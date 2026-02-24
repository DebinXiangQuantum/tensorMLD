# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations

from typing import Any, Optional, Union
import time

import numpy as np
import numpy.typing as npt
import cudaq_qec as qec

from quimb.tensor import TensorNetwork
from autoray import do, to_backend_dtype
import cupy

from .tensor_network_utils.contractors import ContractorConfig, optimize_path
from .tensor_network_utils.noise_models import factorized_noise_model
from .tensor_network_utils.tensor_network_factory import (
    tensor_network_from_logical_observable, tensor_network_from_parity_check,
    tensor_network_from_single_syndrome, tensor_network_from_syndrome_batch)
from .tensor_network_utils.mps_decoder_core import (
    CompressedTNDecoder, OneDChainMPS, compile_to_1d_chain,
    make_parametrized_syndrome_network, set_global_decoder_config,
    timed_chain_decode, _validate_threshold_window)

# Global defaults used by this decoder. They can be updated via
# `set_global_mps_decoder_config`.
GLOBAL_BOND_DIM = 32
GLOBAL_LOW_THRESHOLD = 0.4
GLOBAL_HIGH_THRESHOLD = 0.6
GLOBAL_MAX_CONDITIONAL_ROUNDS = 16


def set_global_mps_decoder_config(
        *,
        bond_dim: Optional[int] = None,
        low_threshold: Optional[float] = None,
        high_threshold: Optional[float] = None,
        max_conditional_rounds: Optional[int] = None) -> None:
    """Set global defaults for `tensor_network_mps_decoder`."""
    global GLOBAL_BOND_DIM
    global GLOBAL_LOW_THRESHOLD
    global GLOBAL_HIGH_THRESHOLD
    global GLOBAL_MAX_CONDITIONAL_ROUNDS

    next_low = (GLOBAL_LOW_THRESHOLD if low_threshold is None else
                float(low_threshold))
    next_high = (GLOBAL_HIGH_THRESHOLD if high_threshold is None else
                 float(high_threshold))
    _validate_threshold_window(next_low, next_high)

    if bond_dim is not None:
        if bond_dim <= 0:
            raise ValueError("bond_dim must be > 0.")
        GLOBAL_BOND_DIM = int(bond_dim)
    if low_threshold is not None:
        GLOBAL_LOW_THRESHOLD = next_low
    if high_threshold is not None:
        GLOBAL_HIGH_THRESHOLD = next_high
    if max_conditional_rounds is not None:
        if max_conditional_rounds <= 0:
            raise ValueError("max_conditional_rounds must be > 0.")
        GLOBAL_MAX_CONDITIONAL_ROUNDS = int(max_conditional_rounds)

    # Keep core defaults in sync.
    set_global_decoder_config(
        bond_dim=GLOBAL_BOND_DIM,
        low_threshold=GLOBAL_LOW_THRESHOLD,
        high_threshold=GLOBAL_HIGH_THRESHOLD,
        max_conditional_rounds=GLOBAL_MAX_CONDITIONAL_ROUNDS,
    )


@qec.decoder("tensor_network_mps_decoder")
class TensorNetworkMPSDecoder:
    """Tensor network decoder with offline MPS compression and online decoding.

    API is intentionally aligned with `tensor_network_decoder` for unified tests.
    """

    code_tn: TensorNetwork
    logical_tn: TensorNetwork
    syndrome_tn: TensorNetwork
    noise_model: TensorNetwork
    full_tn: TensorNetwork
    full_param_tn: TensorNetwork
    check_inds: list[str]
    error_inds: list[str]
    logical_inds: list[str]
    logical_obs_inds: list[str]
    logical_tags: list[str]
    syndrome_param_inds: list[str]
    _dtype: str
    path_single: Any
    path_batch: Any
    slicing_single: Any
    slicing_batch: Any

    def __init__(
        self,
        H: npt.NDArray[Any],
        logical_obs: npt.NDArray[Any],
        noise_model: Union[TensorNetwork, list[float]],
        check_inds: Optional[list[str]] = None,
        error_inds: Optional[list[str]] = None,
        logical_inds: Optional[list[str]] = None,
        logical_tags: Optional[list[str]] = None,
        contract_noise_model: bool = True,
        dtype: str = "float32",
        device: str = "cuda",
        bond_dim: Optional[int] = None,
        offline_chi: Optional[int] = None,
        offline_cutoff: float = 1.0e-12,
        offline_max_steps: int = 100000,
        low_threshold: Optional[float] = None,
        high_threshold: Optional[float] = None,
        max_conditional_rounds: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        qec.Decoder.__init__(self, H)
        self._dtype = dtype
        self.verbose = verbose
        # `bond_dim` is the explicit public knob controlling SVD rank/virtual
        # bond in swap/merge during offline compression.
        if bond_dim is not None:
            self.bond_dim = int(bond_dim)
        elif offline_chi is not None:
            self.bond_dim = int(offline_chi)
        else:
            self.bond_dim = GLOBAL_BOND_DIM
        if self.bond_dim <= 0:
            raise ValueError("bond_dim must be > 0.")

        self.offline_chi = self.bond_dim
        self.offline_cutoff = offline_cutoff
        self.offline_max_steps = offline_max_steps
        self.low_threshold = (GLOBAL_LOW_THRESHOLD if low_threshold is None else
                              float(low_threshold))
        self.high_threshold = (
            GLOBAL_HIGH_THRESHOLD
            if high_threshold is None else float(high_threshold))
        _validate_threshold_window(self.low_threshold, self.high_threshold)
        self.max_conditional_rounds = (
            GLOBAL_MAX_CONDITIONAL_ROUNDS
            if max_conditional_rounds is None else int(max_conditional_rounds))
        if self.max_conditional_rounds <= 0:
            raise ValueError("max_conditional_rounds must be > 0.")
        self.last_decode_timing: dict[str, Any] = {}

        try:
            gpu_available = cupy.cuda.is_available()
        except cupy.cuda.runtime.CUDARuntimeError:
            gpu_available = False
            print(
                "CUDA driver error on first check, assuming no GPU or insufficient driver."
            )

        if gpu_available and "cuda" in device:
            contractor_name = "cutensornet"
            backend = "numpy"
        else:
            if "cuda" in device:
                print("Warning: CUDA is not available. Using CPU backend.")
            contractor_name = "torch"
            backend = "torch"
            device = "cpu"

        num_checks, num_errors = H.shape
        if check_inds is None:
            self.check_inds = [f"s_{i}" for i in range(num_checks)]
        else:
            assert len(check_inds) == num_checks
            self.check_inds = check_inds

        if error_inds is None:
            self.error_inds = [f"e_{i}" for i in range(num_errors)]
        else:
            assert len(error_inds) == num_errors
            self.error_inds = error_inds

        K_logicals = logical_obs.shape[0]
        self.logical_obs_inds = (["obs"] if K_logicals == 1
                                 else [f"obs_{k}" for k in range(K_logicals)])
        self.syndrome_param_inds = [f"syn_param_{i}" for i in range(num_checks)]

        self.parity_check_matrix = H.copy()
        self.code_tn = tensor_network_from_parity_check(
            self.parity_check_matrix,
            col_inds=self.error_inds,
            row_inds=self.check_inds,
        )

        self.replace_logical_observable(
            logical_obs,
            logical_inds=logical_inds,
            logical_tags=logical_tags,
        )

        # Fallback exact network with scalar syndrome vectors.
        self.syndrome_tn = tensor_network_from_single_syndrome(
            [0.0] * len(self.check_inds), self.check_inds)
        self.full_tn = TensorNetwork()
        self.full_tn = self.full_tn.combine(self.code_tn, virtual=True)
        self.full_tn = self.full_tn.combine(self.logical_tn, virtual=True)
        self.full_tn = self.full_tn.combine(self.syndrome_tn, virtual=True)

        # Path defaults for exact fallback methods.
        self.path_single = None if contractor_name == "cutensornet" else "auto"
        self.path_batch = None if contractor_name == "cutensornet" else "auto"
        self.slicing_batch = tuple()
        self.slicing_single = tuple()
        self._batch_size = 1

        self._set_contractor(contractor_name, device, backend, dtype=dtype)

        if isinstance(noise_model, TensorNetwork):
            old_inds = noise_model._outer_inds
            assert len(old_inds) == len(self.error_inds), (
                f"Noise model has {len(old_inds)} open indices, but expected "
                f"{len(self.error_inds)}.")
            ind_map = {oi: ni for oi, ni in zip(old_inds, self.error_inds)}
            noise_model = noise_model.reindex(ind_map)
        else:
            noise_model = factorized_noise_model(self.error_inds, noise_model)
        self.init_noise_model(noise_model, contract=contract_noise_model)

        # Parameterized syndrome network used by offline compression and online decode.
        self.param_syndrome_tn = make_parametrized_syndrome_network(
            self.check_inds, self.syndrome_param_inds)
        self.full_param_tn = TensorNetwork()
        self.full_param_tn = self.full_param_tn.combine(self.code_tn,
                                                        virtual=True)
        self.full_param_tn = self.full_param_tn.combine(self.logical_tn,
                                                        virtual=True)
        self.full_param_tn = self.full_param_tn.combine(self.param_syndrome_tn,
                                                        virtual=True)
        self.full_param_tn = self.full_param_tn.combine(self.noise_model,
                                                        virtual=True)
        preserve_inds = set(self.syndrome_param_inds + self.logical_obs_inds)
        compile_result = compile_to_1d_chain(
            tn=self.full_param_tn.copy(),
            preserve_inds=preserve_inds,
            syndrome_inds=set(self.syndrome_param_inds),
            logical_inds=set(self.logical_obs_inds),
            chi=self.offline_chi,
            cutoff=self.offline_cutoff,
            max_steps=self.offline_max_steps,
            verbose=self.verbose,
        )
        self._compressed_chain = compile_result.chain
        self._compressed_tn = compile_result.compressed_tn
        self.offline_stats = compile_result.offline_stats
        if self.verbose:
            print(
                "[offline] nodes/edges "
                f"{self.offline_stats.initial_nodes}/{self.offline_stats.initial_edges} "
                f"-> {self.offline_stats.active_nodes}/{self.offline_stats.active_edges}, "
                f"max_bond={self.offline_stats.max_bond_dim}, "
                f"steps={self.offline_stats.steps}, "
                f"trunc_err={self.offline_stats.truncation_error:.3e}, "
                f"chain_ready={self._compressed_chain is not None}, "
                f"compressed_tn_ready={self._compressed_tn is not None}")

    def replace_logical_observable(
            self,
            logical_obs: npt.NDArray[Any],
            logical_inds: Optional[list[str]] = None,
            logical_tags: Optional[list[str]] = None) -> None:
        """Replace logical observables.  Supports (K, n) with K >= 1."""
        assert logical_obs.ndim == 2 and logical_obs.shape[1] == len(self.error_inds), (
            f"logical_obs must have shape (K, {len(self.error_inds)}), "
            f"got {logical_obs.shape}.")
        K = logical_obs.shape[0]
        if logical_inds is None:
            self.logical_inds = [f"l_{k}" for k in range(K)]
        else:
            assert len(logical_inds) == K
            self.logical_inds = logical_inds
        self.logical_tags = (logical_tags if logical_tags is not None
                             else [f"LOG_{k}" for k in range(K)])

        self.logical_obs = logical_obs.copy()
        self.logical_tn = tensor_network_from_parity_check(
            self.logical_obs,
            col_inds=self.error_inds,
            row_inds=self.logical_inds,
            tags=self.logical_tags,
        )
        self.logical_tn = self.logical_tn.combine(
            tensor_network_from_logical_observable(self.logical_obs,
                                                   self.logical_inds,
                                                   self.logical_obs_inds,
                                                   self.logical_tags),
            virtual=True)

        if hasattr(self, "full_tn"):
            self.full_tn = TensorNetwork()
            self.full_tn = self.full_tn.combine(self.code_tn, virtual=True)
            self.full_tn = self.full_tn.combine(self.logical_tn, virtual=True)
            self.full_tn = self.full_tn.combine(self.syndrome_tn, virtual=True)
            if hasattr(self, "noise_model"):
                self.full_tn = self.full_tn.combine(self.noise_model,
                                                    virtual=True)
            self._set_tensor_type(self.full_tn)

    def init_noise_model(self,
                         noise_model: TensorNetwork,
                         contract: bool = False) -> None:
        assert isinstance(noise_model, TensorNetwork), (
            "noise_model must be quimb.tensor.TensorNetwork.")
        assert sorted(noise_model.outer_inds()) == sorted(self.error_inds), (
            "Noise model open indices must match decoder error indices.")
        self.noise_model = noise_model
        self._set_tensor_type(self.noise_model)

        self.full_tn = TensorNetwork()
        self.full_tn = self.full_tn.combine(self.code_tn, virtual=True)
        self.full_tn = self.full_tn.combine(self.logical_tn, virtual=True)
        self.full_tn = self.full_tn.combine(self.syndrome_tn, virtual=True)
        self.full_tn = self.full_tn.combine(self.noise_model, virtual=True)
        if contract:
            for ind in self.error_inds:
                self.full_tn.contract_ind(ind)

    def flip_syndromes(self, values: list[float]) -> None:
        assert len(values) == len(self.check_inds)
        assert all(isinstance(v, float) for v in values)

        dtype = to_backend_dtype(self._dtype,
                                 like=self.contractor_config.backend)
        array_args = {"like": self.contractor_config.backend, "dtype": dtype}
        minus = do("array", (1.0, -1.0), **array_args)
        plus = do("array", (1.0, 1.0), **array_args)

        for i in range(len(self.check_inds)):
            tensor = self.syndrome_tn.tensors[next(
                iter(self.syndrome_tn.tag_map[f"SYN_{i}"]))]
            tensor.modify(data=values[i] * minus + (1.0 - values[i]) * plus)

    def _set_contractor(
        self,
        contractor: str,
        device: str,
        backend: str,
        dtype: Optional[str] = None,
    ) -> None:
        self.contractor_config = ContractorConfig(
            contractor_name=contractor,
            backend=backend,
            device=device,
        )
        if dtype is not None:
            self._dtype = dtype
        self._set_tensor_type(self.full_tn)

        is_cutensornet = contractor == "cutensornet"

        def _adjust_default_path_value(val):
            if is_cutensornet:
                return None if val == "auto" else val
            return "auto" if val is None else val

        self.path_batch = _adjust_default_path_value(self.path_batch)
        self.path_single = _adjust_default_path_value(self.path_single)

    def _decode_exact_fallback(self, syndrome: list[float]) -> list[float]:
        """Exact contraction fallback.  Returns K probabilities."""
        self.flip_syndromes(syndrome)
        if self.path_single is None:
            self.optimize_path(optimize=self.path_single)

        K = len(self.logical_obs_inds)
        if K == 1:
            value = self.contractor_config.contractor(
                self.full_tn.get_equation(output_inds=(self.logical_obs_inds[0],)),
                self.full_tn.arrays,
                optimize=self.path_single,
                slicing=self.slicing_single,
                device_id=self.contractor_config.device_id,
            )
            return [float(value[1] / (value[1] + value[0]))]

        value = self.contractor_config.contractor(
            self.full_tn.get_equation(output_inds=tuple(self.logical_obs_inds)),
            self.full_tn.arrays,
            optimize=self.path_single,
            slicing=self.slicing_single,
            device_id=self.contractor_config.device_id,
        )
        arr = np.asarray(value, dtype=np.float64)
        probs: list[float] = []
        for k in range(K):
            axes = tuple(ax for ax in range(K) if ax != k)
            marginal = arr.sum(axis=axes)
            probs.append(float(marginal[1] / (marginal[0] + marginal[1])))
        return probs

    def _online_use_gpu(self) -> bool:
        if cupy is None:
            return False
        try:
            return ("cuda" in self.contractor_config.device
                    and cupy.cuda.is_available())
        except cupy.cuda.runtime.CUDARuntimeError:
            return False

    def decode(self, syndrome: list[float]) -> "qec.DecoderResult":
        """Decode syndrome.  Returns K probabilities (one per logical observable).

        Uses three backends in priority order:
        1. OneDChainMPS — fastest, available when graph is a 1D path.
        2. CompressedTNDecoder — fully contracted MPS, works for any QLDPC code.
        3. Exact contraction fallback — slowest, uses full TN contraction.
        """
        assert len(syndrome) == len(self.check_inds), (
            f"Syndrome length {len(syndrome)} does not match check count "
            f"{len(self.check_inds)}.")
        assert all(isinstance(x, float) for x in syndrome), (
            "Syndrome values must be float.")

        syndrome_values = {
            label: float(syndrome[i])
            for i, label in enumerate(self.syndrome_param_inds)
        }

        # Path 1: 1D chain MPS (fastest)
        if self._compressed_chain is not None:
            _, marginals, latency_ms = timed_chain_decode(
                self._compressed_chain,
                syndrome_values=syndrome_values,
                use_gpu=self._online_use_gpu(),
                low_threshold=self.low_threshold,
                high_threshold=self.high_threshold,
                max_rounds=self.max_conditional_rounds,
            )
            probs = [float(marginals.get(label, 0.5))
                     for label in self.logical_obs_inds]
            self.last_decode_timing = {
                "backend": "cupy" if self._online_use_gpu() else "numpy",
                "method": "1d_chain",
                "latency_ms": latency_ms,
            }
            if self.verbose:
                print(
                    f"[online/1d_chain] p(logical=1)={probs}, "
                    f"latency={latency_ms:.3f} ms")
            result = qec.DecoderResult()
            result.converged = True
            result.result = probs
            return result

        # Path 2: Compressed TN decoder (works for QLDPC codes)
        if self._compressed_tn is not None:
            t0 = time.perf_counter()
            _, marginals = self._compressed_tn.decode_logicals(
                syndrome_values=syndrome_values)
            latency_ms = (time.perf_counter() - t0) * 1e3
            probs = [float(marginals.get(label, 0.5))
                     for label in self.logical_obs_inds]
            self.last_decode_timing = {
                "backend": "numpy",
                "method": "compressed_tn",
                "latency_ms": latency_ms,
            }
            if self.verbose:
                print(
                    f"[online/compressed_tn] p(logical=1)={probs}, "
                    f"latency={latency_ms:.3f} ms")
            result = qec.DecoderResult()
            result.converged = True
            result.result = probs
            return result

        # Path 3: Exact contraction fallback (slowest)
        if self.verbose:
            print("No compressed decoder available; using exact fallback.")
        probs = self._decode_exact_fallback(syndrome)
        self.last_decode_timing = {"backend": "exact_fallback", "method": "exact"}
        result = qec.DecoderResult()
        result.converged = True
        result.result = probs
        return result

    def decode_batch(self,
                     syndrome_batch: npt.NDArray[Any]) -> list["qec.DecoderResult"]:
        assert syndrome_batch.ndim == 2
        assert syndrome_batch.shape[1] == len(self.check_inds)
        output: list[qec.DecoderResult] = []
        for row in syndrome_batch:
            output.append(self.decode([float(x) for x in row]))
        return output

    def optimize_path(self, optimize: Any = None, batch_size: int = -1) -> Any:
        assert isinstance(batch_size, int)
        is_batch = batch_size > 0
        if is_batch:
            fake_batch = np.ones((batch_size, len(self.check_inds)),
                                 dtype=self._dtype)
            tn = TensorNetwork(
                [t for t in self.full_tn.tensors if "SYNDROME" not in t.tags])
            tn = tn.combine(tensor_network_from_syndrome_batch(
                fake_batch, self.check_inds, batch_index="batch_index"),
                            virtual=True)
            self._set_tensor_type(tn)
            output_inds = ("batch_index",) + tuple(self.logical_obs_inds)
        else:
            tn = self.full_tn
            output_inds = tuple(self.logical_obs_inds)

        path, info = optimize_path(optimize, output_inds, tn)
        slices = info.slices if hasattr(info, "slices") else tuple()

        if is_batch:
            self.path_batch = path
            self.slicing_batch = slices
        else:
            self.path_single = path
            self.slicing_single = slices
        return info

    def benchmark_decode_latency(self,
                                 syndrome: list[float],
                                 warmup: int = 10,
                                 repeats: int = 100) -> dict[str, float]:
        assert repeats > 0
        syndrome_values = {
            label: float(syndrome[i])
            for i, label in enumerate(self.syndrome_param_inds)
        }

        # Path 1: 1D chain
        if self._compressed_chain is not None:
            for _ in range(max(0, warmup)):
                timed_chain_decode(
                    self._compressed_chain,
                    syndrome_values=syndrome_values,
                    use_gpu=self._online_use_gpu(),
                    low_threshold=self.low_threshold,
                    high_threshold=self.high_threshold,
                    max_rounds=self.max_conditional_rounds,
                )
            times = []
            for _ in range(repeats):
                _, _, latency_ms = timed_chain_decode(
                    self._compressed_chain,
                    syndrome_values=syndrome_values,
                    use_gpu=self._online_use_gpu(),
                    low_threshold=self.low_threshold,
                    high_threshold=self.high_threshold,
                    max_rounds=self.max_conditional_rounds,
                )
                times.append(latency_ms)
            return {
                "method": "1d_chain",
                "avg_ms": float(np.mean(times)),
                "min_ms": float(np.min(times)),
                "max_ms": float(np.max(times)),
            }

        # Path 2: Compressed TN decoder
        if self._compressed_tn is not None:
            for _ in range(max(0, warmup)):
                self._compressed_tn.decode_logicals(
                    syndrome_values=syndrome_values)
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                self._compressed_tn.decode_logicals(
                    syndrome_values=syndrome_values)
                times.append((time.perf_counter() - t0) * 1e3)
            return {
                "method": "compressed_tn",
                "avg_ms": float(np.mean(times)),
                "min_ms": float(np.min(times)),
                "max_ms": float(np.max(times)),
            }

        # Path 3: Exact fallback
        for _ in range(max(0, warmup)):
            self._decode_exact_fallback(syndrome)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            self._decode_exact_fallback(syndrome)
            times.append((time.perf_counter() - t0) * 1e3)
        return {
            "method": "exact_fallback",
            "avg_ms": float(np.mean(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
        }

    def _set_tensor_type(self, tn: TensorNetwork) -> None:
        assert isinstance(tn, TensorNetwork)
        tn.apply_to_arrays(lambda x: do(
            "array",
            x,
            like=self.contractor_config.backend,
            dtype=to_backend_dtype(self._dtype,
                                   like=self.contractor_config.backend),
        ))
