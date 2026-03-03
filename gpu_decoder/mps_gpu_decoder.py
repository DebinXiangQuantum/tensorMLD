"""
Python ctypes wrapper for the CUDA MPS decoder shared library.

Usage:
    decoder = MpsGpuDecoder("/path/to/chain/")
    decisions, marginals, latency_us = decoder.decode(syndrome_array)
    decoder.close()
"""

import ctypes
import os
import json
import numpy as np
from pathlib import Path
from typing import Optional


def _find_lib():
    """Locate libmps_decoder.so relative to this file."""
    here = Path(__file__).resolve().parent
    candidates = [
        here / "src" / "libmps_decoder.so",
        here / "libmps_decoder.so",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(
        "libmps_decoder.so not found. Run 'make' in gpu_decoder/src/ first.\n"
        f"Searched: {[str(c) for c in candidates]}"
    )


class MpsGpuDecoder:
    """GPU-accelerated MPS decoder wrapping the CUDA C library."""

    def __init__(self, chain_dir: str, lib_path: Optional[str] = None,
                 fp32: Optional[bool] = None):
        """
        Load MPS data from chain_dir and initialize the GPU decoder.

        Args:
            chain_dir: Path to directory containing chain_meta.json, site_*.npy, etc.
            lib_path: Optional explicit path to libmps_decoder.so.
            fp32: Enable FP32 mixed-precision scan. None=default (auto, enabled),
                  True=force on, False=force off.
        """
        if lib_path is None:
            lib_path = _find_lib()

        self._lib = ctypes.CDLL(lib_path)
        self._setup_prototypes()

        # Load metadata for label mappings
        meta_path = os.path.join(chain_dir, "chain_meta.json")
        with open(meta_path) as f:
            self._meta = json.load(f)

        self._syndrome_labels = self._meta["syndrome_labels"]
        self._logical_labels = self._meta["logical_labels"]
        self._K = len(self._logical_labels)
        self._M = len(self._syndrome_labels)

        # Create decoder handle
        self._handle = self._lib.mps_decoder_create(chain_dir.encode("utf-8"))
        if not self._handle:
            raise RuntimeError(f"Failed to create MPS decoder from {chain_dir}")

        # Verify dimensions match
        assert self._lib.mps_decoder_get_num_logicals(self._handle) == self._K
        assert self._lib.mps_decoder_get_num_syndromes(self._handle) == self._M

        # Set FP32 mode if explicitly requested
        if fp32 is not None:
            self.set_fp32(fp32)

    def _setup_prototypes(self):
        lib = self._lib

        lib.mps_decoder_create.argtypes = [ctypes.c_char_p]
        lib.mps_decoder_create.restype = ctypes.c_void_p

        lib.mps_decoder_destroy.argtypes = [ctypes.c_void_p]
        lib.mps_decoder_destroy.restype = None

        lib.mps_decoder_decode.argtypes = [
            ctypes.c_void_p,                          # handle
            ctypes.POINTER(ctypes.c_int),             # syndrome
            ctypes.POINTER(ctypes.c_int),             # decisions
            ctypes.POINTER(ctypes.c_double),          # marginals
            ctypes.POINTER(ctypes.c_double),          # latency_us
            ctypes.c_double,                          # low_thresh
            ctypes.c_double,                          # high_thresh
            ctypes.c_int,                             # max_rounds
        ]
        lib.mps_decoder_decode.restype = ctypes.c_int

        lib.mps_decoder_decode_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
        ]
        lib.mps_decoder_decode_batch.restype = ctypes.c_int

        lib.mps_decoder_decode_profiled.argtypes = [
            ctypes.c_void_p,                          # handle
            ctypes.POINTER(ctypes.c_int),             # syndrome
            ctypes.POINTER(ctypes.c_int),             # decisions
            ctypes.POINTER(ctypes.c_double),          # marginals
            ctypes.POINTER(ctypes.c_double),          # phase_us (6 doubles)
            ctypes.c_double,                          # low_thresh
            ctypes.c_double,                          # high_thresh
            ctypes.c_int,                             # max_rounds
        ]
        lib.mps_decoder_decode_profiled.restype = ctypes.c_int

        lib.mps_decoder_get_num_logicals.argtypes = [ctypes.c_void_p]
        lib.mps_decoder_get_num_logicals.restype = ctypes.c_int

        lib.mps_decoder_get_num_syndromes.argtypes = [ctypes.c_void_p]
        lib.mps_decoder_get_num_syndromes.restype = ctypes.c_int

        lib.mps_decoder_get_num_sites.argtypes = [ctypes.c_void_p]
        lib.mps_decoder_get_num_sites.restype = ctypes.c_int

        lib.mps_decoder_get_max_chi.argtypes = [ctypes.c_void_p]
        lib.mps_decoder_get_max_chi.restype = ctypes.c_int

        lib.mps_decoder_set_fp32.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.mps_decoder_set_fp32.restype = None

        lib.mps_decoder_get_fp32.argtypes = [ctypes.c_void_p]
        lib.mps_decoder_get_fp32.restype = ctypes.c_int

    @property
    def K(self) -> int:
        return self._K

    @property
    def M(self) -> int:
        return self._M

    @property
    def N(self) -> int:
        return self._lib.mps_decoder_get_num_sites(self._handle)

    @property
    def max_chi(self) -> int:
        return self._lib.mps_decoder_get_max_chi(self._handle)

    @property
    def fp32(self) -> bool:
        """Whether FP32 mixed-precision scan is enabled."""
        return bool(self._lib.mps_decoder_get_fp32(self._handle))

    def set_fp32(self, enable: bool):
        """Enable or disable FP32 mixed-precision scan mode."""
        self._lib.mps_decoder_set_fp32(self._handle, 1 if enable else 0)

    @property
    def syndrome_labels(self) -> list:
        return list(self._syndrome_labels)

    @property
    def logical_labels(self) -> list:
        return list(self._logical_labels)

    def decode(self,
               syndrome: np.ndarray,
               low_thresh: float = 0.1,
               high_thresh: float = 0.9,
               max_rounds: int = 10
               ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Decode a single syndrome vector.

        Args:
            syndrome: Integer array of shape (M,) with values 0 or 1.
            low_thresh: Marginal below this → decide 0.
            high_thresh: Marginal above this → decide 1.
            max_rounds: Max conditional refinement rounds.

        Returns:
            (decisions, marginals, latency_us)
            - decisions: int array of shape (K,), 0 or 1
            - marginals: float array of shape (K,), probability of value 1
            - latency_us: GPU decode latency in microseconds
        """
        syndrome = np.asarray(syndrome, dtype=np.int32).ravel()
        assert len(syndrome) == self._M, f"Expected {self._M} syndromes, got {len(syndrome)}"

        decisions = np.zeros(self._K, dtype=np.int32)
        marginals = np.zeros(self._K, dtype=np.float64)
        latency = ctypes.c_double(0.0)

        ret = self._lib.mps_decoder_decode(
            self._handle,
            syndrome.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            decisions.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            marginals.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(latency),
            ctypes.c_double(low_thresh),
            ctypes.c_double(high_thresh),
            ctypes.c_int(max_rounds),
        )
        if ret != 0:
            raise RuntimeError(f"mps_decoder_decode returned error code {ret}")

        return decisions, marginals, latency.value

    def decode_profiled(self,
                        syndrome: np.ndarray,
                        low_thresh: float = 0.1,
                        high_thresh: float = 0.9,
                        max_rounds: int = 10
                        ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Decode with per-phase latency breakdown.

        Returns:
            (decisions, marginals, phase_times)
            - decisions: int array of shape (K,), 0 or 1
            - marginals: float array of shape (K,)
            - phase_times: dict with keys:
                prepare_base_us, forward_scan_us, backward_scan_us,
                marginals_us, refinement_us, total_us
        """
        syndrome = np.asarray(syndrome, dtype=np.int32).ravel()
        assert len(syndrome) == self._M

        decisions = np.zeros(self._K, dtype=np.int32)
        marginals = np.zeros(self._K, dtype=np.float64)
        phase_us = np.zeros(6, dtype=np.float64)

        ret = self._lib.mps_decoder_decode_profiled(
            self._handle,
            syndrome.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            decisions.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            marginals.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            phase_us.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_double(low_thresh),
            ctypes.c_double(high_thresh),
            ctypes.c_int(max_rounds),
        )
        if ret != 0:
            raise RuntimeError(f"mps_decoder_decode_profiled returned error code {ret}")

        phase_times = {
            "prepare_base_us": float(phase_us[0]),
            "forward_scan_us": float(phase_us[1]),
            "backward_scan_us": float(phase_us[2]),
            "marginals_us": float(phase_us[3]),
            "refinement_us": float(phase_us[4]),
            "total_us": float(phase_us[5]),
        }
        return decisions, marginals, phase_times

    def decode_dict(self,
                    syndrome_values: dict[str, float],
                    low_thresh: float = 0.1,
                    high_thresh: float = 0.9,
                    max_rounds: int = 10
                    ) -> tuple[dict[str, int], dict[str, float], float]:
        """
        Decode using dict interface matching OneDChainMPS.decode_logicals().

        Args:
            syndrome_values: Dict mapping syndrome label -> value (0.0 or 1.0).

        Returns:
            (assignments, marginals, latency_us)
            - assignments: dict label -> 0/1
            - marginals: dict label -> probability
            - latency_us: GPU decode latency in microseconds
        """
        syn_arr = np.zeros(self._M, dtype=np.int32)
        for i, label in enumerate(self._syndrome_labels):
            syn_arr[i] = int(round(syndrome_values.get(label, 0.0)))

        decisions, marg_arr, latency_us = self.decode(
            syn_arr, low_thresh, high_thresh, max_rounds)

        assignments = {}
        marginals = {}
        for i, label in enumerate(self._logical_labels):
            assignments[label] = int(decisions[i])
            marginals[label] = float(marg_arr[i])

        return assignments, marginals, latency_us

    def decode_batch(self,
                     syndromes: np.ndarray,
                     low_thresh: float = 0.1,
                     high_thresh: float = 0.9,
                     max_rounds: int = 10
                     ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Decode a batch of syndrome vectors.

        Args:
            syndromes: Integer array of shape (batch_size, M).

        Returns:
            (decisions, marginals, latency_us)
            - decisions: int array of shape (batch_size, K)
            - marginals: float array of shape (batch_size, K)
            - latency_us: total batch latency in microseconds
        """
        syndromes = np.asarray(syndromes, dtype=np.int32)
        if syndromes.ndim == 1:
            syndromes = syndromes.reshape(1, -1)
        batch_size = syndromes.shape[0]
        assert syndromes.shape[1] == self._M

        # Ensure contiguous
        syndromes = np.ascontiguousarray(syndromes)

        decisions = np.zeros((batch_size, self._K), dtype=np.int32)
        marginals = np.zeros((batch_size, self._K), dtype=np.float64)
        latency = ctypes.c_double(0.0)

        decisions_flat = np.ascontiguousarray(decisions)
        marginals_flat = np.ascontiguousarray(marginals)

        ret = self._lib.mps_decoder_decode_batch(
            self._handle,
            syndromes.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(batch_size),
            decisions_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            marginals_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(latency),
            ctypes.c_double(low_thresh),
            ctypes.c_double(high_thresh),
            ctypes.c_int(max_rounds),
        )
        if ret != 0:
            raise RuntimeError(f"mps_decoder_decode_batch returned error code {ret}")

        return decisions_flat, marginals_flat, latency.value

    def close(self):
        """Free GPU resources."""
        if self._handle:
            self._lib.mps_decoder_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
