#!/usr/bin/env python3
"""Visualize multi-logical TN compression step by step.

Builds ONE tensor network with all K logical + M syndrome indices preserved,
compresses it, and draws the pairwise graph at key stages using networkx +
matplotlib.  Designed primarily for debugging BB18 (QCC_18_4_4: n=18, k=4,
d=4) but supports all 9 ISCA revision codes.

Usage:
    python visualize_multi_logical_compression.py --codes QCC_18_4_4
    python visualize_multi_logical_compression.py --codes QCC_18_4_4,LDPC_25_3_4 --workers 2
    python visualize_multi_logical_compression.py --codes all --bond-dim 64
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# Path setup (mirrors export_isca_revision_1d_mps.py)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parent.parent
QEC_PY = WORKSPACE / "cudaqx" / "libs" / "qec" / "python"
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
if str(QEC_PY) not in sys.path:
    sys.path.insert(0, str(QEC_PY))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod


_TNU = QEC_PY / "cudaq_qec" / "plugins" / "decoders" / "tensor_network_utils"
core = _load_module("viz_mps_core", _TNU / "mps_decoder_core.py")
factory = _load_module("viz_tn_factory", _TNU / "tensor_network_factory.py")

from experiments.codes.isca_revision_codes import (
    ISCA_REVISION_CODE_ORDER,
    load_isca_revision_code,
)
# Re-use the multi-logical TN builder from the export script
from experiments.tests.export_isca_revision_1d_mps import (
    build_full_multi_logical_tn,
)


# ---------------------------------------------------------------------------
# Graph drawing
# ---------------------------------------------------------------------------

def _classify_node(node_id: int,
                   preserved_nodes: set[int],
                   nodes: dict[int, Any],
                   syndrome_param_inds: set[str],
                   logical_obs_inds: set[str]) -> str:
    """Classify a node for colouring by tensor type.

    Categories:
      'syndrome'  — preserved node carrying a syndrome param index
      'logical'   — preserved node carrying a logical obs index
      'delta'     — delta tensor (copy tensor, inserted by pairwise graph)
      'noise'     — noise model tensor ([1-p, p] per qubit)
      'hadamard'  — Hadamard matrix from code/logical parity check
      'syn_param' — syndrome parameter tensor (Hadamard linking check↔param)
      'internal'  — fallback
    """
    node = nodes[node_id]
    tags = getattr(node, '_tags', set())

    if node_id in preserved_nodes:
        for nb in node.neighbor:
            if isinstance(nb, str):
                label = core.out_token_to_index(nb)
                if label in syndrome_param_inds:
                    return "syndrome"
                if label in logical_obs_inds:
                    return "logical"
        return "syndrome"

    if core.DELTA_TAG in tags:
        return "delta"
    if "NOISE" in tags:
        return "noise"
    if "SYNDROME_PARAM" in tags or "SYNDROME" in tags:
        return "syn_param"
    # Hadamard from code TN or logical TN
    if any(t.startswith("LOG_") for t in tags):
        return "hadamard_log"
    # Default: Hadamard from code TN (no tags) or other
    decomp = getattr(node, '_decomp_type', '')
    if decomp in ('svd', 'parity'):
        return "hadamard"
    return "internal"


_COLORS = {
    "syndrome":     "#4A90D9",   # blue — preserved syndrome
    "logical":      "#D94A4A",   # red — preserved logical
    "delta":        "#5CB85C",   # green — delta / copy tensor
    "noise":        "#F5A623",   # orange — noise model
    "hadamard":     "#AAAAAA",   # gray — code Hadamard
    "hadamard_log": "#9B59B6",   # purple — logical Hadamard
    "syn_param":    "#1ABC9C",   # teal — syndrome parameter
    "internal":     "#CCCCCC",   # light gray — fallback
}


def draw_pairwise_graph(
    active_nodes: set[int],
    nodes: dict[int, Any],
    preserved_nodes: set[int],
    syndrome_param_inds: set[str],
    logical_obs_inds: set[str],
    title: str = "",
    contracted_edge: Optional[tuple[int, int]] = None,
    cumulative_trunc_error: float = 0.0,
    out_path: Optional[Path] = None,
) -> None:
    """Draw the pairwise graph at a given compression step."""
    G = nx.Graph()

    for nid in active_nodes:
        cls = _classify_node(nid, preserved_nodes, nodes,
                             syndrome_param_inds, logical_obs_inds)
        G.add_node(nid, cls=cls)

    edge_list = []
    for nid in active_nodes:
        for pos, nb in enumerate(nodes[nid].neighbor):
            if isinstance(nb, int) and nb in active_nodes and nb > nid:
                # Compute bond dimension for label
                bd = 1
                if pos < len(nodes[nid].mps):
                    bd = int(nodes[nid].mps[pos].shape[2])
                G.add_edge(nid, nb, bond_dim=bd)
                edge_list.append((nid, nb))

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    if len(G.nodes) == 0:
        ax.text(0.5, 0.5, "Empty graph", ha='center', va='center',
                fontsize=14, transform=ax.transAxes)
    else:
        pos = nx.spring_layout(G, seed=42, k=2.0 / max(1, len(G.nodes)**0.5))

        # Draw nodes
        for cls, color in _COLORS.items():
            cls_nodes = [n for n, d in G.nodes(data=True) if d.get('cls') == cls]
            if cls_nodes:
                nx.draw_networkx_nodes(
                    G, pos, nodelist=cls_nodes, node_color=color,
                    node_size=300, alpha=0.9, ax=ax)

        # Draw edges (highlight contracted edge)
        normal_edges = [(u, v) for u, v in G.edges() if (u, v) != contracted_edge
                        and (v, u) != contracted_edge]
        highlight_edges = [(u, v) for u, v in G.edges() if (u, v) == contracted_edge
                           or (v, u) == contracted_edge]

        nx.draw_networkx_edges(G, pos, edgelist=normal_edges, edge_color='#666666',
                               width=1.0, alpha=0.6, ax=ax)
        if highlight_edges:
            nx.draw_networkx_edges(G, pos, edgelist=highlight_edges,
                                   edge_color='#FF8C00', width=3.0, alpha=0.9, ax=ax)

        # Edge labels (bond dimensions)
        edge_labels = {(u, v): str(d['bond_dim']) for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                     font_size=7, font_color='#333333', ax=ax)

        # Node labels: show id and number of MPS cores
        node_labels = {}
        for n in G.nodes():
            n_cores = len(nodes[n].mps) if n in nodes else 0
            node_labels[n] = f"{n}\n({n_cores})"
        nx.draw_networkx_labels(G, pos, labels=node_labels,
                                font_size=6, font_color='white', ax=ax)

    # Title with metadata
    n_nodes = len(active_nodes)
    n_edges = len(edge_list)
    full_title = (f"{title}\n"
                  f"Nodes: {n_nodes}  Edges: {n_edges}  "
                  f"Trunc error: {cumulative_trunc_error:.2e}")
    ax.set_title(full_title, fontsize=11)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=_COLORS["syndrome"], label="Syndrome (preserved)"),
        Patch(facecolor=_COLORS["logical"], label="Logical (preserved)"),
        Patch(facecolor=_COLORS["delta"], label="Delta / Copy"),
        Patch(facecolor=_COLORS["noise"], label="Noise model"),
        Patch(facecolor=_COLORS["hadamard"], label="Code Hadamard"),
        Patch(facecolor=_COLORS["hadamard_log"], label="Logical Hadamard"),
        Patch(facecolor=_COLORS["syn_param"], label="Syndrome param"),
        Patch(facecolor=_COLORS["internal"], label="Internal"),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=7)

    plt.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
        print(f"  [viz] saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-code processing
# ---------------------------------------------------------------------------

def process_code(
    code_name: str,
    bond_dim: int,
    noise_p: float,
    seed: int,
    output_dir: Path,
    code_dir: Optional[str],
    max_steps: int,
    cutoff: float,
) -> dict[str, Any]:
    """Build multi-logical TN, compress with snapshots, draw key steps."""
    code = load_isca_revision_code(code_name, seed=seed, code_dir=code_dir)
    H = np.asarray(code.hz, dtype=np.float64)
    logical = np.asarray(code.lz, dtype=np.float64)
    if logical.ndim == 1:
        logical = logical.reshape(1, -1)

    K = logical.shape[0]
    M = H.shape[0]
    N = H.shape[1]
    print(f"\n{'='*60}")
    print(f"[{code_name}] N={N}, K={K}, M={M}, d={code.D}")
    print(f"  H shape: {H.shape}, logical shape: {logical.shape}")
    print(f"  Expected preserved sites: {K+M}")
    print(f"{'='*60}")

    code_out = output_dir / code_name
    code_out.mkdir(parents=True, exist_ok=True)

    # Build multi-logical TN
    full_tn, syndrome_param_inds, logical_obs_inds = build_full_multi_logical_tn(
        H=H, logical=logical, noise_p=noise_p,
    )
    preserve_inds = set(syndrome_param_inds + logical_obs_inds)
    syn_set = set(syndrome_param_inds)
    log_set = set(logical_obs_inds)

    print(f"  Syndrome param inds ({len(syndrome_param_inds)}): {syndrome_param_inds}")
    print(f"  Logical obs inds ({len(logical_obs_inds)}): {logical_obs_inds}")

    # Build pairwise graph
    pairwise = core.build_pairwise_graph_from_tn(full_tn, preserve_inds=preserve_inds)
    nodes = core.build_mps_nodes(pairwise, chi=bond_dim, cutoff=cutoff)
    preserved_nodes = set(pairwise.preserved_nodes)

    print(f"  Pairwise graph: {len(nodes)} nodes, "
          f"{pairwise.initial_internal_edges} edges, "
          f"{len(preserved_nodes)} preserved")

    # --- Phase 1: compress_until_preserved with snapshot callback ---
    cumulative_error = [0.0]  # mutable for closure
    step_counter = [0]

    # Detailed per-step log file
    step_log_path = code_out / "compression_log.jsonl"
    step_log_file = open(step_log_path, "w")

    def compress_snapshot_cb(step, active_nodes, nodes_dict,
                             contracted_edge, step_record):
        if step_record is not None:
            cumulative_error[0] += step_record.total_error
            # Write detailed log entry
            log_entry = step_record.to_dict()
            log_entry["cumulative_error"] = cumulative_error[0]
            log_entry["phase"] = "compress"
            step_log_file.write(json.dumps(log_entry) + "\n")
            step_log_file.flush()
            # Print precision-loss annotation
            exact_str = "EXACT" if step_record.is_exact else "LOSSY"
            if not step_record.is_exact or step % 20 == 0:
                print(f"    step {step:4d} [{exact_str}] edge={step_record.edge} "
                      f"eat={step_record.eat_error:.2e} "
                      f"merge={step_record.merge_error:.2e} "
                      f"compress={step_record.compress_error:.2e} "
                      f"max_core_order={step_record.max_mps_core_order}")
        should_draw = (step == 0 or step % 10 == 0)
        if should_draw:
            draw_pairwise_graph(
                active_nodes=active_nodes,
                nodes=nodes_dict,
                preserved_nodes=preserved_nodes,
                syndrome_param_inds=syn_set,
                logical_obs_inds=log_set,
                title=f"{code_name} - Compress step {step}",
                contracted_edge=contracted_edge,
                cumulative_trunc_error=cumulative_error[0],
                out_path=code_out / f"compress_step_{step:04d}.png",
            )
        step_counter[0] = step

    print(f"\n  [Phase 1] compress_until_preserved (chi={bond_dim})...")
    t0 = time.perf_counter()
    offline = core.compress_until_preserved(
        nodes=nodes,
        preserved_nodes=preserved_nodes,
        max_steps=max_steps,
        reverse=True,
        compress_each_step=True,
        verbose=True,
        chi=bond_dim,
        snapshot_callback=compress_snapshot_cb,
    )
    phase1_ms = (time.perf_counter() - t0) * 1e3

    # Draw final state of phase 1
    draw_pairwise_graph(
        active_nodes=offline.active_nodes,
        nodes=offline.nodes,
        preserved_nodes=offline.preserved_nodes,
        syndrome_param_inds=syn_set,
        logical_obs_inds=log_set,
        title=f"{code_name} - After compress_until_preserved ({step_counter[0]} steps)",
        cumulative_trunc_error=offline.stats.truncation_error,
        out_path=code_out / f"compress_final.png",
    )

    n_exact = sum(1 for r in offline.stats.step_errors if r.is_exact)
    n_lossy = sum(1 for r in offline.stats.step_errors if not r.is_exact)
    print(f"  Phase 1 done: {step_counter[0]} steps "
          f"({n_exact} exact, {n_lossy} lossy), "
          f"active={len(offline.active_nodes)}, "
          f"trunc_err={offline.stats.truncation_error:.2e}, "
          f"{phase1_ms:.0f}ms")

    # --- Try direct chain extraction first ---
    chain = core.extract_1d_chain(
        active_nodes=offline.active_nodes,
        nodes=offline.nodes,
        syndrome_label_set=syn_set,
        logical_label_set=log_set,
    )

    merge_error = 0.0
    if chain is None:
        # --- Phase 2: merge_preserved_to_path with snapshot callback ---
        merge_cumulative = [0.0]

        def merge_snapshot_cb(step, active_nodes, nodes_dict,
                              contracted_edge, trunc_error):
            merge_cumulative[0] = trunc_error
            # Log merge step
            merge_log = {
                "step": step, "phase": "merge",
                "cumulative_error": offline.stats.truncation_error + trunc_error,
                "merge_cumulative_error": trunc_error,
            }
            step_log_file.write(json.dumps(merge_log) + "\n")
            step_log_file.flush()
            # Draw every merge step (few steps, all important)
            draw_pairwise_graph(
                active_nodes=active_nodes,
                nodes=nodes_dict,
                preserved_nodes=offline.preserved_nodes,
                syndrome_param_inds=syn_set,
                logical_obs_inds=log_set,
                title=f"{code_name} - Merge step {step}",
                contracted_edge=contracted_edge,
                cumulative_trunc_error=offline.stats.truncation_error + trunc_error,
                out_path=code_out / f"merge_step_{step:04d}.png",
            )

        print(f"\n  [Phase 2] merge_preserved_to_path...")
        t1 = time.perf_counter()
        merge_error = core.merge_preserved_to_path(
            nodes=offline.nodes,
            active_nodes=offline.active_nodes,
            chi=bond_dim,
            max_steps=max_steps,
            verbose=True,
            snapshot_callback=merge_snapshot_cb,
        )
        phase2_ms = (time.perf_counter() - t1) * 1e3
        print(f"  Phase 2 done: merge_err={merge_error:.2e}, {phase2_ms:.0f}ms")

        # Try chain extraction again
        chain = core.extract_1d_chain(
            active_nodes=offline.active_nodes,
            nodes=offline.nodes,
            syndrome_label_set=syn_set,
            logical_label_set=log_set,
        )
    else:
        print(f"  Chain extracted directly after Phase 1 (no merge needed)")

    total_error = offline.stats.truncation_error + merge_error

    # --- Draw final 1D chain layout ---
    if chain is not None:
        draw_pairwise_graph(
            active_nodes=offline.active_nodes,
            nodes=offline.nodes,
            preserved_nodes=offline.preserved_nodes,
            syndrome_param_inds=syn_set,
            logical_obs_inds=log_set,
            title=f"{code_name} - Final 1D chain ({len(chain.sites)} sites)",
            cumulative_trunc_error=total_error,
            out_path=code_out / "final_chain.png",
        )

        # Print summary
        print(f"\n  === Final 1D chain summary for {code_name} ===")
        print(f"  Total sites: {len(chain.sites)}")
        print(f"  Syndrome sites ({len(chain.syndrome_labels)}): {chain.syndrome_labels}")
        print(f"  Logical sites ({len(chain.logical_labels)}): {chain.logical_labels}")
        site_labels = [s.label for s in chain.sites]
        bond_dims = [s.tensor.shape[0] for s in chain.sites]
        phys_dims = [s.tensor.shape[1] for s in chain.sites]
        print(f"  Site order: {site_labels}")
        print(f"  Bond dims (left): {bond_dims}")
        print(f"  Phys dims: {phys_dims}")
        print(f"  Total truncation error: {total_error:.2e}")

        result = {
            "code_name": code_name,
            "status": "ok",
            "N": N, "K": K, "M": M, "D": float(code.D),
            "num_sites": len(chain.sites),
            "syndrome_labels": chain.syndrome_labels,
            "logical_labels": chain.logical_labels,
            "site_labels": site_labels,
            "bond_dims": bond_dims,
            "total_trunc_error": total_error,
        }
    else:
        print(f"\n  WARNING: Chain extraction failed for {code_name}!")
        result = {
            "code_name": code_name,
            "status": "failed",
            "N": N, "K": K, "M": M, "D": float(code.D),
            "total_trunc_error": total_error,
        }

    # Close step log
    step_log_file.close()
    print(f"  [log] Compression log saved to: {step_log_path}")

    # Save result summary
    with open(code_out / "summary.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize multi-logical TN compression step by step.",
    )
    parser.add_argument(
        "--codes",
        type=str,
        default="QCC_18_4_4",
        help=("Comma-separated code names (e.g. QCC_18_4_4,LDPC_25_3_4). "
              "Use 'all' for all 9 ISCA codes."),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel processes (default: 1).",
    )
    parser.add_argument(
        "--bond-dim",
        type=int,
        default=32,
        help="Offline compression bond dimension (chi).",
    )
    parser.add_argument(
        "--noise-p",
        type=float,
        default=0.01,
        help="Independent physical noise probability.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for code loaders.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(WORKSPACE / "experiments" / "results" / "tn_visualization"),
        help="Output directory for images.",
    )
    parser.add_argument(
        "--code-dir",
        type=str,
        default=None,
        help="Override for GND code file directory.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100000,
        help="Maximum offline compression steps.",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=1.0e-12,
        help="SVD cutoff for compression.",
    )
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Parse code list
    if args.codes.lower() == "all":
        code_names = list(ISCA_REVISION_CODE_ORDER)
    else:
        code_names = [c.strip().upper() for c in args.codes.split(",")]
        for name in code_names:
            if name not in ISCA_REVISION_CODE_ORDER:
                print(f"WARNING: '{name}' not in ISCA_REVISION_CODE_ORDER, "
                      f"available: {ISCA_REVISION_CODE_ORDER}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Codes to process: {code_names}")
    print(f"Output directory: {output_dir}")
    print(f"Bond dimension: {args.bond_dim}")
    print(f"Workers: {args.workers}")

    results = []

    if args.workers <= 1:
        # Sequential processing
        for code_name in code_names:
            try:
                result = process_code(
                    code_name=code_name,
                    bond_dim=args.bond_dim,
                    noise_p=args.noise_p,
                    seed=args.seed,
                    output_dir=output_dir,
                    code_dir=args.code_dir,
                    max_steps=args.max_steps,
                    cutoff=args.cutoff,
                )
                results.append(result)
            except Exception as exc:
                print(f"ERROR processing {code_name}: {exc}")
                import traceback
                traceback.print_exc()
                results.append({"code_name": code_name, "status": "error",
                                "error": str(exc)})
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_code,
                    code_name=code_name,
                    bond_dim=args.bond_dim,
                    noise_p=args.noise_p,
                    seed=args.seed,
                    output_dir=output_dir,
                    code_dir=args.code_dir,
                    max_steps=args.max_steps,
                    cutoff=args.cutoff,
                ): code_name
                for code_name in code_names
            }
            for future in as_completed(futures):
                code_name = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    print(f"ERROR processing {code_name}: {exc}")
                    results.append({"code_name": code_name, "status": "error",
                                    "error": str(exc)})

    # Save overall summary
    summary = {
        "codes_processed": len(results),
        "bond_dim": args.bond_dim,
        "noise_p": args.noise_p,
        "results": results,
    }
    summary_path = output_dir / "overall_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nOverall summary saved to: {summary_path}")

    # Print final table
    print(f"\n{'='*70}")
    print(f"{'Code':<20} {'Status':<8} {'Sites':<8} {'Syn':<6} {'Log':<6} {'TruncErr':<12}")
    print(f"{'-'*70}")
    for r in sorted(results, key=lambda x: x.get("code_name", "")):
        name = r.get("code_name", "?")
        status = r.get("status", "?")
        sites = r.get("num_sites", "-")
        syn = len(r.get("syndrome_labels", []))
        log = len(r.get("logical_labels", []))
        err = r.get("total_trunc_error", 0.0)
        if status == "ok":
            print(f"{name:<20} {status:<8} {sites:<8} {syn:<6} {log:<6} {err:<12.2e}")
        else:
            print(f"{name:<20} {status:<8}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
