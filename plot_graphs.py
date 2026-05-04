"""
=============================================================================
plot_graphs.py
Publication-Ready Graph Generator for MARL IoT Research Paper
Author: Prakriti

USAGE:
    cd ~/ns-allinone-3.43/ns-3.43
    python3 scratch/marl-iot/plot_graphs.py

GENERATES:
    1. Throughput comparison bar chart
    2. Latency comparison bar chart
    3. Packet loss comparison bar chart
    4. Jain's Fairness Index comparison
    5. Scalability curve (IoT devices vs metrics)
    6. Fault tolerance impact chart
    7. MARL training convergence curve
    8. Energy consumption comparison
    9. Combined dashboard (all metrics in one figure)
=============================================================================
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for WSL
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
GRAPHS_DIR  = os.path.join(SCRIPT_DIR, "graphs")
os.makedirs(GRAPHS_DIR, exist_ok=True)

# ── Publication style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "font.size"        : 11,
    "axes.titlesize"   : 13,
    "axes.labelsize"   : 12,
    "axes.titleweight" : "bold",
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "legend.fontsize"  : 10,
    "figure.dpi"       : 150,
    "savefig.dpi"      : 200,
})

# ── Colour palette ─────────────────────────────────────────────────────────
C_STATIC = "#E74C3C"   # red    – Static (worst)
C_RR     = "#F39C12"   # orange – Round Robin
C_MARL   = "#27AE60"   # green  – MARL (best)
C_NORMAL = "#2980B9"   # blue   – normal operation
C_FAIL   = "#8E44AD"   # purple – failure scenario


# =============================================================================
# HELPER: load JSON results
# =============================================================================
def load(name):
    path = os.path.join(RESULTS_DIR, f"{name}_results.json")
    if not os.path.exists(path):
        print(f"[WARN] Missing: {path}  — skipping related graphs")
        return None
    with open(path) as f:
        return json.load(f)

def save_fig(fig, filename):
    path = os.path.join(GRAPHS_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"[SAVED] {path}")


# =============================================================================
# 1. BAR CHART — one metric, multiple methods
# =============================================================================
def plot_metric_comparison(methods, values, ylabel, title,
                           filename, colors, unit=""):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(methods))
    bars = ax.bar(x, values, color=colors, width=0.5,
                  edgecolor="white", linewidth=1.2)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(values)*0.02,
                f"{val:.4f}{unit}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(values) * 1.25)
    fig.tight_layout()
    save_fig(fig, filename)


# =============================================================================
# 2. SCALABILITY LINE CHART
# =============================================================================
def plot_scalability(scalability_data):
    if not scalability_data:
        return

    iot_counts  = [d["num_iot"]                for d in scalability_data]
    throughputs = [d["total_throughput_mbps"]   for d in scalability_data]
    latencies   = [d["avg_latency_ms"]          for d in scalability_data]
    losses      = [d["packet_loss_pct"]         for d in scalability_data]
    jfis        = [d["jain_fairness_index"]     for d in scalability_data]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Scalability Analysis: Network Performance vs IoT Device Count",
                 fontsize=14, fontweight="bold", y=1.01)

    specs = [
        (axes[0,0], throughputs, "Throughput (Mbps)",        C_MARL,   "Throughput"),
        (axes[0,1], latencies,   "End-to-End Latency (ms)",  C_STATIC, "Latency"),
        (axes[1,0], losses,      "Packet Loss Rate (%)",     C_FAIL,   "Packet Loss"),
        (axes[1,1], jfis,        "Jain's Fairness Index",    C_NORMAL, "JFI"),
    ]

    for ax, vals, ylabel, color, label in specs:
        ax.plot(iot_counts, vals, "o-", color=color,
                linewidth=2.2, markersize=7, label=label)
        ax.fill_between(iot_counts, vals, alpha=0.12, color=color)
        ax.set_xlabel("Number of IoT Devices")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs IoT Devices")
        ax.set_xticks(iot_counts)
        ax.legend()

    fig.tight_layout()
    save_fig(fig, "04_scalability_analysis.png")


# =============================================================================
# 3. FAULT TOLERANCE CHART
# =============================================================================
def plot_fault_tolerance(fault_data):
    if not fault_data:
        return

    normal  = fault_data.get("normal",  {})
    failure = fault_data.get("failure", {})
    if not normal or not failure:
        return

    metrics = ["Latency (ms)", "Packet Loss (%)"]
    normal_vals  = [normal.get("avg_latency_ms", 0),
                    normal.get("packet_loss_pct", 0)]
    failure_vals = [failure.get("avg_latency_ms", 0),
                    failure.get("packet_loss_pct", 0)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle("Fault Tolerance: Normal vs Base-Station Failure Scenario",
                 fontsize=13, fontweight="bold")

    for i, (ax, metric, n_val, f_val) in enumerate(
            zip(axes, metrics, normal_vals, failure_vals)):
        bars = ax.bar(["Normal", "BS Failure"],
                      [n_val, f_val],
                      color=[C_NORMAL, C_FAIL],
                      width=0.45, edgecolor="white", linewidth=1.2)
        for bar, val in zip(bars, [n_val, f_val]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(n_val, f_val)*0.03,
                    f"{val:.4f}", ha="center", fontsize=11, fontweight="bold")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} Under Failure")
        ax.set_ylim(0, max(n_val, f_val) * 1.3)

    impact_lat  = fault_data.get("latency_impact_ms", 0)
    impact_loss = fault_data.get("loss_impact_pct",   0)
    fig.text(0.5, -0.04,
             f"Failure Impact → Latency: +{impact_lat:.3f}ms  |  "
             f"Packet Loss: +{impact_loss:.4f}%",
             ha="center", fontsize=11,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF3CD"))

    fig.tight_layout()
    save_fig(fig, "05_fault_tolerance.png")


# =============================================================================
# 4. MARL TRAINING CONVERGENCE
# =============================================================================
def plot_marl_convergence(marl_data):
    if not marl_data:
        return

    episodes   = [d["episode"]               for d in marl_data]
    rewards    = [d["avg_reward"]             for d in marl_data]
    latencies  = [d["avg_latency_ms"]         for d in marl_data]
    throughputs= [d["total_throughput_mbps"]  for d in marl_data]
    epsilons   = [d["epsilon"]                for d in marl_data]
    jfis       = [d["jain_fairness_index"]    for d in marl_data]

    # Smooth curves with rolling average
    def smooth(vals, window=3):
        if len(vals) < window:
            return vals
        return np.convolve(vals, np.ones(window)/window, mode="same").tolist()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("MARL Q-Learning Training Convergence",
                 fontsize=14, fontweight="bold")

    # Reward
    ax = axes[0,0]
    ax.plot(episodes, rewards, color=C_MARL, alpha=0.3, linewidth=1)
    ax.plot(episodes, smooth(rewards), color=C_MARL, linewidth=2.2,
            label="Avg Reward (smoothed)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Average Reward")
    ax.set_title("Agent Reward Convergence"); ax.legend()

    # Epsilon decay
    ax = axes[0,1]
    ax.plot(episodes, epsilons, color=C_STATIC, linewidth=2.2)
    ax.fill_between(episodes, epsilons, alpha=0.15, color=C_STATIC)
    ax.set_xlabel("Episode"); ax.set_ylabel("Epsilon (ε)")
    ax.set_title("Exploration Rate Decay")
    ax.text(episodes[-1]*0.6, max(epsilons)*0.5,
            "← Exploring\n→ Exploiting", fontsize=9, color=C_STATIC)

    # Latency over training
    ax = axes[1,0]
    ax.plot(episodes, latencies, color=C_RR, alpha=0.3, linewidth=1)
    ax.plot(episodes, smooth(latencies), color=C_RR, linewidth=2.2,
            label="Latency (smoothed)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Avg Latency (ms)")
    ax.set_title("Latency During Training"); ax.legend()

    # JFI over training
    ax = axes[1,1]
    ax.plot(episodes, jfis, color=C_NORMAL, linewidth=2.2,
            label="Jain's Fairness Index")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Perfect = 1.0")
    ax.set_xlabel("Episode"); ax.set_ylabel("Jain's Fairness Index")
    ax.set_title("Load Fairness During Training")
    ax.set_ylim(0, 1.1); ax.legend()

    fig.tight_layout()
    save_fig(fig, "06_marl_convergence.png")


# =============================================================================
# 5. COMBINED COMPARISON DASHBOARD
# =============================================================================
def plot_dashboard(static, rr, marl_final=None):
    """Single figure with all key metrics — great for paper overview."""

    methods = ["Static LB", "Round Robin"]
    colors  = [C_STATIC, C_RR]

    tputs  = [static.get("total_throughput_mbps", 0),
               rr.get("total_throughput_mbps", 0)]
    lats   = [static.get("avg_latency_ms", 0),
               rr.get("avg_latency_ms", 0)]
    losses = [static.get("packet_loss_pct", 0),
               rr.get("packet_loss_pct", 0)]
    jfis   = [static.get("jain_fairness_index", 0),
               rr.get("jain_fairness_index", 0)]

    if marl_final:
        methods.append("MARL (Q-learn)")
        colors.append(C_MARL)
        tputs.append(marl_final.get("total_throughput_mbps", 0))
        lats.append(marl_final.get("avg_latency_ms", 0))
        losses.append(marl_final.get("packet_loss_pct", 0))
        jfis.append(marl_final.get("jain_fairness_index", 0))

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        "Performance Comparison: Static vs Round Robin vs MARL\n"
        "MARL Framework for Load Balancing in Decentralized IoT Networks",
        fontsize=13, fontweight="bold", y=1.01)

    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    metric_specs = [
        (gs[0,0], tputs,  "Throughput (Mbps)",       "Throughput",         "Mbps"),
        (gs[0,1], lats,   "End-to-End Latency (ms)", "Avg Latency",        "ms"),
        (gs[1,0], losses, "Packet Loss Rate (%)",    "Packet Loss",        "%"),
        (gs[1,1], jfis,   "Jain's Fairness Index",   "Fairness Index",     ""),
    ]

    for spec, vals, ylabel, title, unit in metric_specs:
        ax = fig.add_subplot(spec)
        x = np.arange(len(methods))
        bars = ax.bar(x, vals, color=colors, width=0.5,
                      edgecolor="white", linewidth=1.2)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.02,
                    f"{val:.4f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, max(vals) * 1.3 if max(vals) > 0 else 1)

    # Legend
    patches = [mpatches.Patch(color=c, label=m)
               for m, c in zip(methods, colors)]
    fig.legend(handles=patches, loc="lower center",
               ncol=len(methods), bbox_to_anchor=(0.5, -0.04),
               frameon=True, fontsize=11)

    save_fig(fig, "07_dashboard_comparison.png")


# =============================================================================
# 6. BS LOAD DISTRIBUTION
# =============================================================================
def plot_load_distribution(static, rr, marl_final=None):
    """Shows how evenly load is spread across base stations."""

    fig, axes = plt.subplots(1, 2 + (1 if marl_final else 0),
                             figsize=(5*(2+(1 if marl_final else 0)), 4.5))
    fig.suptitle("Base Station Load Distribution",
                 fontsize=13, fontweight="bold")

    datasets = [("Static LB", static, C_STATIC),
                ("Round Robin", rr, C_RR)]
    if marl_final:
        datasets.append(("MARL", marl_final, C_MARL))

    for ax, (label, data, color) in zip(axes, datasets):
        loads = data.get("bs_loads_mbps", [])
        if not loads:
            loads = [0.1, 0.1, 0.1]  # placeholder
        bs_labels = [f"BS-{i}" for i in range(len(loads))]
        bars = ax.bar(bs_labels, loads, color=color,
                      width=0.5, edgecolor="white", linewidth=1.2)
        for bar, val in zip(bars, loads):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(loads)*0.02,
                    f"{val:.4f}", ha="center", fontsize=10, fontweight="bold")
        jfi = data.get("jain_fairness_index", 0)
        ax.set_title(f"{label}\nJFI = {jfi:.4f}")
        ax.set_ylabel("Load (Mbps)")
        ax.set_ylim(0, max(loads)*1.35 if loads else 1)
        ax.axhline(y=np.mean(loads), color="gray",
                   linestyle="--", alpha=0.6, label=f"Mean={np.mean(loads):.4f}")
        ax.legend(fontsize=9)

    fig.tight_layout()
    save_fig(fig, "08_load_distribution.png")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("="*55)
    print("  MARL IoT Research — Graph Generator")
    print("="*55)
    print(f"  Loading results from: {RESULTS_DIR}")
    print(f"  Saving graphs to    : {GRAPHS_DIR}\n")

    # Load all available results
    static     = load("static")
    rr         = load("round_robin")
    scalability= load("scalability")
    fault      = load("fault_tolerance")
    marl_train = load("marl_training")

    graphs_made = 0

    # ── Graph 1: Throughput comparison ────────────────────────────────────
    if static and rr:
        methods = ["Static LB", "Round Robin"]
        values  = [static["total_throughput_mbps"], rr["total_throughput_mbps"]]
        colors  = [C_STATIC, C_RR]

        # Add MARL final result if available
        marl_final = None
        if marl_train:
            last = marl_train[-1]
            marl_final = {
                "total_throughput_mbps": last["total_throughput_mbps"],
                "avg_latency_ms"       : last["avg_latency_ms"],
                "packet_loss_pct"      : last["packet_loss_pct"],
                "jain_fairness_index"  : last["jain_fairness_index"],
                "bs_loads_mbps"        : last.get("bs_loads_normalized", []),
            }
            methods.append("MARL (Q-learn)")
            values.append(marl_final["total_throughput_mbps"])
            colors.append(C_MARL)

        plot_metric_comparison(
            methods, values,
            "Throughput (Mbps)", "Total Network Throughput Comparison",
            "01_throughput_comparison.png", colors, unit=" Mbps")
        graphs_made += 1

        # ── Graph 2: Latency ──────────────────────────────────────────────
        lat_vals = [static["avg_latency_ms"], rr["avg_latency_ms"]]
        if marl_final:
            lat_vals.append(marl_final["avg_latency_ms"])
        plot_metric_comparison(
            methods, lat_vals,
            "Avg End-to-End Latency (ms)", "End-to-End Latency Comparison",
            "02_latency_comparison.png", colors, unit=" ms")
        graphs_made += 1

        # ── Graph 3: JFI ──────────────────────────────────────────────────
        jfi_vals = [static["jain_fairness_index"], rr["jain_fairness_index"]]
        if marl_final:
            jfi_vals.append(marl_final["jain_fairness_index"])
        plot_metric_comparison(
            methods, jfi_vals,
            "Jain's Fairness Index", "Load Fairness Comparison (1.0 = Perfect)",
            "03_fairness_comparison.png", colors)
        graphs_made += 1

        # ── Graph 7: Dashboard ────────────────────────────────────────────
        plot_dashboard(static, rr, marl_final)
        graphs_made += 1

        # ── Graph 8: Load distribution ────────────────────────────────────
        plot_load_distribution(static, rr, marl_final)
        graphs_made += 1

    # ── Graph 4: Scalability ──────────────────────────────────────────────
    if scalability:
        plot_scalability(scalability)
        graphs_made += 1

    # ── Graph 5: Fault tolerance ──────────────────────────────────────────
    if fault:
        plot_fault_tolerance(fault)
        graphs_made += 1

    # ── Graph 6: MARL convergence ─────────────────────────────────────────
    if marl_train:
        plot_marl_convergence(marl_train)
        graphs_made += 1

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  ✅ {graphs_made} graphs saved to:")
    print(f"     {GRAPHS_DIR}")
    print(f"{'='*55}")
    print("\nGraph files:")
    for f in sorted(os.listdir(GRAPHS_DIR)):
        if f.endswith(".png"):
            size = os.path.getsize(os.path.join(GRAPHS_DIR, f))
            print(f"  {f}  ({size//1024} KB)")
    print("\n✅ Open graphs folder in VS Code to view your paper figures!")
    print("   In VS Code: File → Open Folder →")
    print(f"   {GRAPHS_DIR}")