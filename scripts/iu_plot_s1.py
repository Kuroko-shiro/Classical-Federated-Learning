"""Visualize scenario-1 REAL-DATA (IU X-ray) results.

Plots three runs (FedAvg α=1.0, FedAvg α=0.1, FedProx α=0.1), all 40 rounds,
4 clients, full data, no encoder freezing. Numbers are transcribed from the run
logs (FedAvg runs predate the CSV-logging feature; FedProx has a CSV too).

Figures (real data only, no synthetic comparison):
  1. auroc over rounds        : IID healthy vs strong-non-IID hole + instability
  2. macro_f1 over rounds      : the gap is even starker on macro-F1
  3. FedAvg vs FedProx (α=0.1) : FedProx does NOT close the hole

Run: python scripts/iu_plot_s1.py   (writes PNGs to results/iu/plots/)
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- transcribed round-by-round metrics from the logs -----------------------
# FedAvg, alpha=1.0 (near-IID): client sizes {0:912,1:930,2:341,3:487}
FEDAVG_A10 = {
    "auroc": [0.872,0.908,0.942,0.963,0.956,0.959,0.971,0.979,0.977,0.963,
              0.968,0.966,0.957,0.935,0.953,0.965,0.955,0.971,0.964,0.961,
              0.953,0.927,0.964,0.948,0.961,0.957,0.961,0.936,0.969,0.963,
              0.978,0.967,0.960,0.972,0.973,0.978,0.972,0.977,0.964,0.974],
    "macro_f1": [0.111,0.271,0.410,0.522,0.627,0.614,0.688,0.713,0.650,0.691,
                 0.668,0.622,0.680,0.628,0.627,0.682,0.685,0.642,0.651,0.661,
                 0.671,0.643,0.673,0.665,0.644,0.620,0.684,0.684,0.594,0.715,
                 0.694,0.626,0.606,0.621,0.610,0.677,0.595,0.652,0.631,0.632],
}
# FedAvg, alpha=0.1 (strong non-IID): client sizes {0:21,1:122,2:1977,3:550}
FEDAVG_A01 = {
    "auroc": [0.891,0.887,0.798,0.893,0.884,0.899,0.865,0.883,0.877,0.770,
              0.754,0.750,0.744,0.769,0.745,0.742,0.773,0.718,0.764,0.787,
              0.786,0.766,0.780,0.766,0.786,0.785,0.789,0.784,0.713,0.786,
              0.797,0.790,0.751,0.788,0.788,0.790,0.795,0.779,0.743,0.774],
    "macro_f1": [0.209,0.296,0.071,0.332,0.362,0.346,0.355,0.402,0.376,0.187,
                 0.152,0.181,0.189,0.230,0.205,0.169,0.180,0.217,0.195,0.157,
                 0.207,0.243,0.171,0.181,0.172,0.170,0.148,0.212,0.182,0.235,
                 0.142,0.255,0.159,0.162,0.180,0.249,0.185,0.189,0.225,0.202],
}
# FedProx (mu=0.1), alpha=0.1: same client sizes as FedAvg α=0.1
FEDPROX_A01 = {
    "auroc": [0.887,0.884,0.862,0.798,0.784,0.751,0.768,0.696,0.732,0.753,
              0.745,0.776,0.753,0.763,0.753,0.719,0.729,0.758,0.738,0.774,
              0.771,0.748,0.749,0.744,0.755,0.765,0.745,0.769,0.766,0.770,
              0.779,0.753,0.741,0.732,0.740,0.750,0.784,0.740,0.793,0.785],
    "macro_f1": [0.179,0.265,0.179,0.126,0.173,0.171,0.165,0.209,0.154,0.179,
                 0.186,0.175,0.206,0.171,0.164,0.162,0.172,0.175,0.185,0.179,
                 0.169,0.188,0.215,0.233,0.185,0.202,0.189,0.153,0.155,0.167,
                 0.169,0.179,0.158,0.194,0.162,0.146,0.155,0.182,0.176,0.209],
}
# FedProx (mu=0.1), alpha=1.0 (near-IID): client sizes {0:912,1:930,2:341,3:487}
FEDPROX_A10 = {
    "auroc": [0.877,0.922,0.952,0.974,0.954,0.976,0.977,0.981,0.964,0.974,
              0.972,0.975,0.969,0.966,0.971,0.969,0.866,0.955,0.970,0.972,
              0.973,0.968,0.963,0.965,0.959,0.962,0.965,0.974,0.971,0.964,
              0.960,0.972,0.939,0.939,0.961,0.969,0.962,0.968,0.968,0.968],
    "macro_f1": [0.109,0.340,0.417,0.478,0.585,0.623,0.637,0.612,0.665,0.668,
                 0.663,0.667,0.668,0.671,0.667,0.649,0.548,0.662,0.673,0.688,
                 0.685,0.684,0.678,0.681,0.676,0.721,0.685,0.645,0.612,0.595,
                 0.642,0.632,0.611,0.614,0.623,0.623,0.623,0.675,0.625,0.636],
}

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "iu", "plots")
os.makedirs(OUT, exist_ok=True)


def _rounds(d):
    return list(range(len(d["auroc"])))


def plot_convergence(metric: str, ylabel: str):
    plt.figure(figsize=(9, 5.5))
    r = _rounds(FEDAVG_A10)
    plt.plot(r, FEDAVG_A10[metric], "-o", ms=3, color="tab:green",
             label="FedAvg α=1.0 (near-IID)")
    plt.plot(r, FEDPROX_A10[metric], "--s", ms=3, color="tab:green", alpha=0.6,
             label="FedProx α=1.0")
    plt.plot(r, FEDAVG_A01[metric], "-o", ms=3, color="tab:red",
             label="FedAvg α=0.1 (strong non-IID)")
    plt.plot(r, FEDPROX_A01[metric], "--s", ms=3, color="tab:red", alpha=0.6,
             label="FedProx α=0.1")
    if metric == "auroc":
        plt.axhline(0.9, ls=":", c="gray", alpha=0.7, label="target 0.9")
    plt.xlabel("federated round")
    plt.ylabel(ylabel)
    plt.title(f"Scenario 1 (IU X-ray): {ylabel} over rounds\n"
              f"near-IID (green) converges high & stable; "
              f"strong non-IID (red) stays low & oscillates")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    out = os.path.join(OUT, f"s1_real_{metric}_convergence.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


def plot_fedavg_vs_fedprox():
    plt.figure(figsize=(9, 5.5))
    r = _rounds(FEDAVG_A01)
    plt.plot(r, FEDAVG_A01["auroc"], "-o", ms=3, color="tab:red",
             label="FedAvg α=0.1")
    plt.plot(r, FEDPROX_A01["auroc"], "-s", ms=3, color="tab:orange",
             label="FedProx α=0.1 (μ=0.1)")
    plt.axhline(0.9, ls="--", c="gray", alpha=0.6, label="target 0.9")
    plt.xlabel("federated round")
    plt.ylabel("auroc")
    plt.title("Scenario 1 (IU X-ray): FedAvg vs FedProx under strong non-IID\n"
              "FedProx does not close the hole — classical non-IID fix is insufficient")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    out = os.path.join(OUT, "s1_real_fedavg_vs_fedprox_a01.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


def summary():
    def stat(d, k, lo=14):
        a = np.array(d[k][lo:])
        return a.mean(), a.std()
    print("\n=== scenario 1 real-data summary (rounds 14-39) ===")
    for name, d in [("FedAvg α=1.0", FEDAVG_A10),
                    ("FedProx α=1.0", FEDPROX_A10),
                    ("FedAvg α=0.1", FEDAVG_A01),
                    ("FedProx α=0.1", FEDPROX_A01)]:
        am, asd = stat(d, "auroc")
        fm, fsd = stat(d, "macro_f1")
        print(f"  {name:16} auroc {am:.3f}±{asd:.3f}   macro_f1 {fm:.3f}±{fsd:.3f}")
    print("  communication (all three): 159.5 GB total (1.34e8 params x2 x4 x40)")


if __name__ == "__main__":
    plot_convergence("auroc", "auroc")
    plot_convergence("macro_f1", "macro_f1")
    plot_fedavg_vs_fedprox()
    summary()
