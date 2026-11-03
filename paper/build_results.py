"""Regenerate paper tables, statistics and figures from audited numeric records."""

import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def main():
    data = json.loads((ROOT / "data/audited-results.json").read_text())
    a800 = json.loads((ROOT / "data/a800-results.json").read_text())
    generated = ROOT / "generated"
    generated.mkdir(exist_ok=True)
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    seeds = (53, 59, 61, 67)
    exact = [float(data["exact_endpoints"][str(s)]) for s in seeds]
    n2 = [data["n2_endpoints"][str(s)]["validation_loss"] for s in seeds]
    delta = [b - a for a, b in zip(exact, n2)]
    assert delta == data["phase_d"]["paired_candidate_minus_exact"]
    mean = statistics.mean(delta)
    sem = statistics.stdev(delta) / 2
    lo, hi = mean - 3.182446305284263 * sem, mean + 3.182446305284263 * sem
    upper = mean + 2.3533634348018264 * sem
    assert upper > 0.005
    with (generated / "quality.csv").open("w", newline="") as stream:
        w = csv.writer(stream, lineterminator="\n")
        w.writerow(["seed", "exact_validation_nll", "n2_validation_nll", "paired_difference"])
        w.writerows(zip(seeds, exact, n2, delta))
    rows = [f"{s} & {a:.6f} & {b:.6f} & {d:+.6f} " + r"\\" for s, a, b, d in zip(seeds, exact, n2, delta)]
    rows.append(r"\midrule Mean & " + f"{statistics.mean(exact):.6f} & {statistics.mean(n2):.6f} & {mean:+.6f} " + r"\\")
    (generated / "quality_rows.tex").write_text("\n".join(rows) + "\n")
    (generated / "quality_table.tex").write_text(r"\begin{tabular}{rrrr}" + "\n" + r"\toprule Seed & Exact NLL & N2 NLL & N2 minus exact\\\midrule" + "\n" + "\n".join(rows) + "\n" + r"\bottomrule\end{tabular}" + "\n")
    panels = data["fidelity_panels"]
    applicable = [p for p in panels if p["adam_update_gate_applicable"]]
    summary = {
        "n_seeds": 4, "mean_delta": mean, "two_sided_95_t_ci": [lo, hi],
        "one_sided_95_upper": upper, "operational_mean_gate": mean <= 0.005,
        "operational_all_seed_gate": all(d <= 0.010 for d in delta),
        "statistical_ni_at_0p005": upper < 0.005,
        "fidelity_panels": len(panels), "adam_gated_panels": len(applicable),
        "max_panel_median_grad_error": max(p["median_global_gradient_relative_l2"] for p in panels),
        "max_batch_grad_error": max(p["maximum_global_gradient_relative_l2"] for p in panels),
        "min_panel_median_cosine": min(p["median_global_gradient_cosine"] for p in panels),
        "max_gated_adam_error": max(p["median_full_adamw_parameter_delta_relative_l2"] for p in applicable),
        "max_advisory_adam_error": max(p["median_full_adamw_parameter_delta_relative_l2"] for p in panels if not p["adam_update_gate_applicable"]),
    }
    (generated / "statistics.json").write_text(json.dumps(summary, indent=2) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.5, 3.0), layout="constrained")
    ax.axhline(0, color="#666666", lw=1)
    ax.axhline(0.005, color="#ac4a38", ls="--", lw=1, label="Mean tolerance +0.005")
    ax.scatter(range(4), delta, color="#24656c", s=45, zorder=3)
    ax.errorbar([4.5], [mean], yerr=[[mean - lo], [hi - mean]], fmt="D", capsize=5, color="#222222")
    ax.set_xticks([0, 1, 2, 3, 4.5], [str(s) for s in seeds] + ["Mean / 95% CI"])
    ax.set_ylabel("Validation NLL difference (N2 − exact)")
    ax.set_ylim(-0.023, 0.016)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(figures / "quality_differences.pdf")
    fig.savefig(figures / "quality_differences.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), layout="constrained")
    for ax, length in zip(axes, (512, 4096)):
        d = a800["by_sequence_length"][str(length)]
        x = [d[m + "_median"]["peak_allocated_bytes"] / 1024**3 for m in ("full", "n2")]
        y = [d[m + "_median"]["tokens_per_second"] / 1000 for m in ("full", "n2")]
        ax.scatter(x, y, c=["#24656c", "#ac4a38"], s=65)
        for xx, yy, label in zip(x, y, ("Full checkpoint", "N2")):
            ax.annotate(label, (xx, yy), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
        ax.set_title(f"A800 · sequence {length}")
        ax.set_xlabel("Peak allocated memory (GiB)")
        ax.set_ylabel("Throughput (thousand tokens/s)")
        ax.set_xlim(min(x) - 0.025, max(x) + 0.025)
        ax.set_ylim(7.5, 14.5)
        ax.ticklabel_format(useOffset=False, axis="x")
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(figures / "a800_tradeoff.pdf")
    fig.savefig(figures / "a800_tradeoff.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
