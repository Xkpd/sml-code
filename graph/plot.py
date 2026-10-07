"""Draw three study figures from the agreed tuning and prediction files.

Run from the project folder: python graph/plot.py
See README.md for input locations, optional model selection and interpretation.
All scores and intervals come from results.py; this file only draws them.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from results import CLASSES, evaluate_results, validate_outputs
from run import MODEL_ALIASES, read_config, verify_frozen

MODEL_LABELS = {
    "multinomial_logistic_regression": "Logistic regression",
    "lightgbm": "LightGBM",
    "ft_transformer": "FT-Transformer",
}
MODEL_COLORS = dict(zip(MODEL_LABELS, ("#0072B2", "#009E73", "#CC79A7")))
DOMAIN_STYLES = {
    "FL": {"label": "FL training", "linestyle": "-", "marker": "o", "filled": True},
    "Formal_Lab": {"label": "Lab training", "linestyle": ":", "marker": "s", "filled": False},
}
ANALYSIS_STYLES = {
    "main": {"label": "Natural availability", "marker": "o", "filled": True},
    "matched": {"label": "Matched counts", "marker": "D", "filled": False},
}
FIGURE_NAMES = ("learning_curves.png", "domain_comparison.png", "class_f1.png")
INTERVAL_NOTE = "Means and approximate 95% t intervals across 10 outer folds, after averaging 3 seeds."


def check_destination(destination, results_dir, root=ROOT):
    """Keep generated figures out of the two-file results and recovery folders."""
    destination = Path(destination).resolve()
    for protected in (Path(results_dir).resolve(), Path(root).resolve() / "outputs",
                      Path(root).resolve() / "recovery", Path(root).resolve() / "data",
                      Path(root).resolve() / "splits"):
        if destination == protected or protected in destination.parents:
            raise ValueError(f"Choose a graph folder outside {protected}")
    return destination


def load_reports(results_dir, models, root=ROOT):
    """Validate every requested full run before allowing any figures to be saved."""
    root, results_dir = Path(root), Path(results_dir)
    config = read_config(root)
    missing = [results_dir / model / name for model in models
               for name in ("tuning.csv", "predictions.csv.gz")
               if not (results_dir / model / name).is_file()]
    if missing:
        paths = "\n".join(f"  {path}" for path in missing)
        raise ValueError("Completed result files are missing:\n" + paths +
                         "\nNo figures were generated. Finish the runs or explicitly select "
                         "completed models with --models (for example: --models lr lightgbm).")
    for model in models:
        verify_frozen(root, config, model)
    reports = {}
    for model in models:
        folder = results_dir / model
        print(f"Validating {MODEL_LABELS[model]}...", flush=True)
        validate_outputs(folder / "tuning.csv", folder / "predictions.csv.gz", config,
                         root / "splits", model, require_complete=True)
        print(f"Calculating participant-level scores for {MODEL_LABELS[model]}...", flush=True)
        report = evaluate_results(folder / "predictions.csv.gz")
        # Only these small summaries are needed for the figures. No extra CSV is written.
        reports[model] = {key: report[key] for key in ("summary", "class_summary")}
    return config, reports


def score(reports, model, analysis, domain, size, class_id=None):
    section, metric = ("summary", "macro_f1") if class_id is None else ("class_summary", "f1")
    rows = [row for row in reports[model][section]
            if row["analysis"] == analysis and row["training_domain"] == domain
            and row["subset_size"] == size and row["metric"] == metric
            and (class_id is None or row["class_id"] == class_id)]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one score for {model}/{analysis}/{domain}/n{size}/{class_id}")
    return rows[0]


def interval_errors(row):
    """Use the shared interval; never infer a paired interval from separate bars."""
    mean, low, high = (row[key] for key in ("mean", "ci_low", "ci_high"))
    if mean is None:
        return None
    if not math.isfinite(mean):
        raise ValueError("Nonfinite mean in shared evaluation")
    if low is None and high is None:
        return None
    if low is None or high is None or not (math.isfinite(low) and math.isfinite(high) and low <= mean <= high):
        raise ValueError("Invalid shared confidence interval")
    return [[mean - low], [high - mean]]


def score_limits(rows):
    """Include the full unbounded t intervals, even if they extend outside 0..1."""
    values = [float(row[key]) for row in rows for key in ("mean", "ci_low", "ci_high")
              if row[key] is not None and math.isfinite(row[key])]
    return min([0.0] + values) - .025, max([1.0] + values) + .025


def point(ax, x, row, color, marker="o", filled=True, annotate_support=False):
    if row["mean"] is None:
        ax.text(x, .025, "NA", ha="center", color=color, fontsize=8,
                transform=ax.get_xaxis_transform())
        return
    ax.errorbar(x, row["mean"], yerr=interval_errors(row), fmt=marker,
                color=color, markerfacecolor=color if filled else "white",
                markeredgecolor=color, markeredgewidth=1.2,
                markersize=6, capsize=3, elinewidth=1.2, zorder=3)
    if annotate_support and row["contributing_folds"] < 10:
        ax.annotate(f"n={row['contributing_folds']}", (x, row["mean"]),
                    xytext=(0, 9), textcoords="offset points", ha="center", color=color, fontsize=8)


def decorate(fig, title, subtitle, footer, models):
    fig.suptitle(title, x=.07, ha="left", y=.98, fontsize=17, fontweight="bold")
    fig.text(.07, .915, subtitle, fontsize=10, color="#555555")
    selected = ", ".join(MODEL_LABELS[model] for model in models)
    fig.text(.07, .025, footer + "\nModels shown: " + selected, fontsize=8, color="#555555", va="bottom")
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#DEDEDE", linewidth=.6, zorder=0)
        ax.set_axisbelow(True)


def line_handle(*, color="#444444", linestyle="-", marker=None, filled=True, label=""):
    """Legend handle with the same redundant visual encoding as the data."""
    from matplotlib.lines import Line2D
    return Line2D([], [], color=color, linestyle=linestyle, linewidth=1.8,
                  marker=marker, markersize=6,
                  markerfacecolor=color if filled else "white",
                  markeredgecolor=color, markeredgewidth=1.2, label=label)


def add_model_domain_legends(ax, models, *, model_location="upper left", domain_location="upper right"):
    model_legend = ax.legend(
        handles=[line_handle(color=MODEL_COLORS[model], label=MODEL_LABELS[model]) for model in models],
        title="Model", frameon=False, fontsize=9, title_fontsize=9, loc=model_location)
    ax.add_artist(model_legend)
    ax.legend(handles=[line_handle(linestyle=style["linestyle"], marker=style["marker"],
                                   filled=style["filled"], label=style["label"])
                       for style in DOMAIN_STYLES.values()],
              title="Training domain", frameon=False, fontsize=9,
              title_fontsize=9, loc=domain_location)


def learning_curves(plt, config, reports):
    sizes, models = config["participant_sizes"], list(reports)
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    rows = [score(reports, model, "main", domain, size)
            for model in models for domain in DOMAIN_STYLES for size in sizes]
    for model in models:
        for domain, style in DOMAIN_STYLES.items():
            series = [score(reports, model, "main", domain, size) for size in sizes]
            ax.plot(sizes, [r["mean"] for r in series], color=MODEL_COLORS[model],
                    linestyle=style["linestyle"], linewidth=1.8)
            for size, row in zip(sizes, series):
                point(ax, size, row, MODEL_COLORS[model], style["marker"], style["filled"])
    ax.set(xlabel="Number of labelled training participants", ylabel="Participant Macro-F1",
           xticks=sizes, ylim=score_limits(rows))
    add_model_domain_legends(ax, models)
    decorate(fig, "How do training participants and domain affect recognition?",
             "Main analysis. Colour identifies the model; line style and marker identify the training domain. Test data: unseen FL participants.",
             INTERVAL_NOTE + "\nWindow counts and class composition can differ between FL and Lab.", models)
    fig.subplots_adjust(left=.095, right=.98, bottom=.25, top=.80)
    return fig


def domain_comparison(plt, config, reports):
    size, models = config["matched_participant_sizes"][0], list(reports)
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.8))
    names = [MODEL_LABELS[model].replace("Logistic regression", "Logistic\nregression") for model in models]
    rows = [score(reports, model, analysis, domain, size)
            for model in models for analysis in ("main", "matched") for domain in DOMAIN_STYLES]
    for ax, analysis, title in zip(axes[:2], ("main", "matched"), ("Natural availability", "Matched window counts")):
        for offset, (domain, style) in zip((-.12, .12), DOMAIN_STYLES.items()):
            for i, model in enumerate(models):
                if domain == "Formal_Lab":
                    fl = score(reports, model, analysis, "FL", size)
                    lab = score(reports, model, analysis, "Formal_Lab", size)
                    if fl["mean"] is not None and lab["mean"] is not None:
                        ax.plot([i - .12, i + .12], [fl["mean"], lab["mean"]],
                                color=MODEL_COLORS[model], linewidth=1, alpha=.45, zorder=1)
                point(ax, i + offset, score(reports, model, analysis, domain, size),
                      MODEL_COLORS[model], style["marker"], style["filled"])
        ax.set(title=title, xticks=range(len(models)), xticklabels=names,
               xlim=(-.5, len(models) - .5), ylim=score_limits(rows))
        ax.tick_params(axis="x", labelsize=9)
    axes[0].set_ylabel("Participant Macro-F1")
    axes[0].legend(handles=[line_handle(linestyle="none", marker=style["marker"], filled=style["filled"],
                                        label=style["label"]) for style in DOMAIN_STYLES.values()],
                   title="Training domain", frameon=False, fontsize=9,
                   title_fontsize=9, loc="lower right")
    axes[2].axhline(0, color="#555555", linewidth=1)
    for offset, (analysis, style) in zip((-.12, .12), ANALYSIS_STYLES.items()):
        for i, model in enumerate(models):
            point(axes[2], i + offset, score(reports, model, analysis, "FL_minus_Formal_Lab", size),
                  MODEL_COLORS[model], style["marker"], style["filled"])
    axes[2].set(title="Paired FL minus Lab", ylabel="Difference in participant Macro-F1",
                xticks=range(len(models)), xticklabels=names, xlim=(-.5, len(models) - .5))
    axes[2].tick_params(axis="x", labelsize=9)
    axes[2].legend(handles=[line_handle(linestyle="none", marker=style["marker"], filled=style["filled"],
                                        label=style["label"]) for style in ANALYSIS_STYLES.values()],
                   title="Analysis", frameon=False, fontsize=8, title_fontsize=9, loc="best")
    decorate(fig, f"Does the FL–Lab difference remain after matching?  |  {size} participants",
             "Test data: the same unseen FL participants. Positive paired differences favour FL training.",
             INTERVAL_NOTE + "\nMatching equalises training-window counts within each participant and class; it is not a causal test.", models)
    fig.subplots_adjust(left=.065, right=.99, bottom=.26, top=.82, wspace=.39)
    return fig


def class_f1(plt, config, reports):
    size, models = max(config["participant_sizes"]), list(reports)
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.8), sharey=True)
    names = [MODEL_LABELS[model].replace("Logistic regression", "Logistic\nregression") for model in models]
    rows = [score(reports, model, "main", domain, size, c)
            for model in models for domain in DOMAIN_STYLES for c in range(4)]
    for c, ax in enumerate(axes.flat):
        for offset, (domain, style) in zip((-.12, .12), DOMAIN_STYLES.items()):
            for i, model in enumerate(models):
                point(ax, i + offset, score(reports, model, "main", domain, size, c),
                      MODEL_COLORS[model], style["marker"], style["filled"], annotate_support=True)
        ax.set(title=CLASSES[c].replace("_", " "), xticks=range(len(models)), xticklabels=names,
               xlim=(-.5, len(models) - .5), ylim=score_limits(rows))
        ax.tick_params(axis="x", labelsize=9)
        if c % 2 == 0:
            ax.set_ylabel("Participant-averaged class F1")
    axes[0, 1].legend(handles=[line_handle(linestyle="none", marker=style["marker"], filled=style["filled"],
                                           label=style["label"]) for style in DOMAIN_STYLES.values()],
                      title="Training domain", frameon=False, fontsize=9,
                      title_fontsize=9, loc="lower right")
    decorate(fig, f"Which activities remain difficult?  |  {size} participants",
             "Main analysis. Scores for each activity, evaluated on unseen FL participants.",
             INTERVAL_NOTE + "\nAbsent true classes are omitted; NA = no support. n labels show fewer than 10 contributing folds (no interval)."
             "\nThese class averages must not be averaged again to reconstruct the primary participant Macro-F1.", models)
    fig.subplots_adjust(left=.075, right=.98, bottom=.21, top=.85, hspace=.47, wspace=.15)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_ALIASES), default=list(MODEL_ALIASES),
                        help="Models to show; default: lr lightgbm ft. Each selected model must have its complete run.")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "outputs",
                        help="Parent folder containing canonical model subfolders with the two result files")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "graph", help="Destination for the three PNG figures")
    args = parser.parse_args(argv)
    models = list(dict.fromkeys(MODEL_ALIASES[name] for name in args.models))
    # Keep colours, labels and model order stable even if command-line order differs.
    models = [name for name in MODEL_LABELS if name in models]
    try:
        destination = check_destination(args.output_dir, args.results_dir)
        try:
            import matplotlib
            matplotlib.use("Agg")  # Save images on desktops and servers without requiring a GUI.
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ValueError("Plotting needs Matplotlib. Install it with: python -m pip install matplotlib==3.11.2") from exc
        config, reports = load_reports(args.results_dir, models)
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                             "axes.titlesize": 12, "figure.facecolor": "white", "savefig.facecolor": "white"})
        # Build all three before saving, so missing score groups cannot leave misleading partial figures.
        figures = [make(plt, config, reports) for make in (learning_curves, domain_comparison, class_f1)]
        destination.mkdir(parents=True, exist_ok=True)
        for name, fig in zip(FIGURE_NAMES, figures):
            path = destination / name
            fig.savefig(path, dpi=250, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {path}", flush=True)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Graph generation stopped: {exc}\n")


if __name__ == "__main__":
    main()
