#!/usr/bin/env python3
"""Generate LaTeX report from HP search Phase 2 results.

Table format: one table per category, with main column groups per fault type
(Slugging, Blockage, Leakage, Diverted, All Faults), each subdivided into
AUC and F1 sub-columns. Large tables use longtable for multi-page support.
"""

import os
import csv
import glob
import re

BASE = "/mnt/datassd3/rashinda/DySTGAT/results/pronto/hpsearch_phase2_20260210_050115"

# Experiment categories (from experiment_plan.md)
CATEGORIES = {
    "reference": list(range(1, 3)),
    "ablation": list(range(3, 19)),
    "sensitivity": list(range(19, 69)),
    "combination": list(range(69, 85)),
    "segment_sweep": list(range(85, 121)),
    "cross_validation": list(range(121, 141)),
    "seed_robustness": list(range(141, 145)),
    "task_comparison": list(range(145, 147)),
    "freq_resolution": list(range(147, 149)),
}

CATEGORY_LABELS = {
    "reference": "Reference",
    "ablation": "Ablation Study",
    "sensitivity": "Sensitivity Analysis",
    "combination": "Combination Experiments",
    "segment_sweep": "Segment Sweep",
    "cross_validation": "Cross Validation",
    "seed_robustness": "Seed Robustness",
    "task_comparison": "Task Comparison",
    "freq_resolution": "Frequency Resolution",
}

CATEGORY_DESCRIPTIONS = {
    "reference": "Baseline configurations: full DySTGAT (Phase~2) and DyEdgeGAT (temporal-only ablation).",
    "ablation": "Systematic removal of individual components to measure their contribution.",
    "sensitivity": "Single hyperparameter variations around the baseline configuration.",
    "combination": "Multi-parameter changes exploring combined effects.",
    "segment_sweep": "All 36 pairwise validation segment combinations with test segment fixed at 5.",
    "cross_validation": "10-fold cross-validation with varying test and validation segments.",
    "seed_robustness": "Random seed variation to assess training stability.",
    "task_comparison": "Reconstruction vs.\\ prediction task with different horizons.",
    "freq_resolution": "Frequency bin count variations for spectral feature extraction.",
}

# Metrics per fault type
SUB_METRICS = ["auc_roc", "fused_auc", "f1_score", "best_f1"]
SUB_METRIC_LABELS = {
    "auc_roc": "AUC",
    "fused_auc": "F-AUC",
    "f1_score": "F1",
    "best_f1": "F1*",
}

FAULT_TYPES = ["slugging", "blockage", "leakage", "diverted", "faults_all"]
FAULT_LABELS = {
    "slugging": "Slugging",
    "blockage": "Blockage",
    "leakage": "Leakage",
    "diverted": "Diverted",
    "faults_all": "All Faults",
}

# Categories that are large enough to need longtable
LARGE_CATEGORIES = {"sensitivity", "segment_sweep", "cross_validation"}


def extract_exp_id_and_name(dirname):
    m = re.match(r"exp(\d+)_(.*)", dirname)
    if m:
        return int(m.group(1)), m.group(2)
    return None, dirname


def _load_one_experiment(d, exp_id, exp_name):
    """Load metrics from a single experiment directory."""
    csvs = glob.glob(os.path.join(d, "*/plots/detailed_test_metrics.csv"))
    if not csvs:
        return None

    metrics = {}
    with open(csvs[0]) as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_set = row["test_set"]
            metrics[test_set] = {}
            for k in SUB_METRICS:
                if k in row:
                    metrics[test_set][k] = float(row[k])
                else:
                    metrics[test_set][k] = None

    train_csvs = glob.glob(os.path.join(d, "*/metrics.csv"))
    best_val = None
    epochs = 0
    if train_csvs:
        with open(train_csvs[0]) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            epochs = len(rows)
            if rows:
                best_val = min(float(r["val_loss"]) for r in rows)

    return {
        "name": exp_name,
        "dirname": os.path.basename(d),
        "metrics": metrics,
        "best_val_loss": best_val,
        "epochs": epochs,
    }


# Override specific experiments with results from separate runs
OVERRIDES = {
    1: "/mnt/datassd3/rashinda/DySTGAT/runs/pronto_phase2_full",
}


def load_results():
    results = {}
    exp_dirs = sorted(glob.glob(os.path.join(BASE, "exp*")))

    for d in exp_dirs:
        dirname = os.path.basename(d)
        exp_id, exp_name = extract_exp_id_and_name(dirname)
        if exp_id is None:
            continue

        # Use override path if available
        if exp_id in OVERRIDES:
            r = _load_one_experiment(OVERRIDES[exp_id], exp_id, exp_name)
            if r:
                results[exp_id] = r
                print(f"  exp{exp_id:02d} ({exp_name}): using override from {OVERRIDES[exp_id]}")
                continue

        r = _load_one_experiment(d, exp_id, exp_name)
        if r:
            results[exp_id] = r

    return results


def escape_latex(s):
    return s.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def format_val(val, is_best, is_second):
    s = f"{val:.4f}"
    if is_best:
        return f"\\textbf{{{s}}}"
    elif is_second:
        return f"\\underline{{{s}}}"
    return s


def find_best_second(values_dict):
    vals = [(eid, v) for eid, v in values_dict.items() if v is not None]
    if len(vals) < 2:
        return (vals[0][0] if vals else None), None
    vals.sort(key=lambda x: x[1], reverse=True)
    return vals[0][0], vals[1][0]


def build_rankings(results, valid_ids):
    """Build rankings dict: rankings[(fault_type, metric)] = (best_id, second_id)."""
    rankings = {}
    for ft in FAULT_TYPES:
        for m in SUB_METRICS:
            vals = {}
            for eid in valid_ids:
                r = results[eid]
                if ft in r["metrics"] and r["metrics"][ft].get(m) is not None:
                    vals[eid] = r["metrics"][ft][m]
            rankings[(ft, m)] = find_best_second(vals)
    return rankings


def make_header_rows():
    """Build the two-row header: fault type groups on top, AUC/F1 below."""
    # Top row: multicolumn spans for each fault type
    n_sub = len(SUB_METRICS)  # 2
    top_parts = [
        f"\\multicolumn{{1}}{{c}}{{}}",  # ID
        f"\\multicolumn{{1}}{{c}}{{}}",  # Experiment
    ]
    for ft in FAULT_TYPES:
        top_parts.append(f"\\multicolumn{{{n_sub}}}{{c}}{{\\textbf{{{FAULT_LABELS[ft]}}}}}")
    top_row = " & ".join(top_parts) + " \\\\"

    # cmidrule for each fault group
    cmidrules = []
    col_start = 3  # columns 1=ID, 2=Experiment, then data starts at 3
    for i, ft in enumerate(FAULT_TYPES):
        c_end = col_start + n_sub - 1
        cmidrules.append(f"\\cmidrule(lr){{{col_start}-{c_end}}}")
        col_start = c_end + 1
    cmidrule_row = " ".join(cmidrules)

    # Bottom row: AUC / F1 under each fault group
    bot_parts = ["\\textbf{ID}", "\\textbf{Experiment}"]
    for ft in FAULT_TYPES:
        for m in SUB_METRICS:
            bot_parts.append(f"\\textbf{{{SUB_METRIC_LABELS[m]}}}")
    bot_row = " & ".join(bot_parts) + " \\\\"

    return top_row, cmidrule_row, bot_row


def make_data_row(eid, results, rankings):
    """Build one data row for experiment eid."""
    r = results[eid]
    parts = [str(eid), escape_latex(r["name"])]
    for ft in FAULT_TYPES:
        for m in SUB_METRICS:
            if ft in r["metrics"] and r["metrics"][ft].get(m) is not None:
                val = r["metrics"][ft][m]
                best_id, second_id = rankings[(ft, m)]
                is_best = best_id == eid
                is_second = second_id == eid
                parts.append(format_val(val, is_best, is_second))
            else:
                parts.append("---")
    return " & ".join(parts) + " \\\\"


def col_spec(name_width="3.5cm"):
    """Column specification for the table."""
    n_data = len(FAULT_TYPES) * len(SUB_METRICS)  # 5 * 4 = 20
    return f"rp{{{name_width}}}" + "r" * n_data


def generate_table(results, exp_ids, category_name, use_longtable=False):
    """Generate a full table for one category."""
    valid_ids = [eid for eid in exp_ids if eid in results]
    if not valid_ids:
        return ""

    rankings = build_rankings(results, valid_ids)
    top_row, cmidrule_row, bot_row = make_header_rows()
    cat_label = CATEGORY_LABELS.get(category_name, category_name.replace("_", " ").title())
    cat_desc = CATEGORY_DESCRIPTIONS.get(category_name, "")

    lines = []
    lines.append(f"\\subsection{{{cat_label}}}")
    lines.append(f"{cat_desc} {len(valid_ids)} experiments.\n")

    if use_longtable:
        lines.append("\\tiny")
        lines.append(f"\\begin{{longtable}}{{{col_spec('2cm')}}}")
        # First header
        lines.append("\\toprule")
        lines.append(top_row)
        lines.append(cmidrule_row)
        lines.append(bot_row)
        lines.append("\\midrule")
        lines.append("\\endfirsthead")
        # Continuation header
        lines.append(f"\\multicolumn{{{2 + len(FAULT_TYPES) * len(SUB_METRICS)}}}{{l}}"
                     f"{{\\small\\textit{{\\tablename\\ \\thetable{{}}"
                     f" -- {cat_label} (continued)}}}} \\\\")
        lines.append("\\toprule")
        lines.append(top_row)
        lines.append(cmidrule_row)
        lines.append(bot_row)
        lines.append("\\midrule")
        lines.append("\\endhead")
        # Footer on continuation pages
        lines.append(f"\\midrule \\multicolumn{{{2 + len(FAULT_TYPES) * len(SUB_METRICS)}}}"
                     f"{{r}}{{\\small\\textit{{Continued on next page}}}} \\\\")
        lines.append("\\endfoot")
        # Last footer
        lines.append("\\bottomrule")
        lines.append("\\endlastfoot")
    else:
        lines.append("\\begin{table}[H]")
        lines.append("\\centering")
        lines.append("\\resizebox{\\textwidth}{!}{%")
        lines.append(f"\\begin{{tabular}}{{{col_spec()}}}")
        lines.append("\\toprule")
        lines.append(top_row)
        lines.append(cmidrule_row)
        lines.append(bot_row)
        lines.append("\\midrule")

    # Data rows
    for eid in valid_ids:
        lines.append(make_data_row(eid, results, rankings))

    if use_longtable:
        lines.append(f"\\caption{{{cat_label} --- Metrics per fault type. "
                     f"\\textbf{{Bold}} = best, \\underline{{underline}} = second best.}}")
        lines.append(f"\\label{{tab:{category_name}}}")
        lines.append("\\end{longtable}")
        lines.append("\\normalsize\n")
    else:
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}}")  # closes resizebox
        lines.append(f"\\caption{{{cat_label} --- Metrics per fault type. "
                     f"\\textbf{{Bold}} = best, \\underline{{underline}} = second best.}}")
        lines.append(f"\\label{{tab:{category_name}}}")
        lines.append("\\end{table}\n")

    return "\n".join(lines)


def generate_top_table(results, n=20):
    """Top N experiments ranked by faults_all AUC-ROC."""
    scored = []
    for eid, r in results.items():
        if "faults_all" in r["metrics"]:
            scored.append((eid, r))
    scored.sort(key=lambda x: x[1]["metrics"]["faults_all"]["auc_roc"], reverse=True)
    top = scored[:n]
    top_ids = [eid for eid, _ in top]

    rankings = build_rankings(results, top_ids)
    top_row, cmidrule_row, bot_row = make_header_rows()

    # Modify header to include Rank column
    n_sub = len(SUB_METRICS)
    top_row_parts = [
        f"\\multicolumn{{1}}{{c}}{{}}",  # Rank
        f"\\multicolumn{{1}}{{c}}{{}}",  # ID
        f"\\multicolumn{{1}}{{c}}{{}}",  # Experiment
    ]
    for ft in FAULT_TYPES:
        top_row_parts.append(f"\\multicolumn{{{n_sub}}}{{c}}{{\\textbf{{{FAULT_LABELS[ft]}}}}}")
    top_row_r = " & ".join(top_row_parts) + " \\\\"

    cmidrules_r = []
    col_start = 4
    for ft in FAULT_TYPES:
        c_end = col_start + n_sub - 1
        cmidrules_r.append(f"\\cmidrule(lr){{{col_start}-{c_end}}}")
        col_start = c_end + 1
    cmidrule_row_r = " ".join(cmidrules_r)

    bot_parts_r = ["\\textbf{Rank}", "\\textbf{ID}", "\\textbf{Experiment}"]
    for ft in FAULT_TYPES:
        for m in SUB_METRICS:
            bot_parts_r.append(f"\\textbf{{{SUB_METRIC_LABELS[m]}}}")
    bot_row_r = " & ".join(bot_parts_r) + " \\\\"

    n_data = len(FAULT_TYPES) * len(SUB_METRICS)
    col_spec_r = "rrl" + "r" * n_data

    lines = []
    lines.append("\\subsection{Overall Top 20 Experiments}")
    lines.append(f"Ranked by All Faults AUC-ROC across all {len(scored)} evaluated experiments.\n")
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{col_spec_r}}}")
    lines.append("\\toprule")
    lines.append(top_row_r)
    lines.append(cmidrule_row_r)
    lines.append(bot_row_r)
    lines.append("\\midrule")

    for rank, (eid, r) in enumerate(top, 1):
        parts = [str(rank), str(eid), escape_latex(r["name"])]
        for ft in FAULT_TYPES:
            for m in SUB_METRICS:
                if ft in r["metrics"] and r["metrics"][ft].get(m) is not None:
                    val = r["metrics"][ft][m]
                    best_id, second_id = rankings[(ft, m)]
                    is_best = best_id == eid
                    is_second = second_id == eid
                    parts.append(format_val(val, is_best, is_second))
                else:
                    parts.append("---")
        lines.append(" & ".join(parts) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}}")  # closes resizebox
    lines.append("\\caption{Top 20 experiments ranked by All Faults AUC-ROC. "
                 "\\textbf{Bold} = best, \\underline{underline} = second best within top 20.}")
    lines.append("\\label{tab:top20}")
    lines.append("\\end{table}\n")

    return "\n".join(lines)


BASELINE_DIR = "/mnt/datassd3/rashinda/DySTGAT/runs/pronto_baselines"
BASELINE_METHODS = ["ae", "fnn", "lstm", "lstm_ae", "usad", "gdn", "mtad_gat", "grelen", "dyedgegat"]
BASELINE_DISPLAY_NAMES = {
    "ae": "AE",
    "fnn": "FNN",
    "lstm": "LSTM",
    "lstm_ae": "LSTM-AE",
    "usad": "USAD",
    "gdn": "GDN",
    "mtad_gat": "MTAD-GAT",
    "grelen": "GReLEN",
    "dyedgegat": "DyEdgeGAT",
    "dystgat": "DySTGAT",
}


def load_baseline_metrics():
    """Load all baseline method metrics from pronto_baselines CSVs."""
    all_methods = {}
    for method in BASELINE_METHODS:
        csv_path = os.path.join(BASELINE_DIR, f"{method}_metrics.csv")
        if not os.path.exists(csv_path):
            print(f"  WARNING: {csv_path} not found, skipping")
            continue
        metrics = {}
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = row["test_set"]
                metrics[ts] = {
                    "auc_roc": float(row["auc_roc"]),
                    "fused_auc": None,  # baselines don't have fused_auc
                    "f1_score": float(row["f1"]),  # baseline uses "f1" not "f1_score"
                    "best_f1": float(row["best_f1"]),
                }
        all_methods[method] = metrics
    return all_methods


def generate_baseline_comparison_table(dystgat_metrics):
    """Generate DySTGAT vs baselines comparison table."""
    baseline_data = load_baseline_metrics()

    # Combine: baselines + DySTGAT
    # Use ordered list: DySTGAT first, then baselines
    method_order = ["dystgat"] + BASELINE_METHODS
    all_data = {"dystgat": dystgat_metrics}
    all_data.update(baseline_data)

    # Build rankings across all methods
    rankings = {}
    for ft in FAULT_TYPES:
        for m in SUB_METRICS:
            vals = {}
            for method in method_order:
                if method in all_data and ft in all_data[method]:
                    v = all_data[method][ft].get(m)
                    if v is not None:
                        vals[method] = v
            # Find best and second best
            if len(vals) < 2:
                best = list(vals.keys())[0] if vals else None
                rankings[(ft, m)] = (best, None)
            else:
                sorted_vals = sorted(vals.items(), key=lambda x: x[1], reverse=True)
                rankings[(ft, m)] = (sorted_vals[0][0], sorted_vals[1][0])

    n_sub = len(SUB_METRICS)

    # Header rows
    top_parts = [f"\\multicolumn{{1}}{{c}}{{}}"]  # Method
    for ft in FAULT_TYPES:
        top_parts.append(f"\\multicolumn{{{n_sub}}}{{c}}{{\\textbf{{{FAULT_LABELS[ft]}}}}}")
    top_row = " & ".join(top_parts) + " \\\\"

    cmidrules = []
    col_start = 2
    for ft in FAULT_TYPES:
        c_end = col_start + n_sub - 1
        cmidrules.append(f"\\cmidrule(lr){{{col_start}-{c_end}}}")
        col_start = c_end + 1
    cmidrule_row = " ".join(cmidrules)

    bot_parts = ["\\textbf{Method}"]
    for ft in FAULT_TYPES:
        for m in SUB_METRICS:
            bot_parts.append(f"\\textbf{{{SUB_METRIC_LABELS[m]}}}")
    bot_row = " & ".join(bot_parts) + " \\\\"

    n_data = len(FAULT_TYPES) * n_sub
    cspec = "l" + "r" * n_data

    lines = []
    lines.append("\\subsection{DySTGAT vs.\\ Baseline Methods}")
    lines.append("Comparison of DySTGAT (ID~1, Phase~2 full run) against 9 baseline methods on PRONTO.\n")
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{cspec}}}")
    lines.append("\\toprule")
    lines.append(top_row)
    lines.append(cmidrule_row)
    lines.append(bot_row)
    lines.append("\\midrule")

    for method in method_order:
        if method not in all_data:
            continue
        mdata = all_data[method]
        display = BASELINE_DISPLAY_NAMES.get(method, method)
        parts = [f"\\textbf{{{display}}}" if method == "dystgat" else display]
        for ft in FAULT_TYPES:
            for m in SUB_METRICS:
                if ft in mdata and mdata[ft].get(m) is not None:
                    val = mdata[ft][m]
                    best_method, second_method = rankings[(ft, m)]
                    is_best = best_method == method
                    is_second = second_method == method
                    parts.append(format_val(val, is_best, is_second))
                else:
                    parts.append("---")
        lines.append(" & ".join(parts) + " \\\\")
        # Horizontal line after DySTGAT to separate from baselines
        if method == "dystgat":
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}}")  # closes resizebox
    lines.append("\\caption{DySTGAT vs.\\ baseline methods on PRONTO. "
                 "\\textbf{Bold} = best, \\underline{underline} = second best across all methods. "
                 "F-AUC is only available for DySTGAT (divergence-augmented scoring).}")
    lines.append("\\label{tab:baseline_comparison}")
    lines.append("\\end{table}\n")

    return "\n".join(lines)


def main():
    results = load_results()
    print(f"Loaded {len(results)} experiments with metrics")

    for cat, ids in CATEGORIES.items():
        valid = [i for i in ids if i in results]
        print(f"  {cat}: {len(valid)}/{len(ids)}")

    doc = []
    doc.append(r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=1.0cm,landscape]{geometry}
\usepackage{booktabs}
\usepackage{float}
\usepackage{caption}
\usepackage{longtable}
\usepackage{graphicx}
\usepackage[hidelinks]{hyperref}
\setlength{\tabcolsep}{3pt}

\title{DySTGAT Hyperparameter Search Report\\Phase 2 --- PRONTO Dataset}
\author{Auto-generated}
\date{\today}

\begin{document}
\maketitle
\tableofcontents
\newpage

\section{Summary}

This report summarizes 148 hyperparameter search experiments for DySTGAT on the PRONTO benchmark dataset.
The search covers ablation studies, sensitivity analysis, combination experiments, segment sweep/cross-validation,
seed robustness, task comparison, and frequency resolution studies.

Each table shows per-fault-type results with four metrics:
\begin{itemize}
  \item \textbf{AUC}: Area Under ROC Curve
  \item \textbf{F-AUC}: Fused AUC (divergence-augmented, DySTGAT only)
  \item \textbf{F1}: F1 score at validation-derived threshold
  \item \textbf{F1*}: Best achievable F1 score (oracle threshold)
\end{itemize}

Within each table, the \textbf{best} value per metric column is bold and the \underline{second best} is underlined.

\newpage
""")

    # Overall top 20
    doc.append("\\section{Overall Rankings}\n")
    doc.append(generate_top_table(results))
    doc.append("\\newpage\n")

    # Per-category tables
    doc.append("\\section{Results by Category}\n")

    category_order = [
        "reference", "ablation", "sensitivity", "combination",
        "seed_robustness", "task_comparison", "freq_resolution",
        "segment_sweep", "cross_validation",
    ]

    for cat in category_order:
        ids = CATEGORIES[cat]
        use_long = cat in LARGE_CATEGORIES
        doc.append(generate_table(results, ids, cat, use_longtable=use_long))
        if not use_long:
            doc.append("\\newpage\n")

    # Baseline comparison table (last page)
    doc.append("\\newpage\n")
    doc.append("\\section{Baseline Comparison}\n")
    if 1 in results:
        dystgat_metrics = results[1]["metrics"]
        doc.append(generate_baseline_comparison_table(dystgat_metrics))
    else:
        doc.append("\\textit{DySTGAT ID~1 results not available.}\n")

    doc.append("\\end{document}\n")

    outpath = "/mnt/datassd3/rashinda/DySTGAT/results/pronto/hpsearch_phase2_report.tex"
    with open(outpath, "w") as f:
        f.write("\n".join(doc))
    print(f"\nLaTeX report written to: {outpath}")


if __name__ == "__main__":
    main()
