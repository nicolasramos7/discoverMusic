import argparse, csv, subprocess, datetime, json
from discover_music.evaluation.harness import run_eval, load_personas
from discover_music.utils.paths import EVAL_RESULTS_DIR, ensure_eval_dirs

SUMMARY_CSV = EVAL_RESULTS_DIR / "summary.csv"

def print_summary(results):
    """Prints a readable table of every metric's mean and std to the terminal"""
    print(f"\n  {'source / metric':<32} {'mean':>8}  {'std':>8}")
    print("  " + "-" * 52)
    for source, metric_dict in results.items():
        print(f"\n  {source}")
        for metric, vals in metric_dict.items():
            print(f"    {metric:<28} {vals['mean']:>8.3f}  {vals['std']:>8.3f}")


def flatten_row(results, stage, git_sha, stamp, seed, n_personas):
    """Turns the nested results dict into one flat {column: value} row for the trend CSV.

    Wide format: one column per (source, metric). The `gap_A_minus_B` source is
    shortened to a `gap_` prefix so headline columns stay readable in Sheets.
    """
    row = {
        "stage":       stage,
        "timestamp":   stamp,
        "git_sha":     git_sha,
        "seed":        seed,
        "n_personas":  n_personas,
    }
    for source, metric_dict in results.items():
        prefix = "gap" if source == "gap_A_minus_B" else source
        for metric, vals in metric_dict.items():
            row[f"{prefix}_{metric}"]        = round(vals["mean"], 4)
            row[f"{prefix}_{metric}_std"]    = round(vals["std"], 4)
    return row


def append_summary(row):
    """Appends one run to results/summary.csv, writing the header only on first run."""
    write_header = not SUMMARY_CSV.exists()
    with open(SUMMARY_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the evaluation suite and log results.")
    parser.add_argument(
        "--stage",
        default=None,
        help="Human-readable label for this run, used as the chart x-axis "
             "(e.g. 'baseline', '+genre-weights'). Defaults to the git short SHA.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_eval_dirs()

    sha   = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True).stdout.strip()
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    stage = args.stage or sha

    results = run_eval(seed=args.seed)

    # full nested archive (unchanged)
    (EVAL_RESULTS_DIR / f"{sha}_{stamp}.json").write_text(json.dumps(results, indent=2))

    # flat, append-only trend log — the thing you plot
    row = flatten_row(results, stage, sha, stamp, args.seed, len(load_personas()))
    append_summary(row)

    print_summary(results)          # A, B, A−B vs random/popularity
    print(f"\n  logged stage '{stage}' → {SUMMARY_CSV}")
