import subprocess, datetime, json
from discover_music.evaluation.harness import run_eval
from discover_music.utils.paths import EVAL_RESULTS_DIR, ensure_eval_dirs

def print_summary(results):
    """Prints a readable table of every metric's mean and std to the terminal."""
    print(f"\n  {'source / metric':<32} {'mean':>8}  {'std':>8}")
    print("  " + "-" * 52)
    for source, metric_dict in results.items():
        print(f"\n  {source}")
        for metric, vals in metric_dict.items():
            print(f"    {metric:<28} {vals['mean']:>8.3f}  {vals['std']:>8.3f}")

ensure_eval_dirs()
results = run_eval(seed=42)
sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
stamp = datetime.datetime.now().isoformat(timespec="seconds")
(EVAL_RESULTS_DIR / f"{sha}_{stamp}.json").write_text(json.dumps(results, indent=2))
print_summary(results)          # A, B, A−B vs random/popularity