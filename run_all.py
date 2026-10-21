"""
Run the experiment pipeline end to end, or any subset of its stages.

Stages (run in this order):
    sanity       experiments/sanity_checks/*      -> sanity_check_results/
    preliminary  experiments/preliminary_tests/*  -> results/
    full         experiments/full_tests/*         -> results/   (the results analysed in the report)
    plots        analysis/plot_main.py            -> plots/
    tables       analysis/print_results.py        -> results_display/

Examples:
    python run_all.py                    # full tests + plots (reproduces the report figures)
    python run_all.py plots              # only regenerate plots from the saved results
    python run_all.py full --parallel    # run the three full test scripts concurrently
    python run_all.py all --dry-run      # list everything that would run

Warning: the experiments are slow. The full tests took roughly a day on an M3 MacBook Pro,
with the guided tests being the longest.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STAGES = {
    "sanity": sorted((ROOT / "experiments" / "sanity_checks").glob("*.py")),
    "preliminary": sorted((ROOT / "experiments" / "preliminary_tests").glob("*.py")),
    "full": sorted((ROOT / "experiments" / "full_tests").glob("*.py")),
    "plots": [ROOT / "analysis" / "plot_main.py"],
    "tables": [ROOT / "analysis" / "print_results.py"],
}
DEFAULT_STAGES = ["full", "plots"]


def script_env():
    # make src/ importable even if the package was not installed with `pip install -e .`
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT / "src"), env.get("PYTHONPATH")]))
    return env


def run_sequential(scripts):
    for script in scripts:
        print(f"\n=== Running {script.relative_to(ROOT)} ===", flush=True)
        start = time.time()
        result = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=script_env())
        if result.returncode != 0:
            sys.exit(f"{script.relative_to(ROOT)} failed with exit code {result.returncode}")
        print(f"=== Finished {script.relative_to(ROOT)} in {time.time() - start:.0f}s ===", flush=True)


def run_parallel(scripts):
    # each script writes to its own log so the outputs don't interleave
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    procs = []
    for script in scripts:
        log_path = log_dir / f"{script.stem}.log"
        print(f"Starting {script.relative_to(ROOT)} (log: {log_path.relative_to(ROOT)})", flush=True)
        log = open(log_path, "w")
        proc = subprocess.Popen([sys.executable, "-u", str(script)], cwd=ROOT, env=script_env(),
                                stdout=log, stderr=subprocess.STDOUT)
        procs.append((script, proc, log))

    failed = []
    for script, proc, log in procs:
        proc.wait()
        log.close()
        status = "finished" if proc.returncode == 0 else f"failed with exit code {proc.returncode}"
        print(f"{script.relative_to(ROOT)} {status}", flush=True)
        if proc.returncode != 0:
            failed.append(script)
    if failed:
        sys.exit(f"{len(failed)} script(s) failed, see logs/ for details")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("stages", nargs="*", metavar="stage",
                        help=f"one or more of: {', '.join(STAGES)}, all (default: {' '.join(DEFAULT_STAGES)})")
    parser.add_argument("--parallel", action="store_true",
                        help="run the scripts within each stage concurrently")
    parser.add_argument("--dry-run", action="store_true",
                        help="list the scripts that would run without running them")
    args = parser.parse_args()

    requested = args.stages or DEFAULT_STAGES
    # validated here rather than with argparse choices, which rejects an empty list on older Pythons
    unknown = [s for s in requested if s not in STAGES and s != "all"]
    if unknown:
        parser.error(f"unknown stage(s): {', '.join(unknown)} (choose from {', '.join(STAGES)}, all)")
    # always run stages in pipeline order, regardless of the order they were given in
    stages = [s for s in STAGES if "all" in requested or s in requested]

    for stage in stages:
        scripts = STAGES[stage]
        print(f"\n##### Stage: {stage} ({len(scripts)} script(s)) #####", flush=True)
        if args.dry_run:
            for script in scripts:
                print(f"  {script.relative_to(ROOT)}")
        elif args.parallel and len(scripts) > 1:
            run_parallel(scripts)
        else:
            run_sequential(scripts)


if __name__ == "__main__":
    main()
