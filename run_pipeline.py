"""Quarry base-terrain pipeline orchestrator.

  python run_pipeline.py              run stages 00-06, pausing after each stage
  python run_pipeline.py --auto       no pauses
  python run_pipeline.py --from 03    run 03..06
  python run_pipeline.py --only 05    run only 05
  python run_pipeline.py --new-run    start a new vNNN even if the config is unchanged

Each stage skips its work when its outputs exist for the same config hash.
Everything is logged to logs/run_vNNN.log. The pipeline stops at the first
failed stage.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
import common  # noqa: E402


def run_stage(stage: str, config: str, run_id: str | None, force: bool) -> tuple[int, list[str]]:
    script = ROOT / "scripts" / common.STAGE_SCRIPTS[stage]
    cmd = [sys.executable, str(script), "--config", config]
    if stage != "00":
        cmd += ["--run", run_id]
        if force:
            cmd.append("--force")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            bufsize=1)
    lines = []
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        lines.append(line.rstrip("\n"))
    return proc.wait(), lines


def explain_failure(lines: list[str]) -> str:
    errs =[ln for ln in lines if " ERROR " in ln or ln.startswith("ERROR")]
    if errs:
        return errs[-1].split(" ERROR ", 1)[-1]
    tail = [ln for ln in lines if ln.strip()][-12:]
    return "\n".join(tail)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--from", dest="start", choices=common.STAGES)
    ap.add_argument("--only", choices=common.STAGES)
    ap.add_argument("--auto", action="store_true", help="do not pause between stages")
    ap.add_argument("--new-run", action="store_true", help="start a new run ID")
    ap.add_argument("--force", action="store_true", help="redo stages even if up to date "
                    "(terrain/00_base stays write-once)")
    args = ap.parse_args()

    if args.only:
        stages = [args.only]
    elif args.start:
        stages = common.STAGES[common.STAGES.index(args.start):]
    else:
        stages = list(common.STAGES)

    pre_log: list[str] = []
    if "00" in stages:
        print("\n=== Stage 00 - Environment check ===")
        rc, lines = run_stage("00", args.config, None, False)
        pre_log += lines
        if rc != 0:
            print("\nStopped at stage 00: the environment is not ready. Fix the items listed "
                  "above (the pipeline asks rather than guesses), then run again.")
            return 1
        stages = [s for s in stages if s != "00"]
        if not stages:
            return 0

    try:
        cfg = common.load_config(args.config)
        common.require_aoi(cfg)
        run_id = common.resolve_run_id(cfg, new_run=args.new_run)
        paths = common.init_run(cfg, run_id)
    except common.PipelineError as e:
        print(f"\nStopped: {e}")
        return 1

    log = common.setup_logging(paths, "pipeline")
    log.info("=== Run %s (config hash %s) - stages %s%s ===", run_id, common.config_hash(cfg),
             ", ".join(stages), " [auto]" if args.auto else "")
    if pre_log:
        with open(paths.log_file, "a", encoding="utf-8") as fh:
            fh.write("\n".join(f"[00 check_env] {ln}" for ln in pre_log) + "\n")

    for i, stage in enumerate(stages):
        title = common.STAGE_TITLES[stage]
        print(f"\n=== Stage {stage} - {title} ({run_id}) ===")
        if not args.force and common.stage_up_to_date(paths, cfg, stage):
            log.info("Stage %s up to date for this config - skipped.", stage)
            continue
        t0 = time.time()
        rc, lines = run_stage(stage, args.config, run_id, args.force)
        if rc != 0:
            log.error("Stage %s (%s) FAILED after %.0f s.", stage, title, time.time() - t0)
            print(f"\nStopped at stage {stage} ({title}). Reason:\n  {explain_failure(lines)}")
            print(f"Full log: {common.rel(paths, paths.log_file)}")
            return 1
        log.info("Stage %s finished in %.0f s.", stage, time.time() - t0)

        rec = common.read_json(paths.stage_record(stage))
        last = i == len(stages) - 1
        if not args.auto and not last:
            print(f"\n--- Checkpoint after stage {stage}: review these check images ---")
            for p in rec.get("check_images", []):
                print(f"    {paths.root / p}")
            nxt = stages[i + 1]
            if sys.stdin.isatty():
                ans = input(f"Continue with stage {nxt}? [Y/n] ").strip().lower()
                if ans in ("n", "no", "q"):
                    log.info("Paused by user after stage %s.", stage)
                    print(f"Paused. Resume with: python run_pipeline.py --from {nxt}")
                    return 0
            else:
                log.info("Checkpoint after stage %s (non-interactive): stopping.", stage)
                print(f"Paused for review. Resume with: python run_pipeline.py --from {nxt}"
                      "   (or use --auto to skip checkpoints)")
                return 0

    log.info("=== Run %s complete ===", run_id)
    report = paths.base / "report.md"
    if report.exists():
        print(f"\nDone. Report: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
