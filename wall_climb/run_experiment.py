"""完整课程 + 三项消融 + 固定终点评估。各独立进程训练，无脚本模仿。"""
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
import config as C
from evaluate import evaluate


def run(directory=C.SUITE_DIR, seed=C.SEED, iterations=C.TOTAL_ITERS):
    if iterations < C.ITER_FAIL_END + 1:
        raise ValueError("complete experiment must include iteration 35000")
    out = Path(directory).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise FileExistsError("Use a new --output directory to preserve previous experiment evidence")
    source = Path(__file__).resolve().parent
    root = source.parent
    snapshot = out / "source"
    snapshot.mkdir()
    hashes = {}
    for file in list(source.glob("*.py")) + [root / "launch_2d.py", root / "run.ps1"]:
        shutil.copy2(file, snapshot / file.name)
        hashes[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    for file in root.glob("*.pdf"):
        hashes[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "training", "seed": seed, "iterations": iterations,
        "samples_per_model": iterations * C.ROLLOUT_STEPS,
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "numpy": np.__version__,
        "config": {k: v for k, v in vars(C).items() if k.isupper()},
        "source_sha256": hashes, "selection": "fixed final checkpoint; no best-checkpoint selection",
    }
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    variants = ("full", "no_curriculum", "no_probabilistic", "no_modeling")
    processes = {}
    logfiles = []
    start = time.monotonic()
    try:
        for variant in variants:
            folder = out / variant
            folder.mkdir()
            log = (folder / "stdout.log").open("w", encoding="utf-8")
            logfiles.append(log)
            command = [sys.executable, "-B", "-X", "utf8", "-u", str(root / "launch_2d.py"),
                       "--train", "--variant", variant, "--seed", str(seed),
                       "--iters", str(iterations), "--save-dir", str(folder)]
            command += ["--num-envs", str(C.NUM_ENVS)]
            processes[variant] = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                  cwd=root)
        manifest["training_process_ids"] = {name: process.pid for name,process in processes.items()}
        path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
        last_report = -100.0
        while any(process.poll() is None for process in processes.values()):
            elapsed = time.monotonic() - start
            if elapsed - last_report >= 30:
                progress = {}
                for variant, process in processes.items():
                    file = out / variant / "training.csv"
                    latest = "starting"
                    if file.exists():
                        with file.open("rb") as handle:
                            handle.seek(max(0, file.stat().st_size - 2048))
                            lines = handle.read().decode("utf-8", errors="replace").splitlines()
                            if lines:
                                latest = lines[-1].split(",")[0]
                    progress[variant] = {"iteration": latest, "exit": process.poll()}
                print(f"elapsed {elapsed:.0f}s {progress}", flush=True)
                last_report = elapsed
            time.sleep(1)
        failed = {v: p.returncode for v, p in processes.items() if p.returncode}
        for variant in variants:
            checkpoint = out / variant / "final.pt"
            if not checkpoint.exists():
                failed[variant] = "missing final checkpoint (possibly interrupted)"
            elif torch.load(checkpoint, map_location="cpu").get("iteration") != iterations - 1:
                failed[variant] = "incomplete training iteration"
        if failed:
            manifest.update(status="failed_or_interrupted", errors=failed)
            path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            raise RuntimeError(f"Training failed; inspect stdout.log: {failed}")
    finally:
        for log in logfiles:
            log.close()
    manifest["training_seconds"] = time.monotonic() - start
    manifest["status"] = "evaluating"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    results = {}
    for variant in variants:
        for probability in (1.0, .85):
            name = f"{variant}_p{probability}"
            args = SimpleNamespace(episodes=100, disturb_at=None, seed=10000, demo=False,
                                   checkpoint=str(out / variant / "final.pt"), render=False,
                                   theta=90.0, p_attach=probability, stochastic=False,
                                   output=str(out / f"{name}.json"))
            results[name] = evaluate(args)
    for probability in (1.0, .85):
        name = f"scripted_p{probability}"
        args = SimpleNamespace(episodes=100, disturb_at=None, seed=10000, demo=True,
                               checkpoint=None, render=False, theta=90.0, p_attach=probability,
                               stochastic=False, output=str(out / f"{name}.json"))
        results[name] = evaluate(args)
    args = SimpleNamespace(episodes=100, disturb_at=3.0, seed=10000, demo=False,
                           checkpoint=str(out / "full" / "final.pt"), render=False,
                           theta=90.0, p_attach=.85, stochastic=False,
                           output=str(out / "full_forced_slip.json"))
    results["full_forced_slip"] = evaluate(args)
    manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                    total_seconds=time.monotonic() - start)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Complete experiment saved to {out}", flush=True)
    return results
