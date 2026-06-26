#!/usr/bin/env python3
"""
Generic automated pipeline: train -> render -> metrics for any configured dataset,
with multi-GPU distribution and per-step retries.

On startup, run_all snapshots `constants.py` (as if `train.py` with no extra args) and passes
those values explicitly to every train/render/metrics subprocess, so editing `constants.py`
mid-run does not change behavior for the rest of this run. Use `--no_freeze_constants` to
restore the old behavior (each subprocess reloads `constants.py` from disk).

Usage (from repo root):
  python scripts/run_all.py --dataset TensoIR --cuda_devices 1,2,3
  python scripts/run_all.py --dataset TensoIR --run_dir 2026-03-20_19-39-44 --resume
  python scripts/run_all.py --list_datasets
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Run from repo root so train.py, render.py, metrics.py and outputs/ resolve correctly
REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)

# ---------- Dataset configs (aligned with train_*.sh) ----------
DATASET_CONFIG = {
    "TensoIR": {
        "dataset_type": "TensoIR",
        "dataset_root": "/path/to/datasets/TensoIR",
        "output_root": "./outputs/TensoIR",
        "default_scenes": ["lego", "hotdog", "armadillo", "ficus"],
        "scene_params": {
            "lego": {"offset": "0.1", "geometry_threshold": "0.5"},
            "hotdog": {"offset": "0.1", "geometry_threshold": "0.5"},
            "armadillo": {"offset": "0.1", "geometry_threshold": "0.5"},
            "ficus": {"offset": "0.1", "geometry_threshold": "0.5"},
        },
        "default_iteration": 799,
        "enable_relight": True,
        "envmap_root": "/path/to/datasets/TensoIR/Environment_Maps",
    },
    "Synthetic4Relight": {
        "dataset_type": "Synthetic4Relight",
        "dataset_root": "/path/to/datasets/Synthetic4Relight",
        "output_root": "./outputs/Synthetic4Relight",
        "default_scenes": ["jugs", "chair", "air_baloons", "hotdog"],
        "scene_params": {
            "jugs": {"offset": "0.1", "geometry_threshold": "0.5"},
            "chair": {"offset": "0.1", "geometry_threshold": "0.5"},
            "air_baloons": {"offset": "0.1", "geometry_threshold": "0.5"},
            "hotdog": {"offset": "0.1", "geometry_threshold": "0.5"},
        },
        "default_iteration": 799,
        "enable_relight": True,
        "envmap_root": "/path/to/datasets/Synthetic4Relight/Environment_Maps",
    },
    "RT4Relight": {
        "dataset_type": "RT4Relight",
        "dataset_root": "/path/to/datasets/RT4Relight",
        "output_root": "./outputs/RT4Relight",
        "default_scenes": ["barrels", "plastica"],
        "scene_params": {
            "bread": {"offset": "0.1", "geometry_threshold": "0.5"},
            "toybus": {"offset": "0.1", "geometry_threshold": "0.5"},
            "teacup": {"offset": "0.1", "geometry_threshold": "0.5"},
            "bear": {"offset": "0.1", "geometry_threshold": "0.5"},
            "barrels": {"offset": "0.1", "geometry_threshold": "0.5"},
            "plastica": {"offset": "0.1", "geometry_threshold": "0.5"},
        },
        "default_iteration": 799,
        "enable_relight": True,
        "envmap_root": "/path/to/datasets/RT4Relight/Environment_Maps",
    },
}


def get_base_scene(scene: str) -> str:
    return re.sub(r"[0-9]*$", "", scene)


def get_scene_params(scene: str, cfg: dict) -> dict:
    base = get_base_scene(scene)
    params = cfg.get("scene_params") or {}
    p = params.get(base, {})
    return {
        "offset": p.get("offset", "0.5"),
        "geometry_threshold": p.get("geometry_threshold", "0.3"),
    }


# ---------- Freeze constants.py for whole run_all session ----------
# train.py / render.py / metrics.py import `constants` on each subprocess start; if you edit
# constants.py mid-run, later scenes would see the new file. We snapshot once at startup
# and pass every flag explicitly so later edits do not affect this run.
_STORE_TRUE_FLAGS = frozenset({"relight", "only_relight"})


def _load_constants_snapshot(repo_root: Path) -> dict:
    """Load `constants.args` as if running `train.py` with no extra argv (isolated subprocess)."""
    code = r"""
import json, os, sys
os.chdir(os.environ["REPO_ROOT"])
sys.argv = ["train.py"]
import constants
print(json.dumps(vars(constants.args), default=str))
"""
    env = os.environ.copy()
    env["REPO_ROOT"] = str(repo_root.resolve())
    r = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout or "").strip() or "constants snapshot failed")
    out = (r.stdout or "").strip()
    if not out:
        raise RuntimeError("constants snapshot produced empty stdout")
    # Prefer last line (single JSON object); tolerate stray leading lines
    for line in reversed(out.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    return json.loads(out)


def _namespace_to_cli_args(ns: dict) -> list[str]:
    """Turn argparse Namespace dict into CLI tokens. store_true flags: only emit when True."""
    out: list[str] = []
    for k in sorted(ns.keys()):
        if k.startswith("_"):
            continue
        v = ns[k]
        if v is None:
            continue
        if k in _STORE_TRUE_FLAGS:
            if v:
                out.append(f"--{k}")
            continue
        if isinstance(v, bool):
            out.extend([f"--{k}", "True" if v else "False"])
        elif isinstance(v, (int, float)):
            out.extend([f"--{k}", str(v)])
        else:
            out.extend([f"--{k}", str(v)])
    return out


def _frozen_constants_cli(cfg: dict) -> list[str]:
    return list(cfg.get("frozen_constants_cli") or [])


def _resolve_run_root(run_dir_arg: Optional[str], base_output_root: Path) -> Tuple[Path, str]:
    """Return (run_root, run_id_label). If run_dir_arg is None, create a new timestamp folder."""
    base_output_root = base_output_root.resolve()
    if not run_dir_arg or not str(run_dir_arg).strip():
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return base_output_root / run_id, run_id
    p = Path(run_dir_arg.strip()).expanduser()
    if p.is_absolute():
        run_root = p.resolve()
    elif len(p.parts) == 1:
        # e.g. 2026-03-20_19-39-44 -> <outputs>/2026-03-20_19-39-44
        run_root = (base_output_root / p).resolve()
    else:
        # e.g. outputs/2026-03-20_19-39-44 from repo root
        run_root = (REPO_ROOT / p).resolve()
    return run_root, run_root.name


def _parse_train_progress(log_content: str, total_hint: int) -> tuple[Optional[int], Optional[int], dict]:
    """Parse current/total and optional metrics (rgb, albedo, etc.) from train stdout. Returns (current, total, metrics_dict)."""
    metrics = {}
    progress_m = list(re.finditer(r"(\d+)/(\d+)", log_content))
    current, total = None, None
    if progress_m:
        for m in progress_m:
            c, t = int(m.group(1)), int(m.group(2))
            if 10 <= t <= 20000:
                current, total = c, t
    if total is None and total_hint and total_hint > 0:
        total = total_hint
        for m in re.finditer(r"[Ii]ter[_\s]+(\d+)", log_content):
            current = int(m.group(1))
    for name in ("rgb", "albedo", "roughness", "normal"):
        pat = re.search(rf"[\'\"]?{name}[\'\"]?\s*[=:]\s*([\d.]+)", log_content, re.I)
        if pat:
            try:
                metrics[name] = float(pat.group(1))
            except ValueError:
                pass
    return current, total, metrics


def run_train(scene: str, cuda_device: str, iteration: int, resume: bool, cfg: dict) -> tuple[bool, str]:
    base = get_base_scene(scene)
    params = get_scene_params(scene, cfg)
    output_root = Path(cfg["output_root"])
    ply_dir = output_root / scene / "ply"
    do_resume = resume and ply_dir.exists() and bool(list(ply_dir.glob("iter_*.ply")))
    cmd = [
        sys.executable, "train.py",
        *_frozen_constants_cli(cfg),
        "--dataset_type", cfg["dataset_type"],
        "--dataset_name", scene,
        "--dataset_path", f"{cfg['dataset_root']}/{base}",
        "--output_root", cfg["output_root"],
        "--selfocc_offset_max", params["offset"],
        "--geometry_threshold", params["geometry_threshold"],
    ]
    if do_resume:
        cmd.append("--resume")
    else:
        cmd.extend(["--ply_path", f"{cfg['dataset_root']}/3dgrt/{base}_refined.ply"])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_device
    run_log_path = cfg.get("run_log_path")
    total_hint = int(iteration) + 1 if iteration else 800
    progress_interval = 50
    buffer: list[str] = []
    buffer_max = 100000
    last_logged = [-1]

    def write_progress(current: int, total: int, metrics: dict) -> None:
        if run_log_path and current - last_logged[0] >= progress_interval:
            last_logged[0] = current
            parts = [f"[{scene}] {current}/{total}"]
            if metrics:
                parts.append(" " + " ".join(f"{k}={v:.1f}" for k, v in sorted(metrics.items())))
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                with open(run_log_path, "a", encoding="utf-8") as lf:
                    lf.write(f"[{ts}] " + "".join(parts) + "\n")
            except Exception:
                pass

    def reader() -> None:
        while True:
            line = p.stdout.readline()
            if not line and p.poll() is not None:
                break
            buffer.append(line)
            content = "".join(buffer)
            if len(content) > buffer_max:
                buffer[:] = [content[-buffer_max:]]
            current, total, metrics = _parse_train_progress(content, total_hint)
            if current is not None:
                if total is None:
                    total = total_hint
                write_progress(current, total, metrics)

    try:
        p = subprocess.Popen(
            cmd, env=env, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        t_start = time.perf_counter()
        timeout_sec = 3600 * 24
        reader_thread = threading.Thread(target=reader, daemon=True)
        reader_thread.start()
        try:
            while p.poll() is None:
                if time.perf_counter() - t_start > timeout_sec:
                    p.kill()
                    p.wait()
                    return False, "Timeout"
                time.sleep(30)
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
            return False, "Timeout"
        reader_thread.join(timeout=2)
        if p.returncode != 0:
            return False, "train failed (check run.log)"
        return True, ""
    except Exception as e:
        return False, str(e)


def run_render(scene: str, cuda_device: str, iteration: int, cfg: dict) -> tuple[bool, str]:
    base = get_base_scene(scene)
    params = get_scene_params(scene, cfg)
    output_root = Path(cfg["output_root"])
    ply_dir = output_root / scene / "ply"
    envmap_dir = output_root / scene / "envmap"
    expected_ply = ply_dir / f"iter_{iteration}.ply"
    if expected_ply.exists():
        ply_path = str(expected_ply)
        iter_num = iteration
    else:
        plies = sorted(ply_dir.glob("iter_*.ply"), key=lambda p: int(re.search(r"iter_(\d+)\.ply", p.name).group(1)))
        if not plies:
            return False, f"No ply in {ply_dir}"
        ply_path = str(plies[-1])
        iter_num = int(re.search(r"iter_(\d+)\.ply", plies[-1].name).group(1))
    iter_padded = f"{iter_num:04d}"
    envmap_path = envmap_dir / f"optimized_sgs_{iter_padded}.npy"
    if not envmap_path.exists():
        envmaps = sorted(envmap_dir.glob("optimized_sgs_*.npy"), key=lambda p: p.name)
        envmap_path = envmaps[-1] if envmaps else envmap_path
    cmd = [
        sys.executable, "render.py",
        *_frozen_constants_cli(cfg),
        "--dataset_type", cfg["dataset_type"],
        "--dataset_name", scene,
        "--dataset_path", f"{cfg['dataset_root']}/{base}",
        "--output_root", cfg["output_root"],
        "--ply_path", ply_path,
        "--render_spp", "256",
        "--envmap_init_path", str(envmap_path),
        "--selfocc_offset_max", params["offset"],
        "--geometry_threshold", params["geometry_threshold"],
    ]
    if cfg.get("enable_relight") and cfg.get("envmap_root"):
        cmd.extend(["--relight", "--envmap_root", cfg["envmap_root"]])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_device
    try:
        r = subprocess.run(cmd, env=env, cwd=REPO_ROOT, capture_output=True, text=True, timeout=3600 * 6)
        if r.returncode != 0:
            return False, (r.stderr or r.stdout or "")[-2000:]
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def run_metrics(scene: str, cuda_device: str, cfg: dict) -> tuple[bool, str]:
    base = get_base_scene(scene)
    cmd = [
        sys.executable, "metrics.py",
        *_frozen_constants_cli(cfg),
        "--dataset_type", cfg["dataset_type"],
        "--dataset_name", scene,
        "--dataset_path", f"{cfg['dataset_root']}/{base}",
        "--output_root", cfg["output_root"],
    ]
    if cfg.get("enable_relight") and cfg.get("envmap_root"):
        cmd.extend(["--relight", "--envmap_root", cfg["envmap_root"]])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_device
    try:
        r = subprocess.run(cmd, env=env, cwd=REPO_ROOT, capture_output=True, text=True, timeout=3600 * 2)
        if r.returncode != 0:
            return False, (r.stderr or r.stdout or "")[-2000:]
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def _load_npz_metrics(path: Path) -> Optional[dict]:
    try:
        data = np.load(path, allow_pickle=True)
        out = {}
        for k in data.files:
            v = data[k]
            if isinstance(v, np.ndarray) and v.size == 1:
                out[k] = float(v.flat[0])
            elif isinstance(v, (int, float)):
                out[k] = float(v)
            elif isinstance(v, np.ndarray):
                out[k] = float(np.mean(v))
            else:
                out[k] = str(v)
        return out
    except Exception:
        return None


def load_metrics_for_scene(scene: str, cfg: dict) -> Optional[dict]:
    """Read metrics from scene/renders (results_metrics.json.npz or .npz)."""
    base = Path(cfg["output_root"]) / scene / "renders"
    if not base.exists():
        return None
    main_metrics = None
    for name in ("results_metrics.json.npz", "results_metrics.npz", "results_metrics.json"):
        p = base / name
        if p.exists():
            main_metrics = _load_npz_metrics(p)
            break
    if not main_metrics:
        return None
    relight_psnr, relight_ssim, relight_lpips = [], [], []
    for f in base.iterdir():
        if f.suffix not in (".json", ".npz"):
            continue
        stem = f.stem
        if stem == "results_metrics" or not stem.endswith("_results_metrics"):
            continue
        m = _load_npz_metrics(f)
        if m and "psnr_rgb_mean" in m:
            relight_psnr.append(m["psnr_rgb_mean"])
            relight_ssim.append(m["ssim_rgb_mean"])
            relight_lpips.append(m["lpips_rgb_mean"])
    return {
        "Albedo": {
            "PSNR": main_metrics.get("psnr_albedo_mean"),
            "SSIM": main_metrics.get("ssim_albedo_mean"),
            "LPIPS": main_metrics.get("lpips_albedo_mean"),
        },
        "NVS": {
            "PSNR": main_metrics.get("psnr_rgb_mean"),
            "SSIM": main_metrics.get("ssim_rgb_mean"),
            "LPIPS": main_metrics.get("lpips_rgb_mean"),
        },
        "Relight": {
            "PSNR": float(np.mean(relight_psnr)) if relight_psnr else None,
            "SSIM": float(np.mean(relight_ssim)) if relight_ssim else None,
            "LPIPS": float(np.mean(relight_lpips)) if relight_lpips else None,
        },
        "Roughness": {
            "MSE": main_metrics.get("l2_roughness_mean"),
        },
    }


def run_one_scene(args_tuple: tuple) -> dict:
    scene, cuda_device, iteration, resume, max_retries, cfg = args_tuple
    result = {
        "scene": scene,
        "cuda_device": cuda_device,
        "status": "pending",
        "train_ok": False,
        "render_ok": False,
        "metrics_ok": False,
        "train_retries": 0,
        "render_retries": 0,
        "metrics_retries": 0,
        "train_time_sec": None,
        "last_error": None,
        "metrics": None,
    }
    t_train_start = time.perf_counter()
    for attempt in range(max_retries):
        ok, err = run_train(scene, cuda_device, iteration, resume, cfg)
        result["train_retries"] = attempt + 1
        if ok:
            result["train_ok"] = True
            break
        result["last_error"] = err
        if attempt < max_retries - 1:
            print(f"[{cfg['dataset_type']}] [{scene}] Train attempt {attempt + 1} failed, retrying...", file=sys.stderr)
    result["train_time_sec"] = time.perf_counter() - t_train_start
    if not result["train_ok"]:
        result["status"] = "train_failed"
        return result
    for attempt in range(max_retries):
        ok, err = run_render(scene, cuda_device, iteration, cfg)
        result["render_retries"] = attempt + 1
        if ok:
            result["render_ok"] = True
            break
        result["last_error"] = err
        if attempt < max_retries - 1:
            print(f"[{cfg['dataset_type']}] [{scene}] Render attempt {attempt + 1} failed, retrying...", file=sys.stderr)
    if not result["render_ok"]:
        result["status"] = "render_failed"
        return result
    for attempt in range(max_retries):
        ok, err = run_metrics(scene, cuda_device, cfg)
        result["metrics_retries"] = attempt + 1
        if ok:
            result["metrics_ok"] = True
            break
        result["last_error"] = err
        if attempt < max_retries - 1:
            print(f"[{cfg['dataset_type']}] [{scene}] Metrics attempt {attempt + 1} failed, retrying...", file=sys.stderr)
    if not result["metrics_ok"]:
        result["status"] = "metrics_failed"
        return result
    result["status"] = "ok"
    result["metrics"] = load_metrics_for_scene(scene, cfg)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Run full pipeline (train->render->metrics) for a dataset with retries and optional multi-GPU.")
    ap.add_argument("--dataset", type=str, default=None, help="Dataset name (e.g. TensoIR, Synthetic4Relight, Nerf_Synthetic, RT4Relight). Use --list_datasets to show.")
    ap.add_argument("--list_datasets", action="store_true", help="List configured datasets and exit.")
    ap.add_argument("--cuda_devices", type=str, default="0", help="Comma or space separated, e.g. '2,4' or '2 4'")
    ap.add_argument("--scenes", nargs="*", default=None, help="Scenes to run (default: all for chosen dataset)")
    ap.add_argument("--iteration", type=int, default=None, help="Target iteration (default: dataset default)")
    ap.add_argument("--max_retries", type=int, default=3, help="Max retries per step")
    ap.add_argument("--resume", action="store_true", help="Resume training from checkpoint when available")
    ap.add_argument("--output_json", type=str, default=None, help="Summary JSON path (default: outputs/<dataset>/all_results.json)")
    ap.add_argument("--no_parallel", action="store_true", help="Run scenes sequentially on first device")
    ap.add_argument(
        "--no_freeze_constants",
        action="store_true",
        help="Do not snapshot constants.py at startup. Each train/render/metrics subprocess will re-read constants from disk (editing constants mid-run can affect later scenes).",
    )
    ap.add_argument(
        "--run_dir",
        type=str,
        default=None,
        metavar="NAME_OR_PATH",
        help=(
            "Reuse an existing run folder under outputs/ instead of creating a new YYYY-MM-DD_HH-MM-SS. "
            "Examples: '2026-03-20_19-39-44' (same as outputs/<that>/), or 'outputs/2026-03-20_19-39-44', "
            "or an absolute path. Per-dataset outputs go to <run_dir>/<DatasetName>/; run.log is appended."
        ),
    )
    args = ap.parse_args()

    if args.list_datasets:
        print("Configured datasets:", ", ".join(DATASET_CONFIG.keys()))
        for k, c in DATASET_CONFIG.items():
            print(f"  {k}: scenes={c['default_scenes']}, iteration={c['default_iteration']}")
        return

    if args.dataset and args.dataset not in DATASET_CONFIG:
        print(f"Error: unknown dataset '{args.dataset}'. Use --list_datasets.", file=sys.stderr)
        sys.exit(2)

    # If no dataset is specified, run all configured datasets
    datasets = [args.dataset] if args.dataset else list(DATASET_CONFIG.keys())
    any_failed = False

    # Shared CUDA device parsing for all datasets
    cuda_devices = re.split(r"[\s,]+", args.cuda_devices.strip().strip(","))
    cuda_devices = [d for d in cuda_devices if d] or ["0"]

    # One run root: outputs/<date>/, or reuse an existing directory via --run_dir.
    _ds_out = Path(DATASET_CONFIG[datasets[0]]["output_root"])
    if not _ds_out.is_absolute():
        _ds_out = REPO_ROOT / _ds_out
    base_output_root = _ds_out.resolve().parent
    run_root, run_id = _resolve_run_root(args.run_dir, base_output_root)
    run_root.mkdir(parents=True, exist_ok=True)
    if args.run_dir:
        print(f"[run_all] Using run_dir (no new timestamp): {run_root}")

    frozen_constants_cli: list[str] = []
    if not args.no_freeze_constants:
        try:
            snap = _load_constants_snapshot(REPO_ROOT)
            frozen_constants_cli = _namespace_to_cli_args(snap)
            print(
                f"[run_all] Frozen constants.py snapshot ({len(snap)} argparse fields -> {len(frozen_constants_cli)} CLI tokens). "
                "Later edits to constants.py will not affect this run. "
                "Use --no_freeze_constants to disable."
            )
        except Exception as e:
            print(f"[run_all] Warning: could not freeze constants snapshot: {e}", file=sys.stderr)
            frozen_constants_cli = []

    # Pre-create output dirs and configs for all datasets (for global-queue mode)
    dataset_infos = {}
    for dataset_name in datasets:
        cfg = DATASET_CONFIG[dataset_name].copy()
        cfg["frozen_constants_cli"] = frozen_constants_cli
        cfg["output_root"] = str(run_root / dataset_name)
        output_root_path = Path(cfg["output_root"])
        output_root_path.mkdir(parents=True, exist_ok=True)
        scenes = args.scenes if args.scenes is not None else cfg["default_scenes"]
        iteration = args.iteration if args.iteration is not None else cfg["default_iteration"]
        output_json = str(output_root_path / "all_results.json") if len(datasets) > 1 else (args.output_json or str(output_root_path / "all_results.json"))
        log_path = output_root_path / "run.log"
        cfg["run_log_path"] = str(log_path)
        dataset_infos[dataset_name] = {
            "cfg": cfg,
            "output_root_path": output_root_path,
            "output_json": output_json,
            "log_path": log_path,
            "run_id": run_id,
            "scenes": scenes,
            "iteration": iteration,
        }

    use_global_queue = (
        not args.no_parallel
        and len(cuda_devices) > 1
        and len(datasets) > 1
    )

    if use_global_queue:
        # Single global queue across datasets: when a GPU frees up, assign next task (can be next dataset)
        dataset_log_paths = {name: dataset_infos[name]["log_path"] for name in datasets}

        def log_ds(dataset_name: str, msg: str) -> None:
            line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
            print(msg)
            with open(dataset_log_paths[dataset_name], "a", encoding="utf-8") as f:
                f.write(line)

        global_pending = []
        for dataset_name in datasets:
            info = dataset_infos[dataset_name]
            cfg, scenes, iteration = info["cfg"], info["scenes"], info["iteration"]
            for scene in scenes:
                global_pending.append((dataset_name, scene, cfg, iteration, args.resume, args.max_retries))

        for dataset_name in datasets:
            info = dataset_infos[dataset_name]
            log_ds(dataset_name, f"Run started | dataset: {dataset_name} | output: {info['cfg']['output_root']}")
            log_ds(dataset_name, f"Dataset: {dataset_name} | scenes: {info['scenes']} | CUDA: {cuda_devices} | iter: {info['iteration']}")

        print(f"Global queue: {len(global_pending)} tasks across {datasets} | CUDA: {cuda_devices} | max_retries: {args.max_retries}")

        results_by_dataset = {name: [] for name in datasets}
        free_gpus = list(cuda_devices)
        futures = {}  # future -> (dataset_name, scene, gpu)
        scene_order = {name: {s: i for i, s in enumerate(dataset_infos[name]["scenes"])} for name in datasets}

        with ProcessPoolExecutor(max_workers=len(cuda_devices)) as ex:
            while free_gpus and global_pending:
                dataset_name, scene, cfg, iteration, resume, max_retries = global_pending.pop(0)
                gpu = free_gpus.pop(0)
                full_task = (scene, gpu, iteration, resume, max_retries, cfg)
                fut = ex.submit(run_one_scene, full_task)
                futures[fut] = (dataset_name, scene, gpu)
                log_ds(dataset_name, f"  [{scene}] start (GPU {gpu})")

            while futures:
                done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    if fut not in futures:
                        continue
                    dataset_name, scene, gpu = futures.pop(fut)
                    try:
                        r = fut.result()
                        results_by_dataset[dataset_name].append(r)
                        train_sec = r.get("train_time_sec")
                        tstr = f" {train_sec:.0f}s" if train_sec is not None else ""
                        log_ds(dataset_name, f"  [{scene}] done {r['status']}{tstr}")
                        print(f"Completed: [{dataset_name}] {scene} -> {r['status']}")
                    except Exception as e:
                        results_by_dataset[dataset_name].append({
                            "scene": scene,
                            "cuda_device": gpu,
                            "status": "exception",
                            "last_error": str(e),
                            "train_ok": False,
                            "render_ok": False,
                            "metrics_ok": False,
                            "train_time_sec": None,
                        })
                        log_ds(dataset_name, f"  [{scene}] exception: {e}")
                        print(f"Exception: [{dataset_name}] {scene} -> {e}", file=sys.stderr)
                    free_gpus.append(gpu)
                    if global_pending:
                        dataset_name, scene, cfg, iteration, resume, max_retries = global_pending.pop(0)
                        full_task = (scene, gpu, iteration, resume, max_retries, cfg)
                        new_fut = ex.submit(run_one_scene, full_task)
                        futures[new_fut] = (dataset_name, scene, gpu)
                        log_ds(dataset_name, f"  [{scene}] start (GPU {gpu})")

        for dataset_name in datasets:
            results = results_by_dataset[dataset_name]
            order = scene_order[dataset_name]
            results.sort(key=lambda r: order.get(r["scene"], 999))
            info = dataset_infos[dataset_name]
            summary = {
                "config": {
                    "dataset": dataset_name,
                    "run_id": info["run_id"],
                    "output_root": info["cfg"]["output_root"],
                    "cuda_devices": cuda_devices,
                    "scenes": info["scenes"],
                    "iteration": info["iteration"],
                    "max_retries": args.max_retries,
                    "resume": args.resume,
                },
                "results": results,
                "metrics_summary": {},
                "train_times_sec": {r["scene"]: r.get("train_time_sec") for r in results if r.get("train_time_sec") is not None},
            }
            for r in results:
                m = r.get("metrics")
                if not m and r.get("status") == "ok":
                    m = load_metrics_for_scene(r["scene"], info["cfg"])
                if m:
                    summary["metrics_summary"][r["scene"]] = m
            with open(info["output_json"], "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            total_train_sec = sum(r.get("train_time_sec") or 0 for r in results)
            log_ds(dataset_name, "--- Training time ---")
            for r in results:
                s = r.get("train_time_sec")
                log_ds(dataset_name, f"  {r['scene']}: " + (f"{s:.0f}s" if s is not None else "-"))
            log_ds(dataset_name, f"  Total: {total_train_sec:.0f}s ({total_train_sec/60:.1f} min)")
            log_ds(dataset_name, f"Summary: {info['output_json']}")
            failed = [r["scene"] for r in results if r["status"] != "ok"]
            if failed:
                any_failed = True
                log_ds(dataset_name, f"Failed: {failed}")
                print(f"[{dataset_name}] Failed scenes: {failed}", file=sys.stderr)
            else:
                log_ds(dataset_name, "Done.")
                print(f"All scenes for dataset {dataset_name} completed successfully.")
            # Log test metrics table (from scene/renders/results_metrics.json.npz)
            if summary.get("metrics_summary"):
                log_ds(dataset_name, "--- Test metrics (from renders) ---")
                for sc, m in summary["metrics_summary"].items():
                    nvs = m.get("NVS") or {}
                    alb = m.get("Albedo") or {}
                    rough = m.get("Roughness") or {}
                    p, s = nvs.get("PSNR"), nvs.get("SSIM")
                    ap, rm = alb.get("PSNR"), rough.get("MSE")
                    log_ds(dataset_name, f"  [{sc}] NVS PSNR={(p or 0):.2f} SSIM={(s or 0):.3f} | Albedo PSNR={(ap or 0):.2f} | Roughness MSE={(rm or 0):.4f}")

    else:
        # Per-dataset loop (original behavior: one dataset at a time, or single-GPU / no_parallel)
        for dataset_name in datasets:
            info = dataset_infos[dataset_name]
            cfg = info["cfg"]
            scenes = info["scenes"]
            iteration = info["iteration"]
            output_json = info["output_json"]
            log_path = info["log_path"]

            def log(msg: str) -> None:
                line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
                print(msg)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(line)

            log(f"Run started | dataset: {dataset_name} | output: {cfg['output_root']}")
            log(f"Dataset: {dataset_name} | scenes: {scenes} | CUDA: {cuda_devices} | iter: {iteration}")

            tasks = []
            for i, scene in enumerate(scenes):
                dev = cuda_devices[i % len(cuda_devices)]
                tasks.append((scene, dev, iteration, args.resume, args.max_retries, cfg))

            print(f"Dataset: {dataset_name}, scenes: {scenes}, CUDA: {cuda_devices}, max_retries: {args.max_retries}")

            if args.no_parallel or len(cuda_devices) == 1:
                results = []
                for t in tasks:
                    scene_name = t[0]
                    log(f"  [{scene_name}] start (GPU {t[1]})")
                    r = run_one_scene(t)
                    results.append(r)
                    train_sec = r.get("train_time_sec")
                    tstr = f" {train_sec:.0f}s" if train_sec is not None else ""
                    log(f"  [{scene_name}] done {r['status']}{tstr}")
                    print(f"Completed: {scene_name} -> {r['status']}")
            else:
                results = []
                pending = [(s, iteration, args.resume, args.max_retries, cfg) for s in scenes]
                free_gpus = list(cuda_devices)
                futures = {}
                with ProcessPoolExecutor(max_workers=len(cuda_devices)) as ex:
                    while free_gpus and pending:
                        scene, it, resume, max_retries, c = pending.pop(0)
                        gpu = free_gpus.pop(0)
                        full_task = (scene, gpu, it, resume, max_retries, c)
                        fut = ex.submit(run_one_scene, full_task)
                        futures[fut] = (scene, gpu)
                        log(f"  [{scene}] start (GPU {gpu})")
                    while futures:
                        done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                        for fut in done:
                            if fut not in futures:
                                continue
                            scene, gpu = futures.pop(fut)
                            try:
                                r = fut.result()
                                results.append(r)
                                train_sec = r.get("train_time_sec")
                                tstr = f" {train_sec:.0f}s" if train_sec is not None else ""
                                log(f"  [{scene}] done {r['status']}{tstr}")
                                print(f"Completed: {scene} -> {r['status']}")
                            except Exception as e:
                                results.append({
                                    "scene": scene,
                                    "cuda_device": gpu,
                                    "status": "exception",
                                    "last_error": str(e),
                                    "train_ok": False,
                                    "render_ok": False,
                                    "metrics_ok": False,
                                    "train_time_sec": None,
                                })
                                log(f"  [{scene}] exception: {e}")
                            free_gpus.append(gpu)
                            if pending:
                                scene, it, resume, max_retries, c = pending.pop(0)
                                full_task = (scene, gpu, it, resume, max_retries, c)
                                new_fut = ex.submit(run_one_scene, full_task)
                                futures[new_fut] = (scene, gpu)
                                log(f"  [{scene}] start (GPU {gpu})")
                order = {s: i for i, s in enumerate(scenes)}
                results.sort(key=lambda r: order.get(r["scene"], 999))

            summary = {
                "config": {
                    "dataset": dataset_name,
                    "run_id": info["run_id"],
                    "output_root": cfg["output_root"],
                    "cuda_devices": cuda_devices,
                    "scenes": scenes,
                    "iteration": iteration,
                    "max_retries": args.max_retries,
                    "resume": args.resume,
                },
                "results": results,
                "metrics_summary": {},
                "train_times_sec": {r["scene"]: r.get("train_time_sec") for r in results if r.get("train_time_sec") is not None},
            }
            for r in results:
                m = r.get("metrics")
                if not m and r.get("status") == "ok":
                    m = load_metrics_for_scene(r["scene"], cfg)
                if m:
                    summary["metrics_summary"][r["scene"]] = m
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            total_train_sec = sum(r.get("train_time_sec") or 0 for r in results)
            log("--- Training time ---")
            for r in results:
                s = r.get("train_time_sec")
                log(f"  {r['scene']}: " + (f"{s:.0f}s" if s is not None else "-"))
            log(f"  Total: {total_train_sec:.0f}s ({total_train_sec/60:.1f} min)")
            log(f"Summary: {output_json}")
            failed = [r["scene"] for r in results if r["status"] != "ok"]
            if failed:
                any_failed = True
                log(f"Failed: {failed}")
                print(f"[{dataset_name}] Failed scenes: {failed}", file=sys.stderr)
            else:
                log("Done.")
                print(f"All scenes for dataset {dataset_name} completed successfully.")
            if summary.get("metrics_summary"):
                log("--- Test metrics (from renders) ---")
                for sc, m in summary["metrics_summary"].items():
                    nvs = m.get("NVS") or {}
                    alb = m.get("Albedo") or {}
                    rough = m.get("Roughness") or {}
                    p, s = nvs.get("PSNR"), nvs.get("SSIM")
                    ap, rm = alb.get("PSNR"), rough.get("MSE")
                    log(f"  [{sc}] NVS PSNR={(p or 0):.2f} SSIM={(s or 0):.3f} | Albedo PSNR={(ap or 0):.2f} | Roughness MSE={(rm or 0):.4f}")

    if any_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
