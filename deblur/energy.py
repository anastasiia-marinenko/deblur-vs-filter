"""Time and energy.

Time: a stopwatch (time.perf_counter) around one call.

Energy: CodeCarbon (https://codecarbon.io).
  - NVIDIA GPU is measured directly (NVML)
  - CPU is measured only with Intel RAPL; otherwise CodeCarbon estimates it from TDP and load,
    so the energy is nearly proportional to time. The mode is saved as cpu_tracking.
    On Kaggle it is "cpu_load", i.e. an estimate.
  - tracking_mode="machine": energy of the whole machine (CPU + GPU + RAM), gross_J_per_img
We also measure the idle machine and subtract it (net_J_per_img, saved in efficiency.csv).
"""
from __future__ import annotations

import time
import warnings

import numpy as np


def _tracker(project: str):
    # polls power every second, writes no files
    from codecarbon import OfflineEmissionsTracker
    return OfflineEmissionsTracker(project_name=project, country_iso_code="UKR",
                                   measure_power_secs=1, save_to_file=False,
                                   log_level="error", tracking_mode="machine")


def _energy_joules(tracker) -> tuple[float, dict]:
    # CodeCarbon gives kWh; 1 kWh = 3.6e6 J
    d = tracker.final_emissions_data
    info = {"cpu_model": getattr(d, "cpu_model", ""), "gpu_model": getattr(d, "gpu_model", ""),
            "cpu_J": float(getattr(d, "cpu_energy", 0.0)) * 3.6e6,
            "gpu_J": float(getattr(d, "gpu_energy", 0.0)) * 3.6e6,
            "ram_J": float(getattr(d, "ram_energy", 0.0)) * 3.6e6}
    # how CPU power was obtained: "intel_rapl" = measured, "cpu_load"/"constant" = estimated
    info["cpu_tracking"] = ""
    for hw in getattr(tracker, "_hardware", []):
        if type(hw).__name__ == "CPU":
            info["cpu_tracking"] = str(getattr(hw, "_mode", ""))
    return float(d.energy_consumed) * 3.6e6, info


def measure_idle_power(seconds: float = 20.0) -> float:
    """Mean idle power of the machine, W (to subtract later)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tr = _tracker("idle")
        tr.start()
        t0 = time.perf_counter()
        time.sleep(seconds)
        tr.stop()
        dt = time.perf_counter() - t0
    e, _ = _energy_joules(tr)
    return e / dt


def time_per_item(fn, items, warmup: int = 2, sync=None) -> np.ndarray:
    # seconds per image; the first 2 calls are warm-up (library loading, GPU init) and not counted
    for it in items[:warmup]:
        fn(it)
    out = []
    for it in items:
        t0 = time.perf_counter()
        fn(it)
        if sync:
            sync()        # GPU: wait until it has really finished
        out.append(time.perf_counter() - t0)
    return np.array(out)


def energy_per_item(fn, items, min_seconds: float = 30.0, idle_power_W: float = 0.0,
                    project: str = "method") -> dict:
    """Runs fn over items in a loop for at least min_seconds and returns J per image.
    One 10 ms call is too short for a meter that polls once a second, hence the loop."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tr = _tracker(project)
        tr.start()
        n, t0 = 0, time.perf_counter()
        while n < len(items) or time.perf_counter() - t0 < min_seconds:
            fn(items[n % len(items)])
            n += 1
        tr.stop()
        dt = time.perf_counter() - t0
    e, info = _energy_joules(tr)
    return {"n_runs": n, "seconds": dt, "gross_J_per_img": e / n,
            "net_J_per_img": max(0.0, (e - idle_power_W * dt) / n),       # minus idle
            "avg_power_W": e / dt, **info}