"""Low-overhead host/GPU thermal monitoring for long training jobs."""

from __future__ import annotations

import threading
import time
from typing import Any


class ThermalMonitor:
    """Background GPU/CPU temperature monitor with hysteresis.

    GPU readings use NVML. CPU readings are optional and use psutil when
    available. The monitor never touches model tensors; it only sets an event
    that the training loop checks after an optimizer step.
    """

    def __init__(self, cfg: dict[str, Any]):
        self.gpu_enabled = bool(cfg.get("gpu_temp_monitor_enabled", True))
        self.cpu_enabled = bool(cfg.get("cpu_temp_monitor_enabled", True))
        self.poll_seconds = max(1.0, float(cfg.get("thermal_poll_seconds", 5.0)))
        self.gpu_limit = float(cfg.get("gpu_temp_stop_celsius", 85.0))
        self.gpu_hold = max(1.0, float(cfg.get("gpu_temp_hold_seconds", 120.0)))
        self.gpu_recover = float(cfg.get("gpu_temp_recover_celsius", 82.0))
        self.cpu_limit = float(cfg.get("cpu_temp_stop_celsius", 95.0))
        self.cpu_hold = max(1.0, float(cfg.get("cpu_temp_hold_seconds", 180.0)))
        self.cpu_recover = float(cfg.get("cpu_temp_recover_celsius", 90.0))
        self.stop_event = threading.Event()
        self.shutdown_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.reason = ""
        self._gpu_hot_since: float | None = None
        self._cpu_hot_since: float | None = None
        self._nvml = None
        self._nvml_handles = []

    @property
    def stop_requested(self) -> bool:
        return self.stop_event.is_set()

    def start(self) -> None:
        if not self.gpu_enabled and not self.cpu_enabled:
            print("thermal monitor disabled", flush=True)
            return
        if self.gpu_enabled:
            try:
                import pynvml

                pynvml.nvmlInit()
                self._nvml = pynvml
                self._nvml_handles = [
                    pynvml.nvmlDeviceGetHandleByIndex(index)
                    for index in range(pynvml.nvmlDeviceGetCount())
                ]
                print(
                    f"thermal monitor: GPU NVML active devices={len(self._nvml_handles)} "
                    f"stop={self.gpu_limit:g}C/{self.gpu_hold:g}s",
                    flush=True,
                )
            except Exception as error:
                self._nvml = None
                self._nvml_handles = []
                print(
                    f"thermal monitor: GPU NVML unavailable ({type(error).__name__}: {error}); "
                    "GPU temperature stop disabled",
                    flush=True,
                )
        self.thread = threading.Thread(
            target=self._run,
            name="thermal-monitor",
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        self.shutdown_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(2.0, self.poll_seconds + 1.0))
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass

    def _gpu_temperature(self) -> list[float]:
        if self._nvml is None:
            return []
        values = []
        for handle in self._nvml_handles:
            try:
                values.append(float(self._nvml.nvmlDeviceGetTemperature(
                    handle, self._nvml.NVML_TEMPERATURE_GPU
                )))
            except Exception:
                continue
        return values

    def _cpu_temperature(self) -> list[float]:
        if not self.cpu_enabled:
            return []
        try:
            import psutil

            readings = psutil.sensors_temperatures()
            values = []
            for key, entries in readings.items():
                key_text = str(key).lower()
                for entry in entries:
                    label = str(getattr(entry, "label", "")).lower()
                    if any(
                        marker in key_text or marker in label
                        for marker in (
                            "cpu", "core", "package", "coretemp",
                            "k10temp", "tctl", "tdie",
                        )
                    ) and entry.current is not None:
                        values.append(float(entry.current))
            return values
        except Exception:
            return []

    def _check_domain(
        self,
        label: str,
        values: list[float],
        limit: float,
        recover: float,
        hold: float,
        now: float,
    ) -> None:
        if not values:
            return
        hottest = max(values)
        if hottest >= limit:
            since_name = f"_{label}_hot_since"
            since = getattr(self, since_name)
            if since is None:
                setattr(self, since_name, now)
                print(
                    f"thermal warning: {label} temperature={hottest:.1f}C "
                    f">={limit:g}C; holding for {hold:g}s",
                    flush=True,
                )
            elif now - since >= hold and not self.stop_requested:
                self.reason = (
                    f"{label} temperature {hottest:.1f}C stayed at or above "
                    f"{limit:g}C for {now - since:.0f}s"
                )
                self.stop_event.set()
                print(f"THERMAL STOP REQUESTED: {self.reason}", flush=True)
        elif hottest <= recover:
            setattr(self, f"_{label}_hot_since", None)

    def _run(self) -> None:
        while not self.shutdown_event.wait(self.poll_seconds):
            now = time.monotonic()
            self._check_domain(
                "gpu", self._gpu_temperature(), self.gpu_limit,
                self.gpu_recover, self.gpu_hold, now,
            )
            self._check_domain(
                "cpu", self._cpu_temperature(), self.cpu_limit,
                self.cpu_recover, self.cpu_hold, now,
            )
