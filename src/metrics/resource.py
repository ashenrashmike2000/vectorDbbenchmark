"""
Resource metrics for memory, disk, and CPU usage evaluation.
"""

import os
import time
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import psutil

from src.core.types import ResourceMetrics


def measure_memory_usage() -> Dict[str, int]:
    """
    Measure current memory usage.

    Returns:
        Dictionary with memory metrics in bytes
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    return {
        "rss": mem_info.rss,  # Resident Set Size
        "vms": mem_info.vms,  # Virtual Memory Size
        "shared": getattr(mem_info, 'shared', 0),
        "peak": getattr(mem_info, 'peak_wset', mem_info.rss),  # Peak memory (Windows) or RSS
    }


def measure_disk_usage(path: str) -> Dict[str, int]:
    """
    Measure disk usage for a path.

    Args:
        path: Directory or file path

    Returns:
        Dictionary with disk metrics in bytes
    """
    path = Path(path)

    if not path.exists():
        return {"size": 0, "files": 0}

    if path.is_file():
        return {"size": path.stat().st_size, "files": 1}

    total_size = 0
    file_count = 0

    for item in path.rglob("*"):
        if item.is_file():
            total_size += item.stat().st_size
            file_count += 1

    return {"size": total_size, "files": file_count}


class ResourceMonitor:
    """
    Context manager for monitoring resource usage during operations.
    """

    def __init__(self, sample_interval_sec: float = 0.5):
        self.sample_interval = sample_interval_sec
        self.process = psutil.Process(os.getpid())
        self._monitoring = False
        self._thread: Optional[threading.Thread] = None

        self.cpu_percents: List[float] = []
        self.peak_memory_bytes = 0
        self._start_time = 0
        self._end_time = 0

    def _monitor(self):
        """Background thread function to sample resources."""
        self.peak_memory_bytes = self.process.memory_info().rss
        # Set interval to 0 for the first call to get an initial reading
        psutil.cpu_percent(interval=None)

        while self._monitoring:
            # CPU Usage
            self.cpu_percents.append(psutil.cpu_percent(interval=self.sample_interval))

            # Memory Usage
            current_mem = self.process.memory_info().rss
            if current_mem > self.peak_memory_bytes:
                self.peak_memory_bytes = current_mem

    def __enter__(self):
        self._start_time = time.perf_counter()
        self._monitoring = True
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._monitoring = False
        if self._thread:
            self._thread.join(timeout=self.sample_interval * 2)
        self._end_time = time.perf_counter()
        return False

    @property
    def elapsed_sec(self) -> float:
        """Elapsed time in seconds."""
        return self._end_time - self._start_time

    @property
    def avg_cpu_percent(self) -> float:
        """Average CPU utilization during the monitoring period."""
        return np.mean(self.cpu_percents) if self.cpu_percents else 0.0


def compute_all_resource_metrics(
    build_time_sec: float = 0.0,
    index_size_bytes: int = 0,
    disk_bytes: int = 0,
    ram_bytes_peak: int = 0,
    ram_bytes_steady: int = 0,
    num_vectors: int = 1,
    cpu_utilization: float = 0.0,
) -> ResourceMetrics:
    """
    Compute all resource metrics.
    """
    return ResourceMetrics(
        index_build_time_sec=build_time_sec,
        index_size_bytes=index_size_bytes,
        disk_bytes=disk_bytes,
        ram_bytes_peak=ram_bytes_peak,
        ram_bytes_steady=ram_bytes_steady,
        bytes_per_vector=index_size_bytes / max(num_vectors, 1) if index_size_bytes else 0,
        cpu_utilization_percent=cpu_utilization,
    )
