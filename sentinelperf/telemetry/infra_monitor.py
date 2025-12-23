"""Infrastructure saturation detection for SentinelPerf

Read-only detection of container/host resource saturation.
Does NOT block tests - only adds warnings to reports.

No external dependencies - uses /proc filesystem on Linux.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class InfraSaturationResult:
    """Results from infrastructure saturation check"""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_available_mb: float = 0.0
    cpu_saturated: bool = False
    memory_saturated: bool = False
    warnings: List[str] = field(default_factory=list)
    
    @property
    def is_saturated(self) -> bool:
        return self.cpu_saturated or self.memory_saturated
    
    @property
    def confidence_penalty(self) -> float:
        """Penalty to apply to confidence scores (0.0 to 0.3)"""
        penalty = 0.0
        if self.cpu_saturated:
            penalty += 0.15
        if self.memory_saturated:
            penalty += 0.15
        return min(penalty, 0.3)


class InfraMonitor:
    """
    Lightweight infrastructure monitor using /proc filesystem.
    
    Read-only, no external dependencies.
    """
    
    # Thresholds for saturation detection
    CPU_SATURATION_THRESHOLD = 85.0  # percent
    MEMORY_SATURATION_THRESHOLD = 90.0  # percent
    MEMORY_LOW_MB_THRESHOLD = 512  # MB
    
    def __init__(self):
        self._prev_cpu_stats: Optional[Tuple[int, int]] = None
    
    def check_saturation(self) -> InfraSaturationResult:
        """
        Check for infrastructure saturation.
        
        Returns InfraSaturationResult with current metrics and warnings.
        """
        result = InfraSaturationResult()
        
        # Check CPU
        try:
            cpu_percent = self._get_cpu_percent()
            result.cpu_percent = cpu_percent
            if cpu_percent >= self.CPU_SATURATION_THRESHOLD:
                result.cpu_saturated = True
                result.warnings.append(
                    f"High CPU usage detected ({cpu_percent:.1f}%) - "
                    "test results may be infra-limited"
                )
        except Exception:
            pass  # Silently ignore - not critical
        
        # Check memory
        try:
            mem_percent, mem_available_mb = self._get_memory_stats()
            result.memory_percent = mem_percent
            result.memory_available_mb = mem_available_mb
            
            if mem_percent >= self.MEMORY_SATURATION_THRESHOLD:
                result.memory_saturated = True
                result.warnings.append(
                    f"High memory usage detected ({mem_percent:.1f}%) - "
                    "test results may be infra-limited"
                )
            elif mem_available_mb < self.MEMORY_LOW_MB_THRESHOLD:
                result.memory_saturated = True
                result.warnings.append(
                    f"Low available memory ({mem_available_mb:.0f}MB) - "
                    "test results may be infra-limited"
                )
        except Exception:
            pass  # Silently ignore - not critical
        
        return result
    
    def _get_cpu_percent(self) -> float:
        """Get CPU usage percentage from /proc/stat"""
        try:
            with open('/proc/stat', 'r') as f:
                line = f.readline()
            
            # Parse: cpu user nice system idle iowait irq softirq steal guest guest_nice
            parts = line.split()
            if parts[0] != 'cpu':
                return 0.0
            
            values = [int(x) for x in parts[1:]]
            idle = values[3] + values[4]  # idle + iowait
            total = sum(values)
            
            if self._prev_cpu_stats is None:
                self._prev_cpu_stats = (idle, total)
                return 0.0
            
            prev_idle, prev_total = self._prev_cpu_stats
            self._prev_cpu_stats = (idle, total)
            
            idle_delta = idle - prev_idle
            total_delta = total - prev_total
            
            if total_delta == 0:
                return 0.0
            
            return 100.0 * (1.0 - idle_delta / total_delta)
            
        except (FileNotFoundError, PermissionError, ValueError):
            return 0.0
    
    def _get_memory_stats(self) -> Tuple[float, float]:
        """Get memory usage from /proc/meminfo"""
        try:
            meminfo = {}
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    parts = line.split(':')
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip().split()[0]  # Remove 'kB'
                        meminfo[key] = int(value)
            
            total_kb = meminfo.get('MemTotal', 0)
            available_kb = meminfo.get('MemAvailable', meminfo.get('MemFree', 0))
            
            if total_kb == 0:
                return 0.0, 0.0
            
            used_percent = 100.0 * (1.0 - available_kb / total_kb)
            available_mb = available_kb / 1024.0
            
            return used_percent, available_mb
            
        except (FileNotFoundError, PermissionError, ValueError, KeyError):
            return 0.0, 0.0


# Singleton instance
_monitor: Optional[InfraMonitor] = None


def get_infra_monitor() -> InfraMonitor:
    """Get singleton InfraMonitor instance"""
    global _monitor
    if _monitor is None:
        _monitor = InfraMonitor()
    return _monitor


def check_infra_saturation() -> InfraSaturationResult:
    """Convenience function to check infrastructure saturation"""
    return get_infra_monitor().check_saturation()
