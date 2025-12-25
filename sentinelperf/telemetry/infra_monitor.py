"""Infrastructure saturation detection for SentinelPerf

Read-only detection of container/host resource saturation.
Does NOT block tests - only adds warnings to reports.

No external dependencies - uses /proc filesystem on Linux.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime


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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_available_mb": self.memory_available_mb,
            "saturated": self.is_saturated,
        }


@dataclass
class InfraSnapshot:
    """Snapshot of infrastructure metrics at a specific point"""
    phase: str  # "pre_baseline", "peak_stress", "peak_spike", "end_of_test"
    vus: int = 0
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_available_mb: float = 0.0
    saturated: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "phase": self.phase,
            "vus": self.vus,
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_available_mb": self.memory_available_mb,
            "saturated": self.saturated,
            "timestamp": self.timestamp.isoformat(),
            "notes": self.notes,
        }


@dataclass
class InfraTimeline:
    """Timeline of infrastructure metrics across load phases"""
    snapshots: List[InfraSnapshot] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    data_available: bool = False
    
    @property
    def confidence_penalty(self) -> float:
        """Max confidence penalty from any saturated snapshot"""
        if not self.snapshots:
            return 0.0
        max_penalty = 0.0
        for snap in self.snapshots:
            if snap.saturated:
                penalty = 0.0
                if snap.cpu_percent >= InfraMonitor.CPU_SATURATION_THRESHOLD:
                    penalty += 0.15
                if snap.memory_percent >= InfraMonitor.MEMORY_SATURATION_THRESHOLD:
                    penalty += 0.15
                max_penalty = max(max_penalty, min(penalty, 0.3))
        return max_penalty
    
    def add_snapshot(self, phase: str, vus: int, result: InfraSaturationResult, notes: str = "") -> None:
        """Add a snapshot to the timeline"""
        self.data_available = True
        
        # Auto-generate notes if not provided
        if not notes:
            if result.is_saturated:
                notes = "Resource saturation detected"
            elif result.cpu_percent > 70:
                notes = "Elevated CPU usage"
            elif result.memory_percent > 80:
                notes = "Elevated memory usage"
            else:
                notes = "Normal"
        
        snapshot = InfraSnapshot(
            phase=phase,
            vus=vus,
            cpu_percent=result.cpu_percent,
            memory_percent=result.memory_percent,
            memory_available_mb=result.memory_available_mb,
            saturated=result.is_saturated,
            notes=notes,
        )
        self.snapshots.append(snapshot)
        
        # Collect warnings
        for w in result.warnings:
            if w not in self.warnings:
                self.warnings.append(w)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for state storage"""
        return {
            "data_available": self.data_available,
            "snapshots": [s.to_dict() for s in self.snapshots],
            "warnings": self.warnings,
            "confidence_penalty": self.confidence_penalty,
            # Legacy fields for backward compatibility
            "pre_test": self.snapshots[0].to_dict() if self.snapshots else {},
            "post_test": self.snapshots[-1].to_dict() if self.snapshots else {},
        }


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
        self._timeline: InfraTimeline = InfraTimeline()
    
    def reset_timeline(self) -> None:
        """Reset the timeline for a new test run"""
        self._timeline = InfraTimeline()
    
    def get_timeline(self) -> InfraTimeline:
        """Get the current timeline"""
        return self._timeline
    
    def capture_snapshot(self, phase: str, vus: int = 0, notes: str = "") -> InfraSaturationResult:
        """
        Capture a snapshot at the current point and add to timeline.
        
        Args:
            phase: Load phase name (pre_baseline, peak_stress, peak_spike, end_of_test)
            vus: Number of virtual users at this point
            notes: Optional notes about this snapshot
        
        Returns:
            InfraSaturationResult with current metrics
        """
        result = self.check_saturation()
        self._timeline.add_snapshot(phase, vus, result, notes)
        return result
    
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


def capture_infra_snapshot(phase: str, vus: int = 0, notes: str = "") -> InfraSaturationResult:
    """Convenience function to capture an infrastructure snapshot"""
    return get_infra_monitor().capture_snapshot(phase, vus, notes)


def get_infra_timeline() -> InfraTimeline:
    """Convenience function to get the infrastructure timeline"""
    return get_infra_monitor().get_timeline()


def reset_infra_timeline() -> None:
    """Convenience function to reset the infrastructure timeline"""
    get_infra_monitor().reset_timeline()
