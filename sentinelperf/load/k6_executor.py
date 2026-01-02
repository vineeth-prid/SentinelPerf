"""k6 execution wrapper for SentinelPerf"""

import json
import os
import subprocess
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime

from sentinelperf.load.generator import TestScript, TestType


@dataclass
class K6Metrics:
    """Parsed k6 metrics"""
    # Request counts
    total_requests: int = 0
    failed_requests: int = 0
    
    # Rates
    requests_per_second: float = 0.0
    error_rate: float = 0.0
    
    # Latency (ms)
    latency_avg: float = 0.0
    latency_min: float = 0.0
    latency_max: float = 0.0
    latency_p50: float = 0.0
    latency_p90: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    
    # VUs
    vus_max: int = 0
    iterations: int = 0


@dataclass
class K6Result:
    """Results from k6 test execution"""
    success: bool
    test_name: str
    test_type: str
    
    # Timing
    started_at: datetime
    completed_at: datetime
    duration_seconds: float = 0.0
    
    # Metrics
    metrics: K6Metrics = field(default_factory=K6Metrics)
    
    # Threshold results
    thresholds_passed: bool = True
    threshold_results: Dict[str, bool] = field(default_factory=dict)
    
    # Raw data
    raw_json: Dict[str, Any] = field(default_factory=dict)
    raw_stdout: str = ""
    raw_stderr: str = ""
    exit_code: int = 0
    
    @classmethod
    def from_k6_json(cls, json_data: Dict[str, Any], test_name: str, test_type: str) -> "K6Result":
        """Parse k6 JSON summary output into K6Result"""
        
        metrics_data = json_data.get("metrics", {})
        
        # Parse HTTP request metrics
        http_reqs = metrics_data.get("http_reqs", {}).get("values", {})
        http_req_failed = metrics_data.get("http_req_failed", {}).get("values", {})
        http_req_duration = metrics_data.get("http_req_duration", {}).get("values", {})
        vus_data = metrics_data.get("vus_max", {}).get("values", {})
        iterations_data = metrics_data.get("iterations", {}).get("values", {})
        
        # Calculate metrics
        total_requests = int(http_reqs.get("count", 0))
        error_rate = http_req_failed.get("rate", 0.0)
        failed_requests = int(total_requests * error_rate)
        
        metrics = K6Metrics(
            total_requests=total_requests,
            failed_requests=failed_requests,
            requests_per_second=http_reqs.get("rate", 0.0),
            error_rate=error_rate,
            latency_avg=http_req_duration.get("avg", 0.0),
            latency_min=http_req_duration.get("min", 0.0),
            latency_max=http_req_duration.get("max", 0.0),
            latency_p50=http_req_duration.get("med", http_req_duration.get("p(50)", 0.0)),
            latency_p90=http_req_duration.get("p(90)", 0.0),
            latency_p95=http_req_duration.get("p(95)", 0.0),
            latency_p99=http_req_duration.get("p(99)", 0.0),
            vus_max=int(vus_data.get("max", vus_data.get("value", 0))),
            iterations=int(iterations_data.get("count", 0)),
        )
        
        # Parse threshold results
        thresholds = json_data.get("thresholds", {})
        threshold_results = {}
        thresholds_passed = True
        
        for name, result in thresholds.items():
            if isinstance(result, dict):
                passed = result.get("ok", True)
            else:
                passed = bool(result)
            threshold_results[name] = passed
            if not passed:
                thresholds_passed = False
        
        return cls(
            success=thresholds_passed,
            test_name=test_name,
            test_type=test_type,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            metrics=metrics,
            thresholds_passed=thresholds_passed,
            threshold_results=threshold_results,
            raw_json=json_data,
        )


@dataclass
class InfraMetricPoint:
    """Infrastructure metrics at a specific VU level"""
    vus: int
    cpu_percent: float
    memory_percent: float
    rps: float = 0.0
    latency_p95_ms: float = 0.0
    error_rate: float = 0.0
    saturated: bool = False
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "vus": self.vus,
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "rps": self.rps,
            "latency_p95_ms": self.latency_p95_ms,
            "error_rate": self.error_rate,
            "saturated": self.saturated,
            "timestamp": self.timestamp,
        }


@dataclass
class AutoScaleResult:
    """Results from auto-scaling stress test execution"""
    results: List[K6Result]
    max_vus_attempted: int
    max_vus_reached: int
    stop_reason: str
    executed_stages: List[int]
    # Infra timeline - metrics at each VU increment
    infra_timeline: List[InfraMetricPoint] = field(default_factory=list)
    # Breaking point VUs (if detected near infra saturation)
    breaking_point_vus: int = 0
    # Whether infra saturation occurred near breaking point
    infra_saturated_at_break: bool = False


class K6Executor:
    """
    Executes k6 load tests.
    
    Supports:
    - Local k6 binary
    - Docker-based execution (fallback)
    """
    
    def __init__(
        self,
        k6_path: Optional[str] = None,
        output_dir: Path = None,
        use_docker: bool = False,
    ):
        self.k6_path = k6_path
        self.output_dir = output_dir or Path("./k6-output")
        self.use_docker = use_docker
        self._k6_available = None
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _find_k6(self) -> Optional[str]:
        """Find k6 binary in system PATH"""
        if self.k6_path:
            if Path(self.k6_path).exists():
                return self.k6_path
        
        # Check system PATH
        k6_path = shutil.which("k6")
        if k6_path:
            return k6_path
        
        # Check common locations
        common_paths = [
            "/usr/local/bin/k6",
            "/usr/bin/k6",
            str(Path.home() / ".local" / "bin" / "k6"),
            str(Path.cwd() / "bin" / "k6"),
        ]
        
        for path in common_paths:
            if Path(path).exists():
                return path
        
        return None
    
    def check_available(self) -> bool:
        """Check if k6 is available"""
        if self._k6_available is not None:
            return self._k6_available
        
        k6_path = self._find_k6()
        if k6_path:
            try:
                result = subprocess.run(
                    [k6_path, "version"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self._k6_available = result.returncode == 0
                if self._k6_available:
                    self.k6_path = k6_path
                return self._k6_available
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        
        # Try Docker
        if self.use_docker:
            try:
                result = subprocess.run(
                    ["docker", "run", "--rm", "grafana/k6", "version"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self._k6_available = result.returncode == 0
                return self._k6_available
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        
        self._k6_available = False
        return False
    
    def get_version(self) -> Optional[str]:
        """Get k6 version string"""
        if not self.check_available():
            return None
        
        if self.k6_path:
            result = subprocess.run(
                [self.k6_path, "version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        
        return None
    
    def execute(
        self,
        script: TestScript,
        timeout_seconds: int = 300,
        verbose: bool = False,
    ) -> K6Result:
        """
        Execute a k6 test script.
        
        Args:
            script: TestScript to execute
            timeout_seconds: Maximum execution time
            verbose: Print k6 output
            
        Returns:
            K6Result with test metrics
        """
        if not self.check_available():
            return self._create_error_result(
                script,
                "k6 not available. Install k6 or enable Docker mode.",
            )
        
        # Generate script content
        script_content = script.to_k6_script()
        
        # Write to temp file
        script_file = self.output_dir / f"{script.name}.js"
        with open(script_file, "w") as f:
            f.write(script_content)
        
        started_at = datetime.utcnow()
        
        # Build k6 command
        if self.k6_path:
            cmd = [self.k6_path, "run", "--quiet", str(script_file)]
        else:
            # Docker mode
            cmd = [
                "docker", "run", "--rm",
                "-v", f"{script_file}:/script.js",
                "--network", "host",
                "grafana/k6", "run", "--quiet", "/script.js"
            ]
        
        try:
            if verbose:
                print(f"  Executing: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env={**os.environ, "K6_NO_USAGE_REPORT": "true"},
            )
            
            completed_at = datetime.utcnow()
            duration = (completed_at - started_at).total_seconds()
            
            # Parse JSON output from stdout
            try:
                json_data = json.loads(result.stdout)
                k6_result = K6Result.from_k6_json(
                    json_data,
                    script.name,
                    script.test_type.value,
                )
                k6_result.started_at = started_at
                k6_result.completed_at = completed_at
                k6_result.duration_seconds = duration
                k6_result.raw_stdout = result.stdout
                k6_result.raw_stderr = result.stderr
                k6_result.exit_code = result.returncode
                
                return k6_result
                
            except json.JSONDecodeError:
                # Failed to parse JSON - create result from exit code
                return K6Result(
                    success=result.returncode == 0,
                    test_name=script.name,
                    test_type=script.test_type.value,
                    started_at=started_at,
                    completed_at=completed_at,
                    duration_seconds=duration,
                    raw_stdout=result.stdout,
                    raw_stderr=result.stderr,
                    exit_code=result.returncode,
                )
                
        except subprocess.TimeoutExpired:
            return self._create_error_result(
                script,
                f"Test timed out after {timeout_seconds} seconds",
                started_at=started_at,
            )
        except Exception as e:
            return self._create_error_result(
                script,
                f"Execution error: {str(e)}",
                started_at=started_at,
            )
    
    def _create_error_result(
        self,
        script: TestScript,
        error_message: str,
        started_at: datetime = None,
    ) -> K6Result:
        """Create an error result"""
        now = datetime.utcnow()
        return K6Result(
            success=False,
            test_name=script.name,
            test_type=script.test_type.value,
            started_at=started_at or now,
            completed_at=now,
            raw_stderr=error_message,
            exit_code=1,
        )
    
    def execute_all(
        self,
        scripts: List[TestScript],
        timeout_per_test: int = 300,
        verbose: bool = False,
    ) -> List[K6Result]:
        """Execute multiple test scripts sequentially"""
        results = []
        
        for script in scripts:
            if verbose:
                print(f"  Running {script.test_type.value} test: {script.name}")
            
            result = self.execute(script, timeout_per_test, verbose)
            results.append(result)
            
            if verbose:
                if result.success:
                    print(f"    ✓ Completed: {result.metrics.total_requests} requests, "
                          f"P95: {result.metrics.latency_p95:.0f}ms, "
                          f"Error rate: {result.metrics.error_rate:.1%}")
                else:
                    print(f"    ✗ Failed: {result.raw_stderr[:100]}")
        
        return results
    
    def execute_autoscale_stress(
        self,
        script: TestScript,
        initial_vus: int,
        max_vus_limit: int = 1000,
        step_vus: int = 50,
        step_duration_seconds: int = 30,
        error_threshold: float = 0.05,
        latency_p95_threshold_ms: float = 5000,
        timeout_per_step: int = 120,
        verbose: bool = False,
    ) -> "AutoScaleResult":
        """
        Execute auto-scaling stress test with breaking point detection.
        
        Incrementally increases VUs until:
        - breaking point is detected (error_threshold OR latency_threshold)
        - max_vus_limit is reached
        - infra saturation is detected
        
        Each increment executes a REAL k6 stage.
        
        Args:
            script: Base TestScript with endpoints and config
            initial_vus: Starting VU count
            max_vus_limit: Maximum VUs to attempt (default 1000)
            step_vus: VU increment per stage (e.g., +50 or +100)
            step_duration_seconds: Duration to hold each stage
            error_threshold: Stop if error rate exceeds this (default 0.05 = 5%)
            latency_p95_threshold_ms: Stop if P95 latency exceeds this
            timeout_per_step: Timeout per k6 execution
            verbose: Print progress
            
        Returns:
            AutoScaleResult with all execution data
        """
        from sentinelperf.load.generator import TestType
        
        results = []
        current_vus = initial_vus
        max_vus_attempted = initial_vus
        max_vus_reached = 0
        stop_reason = None
        executed_stages = []
        
        if verbose:
            print(f"  Auto-scale stress: {initial_vus} → {max_vus_limit} VUs (step={step_vus})")
        
        iteration = 0
        while current_vus <= max_vus_limit and stop_reason is None:
            iteration += 1
            max_vus_attempted = current_vus
            
            # Create single-stage script for this VU level
            stage_script = TestScript(
                test_type=TestType.STRESS,
                name=f"autoscale_stage_{current_vus}vus",
                base_url=script.base_url,
                endpoints=script.endpoints,
                stages=[
                    {"duration": "5s", "target": current_vus},  # Ramp up
                    {"duration": f"{step_duration_seconds}s", "target": current_vus},  # Hold
                ],
                thresholds=script.thresholds,
                headers=script.headers,
            )
            
            if verbose:
                print(f"    Stage {iteration}: {current_vus} VUs...")
            
            # Execute this stage - REAL k6 execution
            result = self.execute(stage_script, timeout_per_step, verbose=False)
            result.test_type = f"autoscale_{current_vus}vus"
            results.append(result)
            executed_stages.append(current_vus)
            
            # Capture metrics
            error_rate = result.metrics.error_rate
            latency_p95 = result.metrics.latency_p95
            rps = result.metrics.requests_per_second
            
            if verbose:
                print(f"      → P95: {latency_p95:.0f}ms, "
                      f"Error: {error_rate:.1%}, "
                      f"RPS: {rps:.1f}")
            
            # Check for breaking point: ERROR THRESHOLD
            if error_rate >= error_threshold:
                stop_reason = "breaking_point_error"
                if verbose:
                    print(f"    ✗ Breaking point: Error rate {error_rate:.1%} >= {error_threshold:.1%} at {current_vus} VUs")
                break
            
            # Check for breaking point: LATENCY THRESHOLD
            if latency_p95 >= latency_p95_threshold_ms:
                stop_reason = "breaking_point_latency"
                if verbose:
                    print(f"    ✗ Breaking point: Latency P95 {latency_p95:.0f}ms >= {latency_p95_threshold_ms:.0f}ms at {current_vus} VUs")
                break
            
            # Check for k6 execution failure
            if not result.success and result.exit_code != 0:
                stop_reason = "execution_failure"
                if verbose:
                    print(f"    ✗ Execution failed at {current_vus} VUs")
                break
            
            # Stage completed successfully
            max_vus_reached = current_vus
            
            # Increment for next stage
            current_vus += step_vus
        
        # Determine final stop reason
        if stop_reason is None:
            if current_vus > max_vus_limit:
                stop_reason = "max_limit_reached"
                if verbose:
                    print(f"    ✓ Reached max limit: {max_vus_limit} VUs")
        
        if verbose:
            print(f"  Auto-scale complete: {len(results)} stages, "
                  f"max reached: {max_vus_reached} VUs, "
                  f"reason: {stop_reason}")
        
        return AutoScaleResult(
            results=results,
            max_vus_attempted=max_vus_attempted,
            max_vus_reached=max_vus_reached,
            stop_reason=stop_reason or "unknown",
            executed_stages=executed_stages,
        )
    
    def execute_adaptive(
        self,
        script: TestScript,
        initial_vus: int,
        max_vus: int,
        step: int,
        hold_seconds: int,
        error_threshold: float,
        latency_slope_threshold: float,
        fine_step_divisor: int,
        timeout_per_step: int = 120,
        verbose: bool = False,
    ) -> List[K6Result]:
        """
        Execute adaptive VU escalation.
        
        Increases VUs stepwise while:
        - error_rate < threshold
        - latency slope stable
        
        Switches to smaller increments after degradation detected.
        """
        results = []
        current_vus = initial_vus
        current_step = step
        previous_latency = None
        fine_mode = False
        
        if verbose:
            print(f"  Adaptive mode: {initial_vus} → {max_vus} VUs (step={step})")
        
        iteration = 0
        while current_vus <= max_vus:
            iteration += 1
            
            # Create single-stage script for this VU level
            adaptive_script = TestScript(
                test_type=TestType.STRESS,
                name=f"adaptive_step_{iteration}",
                base_url=script.base_url,
                endpoints=script.endpoints,
                stages=[
                    {"duration": f"{hold_seconds}s", "target": current_vus},
                ],
                thresholds=script.thresholds,
                headers=script.headers,
            )
            
            if verbose:
                mode_str = "[fine]" if fine_mode else "[coarse]"
                print(f"    Step {iteration}: {current_vus} VUs {mode_str}")
            
            # Execute this step
            result = self.execute(adaptive_script, timeout_per_step, verbose=False)
            result.test_type = f"adaptive_{current_vus}vus"
            results.append(result)
            
            if verbose:
                print(f"      → P95: {result.metrics.latency_p95:.0f}ms, "
                      f"Error: {result.metrics.error_rate:.1%}, "
                      f"RPS: {result.metrics.requests_per_second:.1f}")
            
            # Check error threshold
            if result.metrics.error_rate >= error_threshold:
                if verbose:
                    print(f"    ✗ Error threshold breached at {current_vus} VUs")
                break
            
            # Check latency slope
            if previous_latency and previous_latency > 0:
                latency_slope = result.metrics.latency_p95 / previous_latency
                
                if latency_slope >= latency_slope_threshold and not fine_mode:
                    fine_mode = True
                    current_step = max(1, step // fine_step_divisor)
                    if verbose:
                        print(f"    ⚠ Latency slope {latency_slope:.1f}x - switching to fine mode (step={current_step})")
                
                if fine_mode and latency_slope >= latency_slope_threshold:
                    if verbose:
                        print(f"    ✗ Sustained latency degradation at {current_vus} VUs")
                    break
            
            previous_latency = result.metrics.latency_p95
            current_vus += current_step
        
        if verbose:
            print(f"  Adaptive complete: {len(results)} steps executed")
        
        return results
