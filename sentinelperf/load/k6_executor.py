"""k6 execution wrapper for SentinelPerf"""

import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime

from sentinelperf.load.generator import TestScript


@dataclass
class K6Result:
    """Results from k6 test execution"""
    success: bool
    test_name: str
    
    # Timing
    started_at: datetime
    completed_at: datetime
    
    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Rate metrics
    requests_per_second: float = 0.0
    error_rate: float = 0.0
    
    # Latency metrics (ms)
    latency_avg: float = 0.0
    latency_min: float = 0.0
    latency_max: float = 0.0
    latency_p50: float = 0.0
    latency_p90: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    
    # VU metrics
    vus_max: int = 0
    
    # Threshold results
    thresholds_passed: bool = True
    threshold_results: Dict[str, bool] = field(default_factory=dict)
    
    # Raw output
    raw_json: Dict[str, Any] = field(default_factory=dict)
    raw_stdout: str = ""
    raw_stderr: str = ""
    
    @classmethod
    def from_k6_json(cls, json_data: Dict[str, Any], test_name: str) -> "K6Result":
        """Parse k6 JSON summary output into K6Result"""
        
        metrics = json_data.get("metrics", {})
        
        # Extract HTTP request metrics
        http_reqs = metrics.get("http_reqs", {})
        http_req_failed = metrics.get("http_req_failed", {})
        http_req_duration = metrics.get("http_req_duration", {})
        
        # Calculate totals
        total_requests = int(http_reqs.get("values", {}).get("count", 0))
        error_rate = http_req_failed.get("values", {}).get("rate", 0.0)
        failed_requests = int(total_requests * error_rate)
        successful_requests = total_requests - failed_requests
        
        # Extract latency percentiles
        duration_values = http_req_duration.get("values", {})
        
        # Extract threshold results
        root_group = json_data.get("root_group", {})
        thresholds = json_data.get("thresholds", {})
        threshold_results = {}
        thresholds_passed = True
        
        for name, result in thresholds.items():
            passed = result.get("ok", True)
            threshold_results[name] = passed
            if not passed:
                thresholds_passed = False
        
        return cls(
            success=thresholds_passed,
            test_name=test_name,
            started_at=datetime.utcnow(),  # TODO: Extract from k6 output
            completed_at=datetime.utcnow(),
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            requests_per_second=http_reqs.get("values", {}).get("rate", 0.0),
            error_rate=error_rate,
            latency_avg=duration_values.get("avg", 0.0),
            latency_min=duration_values.get("min", 0.0),
            latency_max=duration_values.get("max", 0.0),
            latency_p50=duration_values.get("med", 0.0),
            latency_p90=duration_values.get("p(90)", 0.0),
            latency_p95=duration_values.get("p(95)", 0.0),
            latency_p99=duration_values.get("p(99)", 0.0),
            vus_max=int(metrics.get("vus_max", {}).get("values", {}).get("max", 0)),
            thresholds_passed=thresholds_passed,
            threshold_results=threshold_results,
            raw_json=json_data,
        )


class K6Executor:
    """
    Executes k6 load tests.
    
    Manages k6 binary discovery/bundling and test execution.
    """
    
    def __init__(
        self,
        k6_path: Optional[str] = None,
        output_dir: Path = Path("./k6-output"),
    ):
        self.k6_path = k6_path or self._find_k6()
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _find_k6(self) -> str:
        """
        Find k6 binary in system PATH or bundled location.
        
        Returns:
            Path to k6 binary
            
        Raises:
            RuntimeError: If k6 not found
        """
        # Check system PATH
        k6_path = shutil.which("k6")
        if k6_path:
            return k6_path
        
        # Check common installation locations
        common_paths = [
            "/usr/local/bin/k6",
            "/usr/bin/k6",
            Path.home() / ".local" / "bin" / "k6",
            Path.cwd() / "bin" / "k6",
        ]
        
        for path in common_paths:
            if Path(path).exists():
                return str(path)
        
        raise RuntimeError(
            "k6 not found. Please install k6: https://k6.io/docs/getting-started/installation/"
        )
    
    def check_available(self) -> bool:
        """Check if k6 is available and working"""
        try:
            result = subprocess.run(
                [self.k6_path, "version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def get_version(self) -> Optional[str]:
        """Get k6 version string"""
        try:
            result = subprocess.run(
                [self.k6_path, "version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None
    
    def execute(
        self,
        script: TestScript,
        timeout_seconds: int = 600,
        env_vars: Dict[str, str] = None,
    ) -> K6Result:
        """
        Execute a k6 test script.
        
        Args:
            script: TestScript to execute
            timeout_seconds: Maximum execution time
            env_vars: Additional environment variables
            
        Returns:
            K6Result with test metrics
        """
        # Generate script content
        script_content = script.to_k6_script()
        
        # Write to temporary file
        script_file = self.output_dir / f"{script.name}.js"
        with open(script_file, "w") as f:
            f.write(script_content)
        
        # Prepare output file
        json_output = self.output_dir / f"{script.name}_results.json"
        
        # Build k6 command
        cmd = [
            self.k6_path,
            "run",
            "--summary-export", str(json_output),
            str(script_file),
        ]
        
        # Prepare environment
        env = dict(__import__("os").environ)
        if env_vars:
            env.update(env_vars)
        
        started_at = datetime.utcnow()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=env,
            )
            
            completed_at = datetime.utcnow()
            
            # Parse JSON output
            if json_output.exists():
                with open(json_output, "r") as f:
                    json_data = json.load(f)
                
                k6_result = K6Result.from_k6_json(json_data, script.name)
                k6_result.started_at = started_at
                k6_result.completed_at = completed_at
                k6_result.raw_stdout = result.stdout
                k6_result.raw_stderr = result.stderr
                
                return k6_result
            else:
                # No JSON output, create error result
                return K6Result(
                    success=False,
                    test_name=script.name,
                    started_at=started_at,
                    completed_at=completed_at,
                    raw_stdout=result.stdout,
                    raw_stderr=result.stderr,
                )
                
        except subprocess.TimeoutExpired:
            return K6Result(
                success=False,
                test_name=script.name,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                raw_stderr=f"Test timed out after {timeout_seconds} seconds",
            )
        except Exception as e:
            return K6Result(
                success=False,
                test_name=script.name,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                raw_stderr=str(e),
            )
