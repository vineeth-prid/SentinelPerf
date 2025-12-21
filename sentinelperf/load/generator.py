"""k6 test script generator for SentinelPerf"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class TestType(str, Enum):
    """Types of load tests"""
    LOAD = "load"      # Gradual ramp-up to target VUs
    STRESS = "stress"  # Push beyond normal capacity
    SPIKE = "spike"    # Sudden traffic burst
    SOAK = "soak"      # Extended duration test
    ADAPTIVE = "adaptive"  # Dynamically adjust based on response


@dataclass
class TestStage:
    """A stage in the load test"""
    duration: str  # e.g., "30s", "1m"
    target: int    # Target VUs


@dataclass
class TestScript:
    """Generated k6 test script"""
    test_type: TestType
    name: str
    base_url: str
    endpoints: List[Dict[str, Any]]
    stages: List[TestStage]
    thresholds: Dict[str, List[str]]
    headers: Dict[str, str] = field(default_factory=dict)
    script_content: str = ""
    
    def to_k6_script(self) -> str:
        """Generate k6 JavaScript test script"""
        stages_js = ",\n    ".join([
            f"{{ duration: '{s.duration}', target: {s.target} }}"
            for s in self.stages
        ])
        
        thresholds_js = ",\n    ".join([
            f"'{k}': {v}"
            for k, v in self.thresholds.items()
        ])
        
        headers_js = ",\n      ".join([
            f"'{k}': '{v}'"
            for k, v in self.headers.items()
        ])
        
        # Generate endpoint requests
        requests_js = []
        for ep in self.endpoints:
            method = ep.get("method", "GET").lower()
            path = ep.get("path", "/")
            weight = ep.get("weight", 1)
            
            requests_js.append(f"""
  // {method.upper()} {path} (weight: {weight})
  let res_{len(requests_js)} = http.{method}(`${{BASE_URL}}{path}`, {{
    headers: headers,
  }});
  check(res_{len(requests_js)}, {{
    '{path} status is 2xx': (r) => r.status >= 200 && r.status < 300,
  }});
  sleep(Math.random() * 2);""")
        
        script = f"""// SentinelPerf Generated Test: {self.name}
// Type: {self.test_type.value}
// Generated at: {__import__('datetime').datetime.utcnow().isoformat()}

import http from 'k6/http';
import {{ check, sleep }} from 'k6';
import {{ Rate, Trend }} from 'k6/metrics';

const BASE_URL = '{self.base_url}';

// Custom metrics
const errorRate = new Rate('errors');
const latencyTrend = new Trend('latency_trend');

// Test configuration
export const options = {{
  stages: [
    {stages_js}
  ],
  thresholds: {{
    {thresholds_js}
  }},
}};

// Request headers
const headers = {{
  'Content-Type': 'application/json',
  {headers_js}
}};

export default function () {{
  {''.join(requests_js)}
}}

// Summary handler for JSON output
export function handleSummary(data) {{
  return {{
    'stdout': JSON.stringify(data, null, 2),
  }};
}}
"""
        self.script_content = script
        return script


class TestGenerator:
    """
    Generates k6 test scripts based on telemetry insights.
    
    Creates appropriate test configurations for:
    - Load testing (baseline capacity)
    - Stress testing (breaking point identification)
    - Spike testing (burst handling)
    - Adaptive testing (real-time adjustment)
    """
    
    def __init__(
        self,
        base_url: str,
        auth_headers: Dict[str, str] = None,
    ):
        self.base_url = base_url
        self.auth_headers = auth_headers or {}
    
    def generate_load_test(
        self,
        endpoints: List[Dict[str, Any]],
        initial_vus: int = 1,
        target_vus: int = 10,
        ramp_duration: str = "30s",
        hold_duration: str = "60s",
        error_threshold: float = 0.05,
        p95_threshold_ms: int = 2000,
    ) -> TestScript:
        """Generate a standard load test"""
        
        stages = [
            TestStage(duration=ramp_duration, target=target_vus),
            TestStage(duration=hold_duration, target=target_vus),
            TestStage(duration="10s", target=0),
        ]
        
        thresholds = {
            "http_req_failed": [f"rate<{error_threshold}"],
            "http_req_duration": [f"p(95)<{p95_threshold_ms}"],
        }
        
        return TestScript(
            test_type=TestType.LOAD,
            name="load_test",
            base_url=self.base_url,
            endpoints=endpoints,
            stages=stages,
            thresholds=thresholds,
            headers=self.auth_headers,
        )
    
    def generate_stress_test(
        self,
        endpoints: List[Dict[str, Any]],
        max_vus: int = 100,
        step_duration: str = "30s",
        steps: int = 5,
    ) -> TestScript:
        """Generate a stress test with incremental load"""
        
        vus_per_step = max_vus // steps
        stages = []
        
        for i in range(1, steps + 1):
            stages.append(TestStage(
                duration=step_duration,
                target=vus_per_step * i,
            ))
        
        # Hold at max
        stages.append(TestStage(duration="60s", target=max_vus))
        
        # Ramp down
        stages.append(TestStage(duration="30s", target=0))
        
        thresholds = {
            "http_req_failed": ["rate<0.1"],  # More lenient for stress
            "http_req_duration": ["p(95)<5000"],
        }
        
        return TestScript(
            test_type=TestType.STRESS,
            name="stress_test",
            base_url=self.base_url,
            endpoints=endpoints,
            stages=stages,
            thresholds=thresholds,
            headers=self.auth_headers,
        )
    
    def generate_spike_test(
        self,
        endpoints: List[Dict[str, Any]],
        baseline_vus: int = 10,
        spike_vus: int = 100,
        spike_duration: str = "30s",
    ) -> TestScript:
        """Generate a spike test with sudden load burst"""
        
        stages = [
            TestStage(duration="30s", target=baseline_vus),  # Warmup
            TestStage(duration="10s", target=spike_vus),     # Spike up
            TestStage(duration=spike_duration, target=spike_vus),  # Hold spike
            TestStage(duration="10s", target=baseline_vus),  # Spike down
            TestStage(duration="30s", target=baseline_vus),  # Recovery
            TestStage(duration="10s", target=0),             # Ramp down
        ]
        
        thresholds = {
            "http_req_failed": ["rate<0.2"],  # Allow some failures during spike
            "http_req_duration": ["p(95)<10000"],
        }
        
        return TestScript(
            test_type=TestType.SPIKE,
            name="spike_test",
            base_url=self.base_url,
            endpoints=endpoints,
            stages=stages,
            thresholds=thresholds,
            headers=self.auth_headers,
        )
    
    def generate_adaptive_test(
        self,
        endpoints: List[Dict[str, Any]],
        initial_vus: int = 1,
        max_vus: int = 100,
        step_vus: int = 10,
        step_duration: str = "60s",
    ) -> TestScript:
        """
        Generate an adaptive test for breaking point detection.
        
        This test incrementally increases load until failure thresholds
        are exceeded, then identifies the breaking point.
        """
        
        # Generate incremental stages
        stages = []
        current_vus = initial_vus
        
        while current_vus <= max_vus:
            stages.append(TestStage(
                duration=step_duration,
                target=current_vus,
            ))
            current_vus += step_vus
        
        # Strict thresholds for breaking point detection
        thresholds = {
            "http_req_failed": ["rate<0.05"],
            "http_req_duration": ["p(95)<2000"],
        }
        
        return TestScript(
            test_type=TestType.ADAPTIVE,
            name="adaptive_test",
            base_url=self.base_url,
            endpoints=endpoints,
            stages=stages,
            thresholds=thresholds,
            headers=self.auth_headers,
        )
