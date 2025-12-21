"""k6 test script generator for SentinelPerf"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from enum import Enum
from datetime import datetime


class TestType(str, Enum):
    """Types of load tests"""
    BASELINE = "baseline"  # Validate baseline behavior
    STRESS = "stress"      # Incremental load to find limits
    SPIKE = "spike"        # Sudden traffic burst


@dataclass
class TestStage:
    """A stage in the load test"""
    duration: str  # e.g., "30s", "1m"
    target: int    # Target VUs


@dataclass
class TestEndpoint:
    """Endpoint configuration for testing"""
    path: str
    method: str = "GET"
    weight: int = 1  # Relative frequency
    body: str = ""   # Request body for POST/PUT


@dataclass
class TestScript:
    """Generated k6 test script"""
    test_type: TestType
    name: str
    base_url: str
    endpoints: List[TestEndpoint]
    stages: List[TestStage]
    thresholds: Dict[str, List[str]]
    headers: Dict[str, str] = field(default_factory=dict)
    
    def to_k6_script(self) -> str:
        """Generate deterministic k6 JavaScript test script"""
        
        # Format stages
        stages_js = ",\n    ".join([
            f"{{ duration: '{s.duration}', target: {s.target} }}"
            for s in self.stages
        ])
        
        # Format thresholds
        thresholds_js = ",\n    ".join([
            f"'{k}': {v}"
            for k, v in self.thresholds.items()
        ])
        
        # Format headers
        if self.headers:
            headers_entries = ",\n      ".join([
                f"'{k}': '{v}'"
                for k, v in self.headers.items()
            ])
            headers_js = f"""{{
      'Content-Type': 'application/json',
      {headers_entries}
    }}"""
        else:
            headers_js = "{ 'Content-Type': 'application/json' }"
        
        # Generate weighted endpoint selection
        total_weight = sum(ep.weight for ep in self.endpoints)
        endpoint_cases = []
        cumulative = 0
        
        for i, ep in enumerate(self.endpoints):
            cumulative += ep.weight
            threshold = cumulative / total_weight
            
            if ep.method.upper() == "GET":
                request_code = f"http.get(`${{BASE_URL}}{ep.path}`, {{ headers: headers }})"
            elif ep.method.upper() == "POST":
                body = ep.body or "{}"
                request_code = f"http.post(`${{BASE_URL}}{ep.path}`, '{body}', {{ headers: headers }})"
            elif ep.method.upper() == "PUT":
                body = ep.body or "{}"
                request_code = f"http.put(`${{BASE_URL}}{ep.path}`, '{body}', {{ headers: headers }})"
            elif ep.method.upper() == "DELETE":
                request_code = f"http.del(`${{BASE_URL}}{ep.path}`, null, {{ headers: headers }})"
            else:
                request_code = f"http.get(`${{BASE_URL}}{ep.path}`, {{ headers: headers }})"
            
            endpoint_cases.append(f"""  if (rand < {threshold:.4f}) {{
    res = {request_code};
    endpoint = '{ep.method} {ep.path}';
  }}""")
        
        endpoint_selection = " else ".join(endpoint_cases)
        
        script = f"""// SentinelPerf Generated Test: {self.name}
// Type: {self.test_type.value}
// Target: {self.base_url}
// Generated: {datetime.utcnow().isoformat()}Z

import http from 'k6/http';
import {{ check, sleep }} from 'k6';
import {{ Counter, Rate, Trend }} from 'k6/metrics';

const BASE_URL = '{self.base_url}';

// Custom metrics for SentinelPerf
const requestCount = new Counter('sentinelperf_requests');
const errorCount = new Counter('sentinelperf_errors');
const errorRate = new Rate('sentinelperf_error_rate');
const latencyTrend = new Trend('sentinelperf_latency');

// Test configuration
export const options = {{
  stages: [
    {stages_js}
  ],
  thresholds: {{
    {thresholds_js}
  }},
  // Output JSON summary
  summaryTrendStats: ['avg', 'min', 'med', 'max', 'p(50)', 'p(90)', 'p(95)', 'p(99)'],
}};

// Request headers
const headers = {headers_js};

export default function () {{
  const rand = Math.random();
  let res;
  let endpoint = '';
  
  {endpoint_selection}
  
  // Record metrics
  requestCount.add(1);
  latencyTrend.add(res.timings.duration);
  
  const success = res.status >= 200 && res.status < 400;
  errorRate.add(!success);
  
  if (!success) {{
    errorCount.add(1);
  }}
  
  check(res, {{
    'status is success': (r) => r.status >= 200 && r.status < 400,
  }});
  
  // Small sleep to simulate realistic user behavior
  sleep(0.1 + Math.random() * 0.4);
}}

// Generate JSON summary for SentinelPerf parsing
export function handleSummary(data) {{
  return {{
    'stdout': JSON.stringify({{
      sentinelperf_version: '1.0',
      test_type: '{self.test_type.value}',
      test_name: '{self.name}',
      timestamp: new Date().toISOString(),
      metrics: {{
        http_reqs: data.metrics.http_reqs,
        http_req_duration: data.metrics.http_req_duration,
        http_req_failed: data.metrics.http_req_failed,
        vus: data.metrics.vus,
        vus_max: data.metrics.vus_max,
        iterations: data.metrics.iterations,
        sentinelperf_requests: data.metrics.sentinelperf_requests,
        sentinelperf_errors: data.metrics.sentinelperf_errors,
        sentinelperf_error_rate: data.metrics.sentinelperf_error_rate,
        sentinelperf_latency: data.metrics.sentinelperf_latency,
      }},
      thresholds: data.thresholds,
    }}, null, 2),
  }};
}}
"""
        return script
    
    def save(self, path: str) -> str:
        """Save script to file and return path"""
        content = self.to_k6_script()
        with open(path, 'w') as f:
            f.write(content)
        return path


class TestGenerator:
    """
    Generates k6 test scripts based on baseline behavior.
    
    Creates three test types:
    1. Baseline test - Validates current behavior
    2. Stress test - Incremental load to find breaking point
    3. Spike test - Sudden burst to test resilience
    """
    
    def __init__(
        self,
        base_url: str,
        auth_headers: Dict[str, str] = None,
    ):
        self.base_url = base_url.rstrip('/')
        self.auth_headers = auth_headers or {}
    
    def _parse_endpoints(self, endpoints: List[Dict[str, Any]]) -> List[TestEndpoint]:
        """Convert endpoint dicts to TestEndpoint objects"""
        return [
            TestEndpoint(
                path=ep.get("path", "/"),
                method=ep.get("method", "GET"),
                weight=ep.get("weight", 1),
                body=ep.get("body", ""),
            )
            for ep in endpoints
        ]
    
    def generate_baseline_test(
        self,
        endpoints: List[Dict[str, Any]],
        target_vus: int = 5,
        duration: str = "30s",
        error_threshold: float = 0.05,
        p95_threshold_ms: int = 2000,
    ) -> TestScript:
        """
        Generate baseline validation test.
        
        Purpose: Verify system behaves as expected under normal load.
        """
        stages = [
            TestStage(duration="10s", target=target_vus),  # Ramp up
            TestStage(duration=duration, target=target_vus),  # Hold
            TestStage(duration="5s", target=0),  # Ramp down
        ]
        
        thresholds = {
            "http_req_failed": [f"rate<{error_threshold}"],
            "http_req_duration": [f"p(95)<{p95_threshold_ms}"],
        }
        
        return TestScript(
            test_type=TestType.BASELINE,
            name="baseline_test",
            base_url=self.base_url,
            endpoints=self._parse_endpoints(endpoints),
            stages=stages,
            thresholds=thresholds,
            headers=self.auth_headers,
        )
    
    def generate_stress_test(
        self,
        endpoints: List[Dict[str, Any]],
        start_vus: int = 1,
        max_vus: int = 50,
        step_vus: int = 10,
        step_duration: str = "30s",
    ) -> TestScript:
        """
        Generate stress test with incremental load.
        
        Purpose: Find the breaking point by gradually increasing load.
        """
        stages = []
        current_vus = start_vus
        
        # Initial ramp (shorter)
        stages.append(TestStage(duration="5s", target=start_vus))
        
        # Incremental steps
        while current_vus < max_vus:
            current_vus = min(current_vus + step_vus, max_vus)
            stages.append(TestStage(duration=step_duration, target=current_vus))
        
        # Hold at max (shorter)
        stages.append(TestStage(duration="10s", target=max_vus))
        
        # Ramp down (shorter)
        stages.append(TestStage(duration="5s", target=0))
        
        # More lenient thresholds for stress test - we expect failures
        thresholds = {
            "http_req_failed": ["rate<0.5"],  # Allow up to 50% failures
            "http_req_duration": ["p(95)<10000"],  # 10s timeout
        }
        
        return TestScript(
            test_type=TestType.STRESS,
            name="stress_test",
            base_url=self.base_url,
            endpoints=self._parse_endpoints(endpoints),
            stages=stages,
            thresholds=thresholds,
            headers=self.auth_headers,
        )
    
    def generate_spike_test(
        self,
        endpoints: List[Dict[str, Any]],
        baseline_vus: int = 5,
        spike_vus: int = 50,
        spike_duration: str = "20s",
    ) -> TestScript:
        """
        Generate spike test with sudden load burst.
        
        Purpose: Test system resilience to sudden traffic spikes.
        """
        stages = [
            TestStage(duration="10s", target=baseline_vus),   # Warmup
            TestStage(duration="5s", target=spike_vus),       # Spike up
            TestStage(duration=spike_duration, target=spike_vus),  # Hold spike
            TestStage(duration="5s", target=baseline_vus),    # Spike down
            TestStage(duration="20s", target=baseline_vus),   # Recovery
            TestStage(duration="5s", target=0),               # Ramp down
        ]
        
        # Lenient thresholds during spike
        thresholds = {
            "http_req_failed": ["rate<0.3"],  # Allow up to 30% failures
            "http_req_duration": ["p(95)<8000"],  # 8s during spike
        }
        
        return TestScript(
            test_type=TestType.SPIKE,
            name="spike_test",
            base_url=self.base_url,
            endpoints=self._parse_endpoints(endpoints),
            stages=stages,
            thresholds=thresholds,
            headers=self.auth_headers,
        )
    
    # Keep old methods for backward compatibility
    def generate_load_test(self, endpoints, **kwargs):
        """Alias for generate_baseline_test"""
        return self.generate_baseline_test(
            endpoints,
            target_vus=kwargs.get("target_vus", kwargs.get("initial_vus", 5)),
            duration=kwargs.get("hold_duration", "30s"),
            error_threshold=kwargs.get("error_threshold", 0.05),
            p95_threshold_ms=kwargs.get("p95_threshold_ms", 2000),
        )
    
    def generate_adaptive_test(self, endpoints, **kwargs):
        """Alias for generate_stress_test with adaptive naming"""
        return self.generate_stress_test(
            endpoints,
            start_vus=kwargs.get("initial_vus", 1),
            max_vus=kwargs.get("max_vus", 50),
            step_vus=kwargs.get("step_vus", 10),
            step_duration=kwargs.get("step_duration", "30s"),
        )
