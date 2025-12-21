// SentinelPerf Generated Test: spike_test
// Type: spike
// Target: http://localhost:8765
// Generated: 2025-12-21T18:17:11.796376Z

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';

const BASE_URL = 'http://localhost:8765';

// Custom metrics for SentinelPerf
const requestCount = new Counter('sentinelperf_requests');
const errorCount = new Counter('sentinelperf_errors');
const errorRate = new Rate('sentinelperf_error_rate');
const latencyTrend = new Trend('sentinelperf_latency');

// Test configuration
export const options = {
  stages: [
    { duration: '5s', target: 1 },
    { duration: '3s', target: 15 },
    { duration: '20s', target: 15 },
    { duration: '3s', target: 1 },
    { duration: '10s', target: 1 },
    { duration: '3s', target: 0 }
  ],
  thresholds: {
    'http_req_failed': ['rate<0.3'],
    'http_req_duration': ['p(95)<8000']
  },
  // Output JSON summary
  summaryTrendStats: ['avg', 'min', 'med', 'max', 'p(50)', 'p(90)', 'p(95)', 'p(99)'],
};

// Request headers
const headers = { 'Content-Type': 'application/json' };

export default function () {
  const rand = Math.random();
  let res;
  let endpoint = '';
  
    if (rand < 0.1852) {
    res = http.get(`${BASE_URL}/api/users`, { headers: headers });
    endpoint = 'GET /api/users';
  } else   if (rand < 0.3704) {
    res = http.get(`${BASE_URL}/api/orders`, { headers: headers });
    endpoint = 'GET /api/orders';
  } else   if (rand < 0.5556) {
    res = http.get(`${BASE_URL}/api/products`, { headers: headers });
    endpoint = 'GET /api/products';
  } else   if (rand < 0.6667) {
    res = http.post(`${BASE_URL}/api/users`, '{}', { headers: headers });
    endpoint = 'POST /api/users';
  } else   if (rand < 0.7778) {
    res = http.get(`${BASE_URL}/api/users/:id`, { headers: headers });
    endpoint = 'GET /api/users/:id';
  } else   if (rand < 0.8889) {
    res = http.del(`${BASE_URL}/api/orders/:id`, null, { headers: headers });
    endpoint = 'DELETE /api/orders/:id';
  } else   if (rand < 1.0000) {
    res = http.get(`${BASE_URL}/health`, { headers: headers });
    endpoint = 'GET /health';
  }
  
  // Record metrics
  requestCount.add(1);
  latencyTrend.add(res.timings.duration);
  
  const success = res.status >= 200 && res.status < 400;
  errorRate.add(!success);
  
  if (!success) {
    errorCount.add(1);
  }
  
  check(res, {
    'status is success': (r) => r.status >= 200 && r.status < 400,
  });
  
  // Small sleep to simulate realistic user behavior
  sleep(0.1 + Math.random() * 0.4);
}

// Generate JSON summary for SentinelPerf parsing
export function handleSummary(data) {
  return {
    'stdout': JSON.stringify({
      sentinelperf_version: '1.0',
      test_type: 'spike',
      test_name: 'spike_test',
      timestamp: new Date().toISOString(),
      metrics: {
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
      },
      thresholds: data.thresholds,
    }, null, 2),
  };
}
