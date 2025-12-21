"""Simple test server that degrades under load for testing breaking point detection"""

import asyncio
import time
import random
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from threading import Lock

app = FastAPI()

# Global state for simulating degradation
class LoadState:
    def __init__(self):
        self.lock = Lock()
        self.concurrent_requests = 0
        self.total_requests = 0
        self.start_time = time.time()
    
    def begin_request(self):
        with self.lock:
            self.concurrent_requests += 1
            self.total_requests += 1
            return self.concurrent_requests
    
    def end_request(self):
        with self.lock:
            self.concurrent_requests -= 1
    
    @property
    def rps(self):
        elapsed = time.time() - self.start_time
        return self.total_requests / elapsed if elapsed > 0 else 0

state = LoadState()

# Configuration for degradation simulation
MAX_HEALTHY_CONCURRENT = 5  # Start degrading latency after 5 concurrent
ERROR_THRESHOLD_CONCURRENT = 50  # Much higher - errors only at extreme load
BASE_LATENCY_MS = 20  # Base latency in ms


@app.get("/")
@app.get("/health")
async def health():
    """Health check - always fast"""
    return {"status": "ok", "concurrent": state.concurrent_requests}


@app.get("/api/users")
@app.get("/api/orders")
@app.get("/api/products")
async def api_endpoint():
    """API endpoint that degrades under load"""
    concurrent = state.begin_request()
    
    try:
        # Calculate latency based on load
        if concurrent <= MAX_HEALTHY_CONCURRENT:
            # Healthy: low latency with small variance
            latency = BASE_LATENCY_MS + random.uniform(0, 10)
        elif concurrent <= ERROR_THRESHOLD_CONCURRENT:
            # Degraded: latency increases exponentially
            degradation_factor = (concurrent - MAX_HEALTHY_CONCURRENT) ** 2
            latency = BASE_LATENCY_MS + (degradation_factor * 50) + random.uniform(0, 20)
        else:
            # Overloaded: very high latency
            latency = BASE_LATENCY_MS + 500 + random.uniform(0, 200)
        
        # Simulate the latency
        await asyncio.sleep(latency / 1000)
        
        # Decide if request should error based on load
        error_probability = 0
        if concurrent > MAX_HEALTHY_CONCURRENT:
            # Error probability increases with load
            error_probability = min(0.8, (concurrent - MAX_HEALTHY_CONCURRENT) * 0.15)
        
        if random.random() < error_probability:
            raise HTTPException(status_code=500, detail="Server overloaded")
        
        return {
            "status": "success",
            "concurrent": concurrent,
            "latency_ms": round(latency, 2),
            "total_requests": state.total_requests,
        }
    finally:
        state.end_request()


@app.get("/api/slow")
async def slow_endpoint():
    """Endpoint that is always slow - for latency amplification testing"""
    concurrent = state.begin_request()
    try:
        # Base slow + exponential degradation
        latency = 200 + (concurrent ** 2 * 30)
        await asyncio.sleep(latency / 1000)
        return {"status": "success", "latency_ms": latency}
    finally:
        state.end_request()


if __name__ == "__main__":
    print("Starting test server on port 8765...")
    print(f"- Healthy up to {MAX_HEALTHY_CONCURRENT} concurrent requests")
    print(f"- Starts erroring at {ERROR_THRESHOLD_CONCURRENT}+ concurrent requests")
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="warning")
