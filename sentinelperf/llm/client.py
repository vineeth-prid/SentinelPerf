"""LLM client implementations for SentinelPerf"""

import json
import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import httpx

from sentinelperf.config.schema import LLMConfig


@dataclass
class LLMResponse:
    """Response from LLM"""
    content: str
    model: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    raw_response: Dict[str, Any] = None


class LLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        """Generate completion from LLM"""
        pass
    
    @abstractmethod
    async def check_available(self) -> bool:
        """Check if LLM service is available"""
        pass


class OllamaClient(LLMClient):
    """
    Ollama client for local LLM inference.
    
    Supports Qwen2.5-14B-Instruct (primary) and Qwen2.5-7B-Instruct (fallback).
    
    LLM Usage Rules (enforced at prompt level):
    - LLM may NOT invent metrics
    - LLM may NOT infer causes without observed signals
    - LLM must explain reasoning step-by-step
    - LLM must assign confidence scores based on signal strength
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self.model = config.model
        self.fallback_model = config.fallback_model
        self.timeout = config.timeout
        self.temperature = config.temperature
    
    async def check_available(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                # Check Ollama API
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code != 200:
                    return False
                
                # Check if model is available
                data = response.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                
                # Check for primary or fallback model
                model_base = self.model.split(":")[0]
                fallback_base = self.fallback_model.split(":")[0]
                
                for m in models:
                    if model_base in m or fallback_base in m:
                        return True
                
                return False
                
        except (httpx.RequestError, json.JSONDecodeError):
            return False
    
    async def list_models(self) -> List[str]:
        """List available models in Ollama"""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    return [m.get("name", "") for m in data.get("models", [])]
        except (httpx.RequestError, json.JSONDecodeError):
            pass
        return []
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        """
        Generate completion using Ollama API.
        
        Attempts primary model first, falls back to smaller model on failure.
        """
        temperature = temperature if temperature is not None else self.temperature
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Try primary model first
        result = await self._call_ollama(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        if result:
            return result
        
        # Try fallback model
        result = await self._call_ollama(
            model=self.fallback_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        if result:
            return result
        
        # Return empty response if both fail
        return LLMResponse(
            content="LLM unavailable - using rule-based analysis",
            model="none",
            tokens_used=0,
        )
    
    async def _call_ollama(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Optional[LLMResponse]:
        """Make request to Ollama API"""
        import time
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                
                latency_ms = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    
                    return LLMResponse(
                        content=data.get("message", {}).get("content", ""),
                        model=model,
                        tokens_used=data.get("eval_count", 0),
                        latency_ms=latency_ms,
                        raw_response=data,
                    )
                    
        except (httpx.RequestError, json.JSONDecodeError, asyncio.TimeoutError):
            pass
        
        return None


class MockLLMClient(LLMClient):
    """Mock LLM client for testing"""
    
    async def check_available(self) -> bool:
        return True
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        return LLMResponse(
            content="Mock response - LLM analysis disabled",
            model="mock",
            tokens_used=0,
        )
