"""Configuration schema definitions using Pydantic"""

from typing import Optional, Literal, Dict, Any, List
from pydantic import BaseModel, Field, HttpUrl, field_validator


class AuthConfig(BaseModel):
    """Authentication configuration"""
    method: Literal["bearer", "header", "none"] = "none"
    token: Optional[str] = None
    header_name: Optional[str] = None
    header_value: Optional[str] = None
    
    @field_validator("token", "header_value", mode="before")
    @classmethod
    def resolve_env_vars(cls, v):
        """Resolve environment variables in format ${VAR_NAME}"""
        if v and isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            import os
            env_var = v[2:-1]
            return os.environ.get(env_var, v)
        return v


class TelemetrySourceConfig(BaseModel):
    """Configuration for a single telemetry source"""
    enabled: bool = True
    endpoint: Optional[str] = None
    path: Optional[str] = None  # For log files
    query: Optional[str] = None  # For Prometheus queries


class TelemetryConfig(BaseModel):
    """Telemetry configuration with auto-priority"""
    # Priority: otel > logs > prometheus (auto-detected)
    otel: Optional[TelemetrySourceConfig] = None
    logs: Optional[TelemetrySourceConfig] = None
    prometheus: Optional[TelemetrySourceConfig] = None
    
    def get_active_source(self) -> Optional[str]:
        """Get the highest priority active telemetry source"""
        if self.otel and self.otel.enabled:
            return "otel"
        if self.logs and self.logs.enabled:
            return "logs"
        if self.prometheus and self.prometheus.enabled:
            return "prometheus"
        return None


class LoadConfig(BaseModel):
    """Load testing configuration"""
    # Starting parameters
    initial_vus: int = Field(default=1, ge=1, description="Initial virtual users")
    max_vus: int = Field(default=100, ge=1, description="Maximum virtual users")
    
    # Duration settings
    ramp_duration: str = Field(default="30s", description="Ramp-up duration")
    hold_duration: str = Field(default="60s", description="Hold duration at peak")
    
    # Thresholds
    error_rate_threshold: float = Field(default=0.05, ge=0, le=1, description="Error rate threshold (0-1)")
    p95_latency_threshold_ms: int = Field(default=2000, ge=0, description="P95 latency threshold in ms")
    
    # Adaptive settings
    adaptive_step: int = Field(default=10, ge=1, description="VU increment step for adaptive load")
    adaptive_enabled: bool = Field(default=False, description="Enable adaptive VU escalation")
    adaptive_hold_seconds: int = Field(default=15, ge=5, description="Hold duration per step in adaptive mode")
    adaptive_latency_slope_threshold: float = Field(default=2.0, ge=1.0, description="Max latency slope before switching to fine mode")
    adaptive_fine_step_divisor: int = Field(default=4, ge=2, description="Divisor for fine step after degradation")


class LLMConfig(BaseModel):
    """LLM configuration for analysis"""
    provider: Literal["ollama", "rules", "mock"] = "ollama"
    model: str = "qwen2.5:14b"  # User specifies as qwen2.5:14b
    fallback_model: str = "qwen2.5:7b"
    base_url: str = "http://localhost:11434"
    timeout: int = Field(default=60, ge=1, description="LLM request timeout in seconds")
    temperature: float = Field(default=0.1, ge=0, le=2, description="LLM temperature")


class RecommendationsConfig(BaseModel):
    """Recommendations configuration"""
    enabled: bool = True  # Generate recommendations
    polish_with_llm: bool = False  # LLM polishing OFF by default (token-safe)
    max_recommendations: int = Field(default=5, ge=1, le=10, description="Max recommendations to show")


class TargetConfig(BaseModel):
    """Target application configuration"""
    base_url: str = Field(..., description="Base URL of target application")
    health_endpoint: str = Field(default="/health", description="Health check endpoint")
    endpoints: Optional[List[str]] = Field(default=None, description="Specific endpoints to test")
    
    @field_validator("base_url", mode="before")
    @classmethod
    def normalize_url(cls, v):
        """Remove trailing slash from URL"""
        if v and isinstance(v, str):
            return v.rstrip("/")
        return v


class EnvironmentConfig(BaseModel):
    """Configuration for a specific environment"""
    target: TargetConfig
    auth: AuthConfig = Field(default_factory=AuthConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    load: LoadConfig = Field(default_factory=LoadConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    recommendations: RecommendationsConfig = Field(default_factory=RecommendationsConfig)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SentinelPerfConfig(BaseModel):
    """Root configuration model"""
    version: str = "1.0"
    environments: Dict[str, EnvironmentConfig]
    
    # Resolved config for selected environment
    _active_env: Optional[str] = None
    
    @property
    def target(self) -> TargetConfig:
        """Get target config for active environment"""
        if not self._active_env:
            raise ValueError("No active environment set")
        return self.environments[self._active_env].target
    
    @property
    def auth(self) -> AuthConfig:
        """Get auth config for active environment"""
        if not self._active_env:
            raise ValueError("No active environment set")
        return self.environments[self._active_env].auth
    
    @property
    def telemetry(self) -> TelemetryConfig:
        """Get telemetry config for active environment"""
        if not self._active_env:
            raise ValueError("No active environment set")
        return self.environments[self._active_env].telemetry
    
    @property
    def load(self) -> LoadConfig:
        """Get load config for active environment"""
        if not self._active_env:
            raise ValueError("No active environment set")
        return self.environments[self._active_env].load
    
    @property
    def llm(self) -> LLMConfig:
        """Get LLM config for active environment"""
        if not self._active_env:
            raise ValueError("No active environment set")
        return self.environments[self._active_env].llm
    
    @property
    def recommendations(self) -> RecommendationsConfig:
        """Get recommendations config for active environment"""
        if not self._active_env:
            raise ValueError("No active environment set")
        return self.environments[self._active_env].recommendations
    
    def set_active_environment(self, env_name: str) -> None:
        """Set the active environment"""
        if env_name not in self.environments:
            available = list(self.environments.keys())
            raise ValueError(f"Environment '{env_name}' not found. Available: {available}")
        self._active_env = env_name
