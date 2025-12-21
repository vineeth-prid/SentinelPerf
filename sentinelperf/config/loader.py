"""Configuration loader for SentinelPerf"""

import yaml
from pathlib import Path
from typing import List, Optional

from sentinelperf.config.schema import SentinelPerfConfig, EnvironmentConfig


def load_config(config_path: Path, environment: str) -> SentinelPerfConfig:
    """
    Load and validate configuration from YAML file.
    
    Args:
        config_path: Path to sentinelperf.yaml
        environment: Environment name to activate
        
    Returns:
        Validated SentinelPerfConfig with active environment set
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If environment not found or validation fails
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)
    
    if not raw_config:
        raise ValueError("Configuration file is empty")
    
    # Parse and validate configuration
    config = SentinelPerfConfig(**raw_config)
    
    # Set active environment
    config.set_active_environment(environment)
    
    return config


def validate_config_file(config_path: Path) -> List[str]:
    """
    Validate configuration file without setting active environment.
    
    Args:
        config_path: Path to sentinelperf.yaml
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if not config_path.exists():
        return [f"Configuration file not found: {config_path}"]
    
    try:
        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return [f"YAML parsing error: {e}"]
    
    if not raw_config:
        return ["Configuration file is empty"]
    
    # Check required fields
    if "environments" not in raw_config:
        errors.append("Missing 'environments' section")
        return errors
    
    if not raw_config["environments"]:
        errors.append("No environments defined")
        return errors
    
    # Validate each environment
    for env_name, env_config in raw_config["environments"].items():
        try:
            EnvironmentConfig(**env_config)
        except Exception as e:
            errors.append(f"Environment '{env_name}': {e}")
    
    return errors


def create_example_config() -> str:
    """
    Generate example configuration YAML.
    
    Returns:
        Example configuration as YAML string
    """
    example = {
        "version": "1.0",
        "environments": {
            "local": {
                "target": {
                    "base_url": "http://localhost:8080",
                    "health_endpoint": "/health",
                    "endpoints": ["/api/users", "/api/orders"]
                },
                "auth": {
                    "method": "bearer",
                    "token": "${API_TOKEN}"
                },
                "telemetry": {
                    "otel": {
                        "enabled": True,
                        "endpoint": "http://localhost:4318"
                    },
                    "logs": {
                        "enabled": False,
                        "path": "/var/log/app/access.log"
                    },
                    "prometheus": {
                        "enabled": False,
                        "endpoint": "http://localhost:9090"
                    }
                },
                "load": {
                    "initial_vus": 1,
                    "max_vus": 50,
                    "ramp_duration": "30s",
                    "hold_duration": "60s",
                    "error_rate_threshold": 0.05,
                    "p95_latency_threshold_ms": 2000
                },
                "llm": {
                    "provider": "ollama",
                    "model": "qwen2.5:14b-instruct",
                    "base_url": "http://localhost:11434"
                }
            },
            "staging": {
                "target": {
                    "base_url": "https://staging.example.com",
                    "health_endpoint": "/health"
                },
                "auth": {
                    "method": "header",
                    "header_name": "X-API-Key",
                    "header_value": "${STAGING_API_KEY}"
                },
                "telemetry": {
                    "otel": {
                        "enabled": True,
                        "endpoint": "http://otel-collector.staging:4318"
                    }
                },
                "load": {
                    "initial_vus": 5,
                    "max_vus": 100,
                    "ramp_duration": "60s",
                    "hold_duration": "120s"
                }
            }
        }
    }
    
    return yaml.dump(example, default_flow_style=False, sort_keys=False)
