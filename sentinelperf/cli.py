"""CLI entry point for SentinelPerf Runner"""

import argparse
import sys
from pathlib import Path

from sentinelperf.config.loader import load_config
from sentinelperf.core.agent import SentinelPerfAgent
from sentinelperf.reports.console import print_summary


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser"""
    parser = argparse.ArgumentParser(
        prog="sentinelperf",
        description="SentinelPerf AI - Autonomous Performance Engineering Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sentinelperf run --env=staging
  sentinelperf run --env=production --config=./sentinelperf.yaml
  sentinelperf validate --config=./sentinelperf.yaml
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Execute performance analysis")
    run_parser.add_argument(
        "--env", "-e",
        required=True,
        help="Target environment name (must be defined in config)"
    )
    run_parser.add_argument(
        "--config", "-c",
        default="./sentinelperf.yaml",
        help="Path to configuration file (default: ./sentinelperf.yaml)"
    )
    run_parser.add_argument(
        "--output-dir", "-o",
        default="./sentinelperf-reports",
        help="Directory for output reports (default: ./sentinelperf-reports)"
    )
    run_parser.add_argument(
        "--llm-mode",
        choices=["ollama", "rules", "mock"],
        default="ollama",
        help="LLM mode: ollama (default), rules (rule-based fallback), mock (testing)"
    )
    run_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate configuration file")
    validate_parser.add_argument(
        "--config", "-c",
        default="./sentinelperf.yaml",
        help="Path to configuration file (default: ./sentinelperf.yaml)"
    )
    
    # Version command
    parser.add_argument(
        "--version", "-V",
        action="version",
        version="%(prog)s 0.1.0"
    )
    
    return parser


def cmd_run(args: argparse.Namespace) -> int:
    """Execute the run command"""
    config_path = Path(args.config)
    
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}", file=sys.stderr)
        return 1
    
    try:
        config = load_config(config_path, args.env)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        return 1
    
    if args.verbose:
        print(f"Loaded configuration for environment: {args.env}")
        print(f"Target URL: {config.target.base_url}")
    
    # Initialize and run agent
    agent = SentinelPerfAgent(
        config=config,
        llm_mode=args.llm_mode,
        output_dir=Path(args.output_dir),
        verbose=args.verbose
    )
    
    result = agent.run()
    
    # Print console summary (max 5 lines)
    print_summary(result)
    
    return 0 if result.success else 1


def cmd_validate(args: argparse.Namespace) -> int:
    """Execute the validate command"""
    config_path = Path(args.config)
    
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}", file=sys.stderr)
        return 1
    
    try:
        from sentinelperf.config.loader import validate_config_file
        errors = validate_config_file(config_path)
        
        if errors:
            print("Configuration validation failed:")
            for error in errors:
                print(f"  - {error}")
            return 1
        
        print(f"Configuration valid: {config_path}")
        return 0
        
    except Exception as e:
        print(f"Error validating configuration: {e}", file=sys.stderr)
        return 1


def main() -> int:
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    if args.command == "run":
        return cmd_run(args)
    elif args.command == "validate":
        return cmd_validate(args)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
