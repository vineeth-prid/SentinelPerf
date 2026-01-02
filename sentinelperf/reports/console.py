"""Console output for SentinelPerf"""

from typing import Optional
from sentinelperf.core.state import ExecutionResult, AgentPhase, ExecutionStatus


# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def format_console_output(result: ExecutionResult, execution_id: str = "") -> str:
    """
    Format execution result for console output.
    
    Always prints:
    - Execution status (SUCCESS, SUCCESS_WITH_WARNINGS, FAILED_TO_EXECUTE)
    - Total test cases executed
    - Max VUs reached
    - Reason execution stopped
    
    Returns structured output with clear status indication.
    """
    lines = []
    state = result.state
    
    # Use execution_id from param or state
    exec_id = execution_id or state.execution_id or ""
    exec_id_short = exec_id[:8] if exec_id else "unknown"
    
    # Get execution status
    exec_status = result.get_execution_status()
    test_count = result.get_test_case_count()
    max_vus = result.get_max_vus_reached()
    stop_reason = result.get_stop_reason()
    
    # Line 1: Clear execution status with target URL
    if exec_status == ExecutionStatus.SUCCESS:
        status_label = f"{GREEN}SUCCESS{RESET}"
        status_icon = f"{GREEN}✓{RESET}"
    elif exec_status == ExecutionStatus.SUCCESS_WITH_WARNINGS:
        status_label = f"{YELLOW}SUCCESS_WITH_WARNINGS{RESET}"
        status_icon = f"{YELLOW}⚠{RESET}"
    else:
        status_label = f"{RED}FAILED_TO_EXECUTE{RESET}"
        status_icon = f"{RED}✗{RESET}"
    
    lines.append(f"{status_icon} SentinelPerf [{exec_id_short}]: {status_label}")
    lines.append(f"  Target: {state.target_url}")
    
    # Line 2: Test execution summary (always printed)
    lines.append(f"  Tests executed: {test_count} | Max VUs reached: {max_vus}")
    
    # Line 3: Stop reason (always printed)
    lines.append(f"  Stop reason: {stop_reason}")
    
    # Line 4: Breaking point or root cause summary
    if state.breaking_point and state.breaking_point.vus_at_break > 0:
        bp = state.breaking_point
        lines.append(
            f"  Breaking point: {bp.vus_at_break} VUs @ {bp.rps_at_break:.1f} RPS "
            f"({bp.failure_type})"
        )
    elif state.root_cause and state.root_cause.primary_cause:
        rc = state.root_cause
        confidence_bar = "●" * int(rc.confidence * 5) + "○" * (5 - int(rc.confidence * 5))
        cause = rc.primary_cause
        if len(cause) > 50:
            cause = cause[:50] + "..."
        lines.append(f"  Analysis: {cause} [{confidence_bar}]")
    
    # Line 5: Warnings summary (if any warnings exist)
    warnings = []
    if state.errors:
        warnings.extend(state.errors[:2])
    if state.infra_saturation and state.infra_saturation.get("warnings"):
        warnings.extend(state.infra_saturation.get("warnings", [])[:2])
    if state.root_cause and state.root_cause.llm_mode == "rules":
        warnings.append("LLM unavailable, used rules-based analysis")
    
    if warnings and exec_status == ExecutionStatus.SUCCESS_WITH_WARNINGS:
        warning_summary = warnings[0] if len(warnings) == 1 else f"{warnings[0]} (+{len(warnings)-1} more)"
        if len(warning_summary) > 60:
            warning_summary = warning_summary[:57] + "..."
        lines.append(f"  {YELLOW}Warning: {warning_summary}{RESET}")
    
    # Line 6: Report location
    if result.markdown_report_path:
        lines.append(f"  Report: {result.markdown_report_path}")
    
    return "\n".join(lines)


def print_summary(result: ExecutionResult, execution_id: str = "") -> None:
    """
    Print execution summary to console.
    
    Max 5 lines as per requirements.
    """
    output = format_console_output(result, execution_id)
    print(output)


def print_progress(phase: AgentPhase, message: str, verbose: bool = False) -> None:
    """Print progress message if verbose mode enabled"""
    if verbose:
        phase_icons = {
            AgentPhase.INIT: "⚙",
            AgentPhase.TELEMETRY_ANALYSIS: "📊",
            AgentPhase.TEST_GENERATION: "📝",
            AgentPhase.LOAD_EXECUTION: "🚀",
            AgentPhase.RESULTS_COLLECTION: "📈",
            AgentPhase.BREAKING_POINT_DETECTION: "🎯",
            AgentPhase.ROOT_CAUSE_ANALYSIS: "🔍",
            AgentPhase.RECOMMENDATIONS: "💡",
            AgentPhase.REPORT_GENERATION: "📄",
            AgentPhase.COMPLETE: "✅",
            AgentPhase.ERROR: "❌",
        }
        icon = phase_icons.get(phase, "•")
        print(f"{icon} [{phase.value}] {message}")
