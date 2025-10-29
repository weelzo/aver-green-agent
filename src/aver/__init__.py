"""
AVER (Agent Verification & Error Recovery) Benchmark

A benchmark for evaluating AI agents' ability to detect, diagnose, and recover from errors.

Core Components:
- GreenAgent: Main orchestration engine
- TaskSuite: Task loading and selection
- ReliabilityEvaluator: Detection/Diagnosis/Recovery scoring
- ErrorInjector: Error injection mechanisms
- TraceAnalyzer: Agent execution trace analysis
"""

__version__ = "0.1.0"
__author__ = "AVER Research Team"

from .models import (
    TaskScenario,
    ErrorInjection,
    DetectionSignals,
    RecoveryCriteria,
    EvaluationMetrics,
)

__all__ = [
    "TaskScenario",
    "ErrorInjection",
    "DetectionSignals",
    "RecoveryCriteria",
    "EvaluationMetrics",
]
