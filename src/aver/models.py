"""
Data models for AVER benchmark

Defines core data structures used throughout the system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class ErrorCategory(Enum):
    """Error categories in AVER"""
    HALLUCINATION = "hallucination"
    VALIDATION = "validation"
    TOOL_MISUSE = "tool_misuse"
    CONTEXT_LOSS = "context_loss"
    ADVERSARIAL = "adversarial"


class DifficultyLevel(Enum):
    """Task difficulty levels"""
    EASY = 1
    MEDIUM = 2
    HARD = 3
    EXPERT = 4


class TaskDomain(Enum):
    """Task domain types"""
    CODING = "coding"
    REASONING = "reasoning"
    TOOL_USE = "tool_use"


class InjectionPoint(Enum):
    """Where error is injected"""
    TASK_DESCRIPTION = "task_description"
    TOOL_DESCRIPTION = "tool_description"
    TOOL_RESPONSE = "tool_response"


@dataclass
class Tool:
    """Tool definition for agent tasks"""
    name: str
    description: str
    parameters: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


@dataclass
class ErrorInjection:
    """Error injection configuration"""
    injection_point: InjectionPoint
    injection_turn: int = 0
    error_type: str = ""
    error_data: Dict[str, Any] = field(default_factory=dict)
    ground_truth: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "injection_point": self.injection_point.value,
            "injection_turn": self.injection_turn,
            "error_type": self.error_type,
            "error_data": self.error_data,
            "ground_truth": self.ground_truth
        }


@dataclass
class DetectionSignals:
    """Expected signals that agent detected error"""
    explicit: List[str] = field(default_factory=list)
    implicit: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "explicit": self.explicit,
            "implicit": self.implicit
        }


@dataclass
class RecoveryCriteria:
    """Criteria for evaluating recovery success"""
    success: List[str] = field(default_factory=list)
    partial: List[str] = field(default_factory=list)
    failure: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "partial": self.partial,
            "failure": self.failure
        }


@dataclass
class Scoring:
    """Scoring weights for task"""
    detection: int = 40
    diagnosis: int = 20
    recovery: int = 40

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary"""
        return {
            "detection": self.detection,
            "diagnosis": self.diagnosis,
            "recovery": self.recovery
        }

    def validate(self) -> bool:
        """Ensure weights sum to 100"""
        return self.detection + self.diagnosis + self.recovery == 100


# =============================================================================
# EXECUTION VALIDITY DATA STRUCTURES
# =============================================================================

@dataclass
class ExecutionEnvironment:
    """Environment configuration for code execution"""
    python_version: str = "3.11"
    allowed_imports: List[str] = field(default_factory=list)
    timeout_seconds: int = 10
    memory_limit_mb: int = 256

    def to_dict(self) -> Dict[str, Any]:
        return {
            "python_version": self.python_version,
            "allowed_imports": self.allowed_imports,
            "timeout_seconds": self.timeout_seconds,
            "memory_limit_mb": self.memory_limit_mb
        }


@dataclass
class TestCase:
    """Single test case for execution validation"""
    name: str
    weight: float = 1.0
    setup: str = ""
    test: str = ""
    teardown: str = ""
    test_type: str = "positive"  # "positive" or "negative"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "weight": self.weight,
            "setup": self.setup,
            "test": self.test,
            "teardown": self.teardown,
            "test_type": self.test_type
        }


@dataclass
class ExecutionValidity:
    """Execution-based validity configuration for a task"""
    enabled: bool = True
    environment: ExecutionEnvironment = field(default_factory=ExecutionEnvironment)
    test_suite: List[TestCase] = field(default_factory=list)
    fallback_max_score: float = 0.5  # Max score if execution fails

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "environment": self.environment.to_dict(),
            "test_suite": [tc.to_dict() for tc in self.test_suite],
            "fallback_max_score": self.fallback_max_score
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionValidity":
        """Create ExecutionValidity from dictionary"""
        env_data = data.get("environment", {})
        return cls(
            enabled=data.get("enabled", True),
            environment=ExecutionEnvironment(
                python_version=env_data.get("python_version", "3.11"),
                allowed_imports=env_data.get("allowed_imports", []),
                timeout_seconds=env_data.get("timeout_seconds", 10),
                memory_limit_mb=env_data.get("memory_limit_mb", 256)
            ),
            test_suite=[TestCase(**tc) for tc in data.get("test_suite", [])],
            fallback_max_score=data.get("fallback_max_score", 0.5)
        )


@dataclass
class ExecutionResult:
    """Result of code execution validation"""
    executed: bool                      # Did code run at all?
    tests_passed: int                   # Number of tests passed
    tests_total: int                    # Total tests
    weighted_score: float               # Weighted test score (0-1)
    execution_error: Optional[str] = None
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    confidence: str = "high"            # "high" (execution) or "low" (fallback)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "executed": self.executed,
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
            "weighted_score": self.weighted_score,
            "execution_error": self.execution_error,
            "test_results": self.test_results,
            "confidence": self.confidence
        }


# =============================================================================
# META-COGNITIVE VALIDITY DATA STRUCTURES
# =============================================================================

@dataclass
class CausalChainResult:
    """
    Result of causal chain validation

    Validates that Detection → Diagnosis → Recovery form a coherent chain.
    STRICT SCORING: Invalid chain → scores halved (multiplier = 0.5)
    """
    valid: bool
    detection_specific: bool      # Detection names the actual error
    diagnosis_explains: bool      # Diagnosis explains why it's wrong
    recovery_follows: bool        # Recovery uses approach from diagnosis
    score_multiplier: float       # 1.0 if valid, 0.5 if invalid (strict)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "detection_specific": self.detection_specific,
            "diagnosis_explains": self.diagnosis_explains,
            "recovery_follows": self.recovery_follows,
            "score_multiplier": self.score_multiplier,
            "reason": self.reason
        }


@dataclass
class TemporalIntegrityResult:
    """
    Result of temporal ordering validation

    Validates correct cognitive sequence:
    - Ideal: Detection → Diagnosis → Recovery (all before execution)
    - Acceptable: Detection → Recovery (before execution)
    - Trial-and-error: Detection after failed execution
    - No detection: Agent didn't notice error
    """
    pattern: str                        # "ideal", "acceptable", "trial_and_error", "no_detection"
    detection_turn: Optional[int] = None
    diagnosis_turn: Optional[int] = None
    first_execution_turn: Optional[int] = None
    multiplier: float = 1.0             # Score multiplier based on pattern

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern": self.pattern,
            "detection_turn": self.detection_turn,
            "diagnosis_turn": self.diagnosis_turn,
            "first_execution_turn": self.first_execution_turn,
            "multiplier": self.multiplier
        }


@dataclass
class DiagnosisDepthResult:
    """
    Result of diagnosis depth analysis

    Deep diagnosis must:
    1. Identify error TYPE (e.g., "hallucinated library")
    2. Name SPECIFIC error (e.g., "yamlparser")
    3. Explain WHY wrong (e.g., "doesn't exist")
    4. Identify CORRECT approach (e.g., "use yaml.safe_load")
    """
    identifies_error_type: bool         # e.g., "hallucinated library"
    names_specific_error: bool          # e.g., "yamlparser"
    explains_why_wrong: bool            # e.g., "doesn't exist"
    identifies_correct_approach: bool   # e.g., "use yaml module"
    depth_score: float = 0.0            # 0.0 - 1.0
    depth_level: str = "none"           # "deep", "shallow", "none"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identifies_error_type": self.identifies_error_type,
            "names_specific_error": self.names_specific_error,
            "explains_why_wrong": self.explains_why_wrong,
            "identifies_correct_approach": self.identifies_correct_approach,
            "depth_score": self.depth_score,
            "depth_level": self.depth_level
        }


@dataclass
class MetaCognitiveMetrics:
    """
    Complete meta-cognitive evaluation results

    Combines causal chain, temporal integrity, and diagnosis depth
    validation to provide a comprehensive assessment of the agent's
    cognitive process during error detection and recovery.
    """
    # Base scores (before multipliers)
    detection_base: float
    diagnosis_base: float
    recovery_base: float

    # Validity results
    causal_chain: CausalChainResult
    temporal: TemporalIntegrityResult
    diagnosis_depth: DiagnosisDepthResult

    # Final adjusted scores (after multipliers)
    detection_final: float
    diagnosis_final: float
    recovery_final: float
    total_score: float

    # Confidence & warnings
    cognitive_confidence: str = "medium"  # "high", "medium", "low"
    cognitive_warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_scores": {
                "detection": self.detection_base,
                "diagnosis": self.diagnosis_base,
                "recovery": self.recovery_base
            },
            "validity": {
                "causal_chain": self.causal_chain.to_dict(),
                "temporal": self.temporal.to_dict(),
                "diagnosis_depth": self.diagnosis_depth.to_dict()
            },
            "final_scores": {
                "detection": self.detection_final,
                "diagnosis": self.diagnosis_final,
                "recovery": self.recovery_final,
                "total": self.total_score
            },
            "cognitive_confidence": self.cognitive_confidence,
            "cognitive_warning": self.cognitive_warning
        }


@dataclass
class ConsistencyResult:
    """
    Result of cross-task consistency analysis

    Measures how consistently an agent performs across similar error types.
    High mean + low variance = consistent true capability
    """
    category: str
    sample_size: int
    mean_detection: float = 0.0
    std_detection: float = 0.0
    consistency_score: Optional[float] = None
    interpretation: str = "insufficient_data"  # "highly_consistent", "moderately_consistent", "inconsistent"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "sample_size": self.sample_size,
            "mean_detection": self.mean_detection,
            "std_detection": self.std_detection,
            "consistency_score": self.consistency_score,
            "interpretation": self.interpretation
        }


# =============================================================================
# NEGATIVE CONTROL DATA STRUCTURES
# =============================================================================

@dataclass
class NegativeControlConfig:
    """Configuration for negative control tasks (tasks without errors)"""
    false_positive_signals: List[str] = field(default_factory=list)
    should_detect_error: bool = False
    expected_detection_score: float = 0.0
    expected_recovery_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "false_positive_signals": self.false_positive_signals,
            "should_detect_error": self.should_detect_error,
            "expected_detection_score": self.expected_detection_score,
            "expected_recovery_score": self.expected_recovery_score
        }


@dataclass
class TaskScenario:
    """
    Complete task scenario definition

    Represents one AVER benchmark task including:
    - Task description and tools
    - Error injection configuration
    - Detection signals
    - Recovery criteria
    - Scoring weights
    - Execution validity (test suites for code tasks)
    - Negative control config (for tasks without errors)
    """
    task_id: str
    category: ErrorCategory
    difficulty: DifficultyLevel
    domain: TaskDomain
    task_description: str
    tools: List[Tool]
    error_injection: ErrorInjection
    detection_signals: DetectionSignals
    recovery_criteria: RecoveryCriteria
    scoring: Scoring
    optimal_turns: int = 3
    expected_output: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    # NEW: Execution-based validity
    execution_validity: Optional[ExecutionValidity] = None
    # NEW: Negative control configuration (for tasks without errors)
    negative_control: Optional[NegativeControlConfig] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization"""
        result = {
            "task_id": self.task_id,
            "category": self.category.value,
            "difficulty": self.difficulty.value,
            "domain": self.domain.value,
            "task_description": self.task_description,
            "tools": [tool.to_dict() for tool in self.tools],
            "error_injection": self.error_injection.to_dict(),
            "detection_signals": self.detection_signals.to_dict(),
            "recovery_criteria": self.recovery_criteria.to_dict(),
            "scoring": self.scoring.to_dict(),
            "optimal_turns": self.optimal_turns,
            "expected_output": self.expected_output,
            "metadata": self.metadata
        }
        # Add optional fields if present
        if self.execution_validity:
            result["execution_validity"] = self.execution_validity.to_dict()
        if self.negative_control:
            result["negative_control"] = self.negative_control.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskScenario":
        """Create TaskScenario from dictionary (YAML load)"""
        # Parse execution validity if present
        exec_validity = None
        if "execution_validity" in data:
            exec_validity = ExecutionValidity.from_dict(data["execution_validity"])

        # Parse negative control config if present
        neg_control = None
        if "negative_control" in data:
            nc_data = data["negative_control"]
            neg_control = NegativeControlConfig(
                false_positive_signals=nc_data.get("false_positive_signals", []),
                should_detect_error=nc_data.get("should_detect_error", False),
                expected_detection_score=nc_data.get("expected_detection_score", 0.0),
                expected_recovery_score=nc_data.get("expected_recovery_score", 1.0)
            )

        # Handle "none" injection point for negative control tasks
        injection_point_str = data["error_injection"]["injection_point"]
        if injection_point_str == "none":
            injection_point = InjectionPoint.TASK_DESCRIPTION  # Default, but error_type will be "none"
        else:
            injection_point = InjectionPoint(injection_point_str)

        return cls(
            task_id=data["task_id"],
            category=ErrorCategory(data["category"]) if data["category"] != "negative_control" else ErrorCategory.HALLUCINATION,
            difficulty=DifficultyLevel(data["difficulty"]),
            domain=TaskDomain(data["domain"]),
            task_description=data["task_description"],
            tools=[Tool(**tool) for tool in data.get("tools", [])],
            error_injection=ErrorInjection(
                injection_point=injection_point,
                injection_turn=data["error_injection"].get("injection_turn", 0),
                error_type=data["error_injection"].get("error_type", ""),
                error_data=data["error_injection"].get("error_data", {}),
                ground_truth=data["error_injection"].get("ground_truth", "")
            ),
            detection_signals=DetectionSignals(
                explicit=data.get("detection_signals", {}).get("explicit", []),
                implicit=data.get("detection_signals", {}).get("implicit", [])
            ),
            recovery_criteria=RecoveryCriteria(
                success=data.get("recovery_criteria", {}).get("success", []),
                partial=data.get("recovery_criteria", {}).get("partial", []),
                failure=data.get("recovery_criteria", {}).get("failure", [])
            ),
            scoring=Scoring(**data.get("scoring", {})),
            optimal_turns=data.get("optimal_turns", 3),
            expected_output=data.get("expected_output", ""),
            metadata=data.get("metadata", {}),
            execution_validity=exec_validity,
            negative_control=neg_control
        )

    def is_negative_control(self) -> bool:
        """Check if this is a negative control task (no error injected)"""
        return self.negative_control is not None or self.error_injection.error_type == "none"

    def has_execution_tests(self) -> bool:
        """Check if this task has execution-based validation"""
        return (self.execution_validity is not None and
                self.execution_validity.enabled and
                len(self.execution_validity.test_suite) > 0)


@dataclass
class AgentTurn:
    """Single turn in agent execution trace"""
    turn_number: int
    reasoning: str = ""
    action: str = ""
    tool: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_output: Optional[str] = None
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "turn_number": self.turn_number,
            "reasoning": self.reasoning,
            "action": self.action,
            "tool": self.tool,
            "tool_input": self.tool_input,
            "tool_output": self.tool_output,
            "timestamp": self.timestamp
        }


@dataclass
class AgentTrace:
    """Complete execution trace of agent"""
    task_id: str
    agent_id: str
    turns: List[AgentTurn] = field(default_factory=list)
    final_output: str = ""
    model_name: str = ""  # Track which model was used
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_turn(self, turn: AgentTurn):
        """Add a turn to the trace"""
        self.turns.append(turn)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "model_name": self.model_name,
            "turns": [turn.to_dict() for turn in self.turns],
            "final_output": self.final_output,
            "metadata": self.metadata
        }


@dataclass
class EvaluationMetrics:
    """
    Evaluation results for one agent on one task

    Tracks detection, diagnosis, and recovery scores plus metadata.
    """
    task_id: str
    agent_id: str
    detection_score: float  # 0.0 to 1.0
    diagnosis_score: float  # 0.0 to 1.0
    recovery_score: float   # 0.0 to 1.0
    total_score: float      # 0.0 to 100.0

    # Detailed breakdown
    detection_details: Dict[str, Any] = field(default_factory=dict)
    diagnosis_details: Dict[str, Any] = field(default_factory=dict)
    recovery_details: Dict[str, Any] = field(default_factory=dict)

    # Task context for consistency analysis
    category: str = ""  # Error category
    difficulty: int = 0  # Difficulty level (1-4)
    is_negative_control: bool = False  # True if this was a negative control task

    # Metadata
    model_name: str = ""  # Track which model was tested
    num_turns: int = 0
    execution_time_seconds: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting"""
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "model_name": self.model_name,
            "category": self.category,
            "difficulty": self.difficulty,
            "is_negative_control": self.is_negative_control,
            "scores": {
                "detection": round(self.detection_score, 3),
                "diagnosis": round(self.diagnosis_score, 3),
                "recovery": round(self.recovery_score, 3),
                "total": round(self.total_score, 2)
            },
            "details": {
                "detection": self.detection_details,
                "diagnosis": self.diagnosis_details,
                "recovery": self.recovery_details
            },
            "metadata": {
                "num_turns": self.num_turns,
                "execution_time_seconds": self.execution_time_seconds,
                "timestamp": self.timestamp
            }
        }

    def summary(self) -> str:
        """Get human-readable summary"""
        return (
            f"Task: {self.task_id}\n"
            f"Agent: {self.agent_id}\n"
            f"Model: {self.model_name}\n"
            f"Detection: {self.detection_score:.2f} ({self.detection_score*100:.0f}%)\n"
            f"Diagnosis: {self.diagnosis_score:.2f} ({self.diagnosis_score*100:.0f}%)\n"
            f"Recovery: {self.recovery_score:.2f} ({self.recovery_score*100:.0f}%)\n"
            f"Total Score: {self.total_score:.1f}/100\n"
            f"Turns: {self.num_turns}"
        )


@dataclass
class BenchmarkResults:
    """Aggregated results across multiple tasks/agents"""
    agent_id: str
    task_results: List[EvaluationMetrics] = field(default_factory=list)

    def add_result(self, result: EvaluationMetrics):
        """Add a task result"""
        self.task_results.append(result)

    def aggregate_scores(self) -> Dict[str, float]:
        """Calculate aggregate scores across all tasks"""
        if not self.task_results:
            return {
                "avg_detection": 0.0,
                "avg_diagnosis": 0.0,
                "avg_recovery": 0.0,
                "avg_total": 0.0
            }

        return {
            "avg_detection": sum(r.detection_score for r in self.task_results) / len(self.task_results),
            "avg_diagnosis": sum(r.diagnosis_score for r in self.task_results) / len(self.task_results),
            "avg_recovery": sum(r.recovery_score for r in self.task_results) / len(self.task_results),
            "avg_total": sum(r.total_score for r in self.task_results) / len(self.task_results)
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "agent_id": self.agent_id,
            "num_tasks": len(self.task_results),
            "aggregate_scores": self.aggregate_scores(),
            "task_results": [r.to_dict() for r in self.task_results]
        }
