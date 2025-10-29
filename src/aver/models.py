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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization"""
        return {
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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskScenario":
        """Create TaskScenario from dictionary (YAML load)"""
        return cls(
            task_id=data["task_id"],
            category=ErrorCategory(data["category"]),
            difficulty=DifficultyLevel(data["difficulty"]),
            domain=TaskDomain(data["domain"]),
            task_description=data["task_description"],
            tools=[Tool(**tool) for tool in data["tools"]],
            error_injection=ErrorInjection(
                injection_point=InjectionPoint(data["error_injection"]["injection_point"]),
                injection_turn=data["error_injection"].get("injection_turn", 0),
                error_type=data["error_injection"].get("error_type", ""),
                error_data=data["error_injection"].get("error_data", {}),
                ground_truth=data["error_injection"].get("ground_truth", "")
            ),
            detection_signals=DetectionSignals(
                explicit=data["detection_signals"].get("explicit", []),
                implicit=data["detection_signals"].get("implicit", [])
            ),
            recovery_criteria=RecoveryCriteria(
                success=data["recovery_criteria"].get("success", []),
                partial=data["recovery_criteria"].get("partial", []),
                failure=data["recovery_criteria"].get("failure", [])
            ),
            scoring=Scoring(**data.get("scoring", {})),
            optimal_turns=data.get("optimal_turns", 3),
            expected_output=data.get("expected_output", ""),
            metadata=data.get("metadata", {})
        )


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
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_turn(self, turn: AgentTurn):
        """Add a turn to the trace"""
        self.turns.append(turn)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
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

    # Metadata
    num_turns: int = 0
    execution_time_seconds: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting"""
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
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
