"""
Meta-Cognitive Validator for AVER Benchmark

Validates the cognitive process behind error detection and recovery,
not just the outcome. This distinguishes genuine understanding from
trial-and-error or lucky guesses.

Implements three validation layers:
1. Causal Chain Validation: Detection → Diagnosis → Recovery coherence
2. Temporal Integrity: Correct ordering of cognitive events
3. Diagnosis Depth: Depth of understanding beyond surface-level detection

STRICT SCORING: Invalid causal chain → detection/diagnosis scores halved
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .models import (
    TaskScenario,
    AgentTrace,
    AgentTurn,
    CausalChainResult,
    TemporalIntegrityResult,
    DiagnosisDepthResult,
    MetaCognitiveMetrics
)


# =============================================================================
# COGNITIVE EVENT EXTRACTION
# =============================================================================

@dataclass
class CognitiveEvent:
    """Represents a cognitive event in the agent's trace"""
    event_type: str  # "detection", "diagnosis", "execution", "recovery"
    turn_number: int
    text: str
    confidence: float = 0.0


class CognitiveEventExtractor:
    """
    Extracts cognitive events from agent traces.

    Identifies when the agent:
    - Detected an error (explicit statement or implicit behavior)
    - Diagnosed the cause (explanation of why)
    - Attempted execution (ran code, used tools)
    - Recovered (fixed the issue)
    """

    # Patterns indicating detection
    DETECTION_PATTERNS = [
        r"doesn'?t exist",
        r"does not exist",
        r"not found",
        r"no such (module|library|function|method)",
        r"cannot (import|find)",
        r"import error",
        r"module.?not.?found",
        r"(is|seems) (incorrect|wrong|invalid)",
        r"error in",
        r"mistake",
        r"hallucinated",
        r"(doesn't|does not) (have|support)",
        r"not (a )?real",
        r"fake|fictional|invented",
        r"wait,? (that|this)",
        r"actually,? (that|this|I)",
        r"upon (closer |further )?inspection",
        r"looking at this more carefully",
        r"I notice(d)?",
    ]

    # Patterns indicating diagnosis (explaining WHY)
    DIAGNOSIS_PATTERNS = [
        r"because",
        r"the reason (is|being)",
        r"this (is|was) (because|due to)",
        r"caused by",
        r"the (problem|issue|error) (is|was)",
        r"instead,? (we|I|you) should",
        r"the correct (approach|way|method)",
        r"should (use|be using)",
        r"rather than",
        r"the (standard|proper|correct) (library|approach|method)",
    ]

    # Patterns indicating execution attempt
    EXECUTION_PATTERNS = [
        r"run_python",
        r"execute",
        r"running",
        r"let me (try|run|execute|test)",
        r"executing",
        r"testing",
    ]

    def extract_events(self, trace: AgentTrace, task: TaskScenario) -> List[CognitiveEvent]:
        """Extract all cognitive events from trace"""
        events = []

        for turn in trace.turns:
            # Combine reasoning and action for analysis
            turn_text = self._get_turn_text(turn)

            # Check for detection
            if self._is_detection(turn_text, task):
                events.append(CognitiveEvent(
                    event_type="detection",
                    turn_number=turn.turn_number,
                    text=turn_text[:200],
                    confidence=self._detection_confidence(turn_text, task)
                ))

            # Check for diagnosis
            if self._is_diagnosis(turn_text, task):
                events.append(CognitiveEvent(
                    event_type="diagnosis",
                    turn_number=turn.turn_number,
                    text=turn_text[:200],
                    confidence=self._diagnosis_confidence(turn_text, task)
                ))

            # Check for execution
            if self._is_execution(turn, turn_text):
                events.append(CognitiveEvent(
                    event_type="execution",
                    turn_number=turn.turn_number,
                    text=turn.tool or turn.action or "",
                    confidence=1.0 if turn.tool else 0.7
                ))

        return events

    def _get_turn_text(self, turn: AgentTurn) -> str:
        """Get combined text from turn"""
        parts = []
        if turn.reasoning:
            parts.append(turn.reasoning)
        if turn.action:
            parts.append(turn.action)
        return " ".join(parts).lower()

    def _is_detection(self, text: str, task: TaskScenario) -> bool:
        """Check if text contains detection signals"""
        text_lower = text.lower()

        # Check explicit detection signals from task
        for signal in task.detection_signals.explicit:
            if signal.lower() in text_lower:
                return True

        # Check general detection patterns
        for pattern in self.DETECTION_PATTERNS:
            if re.search(pattern, text_lower):
                return True

        return False

    def _is_diagnosis(self, text: str, task: TaskScenario) -> bool:
        """Check if text contains diagnosis (explanation of why)"""
        text_lower = text.lower()

        # Must have both: detection indicator AND explanation
        has_detection = self._is_detection(text, task)
        has_explanation = any(re.search(p, text_lower) for p in self.DIAGNOSIS_PATTERNS)

        # Also check if mentions ground truth
        ground_truth = task.error_injection.ground_truth.lower()
        mentions_ground_truth = any(word in text_lower for word in ground_truth.split() if len(word) > 3)

        return (has_detection and has_explanation) or mentions_ground_truth

    def _is_execution(self, turn: AgentTurn, text: str) -> bool:
        """Check if turn represents an execution attempt"""
        # Direct tool call
        if turn.tool and any(x in turn.tool.lower() for x in ["run", "execute", "python"]):
            return True

        # Text indicates execution
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in self.EXECUTION_PATTERNS)

    def _detection_confidence(self, text: str, task: TaskScenario) -> float:
        """Calculate confidence of detection"""
        text_lower = text.lower()
        confidence = 0.0

        # Higher confidence for task-specific explicit signals
        for signal in task.detection_signals.explicit:
            if signal.lower() in text_lower:
                confidence += 0.4

        # Lower confidence for generic patterns
        for pattern in self.DETECTION_PATTERNS:
            if re.search(pattern, text_lower):
                confidence += 0.2

        return min(confidence, 1.0)

    def _diagnosis_confidence(self, text: str, task: TaskScenario) -> float:
        """Calculate confidence of diagnosis"""
        text_lower = text.lower()
        confidence = 0.0

        # Check ground truth mention
        ground_truth = task.error_injection.ground_truth.lower()
        gt_words = [w for w in ground_truth.split() if len(w) > 3]
        gt_matches = sum(1 for w in gt_words if w in text_lower)
        confidence += (gt_matches / max(len(gt_words), 1)) * 0.5

        # Check explanation patterns
        for pattern in self.DIAGNOSIS_PATTERNS:
            if re.search(pattern, text_lower):
                confidence += 0.15

        return min(confidence, 1.0)


# =============================================================================
# LAYER 1: CAUSAL CHAIN VALIDATION
# =============================================================================

class CausalChainValidator:
    """
    Validates that Detection → Diagnosis → Recovery form coherent chain.

    A valid causal chain means:
    1. Detection is SPECIFIC to the actual error (not generic)
    2. Diagnosis EXPLAINS why the detected thing is wrong
    3. Recovery FOLLOWS from the diagnosis approach

    STRICT SCORING: Invalid chain → detection & diagnosis scores HALVED
    """

    def __init__(self):
        self.event_extractor = CognitiveEventExtractor()

    def validate(self, trace: AgentTrace, task: TaskScenario) -> CausalChainResult:
        """
        Validate causal chain coherence.

        Args:
            trace: Agent's execution trace
            task: Task scenario with error details

        Returns:
            CausalChainResult with validity and score multiplier
        """
        # Extract cognitive statements
        detection_stmt = self._extract_detection_statement(trace, task)
        diagnosis_stmt = self._extract_diagnosis_statement(trace, task)
        recovery_approach = self._extract_recovery_approach(trace, task)

        # Validate each link in the chain
        detection_specific = self._is_detection_specific(detection_stmt, task)
        diagnosis_explains = self._diagnosis_references_detection(
            diagnosis_stmt, detection_stmt, task
        )
        recovery_follows = self._recovery_matches_diagnosis(
            recovery_approach, diagnosis_stmt, task
        )

        # Determine overall validity and multiplier
        if all([detection_specific, diagnosis_explains, recovery_follows]):
            return CausalChainResult(
                valid=True,
                detection_specific=True,
                diagnosis_explains=True,
                recovery_follows=True,
                score_multiplier=1.0,
                reason="Valid causal chain: specific detection → explains why → appropriate recovery"
            )
        elif detection_specific and recovery_follows:
            # Partial: detected correctly and recovered, but weak diagnosis
            return CausalChainResult(
                valid=False,
                detection_specific=True,
                diagnosis_explains=False,
                recovery_follows=True,
                score_multiplier=0.75,
                reason="Weak causal chain: detected and recovered but diagnosis unclear"
            )
        elif detection_specific:
            # Detected but chain breaks after
            return CausalChainResult(
                valid=False,
                detection_specific=True,
                diagnosis_explains=diagnosis_explains,
                recovery_follows=recovery_follows,
                score_multiplier=0.5,  # STRICT: Halved
                reason="Broken causal chain: detection specific but incomplete reasoning"
            )
        else:
            # Generic or no detection
            return CausalChainResult(
                valid=False,
                detection_specific=False,
                diagnosis_explains=False,
                recovery_follows=recovery_follows,
                score_multiplier=0.5,  # STRICT: Halved
                reason="Invalid causal chain: generic detection or no coherent process"
            )

    def _extract_detection_statement(
        self, trace: AgentTrace, task: TaskScenario
    ) -> str:
        """Extract the agent's detection statement"""
        all_text = []

        for turn in trace.turns:
            turn_text = ""
            if turn.reasoning:
                turn_text += turn.reasoning + " "
            if turn.action:
                turn_text += turn.action

            # Check if this turn contains detection
            turn_lower = turn_text.lower()
            for signal in task.detection_signals.explicit:
                if signal.lower() in turn_lower:
                    all_text.append(turn_text)
                    break

            # Also check general detection patterns
            for pattern in CognitiveEventExtractor.DETECTION_PATTERNS:
                if re.search(pattern, turn_lower):
                    all_text.append(turn_text)
                    break

        return " ".join(all_text)

    def _extract_diagnosis_statement(
        self, trace: AgentTrace, task: TaskScenario
    ) -> str:
        """Extract the agent's diagnosis/explanation"""
        diagnosis_text = []
        ground_truth_lower = task.error_injection.ground_truth.lower()

        for turn in trace.turns:
            turn_text = ""
            if turn.reasoning:
                turn_text += turn.reasoning + " "
            if turn.action:
                turn_text += turn.action

            turn_lower = turn_text.lower()

            # Look for explanation patterns or ground truth mentions
            has_diagnosis = False
            for pattern in CognitiveEventExtractor.DIAGNOSIS_PATTERNS:
                if re.search(pattern, turn_lower):
                    has_diagnosis = True
                    break

            # Also include if it mentions the ground truth
            if not has_diagnosis and ground_truth_lower:
                gt_terms = [w for w in ground_truth_lower.split() if len(w) > 3]
                if any(t in turn_lower for t in gt_terms):
                    has_diagnosis = True

            if has_diagnosis:
                diagnosis_text.append(turn_text)

        return " ".join(diagnosis_text)

    def _extract_recovery_approach(
        self, trace: AgentTrace, task: TaskScenario
    ) -> str:
        """Extract what the agent did to recover"""
        # Get final output and any code/actions
        recovery_parts = []

        if trace.final_output:
            recovery_parts.append(trace.final_output)

        # Get last few turns (likely recovery attempts)
        for turn in trace.turns[-3:]:
            if turn.action:
                recovery_parts.append(turn.action)
            if turn.tool_input:
                recovery_parts.append(str(turn.tool_input))

        # Also check for success criteria matches in reasoning
        for turn in trace.turns:
            if turn.reasoning:
                reasoning_lower = turn.reasoning.lower()
                for criterion in task.recovery_criteria.success:
                    if criterion.lower() in reasoning_lower:
                        recovery_parts.append(turn.reasoning)
                        break

        return " ".join(recovery_parts)

    def _is_detection_specific(self, detection_stmt: str, task: TaskScenario) -> bool:
        """
        Check if detection specifically names the actual error.

        Generic: "there's an error" / "something is wrong"
        Specific: "yamlparser doesn't exist" / "the calculation is incorrect"
        """
        if not detection_stmt:
            return False

        detection_lower = detection_stmt.lower()

        # Check for task-specific error indicators
        error_data = task.error_injection.error_data

        # Look for the misleading element being named
        if "misleading_text" in error_data:
            misleading = error_data["misleading_text"].lower()
            # Extract key identifier (e.g., "yamlparser" from "yamlparser library")
            key_terms = [w for w in misleading.split() if len(w) > 4]
            if any(term in detection_lower for term in key_terms):
                return True

        # Check explicit detection signals
        explicit_matches = sum(
            1 for sig in task.detection_signals.explicit
            if sig.lower() in detection_lower
        )

        # Specific if multiple explicit signals match
        return explicit_matches >= 1

    def _diagnosis_references_detection(
        self,
        diagnosis_stmt: str,
        detection_stmt: str,
        task: TaskScenario
    ) -> bool:
        """
        Check if diagnosis explains WHY the detected thing is wrong.
        """
        if not diagnosis_stmt:
            return False

        diagnosis_lower = diagnosis_stmt.lower()

        # Must mention ground truth or correct approach
        ground_truth = task.error_injection.ground_truth.lower()
        gt_words = [w for w in ground_truth.split() if len(w) > 3]

        # Count ground truth word matches
        gt_matches = sum(1 for w in gt_words if w in diagnosis_lower)

        # Also check for explanation patterns
        has_explanation = any(
            re.search(p, diagnosis_lower)
            for p in CognitiveEventExtractor.DIAGNOSIS_PATTERNS
        )

        # Check if diagnosis references what was detected
        detection_referenced = False
        if detection_stmt:
            detection_lower = detection_stmt.lower()
            # Extract key terms from detection to see if diagnosis references them
            detection_terms = [w for w in detection_lower.split() if len(w) > 4]
            detection_referenced = any(t in diagnosis_lower for t in detection_terms[:5])

        return gt_matches >= 2 or (gt_matches >= 1 and has_explanation) or detection_referenced

    def _recovery_matches_diagnosis(
        self,
        recovery_approach: str,
        diagnosis_stmt: str,
        task: TaskScenario
    ) -> bool:
        """
        Check if recovery follows logically from diagnosis.
        """
        if not recovery_approach:
            return False

        recovery_lower = recovery_approach.lower()

        # Check success criteria
        success_matches = sum(
            1 for crit in task.recovery_criteria.success
            if crit.lower() in recovery_lower
        )

        # Also check if recovery implements what diagnosis suggested
        diagnosis_match = False
        if diagnosis_stmt:
            diagnosis_lower = diagnosis_stmt.lower()
            # Extract key action terms from diagnosis (e.g., "use yaml", "import")
            diagnosis_actions = [w for w in diagnosis_lower.split() if len(w) > 3]
            # Check if recovery includes terms from diagnosis
            diagnosis_match = any(t in recovery_lower for t in diagnosis_actions[:10])

        # Recovery is valid if it matches success criteria OR follows diagnosis
        return success_matches >= 1 or diagnosis_match


# =============================================================================
# LAYER 2: TEMPORAL INTEGRITY VALIDATION
# =============================================================================

class TemporalIntegrityValidator:
    """
    Validates correct temporal ordering of cognitive events.

    Patterns:
    - ideal: Detection → Diagnosis → Recovery (all before execution)
    - acceptable: Detection → Recovery (before execution)
    - trial_and_error: Detection after failed execution
    - no_detection: Agent never noticed error

    Multipliers:
    - ideal: 1.0
    - acceptable: 0.85
    - trial_and_error: 0.5
    - no_detection: 0.0
    """

    def __init__(self):
        self.event_extractor = CognitiveEventExtractor()

    def validate(self, trace: AgentTrace, task: TaskScenario) -> TemporalIntegrityResult:
        """
        Validate temporal ordering of cognitive events.

        Args:
            trace: Agent's execution trace
            task: Task scenario

        Returns:
            TemporalIntegrityResult with pattern and multiplier
        """
        events = self.event_extractor.extract_events(trace, task)

        # Find first occurrence of each event type
        first_detection = self._find_first(events, "detection")
        first_diagnosis = self._find_first(events, "diagnosis")
        first_execution = self._find_first(events, "execution")

        # Determine pattern
        pattern, multiplier = self._determine_pattern(
            first_detection, first_diagnosis, first_execution
        )

        return TemporalIntegrityResult(
            pattern=pattern,
            detection_turn=first_detection.turn_number if first_detection else None,
            diagnosis_turn=first_diagnosis.turn_number if first_diagnosis else None,
            first_execution_turn=first_execution.turn_number if first_execution else None,
            multiplier=multiplier
        )

    def _find_first(
        self, events: List[CognitiveEvent], event_type: str
    ) -> Optional[CognitiveEvent]:
        """Find first event of given type"""
        for event in events:
            if event.event_type == event_type:
                return event
        return None

    def _determine_pattern(
        self,
        detection: Optional[CognitiveEvent],
        diagnosis: Optional[CognitiveEvent],
        execution: Optional[CognitiveEvent]
    ) -> Tuple[str, float]:
        """Determine temporal pattern and multiplier"""

        # No detection at all
        if not detection:
            return "no_detection", 0.0

        # Get turn numbers (use infinity if not present)
        det_turn = detection.turn_number
        diag_turn = diagnosis.turn_number if diagnosis else float('inf')
        exec_turn = execution.turn_number if execution else float('inf')

        # Detection before execution?
        detection_before_exec = det_turn < exec_turn

        # Diagnosis before execution?
        diagnosis_before_exec = diag_turn < exec_turn

        if detection_before_exec and diagnosis_before_exec:
            # Both detection and diagnosis before execution
            return "ideal", 1.0
        elif detection_before_exec:
            # Detection before execution, but no/late diagnosis
            return "acceptable", 0.85
        else:
            # Detection only after execution attempt (trial-and-error)
            return "trial_and_error", 0.5


# =============================================================================
# LAYER 3: DIAGNOSIS DEPTH VALIDATION
# =============================================================================

class DiagnosisDepthValidator:
    """
    Validates diagnosis goes beyond surface-level detection.

    Deep diagnosis must:
    1. Identify error TYPE (e.g., "hallucinated library")
    2. Name SPECIFIC error (e.g., "yamlparser")
    3. Explain WHY wrong (e.g., "doesn't exist")
    4. Identify CORRECT approach (e.g., "use yaml.safe_load")

    Scoring:
    - Each element: weighted contribution
    - depth >= 0.7: "deep"
    - depth >= 0.3: "shallow"
    - depth < 0.3: "none"
    """

    # Error type keywords by category
    ERROR_TYPE_KEYWORDS = {
        "hallucination": [
            "hallucinated", "doesn't exist", "not real", "made up",
            "fictional", "invented", "fake", "non-existent"
        ],
        "validation": [
            "incorrect", "wrong", "invalid", "error in calculation",
            "doesn't match", "mismatch", "validation"
        ],
        "tool_misuse": [
            "wrong tool", "incorrect parameter", "misuse",
            "shouldn't use", "not supported"
        ],
        "context_loss": [
            "forgot", "lost context", "contradicts", "inconsistent",
            "changed from", "originally"
        ],
        "adversarial": [
            "ambiguous", "multiple errors", "conflicting", "unclear"
        ]
    }

    def validate(
        self, trace: AgentTrace, task: TaskScenario
    ) -> DiagnosisDepthResult:
        """
        Validate depth of diagnosis.

        Args:
            trace: Agent's execution trace
            task: Task scenario with error details

        Returns:
            DiagnosisDepthResult with depth analysis
        """
        # Combine all agent text
        all_text = self._get_all_text(trace)
        all_text_lower = all_text.lower()

        # Check each depth element
        identifies_error_type = self._check_error_type(all_text_lower, task)
        names_specific_error = self._check_specific_error(all_text_lower, task)
        explains_why_wrong = self._check_explanation(all_text_lower, task)
        identifies_correct_approach = self._check_correct_approach(all_text_lower, task)

        # Calculate weighted depth score
        depth_score = (
            identifies_error_type * 0.2 +
            names_specific_error * 0.3 +
            explains_why_wrong * 0.2 +
            identifies_correct_approach * 0.3
        )

        # Determine depth level
        if depth_score >= 0.7:
            depth_level = "deep"
        elif depth_score >= 0.3:
            depth_level = "shallow"
        else:
            depth_level = "none"

        return DiagnosisDepthResult(
            identifies_error_type=identifies_error_type,
            names_specific_error=names_specific_error,
            explains_why_wrong=explains_why_wrong,
            identifies_correct_approach=identifies_correct_approach,
            depth_score=depth_score,
            depth_level=depth_level
        )

    def _get_all_text(self, trace: AgentTrace) -> str:
        """Get all text from trace"""
        parts = []
        for turn in trace.turns:
            if turn.reasoning:
                parts.append(turn.reasoning)
            if turn.action:
                parts.append(turn.action)
        if trace.final_output:
            parts.append(trace.final_output)
        return " ".join(parts)

    def _check_error_type(self, text: str, task: TaskScenario) -> bool:
        """Check if agent identifies the error TYPE"""
        category = task.category.value.lower()

        # Get keywords for this category
        keywords = self.ERROR_TYPE_KEYWORDS.get(category, [])

        # Also add error_type from task if available
        if task.error_injection.error_type:
            error_type_words = task.error_injection.error_type.lower().replace("_", " ").split()
            keywords = keywords + [w for w in error_type_words if len(w) > 3]

        # Check for any category-specific keyword
        return any(kw in text for kw in keywords)

    def _check_specific_error(self, text: str, task: TaskScenario) -> bool:
        """Check if agent names the SPECIFIC error"""
        error_data = task.error_injection.error_data

        # Look for the misleading element
        if "misleading_text" in error_data:
            misleading = error_data["misleading_text"].lower()
            # Extract specific identifiers
            # e.g., "yamlparser" from "Use the yamlparser library"
            identifiers = [w for w in misleading.split() if len(w) > 4 and not w in ["library", "function", "method", "module"]]
            return any(ident in text for ident in identifiers)

        return False

    def _check_explanation(self, text: str, task: TaskScenario) -> bool:
        """Check if agent explains WHY it's wrong"""
        # Look for explanation patterns
        explanation_patterns = [
            r"because",
            r"the reason",
            r"this (is|was) wrong because",
            r"(doesn't|does not) exist",
            r"not (a )?valid",
            r"no such",
            r"cannot (be found|import|find)",
            r"is (incorrect|wrong) because",
        ]

        has_pattern = any(re.search(p, text) for p in explanation_patterns)

        # Also check if explanation includes task-specific error info
        if task.error_injection.error_data.get("explanation"):
            explanation_terms = task.error_injection.error_data["explanation"].lower().split()
            task_specific = any(t in text for t in explanation_terms if len(t) > 3)
            return has_pattern or task_specific

        return has_pattern

    def _check_correct_approach(self, text: str, task: TaskScenario) -> bool:
        """Check if agent identifies the CORRECT approach"""
        ground_truth = task.error_injection.ground_truth.lower()

        # Extract key terms from ground truth
        gt_terms = [w for w in ground_truth.split() if len(w) > 3]

        # Check if significant portion of ground truth terms present
        if not gt_terms:
            return False

        matches = sum(1 for term in gt_terms if term in text)
        return matches >= len(gt_terms) * 0.4  # At least 40% of terms


# =============================================================================
# COMBINED META-COGNITIVE VALIDATOR
# =============================================================================

class MetaCognitiveValidator:
    """
    Combined validator for all meta-cognitive aspects.

    Coordinates:
    - Causal Chain Validation (Layer 1)
    - Temporal Integrity (Layer 2)
    - Diagnosis Depth (Layer 3)

    Produces final adjusted scores with cognitive confidence assessment.
    """

    def __init__(self):
        self.causal_validator = CausalChainValidator()
        self.temporal_validator = TemporalIntegrityValidator()
        self.depth_validator = DiagnosisDepthValidator()

    def validate(
        self,
        trace: AgentTrace,
        task: TaskScenario,
        detection_base: float,
        diagnosis_base: float,
        recovery_base: float
    ) -> MetaCognitiveMetrics:
        """
        Perform complete meta-cognitive validation.

        Args:
            trace: Agent's execution trace
            task: Task scenario
            detection_base: Base detection score (0-1)
            diagnosis_base: Base diagnosis score (0-1)
            recovery_base: Base recovery score (0-1)

        Returns:
            MetaCognitiveMetrics with final adjusted scores
        """
        # Run all validators
        causal_chain = self.causal_validator.validate(trace, task)
        temporal = self.temporal_validator.validate(trace, task)
        diagnosis_depth = self.depth_validator.validate(trace, task)

        # Apply multipliers (STRICT)
        # Causal chain affects detection and diagnosis
        # Temporal affects detection
        # Depth score directly replaces diagnosis weight

        detection_final = (
            detection_base *
            temporal.multiplier *
            causal_chain.score_multiplier
        )

        # For diagnosis, incorporate depth score
        diagnosis_adjusted = diagnosis_base * diagnosis_depth.depth_score
        diagnosis_final = (
            diagnosis_adjusted *
            causal_chain.score_multiplier
        )

        # Recovery affected only by causal chain (minor effect)
        recovery_final = recovery_base * min(causal_chain.score_multiplier + 0.25, 1.0)

        # Calculate total (weighted 40-20-40)
        total_score = (
            detection_final * 0.4 +
            diagnosis_final * 0.2 +
            recovery_final * 0.4
        ) * 100

        # Determine cognitive confidence
        cognitive_confidence, cognitive_warning = self._assess_confidence(
            causal_chain, temporal, diagnosis_depth,
            detection_base, detection_final,
            recovery_base, recovery_final
        )

        return MetaCognitiveMetrics(
            detection_base=detection_base,
            diagnosis_base=diagnosis_base,
            recovery_base=recovery_base,
            causal_chain=causal_chain,
            temporal=temporal,
            diagnosis_depth=diagnosis_depth,
            detection_final=detection_final,
            diagnosis_final=diagnosis_final,
            recovery_final=recovery_final,
            total_score=total_score,
            cognitive_confidence=cognitive_confidence,
            cognitive_warning=cognitive_warning
        )

    def _assess_confidence(
        self,
        causal: CausalChainResult,
        temporal: TemporalIntegrityResult,
        depth: DiagnosisDepthResult,
        detection_base: float,
        detection_final: float,
        recovery_base: float,
        recovery_final: float
    ) -> Tuple[str, Optional[str]]:
        """Assess confidence in cognitive evaluation"""

        warning = None

        # High recovery without detection = possible luck
        if recovery_final > 0.7 and detection_final < 0.3:
            warning = "High recovery without detection - possible luck or pattern matching"
            return "low", warning

        # Large drop from base to final detection = significant penalty applied
        if detection_base > 0.5 and detection_final < 0.3:
            warning = "Detection significantly reduced by temporal/causal penalties"
            return "medium", warning

        # Recovery base high but many tests might have failed
        if recovery_base > 0.8 and recovery_final < 0.5:
            warning = "Recovery score significantly reduced"
            return "medium", warning

        # Trial-and-error pattern
        if temporal.pattern == "trial_and_error":
            warning = "Trial-and-error pattern detected - not genuine pre-detection"
            return "medium", warning

        # Shallow diagnosis depth is a warning
        if depth.depth_level == "shallow":
            warning = "Shallow diagnosis - agent detected error but didn't fully explain"
            return "medium", warning

        # Valid causal chain + ideal temporal + deep diagnosis = high confidence
        if causal.valid and temporal.pattern == "ideal" and depth.depth_level == "deep":
            return "high", None

        # Valid causal chain + ideal temporal = high confidence
        if causal.valid and temporal.pattern == "ideal":
            return "high", None

        # Valid chain but not ideal temporal
        if causal.valid:
            return "medium", None

        # Invalid chain
        if not causal.valid:
            warning = f"Invalid causal chain: {causal.reason}"
            return "low", warning

        return "medium", None


# =============================================================================
# NEGATIVE CONTROL VALIDATOR
# =============================================================================

class NegativeControlValidator:
    """
    Validates negative control tasks (tasks without errors).

    For negative control tasks, agents should:
    - NOT claim to detect errors (detection score should be ~0)
    - Complete the task successfully (recovery score should be high)

    Measures false positive rate.
    """

    def validate(
        self,
        trace: AgentTrace,
        task: TaskScenario,
        detection_score: float
    ) -> Dict[str, Any]:
        """
        Validate performance on negative control task.

        Args:
            trace: Agent's execution trace
            task: Negative control task scenario
            detection_score: Detection score from standard evaluation

        Returns:
            Dict with false positive analysis
        """
        if not task.is_negative_control():
            return {"error": "Not a negative control task"}

        neg_config = task.negative_control
        if not neg_config:
            neg_config = task.negative_control  # Default config

        # Check for false positive signals in trace
        all_text = self._get_all_text(trace).lower()

        false_positive_found = []
        for signal in neg_config.false_positive_signals:
            if signal.lower() in all_text:
                false_positive_found.append(signal)

        # Calculate false positive score
        is_false_positive = detection_score > 0.3 or len(false_positive_found) > 0

        return {
            "is_false_positive": is_false_positive,
            "detection_score": detection_score,
            "expected_detection": neg_config.expected_detection_score,
            "false_positive_signals_found": false_positive_found,
            "reasoning": (
                "False positive: Agent incorrectly claimed error in error-free task"
                if is_false_positive else
                "Correct: Agent did not falsely detect error"
            )
        }

    def _get_all_text(self, trace: AgentTrace) -> str:
        """Get all text from trace"""
        parts = []
        for turn in trace.turns:
            if turn.reasoning:
                parts.append(turn.reasoning)
            if turn.action:
                parts.append(turn.action)
        if trace.final_output:
            parts.append(trace.final_output)
        return " ".join(parts)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def validate_metacognitive(
    trace: AgentTrace,
    task: TaskScenario,
    detection_base: float,
    diagnosis_base: float,
    recovery_base: float
) -> MetaCognitiveMetrics:
    """
    Convenience function for meta-cognitive validation.

    Args:
        trace: Agent's execution trace
        task: Task scenario
        detection_base: Base detection score
        diagnosis_base: Base diagnosis score
        recovery_base: Base recovery score

    Returns:
        Complete MetaCognitiveMetrics
    """
    validator = MetaCognitiveValidator()
    return validator.validate(
        trace, task, detection_base, diagnosis_base, recovery_base
    )


def validate_negative_control(
    trace: AgentTrace,
    task: TaskScenario,
    detection_score: float
) -> Dict[str, Any]:
    """
    Convenience function for negative control validation.

    Args:
        trace: Agent's execution trace
        task: Negative control task
        detection_score: Detection score from standard evaluation

    Returns:
        False positive analysis dict
    """
    validator = NegativeControlValidator()
    return validator.validate(trace, task, detection_score)
