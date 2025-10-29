"""
LLM-as-Judge Module

Implements multi-judge consensus for diagnosis scoring.
Based on MAST multi-annotator approach with κ=0.88.
"""

import asyncio
import httpx
from typing import List, Dict, Any
from statistics import median, mean, stdev
import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Judge ensemble (from user specification + literature recommendations)
JUDGE_MODELS = [
    "anthropic/claude-sonnet-4.5",  # Best reasoning
    "openai/gpt-5",                  # Industry standard
    "google/gemini-2.5-pro"          # Diverse perspective
]


class LLMJudge:
    """
    LLM-as-Judge evaluator with multi-judge consensus

    Implements robust diagnosis scoring using 3 judges with voting.
    """

    def __init__(
        self,
        judge_models: List[str] = None,
        api_key: str = None
    ):
        """
        Initialize LLM judge

        Args:
            judge_models: List of models to use as judges
            api_key: OpenRouter API key
        """
        self.judge_models = judge_models or JUDGE_MODELS
        self.api_key = api_key or OPENROUTER_API_KEY

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY required for LLM judge")

    async def judge_diagnosis(
        self,
        trace_reasoning: str,
        error_type: str,
        ground_truth: str
    ) -> Dict[str, Any]:
        """
        Judge if agent correctly diagnosed error using multi-judge consensus

        Args:
            trace_reasoning: Agent's reasoning from trace
            error_type: Type of error
            ground_truth: Correct diagnosis

        Returns:
            Dict with score, individual judgments, agreement
        """
        prompt = self._build_diagnosis_prompt(
            trace_reasoning,
            error_type,
            ground_truth
        )

        # Query all judges
        judgments = []
        for judge_model in self.judge_models:
            try:
                score = await self._query_judge(judge_model, prompt)
                judgments.append({
                    "model": judge_model,
                    "score": score
                })
            except Exception as e:
                print(f"[LLM Judge] Warning: {judge_model} failed: {e}")
                # Continue with other judges

        if not judgments:
            return {
                "score": 0.0,
                "method": "llm_judge_failed",
                "judgments": [],
                "agreement": 0.0
            }

        # Calculate consensus
        scores = [j["score"] for j in judgments]
        consensus_score = median(scores)

        # Calculate inter-judge agreement
        agreement = self._calculate_agreement(scores)

        return {
            "score": consensus_score,
            "method": "llm_judge_ensemble",
            "num_judges": len(judgments),
            "judgments": judgments,
            "agreement": agreement,
            "variance": stdev(scores) if len(scores) > 1 else 0.0
        }

    def _build_diagnosis_prompt(
        self,
        trace_reasoning: str,
        error_type: str,
        ground_truth: str
    ) -> str:
        """Build prompt for diagnosis evaluation"""

        prompt = f"""You are an expert evaluator assessing whether an AI agent correctly diagnosed an error.

ERROR INFORMATION:
Error Type: {error_type}
Correct Diagnosis (Ground Truth): {ground_truth}

AGENT'S REASONING:
{trace_reasoning}

EVALUATION TASK:
Did the agent correctly identify WHY the error occurred?

SCORING RUBRIC:
- Score 1.0: Agent identified the error type AND explained the root cause correctly
- Score 0.5: Agent identified the error type OR partially explained the cause
- Score 0.0: Agent did not identify the error or gave incorrect diagnosis

IMPORTANT:
- Look for understanding of WHY, not just WHAT
- Agent must mention the ground truth concept (or very close equivalent)
- Partial credit for incomplete but directionally correct diagnosis

OUTPUT FORMAT:
Provide ONLY a single number: 0.0, 0.5, or 1.0

Your score:"""

        return prompt

    async def _query_judge(self, model: str, prompt: str) -> float:
        """
        Query a single judge model

        Args:
            model: Judge model name
            prompt: Evaluation prompt

        Returns:
            Score (0.0, 0.5, or 1.0)
        """
        url = "https://openrouter.ai/api/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 10,  # Just need a number
            "temperature": 0.0  # Deterministic
        }

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()

            result = response.json()
            response_text = result["choices"][0]["message"]["content"].strip()

            # Parse score
            score = self._parse_score(response_text)
            return score

    def _parse_score(self, response: str) -> float:
        """Parse score from judge response"""
        # Extract first number found
        import re
        numbers = re.findall(r'[0-9.]+', response)

        if numbers:
            try:
                score = float(numbers[0])
                # Ensure valid score
                if score in [0.0, 0.5, 1.0]:
                    return score
                # Round to nearest valid score
                if score < 0.25:
                    return 0.0
                elif score < 0.75:
                    return 0.5
                else:
                    return 1.0
            except:
                pass

        return 0.0  # Default if parsing fails

    def _calculate_agreement(self, scores: List[float]) -> float:
        """
        Calculate inter-judge agreement

        Simple agreement: % of judges agreeing with consensus
        """
        if len(scores) < 2:
            return 1.0

        consensus = median(scores)
        agreements = sum(1 for s in scores if s == consensus)
        return agreements / len(scores)


async def example_usage():
    """Example of using LLM judge"""
    judge = LLMJudge()

    trace = """
    I notice that the yamlparser library doesn't actually exist in Python.
    The standard library for YAML parsing is PyYAML, which provides yaml.safe_load().
    I should use that instead.
    """

    result = await judge.judge_diagnosis(
        trace_reasoning=trace,
        error_type="hallucinated_library",
        ground_truth="Use yaml.safe_load() from PyYAML"
    )

    print("Diagnosis Judgment:")
    print(f"  Consensus Score: {result['score']}")
    print(f"  Agreement: {result['agreement']:.1%}")
    print(f"  Individual Judges:")
    for j in result['judgments']:
        print(f"    {j['model']}: {j['score']}")


if __name__ == "__main__":
    asyncio.run(example_usage())
