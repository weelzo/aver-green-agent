"""
Consistency Testing Module

Implements pass@k metric from τ-bench to measure agent reliability.
Tests if agents consistently perform well vs one-time luck.
"""

import asyncio
from typing import List, Dict, Any
from statistics import mean, stdev

from .models import TaskScenario, EvaluationMetrics


class ConsistencyTester:
    """
    Tests agent consistency using pass@k metric

    Runs same task k times and measures:
    - pass@k: Success on all k trials
    - variance: Performance consistency
    - reliability score: Weighted metric
    """

    def __init__(self, k: int = 3):
        """
        Initialize consistency tester

        Args:
            k: Number of times to run each task
        """
        self.k = k

    async def test_consistency(
        self,
        run_task_func,
        task: TaskScenario,
        agent_url: str,
        agent_id: str,
        success_threshold: float = 60.0
    ) -> Dict[str, Any]:
        """
        Test task consistency across k runs

        Args:
            run_task_func: Async function to run single task
            task: Task to test
            agent_url: Agent URL
            agent_id: Agent ID
            success_threshold: Score threshold for "success"

        Returns:
            Consistency metrics dictionary
        """
        print(f"[Consistency] Testing {task.task_id} with k={self.k} runs")

        results: List[EvaluationMetrics] = []
        scores: List[float] = []
        successes: List[bool] = []

        # Run k times
        for run in range(self.k):
            print(f"[Consistency]   Run {run + 1}/{self.k}...", end=" ")

            try:
                metrics = await run_task_func(agent_url, agent_id, task)
                results.append(metrics)
                scores.append(metrics.total_score)
                successes.append(metrics.total_score >= success_threshold)
                print(f"Score: {metrics.total_score:.1f}/100")

            except Exception as e:
                print(f"FAILED: {e}")
                # Count as failure
                successes.append(False)
                scores.append(0.0)

        # Calculate metrics
        pass_at_1 = successes[0] if successes else False
        pass_at_k = all(successes)
        pass_any = any(successes)

        avg_score = mean(scores) if scores else 0.0
        score_variance = stdev(scores) if len(scores) > 1 else 0.0

        consistency_metrics = {
            "task_id": task.task_id,
            "k": self.k,
            "runs": len(results),

            # pass@k metrics
            "pass@1": pass_at_1,
            f"pass@{self.k}": pass_at_k,
            "pass_any": pass_any,
            "success_rate": sum(successes) / len(successes) if successes else 0.0,

            # Score metrics
            "avg_score": round(avg_score, 2),
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "variance": round(score_variance, 2),

            # Individual run scores
            "run_scores": [round(s, 1) for s in scores],
            "run_successes": successes,

            # Detailed results
            "detailed_results": [r.to_dict() for r in results]
        }

        # Print summary
        print(f"[Consistency] Summary:")
        print(f"              pass@1: {pass_at_1}")
        print(f"              pass@{self.k}: {pass_at_k}")
        print(f"              Avg: {avg_score:.1f} ± {score_variance:.1f}")

        return consistency_metrics


async def run_consistency_suite(
    green_agent,
    agent_url: str,
    agent_id: str,
    tasks: List[TaskScenario],
    k: int = 3
) -> Dict[str, Any]:
    """
    Run consistency testing on multiple tasks

    Args:
        green_agent: AVER green agent instance
        agent_url: Agent URL
        agent_id: Agent ID
        tasks: List of tasks to test
        k: Number of runs per task

    Returns:
        Aggregate consistency results
    """
    tester = ConsistencyTester(k=k)
    all_results = []

    for i, task in enumerate(tasks, 1):
        print(f"\n[Consistency Suite] Task {i}/{len(tasks)}: {task.task_id}")

        consistency = await tester.test_consistency(
            run_task_func=green_agent._run_single_task,
            task=task,
            agent_url=agent_url,
            agent_id=agent_id
        )

        all_results.append(consistency)

    # Aggregate statistics
    aggregate = {
        "participants": {
            "purple_agent": agent_id
        },
        "agent_id": agent_id,
        "num_tasks": len(tasks),
        "k": k,

        # Overall pass@k rates
        "overall_pass@1": mean([r["pass@1"] for r in all_results]),
        f"overall_pass@{k}": mean([r[f"pass@{k}"] for r in all_results]),

        # Score statistics
        "avg_score": mean([r["avg_score"] for r in all_results]),
        "avg_variance": mean([r["variance"] for r in all_results]),

        # Task-level results
        "task_results": all_results
    }

    print(f"\n[Consistency Suite] AGGREGATE:")
    print(f"  pass@1: {aggregate['overall_pass@1']:.1%}")
    print(f"  pass@{k}: {aggregate[f'overall_pass@{k}']:.1%}")
    print(f"  Avg Score: {aggregate['avg_score']:.1f}")
    print(f"  Avg Variance: {aggregate['avg_variance']:.1f}")

    return aggregate


if __name__ == "__main__":
    print("ConsistencyTester module")
    print("Implements pass@k metric from τ-bench")
    print("\nUsage:")
    print("  tester = ConsistencyTester(k=3)")
    print("  results = await tester.test_consistency(run_func, task, agent_url, agent_id)")
