"""
Test AVER with Realistic Purple Agent

This script:
1. Starts the purple agent server (if not running)
2. Runs AVER green agent against it
3. Collects and displays results

Usage:
    # Terminal 1: Start purple agent
    python scripts/purple_agent_realistic.py

    # Terminal 2: Run test
    python scripts/test_with_purple_agent.py --tasks 1

    # Or test specific task
    python scripts/test_with_purple_agent.py --task aver_hallucination_code_api_2_001
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.aver.green_agent import AVERGreenAgent


async def main():
    parser = argparse.ArgumentParser(description="Test AVER with realistic purple agent")
    parser.add_argument(
        "--agent-url",
        default="http://localhost:8000",
        help="Purple agent URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--task",
        help="Specific task ID to test"
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=1,
        help="Number of random tasks to test (default: 1)"
    )
    parser.add_argument(
        "--category",
        choices=["hallucination", "validation", "tool_misuse", "context_loss", "adversarial"],
        help="Filter by category"
    )
    parser.add_argument(
        "--difficulty",
        type=int,
        choices=[1, 2, 3, 4],
        help="Filter by difficulty level"
    )
    args = parser.parse_args()

    print("="*80)
    print("AVER + Realistic Purple Agent Test")
    print("="*80)
    print(f"Purple Agent: {args.agent_url}")
    print(f"Tasks: {args.task or f'{args.tasks} random'}")
    if args.category:
        print(f"Category: {args.category}")
    if args.difficulty:
        print(f"Difficulty: {args.difficulty}")
    print("="*80)
    print()

    # Check if purple agent is running
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{args.agent_url}/health", timeout=2.0)
            if response.status_code == 200:
                health = response.json()
                print(f"✓ Purple agent is running")
                print(f"  Model: {health.get('model', 'unknown')}")
                print(f"  Type: {health.get('agent_type', 'unknown')}")
                print()
            else:
                print(f"⚠ Purple agent returned status {response.status_code}")
                print()
    except Exception as e:
        print(f"✗ Cannot connect to purple agent at {args.agent_url}")
        print(f"  Error: {e}")
        print()
        print("Please start the purple agent first:")
        print("  python scripts/purple_agent_realistic.py")
        print()
        return

    # Initialize AVER green agent
    green_agent = AVERGreenAgent(
        tasks_dir="tasks",
        results_dir="results",
        use_llm_judge=False
    )

    # Run assessment
    try:
        results = await green_agent.assess_agent(
            agent_url=args.agent_url,
            agent_id="realistic_purple_agent",
            task_id=args.task,
            category=args.category,
            difficulty=args.difficulty,
            num_tasks=args.tasks
        )

        print("\n" + "="*80)
        print("TEST COMPLETE")
        print("="*80)
        print(f"Tasks completed: {len(results)}")
        print()

        if results:
            # Show detailed results
            for i, result in enumerate(results, 1):
                print(f"\n--- Task {i}: {result.task_id} ---")
                print(f"Detection:  {result.detection_score:.2f} - {result.detection_details.get('reasoning', '')}")
                print(f"Diagnosis:  {result.diagnosis_score:.2f} - {result.diagnosis_details.get('reasoning', '')}")
                print(f"Recovery:   {result.recovery_score:.2f} - {result.recovery_details.get('reasoning', '')}")
                print(f"Total:      {result.total_score:.1f}/100")

                # Show detection details
                if result.detection_details.get('explicit_matches'):
                    print(f"\nExplicit detection signals:")
                    for match in result.detection_details['explicit_matches']:
                        print(f"  - Turn {match['turn']}: '{match['signal']}'")

            # Overall summary
            print("\n" + "="*80)
            avg_total = sum(r.total_score for r in results) / len(results)
            avg_detection = sum(r.detection_score for r in results) / len(results)
            avg_diagnosis = sum(r.diagnosis_score for r in results) / len(results)
            avg_recovery = sum(r.recovery_score for r in results) / len(results)

            print(f"AVERAGE SCORES")
            print(f"Detection:  {avg_detection:.2f} ({avg_detection*100:.0f}%)")
            print(f"Diagnosis:  {avg_diagnosis:.2f} ({avg_diagnosis*100:.0f}%)")
            print(f"Recovery:   {avg_recovery:.2f} ({avg_recovery*100:.0f}%)")
            print(f"Total:      {avg_total:.1f}/100")
            print("="*80)

    except Exception as e:
        print(f"\n✗ Error during assessment: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
