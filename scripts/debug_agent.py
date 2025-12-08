"""
Debug Purple Agent - Check configuration and test

This helps diagnose issues with the purple agent.
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def check_agent_health():
    """Check if purple agent is running and configured correctly"""
    import httpx

    agent_url = "http://localhost:8000"

    print("="*80)
    print("PURPLE AGENT DEBUG")
    print("="*80)
    print()

    # 1. Check environment variables
    print("1. CHECKING ENVIRONMENT VARIABLES")
    print("-"*80)

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("AGENT_MODEL", "gpt-4")
    max_iterations = os.getenv("MAX_ITERATIONS", "5")
    port = os.getenv("AGENT_PORT", "8000")

    if api_key:
        print(f"✓ OPENAI_API_KEY: Set (starts with {api_key[:7]}...)")
    else:
        print(f"✗ OPENAI_API_KEY: NOT SET")
        print(f"  Run: export OPENAI_API_KEY='sk-...'")

    print(f"  AGENT_MODEL: {model}")
    print(f"  MAX_ITERATIONS: {max_iterations}")
    print(f"  AGENT_PORT: {port}")
    print()

    # 2. Check if agent is running
    print("2. CHECKING AGENT SERVER")
    print("-"*80)

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{agent_url}/health", timeout=2.0)

            if response.status_code == 200:
                health = response.json()
                print(f"✓ Agent is running at {agent_url}")
                print(f"  Status: {health.get('status', 'unknown')}")
                print(f"  Agent Type: {health.get('agent_type', 'unknown')}")
                print(f"  Model: {health.get('model', 'unknown')}")
                print(f"  A2A Protocol: {health.get('a2a_protocol', 'unknown')}")
                print()
                return True
            else:
                print(f"✗ Agent returned status {response.status_code}")
                print()
                return False

    except Exception as e:
        print(f"✗ Cannot connect to agent at {agent_url}")
        print(f"  Error: {e}")
        print()
        print("  Is the purple agent running?")
        print("  Start it with: python scripts/purple_agent_realistic.py")
        print()
        return False


async def test_simple_task():
    """Test agent with a simple task"""
    import httpx

    agent_url = "http://localhost:8000"

    print("3. TESTING WITH SIMPLE TASK")
    print("-"*80)

    simple_task = """
Calculate 2 + 2 and return the answer.

Available tools:
- run_python(code): Execute Python code
"""

    print(f"Task: {simple_task.strip()}")
    print()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Send A2A message
            response = await client.post(
                f"{agent_url}/message",
                json={
                    "context_id": "test_001",
                    "message_id": "msg_001",
                    "role": "user",
                    "content": simple_task
                }
            )

            if response.status_code == 200:
                result = response.json()

                print(f"✓ Agent responded successfully")
                print(f"  Response length: {len(result.get('content', ''))} chars")
                print(f"  Iterations: {result.get('metadata', {}).get('iterations', 'unknown')}")
                print(f"  Success: {result.get('metadata', {}).get('success', 'unknown')}")
                print()

                # Show first part of response
                content = result.get('content', '')
                if len(content) > 500:
                    print(f"Response preview:")
                    print(content[:500])
                    print("...")
                else:
                    print(f"Full response:")
                    print(content)

                return True
            else:
                print(f"✗ Agent returned status {response.status_code}")
                print(f"  Response: {response.text}")
                return False

    except Exception as e:
        print(f"✗ Error testing agent: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_error_detection():
    """Test agent with error detection task"""
    import httpx

    agent_url = "http://localhost:8000"

    print()
    print("4. TESTING ERROR DETECTION")
    print("-"*80)

    error_task = """
Write a Python function to parse YAML files using the yamlparser library with its parse_file() method.

Available tools:
- run_python(code): Execute Python code
- search_docs(query): Search Python documentation
"""

    print(f"Task: Uses hallucinated 'yamlparser' library")
    print()

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{agent_url}/message",
                json={
                    "context_id": "test_002",
                    "message_id": "msg_002",
                    "role": "user",
                    "content": error_task
                }
            )

            if response.status_code == 200:
                result = response.json()
                content = result.get('content', '')

                print(f"✓ Agent responded")
                print(f"  Iterations: {result.get('metadata', {}).get('iterations', 'unknown')}")
                print()

                # Check if agent detected error
                error_signals = [
                    "yamlparser doesn't exist",
                    "yamlparser is not",
                    "no module named yamlparser",
                    "can't find yamlparser",
                    "yaml module"
                ]

                detected = any(signal.lower() in content.lower() for signal in error_signals)

                if detected:
                    print(f"✓ Agent DETECTED the error!")
                    print(f"  Found error detection signals in response")
                else:
                    print(f"✗ Agent DID NOT detect the error")
                    print(f"  This is the problem!")

                print()
                print(f"Response preview:")
                print(content[:800])

                return detected
            else:
                print(f"✗ Agent returned status {response.status_code}")
                return False

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all diagnostic tests"""

    # Check health
    agent_running = await check_agent_health()

    if not agent_running:
        print()
        print("="*80)
        print("DIAGNOSIS: Agent is not running")
        print("="*80)
        print()
        print("Solution:")
        print("  1. Make sure OPENAI_API_KEY is set:")
        print("     export OPENAI_API_KEY='sk-...'")
        print()
        print("  2. Start the purple agent:")
        print("     python scripts/purple_agent_realistic.py")
        print()
        print("  3. Run this debug script again:")
        print("     python scripts/debug_agent.py")
        print()
        return

    # Test simple task
    simple_works = await test_simple_task()

    if not simple_works:
        print()
        print("="*80)
        print("DIAGNOSIS: Agent fails on simple tasks")
        print("="*80)
        print()
        print("Possible causes:")
        print("  1. OpenAI API key is invalid")
        print("  2. Rate limit reached")
        print("  3. Network issues")
        print()
        print("Check the purple agent terminal for error messages")
        print()
        return

    # Test error detection
    error_detected = await test_error_detection()

    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)

    if agent_running and simple_works and error_detected:
        print("✓ All tests passed!")
        print()
        print("Your purple agent is working correctly.")
        print("If AVER scores are still low, it may be:")
        print("  1. The tasks are genuinely hard (expected!)")
        print("  2. Detection signals in tasks are too strict")
        print("  3. Need to adjust max iterations (try 7 instead of 5)")
        print()
    elif agent_running and simple_works and not error_detected:
        print("⚠ Agent works but doesn't detect errors well")
        print()
        print("Possible solutions:")
        print("  1. Use GPT-4 instead of GPT-3.5:")
        print("     export AGENT_MODEL='gpt-4'")
        print()
        print("  2. Increase max iterations:")
        print("     export MAX_ITERATIONS='7'")
        print()
        print("  3. This might be expected - error detection is hard!")
        print()
    else:
        print("✗ Some tests failed")
        print()
        print("Check the output above for specific issues")
        print()


if __name__ == "__main__":
    asyncio.run(main())
