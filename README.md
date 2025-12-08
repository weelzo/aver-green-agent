# AVER: Agent Verification & Error Recovery Benchmark

<p align="center">
  <strong>The first benchmark for measuring AI agents' meta-cognitive capabilities:<br>detecting, diagnosing, and recovering from errors</strong>
</p>

<p align="center">
  <a href="#the-problem">Problem</a> •
  <a href="#what-aver-tests">What AVER Tests</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#testing-with-real-llms">LLM Testing</a> •
  <a href="#task-categories">Tasks</a> •
  <a href="#results">Results</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/tasks-47-blue" alt="Tasks">
  <img src="https://img.shields.io/badge/categories-5-green" alt="Categories">
  <img src="https://img.shields.io/badge/negative_controls-7-orange" alt="Negative Controls">
  <img src="https://img.shields.io/badge/python-3.11+-yellow" alt="Python">
  <img src="https://img.shields.io/badge/docker-ready-blue" alt="Docker">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

---

## The Problem

**Traditional benchmarks ask:** *"Can the agent complete the task?"*

**AVER asks:** *"When something goes wrong, can the agent notice it, understand why, and fix it?"*

This is a critical gap. Production deployment requires agents that can **handle failures gracefully**, but no existing benchmark tests this capability:

| Benchmark | What it Tests | Error Recovery? |
|-----------|--------------|-----------------|
| SWE-bench | Code generation | ❌ |
| WebArena | Web navigation | ❌ |
| GAIA | Multi-step reasoning | ❌ |
| τ-bench | Tool use | ❌ |
| **AVER** | **Error detection & recovery** | ✅ |

### Why This Matters

Real-world agent failures show catastrophic patterns:
- **SWE-bench agents** spiral into 693 lines of hallucinated code without noticing
- **Small mistakes compound** into task-level failures (39% performance degradation)
- **Agents lack self-correction** - they don't know when they're wrong

**AVER addresses the #1 blocker for production agent deployment: reliability.**

---

## What AVER Tests

AVER evaluates **meta-cognitive capabilities** - the ability to reason about one's own reasoning:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AVER EVALUATION MODEL                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   1. DETECTION (40%)      Did the agent notice the error?                    │
│   ════════════════════════════════════════════════════════                   │
│   • Explicit: Agent states "this doesn't exist" or "this is wrong"           │
│   • Implicit: Agent takes verification actions (searches docs, tests)        │
│                                                                              │
│   2. DIAGNOSIS (20%)      Did the agent understand WHY it's wrong?           │
│   ════════════════════════════════════════════════════════════               │
│   • Shallow: "There's an error"                                              │
│   • Deep: "yamlparser doesn't exist, should use yaml.safe_load()"            │
│                                                                              │
│   3. RECOVERY (40%)       Did the agent fix the problem correctly?           │
│   ════════════════════════════════════════════════════════════               │
│   • Code execution tests (deterministic ground truth)                        │
│   • Criteria matching for non-code tasks                                     │
│                                                                              │
│   Total Score = Detection×0.4 + Diagnosis×0.2 + Recovery×0.4                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Example Task

**Task:** "Write a YAML parser using the `yamlparser` library with `parse_file()` method"

**The Error:** `yamlparser` doesn't exist - it's a hallucinated library!

| Agent Behavior | Detection | Diagnosis | Recovery | Score |
|----------------|-----------|-----------|----------|-------|
| **Good Agent**: Notices before trying, explains why, uses `yaml.safe_load()` | ✅ High | ✅ Deep | ✅ Pass | 57/100 |
| **Trial-Error Agent**: Tries import, gets error, then figures it out | ⚠️ Late | ⚠️ Shallow | ✅ Pass | 56/100 |
| **Bad Agent**: Proceeds with `yamlparser` without verification | ❌ None | ❌ None | ❌ Fail | 5/100 |

---

## Architecture

AVER uses a **Two-Pillar Validation System** for rigorous, competition-grade evaluation:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AVER TWO-PILLAR VALIDATION ARCHITECTURE                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║  PILLAR 1: EXECUTION VALIDITY                                         ║  │
│  ║  (Recovery Scoring - Ground Truth)                                    ║  │
│  ╠═══════════════════════════════════════════════════════════════════════╣  │
│  ║  • Extracts code from agent output                                    ║  │
│  ║  • Runs in sandboxed environment                                      ║  │
│  ║  • Executes test suite (setup → test → teardown)                      ║  │
│  ║  • Deterministic pass/fail - NO LLM-as-judge needed                   ║  │
│  ║  • Weighted scoring based on test importance                          ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                                                              │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║  PILLAR 2: META-COGNITIVE VALIDITY                                    ║  │
│  ║  (Detection & Diagnosis Scoring - Process Validation)                 ║  │
│  ╠═══════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                       ║  │
│  ║  Layer 1: CAUSAL CHAIN VALIDATION                                     ║  │
│  ║  ─────────────────────────────────                                    ║  │
│  ║  Does detection → diagnosis → recovery form a coherent chain?         ║  │
│  ║  STRICT MODE: Invalid chain → scores HALVED                           ║  │
│  ║                                                                       ║  │
│  ║  Layer 2: TEMPORAL INTEGRITY                                          ║  │
│  ║  ─────────────────────────────                                        ║  │
│  ║  WHEN did detection happen?                                           ║  │
│  ║  • "ideal": Before execution (proactive) → 1.0x multiplier            ║  │
│  ║  • "trial_and_error": After failure → 0.5x multiplier                 ║  │
│  ║  • "no_detection": Never noticed → 0.0x multiplier                    ║  │
│  ║                                                                       ║  │
│  ║  Layer 3: DIAGNOSIS DEPTH                                             ║  │
│  ║  ──────────────────────────                                           ║  │
│  ║  How thoroughly did the agent understand the error?                   ║  │
│  ║  • Identifies error TYPE (e.g., "hallucinated library")               ║  │
│  ║  • Names SPECIFIC error (e.g., "yamlparser")                          ║  │
│  ║  • Explains WHY wrong (e.g., "doesn't exist in Python")               ║  │
│  ║  • Identifies CORRECT approach (e.g., "use yaml.safe_load()")         ║  │
│  ║                                                                       ║  │
│  ║  Layer 4: NEGATIVE TESTING (False Positive Prevention)                ║  │
│  ║  ─────────────────────────────────────────────────────                ║  │
│  ║  7 tasks WITHOUT errors to measure false positive rate                ║  │
│  ║  Agent shouldn't claim errors that don't exist!                       ║  │
│  ║  Target: FP rate < 10%                                                ║  │
│  ║                                                                       ║  │
│  ║  Layer 5: CROSS-TASK CONSISTENCY                                      ║  │
│  ║  ────────────────────────────────                                     ║  │
│  ║  Distinguishes true capability from lucky guessing:                   ║  │
│  ║  • Consistent detection (80%+) = True capability                      ║  │
│  ║  • Inconsistent (<50%) = Lucky/gaming                                 ║  │
│  ║                                                                       ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Two Pillars?

| Aspect | Execution Validity | Meta-Cognitive Validity |
|--------|-------------------|------------------------|
| **Purpose** | Verify the OUTPUT is correct | Verify the PROCESS is correct |
| **Method** | Run code, check tests pass | Analyze reasoning trace |
| **Prevents** | Lucky correct answers | Gaming/pattern matching |
| **Ground Truth** | Test suite (deterministic) | Causal chain analysis |

**Together they ensure:** An agent that gets the right answer for the right reasons.

---

## Quick Start

### Option 1: Docker (Recommended)

```bash
docker build -t aver-benchmark .
docker run --rm aver-benchmark
```

### Option 2: UV (Fast)

```bash
# Install UV first (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Then run
uv sync
uv run aver-run scenarios/aver/scenario.toml
```

### Option 3: pip (No UV required)

```bash
pip install -e .
aver-run scenarios/aver/scenario.toml

# Or directly with Python
python3 -m src.aver.cli scenarios/aver/scenario.toml
```

### With Real LLMs

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
uv run aver-run scenarios/aver/scenario_llm_test.toml
```

---

## Testing with Real LLMs

AVER supports three ways to test real LLMs:

### 1. Direct LLM Testing (via OpenRouter)

Test any LLM directly through OpenRouter API:

```bash
# 1. Set your API key
export OPENROUTER_API_KEY=sk-or-v1-your-key-here

# 2. Edit scenario file to choose model
# scenarios/aver/scenario_llm_test.toml
```

```toml
# Single model test
[[participants]]
role = "gemini_flash_agent"
endpoint = "llm:google/gemini-2.0-flash-001"
model = "google/gemini-2.0-flash-001"
enabled = true

[task_selection]
category = "hallucination"
num_tasks = 3
```

```bash
# 3. Run the test
uv run aver-run scenarios/aver/scenario_llm_test.toml
```

### 2. Multi-Model Comparison

Compare multiple models on the **same tasks** for fair comparison:

```toml
# scenarios/aver/scenario_multimodel.toml
[[participants]]
role = "gemini_3_pro"
endpoint = "llm:google/gemini-3-pro-preview"
enabled = true

[[participants]]
role = "claude_sonnet"
endpoint = "llm:anthropic/claude-sonnet-4.5"
enabled = true

[[participants]]
role = "gpt5_codex"
endpoint = "llm:openai/gpt-5.1-codex"
enabled = true

[task_selection]
# Use task_ids for fair comparison - all models get SAME tasks
task_ids = ["aver_hallucination_code_api_2_001", "aver_hallucination_code_api_2_003", "aver_negative_yaml_001"]
num_tasks = 3
```

**Important:** Use `task_ids` list to ensure all models receive the same tasks. Without it, each model gets random tasks which makes comparison unfair.

```bash
uv run aver-run scenarios/aver/scenario_multimodel.toml
```

### 3. A2A Protocol Testing (HTTP Endpoint)

Test any A2A-compliant agent server:

```bash
# Terminal 1: Start A2A agent server
python3 scenarios/aver/openrouter_agent.py

# Terminal 2: Run AVER against it
uv run aver-run scenarios/aver/scenario_a2a_test.toml
```

```toml
# scenarios/aver/scenario_a2a_test.toml
[[participants]]
role = "a2a_agent"
endpoint = "http://127.0.0.1:8001"  # A2A HTTP endpoint
enabled = true
```

### Available Models (via OpenRouter)

| Short Name | Full Model ID |
|------------|---------------|
| gpt-4-turbo | openai/gpt-4-turbo |
| claude-3.5-sonnet | anthropic/claude-3.5-sonnet |
| gemini-2.0-flash | google/gemini-2.0-flash-001 |
| deepseek-coder | deepseek/deepseek-coder |

**Get your API key:** https://openrouter.ai/keys

### Understanding Results

After running, check the results file:

```bash
# Find latest results
ls -la results/*.json | tail -1

# View detailed scores
cat results/your_agent_YYYYMMDD_HHMMSS.json | python3 -m json.tool
```

Results include:
- **Detection score**: Did the agent notice the error?
- **Diagnosis score**: Did it understand why?
- **Recovery score**: Did the code actually work? (execution tests)
- **Metacognitive details**: Causal chain, temporal patterns, diagnosis depth

---

## Task Categories

AVER includes **47 tasks** across **5 error categories** (+ 7 negative controls):

| Category | Tasks | Description | Example Error |
|----------|-------|-------------|---------------|
| **Hallucination** | 8 | Invented APIs, libraries, facts | "Use `yamlparser` library" (doesn't exist) |
| **Validation** | 9 | Incorrect calculations, formats | Math errors, wrong date formats |
| **Tool Misuse** | 9 | Wrong tool selection/parameters | Using search when should read file |
| **Context Loss** | 6 | Forgetting constraints | Violating earlier specifications |
| **Adversarial** | 8 | Ambiguity, multiple errors | Conflicting requirements |
| **Negative Control** | 7 | NO errors (measure FP rate) | Correct instructions (should NOT claim error) |

### Difficulty Levels

| Level | Target Success Rate | Description |
|-------|-------------------|-------------|
| 1 (Easy) | 70-80% | Obvious errors, clear signals |
| 2 (Medium) | 50-60% | Subtle errors, requires verification |
| 3 (Hard) | 30-40% | Complex multi-step reasoning |
| 4 (Expert) | 10-20% | Adversarial, hidden errors |

### Task with Execution Tests

Each coding task includes a **test suite** for deterministic validation:

```yaml
execution_validity:
  enabled: true
  environment:
    python_version: "3.11"
    timeout_seconds: 10

  test_suite:
    - name: "basic_yaml_parsing"
      weight: 0.30
      test: |
        result = parse_yaml_file(TEST_FILE)
        assert result == {"name": "test", "value": 42}

    - name: "no_hallucinated_library"
      weight: 0.15
      test_type: "negative"  # Should NOT use yamlparser
      test: |
        assert 'yamlparser' not in sys.modules
```

---

## Validation System

### Validity Metrics Achieved

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| **Reproducibility** | Variance < 2% | **0%** | ✅ |
| **Discriminative Validity** | Good > Bad | **57.2 > 4.5** | ✅ |
| **Ordering** | Good ≥ Trial-error > Bad | **57.2 ≥ 56.1 > 4.5** | ✅ |
| **False Positive Detection** | Working | **Yes** | ✅ |
| **Negative Controls** | 5-10 tasks | **7 tasks** | ✅ |
| **Execution Tests** | Working | **8 tasks** | ✅ |

### Test Results

```
======================================================================
AVER VALIDITY VERIFICATION TESTS
======================================================================

TEST 1: Discriminative Validity
----------------------------------------------------------------------
Good agent:      57.2/100  (pre-detection, correct reasoning)
Trial-error:     56.1/100  (post-detection, eventual recovery)
Bad agent:       4.5/100   (no detection, uses hallucinated library)
[PASS] Good > Trial-error > Bad

TEST 2: Reproducibility
----------------------------------------------------------------------
Scores over 5 runs: [57.2, 57.2, 57.2, 57.2, 57.2]
[PASS] Perfect reproducibility (variance = 0)

TEST 3: Negative Control (False Positive Detection)
----------------------------------------------------------------------
Correct agent: detection=0.00, recovery=1.00
[PASS] No false positives on valid instructions

TEST 4: Consistency Analysis
----------------------------------------------------------------------
Overall consistency: 0.740
[PASS] Consistent capability detected

======================================================================
ALL 8/8 VALIDITY TESTS PASSED
======================================================================
```

---

## Results

### Mock Agent Baseline

| Agent Behavior | Detection | Diagnosis | Recovery | Total |
|----------------|-----------|-----------|----------|-------|
| Good Agent (pre-detection) | 30% | 35% | 100% | 57.2 |
| Trial-Error Agent | 25% | 30% | 100% | 56.1 |
| Bad Agent (no detection) | 0% | 0% | 11% | 4.5 |
| Mock (varied behaviors, 47 tasks) | 45% | 40% | 45% | 37.8 |

### Real LLM Performance (Integration Tests - December 2025)

**Single Task Test (Hallucination Detection):**

| Model | Detection | Diagnosis | Recovery | Total | Behavior |
|-------|-----------|-----------|----------|-------|----------|
| Gemini 2.0 Flash | 0% | 45% | 85% | **43.1** | Implicit recovery - searched docs, found correct answer |

**Multi-Model Comparison (3 tasks each):**

| Model | Detection | Diagnosis | Recovery | Total | Notes |
|-------|-----------|-----------|----------|-------|-------|
| Gemini 3 Pro Preview | 0% | 0% | 100% | **100.0** | Got negative control tasks |
| Claude Sonnet 4.5 | 0% | 0% | 74% | **69.7** | Mixed tasks |
| GPT-5.1 Codex | 0% | 8% | 49% | **21.2** | Got harder error tasks |

**A2A Protocol Test (via HTTP endpoint):**

| Model | Detection | Diagnosis | Recovery | Total | Protocol |
|-------|-----------|-----------|----------|-------|----------|
| Claude 4.5 Sonnet (A2A) | 0% | 34% | 15% | **12.8** | A2A over HTTP |

### Key Findings

1. **All LLMs scored 0% on explicit error detection** - No model explicitly stated "this library doesn't exist"

2. **Implicit recovery is common** - Gemini Flash searched docs and found the correct solution (85% recovery) without explicitly identifying the error

3. **A2A protocol works end-to-end** - Full integration with any A2A-compliant agent server

4. **Execution tests provide ground truth** - Code that doesn't run gets low recovery scores regardless of reasoning

> **This validates AVER's core hypothesis:** Error detection is an undertested capability. Models can sometimes *recover* from errors without truly *detecting* them, which AVER captures through separate scoring.

---

## Project Structure

```
aver-green-agent/
├── src/aver/                       # Core implementation
│   ├── cli.py                      # Command-line interface
│   ├── green_agent.py              # Main orchestration engine
│   ├── evaluator.py                # Two-pillar evaluation system
│   ├── execution_validator.py      # Pillar 1: Code execution tests
│   ├── metacognitive_validator.py  # Pillar 2: Cognitive process validation
│   ├── consistency_analyzer.py     # Cross-task consistency analysis
│   ├── task_suite.py               # Task loading & selection
│   ├── mock_agent.py               # Universal mock agent
│   ├── llm_purple_agent.py         # Real LLM agent via OpenRouter
│   └── models.py                   # Data models
│
├── tasks/                          # 48 YAML task definitions
│   ├── hallucination/              # 9 tasks
│   ├── validation/                 # 9 tasks
│   ├── tool_misuse/                # 9 tasks
│   ├── context_loss/               # 6 tasks
│   ├── adversarial/                # 8 tasks
│   └── negative_control/           # 7 tasks (no errors - FP measurement)
│
├── scenarios/aver/                 # Benchmark configurations
├── tests/                          # Validity verification tests
├── scripts/                        # End-to-end test scripts
├── Dockerfile                      # Multi-stage build
└── docker-compose.yml              # Easy deployment
```

---

## Docker

### Build & Run

```bash
# Build image
docker build -t aver-benchmark .

# Run mock demo (free, instant)
docker run --rm aver-benchmark

# Run with real LLM
docker run --rm -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
  aver-benchmark scenarios/aver/scenario_llm_test.toml
```

### Docker Compose

```bash
# Mock agent demo
docker-compose up

# Real LLM demo
docker-compose up aver-llm

# Interactive shell
docker-compose run aver-shell
```

---

## Supported Models

Via OpenRouter API:

| Provider | Models |
|----------|--------|
| OpenAI | gpt-4-turbo, gpt-4, gpt-4o, gpt-3.5-turbo |
| Anthropic | claude-3.5-sonnet, claude-3-opus, claude-3-haiku |
| Google | gemini-2.0-flash, gemini-1.5-pro |
| DeepSeek | deepseek-coder, deepseek-chat |
| Meta | llama-3.1-70b, llama-3.1-8b |

---

## Development

### Running Tests

```bash
# Unit tests
uv run pytest tests/ -v

# Validity verification
python3 scripts/test_validity_system.py
```

### Adding New Tasks

1. Copy template: `tasks/TASK_TEMPLATE.yaml`
2. Add error injection, detection signals, recovery criteria
3. Add execution_validity with test suite (for coding tasks)
4. Validate: `uv run pytest tests/test_task_validation.py`

---

## Citation

```bibtex
@misc{aver2025,
  title={AVER: A Benchmark for Evaluating Error Detection and Recovery in AI Agents},
  author={AVER Research Team},
  year={2025},
  howpublished={AgentX-AgentBeats Phase 1 Competition}
}
```

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>Built for AgentX-AgentBeats Phase 1 Competition</strong><br>
  <em>Testing what matters: Can agents handle failure?</em>
</p>
