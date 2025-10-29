# AVER: Agent Verification & Error Recovery Benchmark

**A benchmark for evaluating AI agents' ability to detect, diagnose, and recover from errors**

[![Status](https://img.shields.io/badge/status-in%20development-yellow)]()
[![Phase](https://img.shields.io/badge/phase-Week%201%20Foundation-blue)]()
[![Progress](https://img.shields.io/badge/progress-15%25-orange)]()

---

## Overview

AVER (Agent Verification & Error Recovery) is the **first benchmark specifically designed to measure AI agents' meta-cognitive capabilities**: error detection, diagnosis, and recovery.

While existing benchmarks (SWE-bench, WebArena, GAIA, etc.) test whether agents can complete tasks, **AVER tests what agents do when things go wrong**.

### The Problem

Production deployment of AI agents is blocked by reliability issues:
- Agents spiral into catastrophic failures when errors occur
- Small mistakes compound into task-level failures
- No existing benchmark measures self-correction capabilities

### The Solution

AVER provides:
- **40 carefully designed tasks** across 5 error categories
- **3-level evaluation**: Detection (40%) + Diagnosis (20%) + Recovery (40%)
- **Validated metrics**: False positive rate < 10%, inter-annotator agreement > 80%
- **Open platform integration**: Built on AgentBeats/A2A standard

---

## Quick Start

### Installation

```bash
# Clone repository
cd aver-green-agent

# Create virtual environment
python3 -m venv aver-venv
source aver-venv/bin/activate  # On Windows: aver-venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running AVER

```python
from src.aver.task_suite import TaskSuite
from src.aver.evaluator import ReliabilityEvaluator

# Load task suite
suite = TaskSuite()
suite.load_all_tasks()

# Select a task
task = suite.select_random(category="hallucination", difficulty=2)

# Evaluate agent (pseudo-code - full implementation coming Week 2)
# evaluator = ReliabilityEvaluator()
# metrics = evaluator.evaluate(task, agent_trace)
# print(metrics.summary())
```

---

## Project Structure

```
aver-green-agent/
├── CLAUDE.md                    # Research coordination & paper draft
├── RESEARCH_LOG.md              # Detailed step-by-step research log
├── README.md                    # This file
│
├── src/aver/                    # Core AVER implementation
│   ├── models.py                # Data models (TaskScenario, EvaluationMetrics)
│   ├── task_suite.py            # Task loading and selection
│   ├── evaluator.py             # Reliability evaluation engine
│   ├── green_agent.py           # Main orchestration (coming Week 2)
│   └── error_injector.py        # Error injection (coming Week 2)
│
├── tasks/                       # Task suite (YAML files)
│   ├── TASK_TEMPLATE.yaml       # Template for new tasks
│   ├── hallucination/           # Hallucination detection tasks
│   │   └── aver_hallucination_code_api_2_001.yaml  # First task
│   ├── validation/              # Output validation tasks
│   ├── tool_misuse/             # Tool misuse recovery tasks
│   ├── context_loss/            # Context/consistency tasks
│   └── adversarial/             # Adversarial robustness tasks
│
├── experiments/                 # Research experiments & results
│   ├── difficulty_calibration/
│   ├── baseline_results/
│   └── validation_studies/
│
├── scenarios/                   # AgentBeats scenarios
│   └── aver/
│       ├── scenario.toml        # Test configuration
│       └── baseline_agents/     # Baseline agents
│
├── paper/                       # Research paper content
│   ├── figures/
│   └── tables/
│
└── tests/                       # Unit & integration tests
```

---

## Task Categories

AVER tests 5 types of errors derived from real agent failures:

### 1. Hallucination Detection (8 tasks, 20%)
- Invented APIs, libraries, or facts
- Example: "Use the yamlparser library" (doesn't exist)

### 2. Output Validation (8 tasks, 20%)
- Incorrect calculations, formats, or logic
- Example: Math errors in code generation

### 3. Tool Misuse & Recovery (8 tasks, 20%)
- Wrong tool selection or parameters
- Example: Using search when file read is needed

### 4. Context Loss & Consistency (6 tasks, 15%)
- Forgetting constraints or requirements
- Example: Violating earlier specifications

### 5. Adversarial Robustness (6 tasks, 15%)
- Ambiguity and multiple errors
- Example: Conflicting requirements

---

## Evaluation Methodology

### Three-Level Scoring

AVER evaluates agents on three capabilities:

**1. Detection (40%)**: Did the agent notice the error?
- Explicit: Agent states the error in reasoning
- Implicit: Agent takes verification actions

**2. Diagnosis (20%)**: Did the agent identify why the error occurred?
- Evaluated via pattern matching or LLM-as-judge

**3. Recovery (40%)**: Did the agent successfully fix the problem?
- Evaluated via code execution tests or criteria matching

### Example Scoring

```
Task: aver_hallucination_code_api_2_001
Agent: GPT-4

Detection: 1.0 (Agent stated "yamlparser doesn't exist")
Diagnosis: 1.0 (Identified yaml module as correct approach)
Recovery: 1.0 (Code uses yaml.safe_load() correctly)

Total Score: 100/100
```

---

## Development Status

### Current Phase: Week 1 - Foundation ✅

- [x] AgentBeats tutorial completed
- [x] Research documentation created (CLAUDE.md, RESEARCH_LOG.md)
- [x] Directory structure set up
- [x] Core data models implemented (models.py)
- [x] Task suite management (task_suite.py)
- [x] Evaluation engine (evaluator.py)
- [x] First task created (hallucination YAML parsing)
- [ ] Green agent skeleton (in progress)

**Progress**: 15% complete (Week 1, Day 1)

### Upcoming Milestones

- **Week 2**: Core infrastructure (error injection, trace collection, sandbox)
- **Week 3**: Task development I (20 tasks, baseline testing)
- **Week 4**: Task development II (expand to 40 tasks)
- **Week 5**: **CRITICAL** - Validation studies (false positive/negative rates)
- **Week 6**: Integration & optimization
- **Week 7**: Documentation & paper writing
- **Week 8**: Launch preparation & submission

**Deadline**: December 19, 2025 (8 weeks, 58 days remaining)

---

## Creating New Tasks

1. Copy the task template:
```bash
cp tasks/TASK_TEMPLATE.yaml tasks/{category}/aver_{category}_{type}_{difficulty}_{number}.yaml
```

2. Fill in all sections:
   - Task description (what agent should do)
   - Tools available
   - Error injection details
   - Detection signals (how to know agent noticed)
   - Recovery criteria (how to know agent fixed it)

3. Validate the task:
```python
from src.aver.task_suite import TaskSuite

suite = TaskSuite()
task = suite._load_task_file("path/to/task.yaml")
errors = suite.validate_task(task)
print(errors)  # Should be empty list
```

4. Test with baseline agents (Week 3+)

---

## Research Documents

### Key Files

- **CLAUDE.md**: Main research coordination document
  - Project overview and status
  - Research paper draft (sections 1-9)
  - Experiment log
  - Development timeline
  - ~1,500 lines, continuously updated

- **RESEARCH_LOG.md**: Detailed research tracking
  - Step-by-step activity log
  - Every code change with rationale
  - Problems and solutions
  - Research insights
  - Updated after every significant step

- **AGENTBEATS_TUTORIAL_GUIDE.md**: Architecture analysis
  - Complete walkthrough of AgentBeats tutorial
  - A2A protocol patterns
  - Mapping to AVER implementation

- **AVER_CRITICAL_SUCCESS_FACTORS.md**: Strategy document
  - Task design best practices (40% of success)
  - Evaluation validity guidelines (30% of success)
  - Quality checklists

---

## Contributing

AVER is under active development for AgentX-AgentBeats Phase 1 competition.

### For Researchers

- Review task designs in `tasks/`
- Suggest validation approaches
- Contribute baseline agent implementations

### For Developers

- Implement missing components (see CLAUDE.md for roadmap)
- Add unit tests
- Improve evaluation logic

### For Task Designers

- Create new tasks following template
- Focus on realistic error scenarios
- Ensure clear detection signals and recovery criteria

---

## Citation

```bibtex
@misc{aver2025,
  title={AVER: A Benchmark for Evaluating Error Detection and Recovery in AI Agents},
  author={AVER Research Team},
  year={2025},
  note={In development for AgentX-AgentBeats Phase 1}
}
```

---

## License

[To be determined]

---

## Contact

For questions or feedback, see:
- GitHub Issues

---

**Last Updated**: October 29, 2025
**Status**: Week 1, Day 1 - Foundation Phase
**Progress**: 15% complete
