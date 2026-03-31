# PRBench

Benchmark AI agents on scientific paper reproduction. The agent reads a physics paper and must reproduce its computational results from scratch — writing code, running simulations, and producing numerical output data that matches the paper's figures.

This repository is an open-source release of the evaluation harness. It includes two minimal test tasks (`aaatest_helloworld`, `bbbtest_alphabet`) and one full benchmark task (`task_white_1993`) as a representative sample.

---

## Prerequisites

- Python 3.11+
- Docker (running and accessible)
- Claude Code CLI: `npm install -g @anthropic-ai/claude-code`
- OpenAI Codex CLI (optional): `npm install -g @openai/codex`
- pip dependencies: `pip install -e .`

---

## Environment Variables

Set only the variables for the agent type(s) you intend to use.

```bash
# Claude Code (agent-type: claude)
export CC_API_KEY="<your-api-key>"
export CC_BASE_URL="https://api.anthropic.com"   # optional; default is Anthropic official
export CC_MODEL="claude-sonnet-4-5"              # optional
export CC_SMALL_FAST_MODEL="claude-haiku-4-5-20251001"  # optional

# OpenAI Codex (agent-type: codex)
export CODEX_API_KEY="<your-api-key>"
export CODEX_BASE_URL="https://api.openai.com"   # optional; default is OpenAI official
export CODEX_MODEL="gpt-4o"                      # optional

# OpenCode (agent-type: opencode)
export OPENCODE_API_KEY="<your-api-key>"
export OPENCODE_BASE_URL="..."                   # optional
export OPENCODE_MODEL="openai/gpt-4o"            # required; format: provider/model-name
```

Variable reference:

| Variable | Required | Notes |
|----------|----------|-------|
| `CC_API_KEY` | Yes (claude) | Falls back to `ANTHROPIC_API_KEY` |
| `CC_BASE_URL` | No | Falls back to `ANTHROPIC_BASE_URL` |
| `CC_MODEL` | No | Falls back to `ANTHROPIC_MODEL` |
| `CC_SMALL_FAST_MODEL` | No | Used for lightweight sub-tasks |
| `CODEX_API_KEY` | Yes (codex) | Falls back to `OPENAI_API_KEY` |
| `CODEX_BASE_URL` | No | Falls back to `OPENAI_BASE_URL` |
| `CODEX_MODEL` | No | Falls back to `OPENAI_MODEL` |
| `OPENCODE_API_KEY` | Yes (opencode) | Falls back to `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` depending on model prefix |
| `OPENCODE_BASE_URL` | No | Falls back to corresponding base URL |
| `OPENCODE_MODEL` | Yes (opencode) | Must include provider prefix: `openai/...` or `anthropic/...` |

---

## Running Evaluations

### Single task

```bash
python main.py launch --task-id task_white_1993
python main.py launch --task-id task_white_1993 --agent-type codex
python main.py launch --task-id task_white_1993 --code-only
python main.py launch --task-id task_white_1993 --white-agent-type opencode --green-agent-type claude
```

All parameters:

| Parameter | Required | Default | Notes |
|-----------|----------|---------|-------|
| `--task-id` | Yes | -- | Directory name under `data/tasks/` |
| `--agent-type` | No | `claude` | Default CLI for both agents; `claude`, `codex`, or `opencode` |
| `--white-agent-type` | No | `--agent-type` | Override for the task-solving agent |
| `--green-agent-type` | No | `--agent-type` | Override for the grading agent |
| `--code-only` | No | off | Write code only, skip simulation and CSV output |
| `--results-subdir` | No | `full` or `code_only` | Subdirectory under `results/` |
| `--green-port` | No | 9001 (0=auto) | Port for green agent A2A server |
| `--white-port` | No | 9002 (0=auto) | Port for white agent A2A server |
| `--no-archive` | No | off | Skip archiving workspace to `results/` |

### Batch (parallel)

```bash
python main.py batch --task-ids task_white_1993,aaatest_helloworld -j 2
python main.py batch --task-ids all --agent-type codex --code-only -j 4
```

| Parameter | Required | Default | Notes |
|-----------|----------|---------|-------|
| `--task-ids` | Yes | -- | Comma-separated IDs or `all` |
| `-j` / `--max-concurrent` | No | 2 | Max parallel evaluations |
| (all other `launch` params) | -- | -- | Same semantics |

### Re-grade existing results

```bash
python main.py grade --task-id task_white_1993
python main.py grade --task-id task_white_1993 --code-only --workspace results/full/task_white_1993/workspace_full
```

### Cleanup

Kill leftover agent processes and remove Docker containers:

```bash
python main.py cleanup
```

Note: `cleanup` removes all containers whose ancestor image is `python:3.11-slim`. Do not run it while other `python:3.11-slim` containers are in use.

---

## Architecture

```
+---------------+    A2A      +---------------+   docker exec   +---------------------+
|  Green Agent  | ---------> |  White Agent  | -------------> |  Agent CLI          |
|  (orchestrate | <--------- |  (manage proc)| <------------- |  (inside Docker)    |
|   + grade)    |  polling   |               |  stdout/stderr |  claude / codex /   |
|  port: auto   |            |  port: auto   |                |  opencode           |
+---------------+            +---------------+                +---------------------+
       |
       | docker exec (model-judge grading)
       +------------------------------------------------> same container
```

Lifecycle per evaluation:

1. Allocate 2 free TCP ports
2. Create Docker container; install chosen agent CLI(s) and inject credentials
3. Copy paper images, paper markdown, and input files into `/workspace`
4. Start green agent (port A) and white agent (port B) as child processes
5. Send task config to green agent; green agent orchestrates:
   a. Send instruction to white agent; white agent runs CLI inside Docker
   b. Poll every 30 s until `COMPLETED` (max 3 h)
   c. Copy ground truth into `workspace/_ground_truth/` (after white agent finishes, so it cannot access answers during solving)
   d. Run model-judge grading via CLI inside Docker; write `eval_report.json`
6. Kill green and white processes
7. Remove Docker container
8. Archive `workspace/` to `results/<subdir>/<task_id>/workspace_<mode>`
9. Copy `eval_report.json` to `results/<subdir>/<task_id>/`

---

## Evaluation Modes

| Mode | Flag | What the white agent does | Grading |
|------|------|--------------------------|---------|
| Full | (default) | Write code + run simulation + produce CSV data files | Model-judge reads agent code, output CSVs, and ground truth (metadata + reference CSVs + reference code) |
| Code-only | `--code-only` | Write code only, no execution | Model-judge reads agent code and ground truth (metadata + reference code); no CSV comparison |

---

## Results Directory

```
results/
+-- full/
|   +-- task_white_1993/
|       +-- eval_report.json           # grading report (copy)
|       +-- workspace_full/
|           +-- eval_logs/             # grading prompt, traces, eval_report.json
|           +-- _ground_truth/         # ground truth revealed after solving
|           |   +-- metadata.md
|           |   +-- data/              # reference CSVs
|           |   +-- reproduce/         # reference code
|           +-- _paper_images/         # paper figures given to white agent
|           +-- reproduction/          # agent's code output
|           +-- data/                  # agent's CSV output
+-- code_only/
|   +-- task_white_1993/
|       +-- eval_report.json
|       +-- workspace_code/
+-- full_codex/                        # custom subdir (--results-subdir full_codex)
    +-- ...
```

---

## Task Format

Each task lives in `data/tasks/<task_id>/`. The directory contains:

```
task_<id>/
+-- task.yaml          # task configuration
+-- instruction.md     # sent to the white agent (no ground truth)
+-- metadata.md        # ground truth formulas and methodology (grading only, hidden from white agent)
+-- <paper>.md         # full paper content
+-- data/              # reference CSV files (ground truth data, hidden until grading)
+-- reproduce/         # reference implementation code (hidden until grading)
+-- images*/           # paper figure images, optional
```

### task.yaml

`task.yaml` is the central configuration file. Key fields:

```yaml
task_id: task_white_1993

paper:
  title: "Density-matrix algorithms for quantum renormalization groups"
  author: "Steven R. White"
  doi: "10.1103/PhysRevB.48.10345"
  year: 1993
  paper_file: "white1993.md"       # paper content file (given to white agent)

instruction_file: "instruction.md"
metadata_file: "metadata.md"
ground_truth_data_dir: "data"      # directory with reference CSVs (default: data)
ground_truth_code_dir: "reproduce" # directory with reference code (default: reproduction)

input_files:                       # optional: files needed by white agent that are not ground truth
  - "input_table.csv"

expected_outputs:
  analysis:
    - "reproduction/ANALYSIS.md"
  code:
    - "reproduction/operators.py"
    - "reproduction/dmrg_infinite.py"
    # ...
  data:
    - "data/fig2.csv"
    - "data/fig3.csv"
    # ...

docker:
  image: "python:3.11-slim"
  memory_limit: "8g"
  timeout: 14400                   # seconds; 4 hours max
  pip_install:
    - numpy
    - scipy
    - matplotlib

grading:
  dimensions:
    - name: "methodology_understanding"
      weight: 0.20
      description: >
        Does the agent's ANALYSIS.md correctly describe the DMRG methodology ...
    - name: "code_correctness"
      weight: 0.15
      description: >
        Are all required code files present? Does the code implement DMRG correctly ...
    - name: "data_accuracy"
      weight: 0.50
      description: >
        How well do the generated CSV files match the ground truth reference data ...
    - name: "completeness"
      weight: 0.15
      description: >
        Are all expected output files present and non-trivial ...

code_only:                         # optional: separate config for code-only mode
  instruction_suffix: |
    ## Code-Only Mode
    Write code only. Do NOT run simulations or produce CSV files.
  expected_outputs:
    analysis:
      - "reproduction/ANALYSIS.md"
    code:
      - "reproduction/operators.py"
      # ...
  grading:
    dimensions:
      - name: "methodology_understanding"
        weight: 0.25
        description: ...
      - name: "code_correctness"
        weight: 0.50
        description: ...
```

Grading dimension weights must sum to 1.0 within each mode. The model-judge scores each dimension from 0 to 1, and the final score is a weighted sum.

### instruction.md

`instruction.md` is the only task description given to the white agent. It must be fully self-contained: the agent sees the paper and this file, nothing else.

It should specify:
- What the agent must produce (code files, CSV data files, analysis document)
- Exact output paths and filenames
- CSV column names, row counts, and value ranges for each data file
- Required code structure (modules, classes, function signatures if relevant)
- Banned libraries or practices
- Computational resource constraints (time, memory)

It must not contain ground truth numerical results or reference implementation details.

Example from `task_white_1993`:

```
## 5. Data Files to Reproduce

### Figure 2: Density-Matrix Eigenvalues
File: data/fig2.csv

X values: 1, 2, 3, ..., 50
Columns: alpha,"Open, S=1/2","Open, S=1","Periodic, S=1/2","Periodic, S=1"
```

### metadata.md

`metadata.md` is the ground truth document. It is hidden from the white agent and revealed to the grading agent only after the white agent has finished.

It should contain:
- All key formulas with equation numbers
- Step-by-step algorithm descriptions
- Expected qualitative results for each figure (trends, asymptotic values, etc.)
- A list of banned libraries (repeated from instruction.md, used as a grading criterion)

The grading agent uses `metadata.md` together with `data/` (reference CSVs) and `reproduce/` (reference code) to evaluate the white agent's output.

### data/ (ground truth CSVs)

The `data/` directory contains reference CSV files — one per figure or observable. These are produced by running the reference implementation in `reproduce/`.

CSV format requirements:
- Header row with descriptive column names
- Numeric values with sufficient precision (10 significant figures recommended)
- Column order and names must match what `instruction.md` specifies
- Files must be deterministic (same inputs produce the same output)

### reproduce/ (reference implementation)

The `reproduce/` directory contains a working reference implementation. It is used by the grading agent to understand how the problem should be solved, and serves as the authoritative source for what "correct" code looks like.

The grading agent reads the agent's code alongside the reference code and scores correctness. The reference code is never shown to the white agent.

---

## Designing a New Task

To add a task for a new paper:

1. Create `data/tasks/task_<author>_<year>/`
2. Write `task.yaml` with paper metadata, expected outputs, docker config, and grading rubric
3. Write `instruction.md`: full self-contained instructions, exact output specs, banned libraries
4. Write `metadata.md`: all formulas, algorithms, expected results — ground truth for grading
5. Add the paper as `<paper>.md` (converted from PDF if needed)
6. Write a reference implementation in `reproduce/` and run it to produce reference CSVs in `data/`
7. Verify the task runs end-to-end with the test agent on a simple sanity check

Design principles:
- The paper must have deterministic numerical results (e.g., from a well-defined simulation or algorithm)
- Figures or tables with specific numerical values make the best grading targets
- Banned-library constraints prevent the agent from trivially wrapping an existing package
- The reference implementation documents what the grader expects; write it carefully

---

## Sample Task: task_white_1993

`task_white_1993` is included as the open-source sample task. It reproduces 7 figures from:

> Steven R. White, "Density-matrix algorithms for quantum renormalization groups," *Phys. Rev. B* 48, 10345 (1993). DOI: 10.1103/PhysRevB.48.10345

The agent must implement the DMRG algorithm from scratch (no tensor network libraries), run simulations for spin-1/2 and spin-1 Heisenberg chains, and produce 7 CSV data files matching the paper's figures. This is a representative example of the difficulty and structure of benchmark tasks in PRBench.

---

## Monitoring

```bash
tail -f green_agent.log white_agent.log launcher.log
```
