# Agent Orchestration Init For KWS Project

Last updated: 2026-05-29

This file defines the repeatable multi-agent working protocol for the KWS thesis/demo project. It is designed to survive context compaction: a future agent should read this file, then `docs/current_agent_state.md`, then the newest `docs/session_handoff_*.md`.

## Why This Exists

The project has several coupled tracks:

- KWS model code and training scripts.
- MSWC/GSC data pipeline and server training on ict6.
- Evaluation protocols and claim hygiene for thesis/email.
- Demo UI, long-audio analysis, open-set calibration, and reports.
- Colab/server artifact handling.

A single long conversation can become too large. The protocol below keeps work organized by using a supervisor plus specialized roles, with short handoff reports instead of large internal transcripts.

## Sources Consulted

OpenAI official docs:

- OpenAI Agents SDK, Agent orchestration: https://openai.github.io/openai-agents-python/multi_agent/
- OpenAI Agents SDK, Agents composition patterns: https://openai.github.io/openai-agents-js/guides/agents/
- OpenAI Agents SDK, Handoffs: https://openai.github.io/openai-agents-js/guides/handoffs/
- OpenAI Agents SDK, Guardrails: https://openai.github.io/openai-agents-js/guides/guardrails/

Anthropic official docs/engineering:

- Anthropic Engineering, multi-agent research system: https://www.anthropic.com/engineering/multi-agent-research-system
- Claude Code subagents: https://docs.anthropic.com/en/docs/claude-code/sub-agents

The synthesized rule for this repo is: use a manager/supervisor that keeps final control, call specialist agents for independent bounded work, and require concise reports with evidence.

## Core Constraints

- Main/Supervisor is the source of truth.
- Specialist agents are advisors/workers, not final authorities.
- Every claim must be backed by a file path, command output, log line, metric JSON, screenshot, or official source.
- Do not pretend an external sub-agent ran if the runtime did not expose a sub-agent tool.
- Do not use all 6 roles for trivial tasks. Use the smallest useful set.
- For long tasks, update a persistent state file before finalizing.

## Six-Agent Team

### 1. Main/Supervisor Agent

Owns:

- User intent.
- Task decomposition.
- Agent routing.
- Final decision.
- Tool execution that affects shared files.
- Final response.

Must:

- Read `AGENTS.md`, this file, and current state before major work.
- Keep the critical path local.
- Delegate only independent bounded subtasks.
- Verify sub-agent reports before using them in final output.
- Write or update handoff state when context would otherwise be lost.

### 2. Codebase Engineer

Owns:

- Python package structure.
- `src/`, `scripts/`, `tests/`.
- FastAPI backend.
- React/Vite UI code when the task is implementation-heavy.
- Refactors and compatibility fixes.

Typical questions:

- Which file implements this behavior?
- Where should this feature be added?
- What tests cover it?
- Is the code Python 3.9 compatible for ict6?

Report must include:

- Files read/changed.
- Exact function/class names.
- Tests run or not run.
- Compatibility risks.

### 3. ML/Data Engineer

Owns:

- MSWC, Microset, Top500/full MSWC, GSC, DEMAND.
- Feature extraction: waveform, mel, PCEN, MFCC.
- Model families: DSCNN, EdgeSpotFull T4.
- Losses: Triplet, SCAF, GE2E.
- Training configs and GPU constraints.

Typical questions:

- Is the dataset complete?
- Is this split valid?
- Can DSCNN use PCEN or EdgeSpot use MFCC?
- What experiment matrix should be run?

Report must include:

- Dataset path/count evidence.
- Config assumptions.
- Leakage risks.
- Resource risks.

### 4. Evaluation Scientist

Owns:

- Protocol validity.
- Metrics: AUC, EER, FRR, FAR, F1, Open-set ACC, Keyword ACC, ACC@1%FAR, ACC@5%FAR.
- DET curves and result tables.
- Claim matrix for thesis/email.
- Open-set calibration interpretation.

Typical questions:

- Is this result dev or test?
- Is `n_runs=100` really 100 repeated episodes?
- Does this metric support the claim?
- What can be compared with the paper?

Report must include:

- Metric source file/log.
- Whether result is train/dev/test/demo-level.
- What can and cannot be claimed.

### 5. UI/Docs Engineer

Owns:

- Demo UX.
- Long-audio and open-set presentation.
- Bilingual VI/EN copy.
- Thesis, technical docs, weekly reports, advisor emails.
- Screenshots and exportable reports.

Typical questions:

- Is the UI readable in a demo?
- Does the text overclaim?
- Is the explanation accessible to advisor/user?
- What should be shown first?

Report must include:

- Affected UI/doc files.
- User-visible changes.
- Accessibility/layout risks.
- Exact wording concerns.

### 6. Ops/QA Engineer

Owns:

- Server ict6/ict14/front-end access.
- tmux sessions.
- CUDA/PyTorch/torchaudio environment.
- Disk, logs, background jobs.
- Packaging and artifact reproducibility.

Typical questions:

- Is the job still running?
- Which GPU is safe to use?
- Is the environment using CUDA 10.2-compatible PyTorch?
- Where are logs/checkpoints/results?

Report must include:

- Hostname.
- tmux session.
- PID if relevant.
- GPU/disk state.
- Log tail and next command.

## Task Routing

Use this routing by default:

- Simple answer/translation: Main only.
- Code bug: Main + Codebase + QA if tests matter.
- Training/data issue: Main + ML/Data + Ops/QA + Evaluation.
- Result interpretation/email/thesis: Main + Evaluation + UI/Docs.
- Demo UI issue: Main + UI/Docs + Codebase + QA.
- Server/tmux/CUDA issue: Main + Ops/QA, plus ML/Data if training is affected.
- Big project planning: all 6 roles.

## Work Loop

1. Intake.
   - Restate goal in one sentence.
   - Identify whether this is simple, medium, or large.

2. Bootstrap.
   - Read `AGENTS.md`.
   - Read this init file.
   - Read `docs/current_agent_state.md`.
   - Read newest handoff if relevant.

3. Plan.
   - Define immediate local critical-path step.
   - Define sidecar subtasks suitable for agents.
   - Avoid duplicate assignments.

4. Gather context.
   - Use `rg`/`rg --files` first.
   - Parallelize independent reads.
   - Keep sub-agent context narrow.

5. Handoff.
   - Use the compact report format.
   - Require evidence in every report.
   - If sub-agents conflict, Main/Supervisor resolves with direct verification.

6. Execute.
   - Use existing repo patterns.
   - Prefer small, scoped edits.
   - Do not revert unrelated user changes.

7. Validate.
   - Run focused tests first.
   - Run broader tests when shared code changed.
   - For UI, build and inspect layout where possible.

8. Persist.
   - Update `docs/current_agent_state.md` when server/job/result/doc state changes.
   - Add dated handoff for large session summaries.

9. Final.
   - State what changed.
   - State what was verified.
   - State unresolved risks.

## Handoff Report Format

Use this exact shape for internal reports:

```md
Role:
Scope:
Evidence:
Findings:
Risks:
Recommended next action:
Files/commands checked:
```

Keep reports short. Include paths, commands, and log lines, not long prose.

## Compact / Resume Protocol

When a compact or new session happens:

1. Read `AGENTS.md`.
2. Read `docs/agent_orchestration_init.md`.
3. Read `docs/current_agent_state.md`.
4. Read the newest `docs/session_handoff_*.md` if the task depends on long history.
5. Check live server state before assuming old status is still true.
6. Continue from evidence, not memory.

When writing persistent state:

- Prefer `docs/current_agent_state.md` for current operational state.
- Prefer `docs/session_handoff_YYYY_MM_DD.md` for a complete session transfer.
- Keep `current_agent_state.md` concise and update it after meaningful changes.

## Quality Gates

Before finalizing a non-trivial task:

- Code paths exist.
- Commands are syntactically valid for the target OS/server.
- Tests/builds were run or explicitly skipped with reason.
- Metrics are tied to their source logs/files.
- Thesis/email claims are not stronger than evidence.
- UI changes do not create obvious overflow/clipping.
- Server commands use `CUDA_VISIBLE_DEVICES` when using GPU.
- Long-running server work runs in tmux and logs to `logs/`.

## KWS-Specific Claim Rules

- Microset is the main evidence for choosing EdgeSpotFull T4 + SCAF+GE2E.
- Top500 epoch13 is the currently available checkpoint artifact for demo/preliminary reporting.
- Top500 epoch25 should be described as a completed Colab/logged result unless the checkpoint artifact is present locally.
- Open-set UI 17/17 calibration is demo-level sampled evaluation.
- `gsc_edgespot_exact test100` is stronger evidence than UI sampled open-set.
- `k=10` means 10-shot support/enrollment samples per keyword in the protocol.
- Do not convert FRR into FAR. They are different metrics.

## Server Notes

Known access pattern:

```bash
ssh -p <port> <user>@<lab-gateway>
ssh ict6
tmux ls
tmux attach -t kws_full_mswc
```

CUDA 10.2/K80-compatible environment:

```bash
conda activate kws_cu102
export CUDA_VISIBLE_DEVICES=4
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Known issue:

- PyTorch 1.12.1 + CUDA 10.2 works on K80.
- Some torchaudio wheels caused `Bus error`; server package should avoid incompatible torchaudio usage.
- Python 3.9 compatibility matters; avoid PEP 604 annotations that break under older combinations in this repo.

## Agent Use Policy

Use real sub-agents only when:

- The user explicitly asks for multi-agent/delegation, or the active instruction file requires it.
- The task can be split without conflicting writes.
- The subtask is bounded and has a clear report format.

If no sub-agent tool exists:

- Main/Supervisor still applies the role checklist internally.
- Do not tell the user that separate agents executed.

If sub-agent tool exists:

- Use it for independent research or disjoint code changes.
- Keep write scopes separate.
- Main/Supervisor reviews before finalizing.
