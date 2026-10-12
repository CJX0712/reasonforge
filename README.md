# ReasonForge — Verifiable System-2 Reasoning Engine

> 可验证 System-2 推理引擎 · 作者：**晨星 (Chenxing)**
>
> 把"思维链"拆成可机检的断言，用一族符号验证器判定每一步对错，再用"验证结果"而非"多数投票"决定最终答案。旗舰算法 **VWSC**（验证加权自一致性）与 **CEGAR**（反例引导的抽象精化）让推理既可验证、又可自我修复。核心**零硬依赖**：仅用 Python 标准库即可运行；`sympy` 可增强代数验证，但非必需。

---

## Why this exists

Large language models fail most often as **locally consistent, globally wrong**
chains: one bad step, then every later step correctly derived from that bad
premise. A reader — and a naive majority-vote — cannot tell such a chain apart
from a correct one. The fix is to make every claim *machine-checkable* and let
verification, not popularity, decide.

`reasonforge` is a complete, layered implementation of that idea. It is not a
thin wrapper: every layer carries machine-checkable invariants (see
[`docs/INVARIANTS.md`](docs/INVARIANTS.md)), and the whole engine is exercised
by **101 deterministic tests**.

### What it delivers
- **Six symbolic verifiers** — arithmetic, algebra, propositional logic, finite-domain CSP, restricted Python code, and cross-trace consistency.
- **VWSC** — dependency-aware, depth-weighted, trust-gated, verification-weighted aggregation.
- **CEGAR** — feeds concrete counterexamples back to the model; a vague "try again" barely moves accuracy, a concrete counterexample repairs the chain.
- **Six inference-time strategies** — greedy, self-consistency (Wang et al. 2022), best-of-N verified, VWSC, VWSC-adaptive (early stopping), DVTS (Beeching et al. 2024).
- **Adaptive compute budgeting** — geometric batch growth + four early-stop rules.
- **Deterministic RNG** — bit-identical replays across platforms and CPython versions.
- **Honest benchmarking** — the simulated student collapses wrong answers onto a few *distractors* (not a uniform spread), so majority-vote baselines are honestly weak, exactly as in real LLMs.

---

## The 10-layer architecture

```
L0   numeric primitives          core/numeric.py        (logsumexp, entropy, ECE, Beta, RNG …)
L1   reasoning-trace structures  core/trace.py          (Claim/Step/Trace as a DAG of assertions)
L2   sampling backends           llm/                   (LLMBackend + deterministic MockStudent)
L3   search/strategy contract    search/base.py
L4   verification layer          verifiers/   ★ arith · algebra · logic · csp · code · consistency
L5   aggregation                 aggregate/vwsc.py       ★ VWSC (the flagship)
L6   inference-time strategies    search/      (SC · BoN · DVTS · UCB bandit)
L7   adaptive budgeting          core/budget.py
L8   refinement & calibration    refine/cegar.py · aggregate/calibration.py
L9   benchmark                   bench/
L10  orchestration & audit        engine.py
```

Each layer is independently testable; strategies differ only in *how they spend
compute* and all share the same verification layer, which is what makes them
comparable.

---

## Quick start

```python
from reasonforge import ReasonForge
from reasonforge.bench.tasks import get_task

rf = ReasonForge(seed=42)
out = rf.solve(get_task("arith-01"), strategy="vwsc_adaptive", use_cegar=True)
print(out.answer, out.confidence, out.correct)   # 68 0.8 True
print(rf.explain(out))                            # full audit trail
```

Plug in a real model by subclassing `LLMBackend`:

```python
from reasonforge.llm.base import LLMBackend, GenerationConfig
from reasonforge.core.numeric import DeterministicRNG
from reasonforge.core.trace import Trace

class MyLLM(LLMBackend):
    name = "my-llm"
    def sample(self, task, config: GenerationConfig, rng: DeterministicRNG) -> list[Trace]:
        ...  # return a list of Trace objects built from the model's outputs
```

CEGAR needs the backend to accept feedback — implement `sample_with_feedback`
(or rely on the default which degrades to plain `sample`):

```python
    def supports_feedback(self) -> bool:
        return True
    def sample_with_feedback(self, task, config, rng, feedback):
        ...  # pass the counterexamples back to the model
```

---

## Verifiers (L4) — the trust engine

| Verifier | Claim syntax | Soundness |
|---|---|---|
| `ArithmeticVerifier` | `"12 * 5 = 60"` | Whitelisted AST only; **never** `eval`. |
| `AlgebraVerifier` | `"(x+1)**2 = x**2 + 2*x + 1"` | `sympy` exact, deterministic sampling fallback. |
| `LogicVerifier` | `"A, A -> B \|= B"`, `SAT: …`, `VALID: …` | Recursive-descent parser + Tseitin CNF + DPLL. |
| `ConstraintVerifier` | `"x=6; y=4 ### x+y=10, x>y"` | AC-3 arc consistency + MRV backtracking. |
| `CodeVerifier` | `"… code … ### expected"` | Restricted Python-subset interpreter sandbox (no `exec`/`eval`/`import`/dunder). |
| `ConsistencyVerifier` | `"68"` (vs trace answer) | Canonical-answer match. |

The registry takes the **strictest** verdict (`False > None > True`): one
refutation beats any number of passes. This is the property that keeps the whole
system sound.

---

## The flagship algorithms

### VWSC — Verification-Weighted Self-Consistency (L5)

Three ideas over plain majority vote:

1. **Dependency-aware trust propagation** — a claim whose premises were refuted is
   discounted even if it verifies:
   `eff(c) = base(c)·Π_{d∈deps(c)}(α + (1−α)·eff(d))`, `α = 0.5`.
2. **Depth weighting** — downstream claims weigh more: `w(c) = 1 + β·depth(c)`.
3. **Hard gate + verification-weighted vote** — chains below the trust gate are
   dropped entirely; survivors vote with `Wᵢ = exp(γ·vᵢ)·softmax(logprob)ᵢ`.

### CEGAR — Counterexample-Guided Abstraction Refinement (L8)

```
sample → verify → aggregate → decisive? → stop
                          │
                          └─ not decisive → build feedback (exact wrong step + correct value)
                                          → re-sample conditioned on counterexamples
```

The loop is closed: the verifier *produces* the counterexample, the feedback
*payload* repairs the chain. Vague retry barely helps; a concrete counterexample
repairs it with high probability.

---

## Benchmark (reproducible, seeded)

Built-in suite: **32 tasks** — 12 arithmetic word problems, 8 logic entailments,
6 constraint CSPs, 6 code. All strategies run on the identical seed.

**Headline (full budget = 32 samples):**

| strategy | accuracy | mean samples |
|---|---|---|
| greedy (1 sample) | 0.688 | 1.0 |
| self_consistency (Wang 2022) | 0.938 | 32.0 |
| best_of_n_verified | 0.906 | 32.0 |
| **vwsc** (fixed 32) | **1.000** | 32.0 |
| **vwsc_adaptive** | **1.000** | **9.5** |

`vwsc_adaptive` reaches the same 100% as fixed-budget VWSC while spending ~3×
fewer samples.

**Accuracy vs compute budget** (error rate 0.45, 5 seeds/cell, `results/sweep.json`):

| budget | vwsc | vwsc_adaptive *(samples)* | self_consistency | greedy |
|---|---|---|---|---|
| 4  | 0.950 | 0.888 *(2.3)* | 0.819 | 0.575 |
| 8  | 0.994 | 0.988 *(4.4)* | 0.925 | 0.575 |
| 16 | 1.000 | 0.994 *(8.5)* | 0.969 | 0.575 |
| 32 | 1.000 | 1.000 *(16.5)* | 0.988 | 0.575 |

Two takeaways beyond raw accuracy:
- **Calibration.** Self-consistency's ECE climbs to **0.42** at budget 32 — it is
  *confidently wrong* (a uniform vote has no trust signal). VWSC keeps ECE ≈ 0.07
  because the vote weight is the verification trust, not the vote count.
- **The attractor model is honest.** `greedy` plateaus at 0.575, not at chance —
  wrong answers pile onto a few plausible distractors, exactly the failure mode
  that makes real majority-vote weak. That is the whole reason verification wins.

The sweep is regenerated by:

```bash
python examples/bench_sweep.py --budgets 4,8,16,32 --seeds 5
```

…which writes `results/sweep.json` and a markdown table.

---

## Project layout

```
reasonforge/
├── reasonforge/            # the package
│   ├── core/              # L0 numeric, L1 trace, L7 budget
│   ├── llm/               # L2 backends (MockStudent + LLMBackend contract)
│   ├── verifiers/         # L4 — the six verifiers + registry
│   ├── aggregate/         # L5 VWSC, calibration
│   ├── search/            # L3/L6 strategies (SC, BoN, DVTS, UCB)
│   ├── refine/            # L8 CEGAR
│   ├── bench/             # L9 tasks + runner
│   └── engine.py          # L10 orchestration
├── examples/bench_sweep.py
├── tests/                 # 101 deterministic tests
└── docs/INVARIANTS.md     # every mathematical invariant
```

---

## Install & test

```bash
pip install -e .            # optional; stdlib-only core needs nothing
pip install pytest sympy    # for tests + stronger algebra verification
pytest tests/ -q            # 101 passed
```

Python 3.9+. Zero hard dependencies for the core; `sympy` is an optional
accelerator for algebra verification, `numpy` is never required.

---

## License

MIT — see [LICENSE](LICENSE). Author: **晨星 (Chenxing)**.

> "零重写 · 集大成 · 闭环交付" — built on the world's best verified primitives,
> reusable and extensible, open-sourced and reproducible.
