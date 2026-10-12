# Invariants — machine-checkable correctness properties

> 每一个核心函数都带一条可被独立实现交叉验证的「不变量」。
> 这些是 `reasonforge` 对抗"凭直觉重写"的硬金标准：任意正确实现都必须满足。

Author: 晨星 (Chenxing)

---

## L0 — Numerical primitives (`core/numeric.py`)

| Function | Invariant | Cross-check |
|---|---|---|
| `logsumexp` | `logsumexp([a])==a`; shift-equivariant `logsumexp(x+c)==logsumexp(x)+c` | vs naive `log(sum(exp))` |
| `softmax` | outputs ∈ [0,1], `sum==1` (≤1e-12), monotone in logits | sum + ordering |
| `entropy` | `0 ≤ H ≤ log_b(n)`; uniform maximises | direct bound + sweep |
| `normalized_entropy` | ∈ [0,1]; `n≤1 → 0` | uniform → 1 |
| `gini_simpson` | ∈ [0, 1−1/n]; 0 iff degenerate | formula |
| `kl_divergence` | `≥ 0`, `== 0 ⟺ p==q` | identical vs different |
| `js_divergence` | symmetric, bounded by 1 (base 2) | `JS(p,q)==JS(q,p)` |
| `top_k_filter` | exactly `min(k,n)` finite entries, set preserved | count + set |
| `top_p_filter` | kept mass `≥ p` (up to float error) | recompute mass |
| `wilson_interval` | bounds ∈ [0,1]; `lo ≤ p̂ ≤ hi`; width ↓ as n ↑ | bounds |
| `expected_calibration_error` | `0 ≤ ECE ≤ 1`; `== 0` for perfectly calibrated | bin-conf==bin-acc |
| `beta_sample` | output ∈ [0,1]; mean → `a/(a+b)` | 4000-draw mean |
| `ucb_score` | unvisited arm → `+∞`; monotone in mean | edge cases |
| `DeterministicRNG` | bit-identical replay; `fork` distinct & stable | replay equality |

**Key correctness note (Beta sampler).** `Beta(a,b)` is built as
`X/(X+Y)` with `X~Gamma(a)`, `Y~Gamma(b)` **independent** and the *same* `X`
reused in numerator and denominator — otherwise the mean is a biased
`a'/(a'+b)` and fails the `2/7` test. `beta_sample` draws `x` once and reuses it.

---

## L1 — Trace structures (`core/trace.py`)

- `Trace.validate()` enforces referential integrity (every `dep` resolves to a
  real `Claim.id`) **and** acyclicity (Kahn topo-sort succeeds).
- `normalize_answer()` is **idempotent**: `normalize(normalize(s)) == normalize(s)`
  — currency/whitespace stripped, `"60.0" → "60"`.
- `answer_similarity()` returns 1.0 for two canonical-equivalent answers and is
  symmetric.

---

## L4 — Verifiers (`verifiers/`)

The registry semantics: **strictest verdict wins** — `False > None > True`.
One refutation beats any number of passes. This is what keeps the engine sound.

| Verifier | Soundness invariant |
|---|---|
| `ArithmeticVerifier` | **never** `eval`; whitelisted AST only; `1/0→None` (no crash), `9**9**9→None` (no hang) |
| `AlgebraVerifier` | `sympy` exact identity; sampling fallback deterministic; rejects false identities |
| `LogicVerifier` | `entails`/`valid`/`satisfiable` agree with DPLL on CNF (Tseitin preserves satisfiability) |
| `ConstraintVerifier` | AC-3 returns `False` iff a domain is wiped (incl. **unary** constraints); `check_assignment` exact |
| `CodeVerifier` | sandbox never executes `exec`/`eval`/`import`/dunder/unsafe attrs; caps steps/iters/depth; blocks infinite loops |
| `ConsistencyVerifier` | `answer_similarity ≥ 0.95` ⇒ same answer |

**AC-3 unary constraint fix.** `_revise(xi, xj)` handles binary arcs; the queue
seeds a self-arc `(v, v)` for any single-variable constraint (e.g. `"x > 5"`), so
a value violating it is pruned and `ac3` correctly returns `False` when the
domain empties.

---

## L5 — VWSC (`aggregate/vwsc.py`)

- All-clean traces ⇒ VWSC reduces to majority vote (no harm when nothing is wrong).
- A trace with a refuted claim scores **strictly below** an otherwise identical clean trace.
- Vote weights are non-negative and sum to 1.
- The gate is monotone: raising `gate` never *adds* a surviving chain.

`eff(c) = base(c) · Π_{d∈deps(c)} (α + (1−α)·eff(d))`, with `w(c) = 1 + β·depth(c)`
and `Wᵢ = exp(γ·vᵢ)·softmax(logprob)ᵢ`.

---

## L7 — Budget (`core/budget.py`)

`BudgetScheduler.should_stop` returns exactly one of
`confident | consensus | exhausted | starved`, and never permits more than
`max_traces` samples. Geometric batch growth (`growth=1.6`) is deterministic.

---

## L8 — CEGAR (`refine/cegar.py`)

- Each round is verification→aggregation→(decisive? stop : build feedback→resample).
- `suggest_fix` returns a *machine-derivable* correct value for arithmetic and
  constraint claims; otherwise a sound refutation without a suggestion.
- Bounded: `max_rounds` + `resample_per_round` cap the work. The 3-variable
  constraint task forced through all rounds converges in ≈ 2 s.

---

## L9 / L10 — Benchmark & engine

- Correctness is decided by **canonical answer matching** (`normalize_answer`),
  never raw string equality.
- The benchmark is fully seeded: identical seed ⇒ byte-identical report.
- `ReasonForge.solve` returns a complete audit trail (every trace, every
  verification result, every counterexample, and — under CEGAR — the repair history).
