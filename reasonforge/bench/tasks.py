"""
L9 — Built-in benchmark tasks (内置题库).

Every task carries a *gold chain*: an ordered list of machine-checkable claims
whose last claim pins the answer. This makes the benchmark **self-grounding** —
correctness is decided by the verification layer, not by string matching.

Domains: ``arithmetic`` (12), ``logic`` (8), ``constraint`` (6), ``code`` (6).

Author: 晨星 (Chenxing)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ["Task", "TASKS", "get_task", "by_domain", "DOMAINS", "make_task"]

DOMAINS = ("arithmetic", "logic", "constraint", "code")

# A gold claim is (claim_id, natural-language text, machine expr, kind, deps)
GoldClaim = Tuple[str, str, str, str, Tuple[str, ...]]


@dataclass
class Task:
    id: str
    question: str
    domain: str
    gold_answer: str
    gold_claims: List[GoldClaim] = field(default_factory=list)
    tags: Tuple[str, ...] = ()
    distractors: Tuple[str, ...] = ()
    """Plausible-but-wrong answers that wrong chains collapse onto.

    Real LLMs do not fail uniformly — they fall into a few attractors
    (off-by-one, sign flip, swapped operand). Benchmarks that spread errors
    uniformly make majority vote look far stronger than it really is.
    """

    def __post_init__(self) -> None:
        if self.domain not in DOMAINS:
            raise ValueError(f"unknown domain {self.domain!r}")
        ids = [c[0] for c in self.gold_claims]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate claim ids in task {self.id!r}")
        known = set(ids)
        for c in self.gold_claims:
            for d in c[4]:
                if d not in known:
                    raise ValueError(f"claim {c[0]!r} depends on unknown {d!r}")

    def n_claims(self) -> int:
        return len(self.gold_claims)


def make_task(
    tid: str,
    question: str,
    domain: str,
    gold_answer: str,
    claims: Sequence[Tuple[str, str, str]],
    deps: Optional[Dict[str, Tuple[str, ...]]] = None,
    tags: Sequence[str] = (),
) -> Task:
    """Helper: claims given as ``(id, text, expr)``; kind inferred from domain."""
    deps = deps or {}
    gold: List[GoldClaim] = [
        (cid, text, expr, domain, tuple(deps.get(cid, ()))) for cid, text, expr in claims
    ]
    return Task(id=tid, question=question, domain=domain, gold_answer=gold_answer,
                gold_claims=gold, tags=tuple(tags))


# --------------------------------------------------------------------------
# arithmetic (multi-step word problems)
# --------------------------------------------------------------------------
_ARITH = [
    ("arith-01", "小明买了 12 盒铅笔，每盒 5 支，又单独买了 8 支。一共多少支？", "68",
     [("c1", "12 盒 × 每盒 5 支 = 60 支", "12 * 5 = 60"),
      ("c2", "60 支 + 8 支 = 68 支", "60 + 8 = 68")], {"c2": ("c1",)}),
    ("arith-02", "一件商品原价 200 元，先涨价 15%，再降价 20 元。现价多少？", "210",
     [("c1", "涨价 15% = 200 × 0.15 = 30 元", "15% of 200 = 30"),
      ("c2", "200 + 30 = 230 元", "200 + 30 = 230"),
      ("c3", "230 - 20 = 210 元", "230 - 20 = 210")], {"c2": ("c1",), "c3": ("c2",)}),
    ("arith-03", "一个水池容量 480 升，进水管每小时进 60 升，同时排水管每小时排 20 升。几小时注满？", "12",
     [("c1", "净进水速度 = 60 - 20 = 40 升/小时", "60 - 20 = 40"),
      ("c2", "480 ÷ 40 = 12 小时", "480 / 40 = 12")], {"c2": ("c1",)}),
    ("arith-04", "甲车速度 75 km/h，乙车速度 60 km/h，相向而行，相距 405 km。几小时相遇？", "3",
     [("c1", "相对速度 = 75 + 60 = 135 km/h", "75 + 60 = 135"),
      ("c2", "405 ÷ 135 = 3 小时", "405 / 135 = 3")], {"c2": ("c1",)}),
    ("arith-05", "班级有 36 人，男生占 4/9，女生比男生多多少人？", "4",
     [("c1", "男生 = 36 × 4 ÷ 9 = 16 人", "36 * 4 / 9 = 16"),
      ("c2", "女生 = 36 - 16 = 20 人", "36 - 16 = 20"),
      ("c3", "20 - 16 = 4 人", "20 - 16 = 4")], {"c2": ("c1",), "c3": ("c2",)}),
    ("arith-06", "一本书 320 页，第一天读了 1/8，第二天读了剩下的 1/4。还剩多少页？", "210",
     [("c1", "第一天 = 320 ÷ 8 = 40 页", "320 / 8 = 40"),
      ("c2", "剩余 = 320 - 40 = 280 页", "320 - 40 = 280"),
      ("c3", "第二天 = 280 ÷ 4 = 70 页", "280 / 4 = 70"),
      ("c4", "还剩 280 - 70 = 210 页", "280 - 70 = 210")],
     {"c2": ("c1",), "c3": ("c2",), "c4": ("c3",)}),
    ("arith-07", "某数乘以 7 再加 13 等于 90。这个数是多少？", "11",
     [("c1", "7x + 13 = 90，故 7x = 77", "90 - 13 = 77"),
      ("c2", "x = 77 ÷ 7 = 11", "77 / 7 = 11")], {"c2": ("c1",)}),
    ("arith-08", "仓库有 15 箱货物，每箱 24 件，运走 160 件后还剩多少件？", "200",
     [("c1", "总数 = 15 × 24 = 360 件", "15 * 24 = 360"),
      ("c2", "360 - 160 = 200 件", "360 - 160 = 200")], {"c2": ("c1",)}),
    ("arith-09", "一件工作甲 6 小时完成，乙 3 小时完成。两人合作几小时完成？", "2",
     [("c1", "甲效率 = 1/6，乙效率 = 1/3", "1 / 6 + 1 / 3 = 0.5"),
      ("c2", "1 ÷ 0.5 = 2 小时", "1 / 0.5 = 2")], {"c2": ("c1",)}),
    ("arith-10", "一个长方形长 18 米，宽比长短 5 米。面积是多少平方米？", "234",
     [("c1", "宽 = 18 - 5 = 13 米", "18 - 5 = 13"),
      ("c2", "面积 = 18 × 13 = 234", "18 * 13 = 234")], {"c2": ("c1",)}),
    ("arith-11", "某商品进价 80 元，按 25% 利润率定价，再打八折出售。售价多少？", "80",
     [("c1", "定价 = 80 × 1.25 = 100 元", "80 * 1.25 = 100"),
      ("c2", "八折 = 100 × 0.8 = 80 元", "100 * 0.8 = 80")], {"c2": ("c1",)}),
    ("arith-12", "三个连续偶数之和为 96，最大的数是多少？", "34",
     [("c1", "中间数 = 96 ÷ 3 = 32", "96 / 3 = 32"),
      ("c2", "最大数 = 32 + 2 = 34", "32 + 2 = 34")], {"c2": ("c1",)}),
]

# --------------------------------------------------------------------------
# logic (propositional entailment)
# --------------------------------------------------------------------------
_LOGIC = [
    ("logic-01", "若下雨则地湿。今天下雨了。结论：地湿。这个推理成立吗？", "yes",
     [("c1", "前提 rain 且 rain -> wet，推出 wet", "rain, rain -> wet |= wet")], {}),
    ("logic-02", "若下雨则地湿。地湿。能推出下雨了吗？", "no",
     [("c1", "肯定后件存在反例：地湿但没下雨", "SAT: wet, rain -> wet, not rain")], {}),
    ("logic-03", "若 A 则 B；若 B 则 C；已知 A。能推出 C 吗？", "yes",
     [("c1", "假言三段论：A, A->B, B->C ⊨ C", "A, A -> B, B -> C |= C")], {}),
    ("logic-04", "并非（A 且 B）等价于（非 A 或 非 B）吗？", "yes",
     [("c1", "德摩根律", "not (A and B) <-> (not A or not B)")], {}),
    ("logic-05", "A 或非 A 是否恒真？", "yes",
     [("c1", "排中律", "A or not A")], {}),
    ("logic-06", "已知 A -> B 且 非 B。能推出 非 A 吗？", "yes",
     [("c1", "否定后件式（modus tollens）", "A -> B, not B |= not A")], {}),
    ("logic-07", "A -> B 与 B -> A 是否等价？", "no",
     [("c1", "蕴涵不具对称性：存在 A 真 B 假的反例", "SAT: (A -> B) and not (B -> A)")], {}),
    ("logic-08", "若考试及格则毕业；若毕业则拿学位；已知未拿学位。能推出考试未及格吗？", "yes",
     [("c1", "链式否定后件", "P -> G, G -> D, not D |= not P")], {}),
]

# --------------------------------------------------------------------------
# constraint (finite-domain CSP)
# --------------------------------------------------------------------------
_CONS = [
    ("cons-01", "鸡兔同笼：共 10 个头，28 只脚。鸡有几只？（用 x=鸡, y=兔 表示）", "6",
     [("c1", "x + y = 10，2x + 4y = 28", "x=6; y=4 ### x + y = 10, 2*x + 4*y = 28")], {}),
    ("cons-02", "两个数之和为 17，之差为 3。较大的数是多少？（x=大数, y=小数）", "10",
     [("c1", "x + y = 17, x - y = 3", "x=10; y=7 ### x + y = 17, x - y = 3")], {}),
    ("cons-03", "三数 a,b,c 均为 1..6 的整数，a+b+c=12 且 a<b<c。c 是多少？（取一组解）", "5",
     [("c1", "a=3, b=4, c=5 满足约束与互异", "a=3; b=4; c=5 ### a+b+c = 12, a < b, b < c")], {}),
    ("cons-04", "用 5 元和 3 元邮票凑 37 元，两种都用。5 元邮票几张？（x=5元张数, y=3元张数）", "5",
     [("c1", "5x + 3y = 37, x>0, y>0", "x=5; y=4 ### 5*x + 3*y = 37, x > 0, y > 0")], {}),
    ("cons-05", "一个两位数，十位比个位大 3，交换后比原数小 27。原数是多少？（x=十位, y=个位）", "41",
     [("c1", "x = y + 3 且 (10y+x) = (10x+y) - 27", "x=4; y=1 ### x = y + 3, 10*y + x = 10*x + y - 27")], {}),
    ("cons-06", "甲比乙大 5 岁，两人年龄和为 45。甲多少岁？（x=甲, y=乙）", "25",
     [("c1", "x - y = 5, x + y = 45", "x=25; y=20 ### x - y = 5, x + y = 45")], {}),
]

# --------------------------------------------------------------------------
# code (executable verification)
# --------------------------------------------------------------------------
_CODE = [
    ("code-01", "用代码计算 1 到 100 所有整数之和。", "5050",
     [("c1", "循环累加 1..100", "s = 0\nfor i in range(1, 101):\n    s = s + i\nresult = s###5050")], {}),
    ("code-02", "用代码求 1 到 20 中所有能被 3 整除的数之和。", "63",
     [("c1", "筛选能被 3 整除的数并求和", "s = 0\nfor i in range(1, 21):\n    if i % 3 == 0:\n        s = s + i\nresult = s###63")], {}),
    ("code-03", "用代码计算 10 的阶乘。", "3628800",
     [("c1", "累乘 1..10", "p = 1\nfor i in range(1, 11):\n    p = p * i\nresult = p###3628800")], {}),
    ("code-04", "用代码求斐波那契数列第 15 项（F1=1, F2=1）。", "610",
     [("c1", "迭代求斐波那契", "a = 1\nb = 1\nfor i in range(3, 16):\n    c = a + b\n    a = b\n    b = c\nresult = b###610")], {}),
    ("code-05", "用代码求列表中最大值：[3, 17, 4, 25, 9, 12]。", "25",
     [("c1", "遍历求最大值", "xs = [3, 17, 4, 25, 9, 12]\nm = xs[0]\nfor x in xs:\n    if x > m:\n        m = x\nresult = m###25")], {}),
    ("code-06", "用代码统计 1 到 200 中同时能被 2 和 7 整除的数的个数。", "14",
     [("c1", "计数能被 14 整除的数", "cnt = 0\nfor i in range(1, 201):\n    if i % 14 == 0:\n        cnt = cnt + 1\nresult = cnt###14")], {}),
]


def _make_distractors(answer: str, domain: str) -> Tuple[str, ...]:
    """Derive the attractor answers a wrong chain typically lands on."""
    low = str(answer).strip().lower()
    if domain == "logic" or low in ("yes", "no", "true", "false"):
        opp = "no" if low in ("yes", "true", "1") else "yes"
        return (opp,)
    try:
        v = float(low)
    except ValueError:
        return ()
    if abs(v - round(v)) < 1e-9:
        iv = int(round(v))
        cands = [str(iv + 1), str(iv - 1)]
        if iv != 0:
            cands.append(str(-iv))
        return tuple(dict.fromkeys(cands))
    return (f"{v + 1:.10g}", f"{v - 1:.10g}")


def _build() -> List[Task]:
    out: List[Task] = []
    for tid, q, ans, claims, deps in _ARITH:
        t = make_task(tid, q, "arithmetic", ans, claims, deps, tags=("word-problem",))
        t.distractors = _make_distractors(ans, "arithmetic")
        out.append(t)
    for tid, q, ans, claims, deps in _LOGIC:
        t = make_task(tid, q, "logic", ans, claims, deps, tags=("entailment",))
        t.distractors = _make_distractors(ans, "logic")
        out.append(t)
    for tid, q, ans, claims, deps in _CONS:
        t = make_task(tid, q, "constraint", ans, claims, deps, tags=("csp",))
        t.distractors = _make_distractors(ans, "constraint")
        out.append(t)
    for tid, q, ans, claims, deps in _CODE:
        t = make_task(tid, q, "code", ans, claims, deps, tags=("execution",))
        t.distractors = _make_distractors(ans, "code")
        out.append(t)
    return out


TASKS: List[Task] = _build()


def get_task(task_id: str) -> Task:
    for t in TASKS:
        if t.id == task_id:
            return t
    raise KeyError(f"unknown task {task_id!r}")


def by_domain(domain: str) -> List[Task]:
    return [t for t in TASKS if t.domain == domain]


def summary() -> Dict[str, int]:
    return {d: len(by_domain(d)) for d in DOMAINS}
