"""独立参考检查器。

与核心 (:mod:`polyinfer.inference`) **完全独立**：

* 不调用 ``infer`` / ``Inferencer.unify``；
* 不导入核心的任何函数来得到“期望值”；
* 只把核心产出的 :class:`InferenceResult` 当作待审数据，
  重新实现替换应用、结构同构判定、自由变量计算与闭包/值限制复核。

它核验三件事：

1. **替换良构**：最终替换幂等、无循环（独立的 occurs 检查），
   且右端不含替换变量自身；
2. **全部约束成立**：把最终替换分别作用于每条记录约束的两端，
   结果必须结构同构（含引用不变式逐点相等）；
3. **泛化合法**：每条 let 记录中，核心声称的泛化集合恰好等于
   “绑定类型自由变量 − 环境自由变量”；扩展式必须空泛化（值限制）。

对带 rank-1 签名的 let 额外核验（同样只用自带的替换应用与结构
同构判定，不借用待测统一器）：

4. **签名一致**：绑定类型在最终替换下必须与签名类型结构同构；
5. **刚性集合**：泛化集合恰为签名类型的自由变量（刚性变量集合），
   且这些刚性变量不出现在环境自由变量中（未逃逸）。

检查失败抛 :class:`ReferenceCheckFailure`，这表示核心（或被篡改的结果）
有 bug，而不是被测程序不可类型化——后者在 ``infer`` 阶段就已抛出。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Set

from .errors import PolytypeError
from .types import TCon, TFun, TRef, TVar, Type


class ReferenceCheckFailure(PolytypeError):
    """独立检查器发现核心结果不自洽。正常交付的核心不应触发。"""


# ------------------------------------------------------------ 自实现原语

def ref_apply(subst: Dict[TVar, Type], tp: Type, seen: Set[TVar] | None = None) -> Type:
    """检查器自带的替换应用（带循环检测），不调用核心模块函数。"""
    if seen is None:
        seen = set()
    if isinstance(tp, TVar):
        target = subst.get(tp)
        if target is None:
            return tp
        if tp in seen:
            raise ReferenceCheckFailure(f"替换存在循环链，经过变量 {tp}")
        return ref_apply(subst, target, seen | {tp})
    if isinstance(tp, TFun):
        return TFun(
            ref_apply(subst, tp.arg, seen),
            ref_apply(subst, tp.ret, seen),
        )
    if isinstance(tp, TRef):
        return TRef(ref_apply(subst, tp.inner, seen))
    return tp


def ref_free_vars(tp: Type, acc: Set[TVar] | None = None) -> Set[TVar]:
    """检查器自带的自由变量收集。"""
    if acc is None:
        acc = set()
    if isinstance(tp, TVar):
        acc.add(tp)
    elif isinstance(tp, TFun):
        ref_free_vars(tp.arg, acc)
        ref_free_vars(tp.ret, acc)
    elif isinstance(tp, TRef):
        ref_free_vars(tp.inner, acc)
    return acc


def ref_isomorphic(left: Type, right: Type, path: str = "$") -> None:
    """结构同构判定：构造子必须一一对应；TVar 只允许与自身比较。

    这里**故意不做任何变量绑定/统一**：检查器只验证“最终替换作用后
    两端得到同一个范式类型”。到达这里的类型已被最终替换解析到范式，
    若仍是不同的未绑定变量，则约束确实没被满足。
    """
    if isinstance(left, TVar) and isinstance(right, TVar):
        if left.ident != right.ident:
            raise ReferenceCheckFailure(
                f"约束在 {path} 未解析：左端变量 {left} ≠ 右端变量 {right}"
            )
        return
    if isinstance(left, TCon) and isinstance(right, TCon):
        if left.name != right.name:
            raise ReferenceCheckFailure(f"约束在 {path} 基础类型冲突：{left.name} ≠ {right.name}")
        return
    if isinstance(left, TFun) and isinstance(right, TFun):
        ref_isomorphic(left.arg, right.arg, path + ".arg")
        ref_isomorphic(left.ret, right.ret, path + ".ret")
        return
    if isinstance(left, TRef) and isinstance(right, TRef):
        ref_isomorphic(left.inner, right.inner, path + ".inner")
        return
    raise ReferenceCheckFailure(
        f"约束在 {path} 外构造子冲突：{type(left).__name__} ≠ {type(right).__name__}"
    )


# ------------------------------------------------------------ 结果核验

@dataclass(frozen=True)
class RefCheckReport:
    constraints_checked: int
    generalizations_checked: int
    substitution_size: int


def check_result(result: Any) -> RefCheckReport:
    """核验一份 ``infer`` 返回结果，全部通过才返回统计，否则抛错。"""
    subst = dict(result.subst)

    # 1) 替换良构：右端不含被绑定变量自身（occurs），且整体幂等。
    for var, rhs in subst.items():
        if var in ref_free_vars(rhs):
            raise ReferenceCheckFailure(f"替换不满足 occurs：{var} ↦ {rhs}")
        once = ref_apply(subst, var)
        twice = ref_apply(subst, once)
        ref_isomorphic(once, twice, f"idempotency[{var}]")

    # 2) 每条约束：最终替换作用后两端必须同构。
    for index, c in enumerate(result.constraints):
        l = ref_apply(subst, c.left)
        r = ref_apply(subst, c.right)
        try:
            ref_isomorphic(l, r, f"constraint#{index}")
        except ReferenceCheckFailure as exc:
            raise ReferenceCheckFailure(
                f"{exc}（左端路径 {c.left_path}，右端路径 {c.right_path}，"
                f"来源：{c.note}）"
            ) from exc

    # 3) 泛化：核心声称的集合必须恰好等于合法闭包集合。
    #    值限制由检查器依据契约自行重判（不信任核心字段）：绑定表达式
    #    只有是语法值（字面量/lambda/变量）时才允许闭包泛化；
    #    ref、apply 等扩展式的合法集合只能是空集——这正是引用漏洞的防线。
    value_tags = frozenset({"int", "bool", "lambda", "var"})
    for g in result.generalizations:
        is_value = g.bound_tag in value_tags
        bound_free = ref_free_vars(ref_apply(subst, g.bound_type))
        env_free: Set[TVar] = set()
        for _ename, quantified, body in g.env_snapshot:
            body_free = ref_free_vars(ref_apply(subst, body))
            env_free |= body_free - set(quantified)
        expected = (bound_free - env_free) if is_value else set()
        actual = set(g.generalized)
        if actual != expected:
            raise ReferenceCheckFailure(
                f"let {g.name!r} @ {g.path} 泛化集合不合法："
                f"核心给出 {sorted(v.ident for v in actual)}，"
                f"独立重算（{g.bound_tag} 是{'值' if is_value else '扩展式'}）"
                f"应为 {sorted(v.ident for v in expected)}"
            )
        sig_type = getattr(g, "sig_type", None)
        if sig_type is not None:
            # 4) 实现类型在最终替换下必须与签名类型结构同构。
            bound_final = ref_apply(subst, g.bound_type)
            try:
                ref_isomorphic(bound_final, sig_type, f"signature[{g.name}]")
            except ReferenceCheckFailure as exc:
                raise ReferenceCheckFailure(
                    f"let {g.name!r} @ {g.path} 的实现类型与签名不一致：{exc}"
                ) from exc
            # 5) 泛化集合恰为签名刚性变量，且刚性变量未逃逸进环境。
            sig_free = ref_free_vars(sig_type)
            if actual != sig_free:
                raise ReferenceCheckFailure(
                    f"let {g.name!r} @ {g.path} 泛化集合不等于签名刚性变量集："
                    f"核心给出 {sorted(v.ident for v in actual)}，"
                    f"签名为 {sorted(v.ident for v in sig_free)}"
                )
            leaked = env_free & sig_free
            if leaked:
                raise ReferenceCheckFailure(
                    f"let {g.name!r} @ {g.path} 的刚性变量逃逸进环境："
                    f"{sorted(v.ident for v in leaked)}"
                )

    return RefCheckReport(
        constraints_checked=len(result.constraints),
        generalizations_checked=len(result.generalizations),
        substitution_size=len(subst),
    )
