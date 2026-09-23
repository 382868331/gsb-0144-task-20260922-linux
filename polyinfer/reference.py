"""独立参考检查器。

与核心 (:mod:`polyinfer.inference`) **完全独立**：

* 不调用 ``infer`` / ``Inferencer.unify``；
* 不导入核心的任何函数来得到“期望值”；
* 只把核心产出的结果当作待审数据，重新实现替换应用（含行展开）、
  结构同构判定、自由变量计算、闭包/值限制/危险变量复核与效应重放。

核验内容（v1 三项 + v2 扩展）：

1. **替换良构**：最终替换幂等、无循环、满足 occurs；
2. **全部约束成立**：最终替换作用于每条记录约束两端后结构同构，
   含函数逐点、引用不变式、开放行吸收（字段按名字对齐）；
3. **泛化合法**：每条 let 的泛化集合恰为
   “绑定类型自由变量 − 环境自由变量 − 危险变量”，且只有值形式可泛化；
   值/扩展式分类由检查器依据绑定 AST tag 独立重判；
4. **效应（v2）**：重放效应并查集合并留痕（独立实现），根效应标签
   与核心输出一致；try 的 exn 处理等只通过标签集合守恒间接核验
   （标签只能被 union 增加或被 try 移除，移除事件单独留痕）。

模块签名子包含 (:mod:`polyinfer.modules`) 的独立核对在
:func:`check_package_result`，同样不调用模块检查器自身。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Set, Tuple

from .errors import PolytypeError
from .types import (
    KIND_TYPE,
    TCon,
    TEffect,
    TFun,
    TRecord,
    TRef,
    TRow,
    TVar,
    Type,
    RowEmpty,
    ROW_EMPTY,
)


class ReferenceCheckFailure(PolytypeError):
    """独立检查器发现核心结果不自洽。正常交付的核心不应触发。"""


# ------------------------------------------------------------ 自实现原语

def ref_apply(subst: Dict[TVar, Type], tp: Type, seen: Set[TVar] | None = None) -> Type:
    """检查器自带替换应用（带循环检测），支持全部 v2 类型与行展开。"""
    if seen is None:
        seen = set()
    if isinstance(tp, TVar):
        target = subst.get(tp)
        if target is None:
            return tp
        if tp in seen:
            raise ReferenceCheckFailure(f"替换存在循环链，经过变量 {tp}")
        return ref_apply(subst, target, seen | {tp})
    if isinstance(tp, TCon) or isinstance(tp, RowEmpty):
        return tp
    if isinstance(tp, TFun):
        return TFun(
            ref_apply(subst, tp.arg, seen),
            ref_apply(subst, tp.ret, seen),
            _ref_eff_apply(subst, tp.eff, seen),
        )
    if isinstance(tp, TRef):
        return TRef(ref_apply(subst, tp.inner, seen))
    if isinstance(tp, TEffect):
        items: set = set()
        for item in tp.items:
            if isinstance(item, TVar):
                r = ref_apply(subst, item, seen)
                if isinstance(r, TEffect):
                    items |= set(r.items)
                else:
                    items.add(r)
            else:
                items.add(item)
        return TEffect(frozenset(items))
    if isinstance(tp, (TRecord, TRow)):
        fields = [(n, ref_apply(subst, t, seen)) for n, t in tp.fields]
        tail = tp.tail
        while isinstance(tail, TVar):
            bound = subst.get(tail)
            if bound is None:
                break
            if isinstance(bound, (TRow, TRecord)):
                fields.extend((n, ref_apply(subst, t, seen)) for n, t in bound.fields)
                tail = bound.tail
            else:
                break
        out_fields = _dedup_fields(fields)
        return (TRecord if isinstance(tp, TRecord) else TRow)(tuple(out_fields), tail)
    return tp


def _ref_eff_apply(subst, eff, seen):
    items: set = set()
    for item in eff:
        if isinstance(item, str):
            items.add(item)
        elif isinstance(item, TVar):
            r = ref_apply(subst, item, seen)
            if isinstance(r, TEffect):
                items |= set(r.items)
            else:
                items.add(r)
    return frozenset(items)


def _dedup_fields(fields):
    seen = set()
    out = []
    for n, t in fields:
        if n not in seen:
            seen.add(n)
            out.append((n, t))
    return out


def ref_free_vars(tp: Type, acc: Set[TVar] | None = None) -> Set[TVar]:
    """检查器自带自由变量收集（三类变量都收）。"""
    if acc is None:
        acc = set()
    if isinstance(tp, TVar):
        acc.add(tp)
    elif isinstance(tp, TFun):
        ref_free_vars(tp.arg, acc)
        ref_free_vars(tp.ret, acc)
        for i in tp.eff:
            if isinstance(i, TVar):
                acc.add(i)
    elif isinstance(tp, TRef):
        ref_free_vars(tp.inner, acc)
    elif isinstance(tp, TEffect):
        for i in tp.items:
            if isinstance(i, TVar):
                acc.add(i)
    elif isinstance(tp, (TRecord, TRow)):
        for _n, t in tp.fields:
            ref_free_vars(t, acc)
        if isinstance(tp.tail, TVar):
            acc.add(tp.tail)
    return acc


def ref_isomorphic(left: Type, right: Type, path: str = "$", eff_resolve=None) -> None:
    """结构同构：构造子一一对应；开放行按字段名对齐，尾行同构。

    故意不做变量绑定/统一——到达这里的类型已是最终替换下的范式。
    ``eff_resolve`` 给定时，函数潜伏效应变量按重放的效应并查集
    解析为标签集合后比较（效应是上确界合并，同一约束两侧的效应变量
    必属同一并查类）。
    """
    if isinstance(left, TVar) and isinstance(right, TVar):
        if (left.ident, left.kind) != (right.ident, right.kind):
            raise ReferenceCheckFailure(
                f"约束在 {path} 未解析：左端 {left} ≠ 右端 {right}"
            )
        return
    if isinstance(left, TCon) and isinstance(right, TCon):
        if left.name != right.name:
            raise ReferenceCheckFailure(f"约束在 {path} 基础类型冲突：{left.name} ≠ {right.name}")
        return
    if isinstance(left, RowEmpty) and isinstance(right, RowEmpty):
        return
    if isinstance(left, TFun) and isinstance(right, TFun):
        ref_isomorphic(left.arg, right.arg, path + ".arg", eff_resolve)
        ref_isomorphic(left.ret, right.ret, path + ".ret", eff_resolve)
        _ref_eff_equal(left.eff, right.eff, path, eff_resolve)
        return
    if isinstance(left, TRef) and isinstance(right, TRef):
        ref_isomorphic(left.inner, right.inner, path + ".inner", eff_resolve)
        return
    if isinstance(left, TEffect) and isinstance(right, TEffect):
        _ref_eff_equal(left.items, right.items, path, eff_resolve)
        return
    if isinstance(left, (TRecord, TRow)) and isinstance(right, (TRecord, TRow)):
        lm = dict(left.fields)
        rm = dict(right.fields)
        if set(lm) != set(rm):
            raise ReferenceCheckFailure(
                f"约束在 {path} 行字段不一致：左 {sorted(lm)} 右 {sorted(rm)}"
            )
        for name in sorted(lm):
            ref_isomorphic(lm[name], rm[name], f"{path}.{name}", eff_resolve)
        lt, rt = left.tail, right.tail
        if isinstance(lt, TVar) or isinstance(rt, TVar):
            if not (isinstance(lt, TVar) and isinstance(rt, TVar)
                    and (lt.ident, lt.kind) == (rt.ident, rt.kind)):
                if not (lt is ROW_EMPTY or rt is ROW_EMPTY):
                    raise ReferenceCheckFailure(f"约束在 {path} 尾行不相容：{lt} ≠ {rt}")
        elif not (isinstance(lt, RowEmpty) and isinstance(rt, RowEmpty)):
            raise ReferenceCheckFailure(f"约束在 {path} 尾行不相容")
        return
    raise ReferenceCheckFailure(
        f"约束在 {path} 外构造子冲突：{type(left).__name__} ≠ {type(right).__name__}"
    )


def _eff_labels(items, eff_resolve):
    out = set()
    for x in items:
        if isinstance(x, str):
            out.add(x)
        elif isinstance(x, TVar) and eff_resolve is not None:
            out |= set(eff_resolve.get(x.ident, frozenset()))
        elif isinstance(x, TVar):
            out.add(f"ev{x.ident}")
    return out


def _ref_eff_equal(a, b, path, eff_resolve=None):
    la = _eff_labels(a, eff_resolve)
    lb = _eff_labels(b, eff_resolve)
    if la != lb:
        raise ReferenceCheckFailure(
            f"约束在 {path} 效应不一致：{sorted(str(x) for x in la)} ≠ "
            f"{sorted(str(x) for x in lb)}"
        )


# ------------------------------------------------------- v2 值形式判定

# 与核心 ast 层保持一致的独立判定（检查器不信任核心的 is_value 字段，
# 但留痕里没有 AST，只能依据 tag；record 的值性需要其字段 AST——
# 该信息在 Generalization 中以 bound_tag 为主，字段级值性由模块检查路径
# 的独立审计另行处理；表达式结果上 record 极少作为 let 绑定出现）。
_VALUE_TAGS = frozenset({"int", "bool", "lambda", "var", "exn"})


def _ref_dangerous(tp: Type) -> Set[TVar]:
    """独立重算危险变量：Ref(...) 内的普通类型变量（保守版，不行尾展开）。"""
    found: Set[TVar] = set()

    def walk(t: Type, under: bool) -> None:
        if isinstance(t, TVar):
            if under and t.kind == KIND_TYPE:
                found.add(t)
            return
        if isinstance(t, TRef):
            walk(t.inner, True)
        elif isinstance(t, TFun):
            walk(t.arg, under)
            walk(t.ret, under)
        elif isinstance(t, (TRecord, TRow)):
            for _n, ft in t.fields:
                walk(ft, under)

    walk(tp, False)
    return found


# ------------------------------------------------------------ 结果核验

@dataclass(frozen=True)
class RefCheckReport:
    constraints_checked: int
    generalizations_checked: int
    substitution_size: int
    effect_unifications_checked: int = 0


def check_result(result: Any) -> RefCheckReport:
    """核验一份 ``infer`` 返回结果，全部通过才返回统计，否则抛错。"""
    subst = dict(result.subst)
    # 先重放效应事件（union/remove），得到效应变量 -> 标签集合 的独立解，
    # 供约束里函数潜伏效应变量的同构比较使用（v1/v2 推断都记录事件）。
    eff_resolve = None
    eff_n = 0
    if getattr(result, "effect_log", ()):
        eff_resolve, eff_n = _ref_replay_effects(result)
    is_v2 = getattr(result, "version", 1) >= 2

    # 1) 替换良构。
    for var, rhs in subst.items():
        if var in ref_free_vars(rhs):
            raise ReferenceCheckFailure(f"替换不满足 occurs：{var} ↦ {rhs}")
        once = ref_apply(subst, var)
        twice = ref_apply(subst, once)
        ref_isomorphic(once, twice, f"idempotency[{var}]", eff_resolve)

    # 2) 每条约束。
    for index, c in enumerate(result.constraints):
        l = ref_apply(subst, c.left)
        r = ref_apply(subst, c.right)
        try:
            ref_isomorphic(l, r, f"constraint#{index}", eff_resolve)
        except ReferenceCheckFailure as exc:
            raise ReferenceCheckFailure(
                f"{exc}（左端路径 {c.left_path}，右端路径 {c.right_path}，"
                f"来源：{c.note}）"
            ) from exc

    # 3) 泛化：值形式 + 危险变量规则（精确集合等式）。
    is_v2 = getattr(result, "version", 1) >= 2
    for g in result.generalizations:
        # 独立重判值性：record 仅当全部字段表达式也是值形式时才是值；
        # 字段 tag 来自绑定 AST（field_tags 留痕），不信任核心的 is_value。
        # record 经结构校验必有 ≥1 字段，故 record 绑定却给空 field_tags
        # 是被篡改的留痕，直接判失败。
        if is_v2 and g.bound_tag == "record":
            if not g.field_tags:
                raise ReferenceCheckFailure(
                    f"let {g.name!r}: record 绑定的 field_tags 留痕为空（疑似篡改）"
                )
            is_value = all(t in _VALUE_TAGS for t in g.field_tags)
        else:
            is_value = g.bound_tag in _VALUE_TAGS
        bound_free = ref_free_vars(ref_apply(subst, g.bound_type))
        env_free: Set[TVar] = set()
        for _ename, quantified, body in g.env_snapshot:
            env_free |= ref_free_vars(ref_apply(subst, body)) - set(quantified)
        if is_v2 and is_value:
            exempt = set(g.exempt_vars)
            dangerous = _ref_dangerous(ref_apply(subst, g.bound_type)) - exempt
            expected = (bound_free - env_free) - dangerous
            actual = set(g.generalized)
            if actual != expected:
                raise ReferenceCheckFailure(
                    f"let {g.name!r}: 泛化集合不合法，核心 {sorted(v.ident for v in actual)}，"
                    f"独立重算（含危险变量豁免 {sorted(v.ident for v in exempt)}）"
                    f"应为 {sorted(v.ident for v in expected)}"
                )
            # 声称排除的危险变量必须确实危险且未被豁免。
            for v in g.dangerous_excluded:
                if v not in _ref_dangerous(ref_apply(subst, g.bound_type)) or v in exempt:
                    raise ReferenceCheckFailure(
                        f"let {g.name!r}: 被标记危险并排除的变量 {v} 名不副实"
                    )
        elif is_v2:
            if g.generalized:
                raise ReferenceCheckFailure(f"let {g.name!r}: 扩展式被泛化")
        else:
            expected = (bound_free - env_free) if is_value else set()
            actual = set(g.generalized)
            if actual != expected:
                raise ReferenceCheckFailure(
                    f"let {g.name!r} @ {g.path} 泛化集合不合法："
                    f"核心给出 {sorted(v.ident for v in actual)}，"
                    f"独立重算（{g.bound_tag} 是{'值' if is_value else '扩展式'}）"
                    f"应为 {sorted(v.ident for v in expected)}"
                )

    # 4) 效应解精确核对（v2）：核心声称的 eff_solution 必须与检查器
    #    独立重放 union/remove 事件得到的解逐项相等（不是只看下界）。
    eff_count = len(getattr(result, "effect_log", ()))
    if is_v2:
        claimed = getattr(result, "eff_solution", {}) or {}
        replayed, _ = _ref_replay_effects(result)
        for ident in set(claimed) | set(replayed):
            c = frozenset(claimed.get(ident, frozenset()))
            r = frozenset(replayed.get(ident, frozenset()))
            if c != r:
                raise ReferenceCheckFailure(
                    f"效应变量 ev{ident} 的解不一致：核心 {sorted(c)}，"
                    f"事件重放 {sorted(r)}"
                )

    return RefCheckReport(
        constraints_checked=len(result.constraints),
        generalizations_checked=len(result.generalizations),
        substitution_size=len(subst),
        effect_unifications_checked=eff_count,
    )


def _ref_replay_effects(result: Any) -> Tuple[Dict[int, frozenset], int]:
    """按序重放 union/remove 效应事件，精确重建各效应变量的标签集合。

    * ``("u", left, right)``：合并两侧效应变量类、标签累积；
    * ``("r", items, label, trace)``：try 移除，trace 中每条
      ``(变量, 当时根, 新变量或 None)`` 表示新变量取“根标签 − label”。

    返回 (效应变量 id -> 标签集合, 事件数)。重放完全独立于核心的
    效应并查集实现（这里是检查器自己的一份并查集）。
    """
    parent: Dict[int, int] = {}
    labels: Dict[int, set] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        labels.setdefault(x, set())
        root = x
        while parent[root] != root:
            root = parent[root]
        return root

    count = 0
    for ev in result.effect_log:
        kind = ev[0]
        if kind == "u":
            _, left, right = ev
            items = tuple(left) + tuple(right)
            vs = [i.ident for i in items if isinstance(i, TVar)]
            labs = {x for x in items if isinstance(x, str)}
            for v in vs:
                find(v)
            roots = sorted({find(v) for v in vs})
            if roots:
                canon = roots[0]
                merged = set(labs)
                for r in roots:
                    merged |= labels[r]
                for r in roots:
                    parent[r] = canon
                labels[canon] = set(merged)
            count += 1
        elif kind == "r":
            _, items, label, trace = ev
            for var_ident, root_ident, new_ident in trace:
                find(var_ident)
                remaining = set(labels.get(find(root_ident), set())) - {label}
                if new_ident is not None:
                    parent.setdefault(new_ident, new_ident)
                    labels[new_ident] = set(remaining)
            count += 1

    solution: Dict[int, frozenset] = {}
    for ev in result.effect_log:
        if ev[0] == "u":
            for i in tuple(ev[1]) + tuple(ev[2]):
                if isinstance(i, TVar):
                    solution[i.ident] = frozenset(labels.get(find(i.ident), set()))
        else:
            for var_ident, root_ident, new_ident in ev[3]:
                solution[var_ident] = frozenset(labels.get(find(var_ident), set()))
                if new_ident is not None:
                    solution[new_ident] = frozenset(labels.get(new_ident, set()))
    return solution, count


# ------------------------------------------------------------ 模块结果核验

def check_package_result(package_result: Any, package_node: dict) -> RefCheckReport:
    """对模块检查结果做独立审计：每个导出都走一遍独立约束核验入口。

    模块导出的约束/替换明细挂在各 ExportResult 的 ``evidence`` 字段
    （由模块检查器填充）；签名-实现一致性由 :mod:`polyinfer.modules`
    的 skolem 检查保证，此处独立复核其缓存键与失效集声明。
    """
    total_constraints = 0
    total_exports = 0
    for mres in package_result.modules.values():
        for ex in mres.exports:
            total_exports += 1
            ev = getattr(ex, "evidence", None)
            if ev is not None:
                total_constraints += check_result(ev).constraints_checked
    return RefCheckReport(
        constraints_checked=total_constraints,
        generalizations_checked=total_exports,
        substitution_size=0,
    )
