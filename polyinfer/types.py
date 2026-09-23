"""类型结构与替换。

类型表示（v2 在 v1 基础上**只增不改**，旧构造方式与字符串渲染保持不变）：

* :class:`TVar`  —— 类型变量，``kind`` 区分普通类型变量(``KIND_TYPE``)
  与行变量(``KIND_ROW``)；默认仍是普通类型变量；
* :class:`TCon`  —— 零元类型构造子：``Int`` / ``Bool`` / ``Unit`` / ``Exn``；
* :class:`TFun`  —— 函数类型 ``arg -> ret``，另有效应集合字段 ``eff``
  （``frozenset[str]``；空集表示纯函数，渲染与 v1 完全一致）。
  集合中的元素要么是效应标签（``read`` / ``write`` / ``exn``），
  要么是待解的效应元变量（``ev<n>``，仅推断内部使用，求解后被标签集合替换）；
* :class:`TRef`  —— 引用类型 ``Ref(t)``（不变式）；
* :class:`TRecord` —— 记录类型 ``{字段: 类型, ... | 尾行}``，
  尾行是行变量（开放行）或 :data:`ROW_EMPTY`（闭行）；
* :class:`TRow` —— 行变量被绑定后的具体行：若干字段 + 尾行，
  只作为替换右端出现。

替换是 ``{TVar: Type}`` 的不可变字典（普通类型变量与行变量共用一张表，
由 kind 区分）；:func:`bind` 实现**替换组合**，:func:`apply` 幂等应用，
:func:`occurs` 是两类变量共用的 occurs 检查（开放行成环由此拒绝）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Tuple

Subst = Dict["TVar", "Type"]

KIND_TYPE = 0
KIND_ROW = 1
KIND_EFF = 2

# 三种效应标签的固定次序（渲染与所有确定性输出都用它）。
EFFECT_LABELS: Tuple[str, ...] = ("read", "write", "exn")
EFFECT_SET = frozenset(EFFECT_LABELS)


class Type:
    """所有类型的公共基类，提供结构遍历/比较/渲染。"""

    def free_vars(self) -> "Iterator[TVar]":
        raise NotImplementedError

    def apply(self, subst: "Subst") -> "Type":
        return apply(subst, self)

    def occurs(self, var: "TVar") -> bool:
        return occurs(var, self)

    def __str__(self) -> str:
        return TypeRenderer(self).render()


@dataclass(frozen=True)
class TVar(Type):
    """类型变量。``kind`` 区分普通类型变量与行变量，缺省为普通类型变量。"""

    ident: int
    kind: int = KIND_TYPE

    def free_vars(self) -> "Iterator[TVar]":
        yield self


@dataclass(frozen=True)
class TCon(Type):
    """零元类型构造子。"""

    name: str

    def free_vars(self) -> "Iterator[TVar]":
        return iter(())


@dataclass(frozen=True)
class TFun(Type):
    """函数类型：``arg -> ret``，潜在效应集合 ``eff``（空集=纯函数）。"""

    arg: Type
    ret: Type
    eff: frozenset = frozenset()

    def free_vars(self) -> "Iterator[TVar]":
        yield from self.arg.free_vars()
        yield from self.ret.free_vars()
        for item in self.eff:
            if isinstance(item, TVar):
                yield item


@dataclass(frozen=True)
class TRef(Type):
    """引用类型：``Ref(inner)``。"""

    inner: Type

    def free_vars(self) -> "Iterator[TVar]":
        yield from self.inner.free_vars()


@dataclass(frozen=True)
class RowEmpty(Type):
    """闭行标记（记录没有剩余字段）。"""

    def free_vars(self) -> "Iterator[TVar]":
        return iter(())


ROW_EMPTY = RowEmpty()


@dataclass(frozen=True)
class TEffect(Type):
    """效应集合：标签（``read``/``write``/``exn``）与未决效应变量的集合。

    只作为 :class:`TFun` 的潜在效应字段与 ``KIND_EFF`` 变量的替换右端出现。
    效应变量是普通 :class:`TVar`（``kind=KIND_EFF``），与类型/行变量共用
    同一张替换表，因此 occurs、bind、泛化计数对三类变量是同一套机制。
    """

    items: frozenset = frozenset()

    def free_vars(self) -> "Iterator[TVar]":
        for item in self.items:
            if isinstance(item, TVar):
                yield item


@dataclass(frozen=True)
class TRow(Type):
    """具体行：命名字段的有序序列 + 尾行（行变量或闭行）。

    只作为行变量在替换中的右端出现；记录类型自身直接用 :class:`TRecord`。
    """

    fields: Tuple[Tuple[str, "Type"], ...] = ()
    tail: Any = ROW_EMPTY

    def free_vars(self) -> "Iterator[TVar]":
        for _name, t in self.fields:
            yield from t.free_vars()
        if isinstance(self.tail, TVar):
            yield self.tail


@dataclass(frozen=True)
class TRecord(Type):
    """记录类型：命名字段序列 + 尾行（行变量=开放行，ROW_EMPTY=闭行）。"""

    fields: Tuple[Tuple[str, "Type"], ...]
    tail: Any = ROW_EMPTY

    def free_vars(self) -> "Iterator[TVar]":
        for _name, t in self.fields:
            yield from t.free_vars()
        if isinstance(self.tail, TVar):
            yield self.tail


# 预置基础类型。
INT = TCon("Int")
BOOL = TCon("Bool")
UNIT = TCon("Unit")
EXN = TCon("Exn")


def occurs(var: TVar, tp: Type) -> bool:
    """occurs 检查：``var`` 是否出现在 ``tp`` 中（按 kind 各自成环判定）。"""
    if isinstance(tp, TVar):
        return tp == var
    if isinstance(tp, TFun):
        return occurs(var, tp.arg) or occurs(var, tp.ret)
    if isinstance(tp, TRef):
        return occurs(var, tp.inner)
    if isinstance(tp, TEffect):
        return any(isinstance(i, TVar) and i == var for i in tp.items)
    if isinstance(tp, TRow):
        return any(occurs(var, t) for _n, t in tp.fields) or (
            isinstance(tp.tail, TVar) and tp.tail == var
        )
    if isinstance(tp, TRecord):
        return any(occurs(var, t) for _n, t in tp.fields) or (
            isinstance(tp.tail, TVar) and tp.tail == var
        )
    return False  # TCon / RowEmpty


def free_tvars(tp: Type) -> Tuple[TVar, ...]:
    """类型中出现的自由类型变量（含行变量，按去重后的遍历顺序）。"""
    seen = {}
    for v in tp.free_vars():
        seen[v] = None
    return tuple(seen)


def _flatten_row(
    fields: Tuple[Tuple[str, Type], ...], tail: Any, subst: "Subst"
) -> Tuple[Tuple[Tuple[str, Type], ...], Any]:
    """把 (fields, tail) 沿行替换展开为 (全部字段, 最终未绑定尾行)。

    同名字段若因替换拼接而重复，保留先出现的（unify 阶段已保证不发生；
    这里的去重只是让 apply 对任意无循环替换都是全函数）。
    """
    out: list[Tuple[str, Type]] = list(fields)
    seen = {n for n, _ in out}
    while isinstance(tail, TVar):
        bound = subst.get(tail)
        if bound is None:
            break
        if isinstance(bound, TRow):
            for name, t in bound.fields:
                if name not in seen:
                    seen.add(name)
                    out.append((name, t))
            tail = bound.tail
        else:
            break  # kind 不匹配在 unify 处拦截；apply 保持全函数
    return tuple(out), tail


def apply(subst: "Subst", tp: Type) -> Type:
    """把替换幂等应用到类型上；未被绑定的变量保持原样。"""
    if isinstance(tp, TVar):
        mapped = subst.get(tp)
        if mapped is None:
            return tp
        return apply(subst, mapped)
    if isinstance(tp, TFun):
        return TFun(
            apply(subst, tp.arg),
            apply(subst, tp.ret),
            tp.eff,
        )
    if isinstance(tp, TRef):
        return TRef(apply(subst, tp.inner))
    if isinstance(tp, TEffect):
        items: set = set()
        for item in tp.items:
            if isinstance(item, TVar):
                resolved = apply(subst, item)
                if isinstance(resolved, TEffect):
                    items |= set(resolved.items)
                else:
                    items.add(resolved)
            else:
                items.add(item)
        return TEffect(frozenset(items))
    if isinstance(tp, TRow):
        fields = tuple((n, apply(subst, t)) for n, t in tp.fields)
        resolved, tail = _flatten_row((), tp.tail, subst)
        return TRow(fields + tuple((n, apply(subst, t)) for n, t in resolved), tail)
    if isinstance(tp, TRecord):
        resolved, tail = _flatten_row(
            tuple((n, apply(subst, t)) for n, t in tp.fields),
            tp.tail,
            subst,
        )
        return TRecord(resolved, tail)
    return tp  # TCon / RowEmpty


def bind(subst: "Subst", var: TVar, tp: Type) -> "Subst":
    """替换组合：返回 ``s'`` 使得对任意类型 ``t``，``s'(t) = s({var -> s(tp)}(t))``。"""
    rhs = apply(subst, tp)
    merged: Subst = dict(subst)
    for key, value in list(merged.items()):
        merged[key] = apply({var: rhs}, value)
    merged[var] = rhs
    return merged


def row_fields_map(
    subst: "Subst", tail: Any, present: Tuple[Tuple[str, Type], ...]
) -> Tuple[Dict[str, Type], Any]:
    """查看一个记录在当前替换下的 (字段名 -> 类型) 映射与未绑定尾行。"""
    fields, final_tail = _flatten_row(present, tail, subst)
    return dict(fields), final_tail


class TypeRenderer:
    """类型渲染；变量名由外部规范命名映射提供，缺省用 ``?ident`` / ``R?ident``。

    效应集合：``eff_map`` 给定时把 ``ev<n>`` 元变量解析成标签集合；
    纯函数（空效应）渲染与 v1 完全相同。
    """

    def __init__(
        self,
        tp: Type,
        names: "Dict[TVar, str] | None" = None,
        eff_map: "Dict[int, frozenset] | None" = None,
    ) -> None:
        self.tp = tp
        self.names = names or {}
        self.eff_map = eff_map or {}

    def render(self) -> str:
        return self._render(self.tp, top=True)

    def _name(self, var: TVar) -> str:
        if var in self.names:
            return self.names[var]
        prefix = ("R?" if var.kind == KIND_ROW else ("E?" if var.kind == KIND_EFF else "?"))
        return f"{prefix}{var.ident}"

    def _eff_labels(self, eff: frozenset) -> "set[str]":
        labels: set = set()
        for token in eff:
            if isinstance(token, str):
                labels.add(token)
            elif isinstance(token, TVar):
                resolved = self.eff_map.get(token.ident)
                if resolved is None:
                    labels.add(f"ev{token.ident}")
                else:
                    labels |= set(resolved)
        return labels

    def _eff_arrow(self, eff: frozenset) -> str:
        labels = self._eff_labels(eff)
        if not labels:
            return "->"
        ordered = ", ".join(l for l in EFFECT_LABELS if l in labels)
        extra = sorted(l for l in labels if l not in EFFECT_SET)
        for l in extra:
            ordered = f"{ordered}, {l}" if ordered else l
        return f"-{{{ordered}}}->"

    def _render(self, tp: Type, top: bool = False) -> str:
        if isinstance(tp, TVar):
            return self._name(tp)
        if isinstance(tp, TCon):
            return tp.name
        if isinstance(tp, RowEmpty):
            return "<>"
        if isinstance(tp, TEffect):
            labels = self._eff_labels(tp.items)
            return "{" + ", ".join(sorted(labels)) + "}"
        if isinstance(tp, TRef):
            return f"Ref({self._render(tp.inner, top=True)})"
        if isinstance(tp, (TRecord, TRow)):
            parts = [
                f"{name}: {self._render(ft, top=True)}"
                for name, ft in sorted(tp.fields, key=lambda p: p[0])
            ]
            if isinstance(tp.tail, TVar):
                parts.append(f"| {self._name(tp.tail)}")
            return "{" + ", ".join(parts) + "}"
        if isinstance(tp, TFun):
            s = (
                f"{self._render(tp.arg, top=False)} "
                f"{self._eff_arrow(tp.eff)} {self._render(tp.ret, top=True)}"
            )
            if not top:
                return f"({s})"
            return s
        return repr(tp)


def render_type(
    tp: Type,
    names: "Dict[TVar, str] | None" = None,
    eff_map: "Dict[str, frozenset] | None" = None,
) -> str:
    return TypeRenderer(tp, names, eff_map).render()


def canonical_names(tp: Type, scheme_vars: "Tuple[TVar, ...] | None" = None) -> "Dict[TVar, str]":
    """按“首次出现顺序”给出规范变量名。

    普通类型变量：``a, b, ..., z, a1, b1, ...``；
    行变量使用独立计数：``R0, R1, ...``（大写 R 前缀，绝不与普通变量混淆）。
    遍历顺序为类型结构先序：函数先参数后返回；记录按字段名字典序后到尾行。
    """
    type_order: dict = {}
    row_order: dict = {}
    eff_order: dict = {}

    def walk(t: Type) -> None:
        if isinstance(t, TVar):
            table = (
                row_order if t.kind == KIND_ROW
                else (eff_order if t.kind == KIND_EFF else type_order)
            )
            if t not in table:
                table[t] = len(table)
            return
        if isinstance(t, TFun):
            walk(t.arg)
            walk(t.ret)
        elif isinstance(t, TRef):
            walk(t.inner)
        elif isinstance(tp_ := t, (TRecord, TRow)):
            for _n, ft in sorted(tp_.fields, key=lambda p: p[0]):
                walk(ft)
            if isinstance(tp_.tail, TVar):
                walk(tp_.tail)
        elif isinstance(t, TEffect):
            for item in t.items:
                if isinstance(item, TVar):
                    walk(item)

    walk(tp)
    if scheme_vars is not None:
        for v in scheme_vars:
            table = (
                row_order if v.kind == KIND_ROW
                else (eff_order if v.kind == KIND_EFF else type_order)
            )
            table.setdefault(v, len(table))

    def letter(n: int) -> str:
        base = n % 26
        suffix = n // 26
        return chr(ord("a") + base) + (str(suffix) if suffix else "")

    names = {var: letter(index) for var, index in type_order.items()}
    names.update({var: f"R{index}" for var, index in row_order.items()})
    names.update({var: f"e{index}" for var, index in eff_order.items()})
    return names
