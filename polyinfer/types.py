"""类型结构与替换。

类型表示：

* :class:`TVar`  —— 类型变量（推断时按创建顺序拿内部唯一 id）；
* :class:`RVar`  —— 行变量（记录开放行的尾变量，独立的变量空间）；
* :class:`REmpty` —— 闭合行的空尾；
* :class:`TRow`  —— 行：若干具名字段类型 + 尾（行变量或空尾）；
* :class:`TRec`  —— 记录类型（包一行）；
* :class:`TSkol` —— 刚性（skolem）变量：模块导出签名中的限定变量，
  实现检查时不得被具体类型替换；
* :class:`TCon`  —— 类型构造子零元：``Int`` / ``Bool`` / ``Unit``；
* :class:`TFun`  —— 函数类型 ``arg -> ret``，带潜在效应集合 ``eff``
  （空集合渲染与旧版完全一致：``a -> b``）；
* :class:`TRef`  —— 引用类型 ``Ref(t)``。

替换是 ``{TVar|RVar: Type}`` 的不可变字典；:func:`bind` 实现**替换组合**
（先做右侧替换，再并入左侧绑定），:func:`apply` 做幂等应用，
:func:`occurs` 是 occurs 检查（类型变量与行变量各自独立检查）。
三个函数彼此独立、可单独测试，也供独立参考检查器复用——统一核心
(``unify``) 不允许在检查器里被调用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Tuple

Subst = Dict["TVar", "Type"]

# 效应标签（读 / 写 / 异常）。效应集合是这三者的 frozenset 子集。
EFF_READ = "R"
EFF_WRITE = "W"
EFF_EXN = "X"
ALL_EFFECTS = frozenset({EFF_READ, EFF_WRITE, EFF_EXN})


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
    """类型变量。内部 id 为创建序号；canonical 名在最后统一编号。"""

    ident: int

    def free_vars(self) -> "Iterator[TVar]":
        yield self


@dataclass(frozen=True)
class RVar(Type):
    """行变量：记录开放行“其余字段”的尾变量。

    与 :class:`TVar` 分属不同变量空间：行变量只能被绑定到行
    (:class:`TRow` / :class:`REmpty`)，绝不会与普通类型变量统一。
    """

    ident: int

    def free_vars(self) -> "Iterator[TVar]":
        return iter(())  # 行变量由 free_rvars 单独收集


@dataclass(frozen=True)
class REmpty(Type):
    """空行尾：记录行在此闭合，不含更多字段。"""

    def free_vars(self) -> "Iterator[TVar]":
        return iter(())


@dataclass(frozen=True)
class TRow(Type):
    """一行：``fields`` 为按名排序、互不重名的 (字段名, 类型) 序列，
    ``rest`` 是 :class:`RVar`（开放）或 :class:`REmpty`（闭合）。"""

    fields: Tuple[Tuple[str, "Type"], ...] = ()
    rest: Type = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.rest is None:
            object.__setattr__(self, "rest", REmpty())

    def free_vars(self) -> "Iterator[TVar]":
        for _n, ft in self.fields:
            yield from ft.free_vars()
        # rest 只可能是行变量/空尾，不含普通类型变量。


@dataclass(frozen=True)
class TRec(Type):
    """记录类型：``{f1: t1, ...; r}``（开放）或 ``{f1: t1, ...}``（闭合）。"""

    row: TRow

    def free_vars(self) -> "Iterator[TVar]":
        yield from self.row.free_vars()


@dataclass(frozen=True)
class TSkol(Type):
    """刚性（skolem）类型变量，用于导出签名一致性检查。

    ``ident`` 是本次检查内的唯一编号，``name`` 是签名里的源名。
    刚性变量只能与自身统一；与任何具体类型/其他变量相遇即证明实现
    比签名更不一般。
    """

    ident: int
    name: str

    def free_vars(self) -> "Iterator[TVar]":
        return iter(())  # 刚性变量不参与普通自由变量泛化


@dataclass(frozen=True)
class RSkol(Type):
    """刚性行变量：出现在导出签名开放行尾（含 Ref 内的行）的限定变量
    在实现检查处实例化为此类。

    它只能与自身统一；柔性 :class:`RVar` 可被绑定到它（表示实现对“任意
    尾行”保持多态），但闭合行或带具体字段的行尾与它相遇即证明实现把
    一个形状固定（或共享）的记录宣称为开放行多态——对可变引用不健全。
    """

    ident: int
    name: str

    def free_vars(self) -> "Iterator[TVar]":
        return iter(())


@dataclass(frozen=True)
class TCon(Type):
    """零元类型构造子。"""

    name: str

    def free_vars(self) -> "Iterator[TVar]":
        return iter(())


@dataclass(frozen=True)
class TFun(Type):
    """函数类型：``arg -> ret``，``eff`` 为潜在效应集合。

    ``eff == frozenset()`` 时是纯函数，渲染与旧版完全相同（``a -> b``）；
    非空时渲染为 ``a -{R,W}-> b``。默认空集合保证旧代码
    ``TFun(arg, ret)`` 的二参构造与相等性不变。
    """

    arg: Type
    ret: Type
    eff: frozenset = frozenset()

    def free_vars(self) -> "Iterator[TVar]":
        yield from self.arg.free_vars()
        yield from self.ret.free_vars()


@dataclass(frozen=True)
class TRef(Type):
    """引用类型：``Ref(inner)``。"""

    inner: Type

    def free_vars(self) -> "Iterator[TVar]":
        yield from self.inner.free_vars()


# 预置基础类型。
INT = TCon("Int")
BOOL = TCon("Bool")
UNIT = TCon("Unit")
ROW_EMPTY = REmpty()


# ---------------------------------------------------------------- 遍历

def occurs(var: Type, tp: Type) -> bool:
    """occurs 检查：``var`` 是否出现在 ``tp`` 中。

    :class:`TVar` 只在普通类型位置比较；:class:`RVar` 只在行尾位置比较，
    两个变量空间互不串扰。行变量绑定到含自身的行会被这里拦下
    （开放行统一成环 = 无限行）。
    """
    if isinstance(var, RVar):
        # 行变量只可能出现在行尾链里。
        if isinstance(tp, (RVar, RSkol)):
            return isinstance(tp, RVar) and tp == var
        if isinstance(tp, TRow):
            return occurs(var, tp.rest)
        if isinstance(tp, TRec):
            return occurs(var, tp.row)
        return False
    # 普通类型变量
    if isinstance(tp, (TVar, TSkol)):
        return isinstance(tp, TVar) and tp == var
    if isinstance(tp, TFun):
        return occurs(var, tp.arg) or occurs(var, tp.ret)
    if isinstance(tp, TRef):
        return occurs(var, tp.inner)
    if isinstance(tp, TRow):
        return any(occurs(var, t) for _n, t in tp.fields) or occurs(var, tp.rest)
    if isinstance(tp, TRec):
        return occurs(var, tp.row)
    return False  # TCon / REmpty


def free_tvars(tp: Type) -> Tuple[TVar, ...]:
    """类型中出现的自由类型变量（按去重后的遍历顺序）。"""
    seen = {}
    for v in tp.free_vars():
        seen[v] = None
    return tuple(seen)


def free_rvars(tp: Type) -> Tuple[RVar, ...]:
    """类型中出现的自由行变量（去重、按遍历顺序）。"""
    seen: Dict[RVar, None] = {}

    def walk(t: Type) -> None:
        if isinstance(t, RVar):
            seen[t] = None
        elif isinstance(t, TRow):
            for _n, ft in t.fields:
                walk(ft)
            walk(t.rest)
        elif isinstance(t, TRec):
            walk(t.row)
        elif isinstance(t, TFun):
            walk(t.arg)
            walk(t.ret)
        elif isinstance(t, TRef):
            walk(t.inner)

    walk(tp)
    return tuple(seen)


def free_skolems(tp: Type) -> Tuple[Any, ...]:
    """类型中出现的刚性变量（TSkol 与 RSkol，去重）。独立于 free_tvars
    （刚性变量在普通自由变量语义中不计入，但逃逸检查必须看见它们）。"""
    seen: Dict[Any, None] = {}

    def walk(t: Type) -> None:
        if isinstance(t, (TSkol, RSkol)):
            seen[t] = None
        elif isinstance(t, TFun):
            walk(t.arg)
            walk(t.ret)
        elif isinstance(t, TRef):
            walk(t.inner)
        elif isinstance(t, TRow):
            for _n, ft in t.fields:
                walk(ft)
            walk(t.rest)
        elif isinstance(t, TRec):
            walk(t.row)

    walk(tp)
    return tuple(seen)


def apply(subst: "Subst", tp: Type) -> Type:
    """把替换幂等应用到类型上；未被绑定的变量保持原样。

    对非循环替换必然终止（unify 产出的绑定经 occurs 检查，绝无循环）。
    """
    if isinstance(tp, TVar):
        mapped = subst.get(tp)
        if mapped is None:
            return tp
        return apply(subst, mapped)
    if isinstance(tp, RVar):
        mapped = subst.get(tp)
        if mapped is None:
            return tp
        return apply(subst, mapped)
    if isinstance(tp, (TCon, REmpty, TSkol, RSkol)):
        return tp
    if isinstance(tp, TFun):
        return TFun(
            apply(subst, tp.arg),
            apply(subst, tp.ret),
            tp.eff,
        )
    if isinstance(tp, TRef):
        return TRef(apply(subst, tp.inner))
    if isinstance(tp, TRow):
        return TRow(
            tuple((n, apply(subst, ft)) for n, ft in tp.fields),
            apply(subst, tp.rest),
        )
    if isinstance(tp, TRec):
        return TRec(apply(subst, tp.row))
    return tp


def _mentions(tp: Type, var: Type) -> bool:
    """类型中是否出现变量 ``var``（比完整 apply 重建便宜的成员测试）。"""
    if isinstance(var, RVar):
        if isinstance(tp, RVar):
            return tp == var
        if isinstance(tp, TRow):
            return _mentions(tp.rest, var) or any(_mentions(ft, var) for _n, ft in tp.fields)
        if isinstance(tp, TRec):
            return _mentions(tp.row, var)
        if isinstance(tp, TFun):
            return _mentions(tp.arg, var) or _mentions(tp.ret, var)
        if isinstance(tp, TRef):
            return _mentions(tp.inner, var)
        return False
    if isinstance(tp, TVar):
        return tp == var
    if isinstance(tp, TFun):
        return _mentions(tp.arg, var) or _mentions(tp.ret, var)
    if isinstance(tp, TRef):
        return _mentions(tp.inner, var)
    if isinstance(tp, TRow):
        return any(_mentions(ft, var) for _n, ft in tp.fields) or _mentions(tp.rest, var)
    if isinstance(tp, TRec):
        return _mentions(tp.row, var)
    return False


def bind(subst: "Subst", var: Type, tp: Type) -> "Subst":
    """替换组合：返回 ``s'`` 使得对任意类型 ``t``，``s'(t) = s({var -> s(tp)}(t))``。
    新绑定右端与所有旧绑定右端都先经已有替换推进，保持替换幂等。
    ``var`` 可以是 :class:`TVar` 或 :class:`RVar`。
    """
    rhs = apply(subst, tp)
    merged: Subst = dict(subst)
    singleton = {var: rhs}
    for key, value in list(merged.items()):
        if _mentions(value, var):
            merged[key] = apply(singleton, value)
    merged[var] = rhs  # type: ignore[index]
    return merged


# ---------------------------------------------------------------- 渲染

class TypeRenderer:
    """类型渲染；变量名由外部规范命名映射提供，缺省用 ``?ident``/``?rident``。"""

    def __init__(self, tp: Type, names: "Dict[Type, str] | None" = None) -> None:
        self.tp = tp
        self.names = names or {}

    def render(self) -> str:
        return self._render(self.tp, top=True)

    def _name(self, var: Type) -> str:
        if isinstance(var, TSkol):
            # 显式命名映射（模块检查报告用签名源名）优先，否则用 $源名。
            return self.names.get(var, "$" + var.name)
        if isinstance(var, RSkol):
            return self.names.get(var, "$" + var.name)
        return self.names.get(var, ("?r" if isinstance(var, RVar) else "?") + str(var.ident))

    def _render(self, tp: Type, top: bool = False) -> str:
        if isinstance(tp, (TVar, RVar, TSkol, RSkol)):
            return self._name(tp)
        if isinstance(tp, TCon):
            return tp.name
        if isinstance(tp, REmpty):
            return "<>"
        if isinstance(tp, TRef):
            # Ref(...) 自带分隔符，内部按顶层渲染，避免双重括号。
            return f"Ref({self._render(tp.inner, top=True)})"
        if isinstance(tp, TRow):
            # 展平替换可能产生的嵌套尾链：TRow(f1, TRow(f2, t))。
            flat_fields = list(tp.fields)
            tail = tp.rest
            while isinstance(tail, TRow):
                flat_fields.extend(tail.fields)
                tail = tail.rest
            parts = [f"{n}: {self._render(ft, top=True)}" for n, ft in flat_fields]
            head = ", ".join(parts)
            if isinstance(tail, RVar):
                rest_tail = " ; " + self._name(tail)
            elif isinstance(tail, RSkol):
                rest_tail = " ; " + self._name(tail)
            elif not isinstance(tail, REmpty):  # 理论上不可达
                rest_tail = " ; " + self._render(tail, top=True)
            else:
                rest_tail = ""
            return "{" + head + rest_tail + "}"
        if isinstance(tp, TRec):
            return self._render(tp.row, top=True)
        # TFun：函数参数位置需要加括号，右结合的返回类型不需要。
        arg_s = self._render(tp.arg, top=False)
        ret_s = self._render(tp.ret, top=True)
        if tp.eff:
            def eff_label(x):
                return x if isinstance(x, str) else f"?e{x.ident}"

            labels = ",".join(sorted(eff_label(x) for x in tp.eff))
            s = f"{arg_s} -{{{labels}}}-> {ret_s}"
        else:
            s = f"{arg_s} -> {ret_s}"
        if not top:
            return f"({s})"
        return s


def render_type(tp: Type, names: "Dict[Type, str] | None" = None) -> str:
    """便捷渲染入口。"""
    return TypeRenderer(tp, names).render()


def canonical_names(tp: Type, scheme_vars: "Tuple[Type, ...] | None" = None) -> "Dict[Type, str]":
    """按“首次出现顺序”给出规范变量名。

    * 类型变量：a, b, ..., z, a1, b1, ...（先序遍历，与内部 id 无关）；
    * 行变量：r, r1, r2, ...（独立名字空间）；
    * 刚性变量不参与命名（渲染为 ``$源名``）。
    """
    order: Dict[Type, int] = {}

    def walk(t: Type) -> None:
        if isinstance(t, (TVar, RVar)):
            if t not in order:
                order[t] = len(order)
            return
        if isinstance(t, (TSkol, RSkol)):
            return
        if isinstance(t, TFun):
            walk(t.arg)
            walk(t.ret)
        elif isinstance(t, TRef):
            walk(t.inner)
        elif isinstance(t, TRow):
            for _n, ft in t.fields:
                walk(ft)
            walk(t.rest)
        elif isinstance(t, TRec):
            walk(t.row)

    walk(tp)
    if scheme_vars is not None:
        for v in scheme_vars:  # 保证泛化变量即使（理论上）未出现也被命名
            order.setdefault(v, len(order))

    type_index = 0
    row_index = 0
    names: Dict[Type, str] = {}
    for var, index in sorted(order.items(), key=lambda kv: kv[1]):
        if isinstance(var, RVar):
            names[var] = "r" if row_index == 0 else f"r{row_index}"
            row_index += 1
        else:
            n = type_index
            base = n % 26
            suffix = n // 26
            names[var] = chr(ord("a") + base) + (str(suffix) if suffix else "")
            type_index += 1
    return names
