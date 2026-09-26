"""类型结构与替换。

类型的四元表示：

* :class:`TVar`  —— 类型变量（推断时按创建顺序拿内部唯一 id）；
* :class:`TCon`  —— 类型构造子零元：``Int`` / ``Bool`` / ``Unit``；
* :class:`TFun`  —— 函数类型 ``arg -> ret``；
* :class:`TRef`  —— 引用类型 ``Ref(t)``。

替换是 ``{TVar: Type}`` 的不可变字典；:func:`bind` 实现**替换组合**
（先做右侧替换，再并入左侧绑定），:func:`apply` 做幂等应用，
:func:`occurs` 是 occurs 检查。三个函数彼此独立、可单独测试，
也供独立参考检查器复用——统一核心 (``unify``) 不允许在检查器里被调用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Tuple

Subst = Dict["TVar", "Type"]


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


@dataclass(frozen=True, eq=False)
class RigidTVar(TVar):
    """显式签名里 ``forall`` 引入的刚性（skolem）类型变量。

    刚性变量在其签名检查期间只能与自身统一：任何把它实例化成具体
    类型、与另一个刚性变量合并、或经外层推断变量“逃逸”出绑定的
    做法都必须被拒绝。``eq=False`` 使它绝不与普通 :class:`TVar`
    判等，避免被替换映射误吞。
    """

    name: str = "?"

    def __eq__(self, other: object) -> bool:  # 只允许自身（按 ident）相等
        return isinstance(other, RigidTVar) and other.ident == self.ident

    def __hash__(self) -> int:
        return hash((RigidTVar, self.ident))


@dataclass(frozen=True)
class TCon(Type):
    """零元类型构造子。"""

    name: str

    def free_vars(self) -> "Iterator[TVar]":
        return iter(())


@dataclass(frozen=True)
class TFun(Type):
    """函数类型：``arg -> ret``。"""

    arg: Type
    ret: Type

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


def occurs(var: TVar, tp: Type) -> bool:
    """occurs 检查：``var`` 是否出现在 ``tp`` 中。"""
    if isinstance(tp, TVar):
        return tp == var
    if isinstance(tp, TFun):
        return occurs(var, tp.arg) or occurs(var, tp.ret)
    if isinstance(tp, TRef):
        return occurs(var, tp.inner)
    return False  # TCon


def free_tvars(tp: Type) -> Tuple[TVar, ...]:
    """类型中出现的自由类型变量（按去重后的遍历顺序）。"""
    seen = {}
    for v in tp.free_vars():
        seen[v] = None
    return tuple(seen)


def apply(subst: "Subst", tp: Type) -> Type:
    """把替换幂等应用到类型上；未被绑定的变量保持原样。"""
    if isinstance(tp, TVar):
        mapped = subst.get(tp)
        if mapped is None:
            return tp
        # 替换要求是幂等的；这里再 apply 一次对幂等替换是无操作，
        # 但能在部分替换上也给出范式形式，且对非循环替换必然终止
        # （unify 产出的替换经过 occurs 检查，绝无循环）。
        resolved = apply(subst, mapped)
        return resolved
    if isinstance(tp, TFun):
        return TFun(apply(subst, tp.arg), apply(subst, tp.ret))
    if isinstance(tp, TRef):
        return TRef(apply(subst, tp.inner))
    return tp  # TCon / TVar 未绑定


def bind(subst: "Subst", var: TVar, tp: Type) -> "Subst":
    """替换组合：返回 ``s'`` 使得对任意类型 ``t``，``s'(t) = s({var -> s(tp)}(t))``。

    即新绑定中的右端以及所有旧绑定右端都先经过已有替换，保持替换幂等。
    """
    rhs = apply(subst, tp)
    merged: Subst = dict(subst)
    for key, value in list(merged.items()):
        merged[key] = apply({var: rhs}, value)
    merged[var] = rhs
    return merged


class TypeRenderer:
    """类型渲染；变量名由外部规范命名映射提供，缺省用 ``?ident``。"""

    def __init__(self, tp: Type, names: "Dict[TVar, str] | None" = None) -> None:
        self.tp = tp
        self.names = names or {}

    def render(self) -> str:
        return self._render(self.tp, top=True)

    def _name(self, var: TVar) -> str:
        if isinstance(var, RigidTVar):
            return self.names.get(var, var.name)
        return self.names.get(var, f"?{var.ident}")

    def _render(self, tp: Type, top: bool = False) -> str:
        if isinstance(tp, TVar):
            return self._name(tp)
        if isinstance(tp, TCon):
            return tp.name
        if isinstance(tp, TRef):
            # Ref(...) 自带分隔符，内部按顶层渲染，避免双重括号。
            return f"Ref({self._render(tp.inner, top=True)})"
        # TFun：函数参数位置需要加括号，右结合的返回类型不需要。
        s = f"{self._render(tp.arg, top=False)} -> {self._render(tp.ret, top=True)}"
        if not top:
            return f"({s})"
        return s


def render_type(tp: Type, names: "Dict[TVar, str] | None" = None) -> str:
    """便捷渲染入口。"""
    return TypeRenderer(tp, names).render()


def canonical_names(tp: Type, scheme_vars: "Tuple[TVar, ...] | None" = None) -> "Dict[TVar, str]":
    """按“首次出现顺序”给出规范变量名：a, b, ..., z, a1, b1, ...。

    遍历顺序即类型结构的先序遍历（函数先参数后返回，引用看内部），
    与推断时内部 id 无关，因此重命名内部变量不影响最终主类型字符串。

    ``scheme_vars`` 用于 ``forall ... . t`` 的打印：先列泛化变量（按其在
    类型中首次出现的顺序），其余（未解析）变量排在后面；当前主类型
    推断完成后不存在未解析变量，该参数只是保留扩展点。
    """
    order = {}

    def walk(t: Type) -> None:
        if isinstance(t, TVar):
            if t not in order:
                order[t] = len(order)
            return
        if isinstance(t, TFun):
            walk(t.arg)
            walk(t.ret)
        elif isinstance(t, TRef):
            walk(t.inner)

    walk(tp)
    if scheme_vars is not None:
        for v in scheme_vars:  # 保证泛化变量即使（理论上）未出现也被命名
            order.setdefault(v, len(order))

    def letter(n: int) -> str:
        base = n % 26
        suffix = n // 26
        return chr(ord("a") + base) + (str(suffix) if suffix else "")

    return {var: letter(index) for var, index in sorted(order.items(), key=lambda kv: kv[1])}
