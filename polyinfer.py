"""polyinfer — 带值限制（value restriction）的多态类型推断。

从零实现的 Hindley–Milner 风格推断（Algorithm W 的约束记录变体），
仅依赖 Python 3.14 标准库。输入为纯数据 AST（绝不 eval/exec 输入）。

语言节点：变量、整数、布尔、lambda、apply、let、if、ref、read、write。
类型：Int、Bool、Unit、函数、引用、类型变量。

规则要点：
  * let 仅对语法值（变量 / 字面量 / lambda）泛化（值限制）；
  * lambda 参数单态；
  * write 返回 Unit；
  * 不含递归 let（let 绑定名在其值表达式中不可见）。

公开接口：
  infer(node, *, max_nodes=500, max_steps=1_000_000) -> InferResult
  unify(left, right, *, max_steps=...) -> dict[int, Type]
  apply_substitution / compose_substitutions / free_type_vars
  type_to_string / format_path
错误语义：
  InputError            非法输入（AST 结构/字段非法、超节点上限、参数非法）
  UnboundVariableError  未绑定变量（含路径）
  UnifyError            约束被证明不可满足（含两端 AST 路径）
  OccursError           occurs 检查失败（无限类型），UnifyError 子类
  BudgetExhaustedError  搜索预算耗尽（与“证明无解”严格区分）
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "Node", "Var", "Int", "Bool", "Lam", "App", "Let", "If", "Ref", "Read", "Write",
    "Type", "TInt", "TBool", "TUnit", "TFun", "TRef", "TVar",
    "INT", "BOOL", "UNIT", "Scheme", "Constraint", "InferResult",
    "infer", "unify", "apply_substitution", "compose_substitutions",
    "free_type_vars", "type_to_string", "format_path",
    "InferError", "InputError", "UnboundVariableError",
    "UnifyError", "OccursError", "BudgetExhaustedError",
    "MAX_NODES", "MAX_STEPS",
]

MAX_NODES = 500
MAX_STEPS = 1_000_000


# ---------------------------------------------------------------------------
# AST 节点（纯数据，冻结 dataclass）
# ---------------------------------------------------------------------------

class Node:
    """所有 AST 节点的基类。"""
    __slots__ = ()


@dataclass(frozen=True)
class Var(Node):
    name: str


@dataclass(frozen=True)
class Int(Node):
    value: int


@dataclass(frozen=True)
class Bool(Node):
    value: bool


@dataclass(frozen=True)
class Lam(Node):
    param: str
    body: Node


@dataclass(frozen=True)
class App(Node):
    func: Node
    arg: Node


@dataclass(frozen=True)
class Let(Node):
    name: str
    value: Node
    body: Node


@dataclass(frozen=True)
class If(Node):
    cond: Node
    then: Node
    otherwise: Node


@dataclass(frozen=True)
class Ref(Node):
    expr: Node


@dataclass(frozen=True)
class Read(Node):
    expr: Node


@dataclass(frozen=True)
class Write(Node):
    ref: Node
    value: Node


# ---------------------------------------------------------------------------
# 类型
# ---------------------------------------------------------------------------

class Type:
    """所有类型的基类。"""
    __slots__ = ()


@dataclass(frozen=True)
class TInt(Type):
    def __repr__(self) -> str:  # pragma: no cover - 便于调试
        return "Int"


@dataclass(frozen=True)
class TBool(Type):
    def __repr__(self) -> str:  # pragma: no cover
        return "Bool"


@dataclass(frozen=True)
class TUnit(Type):
    def __repr__(self) -> str:  # pragma: no cover
        return "Unit"


@dataclass(frozen=True)
class TFun(Type):
    arg: Type
    ret: Type


@dataclass(frozen=True)
class TRef(Type):
    inner: Type


@dataclass(frozen=True)
class TVar(Type):
    id: int


INT = TInt()
BOOL = TBool()
UNIT = TUnit()


@dataclass(frozen=True)
class Scheme:
    """类型方案：vars 中被量化的变量 id + 类型体。"""
    vars: frozenset
    type: Type


@dataclass(frozen=True)
class Constraint:
    """一次统一约束：两端类型（生成时的快照）与两端 AST 路径。"""
    left: Type
    right: Type
    left_path: tuple
    right_path: tuple


@dataclass(frozen=True)
class InferResult:
    """成功推导的结果。

    type:         应用最终替换后的主类型（变量 id 为内部 id）。
    type_str:     按首次出现顺序规范命名（a, b, c, ...）的主类型字符串。
    constraints:  推导过程中生成并已全部满足的约束（供独立检查器复核）。
    substitution: 最终替换 {变量 id: 类型}。
    """
    type: Type
    type_str: str
    constraints: tuple
    substitution: dict


# ---------------------------------------------------------------------------
# 错误
# ---------------------------------------------------------------------------

class InferError(Exception):
    """本模块所有推断相关错误的基类。"""


class InputError(InferError, ValueError):
    """非法输入：AST 结构/字段非法、超节点上限、参数非法。"""


class UnboundVariableError(InferError):
    def __init__(self, name: str, path: tuple):
        self.name = name
        self.path = tuple(path)
        super().__init__(f"unbound variable {name!r} at {format_path(self.path)}")


class UnifyError(InferError):
    """约束被证明不可满足。left_path/right_path 为失败约束两端的 AST 路径。"""

    def __init__(self, left: Type, right: Type, left_path: tuple, right_path: tuple):
        self.left = left
        self.right = right
        self.left_path = tuple(left_path)
        self.right_path = tuple(right_path)
        super().__init__(
            "cannot unify "
            f"{_show(left)} (from {format_path(self.left_path)}) with "
            f"{_show(right)} (from {format_path(self.right_path)})"
        )


class OccursError(UnifyError):
    """occurs 检查失败：统一将产生无限类型。"""

    def __init__(self, var: TVar, inner: Type, left_path: tuple, right_path: tuple):
        self.var = var
        self.inner = inner
        UnifyError.__init__(self, var, inner, left_path, right_path)
        self.args = (
            f"occurs check failed: {_show(var)} occurs in {_show(inner)}; "
            f"infinite type rejected "
            f"(constraint from {format_path(self.left_path)} vs "
            f"{format_path(self.right_path)})",
        )


class BudgetExhaustedError(InferError):
    """搜索预算耗尽。与 UnifyError（证明无解）严格区分。"""

    def __init__(self, max_steps: int):
        self.max_steps = max_steps
        super().__init__(
            f"unification step budget exhausted (max_steps={max_steps}); "
            "this means 'budget exhausted', NOT 'proven unsolvable'"
        )


# ---------------------------------------------------------------------------
# 路径与类型的展示
# ---------------------------------------------------------------------------

def format_path(path: tuple) -> str:
    """把 AST 路径元组渲染为 $/let.value/app.func 形式。"""
    if not path:
        return "$"
    return "$/" + "/".join(path)


def _show(t: Type) -> str:
    """内部展示：类型变量用原始 id（?7），用于错误消息。"""
    if isinstance(t, TVar):
        return f"?{t.id}"
    if isinstance(t, TInt):
        return "Int"
    if isinstance(t, TBool):
        return "Bool"
    if isinstance(t, TUnit):
        return "Unit"
    if isinstance(t, TFun):
        arg = _show(t.arg)
        if isinstance(t.arg, TFun):
            arg = f"({arg})"
        return f"{arg} -> {_show(t.ret)}"
    if isinstance(t, TRef):
        inner = _show(t.inner)
        if isinstance(t.inner, TFun):
            inner = f"({inner})"
        return f"ref {inner}"
    raise TypeError(f"not a type: {t!r}")


def _var_name(i: int) -> str:
    base = chr(ord("a") + i % 26)
    suffix = "" if i < 26 else str(i // 26)
    return base + suffix


def type_to_string(t: Type) -> str:
    """按变量首次出现顺序规范命名（a, b, c, ...）并渲染主类型。

    含自由变量时加 forall 前缀，例如 ``forall a. a -> a``。
    """
    names: dict[int, str] = {}

    def visit(x: Type) -> None:
        if isinstance(x, TVar):
            if x.id not in names:
                names[x.id] = _var_name(len(names))
        elif isinstance(x, TFun):
            visit(x.arg)
            visit(x.ret)
        elif isinstance(x, TRef):
            visit(x.inner)
        elif not isinstance(x, (TInt, TBool, TUnit)):
            raise InputError(f"type_to_string: not a type: {x!r}")

    def render(x: Type) -> str:
        if isinstance(x, TVar):
            return names[x.id]
        if isinstance(x, TInt):
            return "Int"
        if isinstance(x, TBool):
            return "Bool"
        if isinstance(x, TUnit):
            return "Unit"
        if isinstance(x, TFun):
            arg = render(x.arg)
            if isinstance(x.arg, TFun):
                arg = f"({arg})"
            return f"{arg} -> {render(x.ret)}"
        if isinstance(x, TRef):
            inner = render(x.inner)
            if isinstance(x.inner, TFun):
                inner = f"({inner})"
            return f"ref {inner}"
        raise InputError(f"type_to_string: not a type: {x!r}")  # pragma: no cover

    visit(t)
    body = render(t)
    if names:
        quants = ", ".join(names.values())
        return f"forall {quants}. {body}"
    return body


# ---------------------------------------------------------------------------
# 替换（substitution）：应用与组合
# ---------------------------------------------------------------------------

def free_type_vars(t: Type) -> set:
    """类型中自由类型变量的 id 集合。"""
    if isinstance(t, TVar):
        return {t.id}
    if isinstance(t, TFun):
        return free_type_vars(t.arg) | free_type_vars(t.ret)
    if isinstance(t, TRef):
        return free_type_vars(t.inner)
    if isinstance(t, (TInt, TBool, TUnit)):
        return set()
    raise InputError(f"free_type_vars: not a type: {t!r}")


def apply_substitution(subst: dict, t: Type) -> Type:
    """把替换 {var_id: Type} 应用到类型上（递归解析链式绑定）。"""
    if isinstance(t, TVar):
        seen = set()
        while isinstance(t, TVar) and t.id in subst and t.id not in seen:
            seen.add(t.id)
            t = subst[t.id]
        if isinstance(t, TVar):
            return t
        return apply_substitution(subst, t)
    if isinstance(t, TFun):
        return TFun(apply_substitution(subst, t.arg),
                    apply_substitution(subst, t.ret))
    if isinstance(t, TRef):
        return TRef(apply_substitution(subst, t.inner))
    if isinstance(t, (TInt, TBool, TUnit)):
        return t
    raise InputError(f"apply_substitution: not a type: {t!r}")


def compose_substitutions(first: dict, second: dict) -> dict:
    """组合替换：结果等价于先应用 first 再应用 second。"""
    out = {k: apply_substitution(second, v) for k, v in first.items()}
    for k, v in second.items():
        out[k] = v
    return out


# ---------------------------------------------------------------------------
# 统一（带 occurs 检查）
# ---------------------------------------------------------------------------

class _State:
    __slots__ = ("next_var", "subst", "constraints", "steps", "max_steps")

    def __init__(self, max_steps: int):
        self.next_var = 0
        self.subst: dict[int, Type] = {}
        self.constraints: list[Constraint] = []
        self.steps = 0
        self.max_steps = max_steps


def _tick(state: _State) -> None:
    state.steps += 1
    if state.steps > state.max_steps:
        raise BudgetExhaustedError(state.max_steps)


def _bind(state: _State, var: TVar, t: Type, ctx: tuple) -> None:
    _tick(state)
    if isinstance(t, TVar) and t.id == var.id:
        return
    if var.id in free_type_vars(t):
        raise OccursError(var, t, ctx[0], ctx[1])
    delta = {var.id: t}
    # 组合：旧替换的值里若引用 var，一并替换，保持替换规范化。
    state.subst = {k: apply_substitution(delta, v) for k, v in state.subst.items()}
    state.subst[var.id] = t


def _unify(state: _State, a: Type, b: Type, ctx: tuple) -> None:
    _tick(state)
    a = apply_substitution(state.subst, a)
    b = apply_substitution(state.subst, b)
    if a == b:
        return
    if isinstance(a, TVar):
        _bind(state, a, b, ctx)
        return
    if isinstance(b, TVar):
        _bind(state, b, a, ctx)
        return
    if isinstance(a, TFun) and isinstance(b, TFun):
        _unify(state, a.arg, b.arg, ctx)
        _unify(state, a.ret, b.ret, ctx)
        return
    if isinstance(a, TRef) and isinstance(b, TRef):
        _unify(state, a.inner, b.inner, ctx)
        return
    raise UnifyError(a, b, ctx[0], ctx[1])


def _constrain(state: _State, a: Type, b: Type,
               left_path: tuple, right_path: tuple) -> None:
    """记录一条约束（含两端 AST 路径）并立即求解。"""
    state.constraints.append(Constraint(a, b, tuple(left_path), tuple(right_path)))
    _unify(state, a, b, (left_path, right_path))


def unify(left: Type, right: Type, *, max_steps: int = MAX_STEPS) -> dict:
    """公开统一入口：统一两个类型，返回替换 {var_id: Type}。

    失败抛 UnifyError / OccursError（路径为空元组）；
    预算耗尽抛 BudgetExhaustedError。
    """
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
        raise InputError(f"max_steps must be a non-negative int, got {max_steps!r}")
    if not isinstance(left, Type) or not isinstance(right, Type):
        raise InputError("unify: both arguments must be Type instances")
    state = _State(max_steps)
    _unify(state, left, right, ((), ()))
    return dict(state.subst)


# ---------------------------------------------------------------------------
# 推断
# ---------------------------------------------------------------------------

def _fresh(state: _State) -> TVar:
    v = TVar(state.next_var)
    state.next_var += 1
    return v


def _is_syntactic_value(node: Node) -> bool:
    """值限制：仅变量、字面量、lambda 是语法值。"""
    return isinstance(node, (Var, Int, Bool, Lam))


def _scheme_ftv(scheme: Scheme, subst: dict) -> set:
    return free_type_vars(apply_substitution(subst, scheme.type)) - set(scheme.vars)


def _env_ftv(env: dict, subst: dict) -> set:
    out: set = set()
    for scheme in env.values():
        out |= _scheme_ftv(scheme, subst)
    return out


def _generalize(state: _State, env: dict, t: Type) -> Scheme:
    t2 = apply_substitution(state.subst, t)
    quantified = free_type_vars(t2) - _env_ftv(env, state.subst)
    return Scheme(frozenset(quantified), t2)


def _instantiate(state: _State, scheme: Scheme) -> Type:
    t = apply_substitution(state.subst, scheme.type)
    mapping = {v: _fresh(state) for v in sorted(scheme.vars)}
    return apply_substitution(mapping, t)


def _infer(node: Node, env: dict, path: tuple, state: _State) -> Type:
    if isinstance(node, Var):
        scheme = env.get(node.name)
        if scheme is None:
            raise UnboundVariableError(node.name, path)
        return _instantiate(state, scheme)

    if isinstance(node, Int):
        return INT
    if isinstance(node, Bool):
        return BOOL

    if isinstance(node, Lam):
        tv = _fresh(state)
        body_env = dict(env)
        body_env[node.param] = Scheme(frozenset(), tv)  # lambda 参数单态
        t_body = _infer(node.body, body_env, path + ("lam.body",), state)
        return TFun(tv, t_body)

    if isinstance(node, App):
        t_func = _infer(node.func, env, path + ("app.func",), state)
        t_arg = _infer(node.arg, env, path + ("app.arg",), state)
        rv = _fresh(state)
        _constrain(state, t_func, TFun(t_arg, rv),
                   path + ("app.func",), path)
        return rv

    if isinstance(node, Let):
        t_val = _infer(node.value, env, path + ("let.value",), state)
        if _is_syntactic_value(node.value):
            scheme = _generalize(state, env, t_val)  # 仅语法值泛化
        else:
            scheme = Scheme(frozenset(), apply_substitution(state.subst, t_val))
        body_env = dict(env)
        body_env[node.name] = scheme
        return _infer(node.body, body_env, path + ("let.body",), state)

    if isinstance(node, If):
        t_cond = _infer(node.cond, env, path + ("if.cond",), state)
        _constrain(state, t_cond, BOOL, path + ("if.cond",), path)
        t_then = _infer(node.then, env, path + ("if.then",), state)
        t_else = _infer(node.otherwise, env, path + ("if.else",), state)
        _constrain(state, t_then, t_else, path + ("if.then",), path + ("if.else",))
        return t_then

    if isinstance(node, Ref):
        return TRef(_infer(node.expr, env, path + ("ref.expr",), state))

    if isinstance(node, Read):
        t = _infer(node.expr, env, path + ("read.expr",), state)
        rv = _fresh(state)
        _constrain(state, t, TRef(rv), path + ("read.expr",), path)
        return rv

    if isinstance(node, Write):
        t_ref = _infer(node.ref, env, path + ("write.ref",), state)
        t_val = _infer(node.value, env, path + ("write.value",), state)
        _constrain(state, t_ref, TRef(t_val),
                   path + ("write.ref",), path + ("write.value",))
        return UNIT

    raise InputError(  # pragma: no cover - _validate 已拦截
        f"{format_path(path)}: unknown node {node!r}")


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------

def _check_name(value, path: tuple, what: str) -> None:
    if not isinstance(value, str) or not value:
        raise InputError(
            f"{format_path(path)}: {what} must be a non-empty str, got {value!r}")


def _validate(node, path: tuple, count: list, max_nodes: int) -> None:
    if not isinstance(node, Node):
        raise InputError(
            f"{format_path(path)}: expected an AST node, got "
            f"{type(node).__name__} {node!r}")
    count[0] += 1
    if count[0] > max_nodes:
        raise InputError(
            f"node limit exceeded: more than {max_nodes} nodes "
            f"(detected at {format_path(path)})")

    if isinstance(node, Var):
        _check_name(node.name, path, "Var.name")
    elif isinstance(node, Int):
        v = node.value
        if isinstance(v, bool):
            raise InputError(
                f"{format_path(path)}: Int.value must be int, got bool {v!r}")
        if isinstance(v, float):
            if math.isnan(v):
                raise InputError(f"{format_path(path)}: NaN is not allowed")
            if math.isinf(v):
                raise InputError(f"{format_path(path)}: Infinity is not allowed")
            raise InputError(
                f"{format_path(path)}: Int.value must be int, got float {v!r}")
        if not isinstance(v, int):
            raise InputError(
                f"{format_path(path)}: Int.value must be int, got "
                f"{type(v).__name__} {v!r}")
    elif isinstance(node, Bool):
        if not isinstance(node.value, bool):
            raise InputError(
                f"{format_path(path)}: Bool.value must be bool, got "
                f"{type(node.value).__name__} {node.value!r}")
    elif isinstance(node, Lam):
        _check_name(node.param, path, "Lam.param")
        _validate(node.body, path + ("lam.body",), count, max_nodes)
    elif isinstance(node, App):
        _validate(node.func, path + ("app.func",), count, max_nodes)
        _validate(node.arg, path + ("app.arg",), count, max_nodes)
    elif isinstance(node, Let):
        _check_name(node.name, path, "Let.name")
        _validate(node.value, path + ("let.value",), count, max_nodes)
        _validate(node.body, path + ("let.body",), count, max_nodes)
    elif isinstance(node, If):
        _validate(node.cond, path + ("if.cond",), count, max_nodes)
        _validate(node.then, path + ("if.then",), count, max_nodes)
        _validate(node.otherwise, path + ("if.else",), count, max_nodes)
    elif isinstance(node, Ref):
        _validate(node.expr, path + ("ref.expr",), count, max_nodes)
    elif isinstance(node, Read):
        _validate(node.expr, path + ("read.expr",), count, max_nodes)
    elif isinstance(node, Write):
        _validate(node.ref, path + ("write.ref",), count, max_nodes)
        _validate(node.value, path + ("write.value",), count, max_nodes)
    else:
        raise InputError(
            f"{format_path(path)}: unknown node type {type(node).__name__}")


def _check_limit(value, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InputError(f"{name} must be a non-negative int, got {value!r}")


def infer(node: Node, *, max_nodes: int = MAX_NODES,
          max_steps: int = MAX_STEPS) -> InferResult:
    """对 AST 做主类型推断（带值限制）。

    参数：
      node:      AST 根节点（纯数据，绝不被执行）。
      max_nodes: 节点数上限（默认 500，本题接口边界）。
      max_steps: 统一步数预算；耗尽抛 BudgetExhaustedError，
                 与 UnifyError（证明无解）严格区分。

    返回 InferResult；失败抛 InferError 子类（见模块 docstring）。
    """
    _check_limit(max_nodes, "max_nodes")
    _check_limit(max_steps, "max_steps")
    _validate(node, (), [0], max_nodes)
    state = _State(max_steps)
    t = _infer(node, {}, (), state)
    final = apply_substitution(state.subst, t)
    return InferResult(
        type=final,
        type_str=type_to_string(final),
        constraints=tuple(state.constraints),
        substitution=dict(state.subst),
    )
