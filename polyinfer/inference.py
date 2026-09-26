"""核心：带值限制的多态类型推断（Algorithm W / Hindley–Milner）。

设计要点
========

* **显式替换线程**：每个 ``ti_*`` 返回 ``(subst, type)``，替换只通过
  :mod:`polyinfer.types` 里的 ``apply``/``bind`` 组合推进；不使用可变并查集，
  推理失败时除了抛出异常不产生任何外部可观察变更（纯函数）。
* **统一**：:meth:`Inferencer.unify` 自带 occurs 检查（无限类型）、构造子
  冲突（类型不一致），引用类型按不变式逐点统一。
* **值限制**：``let`` 只对“非扩展（语法值）”绑定泛化——字面量、lambda、变量；
  ``ref e`` 等其他表达式得到单态类型，堵住引用多态漏洞。
* **lambda 参数单态**：参数以平凡单态类型方案入环境，其类型变量在整个
  函数体内只允许被统一成一个类型。
* **约束留痕**：每次统一（含函数/引用深入后的子等式）都记录参与统一前的
  两端类型与 AST 路径，供独立替换器
  (:mod:`polyinfer.reference`) 事后核验最终替换满足全部等式。

本过程确定性、无回溯：不存在“搜索预算”这一状态。输入超限是资源拒绝
(:class:`NodeLimitError`)，统一失败是构造性的无解证明
(:class:`UnificationError`)，二者类型不同。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
import sys
from typing import Dict, List, Tuple

from . import ast as ast_mod
from .errors import (
    EscapedRigidVariableError,
    InfiniteTypeError,
    SignatureMismatchError,
    TypeMismatchError,
    UnboundVariableError,
    ValueRestrictionConflictError,
)
from .signature import build_signature_type, parse_signature
from .types import (
    BOOL,
    INT,
    UNIT,
    Subst,
    TCon,
    TFun,
    TRef,
    TVar,
    Type,
    apply,
    bind,
    canonical_names,
    free_tvars,
    occurs,
    render_type,
)
from .types import RigidTVar

Path = Tuple[str, ...]

# 非扩展表达式（标准值限制下可安全泛化的语法值）：
# 字面量、lambda；变量也非扩展，但其自由变量均来自环境，泛化集合自然为空。
_VALUE_TAGS = frozenset({"int", "bool", "lambda", "var"})


@dataclass(frozen=True)
class TypeScheme:
    """类型方案 ``forall quantified . t``。"""

    quantified: frozenset
    t: Type


@dataclass(frozen=True)
class Constraint:
    """一条统一等式（统一发生前的形态）及其两端的 AST 路径。"""

    left: Type
    right: Type
    left_path: Path
    right_path: Path
    note: str


@dataclass(frozen=True)
class Generalization:
    """一次 let 事件的留痕，供独立检查器复核闭包条件与值限制。

    * ``bound_tag``：绑定表达式的 tag；独立检查器据此自行判定是否属于
      非扩展表达式（语法值），而不是信任核心的结论；
    * ``bound_type``：泛化时刻（当时替换下）绑定表达式的类型；
    * ``env_snapshot``：泛化时刻环境中各方案的
      （名字, 被限定 TVar 集合, 当时替换下的体类型）。
    独立检查器据此自行重算环境自由变量并比对完整泛化集合，
    而不是信任核心给出的结论。
    """

    path: Path
    name: str
    bound_tag: str
    bound_type: Type
    generalized: Tuple[TVar, ...]
    env_snapshot: Tuple[Tuple[str, frozenset, Type], ...]
    signed: bool = False
    signature_text: str = ""
    rigid_vars: Tuple[TVar, ...] = ()


@dataclass(frozen=True)
class InferenceResult:
    """成功推断结果。

    * ``principal_type``：主类型字符串，变量按类型结构先序遍历的首次出现
      顺序命名为 a, b, ...；
    * ``type``：主类型的结构化表示；
    * ``names``：TVar -> 规范名；
    * ``subst``：最终替换（幂等、无循环）；
    * ``constraints`` / ``generalizations``：核验留痕；
    * ``node_count``：AST 节点数。
    """

    principal_type: str
    type: Type
    names: Dict[TVar, str]
    subst: Subst
    constraints: Tuple[Constraint, ...]
    generalizations: Tuple[Generalization, ...]
    node_count: int

    def verify(self):
        """用**独立**替换器核验本结果（不经过统一核心）。"""
        from .reference import check_result

        return check_result(self)


class Inferencer:
    def __init__(self) -> None:
        self._counter = count()
        self.constraints: List[Constraint] = []
        self.generalizations: List[Generalization] = []
        # 当前正在检查的签名引入的刚性变量集合（用于错误归类）。
        # 当前正在检查的签名上下文栈：(let 路径, 名字, 签名文本, 刚性集合)。
        self.active_sigs: list = []

    # ------------------------------------------------------------ 工具

    def fresh(self) -> TVar:
        return TVar(next(self._counter))

    def fresh_rigid(self, name: str) -> RigidTVar:
        return RigidTVar(next(self._counter), name)

    def instantiate(self, scheme: TypeScheme) -> Type:
        """实例化类型方案：每个被限定变量换一个全新类型变量。"""
        if not scheme.quantified:
            return scheme.t
        mapping = {v: self.fresh() for v in scheme.quantified}
        return _rename(scheme.t, mapping)

    def unify(
        self,
        s: Subst,
        left: Type,
        right: Type,
        left_path: Path,
        right_path: Path,
        note: str,
    ) -> Subst:
        """统一两类型并返回扩展后的替换。

        失败时抛出 :class:`TypeMismatchError` 或 :class:`InfiniteTypeError`，
        路径随类型结构下钻而追加 ``[arg]``/``[ret]``/``[inner]``，
        定位到真正冲突的叶子。
        """
        l = apply(s, left)
        r = apply(s, right)
        self.constraints.append(Constraint(l, r, left_path, right_path, note))

        if isinstance(l, TVar) or isinstance(r, TVar):
            if isinstance(l, TVar):
                return self._unify_var(s, l, r, left_path, right_path)
            return self._unify_var(s, r, l, right_path, left_path)

        if isinstance(l, TCon) and isinstance(r, TCon):
            if l.name == r.name:
                return s
            raise self._sig_or_mismatch(left_path, right_path, l, r, infinite=False)

        if isinstance(l, TFun) and isinstance(r, TFun):
            s1 = self.unify(
                s,
                l.arg,
                r.arg,
                left_path + ("[arg]",),
                right_path + ("[arg]",),
                f"{note}：参数",
            )
            s2 = self.unify(
                s1,
                apply(s1, l.ret),
                apply(s1, r.ret),
                left_path + ("[ret]",),
                right_path + ("[ret]",),
                f"{note}：返回值",
            )
            return s2

        if isinstance(l, TRef) and isinstance(r, TRef):
            return self.unify(
                s,
                l.inner,
                r.inner,
                left_path + ("[inner]",),
                right_path + ("[inner]",),
                f"{note}：引用单元",
            )

        raise self._sig_or_mismatch(left_path, right_path, l, r, infinite=False)

    def _unify_var(self, s: Subst, va: Type, other: Type, va_path: Path, other_path: Path,
                   ) -> Subst:
        """统一其中一侧为（已解析的）类型变量的情形。

        刚性变量（显式签名的 skolem）只能与自身或普通推断变量统一：
        不能实例化成具体类型，也不能与另一个不同的刚性变量合并。
        ``swap`` 仅用于交换路径方向，等式对称。
        """
        if isinstance(other, TVar):
            if va == other:
                return s
            va_rigid = isinstance(va, RigidTVar)
            other_rigid = isinstance(other, RigidTVar)
            if va_rigid and other_rigid:
                raise self._sig_or_mismatch(va_path, other_path, va, other, infinite=False)
            # 刚性 ↔ 普通变量：允许普通变量绑定到刚性（方向取普通变量为键）。
            if va_rigid:
                if occurs(other, va):
                    raise InfiniteTypeError(other_path, va_path, other, va)
                return bind(s, other, va)
            if other_rigid:
                if occurs(va, other):
                    raise InfiniteTypeError(va_path, other_path, va, other)
                return bind(s, va, other)
            if occurs(va, other):
                raise InfiniteTypeError(va_path, other_path, va, other)
            return bind(s, va, other)
        # 刚性变量在签名检查中充当 skolem 目标：
        # 允许实现侧的自由推断变量经统一把它替换成复合类型
        #（实现比签名更一般，如实现 b->b、签名 a）；真正的危险
        # （被外层单态吞并、扩展式泛化）由逃逸检查与值限制检查拦截。
        if isinstance(va, RigidTVar) and not isinstance(other, TVar) and not free_tvars(other):
            # 刚性变量与不含推断变量的具体复合类型冲突（a ≠ Int/Bool/具体函数）。
            # 含自由推断变量的复合类型仍允许：实现可能比签名更一般
            #（如 ?b -> ?b 与 a），其安全性由逃逸检查/值限制把关。
            raise self._sig_or_mismatch(va_path, other_path, va, other, infinite=False)
        if occurs(va, other):
            raise self._sig_or_mismatch(va_path, other_path, va, other, infinite=True)
        return bind(s, va, other)

    def _sig_or_mismatch(self, left_path, right_path, left_type, right_type, infinite):
        """若处于某个签名检查的活跃作用域，抛签名不匹配；否则抛普通统一失败。"""
        if self.active_sigs and _touches_active_rigid(left_type, right_type, self.active_sigs[-1][3]):
            let_path, name, sig_text = self.active_sigs[-1][:3]
            raise SignatureMismatchError(
                let_path, name, sig_text, left_path, right_path, left_type, right_type,
                "occurs 检查失败（无限类型）" if infinite else "构造子/刚性变量冲突",
            )
        if infinite:
            raise InfiniteTypeError(left_path, right_path, left_type, right_type)
        raise TypeMismatchError(left_path, right_path, left_type, right_type)

    # ------------------------------------------------------------ 规则

    def ti(self, node: ast_mod.Node, env: Dict[str, TypeScheme], s: Subst, path: Path):
        tag = node["tag"]
        method = self._DISPATCH.get(tag)
        if method is None:  # 校验阶段已保证不可达
            raise ast_mod.InvalidAstError(path, f"未知节点 tag {tag!r}")
        return method(self, node, env, s, path)

    def _ti_var(self, node, env, s, path):
        name = node["name"]
        scheme = env.get(name)
        if scheme is None:
            raise UnboundVariableError(path, name)
        return s, self.instantiate(scheme)

    def _ti_int(self, node, env, s, path):
        return s, INT

    def _ti_bool(self, node, env, s, path):
        return s, BOOL

    def _ti_lambda(self, node, env, s, path):
        param_var = self.fresh()
        body_env = dict(env)
        # lambda 参数是单态绑定：空限定集的平凡方案，函数体内共享同一个变量。
        body_env[node["param"]] = TypeScheme(frozenset(), param_var)
        s1, body_t = self.ti(node["body"], body_env, s, path + ("body",))
        return s1, TFun(apply(s1, param_var), body_t)

    def _ti_apply(self, node, env, s, path):
        s1, fn_t = self.ti(node["fn"], env, s, path + ("fn",))
        s2, arg_t = self.ti(node["arg"], env, s1, path + ("arg",))
        ret_var = self.fresh()
        # 右侧“参数类型 -> 返回类型”由 apply 节点诱导，路径挂在 apply 节点上。
        s3 = self.unify(
            s2,
            apply(s2, fn_t),
            TFun(arg_t, ret_var),
            path + ("fn",),
            path,
            "apply 要求被应用值是函数",
        )
        return s3, apply(s3, ret_var)

    def _ti_let(self, node, env, s, path):
        name = node["name"]
        sig_text = node.get("sig")
        if sig_text is not None:
            return self._ti_let_signed(node, env, s, path, name, sig_text)

        s1, bound_raw = self.ti(node["bound"], env, s, path + ("bound",))
        bound_t = apply(s1, bound_raw)
        return self._finish_let(node, env, path, name, s1, bound_t, bound_t,
                                signed=False, sig_text="", rigids=())

    def _finish_let(self, node, env, path, name, s1, bound_t, scheme_body,
                    signed, sig_text, rigids):
        bound_tag = node["bound"]["tag"]
        if signed:
            # 签名绑定：量化集合恰为声明的刚性变量；实现中残留的
            # 未约束推断变量按单态处理（签名比推断更具体是允许的）。
            quantified: Tuple[TVar, ...] = ()
        elif bound_tag in _VALUE_TAGS:
            env_free = _env_free_vars(env, s1)
            # 刚性变量绝不允许进入普通泛化（它们只属于签名方案）。
            quantified = tuple(
                v for v in free_tvars(bound_t)
                if v not in env_free and not isinstance(v, RigidTVar)
            )
        else:
            # 值限制：扩展式（ref、apply、read、write、if ...）不泛化。
            quantified = ()
        scheme = TypeScheme(frozenset(quantified), bound_t)

        self.generalizations.append(
            Generalization(
                path=path,
                name=name,
                bound_tag=bound_tag,
                bound_type=bound_t,
                generalized=quantified,
                env_snapshot=tuple(
                    (ename, esch.quantified, apply(s1, esch.t))
                    for ename, esch in env.items()
                ),
                signed=signed,
                signature_text=sig_text,
                rigid_vars=tuple(rigids),
            )
        )

        body_env = dict(env)
        if signed:
            # 签名绑定：方案体取签名形状（应用当前替换后的刚性变量），
            # 量化集合就是声明的刚性变量。
            scheme = TypeScheme(frozenset(rigids), apply(s1, scheme_body))
        body_env[name] = scheme
        s2, body_t = self.ti(node["body"], body_env, s1, path + ("body",))
        return s2, body_t

    def _ti_let_signed(self, node, env, s, path, name, sig_text):
        """带 rank-1 显式签名的 let 检查。

        1. 解析签名并为每个量化变量造**刚性**变量（检查期间保持刚性）；
        2. 在普通推断下推导绑定表达式；
        3. 把实现类型与签名类型统一（失败归类为签名不匹配）；
        4. 值限制：扩展式绑定仍残留刚性变量则拒绝（值限制冲突）；
        5. 逃逸检查：刚性变量被外层作用域的推断变量吞并则拒绝
           （逃逸的刚性变量）——典型于实现捕获外层单态变量；
        6. 绑定按声明的方案（量化变量 = 刚性变量）进入函数体，
           每次使用独立实例化；其余残留的内部推断变量保持单态。
        """
        bound_path = path + ("bound",)
        parsed = parse_signature(sig_text)  # 校验阶段已解析过一次
        rigids, sig_type = build_signature_type(parsed, self.fresh_rigid)

        # 记录检查前外层已存在的推断变量，用于识别“外层单态变量”。
        outer_flex = {v for v in _all_subst_vars(s) if not isinstance(v, RigidTVar)}
        outer_flex |= _env_flex_vars(env)

        frame = (path, name, sig_text, set(rigids))
        self.active_sigs.append(frame)
        try:
            s1, bound_raw = self.ti(node["bound"], env, s, bound_path)
        finally:
            self.active_sigs.pop()
        bound_t = apply(s1, bound_raw)
        sig_t1 = apply(s1, sig_type)

        # 3) 实现类型与签名类型统一（此上下文中刚性刚被移出 active，
        #    但 RigidTVar 自身的统一规则不变；用专门包装归类错误）。
        try:
            s2 = self.unify(
                s1, bound_t, sig_t1,
                bound_path, path + ("sig",),
                f"let {name} 的显式签名检查",
            )
        except (TypeMismatchError, InfiniteTypeError) as exc:
            raise SignatureMismatchError(
                path, name, sig_text,
                exc.left_path, exc.right_path, exc.left_type, exc.right_type,
                "occurs 检查失败（无限类型）" if isinstance(exc, InfiniteTypeError) else "构造子/刚性变量冲突",
            ) from exc

        bound_t2 = apply(s2, bound_raw)
        sig_t2 = apply(s2, sig_type)
        bound_tag = node["bound"]["tag"]

        # 4) 逃逸检查（先于值限制）：刚性变量是否被**外层**推断变量吞并。
        #    内层新生成的推断变量与刚性合并是合法的（它们随绑定消亡/泛化）。
        #    同时把“与刚性等价的内部柔性变量”收集出来：若扩展式绑定后
        #    这些变量仍未解析成具体类型，声明的多态就是被单态计算结果
        #    冒充的，归为值限制冲突。
        dangling_flex: set = set()
        for rigid in rigids:
            resolved_rigid = apply(s2, rigid)
            for flex in outer_flex:
                resolved = apply(s2, flex)
                if resolved == resolved_rigid:
                    raise EscapedRigidVariableError(path, name, sig_text, resolved_rigid, bound_path)
        for v in free_tvars(sig_t2):
            # 签名类型里残留的未绑定柔性变量都来自绑定内部，与某个刚性位置等价。
            if not isinstance(v, RigidTVar):
                dangling_flex.add(v)

        # 5) 值限制：扩展式不能承载声明的多态。
        surviving_rigids = _mentions_any(sig_t2, tuple(apply(s2, r) for r in rigids))
        if bound_tag not in _VALUE_TAGS and (surviving_rigids or dangling_flex):
            raise ValueRestrictionConflictError(path, name, sig_text, bound_tag, bound_path)

        return self._finish_let(
            node, env, path, name, s2, bound_t2, sig_t2,
            signed=True, sig_text=sig_text,
            rigids=tuple(v for v in rigids for v in (apply(s2, v),) if isinstance(v, RigidTVar)),
        )

    def _ti_if(self, node, env, s, path):
        s1, cond_t = self.ti(node["cond"], env, s, path + ("cond",))
        s2 = self.unify(
            s1,
            apply(s1, cond_t),
            BOOL,
            path + ("cond",),
            path,
            "if 的条件必须是 Bool",
        )
        s3, then_t = self.ti(node["then"], env, s2, path + ("then",))
        s4, else_t = self.ti(node["else_"], env, s3, path + ("else_",))
        s5 = self.unify(
            s4,
            apply(s4, then_t),
            apply(s4, else_t),
            path + ("then",),
            path + ("else_",),
            "if 两个分支类型必须一致",
        )
        return s5, apply(s5, then_t)

    def _ti_ref(self, node, env, s, path):
        s1, inner_t = self.ti(node["inner"], env, s, path + ("inner",))
        return s1, TRef(inner_t)

    def _ti_read(self, node, env, s, path):
        s1, ref_t = self.ti(node["ref"], env, s, path + ("ref",))
        inner_var = self.fresh()
        s2 = self.unify(
            s1,
            apply(s1, ref_t),
            TRef(inner_var),
            path + ("ref",),
            path,
            "read 要求引用类型",
        )
        return s2, apply(s2, inner_var)

    def _ti_write(self, node, env, s, path):
        s1, ref_t = self.ti(node["ref"], env, s, path + ("ref",))
        s2, val_t = self.ti(node["value"], env, s1, path + ("value",))
        # 引用不变式：Ref(值类型)。必须接住扩展后的替换继续线程化。
        s3 = self.unify(
            s2,
            apply(s2, ref_t),
            TRef(apply(s2, val_t)),
            path + ("ref",),
            path + ("value",),
            "write 要求左侧引用的单元类型等于右侧表达式类型",
        )
        return s3, UNIT

    _DISPATCH = {
        "var": _ti_var,
        "int": _ti_int,
        "bool": _ti_bool,
        "lambda": _ti_lambda,
        "apply": _ti_apply,
        "let": _ti_let,
        "if": _ti_if,
        "ref": _ti_ref,
        "read": _ti_read,
        "write": _ti_write,
    }


# ---------------------------------------------------------------- 杂项

def _rename(tp: Type, mapping: Dict[TVar, TVar]) -> Type:
    if isinstance(tp, TVar):
        return mapping.get(tp, tp)
    if isinstance(tp, TFun):
        return TFun(_rename(tp.arg, mapping), _rename(tp.ret, mapping))
    if isinstance(tp, TRef):
        return TRef(_rename(tp.inner, mapping))
    return tp


def _env_free_vars(env: Dict[str, TypeScheme], s: Subst):
    """环境的自由类型变量：各方案体类型的自由变量减去其被限定变量。"""
    free = set()
    for scheme in env.values():
        body = apply(s, scheme.t)
        quantified = scheme.quantified
        for v in free_tvars(body):
            if v not in quantified:
                free.add(v)
    return free


def _env_flex_vars(env: Dict[str, TypeScheme]):
    """环境里当前出现的全部普通（非刚性）推断变量。"""
    out = set()
    for scheme in env.values():
        for v in free_tvars(scheme.t):
            if not isinstance(v, RigidTVar):
                out.add(v)
    return out


def _all_subst_vars(s: Subst):
    out = set(s)
    for rhs in s.values():
        out.update(free_tvars(rhs))
    return out


def _mentions_any(tp: Type, vars_: Tuple[TVar, ...]) -> bool:
    wanted = set(vars_)
    return any(v in wanted for v in free_tvars(tp))


def _touches_active_rigid(left: Type, right: Type, rigids: set) -> bool:
    return any(isinstance(v, RigidTVar) and v in rigids for v in (*free_tvars(left), *free_tvars(right)))


# ---------------------------------------------------------------- 入口

def infer(ast: ast_mod.Node, max_nodes: int = 500) -> InferenceResult:
    """推断整棵 AST 的主类型。

    :param ast: 由 :mod:`polyinfer.ast` 构造器或同构 dict 组成的 AST；
    :param max_nodes: 接口节点上限，默认 500；
    :raises InvalidAstError: AST 结构或字面量非法（路径可定位）；
    :raises NodeLimitError: 节点数超过上限（资源拒绝，非类型无解）；
    :raises UnboundVariableError: 出现未绑定变量（含递归 let）；
    :raises TypeMismatchError / InfiniteTypeError: 已证明不可类型化，
        携带失败约束两端的 AST 路径与叶子类型。
    """
    # 接口参数自身的合法性（不静默纠正：bool 不算 int）。
    if isinstance(max_nodes, bool) or not isinstance(max_nodes, int):
        raise TypeError(f"max_nodes 必须是 int，实际为 {type(max_nodes).__name__}")
    if max_nodes < 1:
        raise ValueError(f"max_nodes 必须为正整数，实际为 {max_nodes}")

    # AST 最深可达 max_nodes（500）层，每层校验/ti/替换应用会产生多个
    # Python 帧，临时提高递归上限；无论成功失败都恢复调用前的值，
    # 不给解释器进程留下未说明的全局变更。
    old_limit = sys.getrecursionlimit()
    needed = 200 * max_nodes + 1000
    if needed > old_limit:
        sys.setrecursionlimit(needed)
    try:
        validator = ast_mod.AstValidator(max_nodes)
        validator.validate(ast)  # 先完整校验；此处抛出则不会进入任何推断状态
        checker = Inferencer()
        subst, root_t = checker.ti(ast, {}, {}, ())
        # 深类型的替换应用、规范命名与渲染同样是结构递归，留在提限区间内。
        principal = apply(subst, root_t)
        names = canonical_names(principal)
        principal_str = render_type(principal, names)
    finally:
        sys.setrecursionlimit(old_limit)
    return InferenceResult(
        principal_type=principal_str,
        type=principal,
        names=dict(names),
        subst=subst,
        constraints=tuple(checker.constraints),
        generalizations=tuple(checker.generalizations),
        node_count=validator.count,
    )
