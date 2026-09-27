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
* **互相递归组（letrec）**：整组绑定先入单态占位类型变量（与书写顺序无关），
  在同一套约束下检查全部绑定，整组通过后才逐绑定按值限制泛化；
  泛化闭包排除外部环境与其他组员的自由变量，可修改引用（ref 等扩展式）
  仍不泛化，安全限制不因递归组而放松。
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
    InfiniteTypeError,
    TypeMismatchError,
    UnboundVariableError,
)
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

    # ------------------------------------------------------------ 工具

    def fresh(self) -> TVar:
        return TVar(next(self._counter))

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

        if isinstance(l, TVar):
            if isinstance(r, TVar) and l == r:
                return s
            # 检查 l 是否出现在 r 中（occurs 检查，防止无限类型）。
            if occurs(l, r):
                raise InfiniteTypeError(left_path, right_path, l, r)
            return bind(s, l, r)
        if isinstance(r, TVar):
            if occurs(r, l):
                raise InfiniteTypeError(right_path, left_path, r, l)
            return bind(s, r, l)

        if isinstance(l, TCon) and isinstance(r, TCon):
            if l.name == r.name:
                return s
            raise TypeMismatchError(left_path, right_path, l, r)

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

        raise TypeMismatchError(left_path, right_path, l, r)

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
        s1, bound_raw = self.ti(node["bound"], env, s, path + ("bound",))
        bound_t = apply(s1, bound_raw)
        name = node["name"]

        bound_tag = node["bound"]["tag"]
        if bound_tag in _VALUE_TAGS:
            env_free = _env_free_vars(env, s1)
            quantified = tuple(v for v in free_tvars(bound_t) if v not in env_free)
            scheme = TypeScheme(frozenset(quantified), bound_t)
        else:
            # 值限制：扩展式（ref、apply、read、write、if ...）不泛化。
            quantified = ()
            scheme = TypeScheme(frozenset(), bound_t)

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
            )
        )

        body_env = dict(env)
        body_env[name] = scheme
        s2, body_t = self.ti(node["body"], body_env, s1, path + ("body",))
        return s2, body_t

    def _ti_letrec(self, node, env, s, path):
        bindings = node["bindings"]
        names = [b["name"] for b in bindings]

        # 1) 整组先入单态占位类型变量：分析任一绑定时，组内其他名字
        #    均已可引用，结果与绑定书写顺序无关。
        placeholders = {name: self.fresh() for name in names}
        group_env = dict(env)
        for name in names:
            group_env[name] = TypeScheme(frozenset(), placeholders[name])

        # 2) 同一套替换线程下检查整组：递归出现与定义必须一致。
        s_cur = s
        for b in bindings:
            bpath = path + ("bindings", b["name"])
            s_cur, bound_t = self.ti(b["bound"], group_env, s_cur, bpath)
            s_cur = self.unify(
                s_cur,
                apply(s_cur, placeholders[b["name"]]),
                bound_t,
                bpath,
                bpath,
                f"递归绑定 {b['name']} 的递归用法与其定义类型必须一致",
            )

        # 3) 整组检查结束后才泛化：逐绑定按值限制决定，闭包排除外部
        #    环境与其他组员的自由变量（共享变量保持单态，保证安全）。
        env_free = _env_free_vars(env, s_cur)
        final_types = {name: apply(s_cur, placeholders[name]) for name in names}
        outer_snapshot = tuple(
            (ename, esch.quantified, apply(s_cur, esch.t))
            for ename, esch in env.items()
        )
        body_env = dict(env)
        for b in bindings:
            name = b["name"]
            bound_tag = b["bound"]["tag"]
            final_t = final_types[name]
            others_free = set()
            for other in names:
                if other != name:
                    others_free |= set(free_tvars(final_types[other]))
            if bound_tag in _VALUE_TAGS:
                quantified = tuple(
                    v for v in free_tvars(final_t)
                    if v not in env_free and v not in others_free
                )
            else:
                # 值限制：ref/apply 等扩展式即便在递归组内也不泛化。
                quantified = ()
            snapshot = outer_snapshot + tuple(
                (other, frozenset(), final_types[other])
                for other in names if other != name
            )
            self.generalizations.append(
                Generalization(
                    path=path + ("bindings", name),
                    name=name,
                    bound_tag=bound_tag,
                    bound_type=final_t,
                    generalized=quantified,
                    env_snapshot=snapshot,
                )
            )
            body_env[name] = TypeScheme(frozenset(quantified), final_t)

        s2, body_t = self.ti(node["body"], body_env, s_cur, path + ("body",))
        return s2, body_t

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
        "letrec": _ti_letrec,
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
