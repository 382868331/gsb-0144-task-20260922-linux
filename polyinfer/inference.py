"""核心：带值限制的多态类型推断（Algorithm W / Hindley–Milner）。

v1（:func:`infer` 缺省模式）行为与留痕格式完全保留；
v2（``infer(..., version=2)`` 与模块检查）在同一套数据模型上前扩展：

* **开放行记录**：``record`` / ``field`` / ``extend``；行变量是
  ``KIND_ROW`` 的 :class:`TVar`，行统一吸收差异字段，行变量 occurs
  检查拒绝开放行成环；
* **效应**：``read`` / ``write`` / ``exn`` 三标签 + 潜伏在
  :class:`TFun` 中的效应元变量（``KIND_EFF``）。``try`` 精确移除
  已确定的 ``exn`` 标签（其余标签与未决元变量保留）；
* **值限制 + 危险变量**：扩展式不泛化；值形式泛化时再排除“出现在
  Ref(...) 内的自由类型变量”（携带可变引用的变量不得逃逸泛化）。

模块边界（:mod:`polyinfer.modules`）在此之上做显式签名的 skolem
subsumption：效应包含检查（实现效应 ⊆ 声明效应）与 rigid skolem
检查，堵住引用写入/异常处理谎报纯多态。
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from itertools import count
import sys
from typing import Any, Dict, FrozenSet, List, Tuple

from . import ast as ast_mod
from .errors import (
    DuplicateFieldError,
    InfiniteTypeError,
    MissingFieldError,
    TypeMismatchError,
    UnboundVariableError,
)
from .types import (
    BOOL,
    EXN,
    INT,
    KIND_EFF,
    KIND_ROW,
    KIND_TYPE,
    UNIT,
    ROW_EMPTY,
    RowEmpty,
    Subst,
    TCon,
    TEffect,
    TFun,
    TRecord,
    TRef,
    TRow,
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
EffSet = FrozenSet[Any]  # str 标签 或 KIND_EFF TVar

# v1 值形式；v2 中 record 仅当全部字段子表达式也是值形式时才是值。
_VALUE_TAGS = frozenset({"int", "bool", "lambda", "var", "exn"})


def _is_value(node: ast_mod.Node) -> bool:
    tag = node["tag"]
    if tag in _VALUE_TAGS:
        return True
    if tag == "record":
        return all(_is_value(pair[1]) for pair in node["fields"])
    return False


def dangerous_vars(tp: Type, exempt: "set | None" = None) -> set:
    """出现在任意 Ref(...) 内部的普通类型变量（Tofte 危险变量，保守版）。

    ``exempt`` 中的变量（典型：lambda 参数变量）不按危险变量计——
    它们在每次调用时独立实例化，不产生跨实例共享的可变 cell。
    开放行尾行不展开（抽象行可能携带的字段在实例化点的不变式
    统一处独立把关）。
    """
    exempt = exempt or set()
    found: set = set()

    def walk(t: Type, under_ref: bool) -> None:
        if isinstance(t, TVar):
            if under_ref and t.kind == KIND_TYPE and t not in exempt:
                found.add(t)
            return
        if isinstance(t, TRef):
            walk(t.inner, True)
            return
        if isinstance(t, TFun):
            walk(t.arg, under_ref)
            walk(t.ret, under_ref)
        elif isinstance(t, (TRecord, TRow)):
            for _n, ft in t.fields:
                walk(ft, under_ref)

    walk(tp, False)
    return found


@dataclass(frozen=True)
class TypeScheme:
    """类型方案 ``forall quantified . t``（含普通/行/效应三类变量）。"""

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
    """一次 let 事件的留痕，供独立检查器复核闭包条件、值限制与危险变量。"""

    path: Path
    name: str
    bound_tag: str
    bound_type: Type
    generalized: Tuple[TVar, ...]
    env_snapshot: Tuple[Tuple[str, frozenset, Type], ...]
    # v2 新增（均带缺省，v1 结果/旧测试构造的替换形态不受影响）：
    is_value: bool = False
    dangerous_excluded: Tuple[TVar, ...] = ()
    bound_effect: frozenset = frozenset()
    # 泛化时刻已引入的 lambda 参数变量（危险变量豁免集留痕）。
    exempt_vars: Tuple[TVar, ...] = ()
    # record 绑定的各字段表达式 tag（供独立检查器重判 record 是否全值字段）。
    field_tags: Tuple[str, ...] = ()


@dataclass(frozen=True)
class InferenceResult:
    """成功推断结果。v1 字段全部保留，v2 字段追加在后。"""

    principal_type: str
    type: Type
    names: Dict[TVar, str]
    subst: Subst
    constraints: Tuple[Constraint, ...]
    generalizations: Tuple[Generalization, ...]
    node_count: int
    # v2：
    effects: frozenset = frozenset()          # 根表达式本次求值的效应标签
    eff_solution: Dict[int, frozenset] = dc_field(default_factory=dict)
    effect_log: tuple = ()                     # 有序 union/remove 效应事件
    version: int = 1

    def verify(self):
        """用**独立**替换器核验本结果（不经过统一核心）。"""
        from .reference import check_result

        return check_result(self)


class Inferencer:
    def __init__(self, version: int = 1, start: int = 0) -> None:
        self.version = version
        self._counter = count(start)
        self.constraints: List[Constraint] = []
        self.generalizations: List[Generalization] = []
        # 有序效应事件日志：("u", left, right) 合并；
        # ("r", items, removed_label, new_var_id) try 移除。
        # 独立检查器按序重放即可精确重建每个效应变量类的标签。
        self.effect_log: List[tuple] = []
        # 效应并查集（独立于不可变主替换，按效应变量 ident 索引）：
        # parent 给出类根；labels[root] 是该类累积到的具体效应标签。
        self.eff_parent: Dict[int, int] = {}
        self.eff_labels: Dict[int, frozenset] = {}
        # 所有 lambda 参数引入的类型变量（Tofte：这些变量即使出现在
        # Ref 内也允许在边界泛化——每次调用独立实例化，不产生共享 cell）。
        self.lambda_params: set = set()
        self.nodes_seen = 0
        # 行兄弟登记：开放尾行行变量 -> 在它“上方”确定声明的字段名->类型。
        # extend 的结果形如 {新字段 | ρ}，其中新字段不在被扩展记录 r 的
        # 已知视图里；当 ρ 后来被绑定暴露出同名字段时，必须逐点统一——
        # 这条非局部约束靠登记在尾行绑定时刻补回。登记随实例化复制到
        # 新行变量（多态方案的每次使用各自独立）。
        self.row_siblings: Dict[int, Dict[str, Type]] = {}

    # ------------------------------------------------------------ 工具

    def fresh(self, kind: int = KIND_TYPE) -> TVar:
        v = TVar(next(self._counter), kind)
        if kind == KIND_EFF:
            self.eff_parent[v.ident] = v.ident
            self.eff_labels[v.ident] = frozenset()
        return v

    def instantiate(self, scheme: TypeScheme) -> Type:
        """实例化类型方案：每个被限定变量换一个同 kind 的全新变量。

        效应变量不出现在方案里（泛化时其已决标签已烘焙为具体集合）。
        行兄弟登记随开放行变量复制到新变量——多态方案的每次使用各自
        独立，新使用点的行冲突写在新行变量上，不影响其他实例。
        """
        if not scheme.quantified:
            return scheme.t
        mapping = {v: self.fresh(v.kind) for v in scheme.quantified}
        for old, new in mapping.items():
            if old.kind == KIND_ROW and old.ident in self.row_siblings:
                self.row_siblings[new.ident] = {
                    n: _rename(t, mapping) for n, t in self.row_siblings[old.ident].items()
                }
        return _rename(scheme.t, mapping)

    # ------------------------------------------------------ 效应并查集

    def _eff_find(self, ident: int) -> int:
        root = ident
        chain = []
        while self.eff_parent.get(root, root) != root:
            chain.append(root)
            root = self.eff_parent[root]
        for node in chain:  # 路径压缩
            self.eff_parent[node] = root
        return root

    def eff_union(self, left: EffSet, right: EffSet) -> None:
        """把两个潜伏效应集合中的效应变量类合并、标签累积（不失败）。

        直接标签并入各变量类；两个效应变量同属一次比较时合并类。
        事件记入 ``effect_log`` 供独立检查器按序精确重放。
        """
        self.effect_log.append(("u", frozenset(left), frozenset(right)))
        l_vars = [i.ident for i in left if isinstance(i, TVar)]
        r_vars = [i.ident for i in right if isinstance(i, TVar)]
        labels = {x for x in left | right if isinstance(x, str)}
        roots = sorted({self._eff_find(i) for i in l_vars + r_vars})
        if not roots:
            return
        canon = roots[0]
        merged_labels = frozenset(labels) | frozenset().union(
            *(self.eff_labels.get(r, frozenset()) for r in roots)
        )
        for r in roots[1:]:
            self.eff_parent[r] = canon
        self.eff_labels[canon] = merged_labels
        for r in roots[1:]:
            self.eff_labels[r] = merged_labels

    def eff_resolve(self, items: EffSet) -> "tuple[frozenset, Dict[int, frozenset]]":
        """把 (标签 ∪ 效应变量) 集合解析为纯标签集合，给出各变量类的解。"""
        solution: Dict[int, frozenset] = {}
        out: set = set()
        for item in items:
            if isinstance(item, str):
                out.add(item)
            else:
                root = self._eff_find(item.ident)
                labels = self.eff_labels.get(root, frozenset())
                solution[item.ident] = labels
                out |= set(labels)
        return frozenset(out), solution

    def eff_full_solution(self) -> Dict[int, frozenset]:
        """全部效应变量的最终标签解（独立检查器据此精确核对）。"""
        return {
            ident: frozenset(self.eff_labels.get(self._eff_find(ident), frozenset()))
            for ident in self.eff_parent
        }

    def eff_solution_for(self, tp: Type) -> Dict[int, frozenset]:
        """渲染用：类型中所有效应变量的最终标签解（含非根类）。"""
        sol: Dict[int, frozenset] = {}
        for ev in _effect_vars_of(tp):
            root = self._eff_find(ev.ident)
            sol[ev.ident] = self.eff_labels.get(root, frozenset())
        return sol

    def bake_effects(self, tp: Type) -> Type:
        """把类型中潜伏的效应元变量替换为其当前已决标签集合（边界泛化用）。"""
        if isinstance(tp, TFun):
            labels, _ = self.eff_resolve(tp.eff)
            return TFun(self.bake_effects(tp.arg), self.bake_effects(tp.ret), labels)
        if isinstance(tp, TRef):
            return TRef(self.bake_effects(tp.inner))
        if isinstance(tp, TRecord):
            return TRecord(
                tuple((n, self.bake_effects(t)) for n, t in tp.fields), tp.tail
            )
        if isinstance(tp, TRow):
            return TRow(
                tuple((n, self.bake_effects(t)) for n, t in tp.fields), tp.tail
            )
        return tp

    # ------------------------------------------------------------ 统一

    def unify(
        self,
        s: Subst,
        left: Type,
        right: Type,
        left_path: Path,
        right_path: Path,
        note: str,
    ) -> Subst:
        """统一两类型并返回扩展后的替换（含函数效应合并与开放行统一）。"""
        l = apply(s, left)
        r = apply(s, right)
        self.constraints.append(Constraint(l, r, left_path, right_path, note))

        # 两个都是变量
        if isinstance(l, TVar) and isinstance(r, TVar):
            if l == r:
                return s
            if l.kind != r.kind:
                raise TypeMismatchError(left_path, right_path, l, r)
            target, other = (l, r) if l.ident < r.ident else (r, l)
            if occurs(target, other):
                raise InfiniteTypeError(left_path, right_path, target, other)
            s = bind(s, target, other)
            if target.kind == KIND_ROW:
                # 合并别名两侧的行兄弟登记；同名字段立即逐点统一。
                a = self.row_siblings.get(target.ident, {})
                b = self.row_siblings.get(other.ident, {})
                for n in sorted(set(a) & set(b)):
                    s = self.unify(
                        s, apply(s, a[n]), apply(s, b[n]),
                        left_path + ("[field]", n), right_path + ("[field]", n),
                        f"行别名两侧字段 {n} 冲突",
                    )
                self._merge_row_siblings(target, other)
            return s
        if isinstance(l, TVar):
            return self._bind_var(s, l, r, left_path, right_path)
        if isinstance(r, TVar):
            return self._bind_var(s, r, l, right_path, left_path)

        if isinstance(l, TCon) and isinstance(r, TCon):
            if l.name == r.name:
                return s
            raise TypeMismatchError(left_path, right_path, l, r)

        if isinstance(l, TFun) and isinstance(r, TFun):
            s1 = self.unify(
                s, l.arg, r.arg,
                left_path + ("[arg]",), right_path + ("[arg]",),
                f"{note}：参数",
            )
            self.eff_union(l.eff, r.eff)
            s3 = self.unify(
                s1, apply(s1, l.ret), apply(s1, r.ret),
                left_path + ("[ret]",), right_path + ("[ret]",),
                f"{note}：返回值",
            )
            return s3

        if isinstance(l, TRef) and isinstance(r, TRef):
            return self.unify(
                s, l.inner, r.inner,
                left_path + ("[inner]",), right_path + ("[inner]",),
                f"{note}：引用单元",
            )

        if isinstance(l, TRecord) and isinstance(r, TRecord):
            return self._unify_records(s, l, r, left_path, right_path, note)

        raise TypeMismatchError(left_path, right_path, l, r)

    def _bind_var(self, s, var, tp, left_path, right_path):
        if isinstance(tp, RowEmpty) or not self._kind_ok(var, tp):
            raise TypeMismatchError(left_path, right_path, var, tp)
        if occurs(var, tp):
            raise InfiniteTypeError(left_path, right_path, var, tp)
        return bind(s, var, tp)

    @staticmethod
    def _kind_ok(var: TVar, tp: Type) -> bool:
        if var.kind == KIND_TYPE:
            return not isinstance(tp, (TRow, TEffect, RowEmpty))
        if var.kind == KIND_ROW:
            return isinstance(tp, (TRow, TRecord))
        if var.kind == KIND_EFF:
            return isinstance(tp, TEffect)
        return False

    def _unify_records(self, s, l: TRecord, r: TRecord, left_path, right_path, note):
        """开放行统一。

        展开两侧尾行后：共有字段逐点统一；各自独有字段通过把尾行变量
        绑成 ``{独有字段 | 新尾行}`` 吸收；闭行侧不允许出现对侧独有
        字段；尾行变量被绑入含自身的行时由 occurs 检查拒绝（行成环）。
        """
        l_fields, l_tail = _expand_row(s, l.fields, l.tail)
        r_fields, r_tail = _expand_row(s, r.fields, r.tail)
        l_map = dict(l_fields)
        r_map = dict(r_fields)
        common = sorted(set(l_map) & set(r_map))
        l_only = [n for n in l_map if n not in r_map]
        r_only = [n for n in r_map if n not in l_map]

        # 1) 共有字段逐点一致
        for name in common:
            s = self.unify(
                s, l_map[name], r_map[name],
                left_path + ("[field]", name), right_path + ("[field]", name),
                f"{note}：字段 {name}",
            )
        l_map = {n: apply(s, t) for n, t in l_map.items()}
        r_map = {n: apply(s, t) for n, t in r_map.items()}

        # 2) 尾行吸收
        l_closed = l_tail is ROW_EMPTY
        r_closed = r_tail is ROW_EMPTY

        if l_closed and r_closed:
            if l_only or r_only:
                raise TypeMismatchError(left_path, right_path, l, r)
            return s
        if l_closed:
            if r_only:
                # 左行闭行（确定没有更多字段），右行却多出字段。
                raise TypeMismatchError(left_path, right_path, l, r)
            # ρ_r := {l_only | <>}
            return self._bind_row(s, r_tail, l_only, l_map, ROW_EMPTY,
                                  right_path, left_path)
        if r_closed:
            if l_only:
                raise TypeMismatchError(left_path, right_path, l, r)
            return self._bind_row(s, l_tail, r_only, r_map, ROW_EMPTY,
                                  left_path, right_path)
        # 两侧都开放
        if l_tail == r_tail:
            # 共享同一尾行 ρ：字段集差异必须由 ρ 吸收（ρ := {差异字段 | ρ'}）；
            # 若差异字段的类型里又含 ρ（开放行成环），occurs 检查在此拒绝。
            if not l_only and not r_only:
                return s
            extra_map = dict(l_fields)
            extra_map.update(dict(r_fields))
            new_tail = self.fresh(KIND_ROW)
            diff = sorted(set(l_only) | set(r_only))
            return self._bind_row(s, l_tail, diff, extra_map, new_tail,
                                  left_path, right_path)
        # ρ_l := {r_only | ρ_new}, ρ_r := {l_only | ρ_new}
        new_tail = self.fresh(KIND_ROW)
        s = self._bind_row(s, l_tail, r_only, r_map, new_tail,
                           left_path, right_path)
        s = self._bind_row(s, r_tail, l_only, l_map, new_tail,
                           right_path, left_path)
        return s

    def _bind_row(self, s, row_var, names, source_map, new_tail,
                  left_path, right_path):
        row = TRow(tuple((n, source_map[n]) for n in sorted(names)), new_tail)
        if occurs(row_var, row):
            raise InfiniteTypeError(left_path, right_path, row_var, row)
        # 绑定前：尾行上方已登记的兄弟字段与新暴露字段同名则逐点统一。
        sib = self.row_siblings.get(row_var.ident, {})
        for n in sorted(set(names) & set(sib)):
            s = self.unify(
                s, apply(s, sib[n]), apply(s, source_map[n]),
                left_path + ("[field]", n), right_path + ("[field]", n),
                f"开放行字段 {n} 与该尾行上方已扩展字段冲突",
            )
        s = bind(s, row_var, row)
        # 传播：尾行上方全部字段（旧兄弟 ∪ 新字段）成为新尾行的兄弟。
        if isinstance(new_tail, TVar):
            carried = dict(sib)
            for n in names:
                carried[n] = apply(s, source_map[n])
            reg = self.row_siblings.setdefault(new_tail.ident, {})
            for n, t in sorted(carried.items()):
                cur = reg.get(n)
                if cur is None:
                    reg[n] = t
                else:
                    s = self.unify(
                        s, apply(s, cur), apply(s, t),
                        left_path + ("[field]", n), right_path + ("[field]", n),
                        f"开放行字段 {n} 跨尾行冲突",
                    )
        return s

    def _register_siblings(self, tail, fields):
        """登记开放尾行 ``tail`` 上方确定出现的字段（字面量/扩展字段）。"""
        if not isinstance(tail, TVar):
            return
        reg = self.row_siblings.setdefault(tail.ident, {})
        for n, t in fields:
            reg.setdefault(n, t)

    def _merge_row_siblings(self, kept: TVar, dropped: TVar) -> None:
        """行变量别名 kept ≡ dropped：合并两者上方兄弟字段（同名须一致）。

        一致性在后续统一/绑定时刻由 :meth:`_bind_row` 逐点处理；这里只
        并表（类型留为变量，应用时取最新替换）。
        """
        if kept.kind != KIND_ROW or dropped.kind != KIND_ROW:
            return
        a = self.row_siblings.get(kept.ident, {})
        b = self.row_siblings.get(dropped.ident, {})
        merged = dict(a)
        for n, t in b.items():
            merged.setdefault(n, t)
        self.row_siblings[kept.ident] = merged
        self.row_siblings[dropped.ident] = merged

    # ------------------------------------------------------------ 规则

    def ti(self, node: ast_mod.Node, env: Dict[str, TypeScheme], s: Subst, path: Path):
        """推断表达式，返回 (subst, type, effects)。"""
        tag = node["tag"]
        method = self._DISPATCH.get(tag)
        if method is None:
            raise ast_mod.InvalidAstError(path, f"未知节点 tag {tag!r}")
        self.nodes_seen += 1
        return method(self, node, env, s, path)

    @staticmethod
    def _union(*sets: EffSet) -> EffSet:
        out: set = set()
        for x in sets:
            out |= set(x)
        return frozenset(out)

    def _ti_var(self, node, env, s, path):
        name = node["name"]
        scheme = env.get(name)
        if scheme is None:
            raise UnboundVariableError(path, name)
        return s, self.instantiate(scheme), frozenset()

    def _ti_mvar(self, node, env, s, path):
        key = (node["module"], node["name"])
        scheme = env.get(("mvar",) + key)
        if scheme is None:
            raise UnboundVariableError(path, f"{node['module']}.{node['name']}")
        return s, self.instantiate(scheme), frozenset()

    def _ti_int(self, node, env, s, path):
        return s, INT, frozenset()

    def _ti_bool(self, node, env, s, path):
        return s, BOOL, frozenset()

    def _ti_exn(self, node, env, s, path):
        return s, EXN, frozenset()

    def _ti_lambda(self, node, env, s, path):
        param_var = self.fresh()
        eff_var = self.fresh(KIND_EFF)
        self.lambda_params.add(param_var)
        body_env = dict(env)
        body_env[node["param"]] = TypeScheme(frozenset(), param_var)
        s1, body_t, body_eff = self.ti(node["body"], body_env, s, path + ("body",))
        # 函数潜伏效应 = 函数体效应；把该函数的效应元变量类并入函数体效应。
        self.eff_union(frozenset({eff_var}), body_eff)
        return s1, TFun(apply(s1, param_var), body_t, frozenset({eff_var})), frozenset()

    def _ti_apply(self, node, env, s, path):
        s1, fn_t, fn_eff = self.ti(node["fn"], env, s, path + ("fn",))
        s2, arg_t, arg_eff = self.ti(node["arg"], env, s1, path + ("arg",))
        ret_var = self.fresh()
        ret_eff_var = self.fresh(KIND_EFF)
        s3 = self.unify(
            s2,
            apply(s2, fn_t),
            TFun(arg_t, ret_var, frozenset({ret_eff_var})),
            path + ("fn",),
            path,
            "apply 要求被应用值是函数",
        )
        fn_latent = apply(s3, fn_t)
        latent = fn_latent.eff if isinstance(fn_latent, TFun) else frozenset()
        eff = self._union(
            fn_eff,
            arg_eff,
            latent,
        )
        return s3, apply(s3, ret_var), eff

    def _ti_let(self, node, env, s, path):
        s1, bound_raw, bound_eff = self.ti(node["bound"], env, s, path + ("bound",))
        bound_t0 = apply(s1, bound_raw)
        # 效应在方案中是具体标签（签名不做效应量化）：泛化前把潜伏效应
        # 元变量烘焙为已决标签，于是实例化得到的函数仍带正确效应。
        bound_t = self.bake_effects(bound_t0)
        s1_baked = s1
        name = node["name"]
        bound_tag = node["bound"]["tag"]
        if self.version >= 2:
            is_value = _is_value(node["bound"])
        else:
            is_value = bound_tag in _VALUE_TAGS

        env_free = _env_free_vars(env, s1_baked)
        candidates = tuple(v for v in free_tvars(bound_t) if v not in env_free)
        # λ 参数豁免要按替换后的等价类算：参数变量经统一可能被别名
        # 重定向（p ↦ q），最终代表 q 同样源自参数位置、每次调用独立
        # 实例化，不属于跨实例共享的危险 cell。
        param_reprs = {apply(s1_baked, p) for p in self.lambda_params}
        param_reprs = {v for v in param_reprs if isinstance(v, TVar)}
        dangerous_all = dangerous_vars(bound_t, param_reprs)
        dangerous = tuple(v for v in candidates if v in dangerous_all)
        if is_value:
            quantified = tuple(v for v in candidates if v not in dangerous_all)
        else:
            quantified = ()
        scheme = TypeScheme(frozenset(quantified), bound_t)

        bound_eff_labels, _ = self.eff_resolve(bound_eff)
        exempt_snapshot = tuple(sorted(param_reprs, key=lambda x: (x.kind, x.ident)))
        field_tags = (
            tuple(pair[1]["tag"] for pair in node["bound"]["fields"])
            if node["bound"]["tag"] == "record"
            else ()
        )
        self.generalizations.append(
            Generalization(
                path=path,
                name=name,
                bound_tag=bound_tag,
                bound_type=bound_t,
                generalized=quantified,
                env_snapshot=tuple(
                    (ename, esch.quantified, apply(s1_baked, esch.t))
                    for ename, esch in env.items()
                    if isinstance(ename, str)
                ),
                is_value=is_value,
                dangerous_excluded=dangerous if is_value else (),
                bound_effect=frozenset(bound_eff_labels),
                exempt_vars=exempt_snapshot,
                field_tags=field_tags,
            )
        )

        body_env = dict(env)
        body_env[name] = scheme
        s2, body_t, body_eff = self.ti(node["body"], body_env, s1, path + ("body",))
        return s2, body_t, self._union(
            bound_eff, body_eff
        )

    def _ti_if(self, node, env, s, path):
        s1, cond_t, e_cond = self.ti(node["cond"], env, s, path + ("cond",))
        s2 = self.unify(
            s1, apply(s1, cond_t), BOOL,
            path + ("cond",), path, "if 的条件必须是 Bool",
        )
        s3, then_t, e_then = self.ti(node["then"], env, s2, path + ("then",))
        s4, else_t, e_else = self.ti(node["else_"], env, s3, path + ("else_",))
        s5 = self.unify(
            s4, apply(s4, then_t), apply(s4, else_t),
            path + ("then",), path + ("else_",), "if 两个分支类型必须一致",
        )
        eff = self._union(
            e_cond,
            e_then,
            e_else,
        )
        return s5, apply(s5, then_t), eff

    def _ti_ref(self, node, env, s, path):
        s1, inner_t, eff = self.ti(node["inner"], env, s, path + ("inner",))
        return s1, TRef(inner_t), eff

    def _ti_read(self, node, env, s, path):
        s1, ref_t, e_ref = self.ti(node["ref"], env, s, path + ("ref",))
        inner_var = self.fresh()
        s2 = self.unify(
            s1, apply(s1, ref_t), TRef(inner_var),
            path + ("ref",), path, "read 要求引用类型",
        )
        eff = self._union(e_ref, frozenset({"read"}))
        return s2, apply(s2, inner_var), eff

    def _ti_write(self, node, env, s, path):
        s1, ref_t, e_ref = self.ti(node["ref"], env, s, path + ("ref",))
        s2, val_t, e_val = self.ti(node["value"], env, s1, path + ("value",))
        s3 = self.unify(
            s2, apply(s2, ref_t), TRef(apply(s2, val_t)),
            path + ("ref",), path + ("value",),
            "write 要求左侧引用的单元类型等于右侧表达式类型",
        )
        eff = self._union(
            e_ref,
            e_val,
            frozenset({"write"}),
        )
        return s3, UNIT, eff

    # ------------------------------------------------------ v2: 记录

    def _ti_record(self, node, env, s, path):
        fields = []
        eff: EffSet = frozenset()
        for i, pair in enumerate(node["fields"]):
            fname, fexpr = pair
            s, ft, fe = self.ti(fexpr, env, s, path + ("fields", f"[{i}]", "value"))
            fields.append((fname, ft))
            eff = self._union(eff, fe)
        tail = self.fresh(KIND_ROW)  # 记录字面量默认是开放行
        result_t = TRecord(tuple(fields), tail)
        self._register_siblings(tail, tuple(fields))
        return s, result_t, eff

    def _ti_field(self, node, env, s, path):
        s1, rec_t, rec_eff = self.ti(node["record"], env, s, path + ("record",))
        fname = node["name"]
        rec_resolved = apply(s1, rec_t)
        # 仅当类型已确定为记录且闭行时，缺字段才是构造性拒绝；
        # 仍是未绑定变量/开放行时交给下面的统一吸收。
        if isinstance(rec_resolved, (TRecord, TRow)):
            known_pairs, known_tail = _row_view(s1, rec_resolved)
            if known_tail is ROW_EMPTY and fname not in dict(known_pairs):
                raise MissingFieldError(
                    path + ("record",), path, rec_resolved,
                    TRecord(((fname, self.fresh()),), ROW_EMPTY),
                )
        field_var = self.fresh()
        tail_var = self.fresh(KIND_ROW)
        s2 = self.unify(
            s1,
            rec_resolved,
            TRecord(((fname, field_var),), tail_var),
            path + ("record",),
            path,
            f"字段访问 {fname!r}",
        )
        return s2, apply(s2, field_var), rec_eff

    def _ti_extend(self, node, env, s, path):
        s1, rec_t0, rec_eff = self.ti(node["record"], env, s, path + ("record",))
        # 要求源是记录：与一个“已知字段为空、尾行开放”的记录统一，
        # 由此学到源的字段集合与其（共享的）尾行 ρ。
        src_tail = self.fresh(KIND_ROW)
        s1 = self.unify(
            s1, apply(s1, rec_t0), TRecord((), src_tail),
            path + ("record",), path, "extend 的对象必须是记录",
        )
        known_pairs, final_tail = _row_view(s1, apply(s1, rec_t0))
        old_view = dict(known_pairs)
        new_fields: list[Tuple[str, Type]] = []
        eff = rec_eff
        for i, pair in enumerate(node["fields"]):
            fname, fexpr = pair
            if fname in old_view:
                # 同名字段扩展一律报错（含经行替换暴露出的确定字段）。
                raise DuplicateFieldError(path + ("fields", f"[{i}]"), fname)
            s1, ft, fe = self.ti(fexpr, env, s1, path + ("fields", f"[{i}]", "value"))
            eff = self._union(eff, fe)
            new_fields.append((fname, ft))
            old_view[fname] = ft
        # 非破坏式：结果记录 = 源字段（快照）+ 新字段，**共享源的尾行 ρ**，
        # 但不回写源记录的字段列表——原记录类型本身不增加字段。
        result_fields = known_pairs + tuple(new_fields)
        result = TRecord(result_fields, final_tail)
        # 结果的全部确定字段登记到共享尾行上方：ρ 之后若被绑定暴露
        # 同名字段，_bind_row 会逐点统一（这是“同名字段扩展一律报错”
        # 在抽象尾行情形下的非局部落点）。
        self._register_siblings(final_tail, result_fields)
        return s1, result, eff

    # ------------------------------------------------------ v2: 异常

    def _ti_raise(self, node, env, s, path):
        s1, val_t, eff = self.ti(node["value"], env, s, path + ("value",))
        s2 = self.unify(
            s1, apply(s1, val_t), EXN,
            path + ("value",), path, "raise 的值必须是 Exn",
        )
        return s2, self.fresh(), self._union(
            eff, frozenset({"exn"})
        )

    def _ti_try(self, node, env, s, path):
        s1, body_t, body_eff = self.ti(node["body"], env, s, path + ("body",))
        s2, handler_t, handler_eff = self.ti(
            node["handler"], env, s1, path + ("handler",)
        )
        result_var = self.fresh()
        h_latent_var = self.fresh(KIND_EFF)
        # handler : Exn -> β（潜伏效应在异常路径执行，需计入）。
        s3 = self.unify(
            s2,
            apply(s2, handler_t),
            TFun(EXN, result_var, frozenset({h_latent_var})),
            path + ("handler",),
            path,
            "try 的处理器必须是 Exn -> t 函数",
        )
        s4 = self.unify(
            s3, apply(s3, body_t), apply(s3, result_var),
            path + ("body",), path + ("handler",),
            "try 的正常结果与处理器结果必须一致",
        )
        h_fn = apply(s4, handler_t)
        h_latent = h_fn.eff if isinstance(h_fn, TFun) else frozenset()
        # exn 被精确处理：body 的直接标签去掉 exn；body 的效应变量按其
        # 类当前标签去掉 exn 后由新变量代表（不回写共享类，避免污染
        # 与 handler 等共享的其他用途）。
        remaining = self._eff_remove_label(body_eff, "exn")
        eff = self._union(
            remaining,
            h_latent,
            handler_eff,
        )
        return s4, apply(s4, result_var), eff

    def _eff_remove_label(self, items: EffSet, label: str) -> EffSet:
        """从 (标签 ∪ 效应变量) 集合中语义移除一个标签（try 处理）。

        纯标签直接过滤；同一并查类根只生成一个新效应变量代表“去掉该
        标签后的剩余标签”。移除事件记入 ``effect_log``，检查器据此
        精确重建（含 write 保留、exn 被移除）。try 是其结果效应构造的
        最后一步，新变量之后不再与旧类回并。
        """
        out: set = {x for x in items if isinstance(x, str) and x != label}
        root_to_new: Dict[int, TVar] = {}
        trace: list = []
        for item in items:
            if not isinstance(item, TVar):
                continue
            root = self._eff_find(item.ident)
            labels = frozenset(self.eff_labels.get(root, frozenset())) - {label}
            nv = None
            if labels:
                nv = root_to_new.get(root)
                if nv is None:
                    nv = self.fresh(KIND_EFF)
                    root_to_new[root] = nv
                    self.eff_labels[self._eff_find(nv.ident)] = labels
                out.add(nv)
            trace.append((item.ident, root, None if nv is None else nv.ident))
        self.effect_log.append(
            ("r", frozenset(items), label, tuple(trace))
        )
        return frozenset(out)

    _DISPATCH = {
        "var": _ti_var,
        "mvar": _ti_mvar,
        "int": _ti_int,
        "bool": _ti_bool,
        "exn": _ti_exn,
        "lambda": _ti_lambda,
        "apply": _ti_apply,
        "let": _ti_let,
        "if": _ti_if,
        "ref": _ti_ref,
        "read": _ti_read,
        "write": _ti_write,
        "record": _ti_record,
        "field": _ti_field,
        "extend": _ti_extend,
        "raise": _ti_raise,
        "try": _ti_try,
    }


# ---------------------------------------------------------------- 行工具

def _expand_row(s: Subst, fields, tail):
    """沿替换展开 (fields, tail) 为 ((名字, 类型)..., 最终尾行)。"""
    out = list(fields)
    while isinstance(tail, TVar):
        bound = s.get(tail)
        if bound is None:
            break
        if isinstance(bound, TRow):
            out.extend(bound.fields)
            tail = bound.tail
        elif isinstance(bound, TRecord):
            out.extend(bound.fields)
            tail = bound.tail
        else:
            break
    return tuple((n, apply(s, t)) for n, t in out), tail


def _row_view(s: Subst, tp: Type):
    """类型的记录字段视图：((名字, 类型)..., 尾行)；非记录给空视图。"""
    tp = apply(s, tp)
    if isinstance(tp, (TRecord, TRow)):
        return _expand_row(s, tp.fields, tp.tail)
    return (), ROW_EMPTY


# ---------------------------------------------------------------- 杂项

def _rename(tp: Type, mapping: Dict[TVar, TVar]) -> Type:
    if isinstance(tp, TVar):
        return mapping.get(tp, tp)
    if isinstance(tp, TFun):
        eff = frozenset(
            mapping.get(i, i) if isinstance(i, TVar) else i for i in tp.eff
        )
        return TFun(_rename(tp.arg, mapping), _rename(tp.ret, mapping), eff)
    if isinstance(tp, TRef):
        return TRef(_rename(tp.inner, mapping))
    if isinstance(tp, TRecord):
        return TRecord(
            tuple((n, _rename(t, mapping)) for n, t in tp.fields),
            _rename(tp.tail, mapping) if isinstance(tp.tail, TVar) else tp.tail,
        )
    if isinstance(tp, TRow):
        return TRow(
            tuple((n, _rename(t, mapping)) for n, t in tp.fields),
            _rename(tp.tail, mapping) if isinstance(tp.tail, TVar) else tp.tail,
        )
    if isinstance(tp, TEffect):
        return TEffect(frozenset(mapping.get(i, i) if isinstance(i, TVar) else i
                                 for i in tp.items))
    return tp


def _env_free_vars(env: Dict[str, TypeScheme], s: Subst):
    """环境自由变量：各方案体类型自由变量减去其被限定变量（按 kind 区分）。"""
    free = set()
    for key, scheme in env.items():
        body = apply(s, scheme.t)
        for v in free_tvars(body):
            if v not in scheme.quantified:
                free.add(v)
    return free


def _effect_vars_of(tp: Type):
    """收集类型（含函数潜伏效应）里出现的全部效应变量。"""
    found = {}

    def walk(t: Type) -> None:
        if isinstance(t, TFun):
            for item in t.eff:
                if isinstance(item, TVar) and item.kind == KIND_EFF:
                    found[item] = None
            walk(t.arg)
            walk(t.ret)
        elif isinstance(t, TRef):
            walk(t.inner)
        elif isinstance(t, (TRecord, TRow)):
            for _n, ft in t.fields:
                walk(ft)

    walk(tp)
    return tuple(found)


def _strip_effects(tp: Type) -> Type:
    """v1 渲染用：把所有函数效应集合清空（v1 类型不含效应标注）。"""
    if isinstance(tp, TFun):
        return TFun(_strip_effects(tp.arg), _strip_effects(tp.ret), frozenset())
    if isinstance(tp, TRef):
        return TRef(_strip_effects(tp.inner))
    if isinstance(tp, TRecord):
        return TRecord(
            tuple((n, _strip_effects(t)) for n, t in tp.fields), tp.tail
        )
    return tp


# ---------------------------------------------------------------- 入口

def infer(ast: ast_mod.Node, max_nodes: int = 500, version: int = 1) -> InferenceResult:
    """推断整棵表达式 AST 的主类型。

    ``version=1``（缺省）：只接受 v1 语法，行为、异常与留痕与旧版一致。
    ``version=2``：额外接受记录/开放行、扩展、字段访问、raise/try/exn
    （``mvar`` 只允许出现在模块检查上下文），结果附带效应信息。
    """
    if isinstance(max_nodes, bool) or not isinstance(max_nodes, int):
        raise TypeError(f"max_nodes 必须是 int，实际为 {type(max_nodes).__name__}")
    if max_nodes < 1:
        raise ValueError(f"max_nodes 必须为正整数，实际为 {max_nodes}")
    if version not in (1, 2):
        raise ValueError(f"version 只支持 1 或 2，实际为 {version}")

    old_limit = sys.getrecursionlimit()
    needed = 200 * max_nodes + 1000
    if needed > old_limit:
        sys.setrecursionlimit(needed)
    try:
        validator = ast_mod.ExprValidator(max_nodes, version=version)
        validator.check(ast, (), scope=())
        checker = Inferencer(version=version)
        subst, root_t, root_eff_raw = checker.ti(ast, {}, {}, ())
        principal = apply(subst, root_t)
        if version == 1:
            # v1 契约：函数类型不带效应标注，主类型渲染保持旧格式。
            principal = _strip_effects(principal)
            names = canonical_names(principal)
            principal_str = render_type(principal, names)
            root_labels: frozenset = frozenset()
            eff_solution = {}
        else:
            names = canonical_names(principal)
            root_labels, _ = checker.eff_resolve(root_eff_raw)
            eff_solution = checker.eff_full_solution()
            principal_str = render_type(principal, names, eff_solution)
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
        effects=frozenset(root_labels),
        eff_solution=dict(eff_solution),
        effect_log=tuple(checker.effect_log),
        version=version,
    )
