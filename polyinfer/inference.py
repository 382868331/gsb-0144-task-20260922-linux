"""核心：带值限制的多态类型推断（Algorithm W / Hindley–Milner）。

支持两个语言版本（``infer(..., language=...)``）：

* ``"v1"``（默认）：与初版完全一致——10 个表达式节点、严格值限制、
  纯函数类型渲染 ``a -> b``；旧调用、旧异常类型、旧确定性结果不变。
* ``"v2"``：在 v1 之上增加开放行记录（record/get/extend）、三种效应
  （R 读 / W 写 / X 异常，含函数潜在效应与 try 对 X 的处理）、
  未决定应占位 evar。模块系统（:mod:`polyinfer.modules`）只在 v2 上工作。

设计要点
========

* **显式替换线程**：每个 ``ti_*`` 返回 ``(subst, type, effects)``，
  替换只通过 :mod:`polyinfer.types` 里的 ``apply``/``bind`` 组合推进；
  不使用可变并查集，失败只抛异常（纯函数语义）。
* **统一**：:meth:`Inferencer.unify` 自带 occurs 检查（含行变量成环）、
  构造子冲突、引用不变式、函数潜在效应集合一致。
* **效应**：直接效应随表达式收集；lambda 把体效应收为函数潜在效应；
  apply 放出被应用函数的潜在效应；``try`` 移除 body 的 X（R/W 保留）；
  ``evar`` 引入效应变量，最终仍有未决效应变量时结果如实标记 undecided。
* **值限制**：``let`` 只对非扩展（语法值）绑定泛化；v2 中记录字面量
  当且仅当所有字段都是语法值时才算值。
* **模块边界**：导出签名中的限定变量在实现检查时实例化为刚性 skolem；
  skolem 若经实现内分配的可变引用（0 层单元变量）逃逸，即拒绝——
  携带可变引用的类型变量不能在模块边界被“泛化”出来。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

from . import ast as ast_mod
from .errors import (
    EffectMismatchError,
    InfiniteTypeError,
    RecordFieldError,
    SkolemEscapeError,
    TypeMismatchError,
    UnboundVariableError,
)
from .types import (
    BOOL,
    INT,
    UNIT,
    REmpty,
    RSkol,
    RVar,
    Subst,
    TCon,
    TFun,
    TRec,
    TRef,
    TRow,
    TSkol,
    TVar,
    Type,
    apply,
    bind,
    canonical_names,
    free_rvars,
    free_skolems,
    free_tvars,
    occurs,
    render_type,
)

Path = Tuple[Any, ...]

# v1 的语法值集合；v2 额外按字段判定 record。
V1_VALUE_TAGS = frozenset({"int", "bool", "lambda", "var"})
V2_VALUE_TAGS = frozenset(V1_VALUE_TAGS | {"record"})


# ---------------------------------------------------------------- 效应变量

@dataclass(frozen=True)
class EffVar:
    """效应变量（仅由 ``evar`` 占位与含它的函数类型引入）。"""

    ident: int


# 效应：标签（"R"/"W"/"X"）与 EffVar 的冻结集合。
Effect = frozenset


def _eff_labels(eff: Effect) -> Set[str]:
    return {x for x in eff if isinstance(x, str)}


# ---------------------------------------------------------------- 留痕数据

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

    * ``bound_tag``：绑定表达式的 tag（记录时检查器自行递归判定 record
      字段是否全为语法值，不信任核心结论）；
    * ``bound_fields``：record 绑定的各字段表达式 tag（否则为空元组）；
    * ``bound_type``：泛化时刻绑定表达式的类型；
    * ``env_snapshot``：泛化时刻环境各方案的
      （名字, 被限定 TVar/RVar 集合, 当时替换下的体类型）。
    """

    path: Path
    name: str
    bound_tag: str
    bound_type: Type
    generalized: Tuple[Type, ...]
    env_snapshot: Tuple[Tuple[str, frozenset, Type], ...]
    bound_fields: Tuple[str, ...] = ()
    # 核心对“绑定是否为值”的判定留档；独立检查器自行重判，不信任此字段。
    bound_value_shape_ok: bool = True


@dataclass(frozen=True)
class EffectTrace:
    """效应推导留痕：每个产生效应的结构节点一条，供独立核对。"""

    path: Path
    tag: str
    produced: Tuple[str, ...]  # 该节点自身产生的具体标签
    latent: Tuple[str, ...] = ()  # lambda/签名函数收束的潜在标签
    handled: Tuple[str, ...] = ()  # try 移除的标签


@dataclass(frozen=True)
class RowAbsence:
    """开放行缺席约束留痕：``extend {e with f=v}`` 要求 e 的开放行尾
    ``row`` 中不含字段 ``field``。独立检查器据此在最终替换上复核，
    不信任求解器是否真的拦住了“同名字段塞回同一尾”的成环情形。"""

    path: Path
    field: str
    row: Any


@dataclass(frozen=True)
class InferenceResult:
    """成功推断结果（v1 字段语义与初版完全一致）。"""

    principal_type: str
    type: Type
    names: Dict[Type, str]
    subst: Subst
    constraints: Tuple[Constraint, ...]
    generalizations: Tuple[Generalization, ...]
    node_count: int
    # ---- v2 附加（v1 下取缺省，不改变旧字段与主类型字符串）----
    language: str = "v1"
    effects: frozenset = frozenset()
    effects_undecided: bool = False
    effect_traces: Tuple[EffectTrace, ...] = ()
    row_absences: Tuple[RowAbsence, ...] = ()
    # 效应变量替换（int ident -> 已解析效应集合），独立检查器据此解析
    # 约束记录中的潜在效应（约束是统一前形态，可能含未解析 EffVar）。
    eff_subst: Tuple[Tuple[int, frozenset], ...] = ()
    # 源 AST 引用：独立参考检查器需要它重算效应/值形状（不参与核心计算）。
    source_ast: Any = None

    def verify(self):
        """用**独立**替换器核验本结果（不经过统一核心）。"""
        from .reference import check_result

        return check_result(self)


def is_syntactic_value(node: ast_mod.Node) -> bool:
    """语法值判定（严格值限制，v2）：

    字面量、lambda、变量总是值；record 当且仅当每个字段表达式是值；
    get/extend 作用于不可变记录、且所有部件都是值时也算值
    （投影/扩展不分配可变状态、不产生效应）。其余（apply/ref/read/
    write/if/raise/try/evar）一律是扩展式。
    """
    tag = node["tag"]
    if tag in V1_VALUE_TAGS:
        return True
    if tag == "record":
        return all(is_syntactic_value(pair[1]) for pair in node["fields"])
    if tag == "get":
        return is_syntactic_value(node["record"])
    if tag == "extend":
        return (
            is_syntactic_value(node["record"])
            and is_syntactic_value(node["value"])
            and is_syntactic_value(node["rest"])
        )
    return False


# ---------------------------------------------------------------- 推断器

class Inferencer:
    def __init__(self, language: str = "v1") -> None:
        self.language = language
        # v1 完全不做效应收集：保证旧主类型字符串（含 Ref(a) -> Unit）不变。
        self.collect_effects = language == "v2"
        self._counter = count()
        self._eff_counter = count()
        self.constraints: List[Constraint] = []
        self.generalizations: List[Generalization] = []
        self.effect_traces: List[EffectTrace] = []
        self.row_absences: List[RowAbsence] = []
        # 效应变量替换：{int(ident): Effect}。
        self.eff_subst: Dict[int, Effect] = {}
        # 0 层（模块顶层，非任何 lambda 内）的“不逃逸变量”：
        # 严格值限制下，扩展式（apply/ref/read/write/if/raise/try/evar
        # 及含扩展式字段的 record）在顶层产生的自由类型/行变量不得在模块
        # 边界被绑定到刚性签名变量。这既包含本模块字面 ref 分配的单元变量，
        # 也包含经外部多态函数（如另一模块的 mk）应用产生的自由变量。
        self.unsafe_vars: Set[Any] = set()
        # 行变量的“缺席字段”事实：extend {r with f=v} 要求 f 不出现在
        # r 的开放尾中。键为行变量，值为该尾被声称不含的字段名集合。
        # 行变量被绑定到含这些字段的行时，必须报记录字段冲突
        # （否则同一字段会经尾展开出现两次——开放行成环的核心防线）。
        self.row_absent: Dict[RVar, Set[str]] = {}
        # skolem -> 经逃逸检查时使用的信息
        self.level = 0
        # 未捕获 raise 的处理栈：try 帧保存 handler 参数变量。
        self.raise_frames: List[TVar] = []
        # 模块检查时由 ModuleChecker 注入：{(模块名, 导出名): TypeScheme}。
        self.external_env: Dict[tuple, TypeScheme] | None = None

    # ------------------------------------------------------------ 工具

    def fresh(self) -> TVar:
        return TVar(next(self._counter))

    def fresh_rvar(self) -> RVar:
        return RVar(next(self._counter))

    def fresh_skol(self, name: str) -> TSkol:
        return TSkol(next(self._counter), name)

    def fresh_rskol(self, name: str) -> RSkol:
        return RSkol(next(self._counter), name)

    def fresh_effvar(self) -> EffVar:
        return EffVar(next(self._eff_counter))

    def instantiate(self, scheme: TypeScheme) -> Type:
        """实例化类型方案：每个被限定变量换一个全新变量。"""
        if not scheme.quantified:
            return scheme.t
        mapping: Dict[Type, Type] = {}
        for v in scheme.quantified:
            mapping[v] = self.fresh_rvar() if isinstance(v, RVar) else self.fresh()
        return _rename(scheme.t, mapping)

    def eff_resolve(self, eff: Effect, seen: Optional[Set[int]] = None) -> Effect:
        """把效应集合解析到范式（具体标签 ∪ 残余效应变量）。"""
        if seen is None:
            seen = set()
        out: Set[Any] = set()
        for item in eff:
            if isinstance(item, EffVar):
                mapped = self.eff_subst.get(item.ident)
                if mapped is None:
                    out.add(item)
                else:
                    if item.ident in seen:
                        out.add(item)  # 环防御：效应绑定无环，这里不可达
                        continue
                    out |= self.eff_resolve(mapped, seen | {item.ident})
            else:
                out.add(item)
        return frozenset(out)

    def _join(self, *effs: Effect) -> Effect:
        """效应并集（v1 下恒为空集，旧行为完全不变）。"""
        if not self.collect_effects:
            return frozenset()
        out: Set[Any] = set()
        for e in effs:
            out |= e
        return self.eff_resolve(frozenset(out))

    def bind_effvar(self, ident: int, eff: Effect) -> None:
        """把效应变量绑定到已解析效应；旧绑定右端同步推进（替换组合）。

        注意替换时要把像**展平**并入（而不是把整个效应集合当成一个元素）。
        """
        rhs = self.eff_resolve(eff)
        target = EffVar(ident)
        for k in list(self.eff_subst):
            new_items: Set[Any] = set()
            for x in self.eff_subst[k]:
                if x == target:
                    new_items |= set(rhs)
                else:
                    new_items.add(x)
            self.eff_subst[k] = self.eff_resolve(frozenset(new_items))
        self.eff_subst[ident] = rhs

    def unify_effects(
        self,
        left: Effect,
        right: Effect,
        left_path: Path,
        right_path: Path,
        note: str,
    ) -> None:
        """统一两个潜在效应集合（idempotent，类似行尾吸收）。

        设两侧范式为“具体标签 la/lb ∪ 效应变量 va/vb”：

        * 两侧都无变量：标签集合必须完全相等；
        * 仅一侧有变量：另一侧的具体标签必须包含本侧标签，差额由变量吸收；
        * 两侧都有变量：引入一个共享新效应变量 γ，
          左变量 ↦ 右侧独有标签 ∪ {γ}，右变量 ↦ 左侧独有标签 ∪ {γ}。

        例：``{β}`` 与 ``{W}`` 得 β ↦ {W}；``{W}`` 与 ``{}`` 直接冲突。
        """
        a = self.eff_resolve(left)
        b = self.eff_resolve(right)
        la = _eff_labels(a)
        lb = _eff_labels(b)
        va = {x for x in a if isinstance(x, EffVar)}
        vb = {x for x in b if isinstance(x, EffVar)}

        if not va and not vb:
            if la != lb:
                raise EffectMismatchError(
                    left_path, right_path, frozenset(la), frozenset(lb), note
                )
            return
        if va and not vb:
            if not (la <= lb):
                raise EffectMismatchError(
                    left_path, right_path, frozenset(la), frozenset(lb), note
                )
            image = frozenset(lb - la)
            for v in va:
                self.bind_effvar(v.ident, image)
            return
        if vb and not va:
            if not (lb <= la):
                raise EffectMismatchError(
                    left_path, right_path, frozenset(la), frozenset(lb), note
                )
            image = frozenset(la - lb)
            for w in vb:
                self.bind_effvar(w.ident, image)
            return
        # 两侧都有变量：共享新尾 γ。
        gamma = self.fresh_effvar()
        for v in va:
            self.bind_effvar(v.ident, frozenset(lb - la) | frozenset({gamma}))
        for w in vb:
            self.bind_effvar(w.ident, frozenset(la - lb) | frozenset({gamma}))

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
        """统一两类型并返回扩展后的替换（路径随结构下钻）。"""
        l = apply(s, left)
        r = apply(s, right)
        self.constraints.append(Constraint(l, r, left_path, right_path, note))

        # 刚性 skolem 规则：
        # * skolem 只与自身（ident 相同）相等；
        # * 柔性变量可以被绑定到 skolem（实现侧自由变量被确定为该刚性量，
        #   这正是“实现至少与签名一样一般”的含义），但 0 层可变单元变量
        #   绑定到含 skolem 的类型时由 _check_skolem_escape 拒绝；
        # * skolem 与具体类型（Int/函数/记录…）相遇即证明实现不够一般。
        if isinstance(l, TSkol) and isinstance(r, TSkol):
            if l.ident == r.ident:
                return s
            raise TypeMismatchError(left_path, right_path, l, r)
        if isinstance(l, TSkol):
            if isinstance(r, TVar):
                if occurs(r, l):
                    raise InfiniteTypeError(right_path, left_path, r, l)
                return self.do_bind(s, r, l)
            raise TypeMismatchError(left_path, right_path, l, r)
        if isinstance(r, TSkol):
            if isinstance(l, TVar):
                if occurs(l, r):
                    raise InfiniteTypeError(left_path, right_path, l, r)
                return self.do_bind(s, l, r)
            raise TypeMismatchError(left_path, right_path, l, r)

        if isinstance(l, TVar):
            if isinstance(r, TVar) and l == r:
                return s
            if occurs(l, r):
                raise InfiniteTypeError(left_path, right_path, l, r)
            return self.do_bind(s, l, r)
        if isinstance(r, TVar):
            if occurs(r, l):
                raise InfiniteTypeError(right_path, left_path, r, l)
            return self.do_bind(s, r, l)

        # 行变量（只可能出现在行尾位置；这里处理裸行变量相遇）。
        if isinstance(l, RVar) or isinstance(r, RVar):
            return self._unify_row_terminals(s, l, r, left_path, right_path)

        if isinstance(l, TCon) and isinstance(r, TCon):
            if l.name == r.name:
                return s
            raise TypeMismatchError(left_path, right_path, l, r)

        if isinstance(l, TFun) and isinstance(r, TFun):
            self.unify_effects(
                l.eff, r.eff,
                left_path + ("[eff]",), right_path + ("[eff]",),
                f"{note}：函数潜在效应",
            )
            s1 = self.unify(
                s, l.arg, r.arg,
                left_path + ("[arg]",), right_path + ("[arg]",),
                f"{note}：参数",
            )
            s2 = self.unify(
                s1, apply(s1, l.ret), apply(s1, r.ret),
                left_path + ("[ret]",), right_path + ("[ret]",),
                f"{note}：返回值",
            )
            return s2

        if isinstance(l, TRef) and isinstance(r, TRef):
            return self.unify(
                s, l.inner, r.inner,
                left_path + ("[inner]",), right_path + ("[inner]",),
                f"{note}：引用单元",
            )

        if isinstance(l, TRec) and isinstance(r, TRec):
            return self._unify_rows(
                s, l.row, r.row, left_path, right_path, note
            )

        # TRow 裸行（记录内部使用）相遇。
        if isinstance(l, TRow) and isinstance(r, TRow):
            return self._unify_rows(s, l, r, left_path, right_path, note)

        raise TypeMismatchError(left_path, right_path, l, r)

    def _check_skolem_escape(self, flex: Type, target: Type) -> None:
        """flex 将被绑定到 target（含 skolem）：若 flex 是 0 层扩展式
        产生的不逃逸变量，则拒绝——严格值限制下它不能经模块边界泛化。"""
        if self.level != 0:
            return
        if flex not in self.unsafe_vars:
            return
        skol = [t for t in free_tvars(target) if isinstance(t, TSkol)]
        if skol:
            raise SkolemEscapeError(flex, target, skol)

    def do_bind(self, s: Subst, var: Type, tp: Type) -> Subst:
        """统一用的绑定入口：bind 之后在 0 层传播“不逃逸”标记。

        * 若被绑定变量不安全，其右端出现的自由变量也标记为不安全；
        * 若右端含不安全变量，被绑定变量也标记为不安全；
        * 右端含 skolem 且变量不安全时，由 :meth:`_check_skolem_escape` 拒绝。
        这样替换链上的传递（β↦γ 后 γ↦$a）不会绕过逃逸防线。
        """
        rhs = apply(s, tp)
        # 行变量缺席约束：var 被绑定的行不得提供其缺席字段；
        # 缺席事实沿新尾终端继续传播（开放尾吸收字段时的成环防线）。
        if isinstance(var, RVar) and var in self.row_absent:
            absent = self.row_absent[var]
            if isinstance(rhs, TRow):
                fields_in_rhs, end_terminal = self._norm_row(s, rhs)
            else:
                fields_in_rhs, end_terminal = {}, rhs
            clash = absent & set(fields_in_rhs)
            if clash:
                raise RecordFieldError(
                    (), (), sorted(clash)[0],
                    f"开放行尾 {var} 已声明不含字段 {sorted(clash)}（extend 约束），"
                    f"统一却把该字段塞入同一行（开放行成环/重复）",
                )
            if isinstance(end_terminal, RVar):
                merged = absent | self.row_absent.get(end_terminal, set())
                self.row_absent[end_terminal] = set(merged)
        # 注意：unsafe_vars 的**标记**只在 0 层发生（顶层扩展式），但检查
        # 不能受当前 level 限制——该变量可能在导出 lambda 体内（level 1）
        # 才被绑定到签名 skolem；顶层事实一旦成立，任何层级都不得洗白。
        target_vars = set(_all_free(rhs))
        target_skol = list(free_skolems(rhs))
        if var in self.unsafe_vars and target_skol:
            raise SkolemEscapeError(var, rhs, target_skol)
        if var in self.unsafe_vars:
            self.unsafe_vars |= {v for v in target_vars if not isinstance(v, (TSkol, RSkol))}
        if any(v in self.unsafe_vars for v in target_vars):
            if target_skol:
                raise SkolemEscapeError(var, rhs, target_skol)
            if isinstance(var, (TVar, RVar)):
                self.unsafe_vars.add(var)
        return bind(s, var, rhs)

    # -- 行统一 -------------------------------------------------------

    def _norm_row(self, s: Subst, row: TRow):
        """沿替换把行尾展平：返回 (排序字段 dict, 终端 RVar/REmpty)。"""
        fields: Dict[str, Type] = {}
        cur: Type = row
        guard = 0
        while isinstance(cur, TRow):
            for n, t in cur.fields:
                if n in fields:
                    # 同一侧经替换带入重复字段：行成环/重复，按记录字段错误拒绝。
                    raise RecordFieldError(
                        (), (), n,
                        f"行的尾变量展开后两次提供字段 {n!r}（开放行成环/重复）",
                    )
                fields[n] = t
            cur = apply(s, cur.rest)
            guard += 1
            if guard > len(s) + 4:  # 防御性：替换无环，正常不会触发
                raise InfiniteTypeError((), (), cur, cur)
        if not isinstance(cur, (RVar, REmpty, RSkol)):
            raise TypeMismatchError((), (), cur, REmpty())
        return fields, cur

    def _unify_row_terminals(self, s, l, r, left_path, right_path):
        """裸行变量/刚性行变量（记录外）相遇：只允许同类自身相等，
        或柔性行变量吸收为刚性行变量。"""
        if isinstance(l, RVar) and isinstance(r, RVar):
            if l == r:
                return s
            if occurs(l, r):
                raise InfiniteTypeError(left_path, right_path, l, r)
            return self.do_bind(s, l, r)
        if isinstance(l, RSkol) and isinstance(r, RSkol):
            if l.ident == r.ident:
                return s
            raise TypeMismatchError(left_path, right_path, l, r)
        if isinstance(l, REmpty) and isinstance(r, REmpty):
            return s
        if isinstance(l, RVar) and isinstance(r, RSkol):
            return self.do_bind(s, l, r)
        if isinstance(r, RVar) and isinstance(l, RSkol):
            return self.do_bind(s, r, l)
        raise TypeMismatchError(left_path, right_path, l, r)

    def _unify_rows(
        self, s: Subst, r1: TRow, r2: TRow,
        left_path: Path, right_path: Path, note: str,
    ) -> Subst:
        f1, t1 = self._norm_row(s, r1)
        f2, t2 = self._norm_row(s, r2)
        names1 = set(f1)
        names2 = set(f2)
        # 共有字段：类型逐点统一。
        for n in sorted(names1 & names2):
            s = self.unify(
                s, f1[n], f2[n],
                left_path + ("[field]", n), right_path + ("[field]", n),
                f"{note}：字段 {n}",
            )
        only1 = names1 - names2
        only2 = names2 - names1

        if isinstance(t1, REmpty) and isinstance(t2, REmpty):
            if only1 or only2:
                missing = sorted(only1 | only2)
                raise RecordFieldError(
                    left_path, right_path, missing[0],
                    f"闭合记录缺少/多出字段 {missing}（{note}）",
                )
            return s

        if isinstance(t1, RVar) and isinstance(t2, REmpty):
            # f1 | t1 ≡ f2 | <>：t1 必须恰好吸收 only2；only1 必须为空。
            if only1:
                raise RecordFieldError(
                    left_path, right_path, sorted(only1)[0],
                    f"开放记录含闭合记录没有的字段 {sorted(only1)}（{note}）",
                )
            tail_row = TRow(
                tuple((n, f2[n]) for n in sorted(only2)), REmpty()
            )
            if occurs(t1, tail_row):
                raise InfiniteTypeError(left_path, right_path, t1, tail_row)
            return self.do_bind(s, t1, tail_row)

        if isinstance(t2, RVar) and isinstance(t1, REmpty):
            if only2:
                raise RecordFieldError(
                    left_path, right_path, sorted(only2)[0],
                    f"开放记录含闭合记录没有的字段 {sorted(only2)}（{note}）",
                )
            tail_row = TRow(
                tuple((n, f1[n]) for n in sorted(only1)), REmpty()
            )
            if occurs(t2, tail_row):
                raise InfiniteTypeError(right_path, left_path, t2, tail_row)
            return self.do_bind(s, t2, tail_row)

        # 刚性行尾与闭合行：形状固定的记录不能冒充开放行多态（引用不健全）。
        if isinstance(t1, RSkol) and isinstance(t2, REmpty) or (
            isinstance(t2, RSkol) and isinstance(t1, REmpty)
        ):
            extra = sorted(only1 | only2)
            raise RecordFieldError(
                left_path, right_path, extra[0] if extra else "<row>",
                f"闭合行尾与刚性开放行变量无法统一"
                f"（{'多余字段 ' + str(extra) if extra else '开放 vs 闭合'}；{note}）",
            )

        # 一侧刚性行变量（签名）、一侧柔性行变量（实现）：
        # 刚性侧代表“任意剩余”，不得被绑定；柔性尾必须被确定为
        # “刚性侧独有字段 | 同一刚性尾”；柔性侧自己多出的字段
        # （签名没提到、实现却要求）证明实现更不一般，必须拒绝。
        if isinstance(t1, RSkol) and isinstance(t2, RVar):
            if only2:
                raise RecordFieldError(
                    left_path, right_path, sorted(only2)[0],
                    f"实现行含签名开放行没有的字段 {sorted(only2)}"
                    f"（实现比签名更不一般；{note}）",
                )
            tail_row = TRow(
                tuple((n, f1[n]) for n in sorted(only1)), t1
            )
            if occurs(t2, tail_row):
                raise InfiniteTypeError(right_path, left_path, t2, tail_row)
            return self.do_bind(s, t2, tail_row)
        if isinstance(t2, RSkol) and isinstance(t1, RVar):
            if only1:
                raise RecordFieldError(
                    left_path, right_path, sorted(only1)[0],
                    f"实现行含签名开放行没有的字段 {sorted(only1)}"
                    f"（实现比签名更不一般；{note}）",
                )
            tail_row = TRow(
                tuple((n, f2[n]) for n in sorted(only2)), t2
            )
            if occurs(t1, tail_row):
                raise InfiniteTypeError(left_path, right_path, t1, tail_row)
            return self.do_bind(s, t1, tail_row)

        # 两侧都是刚性行变量。
        if isinstance(t1, RSkol) and isinstance(t2, RSkol):
            if t1.ident != t2.ident:
                raise TypeMismatchError(left_path, right_path, t1, t2)
            if only1 or only2:
                raise RecordFieldError(
                    left_path, right_path, sorted(only1 | only2)[0],
                    f"同一刚性开放行两端字段不一致 {sorted(only1 | only2)}（{note}）",
                )
            return s

        # 两侧都是柔性行变量。
        assert isinstance(t1, RVar) and isinstance(t2, RVar), (t1, t2)
        if t1 == t2:
            # 同一个开放尾：独占字段必须为空，否则该字段被提供两次。
            if only1 or only2:
                raise RecordFieldError(
                    left_path, right_path, sorted(only1 | only2)[0],
                    f"同一开放行的两端提供了互斥字段 {sorted(only1 | only2)}（{note}）",
                )
            return s
        # 两侧都是柔性行变量：交叉吸收。
        # f1|ρ1 ≡ f2|ρ2 ⇒ ρ1 ↦ only2 | γ，ρ2 ↦ only1 | γ（共享新尾 γ）：
        # 代入后两侧都等于 shared∪only1∪only2 | γ。
        fresh_tail = self.fresh_rvar()
        row_for_t1 = TRow(
            tuple((n, f2[n]) for n in sorted(only2)), fresh_tail
        )
        row_for_t2 = TRow(
            tuple((n, f1[n]) for n in sorted(only1)), fresh_tail
        )
        if occurs(t1, row_for_t1) or occurs(t2, row_for_t2):
            raise InfiniteTypeError(left_path, right_path, t1, row_for_t1)
        s = self.do_bind(s, t1, row_for_t1)
        s = self.do_bind(s, t2, row_for_t2)
        return s

    # ------------------------------------------------------------ 规则

    def ti(self, node: ast_mod.Node, env: Dict[str, TypeScheme], s: Subst, path: Path):
        tag = node["tag"]
        method = self._DISPATCH.get(tag)
        if method is None:  # 校验阶段已保证不可达（xvar 由模块检查器处理）
            raise ast_mod.InvalidAstError(path, f"未知节点 tag {tag!r}")
        s2, t, eff = method(self, node, env, s, path)
        # 模块顶层（0 层）的扩展式产生的自由变量一律标记为“不逃逸”：
        # 严格值限制下它们不得经模块边界绑定到刚性签名变量（也不得在
        # 顶层 let 中泛化）。值表达式不标记。
        if self.level == 0 and not is_syntactic_value(node):
            for v in _all_free(apply(s2, t)):
                if not isinstance(v, TSkol):
                    self.unsafe_vars.add(v)
        return s2, t, eff

    def _ti_var(self, node, env, s, path):
        name = node["name"]
        scheme = env.get(name)
        if scheme is None:
            raise UnboundVariableError(path, name)
        return s, self.instantiate(scheme), frozenset()

    def _ti_xvar(self, node, env, s, path):
        """跨模块外部引用（仅模块检查时可达；单表达式校验已拒绝 xvar）。"""
        if self.external_env is None:
            raise ast_mod.InvalidAstError(path, "xvar 出现在非模块推断中")
        key = (node["module"], node["name"])
        scheme = self.external_env.get(key)
        if scheme is None:  # ModuleValidator 已保证不可达
            raise UnboundVariableError(path, f"{key[0]}.{key[1]}")
        return s, self.instantiate(scheme), frozenset()

    def _ti_int(self, node, env, s, path):
        return s, INT, frozenset()

    def _ti_bool(self, node, env, s, path):
        return s, BOOL, frozenset()

    def _ti_lambda(self, node, env, s, path):
        param_var = self.fresh()
        body_env = dict(env)
        body_env[node["param"]] = TypeScheme(frozenset(), param_var)
        self.level += 1
        try:
            s1, body_t, body_eff = self.ti(node["body"], body_env, s, path + ("body",))
        finally:
            self.level -= 1
        latent = self.eff_resolve(body_eff) if self.collect_effects else frozenset()
        if self.collect_effects and (_eff_labels(latent)
                                    or any(isinstance(x, EffVar) for x in latent)):
            self.effect_traces.append(
                EffectTrace(
                    path=path, tag="lambda",
                    produced=(), latent=tuple(sorted(_eff_labels(latent))),
                )
            )
        ft = TFun(apply(s1, param_var), body_t, latent)
        # lambda 自身是值：直接效应为空（体效应被收进潜在效应）。
        return s1, ft, frozenset()

    def _ti_apply(self, node, env, s, path):
        s1, fn_t, e1 = self.ti(node["fn"], env, s, path + ("fn",))
        s2, arg_t, e2 = self.ti(node["arg"], env, s1, path + ("arg",))
        ret_var = self.fresh()
        # 应用点为函数潜在效应引入全新效应变量：多态函数每次使用独立实例，
        # 不会把不同调用点的效应相互钉死。v1 下 collect_effects=False，恒空。
        beta = frozenset({self.fresh_effvar()}) if self.collect_effects else frozenset()
        fn_resolved = apply(s2, fn_t)
        s3 = self.unify(
            s2, fn_resolved, TFun(arg_t, ret_var, beta),
            path + ("fn",), path,
            "apply 要求被应用值是函数",
        )
        latent_t = apply(s3, fn_t)
        latent_eff = latent_t.eff if isinstance(latent_t, TFun) else frozenset()
        eff = self._join(e1, e2, self.eff_resolve(latent_eff)) if self.collect_effects else frozenset()
        return s3, apply(s3, ret_var), eff

    def _ti_let(self, node, env, s, path):
        s1, bound_raw, bound_eff = self.ti(node["bound"], env, s, path + ("bound",))
        bound_t = apply(s1, bound_raw)
        name = node["name"]

        bound_node = node["bound"]
        bound_tag = bound_node["tag"]
        is_value = (
            is_syntactic_value(bound_node)
            if self.language == "v2"
            else bound_tag in V1_VALUE_TAGS
        )
        if is_value:
            env_free = _env_free_vars(env, s1)
            generalized = tuple(
                v for v in _all_free(bound_t)
                if v not in env_free and not isinstance(v, TSkol)
            )
            scheme = TypeScheme(frozenset(generalized), bound_t)
        else:
            generalized = ()
            scheme = TypeScheme(frozenset(), bound_t)

        bound_fields = (
            tuple(pair[0] for pair in bound_node["fields"])
            if bound_tag == "record" else ()
        )
        self.generalizations.append(
            Generalization(
                path=path,
                name=name,
                bound_tag=bound_tag,
                bound_type=bound_t,
                generalized=generalized,
                env_snapshot=tuple(
                    (ename, esch.quantified, apply(s1, esch.t))
                    for ename, esch in env.items()
                ),
                bound_fields=bound_fields,
                bound_value_shape_ok=is_value,
            )
        )

        body_env = dict(env)
        body_env[name] = scheme
        s2, body_t, body_eff = self.ti(node["body"], body_env, s1, path + ("body",))
        return s2, body_t, self._join(bound_eff, body_eff)

    def _ti_if(self, node, env, s, path):
        s1, cond_t, e1 = self.ti(node["cond"], env, s, path + ("cond",))
        s2 = self.unify(
            s1, apply(s1, cond_t), BOOL,
            path + ("cond",), path, "if 的条件必须是 Bool",
        )
        s3, then_t, e2 = self.ti(node["then"], env, s2, path + ("then",))
        s4, else_t, e3 = self.ti(node["else_"], env, s3, path + ("else_",))
        s5 = self.unify(
            s4, apply(s4, then_t), apply(s4, else_t),
            path + ("then",), path + ("else_",), "if 两个分支类型必须一致",
        )
        return s5, apply(s5, then_t), self._join(e1, e2, e3)

    def _ti_ref(self, node, env, s, path):
        s1, inner_t, e1 = self.ti(node["inner"], env, s, path + ("inner",))
        # ref 是扩展式：其单元内自由变量在 0 层不逃逸（ti 分发处也会按
        # 扩展式统一标记；这里额外立即标记，保证同表达式内后续绑定即受约束）。
        if self.level == 0:
            for v in _all_free(inner_t):
                if not isinstance(v, TSkol):
                    self.unsafe_vars.add(v)
        return s1, TRef(inner_t), e1

    def _ti_read(self, node, env, s, path):
        s1, ref_t, e1 = self.ti(node["ref"], env, s, path + ("ref",))
        inner_var = self.fresh()
        s2 = self.unify(
            s1, apply(s1, ref_t), TRef(inner_var),
            path + ("ref",), path, "read 要求引用类型",
        )
        if self.collect_effects:
            self.effect_traces.append(
                EffectTrace(path=path, tag="read", produced=("R",))
            )
        return s2, apply(s2, inner_var), self._join(e1, frozenset({"R"}))

    def _ti_write(self, node, env, s, path):
        s1, ref_t, e1 = self.ti(node["ref"], env, s, path + ("ref",))
        s2, val_t, e2 = self.ti(node["value"], env, s1, path + ("value",))
        s3 = self.unify(
            s2, apply(s2, ref_t), TRef(apply(s2, val_t)),
            path + ("ref",), path + ("value",),
            "write 要求左侧引用的单元类型等于右侧表达式类型",
        )
        if self.collect_effects:
            self.effect_traces.append(
                EffectTrace(path=path, tag="write", produced=("W",))
            )
        return s3, UNIT, self._join(e1, e2, frozenset({"W"}))

    # -- v2: 记录 -----------------------------------------------------

    def _ti_record(self, node, env, s, path):
        fields: List[Tuple[str, Type]] = []
        eff: Effect = frozenset()
        seen = set()
        for i, pair in enumerate(node["fields"]):
            fname, fexpr = pair
            if fname in seen:  # 校验器已拦，双保险
                raise RecordFieldError(path + ("fields", i), (), fname,
                                       f"记录字段重名 {fname!r}")
            seen.add(fname)
            s, ft, fe = self.ti(fexpr, env, s, path + ("fields", i, 1))
            fields.append((fname, ft))
            eff = self._join(eff, fe)
        fields.sort(key=lambda p: p[0])
        # 字面量行闭合：开放行只能经 extend/签名传入（避免 {f=1} 假装含 g）。
        return s, TRec(TRow(tuple(fields), REmpty())), eff

    def _ti_get(self, node, env, s, path):
        s1, rec_t, e1 = self.ti(node["record"], env, s, path + ("record",))
        field_var = self.fresh()
        tail = self.fresh_rvar()
        s2 = self.unify(
            s1, apply(s1, rec_t),
            TRec(TRow(((node["field"], field_var),), tail)),
            path + ("record",), path,
            f"get 要求记录含字段 {node['field']!r}",
        )
        return s2, apply(s2, field_var), e1

    def _ti_extend(self, node, env, s, path):
        s1, rec_t, e1 = self.ti(node["record"], env, s, path + ("record",))
        # 先把被扩展表达式钉成“某开放记录”：TRec({} ; ρ)。
        rho = self.fresh_rvar()
        s1 = self.unify(
            s1, apply(s1, rec_t), TRec(TRow((), rho)),
            path + ("record",), path,
            "extend 要求被扩展值是记录",
        )
        s2, val_t, e2 = self.ti(node["value"], env, s1, path + ("value",))
        s3, rest_t, e3 = self.ti(node["rest"], env, s2, path + ("rest",))
        # rest 必须是同一条行（共享同一个开放尾 ρ）；共有字段会逐点统一。
        s4 = self.unify(
            s3, apply(s3, rest_t), TRec(TRow((), rho)),
            path + ("rest",), path + ("record",),
            "extend 的行尾来源必须与被扩展记录是同一条记录行",
        )
        resolved = apply(s4, TRec(TRow((), rho)))
        if not isinstance(resolved, TRec):  # 不可达：上面已钉成记录
            raise TypeMismatchError(path + ("record",), path, resolved, TRec(TRow()))
        fields, terminal = self._norm_row(s4, resolved.row)
        fname = node["field"]
        if fname in fields:
            raise RecordFieldError(
                path + ("record",), path + ("field",), fname,
                f"扩展同名的已存在字段 {fname!r} 一律报错（不改变原记录）",
            )
        # 缺席约束：新字段不得经开放尾再次出现。把 fname 记在尾变量上，
        # 后续 get/统一把同名字段塞回该行时由 do_bind 拦截（开放行成环）。
        if isinstance(terminal, RVar):
            self.row_absent.setdefault(terminal, set()).add(fname)
            self.row_absences.append(RowAbsence(path=path, field=fname, row=terminal))
        new_fields = dict(fields)
        new_fields[fname] = apply(s4, val_t)
        out = TRec(
            TRow(tuple((n, new_fields[n]) for n in sorted(new_fields)), terminal)
        )
        return s4, out, self._join(e1, e2, e3)

    # -- v2: 异常 -----------------------------------------------------

    def _ti_raise(self, node, env, s, path):
        s1, val_t, e1 = self.ti(node["value"], env, s, path + ("value",))
        if self.raise_frames:
            target = self.raise_frames[-1]
            s1 = self.unify(
                s1, val_t, target,
                path + ("value",), path + ("[handler-param]",),
                "raise 的值类型必须与最近 try 的处理器参数一致",
            )
        result_var = self.fresh()
        if self.collect_effects:
            self.effect_traces.append(
                EffectTrace(path=path, tag="raise", produced=("X",))
            )
        return s1, result_var, self._join(e1, frozenset({"X"}))

    def _ti_try(self, node, env, s, path):
        param_var = self.fresh()
        self.raise_frames.append(param_var)
        try:
            s1, body_t, body_eff = self.ti(node["body"], env, s, path + ("body",))
        finally:
            self.raise_frames.pop()
        body_res = self.eff_resolve(body_eff) if self.collect_effects else frozenset()
        # X 在此被处理：从 body 效应中移除具体标签 X（未决效应变量保留）。
        handled = {"X"} if "X" in body_res else set()
        after = frozenset(x for x in body_res if x != "X")
        if self.collect_effects and handled:
            self.effect_traces.append(
                EffectTrace(path=path, tag="try", produced=(), handled=("X",))
            )
        handler_env = dict(env)
        handler_env[node["param"]] = TypeScheme(frozenset(), apply(s1, param_var))
        s2, handler_t, handler_eff = self.ti(
            node["handler"], handler_env, s1, path + ("handler",)
        )
        s3 = self.unify(
            s2, apply(s2, body_t), apply(s2, handler_t),
            path + ("body",), path + ("handler",),
            "try 的正常结果与处理器结果类型必须一致",
        )
        eff = self._join(after, handler_eff)
        return s3, apply(s3, body_t), eff

    def _ti_evar(self, node, env, s, path):
        """未决定应占位：类型全新变量、效应全新效应变量；如实保留为未决。"""
        ev = self.fresh_effvar()
        if self.collect_effects:
            self.effect_traces.append(
                EffectTrace(path=path, tag="evar", produced=("?",))
            )
        return s, self.fresh(), frozenset({ev})

    _DISPATCH = {
        "var": _ti_var,
        "xvar": _ti_xvar,
        "int": _ti_int,
        "bool": _ti_bool,
        "lambda": _ti_lambda,
        "apply": _ti_apply,
        "let": _ti_let,
        "if": _ti_if,
        "ref": _ti_ref,
        "read": _ti_read,
        "write": _ti_write,
        "record": _ti_record,
        "get": _ti_get,
        "extend": _ti_extend,
        "raise": _ti_raise,
        "try": _ti_try,
        "evar": _ti_evar,
    }


# ---------------------------------------------------------------- 杂项

def _rename(tp: Type, mapping: Dict[Type, Type]) -> Type:
    if isinstance(tp, (TVar, RVar)):
        return mapping.get(tp, tp)
    if isinstance(tp, TFun):
        return TFun(_rename(tp.arg, mapping), _rename(tp.ret, mapping), tp.eff)
    if isinstance(tp, TRef):
        return TRef(_rename(tp.inner, mapping))
    if isinstance(tp, TRow):
        return TRow(
            tuple((n, _rename(ft, mapping)) for n, ft in tp.fields),
            _rename(tp.rest, mapping),
        )
    if isinstance(tp, TRec):
        return TRec(_rename(tp.row, mapping))
    return tp


def _all_free(tp: Type) -> Tuple[Type, ...]:
    """普通类型变量与行变量，按先序去重。"""
    out: Dict[Type, None] = {}
    for v in free_tvars(tp):
        out[v] = None
    for v in free_rvars(tp):
        out[v] = None
    return tuple(out)


def _env_free_vars(env: Dict[str, TypeScheme], s: Subst):
    """环境的自由类型变量（含行变量）：各方案体自由变量减去被限定变量。"""
    free = set()
    for scheme in env.values():
        body = apply(s, scheme.t)
        quantified = scheme.quantified
        for v in _all_free(body):
            if v not in quantified:
                free.add(v)
    return free


# ---------------------------------------------------------------- 入口

def infer(
    ast: ast_mod.Node,
    max_nodes: int = 500,
    language: str = "v1",
) -> InferenceResult:
    """推断单表达式 AST 的主类型。

    :param language: ``"v1"``（默认，与初版完全一致）或 ``"v2"``
        （记录/开放行、R/W/X 效应、evar）。
    """
    if isinstance(max_nodes, bool) or not isinstance(max_nodes, int):
        raise TypeError(f"max_nodes 必须是 int，实际为 {type(max_nodes).__name__}")
    if max_nodes < 1:
        raise ValueError(f"max_nodes 必须为正整数，实际为 {max_nodes}")
    if language not in ("v1", "v2"):
        raise ValueError(f"language 必须是 'v1' 或 'v2'，实际为 {language!r}")

    old_limit = sys.getrecursionlimit()
    needed = 200 * max_nodes + 1000
    if needed > old_limit:
        sys.setrecursionlimit(needed)
    try:
        validator = ast_mod.AstValidator(max_nodes, language=language)
        validator.validate(ast)
        checker = Inferencer(language=language)
        subst, root_t, root_eff = checker.ti(ast, {}, {}, ())
        principal = apply(subst, root_t)
        names = canonical_names(principal)
        principal_str = render_type(principal, names)
        resolved_eff = checker.eff_resolve(root_eff)
        undecided = any(isinstance(x, EffVar) for x in resolved_eff)
        concrete_eff = frozenset(x for x in resolved_eff if isinstance(x, str))
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
        language=language,
        effects=concrete_eff,
        effects_undecided=undecided,
        effect_traces=tuple(checker.effect_traces),
        row_absences=tuple(checker.row_absences),
        eff_subst=tuple(checker.eff_subst.items()),
        source_ast=ast,
    )
