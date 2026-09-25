"""诊断模式：推断失败时给出按编号确定的极小冲突约束集。

普通入口 :func:`polyinfer.infer` 失败时只报告第一次统一失败的两个位置；
本模块在它旁边提供 :func:`diagnose`，回答“哪几条输入约束合起来才有问题”。

编号与重放
==========

* **稳定编号**：先用一次容错遍历（统一失败不抛出、记录后继续）枚举整棵
  AST 诱导的全部顶层统一等式，按遍历中的首次出现顺序编号。顶层等式以
  “两端 AST 路径对”为稳定键——键只取决于 AST 结构，与替换内容无关，
  因此同一程序每次诊断编号一致；函数/引用下钻出的子等式（路径带
  ``[arg]``/``[ret]``/``[inner]`` 后缀）共享所属顶层键。
* **冻结重放**：每一次试探都深拷贝冻结的原始 AST、配全新的
  :class:`Inferencer`（空替换、空计数器、空留痕）；绝不复用上一次
  失败遍历的并查集/替换。跳过某条等式 = 该顶层统一及其全部下钻
  完全不发生。

极小化算法
==========

逆编号序单轮删除：对编号从大到小的每条约束，重放“当前候选集 − 该条”，
删除后**仍冲突**就保留删除（不要求最小基数）。一轮结束后逐条冻结重放
验证：最终冲突核中任意删去一条都必须可满足——“包含极小（irredundant）”
是验证出来的，不是口头声明。

程序可能含两处互相独立的错误：找到一个核之后，把该核整体跳过再重放，
若仍冲突，则在剩余约束上再跑一轮逆序删除提取下一个核，直到剩余集合
可满足（或诊断预算耗尽）。因此两处独立错误会分别报出两个核。

判定的正确性依据：一阶语法合一定理——等式集可合一当且仅当任意处理
顺序都成功（mgu 存在且唯一）。所以 SAT/UNSAT 一律用**严格**重放判定
（泛化集合在成功前缀的部分 mgu 下精确计算，值限制一条不放宽）；
容错遍历只负责枚举编号与来源留痕，不参与任何可满足性结论。

逆序删除依赖单调性：跳过的等式越多，要求越松，严格重放只会从 UNSAT
变 SAT、不会反过来。删除一条等式只可能让某个变量少被绑定，于是 let
泛化集合只可能更大或相等（值限制的“扩展式不泛化”判定不依赖替换，
不随之改变），各使用点拿到的实例更新鲜、更独立，构造子冲突与 occurs
失败只可能消失。因此“试删后仍 UNSAT”就足以永久删除该条。

预算
====

诊断有独立预算 ``max_checks``（诊断遍历次数：容错枚举 1 次、每轮
“剩余是否仍冲突”判定、每次删除试探、每核的逐条成员验证都计数；
开头那次与普通入口完全相同的严格尝试为复刻主类型，不重复计费）。
预算耗尽时返回已经找到的核与尚未缩减完的候选，``minimal=False``，
绝不假称极小。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from . import ast as ast_mod
from .errors import (
    InfiniteTypeError,
    PolytypeError,
    UnificationError,
    format_path,
)
from .inference import (
    BindTrace,
    Inferencer,
    InferenceResult,
    InstantiationTrace,
    _recursion_room,
    _validate_max_nodes,
    infer,
)
from .types import Type, free_tvars

Path = Tuple[str, ...]
Key = Tuple[Path, Path]
_DRILL_SUFFIXES = ("[arg]", "[ret]", "[inner]")


# ---------------------------------------------------------------- 数据结构

@dataclass(frozen=True)
class ConstraintInfo:
    """一条带稳定编号的顶层统一约束。

    约束身份由编号与两端 AST 路径键确定；``sample_left``/``sample_right``
    只是容错枚举遍历当时的形态，供展示与来源追踪，不参与任何判定。
    ``sub_count`` 是该顶层等式下钻出的子等式条数
    （函数参数/返回值、引用单元）。
    """

    index: int
    left_path: Path
    right_path: Path
    note: str
    sub_count: int
    sample_left: Type
    sample_right: Type


@dataclass(frozen=True)
class ConflictEvidence:
    """某次统一失败的构造性证据；``kind`` 区分 occurs 与构造子冲突。

    这让“出现检查失败（无限类型）”与“普通类型构造子不合”在诊断输出中
    仍然是两类不同结论。
    """

    kind: str  # "infinite-type" | "type-mismatch"
    message: str
    left_path: Path
    right_path: Path
    left_type: Type
    right_type: Type
    constraint_index: Optional[int]

    @property
    def is_occurs(self) -> bool:
        return self.kind == "infinite-type"


@dataclass(frozen=True)
class ConflictCore:
    """一个冲突核：``members`` 合起来不可满足。

    * ``irredundant=True``：每条成员都经“删去它即可满足”的冻结重放
      验证（包含极小，不保证最小基数）；
    * ``irredundant=False``：预算在缩减/验证途中耗尽，这是未完成候选。
    """

    members: Tuple[ConstraintInfo, ...]
    evidence: ConflictEvidence
    irredundant: bool


@dataclass(frozen=True)
class BindingHop:
    """变量来源链中的一跳：?var 被第 ``constraint_index`` 条约束统一为 rhs。"""

    var_ident: int
    rhs: Type
    constraint_index: int
    note: str


@dataclass(frozen=True)
class DiagnosisResult:
    """诊断结果。

    成功（``ok=True``）时：``result`` 即普通 :func:`infer` 的返回值，
    ``principal_type`` 与逐约束核验结论与普通入口完全一致。

    失败时：``cores`` 给出一个或多个冲突核。恰好一个且
    ``minimal=True`` 是标准的包含极小冲突集；多个核表示多处互相独立的
    错误（每核整体跳过之后剩余集合才可满足/继续找下一核）。
    预算耗尽时最后一个候选 ``irredundant=False`` 且 ``minimal=False``，
    绝不假称极小。
    """

    ok: bool
    principal_type: Optional[str]
    result: Optional[InferenceResult]
    node_count: int
    constraints: Tuple[ConstraintInfo, ...]
    cores: Tuple[ConflictCore, ...]
    bind_trace: Tuple[BindTrace, ...]
    instantiation_trace: Tuple[InstantiationTrace, ...]
    checks_used: int
    check_budget: int
    budget_exhausted: bool

    # -- 便捷视图 ------------------------------------------------------

    @property
    def minimal(self) -> bool:
        return (
            not self.ok
            and not self.budget_exhausted
            and bool(self.cores)
            and all(c.irredundant and c.members for c in self.cores)
        )

    @property
    def conflict(self) -> Optional[Tuple[ConstraintInfo, ...]]:
        """第一个（单错误程序即唯一的）冲突核的成员，编号升序。"""
        return self.cores[0].members if self.cores else None

    @property
    def evidence(self) -> Optional[ConflictEvidence]:
        return self.cores[0].evidence if self.cores else None

    def conflict_indices(self) -> Tuple[int, ...]:
        return tuple(c.index for c in self.cores[0].members) if self.cores else ()

    def provenance(self, var_ident: int) -> Tuple[BindingHop, ...]:
        """追踪内部类型变量 ?var_ident 的替换来源链。

        依次列出“?n 被哪条编号约束绑定成什么”，右端若含新变量则沿
        第一个新变量继续；这是“替换/统一来源”的逐条留痕（来自容错
        枚举遍历，重放间内部变量编号确定，因为遍历顺序与 fresh 顺序
        完全相同）。
        """
        index_of: Dict[Key, int] = {
            (c.left_path, c.right_path): c.index for c in self.constraints
        }
        chain: List[BindingHop] = []
        seen = set()
        current = var_ident
        for hop in self.bind_trace:
            if hop.var_ident != current or current in seen:
                continue
            seen.add(current)
            chain.append(
                BindingHop(
                    var_ident=hop.var_ident,
                    rhs=hop.rhs,
                    constraint_index=index_of.get(hop.key, -1),
                    note=hop.note,
                )
            )
            nxt = [v.ident for v in free_tvars(hop.rhs)]
            if not nxt:
                break
            current = nxt[0]
        return tuple(chain)


# ---------------------------------------------------------------- 预算与重放

class DiagnosticBudgetExhausted(PolytypeError):
    """内部控制信号：诊断遍历次数超出独立预算。

    在 :func:`diagnose` 内部被捕获并转换为 ``minimal=False`` 的结果，
    不会泄漏给调用方。
    """


class _Budget:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0

    def tick(self) -> None:
        self.used += 1
        if self.used > self.limit:
            raise DiagnosticBudgetExhausted


def _strip_drill(path: Path) -> Path:
    while path and path[-1] in _DRILL_SUFFIXES:
        path = path[:-1]
    return tuple(path)


def _key_of_error(exc: UnificationError) -> Key:
    return _strip_drill(exc.left_path), _strip_drill(exc.right_path)


@dataclass(frozen=True)
class _Replay:
    ok: bool
    error: Optional[UnificationError]


def _replay(
    frozen_ast: ast_mod.Node,
    max_nodes: int,
    skip: "frozenset[Key]",
    budget: _Budget,
    *,
    tolerant: bool,
    trace: bool,
) -> Tuple[_Replay, Inferencer, ast_mod.AstValidator]:
    """一次冻结重放：深拷贝原始 AST + 全新 Inferencer（空替换/空计数器）。

    严格重放（``tolerant=False``）的成功/失败就是该等式集的
    SAT/UNSAT；容错重放仅用于枚举编号与来源留痕。
    """
    budget.tick()
    checker = Inferencer(tolerant=tolerant, skip_keys=skip, trace=trace)
    validator = ast_mod.AstValidator(max_nodes)
    with _recursion_room(max_nodes):
        # 深拷贝本身沿 AST 结构递归，必须与校验/ti 同在提限区间内。
        ast_copy = copy.deepcopy(frozen_ast)
        validator.validate(ast_copy)
        checker.ti(ast_copy, {}, {}, ())
    if tolerant and checker.failures:
        exc, _key = checker.failures[0]
        return _Replay(False, exc), checker, validator
    return _Replay(True, None), checker, validator


def _strict_replay(
    frozen_ast: ast_mod.Node,
    max_nodes: int,
    skip: "frozenset[Key]",
    budget: _Budget,
) -> _Replay:
    """严格重放并把统一失败归一为 _Replay（其他错误不应出现：向上抛）。"""
    try:
        outcome, _, _ = _replay(
            frozen_ast, max_nodes, skip, budget,
            tolerant=False, trace=False,
        )
    except UnificationError as exc:
        return _Replay(False, exc)
    return outcome


def _evidence(exc: UnificationError, index_of: Dict[Key, int]) -> ConflictEvidence:
    kind = "infinite-type" if isinstance(exc, InfiniteTypeError) else "type-mismatch"
    return ConflictEvidence(
        kind=kind,
        message=str(exc),
        left_path=exc.left_path,
        right_path=exc.right_path,
        left_type=exc.left_type,
        right_type=exc.right_type,
        constraint_index=index_of.get(_key_of_error(exc)),
    )


# ---------------------------------------------------------------- 入口

def diagnose(
    ast: ast_mod.Node,
    max_nodes: int = 500,
    max_checks: int = 512,
) -> DiagnosisResult:
    """诊断入口：与 :func:`polyinfer.infer` 并列，不改变其任何行为。

    :param ast: AST（构造器或同构 dict），本函数不修改调用方对象；
    :param max_nodes: 与普通入口相同的节点上限；
    :param max_checks: 诊断独立预算（遍历次数），默认 512；
    :returns: :class:`DiagnosisResult`。AST 非法、节点超限、未绑定变量
        等“类型冲突之前”的错误与普通入口一样直接抛出；统一冲突永不
        外抛，而是进入冲突核结果（预算耗尽也不假称极小）。
    """
    _validate_max_nodes(max_nodes)
    if isinstance(max_checks, bool) or not isinstance(max_checks, int):
        raise TypeError(f"max_checks 必须是 int，实际为 {type(max_checks).__name__}")
    if max_checks < 1:
        raise ValueError(f"max_checks 必须为正整数，实际为 {max_checks}")

    budget = _Budget(max_checks)

    # 冻结原始输入：后续所有重放都从这份深拷贝出发。深拷贝沿 AST
    # 结构递归，与校验/ti 一样需要为深程序临时提高递归上限。
    with _recursion_room(max_nodes):
        frozen_ast = copy.deepcopy(ast)

    # 0) 与普通入口完全相同的严格尝试（复刻其行为；成功则主类型必须
    #    逐字一致）。AST 非法 / 节点超限 / 未绑定变量在此原样抛出。
    success_result = None
    strict_error: Optional[UnificationError] = None
    try:
        success_result = infer(frozen_ast, max_nodes)
    except UnificationError as exc:
        strict_error = exc

    # 1) 容错枚举遍历：全部顶层约束稳定编号 + 来源留痕。
    cores: List[ConflictCore] = []
    exhausted = False

    # 枚举产物的缺省值（预算在枚举自身耗尽时仍可安全返回）。
    node_count = 0
    keys: List[Key] = []
    index_of: Dict[Key, int] = {}
    all_infos: Tuple[ConstraintInfo, ...] = ()
    info_by_key: Dict[Key, ConstraintInfo] = {}
    bind_trace: Tuple[BindTrace, ...] = ()
    inst_trace: Tuple[InstantiationTrace, ...] = ()
    # 缩减中途的状态（预算耗尽时用于交回未完成候选）。
    pending_active: Optional[set] = None
    pending_evidence: Optional[ConflictEvidence] = None

    try:
        _enum_outcome, enum_checker, validator = _replay(
            frozen_ast, max_nodes, frozenset(), budget,
            tolerant=True, trace=True,
        )
        node_count = validator.count
        keys, index_of, all_infos = _number_constraints(enum_checker)
        info_by_key = {(c.left_path, c.right_path): c for c in all_infos}
        bind_trace = tuple(enum_checker.bind_trace)
        inst_trace = tuple(enum_checker.instantiation_trace)

        if success_result is not None:
            return DiagnosisResult(
                ok=True,
                principal_type=success_result.principal_type,
                result=success_result,
                node_count=node_count,
                constraints=all_infos,
                cores=(),
                bind_trace=bind_trace,
                instantiation_trace=inst_trace,
                checks_used=budget.used,
                check_budget=max_checks,
                budget_exhausted=False,
            )

        # 2) 逐核提取。每轮在“尚未归属任何核”的约束 remaining 上做
        #    逆编号序单轮删除；核取出后整体跳过，再看剩余是否仍冲突。
        remaining = set(keys)
        while True:
            skip_rest = frozenset(set(keys) - remaining)
            probe = _strict_replay(frozen_ast, max_nodes, skip_rest, budget)
            if probe.ok:
                break  # 剩余约束可满足：没有更多独立错误。
            # 进入新一轮缩减：记录未完成候选（预算可能在此轮中途耗尽）。
            pending_active = set(remaining)
            pending_evidence = _evidence(probe.error, index_of)

            active = set(remaining)
            # 逆编号序：编号大的先试删（编号确定，结果因此确定）。
            for key in sorted(active, key=lambda k: index_of[k], reverse=True):
                trial_skip = frozenset(set(keys) - (active - {key}))
                outcome = _strict_replay(frozen_ast, max_nodes, trial_skip, budget)
                if not outcome.ok:
                    active.discard(key)  # 删除后仍冲突：保留删除。
                    pending_active = set(active)
                    pending_evidence = _evidence(outcome.error, index_of)

            # active 即本轮冲突核候选；取它整体仍冲突的证据。
            core_skip = frozenset(set(keys) - active)
            core_outcome = _strict_replay(frozen_ast, max_nodes, core_skip, budget)
            if core_outcome.ok:
                # 理论不可达（逆序删除的每一步都保持 UNSAT）；防御性兜底。
                break

            # 3) 包含极小验证：核中逐条移除都必须可满足（冻结重放）。
            members_sorted = sorted(active, key=lambda k: index_of[k])
            irredundant = True
            for key in members_sorted:  # 编号升序验证，顺序同样确定
                outcome = _strict_replay(
                    frozen_ast, max_nodes,
                    frozenset(set(keys) - (active - {key})), budget,
                )
                if not outcome.ok:
                    irredundant = False
                    break

            cores.append(
                ConflictCore(
                    members=tuple(info_by_key[k] for k in members_sorted),
                    evidence=_evidence(core_outcome.error, index_of),
                    irredundant=irredundant,
                )
            )
            remaining -= active
            # 本轮核已完整归属；清掉未完成候选标记。
            pending_active = None
            pending_evidence = None

        pending_infos = None
    except DiagnosticBudgetExhausted:
        exhausted = True
        pending_infos = None
        if pending_active and index_of:
            pending_infos = tuple(
                info_by_key[k]
                for k in sorted(pending_active, key=lambda kk: index_of[kk])
            )
            if pending_evidence is None and strict_error is not None:
                pending_evidence = _evidence(strict_error, index_of)
            if pending_evidence is not None:
                cores.append(
                    ConflictCore(
                        members=pending_infos,
                        evidence=pending_evidence,
                        irredundant=False,
                    )
                )
                pending_infos = None
        elif not cores and strict_error is not None:
            # 预算在任何缩减开始前耗尽：附空成员核仅承载原始失败证据。
            cores.append(
                ConflictCore(
                    members=(),
                    evidence=_evidence(strict_error, index_of),
                    irredundant=False,
                )
            )

    return DiagnosisResult(
        ok=False,
        principal_type=None,
        result=None,
        node_count=node_count,
        constraints=all_infos,
        cores=tuple(cores),
        bind_trace=bind_trace,
        instantiation_trace=inst_trace,
        checks_used=min(budget.used, max_checks),
        check_budget=max_checks,
        budget_exhausted=exhausted,
    )


# ---------------------------------------------------------------- 编号表

def _number_constraints(
    enum_checker: Inferencer,
) -> Tuple[List[Key], Dict[Key, int], Tuple[ConstraintInfo, ...]]:
    """从容错枚举遍历的留痕构造稳定编号表。"""
    keys: List[Key] = []
    seen: set = set()
    samples: Dict[Key, Tuple[Type, Type, str]] = {}
    sub_counts: Dict[Key, int] = {}
    for c in enum_checker.constraints:
        key = (_strip_drill(c.left_path), _strip_drill(c.right_path))
        if key not in seen:
            seen.add(key)
            keys.append(key)
            samples[key] = (c.left, c.right, c.note)
            sub_counts[key] = 1
        else:
            sub_counts[key] += 1

    index_of = {key: i for i, key in enumerate(keys)}
    infos = tuple(
        ConstraintInfo(
            index=index_of[key],
            left_path=key[0],
            right_path=key[1],
            note=samples[key][2],
            sub_count=sub_counts[key],
            sample_left=samples[key][0],
            sample_right=samples[key][1],
        )
        for key in keys
    )
    return keys, index_of, infos


# ---------------------------------------------------------------- 渲染

def format_diagnosis(diag: DiagnosisResult) -> str:
    """把诊断结果渲染成中文多行说明（供 REPL / demo 使用）。"""
    lines: List[str] = []
    if diag.ok:
        lines.append(f"推断成功，主类型：{diag.principal_type}")
        lines.append(f"共 {len(diag.constraints)} 条统一约束，全部成立；"
                     f"诊断遍历 {diag.checks_used} 次（独立预算 {diag.check_budget}）。")
        return "\n".join(lines)

    lines.append("推断失败。")
    lines.append(f"统一约束共 {len(diag.constraints)} 条，稳定编号如下：")
    for c in diag.constraints:
        lines.append(
            f"  #{c.index:<3} {c.note}\n"
            f"        左端 {format_path(c.left_path)}：{c.sample_left}\n"
            f"        右端 {format_path(c.right_path)}：{c.sample_right}"
            + (f"\n        （含 {c.sub_count} 条下钻子等式）" if c.sub_count > 1 else "")
        )

    if not diag.cores:
        lines.append("")
        lines.append("诊断预算不足，未能给出冲突候选。")
    else:
        many = len(diag.cores) > 1
        for i, core in enumerate(diag.cores):
            idxs = ", ".join(f"#{c.index}" for c in core.members)
            head = f"冲突核 {i + 1}/{len(diag.cores)}" if many else "冲突核"
            tag = "包含极小冲突集（逐条移除验证通过）" if core.irredundant \
                else "未缩减完的冲突候选（不宣称极小）"
            lines.append("")
            lines.append(f"{head}：{tag}，{len(core.members)} 条：{idxs}")
            for c in core.members:
                lines.append(f"  #{c.index} {c.note}")
            lines.append(_format_evidence(core.evidence, prefix="  冲突证据："))

    if diag.instantiation_trace:
        lines.append("")
        lines.append("实例化来源（变量使用点 -> 绑定来源）：")
        for tr in diag.instantiation_trace:
            src_kind = "let 绑定" if tr.source.kind == "let" else "lambda 形参"
            if tr.polymorphic:
                poly = "多态新实例（重命名 %s）" % ", ".join(
                    f"?{a}->?{b}" for a, b in tr.renaming
                )
            elif tr.source.bound_tag not in ("int", "bool", "lambda", "var"):
                poly = "单态共享（值限制：扩展式不泛化）"
            else:
                poly = "单态共享"
            lines.append(
                f"  {tr.name!r} @ {format_path(tr.use_path)} <- "
                f"{src_kind} @ {format_path(tr.source.path)}；{poly}"
            )

    if diag.budget_exhausted:
        lines.append("")
        lines.append(f"注意：诊断预算 {diag.check_budget} 次遍历已耗尽，"
                     "以上含未缩减完的冲突证据，不宣称极小。")
    lines.append("")
    lines.append(f"诊断遍历 {diag.checks_used} 次（独立预算 {diag.check_budget}）。")
    return "\n".join(lines)


def _format_evidence(ev: ConflictEvidence, prefix: str = "") -> str:
    kind = "occurs 检查失败（无限类型）" if ev.is_occurs else "构造子不一致（普通类型不合）"
    where = f"（叶子落在约束 #{ev.constraint_index}）" if ev.constraint_index is not None else ""
    pad = " " * len(prefix)
    return (
        f"{prefix}{kind}{where}\n"
        f"{pad}左端 {ev.left_type} @ {format_path(ev.left_path)}\n"
        f"{pad}右端 {ev.right_type} @ {format_path(ev.right_path)}"
    )
