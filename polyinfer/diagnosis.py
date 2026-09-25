"""诊断模式：在不可类型化时给出**按编号确定的包含极小冲突集**。

与普通入口 :func:`polyinfer.infer` 并列的诊断入口 :func:`diagnose`：

1. **约束稳定编号**：先对原始（冻结的）AST 做一次纯结构遍历，按确定的
   先序顺序给每条"顶层统一义务"编号（apply / if 条件 / if 分支 /
   read / write 各诱发一条；函数与引用的结构下钻属于同一条义务）。
   编号只取决于 AST，与推断状态、替换、重放次数都无关。
2. **来源追踪**：每条统一等式记录所属义务编号；多态方案的每次实例化
   记录使用点 → 方案引入位置（let/lambda）的变量对应；每次替换绑定
   记录是哪条义务把哪个变量绑成了什么。
3. **重放极小化**：Algorithm W 首个失败即停止。诊断从**全部义务**出发，
   按编号**逆序**逐条尝试删除——用冻结的原始约束重放：删除后仍冲突，
   就保留删除；删除后可满足，则该约束必要、恢复。一轮结束。
   每一轮都是全新推断器、全新计数器、空替换，绝不携带上一轮的
   并查集或替换。occurs（``"occurs"``）与普通构造子不合
   （``"mismatch"``）分别标记。
4. **逐条必要性核验**：对最终集合中每一条再单独重放"删去它"的场景，
   必须可满足——确认包含极小（不追求最小基数）。
5. **独立预算**：诊断有自己的重放轮数预算；耗尽时只返回尚未缩减完的
   冲突证据，``minimal`` 为 ``False``，绝不假称极小。

正确性不依赖任何"放宽规则"的假设：初始完整重放构造性地证明全集不可
满足；此后每一次"删除后仍冲突"的重放都**直接证明**删除后的剩余集合
仍不可满足（因此该丢弃合法），最终候选集的不可满足性由最后一次这样
的重放兜底；逐条核验再证明删去候选集中任意一条都可满足。跳过义务
等号于不施加该等式（少做若干变量绑定），不会改变值限制、作用域与
occurs 检查，也不会引入新的绑定，因此重放所用的推断核心与普通入口
完全一致。

值限制、作用域、occurs 检查、字面量契约在诊断路径上**一字未松**；
诊断不引入任何新语法或新类型系统成分。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from . import ast as ast_mod
from .errors import (
    UnificationError,
    format_path,
)
from .inference import (
    Constraint,
    Generalization,
    InstantiationEvent,
    run_infer_partial,
)

Path = Tuple[str, ...]
ObligationId = ast_mod.ObligationId

# 每种义务的人类可读说明（与核心诱发等式时的 note 一致）。
_OBLIGATION_NOTES: Dict[str, str] = {
    "apply": "apply 要求被应用值是函数",
    "if_cond": "if 的条件必须是 Bool",
    "if_branches": "if 两个分支类型必须一致",
    "read": "read 要求引用类型",
    "write": "write 要求左侧引用的单元类型等于右侧表达式类型",
}


# ------------------------------------------------------------ 约束编号

@dataclass(frozen=True)
class ObligationInfo:
    """一条带稳定编号的输入约束（顶层统一义务）。"""

    number: int
    kind: str
    path: Path
    note: str

    @property
    def id(self) -> ObligationId:
        return (self.kind, self.path)

    def format_location(self) -> str:
        return f"#{self.number} [{self.kind}] {format_path(self.path)}"


def enumerate_obligations(node: ast_mod.Node) -> Tuple[ObligationInfo, ...]:
    """对冻结 AST 做纯结构遍历，按确定的先序顺序给义务编号（从 1 起）。

    编号顺序：先序访问节点；同一 ``if`` 节点上先条件后分支
    （:data:`polyinfer.ast.OBLIGATION_KINDS` 的固定顺序）。
    结果只依赖 AST 本身，不跑任何推断。
    """
    catalog: List[ObligationInfo] = []

    def add(kind: str, p: Path) -> None:
        catalog.append(ObligationInfo(len(catalog) + 1, kind, p, _OBLIGATION_NOTES[kind]))

    def walk(nd: ast_mod.Node, path: Path) -> None:
        tag = nd["tag"]
        if tag == "apply":
            add("apply", path)
        elif tag == "if":
            add("if_cond", path)
            add("if_branches", path)
        elif tag == "read":
            add("read", path)
        elif tag == "write":
            add("write", path)

        if tag == "lambda":
            walk(nd["body"], path + ("body",))
        elif tag == "apply":
            walk(nd["fn"], path + ("fn",))
            walk(nd["arg"], path + ("arg",))
        elif tag == "let":
            walk(nd["bound"], path + ("bound",))
            walk(nd["body"], path + ("body",))
        elif tag == "if":
            walk(nd["cond"], path + ("cond",))
            walk(nd["then"], path + ("then",))
            walk(nd["else_"], path + ("else_",))
        elif tag == "ref":
            walk(nd["inner"], path + ("inner",))
        elif tag == "read":
            walk(nd["ref"], path + ("ref",))
        elif tag == "write":
            walk(nd["ref"], path + ("ref",))
            walk(nd["value"], path + ("value",))

    walk(node, ())
    return tuple(catalog)


# ------------------------------------------------------------ 报告数据

@dataclass(frozen=True)
class ConflictEvidence:
    """冲突集中一条编号约束的证据。

    ``witness`` 为真表示：在只含极小冲突集的最终重放中，正是这条义务
    的等式率先失败（叶子两端类型即该失败的两端）；其余成员是共同导致
    不可满足的参与者，其两端类型取自该重放中该义务施加时的等式形态。
    """

    number: int
    kind: str            # 义务种类 apply/if_cond/...
    failure: str         # "mismatch" | "occurs" | ""（参与者未率先失败）
    obligation: ObligationId
    left_path: Path
    right_path: Path
    left_type: str
    right_type: str
    note: str
    witness: bool


@dataclass(frozen=True)
class BindInfo:
    """一次替换绑定的来源：哪个编号义务把变量 ``?var_ident`` 绑成 ``rhs``。"""

    var_ident: int
    rhs: str
    obligation_number: Optional[int]
    note: str


@dataclass(frozen=True)
class Provenance:
    """只含极小冲突集的最终重放所留下的完整来源轨迹。

    所有 TVar 编号都属于**该次重放**（每次重放独立计数，不跨轮）：

    * ``instantiations``：多态实例化，使用点 → 方案引入位置的变量对应；
    * ``binds``：每次替换绑定由哪条编号义务诱发（``obligation_number``
      为 ``None`` 表示无输入义务的内部绑定）；
    * ``unifications``：每条实际施加的等式（``Constraint.obligation``
      为其义务身份，可用 :meth:`DiagnosisReport.obligation` 反查编号）；
    * ``witness_number`` / ``witness_kind``：率先失败的义务编号与异常种类。
    """

    instantiations: Tuple[InstantiationEvent, ...]
    binds: Tuple[BindInfo, ...]
    unifications: Tuple[Constraint, ...]
    generalizations: Tuple[Generalization, ...]
    witness_number: int
    witness_kind: str


@dataclass(frozen=True)
class CoreReport:
    """一个包含极小冲突核及其证据与来源轨迹。

    * ``numbers``：核内约束编号（按编号升序）；删去其中任意一条后，
      只含剩余核成员的重放都可满足（``verified`` 为真时已逐条重放确认）；
    * ``conflicts``：核内每条义务的证据，其中恰有一条 ``witness`` 为真
      （只含本核的重放中率先失败的那条）；
    * ``provenance``：该次重放的实例化/绑定/统一来源轨迹；
    * ``verified``：本核是否已完成逆序缩减与逐条必要性核验。
    """

    numbers: Tuple[int, ...]
    conflicts: Tuple[ConflictEvidence, ...]
    provenance: Optional[Provenance]
    verified: bool


@dataclass(frozen=True)
class DiagnosisReport:
    """:func:`diagnose` 的结果。

    * ``ok``：原程序可类型化（``principal_type`` 与 :func:`polyinfer.infer`
      的结果逐字相同）；
    * ``cores``：互不相交的包含极小冲突核，按发现顺序排列；第一个核
      ``cores[0]`` 完全由编号确定（同一 AST 多次诊断结果一致）；
    * ``conflict_numbers``：**首个**核的编号（便捷字段），即编辑器首选
      高亮的"哪几条输入约束合起来才有问题"；
    * ``minimal``：所有已报告核都完成缩减与逐条验证，且核外再无独立
      冲突；不追求最小基数；预算耗尽恒假；
    * ``budget_exhausted``：独立重放预算是否耗尽——为真时核可能未缩减
      完，不假称极小；
    * ``replays_used``：实际消耗的重放次数（含首次完整推断）。
    """

    ok: bool
    principal_type: Optional[str]
    obligations: Tuple[ObligationInfo, ...]
    cores: Tuple[CoreReport, ...]
    minimal: bool
    budget_exhausted: bool
    replays_used: int
    budget: int

    @property
    def conflict_numbers(self) -> Tuple[int, ...]:
        return self.cores[0].numbers if self.cores else ()

    @property
    def conflicts(self) -> Tuple[ConflictEvidence, ...]:
        out: List[ConflictEvidence] = []
        for core in self.cores:
            out.extend(core.conflicts)
        return tuple(out)

    def obligation(self, number: int) -> ObligationInfo:
        return self.obligations[number - 1]

    def describe(self) -> str:
        """生成给编辑器/人看的多行说明。"""
        lines: List[str] = []
        if self.ok:
            lines.append(f"程序可类型化，主类型：{self.principal_type}")
            return "\n".join(lines)
        if len(self.cores) == 1:
            lines.append(
                f"包含极小冲突集（按编号确定）：{self._fmt_nums(self.cores[0].numbers)}"
                if self.minimal
                else f"冲突约束（未完成极小化，不宣称极小）：{self._fmt_nums(self.cores[0].numbers)}"
            )
            lines.append("这些输入约束合在一起才不可满足；删去其中任意一条后剩余约束均可满足。")
        else:
            head = "发现多处相互独立的冲突，各给出一个包含极小冲突集"
            if not self.minimal:
                head += "（预算受限，未全部完成极小化，不宣称极小）"
            lines.append(head + "：")
            for idx, core in enumerate(self.cores, 1):
                lines.append(f"  冲突集 {idx}：{self._fmt_nums(core.numbers)}")
        if self.budget_exhausted:
            lines.append(f"注意：重放预算（{self.budget} 轮）已耗尽，以上可能未缩减完。")
        for idx, core in enumerate(self.cores, 1):
            if len(self.cores) > 1:
                lines.append(f"-- 冲突集 {idx} 的证据 --")
            for ev in core.conflicts:
                info = self.obligation(ev.number)
                role = "率先失败" if ev.witness else "共同参与者"
                if ev.failure == "occurs":
                    role += "，occurs 无限类型"
                elif ev.failure == "mismatch":
                    role += "，构造子不合"
                lines.append(
                    f"  {info.format_location()}（{role}）\n"
                    f"      约束：{ev.note}\n"
                    f"      左端 {ev.left_type} @ {format_path(ev.left_path)}\n"
                    f"      右端 {ev.right_type} @ {format_path(ev.right_path)}"
                )
        return "\n".join(lines)

    @staticmethod
    def _fmt_nums(numbers: Tuple[int, ...]) -> str:
        return ", ".join(f"#{n}" for n in numbers) or "（无）"


# ------------------------------------------------------------ 诊断主体

def _evidence_from_error(number: int, info: ObligationInfo, err: UnificationError) -> ConflictEvidence:
    return ConflictEvidence(
        number=number,
        kind=info.kind,
        failure=err.kind,  # "mismatch" | "occurs"
        obligation=info.id,
        left_path=err.left_path,
        right_path=err.right_path,
        left_type=str(err.left_type),
        right_type=str(err.right_type),
        note=err.note or info.note,
        witness=True,
    )


def _evidence_from_constraint(number: int, info: ObligationInfo, c: Constraint) -> ConflictEvidence:
    return ConflictEvidence(
        number=number,
        kind=info.kind,
        failure="",
        obligation=info.id,
        left_path=c.left_path,
        right_path=c.right_path,
        left_type=str(c.left),
        right_type=str(c.right),
        note=c.note or info.note,
        witness=False,
    )


def _reduce_core(active, by_id, catalog, replay, budget, used, known_error):
    """在 ``active``（frozenset 义务 id）内部做逆序贪心缩减 + 逐条核验。

    前提：``active`` 已由 ``known_error``（一次重放的统一失败）证明不可
    满足。返回 ``(kept, verified, exhausted, err_f, checker_f)``：
    ``kept`` 为缩减后的 id 集合；``verified`` 为是否完成逐条必要性核验；
    ``err_f`` 为只含缩减结果的见证重放失败（预算不足无法重放时回退为
    ``known_error``）。每次判定都是一次使用冻结原始约束的全新重放。
    """
    kept = set(active)
    exhausted = False

    for info in reversed(catalog):
        if info.id not in kept:
            continue
        if used[0] >= budget:
            exhausted = True
            break
        _r, _c, err = replay(frozenset(kept - {info.id}))
        if err is not None:
            kept.discard(info.id)  # 删除后仍冲突 ⇒ 不必要，保留删除

    verified = not exhausted
    if verified:
        for info in catalog:
            if info.id not in kept:
                continue
            if used[0] >= budget:
                exhausted = True
                verified = False
                break
            _r, _c, err = replay(frozenset(kept - {info.id}))
            if err is not None:
                verified = False  # 防御性：单调性下不应发生

    # 只含本核的重放：拿见证义务与来源轨迹；同时再次确认本核自身冲突。
    checker_f = None
    err_f = None
    if not exhausted and used[0] < budget:
        _rf, checker_f, err_f = replay(frozenset(kept))
        if err_f is None:
            verified = False  # 核自身竟可满足：不宣称极小
    else:
        # 预算不足以做只含本核的重放：回退用调用方已知的失败做见证，
        # 但它可能来自更大的 active，证据只标见证义务，核不宣称已验证。
        err_f = known_error
    return kept, verified, exhausted, err_f, checker_f


def _build_core(kept, err_f, checker_f, by_id, catalog, verified):
    """由只含本核的重放结果组装 :class:`CoreReport`。"""
    final_ids = [info.id for info in catalog if info.id in kept]
    evidence: Dict[ObligationId, ConflictEvidence] = {}

    if err_f is not None and err_f.obligation in by_id:
        witness_ob = err_f.obligation
        top_by_ob: Dict[ObligationId, Constraint] = {}
        for c in (checker_f.constraints if checker_f else ()):
            if c.obligation and c.obligation not in top_by_ob:
                top_by_ob[c.obligation] = c
        for ob in final_ids:
            info = by_id[ob]
            if ob == witness_ob:
                evidence[ob] = _evidence_from_error(info.number, info, err_f)
            else:
                c = top_by_ob.get(ob)
                evidence[ob] = (
                    _evidence_from_constraint(info.number, info, c)
                    if c is not None
                    else ConflictEvidence(
                        number=info.number, kind=info.kind, failure="",
                        obligation=info.id, left_path=info.path, right_path=info.path,
                        left_type="（预算内未取得叶子证据）", right_type="（预算内未取得叶子证据）",
                        note=info.note, witness=False,
                    )
                )
        # 只在真正跑过只含本核的见证重放时才给出轨迹；预算耗尽回退
        # （checker_f 为 None）不给看似完整、实则来自更大集合的轨迹。
        provenance = Provenance(
            instantiations=tuple(checker_f.events),
            binds=tuple(
                BindInfo(
                    var_ident=b.var_ident,
                    rhs=b.rhs,
                    obligation_number=(by_id[b.obligation].number if b.obligation else None),
                    note=b.note,
                )
                for b in checker_f.bind_events
            ),
            unifications=tuple(checker_f.constraints),
            generalizations=tuple(checker_f.generalizations),
            witness_number=by_id[witness_ob].number,
            witness_kind=err_f.kind,
        ) if checker_f is not None else None
    else:
        # 预算耗尽未能做见证重放：只给定位证据，不伪造叶子类型。
        provenance = None
        for ob in final_ids:
            info = by_id[ob]
            evidence[ob] = ConflictEvidence(
                number=info.number, kind=info.kind, failure="",
                obligation=info.id, left_path=info.path, right_path=info.path,
                left_type="（预算内未取得叶子证据）", right_type="（预算内未取得叶子证据）",
                note=info.note, witness=False,
            )

    ordered = [evidence[ob] for ob in final_ids]
    return CoreReport(
        numbers=tuple(e.number for e in ordered),
        conflicts=tuple(ordered),
        provenance=provenance,
        verified=verified,
    )


def diagnose(node: ast_mod.Node, max_nodes: int = 500, budget: int = 4096) -> DiagnosisReport:
    """诊断入口：返回 :class:`DiagnosisReport`，绝不抛统一失败异常。

    :param node: AST（与 :func:`polyinfer.infer` 同一契约、同一套校验）；
    :param max_nodes: 节点上限（资源拒绝仍由原异常类表达）；
    :param budget: 诊断**独立**的重放预算（推断轮数上限，含首轮）；
      耗尽后返回未缩减完的核且 ``minimal=False``。

    非法 AST、未绑定变量、节点超限照常抛原有异常——它们不是约束冲突。

    流程：约束编号在任何推断之前由冻结 AST 确定；先完整重放，成功则
    返回与普通入口一致的主类型；否则迭代找出互不相交的冲突核——
    每轮把已发现核的成员全部跳过，用全新推断器、空替换重放冻结的
    原始约束，直到某轮成功（核外再无冲突）或预算耗尽。每个核内部按
    编号逆序贪心缩减并逐条重放验证包含极小性。
    """
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise TypeError(f"budget 必须是 int，实际为 {type(budget).__name__}")
    if budget < 1:
        raise ValueError(f"budget 必须为正整数，实际为 {budget}")

    # 先做与普通入口完全相同的输入校验：非法 AST / 未绑定变量 / 超节点
    # 上限在这里抛原异常，它们不是约束冲突，编号遍历只面对合法 AST。
    validator = ast_mod.AstValidator(max_nodes)
    validator.validate(node)

    # 约束编号在任何推断之前由冻结 AST 确定。
    catalog = enumerate_obligations(node)
    by_id: Dict[ObligationId, ObligationInfo] = {info.id: info for info in catalog}
    all_ids = frozenset(by_id)

    used_box = [0]

    def replay(active: frozenset):
        """用冻结原始约束重放：active 之外的义务整体跳过。

        每轮全新 Inferencer + 全新计数器 + 空替换，绝不携带上一轮的
        并查集或替换；失败轮也返回截至失败点的来源轨迹。"""
        used_box[0] += 1
        return run_infer_partial(
            node, max_nodes=max_nodes, skipped_obligations=all_ids - active
        )

    # 0) 首次完整推断。
    result0, _c0, err0 = replay(all_ids)
    if err0 is None:
        return DiagnosisReport(
            ok=True,
            principal_type=result0.principal_type,
            obligations=catalog,
            cores=(),
            minimal=True,
            budget_exhausted=False,
            replays_used=used_box[0],
            budget=budget,
        )

    # 1) 迭代找互不相交的冲突核。
    #    首轮 active=全集 的不可满足性已由 err0 证明，直接缩减；
    #    此后每轮先重放确认"去掉已发现核后是否仍冲突"。
    removed: set = set()
    cores: List[CoreReport] = []
    exhausted = False
    all_verified = True
    err = err0
    active = all_ids

    while err is not None:
        if not err.obligation:
            exhausted = True  # 理论不可达，保守处理
            break

        if used_box[0] >= budget:
            # 连一次试删重放都付不起：只把当前见证义务作为未缩减证据，
            # 不把整个 active 误报为冲突集，也不宣称极小或成核完整性。
            info = by_id[err.obligation]
            cores.append(CoreReport(
                (info.number,),
                (_evidence_from_error(info.number, info, err),),
                None, False,
            ))
            all_verified = False
            exhausted = True
            break

        kept, verified, ex, err_f, checker_f = _reduce_core(
            active, by_id, catalog, replay, budget, used_box, err
        )
        exhausted = ex
        all_verified = all_verified and verified
        cores.append(_build_core(kept, err_f, checker_f, by_id, catalog, verified))
        removed |= kept
        if exhausted:
            break

        active = all_ids - removed
        if used_box[0] >= budget:
            exhausted = True
            break
        _r, _c, err = replay(active)  # 成功 => 核外再无独立冲突

    return DiagnosisReport(
        ok=False,
        principal_type=None,
        obligations=catalog,
        cores=tuple(cores),
        minimal=all_verified and not exhausted,
        budget_exhausted=exhausted,
        replays_used=used_box[0],
        budget=budget,
    )
