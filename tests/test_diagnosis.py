"""诊断模式（polyinfer.diagnose）测试。

覆盖：

* 成功时主类型与普通入口逐字一致；
* 约束稳定编号（纯 AST 结构遍历，与推断状态无关）；
* 单义务冲突核、多义务"合起来才有问题"的包含极小核；
* occurs 与普通构造子不合可区分；
* 两处相互独立的错误分别成核，且核的选择按编号确定（不追求最小基数）；
* 值限制不因追踪放宽（共享引用不泛化、单态无实例化事件）；
* 同名变量遮蔽；
* 逐条移除最终核中的约束都可满足（独立重放白盒复核）；
* 实例化 / 替换绑定 / 统一的来源轨迹可追溯到义务编号；
* 独立预算耗尽时不假称极小；
* 每轮重放使用冻结原始约束（结果确定性、计数器不跨轮）；
* 非法 AST / 未绑定变量 / 超节点上限照常抛原异常。
"""

import unittest

from polyinfer import A, diagnose, infer
from polyinfer.ast import OBLIGATION_KINDS
from polyinfer.diagnosis import (
    ConflictEvidence,
    DiagnosisReport,
    diagnose,
    enumerate_obligations,
)
from polyinfer.errors import (
    InvalidAstError,
    NodeLimitError,
    UnboundVariableError,
)
from polyinfer.inference import run_infer_partial


# ----------------------------------------------------------- 测试夹具

def multi_member_program():
    """λx. let a = x 1 in (if x then 1 else 2)

    #1 apply(x,1) 要求 x : Int -> a；#2 if 条件要求 x : Bool；
    #3 if 两分支 Int=Int（无辜）。单独 #1 或 #2 都可满足，
    合起来才冲突，包含极小核必须是 {#1, #2}。
    """
    return A.lambda_(
        "x",
        A.let(
            "a",
            A.apply(A.var("x"), A.int_(1)),
            A.if_(A.var("x"), A.int_(1), A.int_(2)),
        ),
    )


def two_independent_bugs_program():
    """λx. λy.
         let a = x 1 in
         let b = y 1 in
         if x then (if y then 1 else 2)
                 else (if y then 3 else 4)

    x 的矛盾：#1 apply(x,1) 与 #3 外层 if 条件；
    y 的矛盾：#2 apply(y,1) 与 #5/#7 两个内层 if 条件（#5 与 #7
    对 y 的要求相同，彼此冗余——保留小编号 #5 体现"按编号确定、
    不要求最小基数"）。期望两个互不相交的核 {#1,#3}、{#2,#5}。
    """
    return A.lambda_(
        "x",
        A.lambda_(
            "y",
            A.let(
                "a",
                A.apply(A.var("x"), A.int_(1)),
                A.let(
                    "b",
                    A.apply(A.var("y"), A.int_(1)),
                    A.if_(
                        A.var("x"),
                        A.if_(A.var("y"), A.int_(1), A.int_(2)),
                        A.if_(A.var("y"), A.int_(3), A.int_(4)),
                    ),
                ),
            ),
        ),
    )


# ----------------------------------------------------------- 编号与成功路径

class EnumerationTests(unittest.TestCase):
    def test_kind_table_complete(self):
        self.assertEqual(OBLIGATION_KINDS, ("apply", "if_cond", "if_branches", "read", "write"))

    def test_no_obligations_for_pure_values(self):
        self.assertEqual(enumerate_obligations(A.lambda_("x", A.var("x"))), ())
        self.assertEqual(enumerate_obligations(A.let("x", A.int_(1), A.var("x"))), ())

    def test_numbering_is_preorder_and_stable(self):
        node = A.if_(A.int_(1), A.int_(2), A.int_(3))
        infos = enumerate_obligations(node)
        self.assertEqual([(i.number, i.kind) for i in infos], [(1, "if_cond"), (2, "if_branches")])

        node2 = A.apply(A.int_(1), A.int_(2))
        infos2 = enumerate_obligations(node2)
        self.assertEqual([(i.number, i.kind, i.path) for i in infos2],
                         [(1, "apply", ())])

    def test_numbering_of_two_bug_program(self):
        infos = enumerate_obligations(two_independent_bugs_program())
        self.assertEqual(
            [(i.number, i.kind) for i in infos],
            [
                (1, "apply"),         # let a = x 1
                (2, "apply"),         # let b = y 1
                (3, "if_cond"),       # 外层 if x
                (4, "if_branches"),
                (5, "if_cond"),       # then 分支内层 if y
                (6, "if_branches"),
                (7, "if_cond"),       # else 分支内层 if y
                (8, "if_branches"),
            ],
        )
        # 纯 AST 函数：同一棵树多次枚举逐字相等，与是否跑过推断无关。
        self.assertEqual(enumerate_obligations(two_independent_bugs_program()), infos)

    def test_paths_locate_inducing_node(self):
        infos = enumerate_obligations(multi_member_program())
        apply_info = infos[0]
        cond_info = infos[1]
        self.assertEqual(apply_info.path, ("body", "bound"))
        self.assertEqual(cond_info.path, ("body", "body"))


class SuccessPathTests(unittest.TestCase):
    def _assert_same_as_infer(self, node):
        r = diagnose(node)
        self.assertIsInstance(r, DiagnosisReport)
        self.assertTrue(r.ok)
        self.assertEqual(r.principal_type, infer(node).principal_type)
        self.assertEqual(r.conflict_numbers, ())
        self.assertEqual(r.cores, ())
        self.assertFalse(r.budget_exhausted)

    def test_success_still_records_instantiation_provenance(self):
        # 成功路径上普通 infer 的结果同样携带实例化来源事件：
        # id 被使用两次，每次一个实例，使用点分别回溯到根 let。
        prog = A.let(
            "id",
            A.lambda_("x", A.var("x")),
            A.if_(
                A.apply(A.var("id"), A.bool_(True)),
                A.apply(A.var("id"), A.int_(1)),
                A.int_(2),
            ),
        )
        result = infer(prog)
        id_events = [e for e in result.events if e.name == "id"]
        self.assertEqual(len(id_events), 2)
        self.assertEqual({e.origin for e in id_events}, {()})  # 根 let
        self.assertEqual(
            sorted(e.use_path for e in id_events),
            sorted([("body", "cond", "fn"), ("body", "then", "fn")]),
        )
        for e in id_events:
            self.assertEqual(len(e.renames), 1)  # ∀a.a 每次一个新生变量
        # 两次实例化的新生变量互不相同（使用点真正拿到独立副本）。
        news = [e.renames[0][1].ident for e in id_events]
        self.assertEqual(len(set(news)), 2)

    def test_identity(self):
        self._assert_same_as_infer(A.lambda_("x", A.var("x")))

    def test_compose(self):
        expr = A.lambda_(
            "f",
            A.lambda_(
                "g",
                A.lambda_("x", A.apply(A.var("f"), A.apply(A.var("g"), A.var("x")))),
            ),
        )
        self._assert_same_as_infer(expr)

    def test_polymorphic_let_and_ref_roundtrip(self):
        prog = A.let(
            "id",
            A.lambda_("x", A.var("x")),
            A.if_(
                A.apply(A.var("id"), A.bool_(True)),
                A.apply(A.var("id"), A.int_(1)),
                A.int_(2),
            ),
        )
        self._assert_same_as_infer(prog)
        self.assertEqual(diagnose(prog).principal_type, "Int")
        self._assert_same_as_infer(A.lambda_("r", A.write_(A.var("r"), A.read(A.var("r")))))

    def test_shadowing_success(self):
        prog = A.let("x", A.int_(1), A.let("x", A.bool_(True), A.var("x")))
        self._assert_same_as_infer(prog)
        self.assertEqual(diagnose(prog).principal_type, "Bool")


# ----------------------------------------------------------- 冲突核

class SingleConflictTests(unittest.TestCase):
    def test_if_condition_int(self):
        r = diagnose(A.if_(A.int_(1), A.int_(2), A.int_(3)))
        self.assertFalse(r.ok)
        self.assertEqual(r.conflict_numbers, (1,))
        self.assertTrue(r.minimal)
        self.assertFalse(r.budget_exhausted)
        core = r.cores[0]
        self.assertEqual(core.numbers, (1,))
        self.assertTrue(core.verified)
        ev = core.conflicts[0]
        self.assertTrue(ev.witness)
        self.assertEqual(ev.failure, "mismatch")
        self.assertEqual(ev.left_type, "Int")
        self.assertEqual(ev.right_type, "Bool")
        self.assertEqual(core.provenance.witness_number, 1)
        self.assertEqual(core.provenance.witness_kind, "mismatch")

    def test_self_application_is_occurs_not_mismatch(self):
        r = diagnose(A.lambda_("x", A.apply(A.var("x"), A.var("x"))))
        self.assertEqual(r.conflict_numbers, (1,))
        self.assertTrue(r.minimal)
        core = r.cores[0]
        self.assertEqual(core.provenance.witness_kind, "occurs")
        ev = core.conflicts[0]
        self.assertEqual(ev.failure, "occurs")
        self.assertTrue(ev.witness)
        self.assertIn("->", ev.right_type)  # 展开式是函数类型

    def test_mismatch_and_occurs_are_distinguishable(self):
        r_mismatch = diagnose(A.read(A.int_(1)))
        r_occurs = diagnose(A.lambda_("x", A.apply(A.var("x"), A.var("x"))))
        self.assertEqual(r_mismatch.cores[0].provenance.witness_kind, "mismatch")
        self.assertEqual(r_occurs.cores[0].provenance.witness_kind, "occurs")
        # 异常基类相同，kind 字段不同；编辑器可据此分别措辞。
        self.assertNotEqual(
            r_mismatch.cores[0].provenance.witness_kind,
            r_occurs.cores[0].provenance.witness_kind,
        )


class MultiMemberCoreTests(unittest.TestCase):
    def test_core_needs_both_constraints(self):
        node = multi_member_program()
        r = diagnose(node)
        self.assertEqual(r.conflict_numbers, (1, 2))
        self.assertTrue(r.minimal)
        core = r.cores[0]
        witness = [e for e in core.conflicts if e.witness]
        participants = [e for e in core.conflicts if not e.witness]
        self.assertEqual(len(witness), 1)
        self.assertEqual(witness[0].number, 2)  # #1 先把 x 绑成 Int->a，#2 随即失败
        self.assertEqual(witness[0].failure, "mismatch")
        self.assertEqual({e.number for e in participants}, {1})

    def test_each_member_removal_is_satisfiable_independently(self):
        """白盒复核：核 {#1,#2} 删去任意一条后，只含剩余成员的重放成功。"""
        node = multi_member_program()
        r = diagnose(node)
        catalog = r.obligations
        all_ids = frozenset(i.id for i in catalog)
        core_ids = frozenset(r.obligation(n).id for n in r.conflict_numbers)
        for n in r.conflict_numbers:
            kept = core_ids - {r.obligation(n).id}
            result, _checker, err = run_infer_partial(
                node, skipped_obligations=all_ids - kept
            )
            self.assertIsNone(err, f"删去 #{n} 后剩余核约束应可满足")
            self.assertIsNotNone(result)

    def test_each_individual_member_alone_is_satisfiable(self):
        """更强的直觉验证：#1 单独、#2 单独施加时程序都可类型化。"""
        node = multi_member_program()
        catalog = enumerate_obligations(node)
        all_ids = frozenset(i.id for i in catalog)
        for info in catalog:
            if info.number not in (1, 2):
                continue
            _r, _c, err = run_infer_partial(node, skipped_obligations=all_ids - {info.id})
            self.assertIsNone(err, f"义务 #{info.number} 单独不应构成冲突")

    def test_innocent_obligation_excluded(self):
        # #3（if 两分支 Int=Int）与冲突无关，不得进入核。
        r = diagnose(multi_member_program())
        self.assertNotIn(3, r.conflict_numbers)


class TwoIndependentBugsTests(unittest.TestCase):
    def test_two_disjoint_cores_found(self):
        r = diagnose(two_independent_bugs_program())
        self.assertFalse(r.ok)
        self.assertEqual(len(r.cores), 2)
        self.assertEqual(r.cores[0].numbers, (1, 3))
        self.assertEqual(r.cores[1].numbers, (2, 5))
        self.assertTrue(all(c.verified for c in r.cores))
        self.assertTrue(r.minimal)

    def test_each_core_is_individually_conflicting_and_minimal(self):
        node = two_independent_bugs_program()
        r = diagnose(node)
        catalog = r.obligations
        all_ids = frozenset(i.id for i in catalog)
        for core in r.cores:
            ids = frozenset(r.obligation(n).id for n in core.numbers)
            # 只含本核 => 必须冲突。
            _r1, _c1, err1 = run_infer_partial(node, skipped_obligations=all_ids - ids)
            self.assertIsNotNone(err1)
            # 删去核内任意一条 => 必须成功（在"只含本核"的语境下）。
            for n in core.numbers:
                _r2, _c2, err2 = run_infer_partial(
                    node, skipped_obligations=all_ids - (ids - {r.obligation(n).id})
                )
                self.assertIsNone(err2, f"核 {core.numbers} 删去 #{n} 后应可满足")

    def test_removing_first_core_reveals_second(self):
        """编辑器视角：修好第一组后，诊断应只剩第二组。"""
        node = two_independent_bugs_program()
        r = diagnose(node)
        first = frozenset(r.obligation(n).id for n in r.cores[0].numbers)
        all_ids = frozenset(i.id for i in r.obligations)
        # 用"跳过第一组"模拟修复，直接对剩余约束做诊断式重放：
        # 剩余约束仍不可满足，且失败义务属于第二核。
        _r, _c, err = run_infer_partial(node, skipped_obligations=first)
        self.assertIsNotNone(err)
        self.assertIn(err.obligation, all_ids - first)
        self.assertEqual(r.obligation(r.cores[1].provenance.witness_number).id,
                         err.obligation)

    def test_cores_disjoint_and_deterministic(self):
        node = two_independent_bugs_program()
        r1 = diagnose(node)
        r2 = diagnose(node)
        self.assertEqual(r1.cores, r2.cores)
        sets = [set(c.numbers) for c in r1.cores]
        self.assertEqual(set.intersection(*sets), set())
        self.assertEqual(r1.replays_used, r2.replays_used)


# ----------------------------------------------------------- 来源追踪

class ProvenanceTests(unittest.TestCase):
    def test_instantiation_and_bind_trace_to_numbers(self):
        # let id = λz.z in if (id 1) then 2 else 3
        # #3 apply(id,1)：实例化 id 并把实例变量绑成 Int；
        # #1 if 条件要求 Bool；#2 分支等式无辜。核 = {#1,#3}。
        node = A.let(
            "id",
            A.lambda_("z", A.var("z")),
            A.if_(A.apply(A.var("id"), A.int_(1)), A.int_(2), A.int_(3)),
        )
        r = diagnose(node)
        self.assertEqual(r.conflict_numbers, (1, 3))
        prov = r.cores[0].provenance
        self.assertEqual(prov.witness_number, 1)
        # 实例化来源：使用点（body/cond/fn 的 var）回溯到根 let。
        inst = [e for e in prov.instantiations if e.name == "id"]
        self.assertEqual(len(inst), 1)
        self.assertEqual(inst[0].origin, ())  # let 位于根
        self.assertEqual(inst[0].use_path, ("body", "cond", "fn"))
        self.assertTrue(inst[0].renames)  # 被限定变量 -> 新生变量对应非空
        # 替换绑定来源：把实例变量绑成 Int 的那次绑定来自 #3。
        int_binds = [b for b in prov.binds if b.rhs == "Int"]
        self.assertTrue(any(b.obligation_number == 3 for b in int_binds))
        # 统一等式都能反查到义务编号。
        for c in prov.unifications:
            self.assertTrue(c.obligation)

    def test_value_restriction_not_relaxed_no_r_instantiation(self):
        """共享引用在值限制下单态：失败核的来源轨迹里不得有 r 的实例化。"""
        hole = A.let(
            "r",
            A.ref(A.lambda_("z", A.var("z"))),
            A.let("u", A.write_(A.var("r"), A.int_(1)), A.read(A.var("r"))),
        )
        r = diagnose(hole)
        self.assertTrue(r.minimal)
        prov = r.cores[0].provenance
        self.assertIsNone(next((e for e in prov.instantiations if e.name == "r"), None))
        # let r=ref(...) 的泛化事件必须为空泛化（值限制留痕仍在）。
        self.assertTrue(all(() == g.generalized for g in prov.generalizations))

    def test_value_restriction_holds_even_when_obligations_skipped(self):
        """白盒：跳过义务的重放**成功**后，ref 绑定依旧空泛化——
        追踪/跳过机制绝不放宽值限制。"""
        # let r = ref(1) in (r := true; !r)：write 义务是 #2（整程序义务
        # 编号：内层 let 的 write）。跳过后读 r : Int 成功；
        # 关键：r 的方案 generalized 必须仍为空（ref 是扩展式）。
        node = A.let(
            "r", A.ref(A.int_(1)),
            A.let("u", A.write_(A.var("r"), A.bool_(True)), A.read(A.var("r"))),
        )
        catalog = enumerate_obligations(node)
        write_info = next(i for i in catalog if i.kind == "write")
        # 只跳过 write 义务：程序不再施加 Ref(Int)=Ref(Bool) 的等式。
        result, checker, err = run_infer_partial(
            node, skipped_obligations=frozenset({write_info.id})
        )
        self.assertIsNone(err)
        self.assertEqual(result.principal_type, "Int")
        gen_r = next(g for g in result.generalizations if g.name == "r")
        self.assertEqual(gen_r.bound_tag, "ref")
        self.assertEqual(gen_r.generalized, ())
        # 独立检查器也必须认可这份跳过重放的成功结果。
        result.verify()

    def test_occurs_core_removal_is_satisfiable(self):
        # λx. x x：唯一义务 #1；跳过它后 body 只是 x，程序可类型化。
        node = A.lambda_("x", A.apply(A.var("x"), A.var("x")))
        r = diagnose(node)
        self.assertEqual(r.cores[0].provenance.witness_kind, "occurs")
        info = r.obligation(1)
        result, _c, err = run_infer_partial(
            node, skipped_obligations=frozenset({info.id})
        )
        self.assertIsNone(err)
        # 参数 b 与（未施加等式而保持自由的）返回变量 c：
        self.assertEqual(result.principal_type, "a -> b")

    def test_replay_uses_frozen_constraints_and_fresh_state(self):
        # 两次诊断的最终见证轨迹逐字相同：每轮重放的 TVar 计数器都从 0
        # 开始，且使用的是同一冻结 AST，而非上一轮失败后的状态。
        node = two_independent_bugs_program()
        r1 = diagnose(node)
        r2 = diagnose(node, budget=4096)
        self.assertEqual(
            [c.provenance.witness_number for c in r1.cores],
            [c.provenance.witness_number for c in r2.cores],
        )
        leaf1 = r1.cores[0].conflicts[0]
        leaf2 = r2.cores[0].conflicts[0]
        self.assertEqual((leaf1.left_type, leaf1.right_type),
                         (leaf2.left_type, leaf2.right_type))


# ----------------------------------------------------------- 遮蔽

class ShadowingDiagnosisTests(unittest.TestCase):
    def test_conflict_resolves_to_inner_shadowed_variable(self):
        # let x = 1 in let x = true in x 1：内层 x : Bool 被当函数用。
        node = A.let(
            "x", A.int_(1),
            A.let("x", A.bool_(True), A.apply(A.var("x"), A.int_(0))),
        )
        r = diagnose(node)
        self.assertEqual(r.conflict_numbers, (1,))
        ev = r.cores[0].conflicts[0]
        # 诱发节点是内层 body 的 apply；外层 let 不受指责。
        self.assertEqual(ev.kind, "apply")
        self.assertEqual(ev.obligation[1], ("body", "body"))
        self.assertIn("Bool", ev.left_type + ev.right_type)

    def test_shadowed_usage_chain_attribution(self):
        # 外层 let 无关。先序编号：if 的两个义务 #1/#2 先登记，其 cond
        # 子树里的 apply 是 #3；核应为 {#3} 单独成核。
        node = A.let(
            "x", A.int_(1),
            A.let(
                "x", A.bool_(True),
                A.if_(A.apply(A.var("x"), A.int_(0)), A.int_(1), A.int_(2)),
            ),
        )
        r = diagnose(node)
        self.assertEqual(r.conflict_numbers, (3,))
        self.assertTrue(r.minimal)
        ev = r.cores[0].conflicts[0]
        self.assertEqual(ev.obligation[1], ("body", "body", "cond"))
        self.assertIn("Bool", ev.left_type + ev.right_type)

    def test_polymorphic_shadowing_origin_attribution(self):
        # 内外两层同名 id：内层 body 里 id 的实例化事件必须回溯到**内层**
        # let（路径 (body,)），而不是外层根 let（()）。
        prog = A.let(
            "id",
            A.lambda_("x", A.var("x")),
            A.let(
                "id",
                A.lambda_("y", A.var("y")),
                A.apply(A.var("id"), A.int_(1)),
            ),
        )
        result = infer(prog)
        id_events = [e for e in result.events if e.name == "id"]
        self.assertEqual(len(id_events), 1)
        self.assertEqual(id_events[0].origin, ("body",))
        self.assertEqual(id_events[0].use_path, ("body", "body", "fn"))
        # 诊断成功路径一致。
        self.assertTrue(diagnose(prog).ok)


# ----------------------------------------------------------- 预算

class BudgetTests(unittest.TestCase):
    def test_budget_one_marks_exhausted_and_not_minimal(self):
        node = two_independent_bugs_program()
        r = diagnose(node, budget=1)
        self.assertTrue(r.budget_exhausted)
        self.assertFalse(r.minimal)
        # 有未缩减完的证据，但核标记为未验证。
        self.assertGreaterEqual(len(r.cores), 1)
        self.assertFalse(r.cores[0].verified)
        # describe 必须明确告知未宣称极小。
        self.assertIn("不宣称极小", r.describe())

    def test_sufficient_budget_completes(self):
        node = two_independent_bugs_program()
        r = diagnose(node, budget=4096)
        self.assertFalse(r.budget_exhausted)
        self.assertTrue(r.minimal)
        self.assertEqual([c.numbers for c in r.cores], [(1, 3), (2, 5)])
        self.assertLess(r.replays_used, 4096)

    def test_budget_validated(self):
        with self.assertRaises(TypeError):
            diagnose(A.int_(1), budget=True)
        with self.assertRaises(TypeError):
            diagnose(A.int_(1), budget=1.5)
        with self.assertRaises(ValueError):
            diagnose(A.int_(1), budget=0)

    def test_budget_does_not_affect_success_path(self):
        # 可类型化程序首轮即成功，任何预算 >=1 都给出相同主类型。
        node = A.lambda_("x", A.var("x"))
        for b in (1, 2, 64):
            r = diagnose(node, budget=b)
            self.assertTrue(r.ok)
            self.assertEqual(r.principal_type, "a -> a")


# ----------------------------------------------------------- 输入透传

class InputPassthroughTests(unittest.TestCase):
    def test_invalid_ast_still_raises(self):
        with self.assertRaises(InvalidAstError):
            diagnose(42)
        with self.assertRaises(InvalidAstError):
            diagnose({"tag": "int", "value": True})  # bool 不静默当 int

    def test_unbound_variable_still_raises(self):
        with self.assertRaises(UnboundVariableError):
            diagnose(A.var("oops"))
        with self.assertRaises(UnboundVariableError):
            diagnose(A.let("f", A.var("f"), A.int_(1)))  # 递归 let

    def test_node_limit_still_resource_error(self):
        chain = A.int_(0)
        for _ in range(250):
            chain = A.apply(chain, A.int_(0))
        with self.assertRaises(NodeLimitError):
            diagnose(chain)

    def test_float_rejected_identically_to_plain_infer(self):
        with self.assertRaises(InvalidAstError):
            diagnose({"tag": "int", "value": 2.5})


# ----------------------------------------------------------- 报告呈现

class ReportPresentationTests(unittest.TestCase):
    def test_describe_success(self):
        text = diagnose(A.lambda_("x", A.var("x"))).describe()
        self.assertIn("a -> a", text)

    def test_describe_failure_lists_numbers_and_locations(self):
        r = diagnose(multi_member_program())
        text = r.describe()
        self.assertIn("#1", text)
        self.assertIn("#2", text)
        self.assertNotIn("#3", text)
        self.assertIn("apply", text)
        self.assertIn("if_cond", text)
        self.assertIn("<root>/body/bound", text)

    def test_describe_two_cores(self):
        text = diagnose(two_independent_bugs_program()).describe()
        self.assertIn("冲突集 1", text)
        self.assertIn("冲突集 2", text)

    def test_obligation_lookup(self):
        r = diagnose(A.if_(A.int_(1), A.int_(2), A.int_(3)))
        self.assertEqual(r.obligation(1).kind, "if_cond")
        self.assertEqual(r.obligation(2).kind, "if_branches")


if __name__ == "__main__":
    unittest.main()
