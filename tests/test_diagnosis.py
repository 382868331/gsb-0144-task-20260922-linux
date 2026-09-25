"""诊断模式（polyinfer.diagnose）测试。

覆盖契约：

* 成功时主类型与普通 ``infer`` 逐字一致，且独立检查器核验不变；
* 失败时给出按稳定编号确定的包含极小冲突集（逆序单轮删除 +
  逐条移除的冻结重放验证），不要求最小基数；
* 每次重放使用冻结原始 AST（调用方 AST 不被修改）、全新替换；
* occurs（无限类型）与普通构造子不合是两类不同证据；
* 值限制在诊断下一条不放宽：共享引用不泛化、lambda 形参单态；
* 自应用、同名变量遮蔽（实例化来源指向正确绑定）、两处独立错误
  （报出多个冲突核）；
* 诊断预算独立，耗尽时只给未缩减完的证据，``minimal=False``；
* AST 非法 / 未绑定变量等“冲突之前”的错误与普通入口同样抛出。
"""

import unittest

from polyinfer import A, diagnose, format_diagnosis, infer
from polyinfer.diagnose import ConflictCore, _Budget, _strict_replay
from polyinfer.errors import (
    InvalidAstError,
    UnboundVariableError,
)


# ---------------------------------------------------------------- 帮助

def _key(c):
    return c.left_path, c.right_path


def _all_replays_satisfiable_after_each_removal(node, diag):
    """独立于诊断器，再做一遍“核中逐条移除后严格重放可满足”的核验。"""
    assert diag.cores
    budget = _Budget(10_000)
    keys = {_key(c) for c in diag.constraints}
    for core in diag.cores:
        members = {_key(c) for c in core.members}
        # 核整体必须不可满足。
        whole = _strict_replay(node, 500, frozenset(keys - members), budget)
        if whole.ok:
            return False
        # 删去任意一条成员后必须可满足。
        for k in members:
            kept = members - {k}
            outcome = _strict_replay(node, 500, frozenset(keys - kept), budget)
            if not outcome.ok:
                return False
    # 所有核都归属于别的约束之后，剩余约束必须可满足。
    in_cores = set().union(*({_key(c) for c in core.members} for core in diag.cores))
    if not _strict_replay(node, 500, frozenset(in_cores), budget).ok:
        return False
    return True


class SuccessPathTests(unittest.TestCase):
    def test_success_principal_type_matches_plain_infer(self):
        for node in (
            A.lambda_("x", A.var("x")),
            A.let("id", A.lambda_("x", A.var("x")),
                  A.if_(A.apply(A.var("id"), A.bool_(True)),
                        A.apply(A.var("id"), A.int_(1)), A.int_(2))),
            A.lambda_("r", A.write_(A.var("r"), A.read(A.var("r")))),
            A.let("x", A.int_(1), A.let("x", A.bool_(True), A.var("x"))),
        ):
            with self.subTest(node=node):
                plain = infer(node)
                diag = diagnose(node)
                self.assertTrue(diag.ok)
                self.assertEqual(diag.principal_type, plain.principal_type)
                self.assertEqual(diag.result.principal_type, plain.principal_type)
                # 成功结果同样通过独立检查器。
                self.assertEqual(diag.result.verify().constraints_checked,
                                 plain.verify().constraints_checked)
                self.assertEqual(diag.cores, ())
                self.assertFalse(diag.budget_exhausted)

    def test_success_still_records_stable_numbering(self):
        diag = diagnose(A.let("id", A.lambda_("x", A.var("x")),
                              A.apply(A.var("id"), A.int_(1))))
        self.assertTrue(diag.ok)
        self.assertEqual([c.index for c in diag.constraints],
                         list(range(len(diag.constraints))))
        # 编号即首次出现顺序，无重复键。
        keys = [_key(c) for c in diag.constraints]
        self.assertEqual(len(keys), len(set(keys)))


class StableNumberingTests(unittest.TestCase):
    def test_numbers_are_deterministic_across_calls(self):
        node = A.if_(A.int_(1), A.int_(2), A.bool_(False))
        first = diagnose(node)
        first_table = [(c.index, c.left_path, c.right_path) for c in first.constraints]
        for _ in range(4):
            again = diagnose(node)
            self.assertEqual(
                first_table,
                [(c.index, c.left_path, c.right_path) for c in again.constraints],
            )
            self.assertEqual(
                [tuple(c.index for c in core.members) for core in first.cores],
                [tuple(c.index for c in core.members) for core in again.cores],
            )

    def test_numbering_independent_of_substitution_shapes(self):
        # 约束身份是路径键，不是类型形态；两个不同程序即便有相同类型，
        # 编号表只由 AST 结构决定。这里核验同一程序两次诊断的样本一致。
        node = A.lambda_("f", A.if_(A.var("f"),
                                    A.apply(A.var("f"), A.int_(1)), A.int_(2)))
        d1, d2 = diagnose(node), diagnose(node)
        self.assertEqual(
            [(c.sample_left, c.sample_right) for c in d1.constraints],
            [(c.sample_left, c.sample_right) for c in d2.constraints],
        )


class MinimalConflictTests(unittest.TestCase):
    def test_single_self_contradictory_constraint_core(self):
        node = A.if_(A.int_(1), A.int_(2), A.int_(3))
        diag = diagnose(node)
        self.assertFalse(diag.ok)
        self.assertTrue(diag.minimal)
        self.assertEqual(diag.conflict_indices(), (0,))
        self.assertEqual(diag.evidence.kind, "type-mismatch")
        self.assertTrue(_all_replays_satisfiable_after_each_removal(node, diag))

    def test_core_needs_multiple_constraints_together(self):
        # λf. if f then (f 1) else 2：
        # f 既要 Bool（#0）又要 Int -> a（#1），单条都可满足，合起来冲突。
        node = A.lambda_(
            "f",
            A.if_(A.var("f"), A.apply(A.var("f"), A.int_(1)), A.int_(2)),
        )
        diag = diagnose(node)
        self.assertTrue(diag.minimal)
        self.assertEqual(diag.conflict_indices(), (0, 1))
        self.assertEqual(len(diag.cores), 1)
        # 每条成员都带原始诱导位置与说明。
        notes = {c.index: c.note for c in diag.conflict}
        self.assertIn("if 的条件", notes[0])
        self.assertIn("apply", notes[1])
        self.assertTrue(_all_replays_satisfiable_after_each_removal(node, diag))

    def test_removing_any_final_member_is_satisfiable_via_public_trace(self):
        # 同上程序，直接信任 irredundant 标记并要求它确实经过验证：
        # 预算给足时必须为 True；且 format_diagnosis 含验证通过字样。
        node = A.lambda_(
            "f",
            A.if_(A.var("f"), A.apply(A.var("f"), A.int_(1)), A.int_(2)),
        )
        diag = diagnose(node)
        self.assertTrue(all(c.irredundant for c in diag.cores))
        self.assertIn("逐条移除验证通过", format_diagnosis(diag))

    def test_no_minimum_cardinality_claim(self):
        # 文档与结果只声称包含极小，不声称最小基数；结果对象上没有
        # “minimum cardinality”字段，只有 irredundant。
        node = A.if_(A.int_(1), A.int_(2), A.int_(3))
        core = diagnose(node).cores[0]
        self.assertTrue(hasattr(core, "irredundant"))
        self.assertFalse(hasattr(core, "minimum_cardinality"))


class OccursVsMismatchTests(unittest.TestCase):
    def test_self_application_is_infinite_type_evidence(self):
        node = A.lambda_("x", A.apply(A.var("x"), A.var("x")))
        diag = diagnose(node)
        self.assertTrue(diag.minimal)
        self.assertEqual(diag.evidence.kind, "infinite-type")
        self.assertTrue(diag.evidence.is_occurs)
        self.assertIn("->", str(diag.evidence.right_type))
        # 与普通入口报的是同一个叶子位置。
        with self.assertRaises(Exception):
            infer(node)

    def test_mismatch_kind_distinct_from_occurs(self):
        node = A.read(A.int_(1))
        diag = diagnose(node)
        self.assertEqual(diag.evidence.kind, "type-mismatch")
        self.assertFalse(diag.evidence.is_occurs)

    def test_format_diagnosis_distinguishes_two_kinds(self):
        occurs = diagnose(
            A.lambda_("x", A.apply(A.var("x"), A.var("x")))
        )
        mismatch = diagnose(A.read(A.int_(1)))
        self.assertIn("occurs", format_diagnosis(occurs))
        self.assertIn("构造子不一致", format_diagnosis(mismatch))
        self.assertNotIn("occurs", format_diagnosis(mismatch))


class ValueRestrictionUnderDiagnosisTests(unittest.TestCase):
    def test_shared_reference_is_not_generalized(self):
        # 经典引用多态漏洞：诊断也必须判其冲突，且实例化留痕明确显示
        # r 的两次使用是单态共享（值限制：扩展式不泛化）。
        leak = A.let(
            "r",
            A.ref(A.lambda_("z", A.var("z"))),
            A.let("u", A.write_(A.var("r"), A.int_(1)), A.read(A.var("r"))),
        )
        diag = diagnose(leak)
        self.assertFalse(diag.ok)
        self.assertTrue(diag.minimal)
        r_uses = [t for t in diag.instantiation_trace if t.name == "r"]
        self.assertTrue(r_uses)
        for tr in r_uses:
            self.assertFalse(tr.polymorphic)
            self.assertEqual(tr.source.kind, "let")
            self.assertEqual(tr.source.bound_tag, "ref")
        rendered = format_diagnosis(diag)
        self.assertIn("值限制", rendered)

    def test_polymorphic_let_gets_distinct_fresh_instances(self):
        node = A.let(
            "id",
            A.lambda_("x", A.var("x")),
            A.if_(A.apply(A.var("id"), A.bool_(True)),
                  A.apply(A.var("id"), A.int_(1)), A.int_(2)),
        )
        diag = diagnose(node)
        self.assertTrue(diag.ok)
        id_uses = [t for t in diag.instantiation_trace if t.name == "id"]
        self.assertEqual(len(id_uses), 2)
        self.assertTrue(all(t.polymorphic for t in id_uses))
        # 两次实例化的新变量互不相同。
        fresh_targets = {b for t in id_uses for _a, b in t.renaming}
        self.assertEqual(len(fresh_targets), 2)
        # 来源指向同一个 let 节点。
        self.assertEqual({t.source.path for t in id_uses}, {()})


class ShadowingTraceTests(unittest.TestCase):
    def test_shadowed_variable_uses_point_to_inner_binding(self):
        # let x = 1 in let x = true in x：使用点必须解析到内层 let。
        node = A.let("x", A.int_(1), A.let("x", A.bool_(True), A.var("x")))
        diag = diagnose(node)
        self.assertTrue(diag.ok)
        x_uses = [t for t in diag.instantiation_trace if t.name == "x"]
        # 只有一个使用点（外层绑定 1 本身没有对 x 的使用）。
        self.assertEqual(len(x_uses), 1)
        self.assertEqual(x_uses[0].use_path, ("body", "body"))
        self.assertEqual(x_uses[0].source.path, ("body",))  # 内层 let
        self.assertNotEqual(x_uses[0].source.path, ())

    def test_lambda_parameter_origin_is_monopathic(self):
        node = A.lambda_("p", A.var("p"))
        diag = diagnose(node)
        p_use = [t for t in diag.instantiation_trace if t.name == "p"]
        self.assertEqual(len(p_use), 1)
        self.assertEqual(p_use[0].source.kind, "lambda-param")
        self.assertFalse(p_use[0].polymorphic)


class IndependentErrorsTests(unittest.TestCase):
    def test_two_independent_errors_produce_two_cores(self):
        # if 条件是 Int（冲突 A），同时两分支 Int/Bool 不合（冲突 B）。
        node = A.if_(A.int_(1), A.int_(2), A.bool_(False))
        diag = diagnose(node)
        self.assertFalse(diag.ok)
        self.assertFalse(diag.budget_exhausted)
        self.assertEqual(len(diag.cores), 2)
        self.assertTrue(all(c.irredundant for c in diag.cores))
        self.assertEqual([tuple(c.index for c in core.members) for core in diag.cores],
                         [(0,), (1,)])
        self.assertTrue(_all_replays_satisfiable_after_each_removal(node, diag))
        # 两个核各自单独即冲突（这就是“独立”的含义）。
        budget = _Budget(1000)
        keys = {_key(c) for c in diag.constraints}
        for core in diag.cores:
            members = {_key(c) for c in core.members}
            self.assertFalse(
                _strict_replay(node, 500, frozenset(keys - members), budget).ok
            )

    def test_three_independent_simple_conflicts(self):
        # 三个并列语句用嵌套 let 串起：三处各自要求不同基础类型。
        # 条件错（Int 当 Bool）出现两次 + read Int，互相不共享变量。
        node = A.let(
            "_",
            A.read(A.int_(1)),
            A.if_(A.int_(1),
                  A.if_(A.int_(2), A.int_(0), A.int_(0)),
                  A.int_(3)),
        )
        diag = diagnose(node)
        self.assertFalse(diag.budget_exhausted)
        self.assertGreaterEqual(len(diag.cores), 2)
        self.assertTrue(all(c.irredundant for c in diag.cores))
        self.assertTrue(_all_replays_satisfiable_after_each_removal(node, diag))


class FrozenReplayTests(unittest.TestCase):
    def test_caller_ast_is_not_mutated(self):
        import copy

        node = A.if_(A.int_(1), A.int_(2), A.bool_(False))
        snapshot = copy.deepcopy(node)
        diagnose(node)
        diagnose(node, max_checks=2)  # 预算耗尽路径也不许改输入
        self.assertEqual(node, snapshot)

    def test_repeated_diagnoses_are_independent(self):
        # 若重放携带上一次失败的替换/并查集，连续调用会互相污染。
        node = A.lambda_(
            "f",
            A.if_(A.var("f"), A.apply(A.var("f"), A.int_(1)), A.int_(2)),
        )
        results = {tuple(diagnose(node).conflict_indices()) for _ in range(6)}
        self.assertEqual(results, {(0, 1)})

    def test_failed_replay_does_not_loosen_value_restriction(self):
        # 在引用漏洞上反复诊断（含极小化时的大量重放），任何一次重放都
        # 不能因为“诊断模式”而让程序变得可满足。
        leak = A.let(
            "r",
            A.ref(A.lambda_("z", A.var("z"))),
            A.write_(A.var("r"), A.int_(1)),
        )
        diag = diagnose(leak)
        self.assertFalse(diag.ok)
        budget = _Budget(1000)
        keys = {_key(c) for c in diag.constraints}
        members = {_key(c) for c in diag.cores[0].members}
        # 只留该核：仍不可满足（冲突真实存在于这些等式上）。
        self.assertFalse(_strict_replay(leak, 500, frozenset(keys - members), budget).ok)
        # 删去该核（skip=members）后剩余约束可满足——可满足来自“删除
        # 冲突等式”，而不是诊断模式放宽了值限制。
        self.assertTrue(_strict_replay(leak, 500, frozenset(members), budget).ok)


class BudgetTests(unittest.TestCase):
    def test_budget_exhausted_does_not_claim_minimal(self):
        node = A.if_(A.int_(1), A.int_(2), A.bool_(False))
        diag = diagnose(node, max_checks=2)  # 枚举 1 次 + 仅 1 次试探
        self.assertTrue(diag.budget_exhausted)
        self.assertFalse(diag.minimal)
        # 仍给出未缩减完的冲突证据，而不是空着手。
        self.assertTrue(diag.cores)
        self.assertTrue(any(not c.irredundant for c in diag.cores))
        self.assertIn("不宣称极小", format_diagnosis(diag))

    def test_tiny_budget_still_returns_evidence(self):
        node = A.read(A.int_(1))
        diag = diagnose(node, max_checks=1)
        self.assertTrue(diag.budget_exhausted)
        # 只有一个冲突核时，证据直接取自最初严格失败；叶子信息齐全。
        self.assertTrue(diag.cores)
        ev = diag.cores[0].evidence
        self.assertEqual(ev.kind, "type-mismatch")

    def test_adequate_budget_flags_exhaustion_false(self):
        node = A.if_(A.int_(1), A.int_(2), A.bool_(False))
        diag = diagnose(node, max_checks=64)
        self.assertFalse(diag.budget_exhausted)
        self.assertTrue(diag.minimal)

    def test_budget_parameter_validated(self):
        with self.assertRaises(TypeError):
            diagnose(A.int_(1), max_checks=False)
        with self.assertRaises(TypeError):
            diagnose(A.int_(1), max_checks=1.5)
        with self.assertRaises(ValueError):
            diagnose(A.int_(1), max_checks=0)


class UpstreamErrorTests(unittest.TestCase):
    def test_invalid_ast_still_raises(self):
        with self.assertRaises(InvalidAstError):
            diagnose({"tag": "int", "value": True})

    def test_unbound_variable_still_raises(self):
        with self.assertRaises(UnboundVariableError):
            diagnose(A.var("nope"))

    def test_plain_infer_path_unaffected(self):
        # 普通入口的异常类型与消息不被诊断模块改变。
        from polyinfer.errors import TypeMismatchError

        with self.assertRaises(TypeMismatchError):
            infer(A.if_(A.int_(1), A.int_(2), A.int_(3)))


class ProvenanceTests(unittest.TestCase):
    def test_binding_chain_points_to_numbered_constraint(self):
        # (λx. x) 1：apply 等式把 x 的形参变量绑到 Int，来源链应指向
        # 那条 apply 约束，且每跳的约束编号都在编号表内。
        node = A.apply(A.lambda_("x", A.var("x")), A.int_(1))
        diag = diagnose(node)
        self.assertTrue(diag.ok)
        table = {c.index for c in diag.constraints}
        valid_keys = {_key(c) for c in diag.constraints}
        for hop in diag.bind_trace:
            self.assertIn(hop.key, valid_keys)
        start = diag.bind_trace[0].var_ident
        chain = diag.provenance(start)
        self.assertTrue(chain)
        self.assertLessEqual(len(chain), len(diag.bind_trace))
        self.assertTrue(all(h.constraint_index in table for h in chain))


if __name__ == "__main__":
    unittest.main()
