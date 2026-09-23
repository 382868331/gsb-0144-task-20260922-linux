"""v2 / 模块层的篡改检测：证明独立检查器确实重算效应、缺席约束、
模块导出效应与泛化资格，而不是信任求解器给出的结论或布尔标志。
"""

import unittest
from dataclasses import replace

from polyinfer import A, infer
from polyinfer.reference import (
    ReferenceCheckFailure,
    check_module_report,
    check_result,
)


class EffectResultTamperTests(unittest.TestCase):
    def _write_result(self):
        return infer(A.write_(A.ref(A.int_(1)), A.int_(2)), language="v2")

    def test_effect_set_lie_is_caught(self):
        # 真实效应是 W；谎报为空集必须被独立效应重算抓到。
        r = self._write_result()
        tampered = replace(r, effects=frozenset())
        with self.assertRaises(ReferenceCheckFailure):
            check_result(tampered)

    def test_extra_effect_lie_is_caught(self):
        r = infer(A.int_(1), language="v2")  # 真实纯
        tampered = replace(r, effects=frozenset({"W"}))
        with self.assertRaises(ReferenceCheckFailure):
            check_result(tampered)

    def test_undecided_flag_lie_caught(self):
        r = infer(A.evar(), language="v2")  # 真实未决
        self.assertTrue(r.effects_undecided)
        tampered = replace(r, effects_undecided=False)
        with self.assertRaises(ReferenceCheckFailure):
            check_result(tampered)

    def test_false_undecided_on_determined_program_caught(self):
        r = infer(A.read(A.ref(A.int_(1))), language="v2")
        tampered = replace(r, effects_undecided=True)
        with self.assertRaises(ReferenceCheckFailure):
            check_result(tampered)

    def test_handled_exception_with_write_still_audited(self):
        node = A.try_(A.raise_(A.int_(1)), "e",
                      A.write_(A.ref(A.int_(0)), A.int_(2)))
        r = infer(node, language="v2")
        self.assertEqual(set(r.effects), {"W"})
        # 谎报 X 已让整个表达式纯净：必须被抓到（W 不是 try 能移除的）。
        tampered = replace(r, effects=frozenset())
        with self.assertRaises(ReferenceCheckFailure):
            check_result(tampered)


class RowAbsenceTamperTests(unittest.TestCase):
    def test_subst_inserting_forbidden_field_into_absence_row_caught(self):
        # 程序含 extend 缺席约束：{r with g=...} 要求 r 的尾不含 g。
        node = A.lambda_("x", A.let(
            "y", A.extend(A.var("x"), "g", A.int_(1)),
            A.get(A.var("y"), "g"),
        ))
        r = infer(node, language="v2")
        check_result(r)  # 原结果通过
        self.assertTrue(r.row_absences)
        # 篡改替换：把缺席尾变量绑到含 g 的行（模拟求解器漏检成环）。
        from polyinfer.types import TRow, TRec, REmpty, TVar

        absence_row = r.row_absences[0].row
        subst = dict(r.subst)
        subst[absence_row] = TRow((("g", TVar(99999)),), REmpty())
        tampered = replace(r, subst=subst)
        with self.assertRaises(ReferenceCheckFailure):
            check_result(tampered)


class ValueRestrictionV2TamperTests(unittest.TestCase):
    def test_record_generalization_lie_caught(self):
        # let r = ref(...) 是扩展式（record 内含 ref），不得泛化。
        node = A.let(
            "r", A.record([("c", A.ref(A.int_(1)))]),
            A.var("r"),
        )
        r = infer(node, language="v2")
        check_result(r)
        gens = list(r.generalizations)
        self.assertEqual(gens[0].generalized, ())
        # 谎报泛化集合（塞进一个不存在的行/类型变量）：必须被抓。
        from polyinfer.types import RVar

        g = replace(gens[0], generalized=(RVar(99998),))
        tampered = replace(r, generalizations=(g,))
        with self.assertRaises(ReferenceCheckFailure):
            check_result(tampered)


class ModuleReportTamperTests(unittest.TestCase):
    def _bundle_and_report(self):
        from polyinfer.modules import check_modules

        M = A.make_module("M",
            {"put": "Int -{W}-> Unit", "get": "Int -{R}-> Int"},
            {"cell": A.ref(A.int_(0)),
             "put": A.lambda_("_", A.write_(A.var("cell"), A.var("_"))),
             "get": A.lambda_("_", A.read(A.var("cell")))})
        _, report = check_modules({"M": M})
        check_module_report(report)
        return report

    def test_export_effect_lie_caught(self):
        report = self._bundle_and_report()
        traces = list(report.traces)
        mt = traces[0]
        # 篡改：把 put 的观察效应抹成纯（空集），与签名 W 不一致。
        bad_sigs = tuple(
            replace(es, observed_effects=())
            if es.export == "put" else es
            for es in mt.export_effects
        )
        bad_mt = replace(mt, export_effects=bad_sigs)
        tampered = replace(report, traces=(bad_mt,))
        with self.assertRaises(ReferenceCheckFailure):
            check_module_report(tampered)

    def test_module_constraint_tamper_caught(self):
        from polyinfer.types import BOOL

        report = self._bundle_and_report()
        mt = report.traces[0]
        constraints = list(mt.constraints)
        tampered_c = list(constraints)
        idx = next(i for i, c in enumerate(tampered_c) if "Int" in str(c.left)
                   or "Int" in str(c.right))
        tampered_c[idx] = replace(tampered_c[idx], left=BOOL)
        bad_mt = replace(mt, constraints=tuple(tampered_c))
        bad_report = replace(report, traces=(bad_mt,))
        with self.assertRaises(ReferenceCheckFailure):
            check_module_report(bad_report)

    def test_internal_expansive_generalization_lie_caught(self):
        # 模块含一个扩展式内部定义（record 里放 ref）；它必须单态。
        # 人为把它的泛化记录改成“泛化了一个自由变量”，独立检查器从源 AST
        # 重判值形状后必须拒绝（不信任 trace 结论）。
        from polyinfer.modules import check_modules
        from polyinfer.types import RVar

        M = A.make_module("M", {"k": "Int"},
            {"k": A.int_(0),
             "box": A.record([("c", A.ref(A.int_(1)))])})
        _, report = check_modules({"M": M})
        mt = report.traces[0]
        gens = {g.name: g for g in mt.generalizations}
        self.assertIn("box", gens)
        self.assertEqual(gens["box"].generalized, ())
        lie = replace(gens["box"], generalized=(RVar(424242),))
        new_gens = tuple(lie if g.name == "box" else g for g in mt.generalizations)
        bad_mt = replace(mt, generalizations=new_gens)
        bad_report = replace(report, traces=(bad_mt,))
        with self.assertRaises(ReferenceCheckFailure):
            check_module_report(bad_report)


if __name__ == "__main__":
    unittest.main()
