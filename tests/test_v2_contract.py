"""v2 输入契约与独立检查器篡改检测。

覆盖：v1 入口拒绝 v2 语法；version 参数校验；mvar 上下文限制；
模块包结构错误；独立检查器对效应、行、危险变量/field_tags 留痕的
篡改检测；以及一个小规模类型片段的**穷举**独立参考核对。
"""

import unittest
from dataclasses import replace

from polyinfer import A, infer
from polyinfer.ast import (
    export, module, package, rvar, signature, tcon, tfun,
    trecord, tref, tvar,
)
from polyinfer.errors import (
    InvalidAstError,
    ModuleStructureError,
    UnboundVariableError,
)
from polyinfer.modules import ModuleChecker, check_package
from polyinfer.reference import ReferenceCheckFailure, check_result


class VersionContractTests(unittest.TestCase):
    def test_v1_rejects_record_node(self):
        with self.assertRaises(InvalidAstError):
            infer(A.record([("x", A.int_(1))]))  # 缺省 version=1

    def test_v1_rejects_raise_node(self):
        with self.assertRaises(InvalidAstError):
            infer(A.raise_(A.exn("E")))

    def test_bad_version_value(self):
        with self.assertRaises(ValueError):
            infer(A.int_(1), version=3)

    def test_v2_accepts_v1_nodes(self):
        r = infer(A.lambda_("x", A.var("x")), version=2)
        self.assertEqual(r.principal_type, "a -> a")
        self.assertEqual(r.version, 2)

    def test_mvar_rejected_in_standalone_expression(self):
        with self.assertRaises(InvalidAstError):
            infer(A.mvar("M", "f"), max_nodes=1000, version=2)

    def test_raise_requires_exn_literal_type(self):
        # raise 非 Exn 在推断期统一失败。
        from polyinfer.errors import TypeMismatchError
        with self.assertRaises(TypeMismatchError):
            infer(A.raise_(A.int_(1)), max_nodes=1000, version=2)


class PackageStructureTests(unittest.TestCase):
    def _good(self):
        return package({
            "M": module([], [
                export("f", signature([], tfun(tcon("Int"), tcon("Int"))),
                       A.lambda_("x", A.var("x"))),
            ]),
        })

    def test_package_must_have_modules_key(self):
        with self.assertRaises(InvalidAstError):
            ModuleChecker({"nope": {}})

    def test_empty_modules_rejected(self):
        with self.assertRaises(InvalidAstError):
            ModuleChecker({"modules": {}})

    def test_import_missing_rejected(self):
        bad = package({
            "M": module(["Ghost"], [
                export("f", signature([], tfun(tcon("Int"), tcon("Int"))),
                       A.lambda_("x", A.var("x"))),
            ]),
        })
        with self.assertRaises(ModuleStructureError):
            check_package(bad)

    def test_mvar_to_not_imported_rejected(self):
        bad = package({
            "M": module(["N"], [
                export("f", signature([], tfun(tcon("Int"), tcon("Int"))),
                       A.lambda_("x", A.var("x"))),
            ]),
            "N": module([], [
                export("g", signature([], tfun(tcon("Int"), tcon("Int"))),
                       A.lambda_("y", A.var("y"))),
            ]),
            "P": module([], [
                export("h", signature([], tfun(tcon("Int"), tcon("Int"))),
                       A.lambda_("z", A.apply(A.mvar("N", "g"), A.var("z")))),
            ]),
        })
        with self.assertRaises(InvalidAstError):
            check_package(bad)

    def test_signature_unknown_type_con_rejected(self):
        bad = package({
            "M": module([], [
                export("f", signature([], tfun(tcon("String"), tcon("Int"))),
                       A.lambda_("x", A.var("x"))),
            ]),
        })
        with self.assertRaises(ModuleStructureError):
            check_package(bad)

    def test_signature_unbound_type_var_rejected(self):
        bad = package({
            "M": module([], [
                export("f", signature([], tfun(tvar("a"), tvar("a"))),
                       A.lambda_("x", A.var("x"))),
            ]),
        })
        with self.assertRaises(ModuleStructureError):
            check_package(bad)

    def test_signature_unbound_row_var_rejected(self):
        bad = package({
            "M": module([], [
                export("f",
                       signature([], trecord([("x", tcon("Int"))], rvar("r"))),
                       A.record([("x", A.int_(1))])),
            ]),
        })
        with self.assertRaises(ModuleStructureError):
            check_package(bad)

    def test_signature_bad_effect_label_rejected(self):
        bad = package({
            "M": module([], [
                export("f",
                       signature([], tfun(tcon("Int"), tcon("Int"), ["launch"])),
                       A.lambda_("x", A.var("x"))),
            ]),
        })
        with self.assertRaises(ModuleStructureError):
            check_package(bad)

    def test_row_var_only_allowed_at_record_tail(self):
        # rvar 出现在字段类型位置（而非 trecord.tail）必须被结构拒绝。
        bad_sig = {
            "vars": [], "row_vars": ["r"],
            "type": trecord([("self", rvar("r"))], rvar("r")),
        }
        bad = package({
            "M": module([], [
                export("loop", bad_sig, A.record([("self", A.int_(1))])),
            ]),
        })
        with self.assertRaises(ModuleStructureError):
            check_package(bad)


class V2TamperDetectionTests(unittest.TestCase):
    def _read_result(self):
        return infer(
            A.lambda_("rr", A.read(A.var("rr"))),
            max_nodes=1000, version=2,
        )

    def test_effect_solution_tamper_caught(self):
        r = self._read_result()
        fake = {k: frozenset() for k in r.eff_solution}
        with self.assertRaises(ReferenceCheckFailure):
            check_result(replace(r, eff_solution=fake))

    def test_effect_log_tamper_caught(self):
        r = self._read_result()
        # 删掉全部 union 事件：重放解为空，与声称的 read 不一致。
        with self.assertRaises(ReferenceCheckFailure):
            check_result(replace(r, effect_log=()))

    def test_field_tags_tamper_caught(self):
        r = infer(
            A.let("box", A.record([("c", A.int_(1))]), A.var("box")),
            max_nodes=1000, version=2,
        )
        # 全值字段记录本应泛化；把 field_tags 篡改为空触发结构一致性失败。
        g = replace(r.generalizations[0], field_tags=())
        with self.assertRaises(ReferenceCheckFailure):
            check_result(replace(r, generalizations=(g,)))

    def test_over_generalize_row_var_caught(self):
        r = infer(
            A.let("box", A.record([("c", A.int_(1))]), A.var("box")),
            max_nodes=1000, version=2,
        )
        from polyinfer.types import TVar
        g0 = r.generalizations[0]
        g = replace(g0, generalized=g0.generalized + (TVar(999_001, 1),))
        with self.assertRaises(ReferenceCheckFailure):
            check_result(replace(r, generalizations=(g,)))


class ExhaustiveSmallReferenceTests(unittest.TestCase):
    """小规模效应格的**完整**枚举核对。

    独立参考域：单个 ``Ref(a) -> ...`` lambda，其函数体对同一引用做
    或不做 read、做或不做 write——``{read 否/是} × {write 否/是}``
    共 4 个元素，是该效应格的全部元素（非抽样）。期望主类型/效应由
    本测试依据效应语义独立合成（read 改返回类型为 a，write 产出 Unit；
    read+write 顺序下结果类型取最后 write 的 Unit），不调用被测求解器。
    """

    def _program(self, do_read: bool, do_write: bool):
        if do_read and do_write:
            # 先读后写：返回 Unit；效应 {read, write}
            return A.lambda_(
                "r",
                A.write_(A.var("r"), A.read(A.var("r"))),
            )
        if do_read:
            return A.lambda_("r", A.read(A.var("r")))
        if do_write:
            return A.lambda_("r", A.write_(A.var("r"), A.int_(1)))
        return A.lambda_("r", A.var("r"))  # 退化：不触碰引用，类型 Ref(a)->Ref(a)

    def _expected(self, do_read: bool, do_write: bool):
        eff = set()
        if do_read:
            eff.add("read")
        if do_write:
            eff.add("write")
        if do_read and do_write:
            arrow = "Ref(a) -{read, write}-> Unit"
        elif do_read:
            arrow = "Ref(a) -{read}-> a"
        elif do_write:
            arrow = "Ref(Int) -{write}-> Unit"
        else:
            arrow = "a -> a"  # 不触碰参数：退化为恒等（类型不被钉成 Ref）
        return arrow, frozenset(eff)

    def test_full_effect_lattice_enumerated(self):
        cases = 0
        for do_read in (False, True):
            for do_write in (False, True):  # 完整 2×2，非抽样
                cases += 1
                want_type, want_eff = self._expected(do_read, do_write)
                r = infer(self._program(do_read, do_write),
                          max_nodes=2000, version=2)
                check_result(r)
                self.assertEqual(r.principal_type, want_type)
                arrow_eff: set = set()
                if "-{" in r.principal_type:
                    inside = r.principal_type.split("-{")[1].split("}")[0]
                    arrow_eff = {x.strip() for x in inside.split(",") if x.strip()}
                self.assertEqual(arrow_eff, set(want_eff))
        self.assertEqual(cases, 4)  # 完整格的元素数

    def test_repeated_runs_identical(self):
        # 确定性：完整格中每个程序多次推断，主类型/效应/约束数完全一致。
        for do_read in (False, True):
            for do_write in (False, True):
                node = self._program(do_read, do_write)
                r1 = infer(node, max_nodes=2000, version=2)
                r2 = infer(node, max_nodes=2000, version=2)
                self.assertEqual(r1.principal_type, r2.principal_type)
                self.assertEqual(r1.effects, r2.effects)
                self.assertEqual(len(r1.constraints), len(r2.constraints))
                self.assertEqual(
                    tuple(sorted(r1.eff_solution.items())),
                    tuple(sorted(r2.eff_solution.items())),
                )


if __name__ == "__main__":
    unittest.main()
