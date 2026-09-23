"""v2 补充边界：语言版本隔离、v1 漏洞在 v2 仍堵、大记录、确定性、
以及未决效应不能经显式模块签名洗白。
"""

import unittest

from polyinfer import A, infer
from polyinfer.errors import (
    EffectMismatchError,
    InvalidAstError,
    TypeMismatchError,
)


class LanguageIsolationTests(unittest.TestCase):
    def test_v2_tags_rejected_in_v1(self):
        for node in (
            A.record([("a", A.int_(1))]),
            A.get(A.var("x"), "a"),
            A.raise_(A.int_(1)),
            A.evar(),
        ):
            with self.subTest(node=node["tag"]):
                with self.assertRaises(InvalidAstError):
                    infer(node)  # 默认 v1

    def test_bad_language_value_rejected(self):
        with self.assertRaises(ValueError):
            infer(A.int_(1), language="v3")

    def test_v1_rendering_unchanged_under_v2_engine_paths(self):
        # 同一程序 v1 与 v2 的主类型字符串必须一致（纯部分渲染兼容）。
        node = A.lambda_("x", A.var("x"))
        self.assertEqual(
            infer(node).principal_type,
            infer(node, language="v2").principal_type,
        )


class ValueRestrictionV2Tests(unittest.TestCase):
    def test_classic_ref_hole_still_closed_in_v2(self):
        # let r = ref(λz.z) in r := 1; !r 在 v2 仍必须失败。
        leak = A.let(
            "r", A.ref(A.lambda_("z", A.var("z"))),
            A.let("_u", A.write_(A.var("r"), A.int_(1)), A.read(A.var("r"))),
        )
        with self.assertRaises(TypeMismatchError):
            infer(leak, language="v2")

    def test_expansive_record_field_not_generalized(self):
        # record 内含 ref 字段 → 整个 record 是扩展式，不泛化其中单元变量。
        node = A.let(
            "r", A.record([("c", A.ref(A.lambda_("x", A.var("x"))))]),
            A.let("_u", A.write_(A.get(A.var("r"), "c"), A.int_(1)),
                  A.read(A.get(A.var("r"), "c"))),
        )
        with self.assertRaises(TypeMismatchError):
            infer(node, language="v2")

    def test_pure_record_value_is_generalized(self):
        # 全值 record 绑定可泛化，字段访问能取两种实例（两个独立程序）。
        def use_int():
            return infer(A.let("p", A.record([("a", A.lambda_("x", A.var("x")))]),
                               A.apply(A.get(A.var("p"), "a"), A.int_(1))),
                         language="v2")

        def use_bool():
            return infer(A.let("p", A.record([("a", A.lambda_("x", A.var("x")))]),
                               A.apply(A.get(A.var("p"), "a"), A.bool_(False))),
                         language="v2")

        r1 = use_int()
        r2 = use_bool()
        r1.verify()
        r2.verify()
        self.assertEqual((r1.principal_type, r2.principal_type), ("Int", "Bool"))


class LargeRecordTests(unittest.TestCase):
    def test_twenty_field_record_roundtrip(self):
        fields = [(f"f{i}", A.int_(i)) for i in range(20)]
        node = A.lambda_(
            "r",
            A.if_(A.bool_(True),
                  A.get(A.var("r"), "f0"),
                  A.get(A.var("r"), "f19")),
        )
        r = infer(node, language="v2")
        r.verify()
        self.assertIn("f0: a", r.principal_type)
        self.assertIn("f19: a", r.principal_type)

    def test_twenty_field_literal_closed(self):
        fields = [(f"f{i}", A.int_(i)) for i in range(20)]
        r = infer(A.record(fields), language="v2")
        r.verify()
        self.assertEqual(r.principal_type.count(":"), 20)
        self.assertNotIn(";", r.principal_type)  # 字面量闭合


class DeterminismTests(unittest.TestCase):
    def test_same_input_same_output(self):
        import json

        node = A.let(
            "f", A.lambda_("x", A.get(A.var("x"), "k")),
            A.apply(A.var("f"), A.record([("k", A.int_(1)), ("m", A.bool_(True))])),
        )
        outs = []
        for _ in range(5):
            r = infer(node, language="v2")
            outs.append((r.principal_type,
                         tuple(sorted(str(x) for x in r.effects)),
                         r.node_count,
                         tuple((c.note, str(c.left), str(c.right))
                               for c in r.constraints)))
        self.assertEqual(len(set(outs)), 1)
        # 主类型字符串是确定性 JSON 可序列化数据的一部分。
        json.dumps(outs[0][0])


class ModuleEvarRejectionTests(unittest.TestCase):
    def test_export_with_evar_rejected(self):
        from polyinfer.modules import check_modules

        # 导出体含未决效应占位，却给了完全显式的纯签名：拒绝，不洗白。
        M = A.make_module("M", {"f": "Int -> Int"},
                          {"f": A.lambda_("x", A.evar())})
        with self.assertRaises(EffectMismatchError):
            check_modules({"M": M})

    def test_fresh_cell_external_function_is_sound(self):
        # 外部模块每次调用新建单元：a -> Ref(a) 健全多态，调用方写入 Int
        # 不影响多态资格（单元是调用点新建，不是跨模块共享）。
        from polyinfer.modules import check_modules
        from polyinfer.reference import check_module_report

        F = A.make_module("F", {"mk": "forall a. a -> Ref(a)"},
                          {"mk": A.lambda_("x", A.ref(A.var("x")))})
        U = A.make_module("U", {"go": "Int -{W}-> Unit"},
            {"go": A.lambda_("x",
                A.let("c", A.apply(A.xvar("F", "mk"), A.var("x")),
                      A.write_(A.var("c"), A.int_(2))))},
            imports=("F",))
        _, report = check_modules({"F": F, "U": U})
        check_module_report(report)


if __name__ == "__main__":
    unittest.main()
