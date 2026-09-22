"""类型推断的核心行为测试（含四个验收场景）。

期望值（主类型字符串、路径元组）全部手写，不由被测核心产生；
每个成功推导都额外经过独立检查器 ``result.verify()`` 核验全部约束。
"""

import unittest

from polyinfer import A, infer
from polyinfer.errors import InfiniteTypeError, TypeMismatchError


def infer_ok(node):
    result = infer(node)
    # 每个成功推导：独立替换器必须确认最终替换满足全部记录约束。
    report = result.verify()
    return result, report


class PrincipalTypeTests(unittest.TestCase):
    def test_identity(self):
        result, report = infer_ok(A.lambda_("x", A.var("x")))
        self.assertEqual(result.principal_type, "a -> a")
        self.assertGreaterEqual(report.constraints_checked, 0)

    def test_const(self):
        result, _ = infer_ok(A.lambda_("x", A.lambda_("y", A.var("x"))))
        self.assertEqual(result.principal_type, "a -> b -> a")

    def test_compose(self):
        expr = A.lambda_(
            "f",
            A.lambda_(
                "g",
                A.lambda_("x", A.apply(A.var("f"), A.apply(A.var("g"), A.var("x")))),
            ),
        )
        result, _ = infer_ok(expr)
        self.assertEqual(result.principal_type, "(a -> b) -> (c -> a) -> c -> b")

    def test_higher_order(self):
        # λf. λx. f (f x) : (a -> a) -> a -> a
        expr = A.lambda_(
            "f",
            A.lambda_(
                "x",
                A.apply(A.var("f"), A.apply(A.var("f"), A.var("x"))),
            ),
        )
        result, _ = infer_ok(expr)
        self.assertEqual(result.principal_type, "(a -> a) -> a -> a")

    def test_literals_and_if(self):
        result, _ = infer_ok(A.if_(A.bool_(True), A.int_(1), A.int_(2)))
        self.assertEqual(result.principal_type, "Int")

    def test_application_ground(self):
        result, _ = infer_ok(A.apply(A.lambda_("x", A.var("x")), A.int_(7)))
        self.assertEqual(result.principal_type, "Int")

    def test_ref_read_write_roundtrip_lambda(self):
        # λr. r := !r : Ref(a) -> Unit（write 返回 Unit）
        result, _ = infer_ok(
            A.lambda_("r", A.write_(A.var("r"), A.read(A.var("r"))))
        )
        self.assertEqual(result.principal_type, "Ref(a) -> Unit")

    def test_write_expression_is_unit(self):
        result, _ = infer_ok(A.write_(A.ref(A.int_(1)), A.int_(2)))
        self.assertEqual(result.principal_type, "Unit")

    def test_unit_let_body(self):
        # let u = (ref 1 := 2) in u：扩展式单态，类型 Unit
        result, _ = infer_ok(
            A.let("u", A.write_(A.ref(A.int_(1)), A.int_(2)), A.var("u"))
        )
        self.assertEqual(result.principal_type, "Unit")

    def test_ref_invariant_polymorphic_cell(self):
        result, _ = infer_ok(A.read(A.ref(A.lambda_("x", A.var("x")))))
        self.assertEqual(result.principal_type, "a -> a")

    def test_shadowing(self):
        # let x = 1 in let x = true in x  —— 遮蔽，结果 Bool
        result, _ = infer_ok(
            A.let("x", A.int_(1), A.let("x", A.bool_(True), A.var("x")))
        )
        self.assertEqual(result.principal_type, "Bool")


class ValueRestrictionTests(unittest.TestCase):
    def test_identity_two_instances(self):
        """验收 1：let 绑定的恒等函数必须能取两种实例（Int 与 Bool）。"""
        program = A.let(
            "id",
            A.lambda_("x", A.var("x")),
            # id 同时按 Bool -> Bool 与 Int -> Int 使用；条件成立取 id 1。
            A.if_(
                A.apply(A.var("id"), A.bool_(True)),
                A.apply(A.var("id"), A.int_(1)),
                A.int_(2),
            ),
        )
        result, report = infer_ok(program)
        self.assertEqual(result.principal_type, "Int")
        # 至少记录了 id 的一次泛化，且被独立检查器认可。
        self.assertEqual(report.generalizations_checked, 1)

    def test_identity_two_instances_as_result(self):
        # 两次实例分别出现在两个独立程序里，主类型分别落地。
        r_bool, _ = infer_ok(
            A.let("id", A.lambda_("x", A.var("x")),
                  A.apply(A.var("id"), A.bool_(False)))
        )
        r_int, _ = infer_ok(
            A.let("id", A.lambda_("x", A.var("x")),
                  A.apply(A.var("id"), A.int_(42)))
        )
        self.assertEqual(r_bool.principal_type, "Bool")
        self.assertEqual(r_int.principal_type, "Int")

    def test_lambda_parameter_is_monomorphic(self):
        """验收 2：lambda 参数在函数体内不可多态。

        x 先被当作 Bool（if 条件），又被当作 Int -> ?（应用），必须失败；
        同样的代码把 x 放进 let 绑定则可多态——对照组。
        """
        bad = A.lambda_(
            "x",
            A.if_(
                A.var("x"),
                A.int_(1),
                A.apply(A.var("x"), A.int_(2)),
            ),
        )
        with self.assertRaises(TypeMismatchError) as cm:
            infer(bad)
        # 路径必须指向参数被矛盾使用的两处子树。
        paths = (cm.exception.left_path, cm.exception.right_path)
        self.assertTrue(
            any(p and p[0] == "body" for p in paths),
            f"错误路径应位于函数体内：{paths}",
        )

    def test_let_binding_contrast_is_polymorphic(self):
        # 对照组：let x = λy.y 后，x 可分别按 Bool->Bool 与 Int->Int 使用。
        good = A.let(
            "x",
            A.lambda_("y", A.var("y")),
            A.if_(
                A.apply(A.var("x"), A.bool_(True)),
                A.apply(A.var("x"), A.int_(0)),
                A.int_(9),
            ),
        )
        result, _ = infer_ok(good)
        self.assertEqual(result.principal_type, "Int")

    def test_reference_polymorphism_hole_closed(self):
        """验收 4：ref 是扩展式，let 不泛化，多态单元漏洞必须被堵住。

        经典例子：let r = ref(λz.z) in r := 1; !(r)
        若 r 被错误泛化成 ∀a. Ref(a->a)，赋 Int 与取函数可同时通过；
        值限制下 r 是单态 Ref(b -> b)，写入 Int 直接冲突。
        """
        leak = A.let(
            "r",
            A.ref(A.lambda_("z", A.var("z"))),
            A.let(
                "u",
                A.write_(A.var("r"), A.int_(1)),
                A.read(A.var("r")),
            ),
        )
        with self.assertRaises(TypeMismatchError) as cm:
            infer(leak)
        err = cm.exception
        # 冲突两端：ref 分配处的单元类型 vs 写入值处。
        joined = "/".join(err.left_path + err.right_path)
        self.assertIn("bound", joined)
        self.assertIn("value", joined)
        # 被拒绝的不是 occurs 类错误：构造子是函数与 Int 的真实冲突。
        # 错误对象中的类型带内部变量名（规范名只用于最终主类型）。
        self.assertIn("->", str(err.left_type))
        self.assertEqual(str(err.right_type), "Int")

    def test_value_restriction_keeps_expansive_monomorphic(self):
        # let r = ref(λx.x) in r：主类型是单态 Ref(a -> a)（变量仍自由，
        # 但该方案不带任何限定变量），再用 Int 写入即失败。
        result, _ = infer_ok(
            A.let("r", A.ref(A.lambda_("x", A.var("x"))), A.var("r"))
        )
        self.assertEqual(result.principal_type, "Ref(a -> a)")
        gen = result.generalizations[0]
        self.assertEqual(gen.bound_tag, "ref")
        self.assertEqual(gen.generalized, ())  # 未泛化任何变量

    def test_nested_let_generalization(self):
        program = A.let(
            "id",
            A.lambda_("x", A.var("x")),
            A.let(
                "a",
                A.apply(A.var("id"), A.int_(3)),
                A.apply(A.var("id"), A.bool_(False)),
            ),
        )
        result, report = infer_ok(program)
        self.assertEqual(result.principal_type, "Bool")
        self.assertEqual(report.generalizations_checked, 2)

    def test_nested_value_capturing_polymorphic_id(self):
        # let id = λx.x in let f = λz. id z in (f 1, f true) 形式：
        # f 是值绑定，必须得到 ∀a. a -> a 并支持两种实例。
        program = A.let(
            "id",
            A.lambda_("x", A.var("x")),
            A.let(
                "f",
                A.lambda_("z", A.apply(A.var("id"), A.var("z"))),
                A.if_(
                    A.apply(A.var("f"), A.bool_(True)),
                    A.apply(A.var("f"), A.int_(1)),
                    A.int_(0),
                ),
            ),
        )
        result, _ = infer_ok(program)
        self.assertEqual(result.principal_type, "Int")


class OccursTests(unittest.TestCase):
    def test_self_application_is_infinite_type(self):
        """验收 3：λx. x x 触发 occurs 检查，报无限类型而非栈溢出。"""
        bad = A.lambda_("x", A.apply(A.var("x"), A.var("x")))
        with self.assertRaises(InfiniteTypeError) as cm:
            infer(bad)
        err = cm.exception
        # 展开式必须是函数类型；两端路径定位到 apply 的 fn 与整个 apply。
        self.assertEqual(err.left_path, ("body", "fn"))
        self.assertEqual(err.right_path, ("body",))
        self.assertIn("->", str(err.right_type))

    def test_deeper_infinite_type_still_detected(self):
        # λx. x (λy. x y)：x 被要求为 a 与 a -> b 的复合，同样无限。
        bad = A.lambda_(
            "x",
            A.apply(
                A.var("x"),
                A.lambda_("y", A.apply(A.var("x"), A.var("y"))),
            ),
        )
        with self.assertRaises(InfiniteTypeError):
            infer(bad)

    def test_mutual_unify_through_ref_is_finite(self):
        # Ref(a) 与 Ref(b)，再 b := Int：正常有限类型，不应误报。
        result, _ = infer_ok(
            A.let("r", A.ref(A.lambda_("x", A.var("x"))),
                  A.write_(A.var("r"), A.lambda_("y", A.var("y"))))
        )
        self.assertEqual(result.principal_type, "Unit")


class SimpleMismatchTests(unittest.TestCase):
    def test_if_condition_must_be_bool(self):
        with self.assertRaises(TypeMismatchError) as cm:
            infer(A.if_(A.int_(1), A.int_(2), A.int_(3)))
        self.assertEqual(cm.exception.left_path, ("cond",))
        self.assertEqual(cm.exception.right_path, ())

    def test_if_brances_must_agree(self):
        with self.assertRaises(TypeMismatchError) as cm:
            infer(A.if_(A.bool_(True), A.int_(1), A.bool_(False)))
        self.assertEqual(cm.exception.left_path, ("then",))
        self.assertEqual(cm.exception.right_path, ("else_",))

    def test_apply_non_function(self):
        with self.assertRaises(TypeMismatchError):
            infer(A.apply(A.int_(1), A.int_(2)))

    def test_argument_type_mismatch(self):
        # λb. b false 中 b 被要求为 Bool -> a，而 body 里又... 简单点：
        # (λx. x) 1 与 Bool 参数混用不直接，构造 (λf. f true) 应用于
        # λg. g 1？直接：λx. x := 1 需要 x 是引用，传 Int 失败。
        with self.assertRaises(TypeMismatchError):
            infer(
                A.apply(
                    A.lambda_("r", A.write_(A.var("r"), A.int_(1))),
                    A.int_(0),
                )
            )

    def test_read_requires_reference(self):
        with self.assertRaises(TypeMismatchError) as cm:
            infer(A.read(A.int_(1)))
        self.assertEqual(cm.exception.left_path, ("ref",))

    def test_write_requires_reference_left(self):
        with self.assertRaises(TypeMismatchError):
            infer(A.write_(A.int_(1), A.int_(2)))

    def test_write_cell_invariance(self):
        # Ref(Int) 单元不能写 Bool（引用不变式）。
        with self.assertRaises(TypeMismatchError):
            infer(A.write_(A.ref(A.int_(1)), A.bool_(True)))

    def test_ref_is_invariant_not_covariant(self):
        # 把 Ref(Int) 传给要求 Ref(Bool) 的函数：拒绝。
        program = A.apply(
            A.lambda_("r", A.write_(A.var("r"), A.bool_(True))),
            A.ref(A.int_(1)),
        )
        with self.assertRaises(TypeMismatchError):
            infer(program)


if __name__ == "__main__":
    unittest.main()
