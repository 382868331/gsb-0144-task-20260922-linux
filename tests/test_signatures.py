"""rank-1 类型签名（let 绑定处的显式量化）测试。

对照三组场景：

* 恒等函数：带签名后在同一作用域内按两种不同类型调用；
* 捕获外部单态变量的闭包：具体签名合法，多态签名导致刚性变量逃逸；
* 含引用的反例：签名不得把单态引用/扩展式包装成可任意实例化的多态值。

错误分类：签名不匹配 (SignatureMismatchError)、逃逸的刚性变量
(RigidEscapeError)、值限制冲突 (ValueRestrictionConflictError)，
且都带 let 表达式位置。独立参考检查器（不借用待测统一器）对
签名事件做替换级复核与篡改检测。
"""

import unittest
from dataclasses import replace

from polyinfer import A, infer
from polyinfer.errors import (
    InvalidAstError,
    RigidEscapeError,
    SignatureMismatchError,
    ValueRestrictionConflictError,
)
from polyinfer.reference import ReferenceCheckFailure
from polyinfer.types import INT

ID_SIG = A.sig(["a"], A.tfun(A.tvar("a"), A.tvar("a")))
INT2INT = A.sig([], A.tfun(A.tcon("Int"), A.tcon("Int")))


def signed_identity_used_at_two_types():
    """let id : forall a. a -> a = λx.x in id 分别按 Bool、Int 调用。"""
    return A.let(
        "id",
        A.lambda_("x", A.var("x")),
        A.if_(
            A.apply(A.var("id"), A.bool_(True)),
            A.apply(A.var("id"), A.int_(1)),
            A.int_(2),
        ),
        sig=ID_SIG,
    )


class TestSignedIdentity(unittest.TestCase):
    def test_identity_two_instances(self):
        result = infer(signed_identity_used_at_two_types())
        self.assertEqual(result.principal_type, "Int")
        report = result.verify()  # 独立替换器复核，含签名事件
        self.assertEqual(report.generalizations_checked, 1)

    def test_signature_event_recorded(self):
        result = infer(signed_identity_used_at_two_types())
        (gen,) = result.generalizations
        self.assertEqual(gen.sig_quantified, ("a",))
        self.assertIsNotNone(gen.sig_type)
        self.assertEqual(len(gen.generalized), 1)

    def test_unsigned_inference_unchanged(self):
        # 没有签名时保持原推断：恒等函数主类型仍是 a -> a。
        result = infer(A.lambda_("x", A.var("x")))
        self.assertEqual(result.principal_type, "a -> a")

    def test_signature_may_be_more_specific(self):
        # 签名 Int -> Int 比推断的 a -> a 更具体：合法，且结果单态化。
        prog = A.let(
            "f",
            A.lambda_("x", A.var("x")),
            A.apply(A.var("f"), A.int_(1)),
            sig=INT2INT,
        )
        self.assertEqual(infer(prog).principal_type, "Int")

    def test_adjacent_scopes_do_not_leak(self):
        # 相邻两个 let 各自量化 'a'：作用域互不相干，都合法。
        prog = A.let(
            "f",
            A.lambda_("x", A.var("x")),
            A.let(
                "g",
                A.lambda_("y", A.apply(A.var("f"), A.var("y"))),
                A.apply(A.var("g"), A.int_(1)),
                sig=ID_SIG,
            ),
            sig=ID_SIG,
        )
        self.assertEqual(infer(prog).principal_type, "Int")


class TestSignatureMismatch(unittest.TestCase):
    def test_signature_more_general_than_body(self):
        # forall a b. a -> b 声称得比恒等函数更一般：刚性 b 被迫等于 a。
        prog = A.let(
            "f",
            A.lambda_("x", A.var("x")),
            A.var("f"),
            sig=A.sig(["a", "b"], A.tfun(A.tvar("a"), A.tvar("b"))),
        )
        with self.assertRaises(SignatureMismatchError) as ctx:
            infer(prog)
        self.assertEqual(ctx.exception.path, ())  # let 在根位置

    def test_concrete_signature_mismatch(self):
        # λx. true 的类型是 a -> Bool，与 Int -> Int 对不上。
        prog = A.let(
            "f",
            A.lambda_("x", A.bool_(True)),
            A.var("f"),
            sig=INT2INT,
        )
        with self.assertRaises(SignatureMismatchError) as ctx:
            infer(prog)
        exc = ctx.exception
        self.assertEqual(exc.name, "f")
        # 错误带失败约束两端位置。
        self.assertTrue(exc.left_path)
        self.assertTrue(exc.right_path)

    def test_body_forces_rigid_var_to_concrete(self):
        # λx. 1 把签名里的刚性 a 逼成 Int。
        prog = A.let("f", A.lambda_("x", A.int_(1)), A.var("f"), sig=ID_SIG)
        with self.assertRaises(SignatureMismatchError):
            infer(prog)

    def test_mismatch_inside_nested_let_path(self):
        prog = A.let(
            "outer",
            A.lambda_("x", A.var("x")),
            A.let("f", A.lambda_("x", A.int_(1)), A.var("f"), sig=ID_SIG),
        )
        with self.assertRaises(SignatureMismatchError) as ctx:
            infer(prog)
        self.assertEqual(ctx.exception.path, ("body",))


class TestRigidEscape(unittest.TestCase):
    def test_closure_capturing_outer_monomorphic_var(self):
        # λy. let f : forall a. a -> a = λz. y in 0
        # f 的类型实际是 ?z -> ?y；把 ?y 绑到刚性 a 会让 a 逃逸进外层
        # lambda 参数的类型——签名作用域不得穿过外层绑定。
        prog = A.lambda_(
            "y",
            A.let("f", A.lambda_("z", A.var("y")), A.int_(0), sig=ID_SIG),
        )
        with self.assertRaises(RigidEscapeError) as ctx:
            infer(prog)
        exc = ctx.exception
        self.assertEqual(exc.variable, "a")
        self.assertEqual(exc.path, ("body",))

    def test_capture_with_concrete_signature_is_legal(self):
        # 对照组：同一闭包配具体签名 Int -> Int 合法（无刚性变量），
        # 且把外层单态参数细化为 Int。
        prog = A.lambda_(
            "y",
            A.let(
                "f",
                A.lambda_("w", A.var("y")),
                A.apply(A.var("f"), A.int_(1)),
                sig=INT2INT,
            ),
        )
        self.assertEqual(infer(prog).principal_type, "Int -> Int")

    def test_polymorphic_signature_normal_use(self):
        # 对照组：const 配 forall a b. a -> b -> a 正常使用（不捕获）。
        prog = A.lambda_(
            "y",
            A.let(
                "const",
                A.lambda_("x", A.lambda_("z", A.var("x"))),
                A.apply(A.apply(A.var("const"), A.var("y")), A.int_(1)),
                sig=A.sig(
                    ["a", "b"],
                    A.tfun(A.tvar("a"), A.tfun(A.tvar("b"), A.tvar("a"))),
                ),
            ),
        )
        self.assertEqual(infer(prog).principal_type, "a -> a")


class TestValueRestrictionConflict(unittest.TestCase):
    def test_expansive_bound_with_polymorphic_signature(self):
        # (λx.x)(λx.x) 是扩展式：签名要求泛化即冲突。
        prog = A.let(
            "f",
            A.apply(A.lambda_("x", A.var("x")), A.lambda_("x", A.var("x"))),
            A.var("f"),
            sig=ID_SIG,
        )
        with self.assertRaises(ValueRestrictionConflictError) as ctx:
            infer(prog)
        exc = ctx.exception
        self.assertEqual(exc.bound_tag, "apply")
        self.assertEqual(exc.quantified, ("a",))
        self.assertEqual(exc.path, ())

    def test_ref_cannot_be_wrapped_as_polymorphic(self):
        # 把单态引用包装成可任意实例化的 forall a. Ref(a)：拒绝。
        prog = A.let(
            "r",
            A.ref(A.int_(0)),
            A.var("r"),
            sig=A.sig(["a"], A.tref(A.tvar("a"))),
        )
        with self.assertRaises(ValueRestrictionConflictError):
            infer(prog)

    def test_ref_hole_stays_closed_with_signature(self):
        # 引用多态漏洞的签名版：let r : forall a. Ref(a->a) = ref(λz.z)。
        prog = A.let(
            "r",
            A.ref(A.lambda_("z", A.var("z"))),
            A.let("_u", A.write_(A.var("r"), A.int_(1)), A.read(A.var("r"))),
            sig=A.sig(["a"], A.tref(A.tfun(A.tvar("a"), A.tvar("a")))),
        )
        with self.assertRaises(ValueRestrictionConflictError):
            infer(prog)

    def test_expansive_bound_with_monomorphic_signature_ok(self):
        # 对照组：扩展式 + 无变量签名不触发值限制冲突。
        prog = A.let(
            "f",
            A.apply(A.lambda_("x", A.var("x")), A.lambda_("x", A.var("x"))),
            A.apply(A.var("f"), A.int_(1)),
            sig=INT2INT,
        )
        self.assertEqual(infer(prog).principal_type, "Int")


class TestSignatureValidation(unittest.TestCase):
    def test_unquantified_variable_rejected(self):
        prog = A.let(
            "f",
            A.lambda_("x", A.var("x")),
            A.var("f"),
            sig=A.sig(["a"], A.tfun(A.tvar("b"), A.tvar("a"))),
        )
        with self.assertRaises(InvalidAstError) as ctx:
            infer(prog)
        self.assertEqual(ctx.exception.path[:2], ("sig", "type"))

    def test_duplicate_quantified_names_rejected(self):
        prog = A.let(
            "f",
            A.lambda_("x", A.var("x")),
            A.var("f"),
            sig=A.sig(["a", "a"], A.tvar("a")),
        )
        with self.assertRaises(InvalidAstError):
            infer(prog)

    def test_unknown_base_type_rejected(self):
        prog = A.let(
            "x",
            A.int_(1),
            A.var("x"),
            sig=A.sig([], A.tcon("Float")),
        )
        with self.assertRaises(InvalidAstError):
            infer(prog)

    def test_malformed_signature_rejected(self):
        prog = A.let(
            "f",
            A.lambda_("x", A.var("x")),
            A.var("f"),
            sig={"forall": ["a"], "kind": "sig"},
        )
        with self.assertRaises(InvalidAstError):
            infer(prog)


class TestIndependentSignatureCheck(unittest.TestCase):
    """独立参考检查器对签名事件的复核与篡改检测。"""

    def test_verify_passes_on_signed_program(self):
        report = infer(signed_identity_used_at_two_types()).verify()
        self.assertGreater(report.constraints_checked, 0)

    def test_tampered_sig_type_detected(self):
        result = infer(signed_identity_used_at_two_types())
        gen = result.generalizations[0]
        forged = replace(gen, sig_type=INT)
        hacked = replace(result, generalizations=(forged,))
        with self.assertRaises(ReferenceCheckFailure):
            hacked.verify()

    def test_tampered_generalized_set_detected(self):
        result = infer(signed_identity_used_at_two_types())
        gen = result.generalizations[0]
        forged = replace(gen, generalized=())
        hacked = replace(result, generalizations=(forged,))
        with self.assertRaises(ReferenceCheckFailure):
            hacked.verify()


if __name__ == "__main__":
    unittest.main()

