"""rank-1 显式类型签名的行为测试。

覆盖：恒等函数签名 + 两次不同类型调用、签名可更具体、签名不匹配、
捕获外层单态变量的闭包（刚性变量逃逸）、含引用的值限制反例、
量化变量作用域不穿过相邻绑定、签名语法错误，以及一个**不借用
被测统一器**的小替换参照（``signature_oracle``）。
"""

import unittest

from polyinfer import (
    A,
    infer,
    parse_signature,
)
from polyinfer.errors import (
    EscapedRigidVariableError,
    InvalidAstError,
    SignatureMismatchError,
    ValueRestrictionConflictError,
)

from tests.signature_oracle import check_signatures


def twice_identity_let(sig=None):
    # let id = λx.x in (id true ? id 1 : 2)：两次不同类型的调用。
    return A.let(
        "id",
        A.lambda_("x", A.var("x")),
        A.if_(
            A.apply(A.var("id"), A.bool_(True)),
            A.apply(A.var("id"), A.int_(1)),
            A.int_(2),
        ),
        sig=sig,
    )


class SignedLetSuccessTests(unittest.TestCase):
    def test_signed_identity_two_instantiations(self):
        result = infer(twice_identity_let("forall a. a -> a"))
        self.assertEqual(result.principal_type, "Int")
        # 独立检查器：统一约束 + 签名方案各自成立。
        result.verify()
        value_signed, expansive_signed = check_signatures(result)
        self.assertEqual((value_signed, expansive_signed), (1, 0))

    def test_unsigned_inference_preserved(self):
        # 无签名：旧推断行为完全保留，同样的程序可类型化。
        self.assertEqual(infer(twice_identity_let()).principal_type, "Int")

    def test_signature_more_specific_than_inference(self):
        # 推断主类型是 a -> a；签名要求 Int -> Int（更具体），只按 Int 使用合法。
        node = A.let(
            "id",
            A.lambda_("x", A.var("x")),
            A.apply(A.var("id"), A.int_(3)),
            sig="Int -> Int",
        )
        result = infer(node)
        self.assertEqual(result.principal_type, "Int")
        result.verify()

    def test_more_general_implementation_accepted(self):
        # 实现 λx. x 是 a -> a；签名只要求 Int -> Int（逆向：签名更窄），
        # 且实现中自由推断变量可吸收签名形状——合法的“实现更一般”。
        node = A.let(
            "k",
            A.lambda_("x", A.var("x")),
            A.apply(A.var("k"), A.int_(0)),
            sig="Int -> Int",
        )
        infer(node).verify()

    def test_ref_signature_monomorphic_ok(self):
        # 扩展式 + 单态签名是允许的：let r: Ref(Int) = ref 1 in !r
        node = A.let(
            "r",
            A.ref(A.int_(1)),
            A.read(A.var("r")),
            sig="Ref(Int)",
        )
        result = infer(node)
        self.assertEqual(result.principal_type, "Int")
        result.verify()
        _, expansive_signed = check_signatures(result)
        self.assertEqual(expansive_signed, 1)

    def test_adjacent_bindings_have_independent_quantifiers(self):
        # 相邻两个绑定都用名字 a，但互不影响：id2 作用于 (id false)。
        node = A.let(
            "id",
            A.lambda_("x", A.var("x")),
            A.let(
                "id2",
                A.lambda_("y", A.var("y")),
                A.apply(A.var("id2"), A.apply(A.var("id"), A.bool_(False))),
                sig="forall b. b -> b",
            ),
            sig="forall a. a -> a",
        )
        result = infer(node)
        self.assertEqual(result.principal_type, "Bool")
        result.verify()

    def test_outer_polymorphic_arg_capture_is_fine(self):
        # 捕获外层 lambda 参数不必然逃逸：参数用法本就可随签名实例化。
        node = A.lambda_(
            "x",
            A.let(
                "id",
                A.lambda_("y", A.var("y")),
                A.apply(A.var("id"), A.var("x")),
                sig="forall a. a -> a",
            ),
        )
        self.assertEqual(infer(node).principal_type, "a -> a")


class SignatureMismatchTests(unittest.TestCase):
    def test_body_returns_wrong_type(self):
        node = A.let(
            "f",
            A.lambda_("x", A.bool_(True)),
            A.apply(A.var("f"), A.int_(1)),
            sig="Int -> Int",
        )
        with self.assertRaises(SignatureMismatchError) as cm:
            infer(node)
        exc = cm.exception
        self.assertEqual(exc.path, ())
        self.assertEqual(exc.name, "f")
        # 路径要深入到真正冲突的叶子（返回值），并指向绑定与 sig 两侧。
        self.assertIn("[ret]", exc.left_path)
        self.assertIn("sig", exc.right_path)

    def test_rigid_forced_concrete_inside_body(self):
        # λx. if x then x else x 要求 x = Bool；刚性 a 不能被悄悄统一成 Bool。
        node = A.let(
            "id",
            A.lambda_("x", A.if_(A.var("x"), A.var("x"), A.var("x"))),
            A.apply(A.var("id"), A.int_(1)),
            sig="forall a. a -> a",
        )
        with self.assertRaises(SignatureMismatchError):
            infer(node)

    def test_two_distinct_rigids_forced_equal(self):
        # 签名 forall a b. a -> b；实现 λx.x 迫使 a = b（两个不同刚性合并）。
        node = A.let(
            "f",
            A.lambda_("x", A.var("x")),
            A.apply(A.var("f"), A.int_(1)),
            sig="forall a b. a -> b",
        )
        with self.assertRaises(SignatureMismatchError):
            infer(node)


class EscapedRigidTests(unittest.TestCase):
    def test_closure_capturing_outer_monomorphic_var(self):
        # λx. let f : forall a. a -> a = λy. x in (f true; f 1)
        # 实现要求参数类型 == 外层单态 x：a 若逃逸，两次调用会污染 x。
        node = A.lambda_(
            "x",
            A.let(
                "f",
                A.lambda_("y", A.var("x")),
                A.if_(
                    A.apply(A.var("f"), A.bool_(True)),
                    A.apply(A.var("f"), A.int_(1)),
                    A.int_(0),
                ),
                sig="forall a. a -> a",
            ),
        )
        with self.assertRaises(EscapedRigidVariableError) as cm:
            infer(node)
        exc = cm.exception
        self.assertEqual(exc.name, "f")
        self.assertEqual(exc.bound_path, ("body", "bound"))
        self.assertEqual(str(exc.variable), exc.variable.name)


class ValueRestrictionConflictTests(unittest.TestCase):
    def test_polymorphic_sig_on_ref_binding(self):
        # let id = λz.z in let r : forall a. Ref(a) = ref id in r := 1; !r
        # ref id 中 id 实例化成裸柔性变量，与刚性 a 统一；若允许扩展式
        # 泛化，引用多态漏洞就被显式签名包装复活，必须由值限制拦截。
        node = A.let(
            "r",
            A.ref(A.var("id")),
            A.let(
                "u",
                A.write_(A.var("r"), A.int_(1)),
                A.read(A.var("r")),
            ),
            sig="forall a. Ref(a)",
        )
        node = A.let(
            "id",
            A.lambda_("z", A.var("z")),
            node,
        )
        with self.assertRaises(ValueRestrictionConflictError) as cm:
            infer(node)
        exc = cm.exception
        self.assertEqual(exc.bound_tag, "ref")
        self.assertEqual(exc.path[-1], "body")  # 外层无签名 let 的 body 内
        self.assertEqual(exc.bound_path, ("body", "bound"))

    def test_polymorphic_sig_on_apply_binding(self):
        # 扩展式（函数应用结果）同样不能靠签名包装成多态。
        node = A.let(
            "v",
            A.apply(A.lambda_("x", A.var("x")), A.lambda_("z", A.var("z"))),
            A.apply(A.var("v"), A.int_(1)),
            sig="forall a. a -> a",
        )
        with self.assertRaises((ValueRestrictionConflictError, SignatureMismatchError)):
            infer(node)


class SignatureSyntaxTests(unittest.TestCase):
    def test_parser_accepts_shapes(self):
        p = parse_signature("forall a b. a -> Ref(b) -> Int")
        self.assertEqual(p.quantified, ("a", "b"))

    def test_unquantified_var_rejected(self):
        with self.assertRaises(InvalidAstError):
            parse_signature("a -> Int")

    def test_rank2_rejected(self):
        with self.assertRaises(InvalidAstError):
            parse_signature("forall a. (forall b. b -> b) -> a")

    def test_duplicate_and_unknown_constructor(self):
        with self.assertRaises(InvalidAstError):
            parse_signature("forall a a. a -> a")
        with self.assertRaises(InvalidAstError):
            parse_signature("forall a. String -> a")

    def test_invalid_sig_field_rejected_at_let(self):
        node = A.let("x", A.int_(1), A.var("x"), sig=123)
        with self.assertRaises(InvalidAstError):
            infer(node)


class OracleIndependenceTests(unittest.TestCase):
    """参照检查不借用统一器，且能抓出针对签名记录的篡改。"""

    def _signed_result(self):
        return infer(twice_identity_let("forall a. a -> a"))

    def test_oracle_passes_on_genuine_result(self):
        value_signed, expansive_signed = check_signatures(self._signed_result())
        self.assertEqual(value_signed, 1)

    def test_oracle_catches_fabricated_generalization(self):
        import dataclasses

        from polyinfer import TVar

        result = self._signed_result()
        gens = list(result.generalizations)
        g = dataclasses.replace(gens[0], generalized=gens[0].generalized + (TVar(9999),))
        tampered = dataclasses.replace(result, generalizations=(g,))
        with self.assertRaises(AssertionError):
            check_signatures(tampered)

    def test_oracle_catches_value_restriction_tampering(self):
        import dataclasses

        from polyinfer import RigidTVar

        result = self._signed_result()
        gens = list(result.generalizations)
        # 把语法值绑定伪装成扩展式却仍保留刚性变量：值限制被绕过的形态。
        fake_rigid = RigidTVar(gens[0].rigid_vars[0].ident, "a")
        g = dataclasses.replace(gens[0], bound_tag="ref", rigid_vars=(fake_rigid,))
        tampered = dataclasses.replace(result, generalizations=(g,))
        with self.assertRaises(AssertionError):
            check_signatures(tampered)


if __name__ == "__main__":
    unittest.main()
