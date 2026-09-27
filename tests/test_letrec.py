"""letrec 互相递归绑定组的推断测试。

覆盖本特性的契约要点：

* 组内函数互相调用成功（even/odd 经典例子），结果与绑定书写顺序无关；
* 整组在同一套约束下检查：跨绑定的类型冲突被定位拒绝；
* 整组检查结束后才泛化：未被组员共享的变量可在 body 多态使用，
  被其他组员共享的变量保持单态；
* 值限制不因递归组放松：ref 等扩展式绑定在组内依然不泛化，
  引用多态漏洞仍然被拦截；
* 结构校验：空组、重复名字、非 list/非 dict 绑定均被拒绝；
* 普通非递归表达式行为不变（冒烟对照）。
"""

import unittest

from polyinfer import A, infer
from polyinfer.errors import (
    InvalidAstError,
    TypeMismatchError,
    UnboundVariableError,
)


def even_odd_group(first, second):
    """even/odd 互相递归，iszero/pred 由外层 lambda 提供。"""
    even = (
        "even",
        A.lambda_(
            "n",
            A.if_(
                A.apply(A.var("iszero"), A.var("n")),
                A.bool_(True),
                A.apply(A.var("odd"), A.apply(A.var("pred"), A.var("n"))),
            ),
        ),
    )
    odd = (
        "odd",
        A.lambda_(
            "n",
            A.if_(
                A.apply(A.var("iszero"), A.var("n")),
                A.bool_(False),
                A.apply(A.var("even"), A.apply(A.var("pred"), A.var("n"))),
            ),
        ),
    )
    pair = {"even": even, "odd": odd}
    return A.lambda_(
        "iszero",
        A.lambda_(
            "pred",
            A.letrec([pair[first], pair[second]], A.var("even")),
        ),
    )


class MutualRecursionTest(unittest.TestCase):
    def test_even_odd_principal_type(self):
        result = infer(even_odd_group("even", "odd"))
        self.assertEqual(
            result.principal_type, "(a -> Bool) -> (a -> a) -> a -> Bool"
        )
        result.verify()  # 独立替换器核验全部约束与泛化事件

    def test_binding_order_irrelevant(self):
        forward = infer(even_odd_group("even", "odd"))
        backward = infer(even_odd_group("odd", "even"))
        self.assertEqual(forward.principal_type, backward.principal_type)

    def test_self_recursion_single_binding(self):
        # letrec loop = λx. loop x in loop —— 自递归主类型 a -> b
        node = A.letrec(
            [("loop", A.lambda_("x", A.apply(A.var("loop"), A.var("x"))))],
            A.var("loop"),
        )
        result = infer(node)
        self.assertEqual(result.principal_type, "a -> b")
        result.verify()

    def test_group_visible_in_body(self):
        node = A.letrec(
            [("f", A.lambda_("x", A.var("x")))],
            A.apply(A.var("f"), A.int_(1)),
        )
        self.assertEqual(infer(node).principal_type, "Int")

    def test_letrec_does_not_leak_scope(self):
        node = A.letrec(
            [("f", A.lambda_("x", A.var("x")))],
            A.var("f"),
        )
        # 组外同名变量未绑定
        with self.assertRaises(UnboundVariableError):
            infer(A.lambda_("y", A.apply(A.var("f"), A.var("y"))))
        self.assertEqual(infer(node).principal_type, "a -> a")


class ConflictTest(unittest.TestCase):
    def test_cross_binding_type_conflict(self):
        # f 的条件要求参数是 Bool；g 把 Int 参数直接传给 f —— 整组冲突。
        bad = A.letrec(
            [
                (
                    "f",
                    A.lambda_(
                        "x",
                        A.if_(
                            A.var("x"),
                            A.apply(A.var("g"), A.int_(1)),
                            A.apply(A.var("g"), A.int_(2)),
                        ),
                    ),
                ),
                ("g", A.lambda_("y", A.apply(A.var("f"), A.var("y")))),
            ],
            A.var("f"),
        )
        with self.assertRaises(TypeMismatchError) as ctx:
            infer(bad)
        # 失败约束两端路径定位到组内绑定
        self.assertIn("bindings", ctx.exception.left_path)
        self.assertIn("bindings", ctx.exception.right_path)

    def test_conflict_detected_regardless_of_order(self):
        mk = lambda flip: A.letrec(  # noqa: E731
            [
                (
                    "f",
                    A.lambda_(
                        "x",
                        A.if_(
                            A.var("x"),
                            A.apply(A.var("g"), A.int_(1)),
                            A.apply(A.var("g"), A.int_(2)),
                        ),
                    ),
                ),
                ("g", A.lambda_("y", A.apply(A.var("f"), A.var("y")))),
            ][:: -1 if flip else 1],
            A.var("f"),
        )
        with self.assertRaises(TypeMismatchError):
            infer(mk(False))
        with self.assertRaises(TypeMismatchError):
            infer(mk(True))


class GeneralizationTest(unittest.TestCase):
    def test_generalization_after_group_check(self):
        # f = λx.x 与 h = λz.z 互不引用：两者都可泛化，
        # body 中 f 同时按 Bool->Bool 与 Int->Int 使用。
        node = A.letrec(
            [
                ("f", A.lambda_("x", A.var("x"))),
                ("h", A.lambda_("z", A.var("z"))),
            ],
            A.if_(
                A.apply(A.var("f"), A.bool_(True)),
                A.apply(A.var("f"), A.int_(1)),
                A.apply(A.var("h"), A.int_(2)),
            ),
        )
        result = infer(node)
        self.assertEqual(result.principal_type, "Int")
        result.verify()
        generalized = {g.name: g.generalized for g in result.generalizations}
        self.assertEqual(len(generalized["f"]), 1)
        self.assertEqual(len(generalized["h"]), 1)

    def test_shared_variable_stays_monomorphic(self):
        # g 的定义里使用了 f，二者共享类型变量：f 不在 body 多态。
        node = A.letrec(
            [
                ("f", A.lambda_("x", A.var("x"))),
                ("g", A.lambda_("y", A.apply(A.var("f"), A.var("y")))),
            ],
            A.if_(
                A.apply(A.var("f"), A.bool_(True)),
                A.apply(A.var("f"), A.int_(1)),
                A.int_(2),
            ),
        )
        with self.assertRaises(TypeMismatchError):
            infer(node)


class ValueRestrictionTest(unittest.TestCase):
    def test_ref_binding_in_group_not_generalized(self):
        # 引用多态漏洞不能借递归组绕过：r 是扩展式绑定，保持单态，
        # 写入 Int 后按函数读出使用必须被拒绝。
        hole = A.letrec(
            [
                ("r", A.ref(A.lambda_("z", A.var("z")))),
                ("w", A.lambda_("u", A.write_(A.var("r"), A.int_(1)))),
            ],
            A.apply(A.read(A.var("r")), A.bool_(True)),
        )
        with self.assertRaises(TypeMismatchError):
            infer(hole)

    def test_expansive_binding_empty_generalization(self):
        node = A.letrec(
            [
                ("r", A.ref(A.int_(0))),
                ("f", A.lambda_("x", A.var("x"))),
            ],
            A.var("f"),
        )
        result = infer(node)
        self.assertEqual(result.principal_type, "a -> a")
        by_name = {g.name: g for g in result.generalizations}
        self.assertEqual(by_name["r"].generalized, ())
        self.assertEqual(len(by_name["f"].generalized), 1)
        result.verify()


class ValidationTest(unittest.TestCase):
    def test_empty_bindings_rejected(self):
        with self.assertRaises(InvalidAstError):
            infer({"tag": "letrec", "bindings": [], "body": A.int_(1)})

    def test_duplicate_names_rejected(self):
        dup = A.letrec(
            [("f", A.lambda_("x", A.var("x"))), ("f", A.lambda_("y", A.var("y")))],
            A.var("f"),
        )
        with self.assertRaises(InvalidAstError):
            infer(dup)

    def test_bindings_must_be_list(self):
        with self.assertRaises(InvalidAstError):
            infer({"tag": "letrec", "bindings": "nope", "body": A.int_(1)})

    def test_binding_shape_checked(self):
        bad = {
            "tag": "letrec",
            "bindings": [{"name": "f"}],
            "body": A.int_(1),
        }
        with self.assertRaises(InvalidAstError):
            infer(bad)

    def test_unbound_outside_group_still_rejected(self):
        node = A.letrec(
            [("f", A.lambda_("x", A.apply(A.var("missing"), A.var("x"))))],
            A.var("f"),
        )
        with self.assertRaises(UnboundVariableError):
            infer(node)

    def test_plain_let_still_nonrecursive(self):
        node = A.let("f", A.lambda_("x", A.apply(A.var("f"), A.var("x"))), A.var("f"))
        with self.assertRaises(UnboundVariableError):
            infer(node)


if __name__ == "__main__":
    unittest.main()

