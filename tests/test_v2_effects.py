"""v2 效应（read/write/exn）与异常处理的推断测试。

并列规则（README 固定）：

* 效应是潜伏在函数类型上的标签集合：``a -{read,write}-> b``；
* 应用取被应用值/参数/被调潜伏效应的并集；纯函数无箭头标注；
* ``raise`` 引入 ``exn``；``try body handler`` 精确移除 body 的
  ``exn`` 标签，**不**移除 read/write，handler 须为 ``Exn -> t``；
* 未决效应元变量在推断期上确界合并，最终烘焙为具体标签。
"""

import unittest

from polyinfer import A, infer
from polyinfer.errors import TypeMismatchError


def ok(node):
    r = infer(node, max_nodes=3000, version=2)
    r.verify()
    return r


class ReadWriteEffectTests(unittest.TestCase):
    def test_pure_identity_has_no_effect_annotation(self):
        r = ok(A.lambda_("x", A.var("x")))
        self.assertEqual(r.principal_type, "a -> a")
        self.assertEqual(set(r.effects), set())

    def test_read_latent_effect(self):
        r = ok(A.lambda_("rr", A.read(A.var("rr"))))
        self.assertEqual(r.principal_type, "Ref(a) -{read}-> a")

    def test_write_latent_effect_and_unit(self):
        r = ok(A.lambda_("rr", A.write_(A.var("rr"), A.int_(1))))
        self.assertEqual(r.principal_type, "Ref(Int) -{write}-> Unit")

    def test_read_and_write_combine(self):
        body = A.write_(A.var("rr"), A.read(A.var("rr")))
        r = ok(A.lambda_("rr", body))
        self.assertEqual(r.principal_type, "Ref(a) -{read, write}-> Unit")

    def test_application_collects_callee_latent_effect(self):
        # λf. f() 形式：f 带 write，则结果函数也带 write。
        f = A.lambda_("u", A.write_(A.ref(A.int_(0)), A.int_(1)))
        prog = A.lambda_("z", A.apply(f, A.int_(0)))
        r = ok(prog)
        self.assertIn("write", r.principal_type)

    def test_pure_function_passed_to_higher_order_stays_pure(self):
        prog = A.apply(
            A.lambda_("f", A.apply(A.var("f"), A.int_(1))),
            A.lambda_("x", A.var("x")),
        )
        r = ok(prog)
        self.assertEqual(r.principal_type, "Int")
        self.assertEqual(set(r.effects), set())

    def test_write_effect_flows_through_two_modules_style_chain(self):
        # 表达式层面模拟：g 带 write，f 调用 g，f 也必须带 write。
        g = A.lambda_("r", A.write_(A.var("r"), A.int_(1)))
        prog = A.lambda_("s", A.apply(g, A.var("s")))
        r = ok(prog)
        self.assertIn("write", r.principal_type)

    def test_ref_invariant_still_enforced_under_effects(self):
        with self.assertRaises(TypeMismatchError):
            infer(
                A.write_(A.ref(A.int_(1)), A.bool_(True)),
                max_nodes=1000, version=2,
            )


class ExceptionEffectTests(unittest.TestCase):
    def test_raise_requires_exn_type(self):
        with self.assertRaises(TypeMismatchError):
            infer(A.raise_(A.int_(1)), max_nodes=1000, version=2)

    def test_raise_introduces_exn(self):
        r = ok(A.lambda_("x", A.raise_(A.exn("Boom"))))
        self.assertIn("exn", r.principal_type)

    def test_handler_must_be_exn_to_t(self):
        # handler 是 Int -> t 而非 Exn -> t：拒绝。
        with self.assertRaises(TypeMismatchError):
            infer(
                A.try_(A.int_(1), A.lambda_("e", A.var("e"))),
                max_nodes=1000, version=2,
            )

    def test_try_removes_exn_when_handled(self):
        # try (raise E) (λe. 0)：exn 被处理，结果纯。
        prog = A.try_(A.raise_(A.exn("E")), A.lambda_("e", A.int_(0)))
        r = ok(prog)
        self.assertEqual(r.principal_type, "Int")
        self.assertEqual(set(r.effects), set())

    def test_exception_handled_but_write_effect_remains(self):
        """关键反例：异常被 try 处理后，write 效应仍然存在。"""
        prog = A.lambda_(
            "rr",
            A.try_(
                A.let(
                    "_u",
                    A.write_(A.var("rr"), A.int_(1)),
                    A.raise_(A.exn("E")),
                ),
                A.lambda_("e", A.int_(0)),
            ),
        )
        r = ok(prog)
        self.assertEqual(r.principal_type, "Ref(Int) -{write}-> Int")
        self.assertNotIn("exn", r.principal_type)
        self.assertIn("write", r.principal_type)

    def test_exception_handled_but_read_effect_remains(self):
        prog = A.lambda_(
            "rr",
            A.try_(A.read(A.var("rr")), A.lambda_("e", A.int_(0))),
        )
        r = ok(prog)
        self.assertIn("read", r.principal_type)
        self.assertNotIn("exn", r.principal_type)

    def test_handler_latent_effect_is_collected(self):
        # handler 内部写引用：try 的整体效应仍带 write。
        prog = A.lambda_(
            "rr",
            A.try_(
                A.raise_(A.exn("E")),
                A.lambda_("e", A.let(
                    "_u", A.write_(A.var("rr"), A.int_(2)), A.int_(0))),
            ),
        )
        r = ok(prog)
        self.assertIn("write", r.principal_type)

    def test_try_branches_must_agree_in_type(self):
        # body 正常结果钉成 Int，handler 却返回 Bool：拒绝。
        body = A.if_(A.bool_(True), A.raise_(A.exn("E")), A.int_(1))
        with self.assertRaises(TypeMismatchError):
            infer(
                A.try_(body, A.lambda_("e", A.bool_(True))),
                max_nodes=1000, version=2,
            )

    def test_exn_value_polymorphic_identity_over_exn(self):
        # exn 字面量类型是 Exn；λe.e 应用于 exn 得 Exn。
        prog = A.apply(A.lambda_("e", A.var("e")), A.exn("X"))
        r = ok(prog)
        self.assertEqual(r.principal_type, "Exn")


class ValueRestrictionWithEffectsTests(unittest.TestCase):
    def test_write_function_value_binding_is_polymorphic_shape(self):
        # write 函数本身是 lambda（值形式），其引用类型变量可多态；
        # 但潜伏 write 效应每次实例化都保留（效应不被“洗掉”）。
        prog = A.let(
            "setter",
            A.lambda_("rr", A.write_(A.var("rr"), A.int_(1))),
            A.apply(A.var("setter"), A.ref(A.int_(2))),
        )
        r = ok(prog)
        self.assertEqual(r.principal_type, "Unit")
        self.assertIn("write", set(r.effects))

    def test_shared_ref_between_two_uses_is_monomorphic_cell(self):
        # 两个表达式片段共享同一个 ref 值：第一次写 Int 后，第二次读
        # 的类型也必须是 Int（cell 不变式 + 单态）。
        prog = A.let(
            "cell",
            A.ref(A.int_(0)),
            A.let(
                "_u",
                A.write_(A.var("cell"), A.int_(9)),
                A.read(A.var("cell")),
            ),
        )
        r = ok(prog)
        self.assertEqual(r.principal_type, "Int")

    def test_dangerous_ref_var_not_generalized_blocks_two_types(self):
        # 经典漏洞在 v2 仍然被堵：ref(λz.z) 不能当 ∀a. Ref(a→a)。
        leak = A.let(
            "r",
            A.ref(A.lambda_("z", A.var("z"))),
            A.let(
                "_u",
                A.write_(A.var("r"), A.int_(1)),
                A.read(A.var("r")),
            ),
        )
        with self.assertRaises(TypeMismatchError):
            infer(leak, max_nodes=3000, version=2)


class EndToEndAllThreeFeatureGroupsTests(unittest.TestCase):
    def test_open_record_extend_effects_exceptions_together(self):
        """一个程序同时用到三组新能力：

        1) 开放行记录 + 非破坏式扩展（box -> box2 加 tag 字段）；
        2) 共享可变引用的 read/write 效应；
        3) raise/try（exn 被处理），写/读效应保留。
        结果为 Int，本次效应 {read, write}（无 exn）。
        """
        prog = A.let(
            "box", A.record([("cell", A.ref(A.int_(1)))]),
            A.let(
                "box2", A.extend(A.var("box"), [("tag", A.bool_(True))]),
                A.let(
                    "_u", A.write_(A.field_(A.var("box2"), "cell"), A.int_(2)),
                    A.try_(
                        A.raise_(A.exn("E")),
                        A.lambda_("e", A.read(A.field_(A.var("box2"), "cell"))),
                    ),
                ),
            ),
        )
        r = ok(prog)
        self.assertEqual(r.principal_type, "Int")
        self.assertEqual(set(r.effects), {"read", "write"})
        # record 字段含 ref（扩展式）：box 不被泛化（值限制 + 危险变量）。
        box_gen = next(g for g in r.generalizations if g.name == "box")
        self.assertEqual(box_gen.generalized, ())
        self.assertEqual(box_gen.field_tags, ("ref",))


if __name__ == "__main__":
    unittest.main()
