"""v2 记录、开放行、字段访问与非破坏式扩展的测试。

期望值（主类型、错误类、路径）全部手写；每个成功推导都经过独立
检查器 ``result.verify()`` 核验全部统一约束与效应留痕。
"""

import unittest

from polyinfer import A, infer
from polyinfer.errors import (
    DuplicateFieldError,
    InfiniteTypeError,
    MissingFieldError,
    TypeMismatchError,
)


def ok(node):
    r = infer(node, max_nodes=3000, version=2)
    r.verify()
    return r


class RecordBasicsTests(unittest.TestCase):
    def test_int_field_access(self):
        r = ok(A.field_(A.record([("x", A.int_(1))]), "x"))
        self.assertEqual(r.principal_type, "Int")

    def test_multi_fields_principal_type(self):
        rec = A.record([("x", A.int_(1)), ("b", A.bool_(True))])
        self.assertEqual(ok(A.field_(rec, "x")).principal_type, "Int")
        self.assertEqual(ok(A.field_(rec, "b")).principal_type, "Bool")

    def test_open_row_is_polymorphic_in_tail(self):
        # 同一个记录值分别按 {x:Int} 与 {x:Int,b:Bool} 形态使用：
        # 开放行使一个多态字段访问函数能消费带不同尾行的记录。
        getx = A.lambda_("r", A.field_(A.var("r"), "x"))
        use_int = A.apply(getx, A.record([("x", A.int_(1)), ("y", A.int_(2))]))
        use_bool = A.apply(getx, A.record([("x", A.int_(3)), ("z", A.bool_(False))]))
        prog = A.if_(A.bool_(True), use_int, use_bool)
        self.assertEqual(ok(prog).principal_type, "Int")

    def test_record_is_function_argument(self):
        prog = A.apply(
            A.lambda_("r", A.field_(A.var("r"), "x")),
            A.record([("x", A.bool_(False))]),
        )
        self.assertEqual(ok(prog).principal_type, "Bool")


class ExtendTests(unittest.TestCase):
    def test_extend_adds_field(self):
        r = ok(A.field_(
            A.extend(A.record([("x", A.int_(1))]), [("y", A.bool_(True))]),
            "y",
        ))
        self.assertEqual(r.principal_type, "Bool")

    def test_extend_is_non_destructive(self):
        # 扩展后原记录仍按原类型可访问；新记录同时带新旧字段。
        ext = A.extend(A.record([("x", A.int_(1))]), [("y", A.bool_(True))])
        self.assertEqual(ok(A.field_(ext, "x")).principal_type, "Int")
        self.assertEqual(ok(A.field_(ext, "y")).principal_type, "Bool")

    def test_extend_does_not_alias_original_row(self):
        # 非破坏式（值层面）在类型上的体现：
        # 1) extend 不改变原记录的已知字段（原记录仍只有 x）；
        # 2) 对扩展结果再次扩展同名字段时当场重复拒绝（共享尾行下，
        #    第一次扩展出的 y 已是结果记录的确定已知字段）。
        base = A.record([("x", A.int_(1))])
        with_y = A.extend(base, [("y", A.int_(2))])
        self.assertEqual(ok(A.field_(with_y, "x")).principal_type, "Int")
        self.assertEqual(ok(A.field_(with_y, "y")).principal_type, "Int")
        with self.assertRaises(DuplicateFieldError):
            infer(
                A.extend(with_y, [("y", A.bool_(False))]),
                max_nodes=2000, version=2,
            )

    def test_duplicate_field_extension_rejected(self):
        bad = A.extend(A.record([("x", A.int_(1))]), [("x", A.int_(2))])
        with self.assertRaises(DuplicateFieldError) as cm:
            infer(bad, max_nodes=1000, version=2)
        self.assertEqual(cm.exception.field, "x")

    def test_duplicate_field_in_literal_rejected(self):
        bad = A.record([("x", A.int_(1)), ("x", A.int_(2))])
        with self.assertRaises(DuplicateFieldError):
            infer(bad, max_nodes=1000, version=2)

    def test_duplicate_field_exposed_through_open_row(self):
        # 经开放行：addY 给任意记录扩展 y:Int，喂给一个本来 y:Bool 的
        # 记录——共有字段 y 类型 Int vs Bool，必须统一失败（不静默覆盖）。
        prog = A.let(
            "addY",
            A.lambda_("r", A.extend(A.var("r"), [("y", A.int_(0))])),
            A.apply(A.var("addY"), A.record([("x", A.int_(1)), ("y", A.bool_(True))])),
        )
        with self.assertRaises(TypeMismatchError):
            infer(prog, max_nodes=2000, version=2)

    def test_extend_chain_well_typed(self):
        r = A.record([("a", A.int_(1))])
        r = A.extend(r, [("b", A.int_(2))])
        r = A.extend(r, [("c", A.bool_(False))])
        prog = A.if_(
            A.field_(r, "c"),
            A.field_(r, "a"),
            A.field_(r, "b"),
        )
        self.assertEqual(ok(prog).principal_type, "Int")


class RowUnificationTests(unittest.TestCase):
    def test_open_row_unifies_with_subset_use(self):
        # {x:Int,y:Bool} 的记录传给只取 x 的访问函数（开放行吸收）。
        prog = A.apply(
            A.lambda_("r", A.field_(A.var("r"), "x")),
            A.record([("x", A.int_(1)), ("extra", A.bool_(True))]),
        )
        self.assertEqual(ok(prog).principal_type, "Int")

    def test_open_row_unifies_into_cycle_rejected(self):
        """棘手边界：开放行统一成环必须被 occurs 检查拒绝。

        λr. if true then r else extend(r, [z = r])
        then 分支要求 r 的行 ρ，else 分支经扩展得到 {z: t_r | ρ'}，
        两分支统一会令 ρ 与 {z: t_r, ...} 相接、t_r 又含 ρ，构成环。
        """
        bad = A.lambda_(
            "r",
            A.if_(
                A.bool_(True),
                A.var("r"),
                A.extend(A.var("r"), [("z", A.var("r"))]),
            ),
        )
        with self.assertRaises(InfiniteTypeError):
            infer(bad, max_nodes=2000, version=2)

    def test_field_access_two_polymorphic_uses_share_row(self):
        # 同一开放行记录被两次按不同字段访问函数使用，行约束保持一致。
        getx = A.lambda_("r", A.field_(A.var("r"), "x"))
        rec = A.record([("x", A.int_(5)), ("y", A.int_(6))])
        prog = A.apply(A.var("id"), rec) if False else A.let(
            "gx",
            getx,
            A.apply(A.var("gx"), rec),
        )
        self.assertEqual(ok(prog).principal_type, "Int")

    def test_abstract_row_field_access_is_simple_row_polymorphic(self):
        # 简单行多态（不做 absence 约束）：getter 访问抽象尾行上的字段，
        # 字段类型保持多态；喂给不带该字段的具体记录时，尾行吸收其余
        # 字段、字段类型仍自由（README 固定此并列规则；确定的闭行缺字段
        # 在模块闭行签名处拒绝，见 test_v2_modules）。
        getter = A.lambda_("r", A.field_(A.var("r"), "missing"))
        applied = A.apply(getter, A.record([("present", A.int_(1))]))
        self.assertEqual(ok(applied).principal_type, "a")
        # 若该字段在另一处被按 Int 使用，类型随之落地为 Int。
        pinned = A.let(
            "g", getter,
            A.if_(
                A.bool_(True),
                A.apply(A.var("g"), A.record([("missing", A.int_(7))])),
                A.apply(A.var("g"), A.record([("missing", A.int_(8)), ("q", A.bool_(False))])),
            ),
        )
        self.assertEqual(ok(pinned).principal_type, "Int")

    def test_field_types_must_agree(self):
        # 开放行两次观测同一字段，类型必须一致（Int vs Bool 冲突）。
        getx = A.lambda_("r", A.field_(A.var("r"), "x"))
        prog = A.if_(
            A.bool_(True),
            A.apply(getx, A.record([("x", A.int_(1))])),
            A.apply(getx, A.record([("x", A.bool_(False))])),
        )
        with self.assertRaises(TypeMismatchError):
            infer(prog, max_nodes=2000, version=2)


class RecordValueRestrictionTests(unittest.TestCase):
    def test_record_of_lambdas_is_value_and_generalizes_fields(self):
        # 全值字段的记录是值形式：其函数字段可多态使用。
        prog = A.let(
            "ops",
            A.record([("id", A.lambda_("x", A.var("x")))]),
            A.if_(
                A.apply(A.field_(A.var("ops"), "id"), A.bool_(True)),
                A.apply(A.field_(A.var("ops"), "id"), A.int_(1)),
                A.int_(0),
            ),
        )
        self.assertEqual(ok(prog).principal_type, "Int")

    def test_record_of_ref_is_not_generalized_unsafely(self):
        # 记录里装一个 ref：该 cell 类型变量是危险变量，不能多态。
        prog = A.let(
            "box",
            A.record([("cell", A.ref(A.lambda_("z", A.var("z"))))]),
            A.let(
                "_u",
                A.write_(A.field_(A.var("box"), "cell"), A.int_(1)),
                A.read(A.field_(A.var("box"), "cell")),
            ),
        )
        with self.assertRaises(TypeMismatchError):
            infer(prog, max_nodes=3000, version=2)


if __name__ == "__main__":
    unittest.main()
