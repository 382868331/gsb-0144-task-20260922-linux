"""v2：开放行记录、字段访问、不改变原记录的扩展。

期望值全部手写；每个成功推导都经独立检查器 ``verify()`` 核验
（含效应留痕与独立效应重算）。
"""

import unittest

from polyinfer import A, infer
from polyinfer.errors import InfiniteTypeError, RecordFieldError, TypeMismatchError


def infer_ok(node):
    result = infer(node, language="v2")
    report = result.verify()
    return result, report


class RecordBasicTests(unittest.TestCase):
    def test_closed_record_literal(self):
        r, _ = infer_ok(A.record([("a", A.int_(1)), ("b", A.bool_(True))]))
        # 字段按名排序渲染。
        self.assertEqual(r.principal_type, "{a: Int, b: Bool}")

    def test_get_field_principal(self):
        r, _ = infer_ok(A.lambda_("x", A.get(A.var("x"), "a")))
        self.assertEqual(r.principal_type, "{a: a ; r} -> a")

    def test_get_nested_field(self):
        expr = A.lambda_("x", A.get(A.get(A.var("x"), "inner"), "k"))
        r, _ = infer_ok(expr)
        # 内行变量先出现得 r，外行得 r1。
        self.assertEqual(r.principal_type, "{inner: {k: a ; r} ; r1} -> a")

    def test_open_row_is_polymorphic_two_instances(self):
        # 同一 f 分别用于 {a,b} 与 {a,c} 两种记录（开放行多态）。
        prog = A.let(
            "f", A.lambda_("x", A.get(A.var("x"), "a")),
            A.if_(
                A.bool_(True),
                A.apply(A.var("f"), A.record([("a", A.int_(1)), ("b", A.bool_(True))])),
                A.apply(A.var("f"), A.record([("a", A.int_(2)), ("c", A.int_(3))])),
            ),
        )
        r, rep = infer_ok(prog)
        self.assertEqual(r.principal_type, "Int")
        self.assertEqual(rep.generalizations_checked, 1)

    def test_missing_field_on_closed_record_rejected(self):
        with self.assertRaises(RecordFieldError):
            infer(A.get(A.record([("a", A.int_(1))]), "b"), language="v2")

    def test_duplicate_literal_fields_rejected_at_validation(self):
        from polyinfer.errors import InvalidAstError

        with self.assertRaises(InvalidAstError):
            infer(A.record([("a", A.int_(1)), ("a", A.int_(2))]), language="v2")

    def test_field_type_mismatch(self):
        # {a: Int} 传给要求 {a: Bool} 的函数。
        f = A.lambda_("x", A.get(A.var("x"), "a"))
        prog = A.apply(f, A.record([("a", A.bool_(True))]))
        # f 多态，这里取 Bool 实例，合法；构造真正冲突：
        bad = A.let(
            "g", A.lambda_("x", A.if_(A.get(A.var("x"), "a"),
                                      A.int_(1), A.int_(2))),
            A.apply(A.var("g"), A.record([("a", A.int_(1))])),
        )
        with self.assertRaises(TypeMismatchError):
            infer(bad, language="v2")


class ExtendTests(unittest.TestCase):
    def test_extend_adds_field_keeps_original(self):
        # let r = {a=1} in get ({r with b=true}) "a"  => Int；原 r 未被改变。
        prog = A.let(
            "r", A.record([("a", A.int_(1))]),
            A.get(A.extend(A.var("r"), "b", A.bool_(True)), "a"),
        )
        r, _ = infer_ok(prog)
        self.assertEqual(r.principal_type, "Int")

    def test_extend_result_type_has_both_fields(self):
        f = A.lambda_("r", A.extend(A.var("r"), "b", A.int_(0)))
        r, _ = infer_ok(f)
        # 入参开放 {b 之外的字段 ; ρ}，结果闭合含 a（来自 ρ 吸收）...
        # 实际：r: {a: t ; ρ1}，结果 {a:t, b:Int ; ρ1}（ρ1 保持开放）。
        self.assertIn("b: Int", r.principal_type)

    def test_extend_existing_field_always_errors(self):
        # 同名字段扩展一律报错，即使值类型相同。
        bad = A.let(
            "r", A.record([("a", A.int_(1))]),
            A.extend(A.var("r"), "a", A.int_(2)),
        )
        with self.assertRaises(RecordFieldError) as cm:
            infer(bad, language="v2")
        self.assertEqual(cm.exception.field, "a")

    def test_extend_open_two_instances_share_no_row(self):
        # f(r) = get ({r with z=0}) "z"；两个不同形状的记录实例彼此独立，
        # 开放行不被第一次使用钉死。
        f = A.lambda_("r", A.get(A.extend(A.var("r"), "z", A.int_(0)), "z"))
        r1, _ = infer_ok(A.apply(f, A.record([("x", A.int_(1))])))
        r2, _ = infer_ok(A.apply(f, A.record([("x", A.int_(2)),
                                              ("y", A.bool_(False))])))
        self.assertEqual(r1.principal_type, "Int")
        self.assertEqual(r2.principal_type, "Int")

    def test_extend_branches_must_agree(self):
        # 同一 if 中两分支分别取 Int 字段与 Bool 字段：真实类型冲突。
        prog = A.let(
            "f", A.lambda_("r", A.extend(A.var("r"), "z", A.int_(0))),
            A.if_(
                A.bool_(True),
                A.get(A.apply(A.var("f"), A.record([("x", A.int_(1))])), "z"),
                A.get(A.apply(A.var("f"), A.record([("x", A.int_(2)),
                                                    ("y", A.bool_(False))])), "y"),
            ),
        )
        # then 取 Int 字段、else 取 Bool 字段：if 分支类型冲突
        # （或记录行字段冲突，二者都是构造性拒绝）。
        with self.assertRaises((RecordFieldError, TypeMismatchError)):
            infer(prog, language="v2")


class OpenRowCycleTests(unittest.TestCase):
    def test_open_row_unify_into_cycle_rejected(self):
        # λx. let y = {x with f=1} in get x.f
        # x 被 extend 钉成不含 f 的开放行，又被 get 要求含 f：
        # 尾展开后 f 被提供两次（开放行被强制成环/重复）。
        cyc = A.lambda_("x", A.let(
            "y", A.extend(A.var("x"), "f", A.int_(1)),
            A.get(A.var("x"), "f"),
        ))
        with self.assertRaises((RecordFieldError, InfiniteTypeError)):
            infer(cyc, language="v2")

    def test_open_row_tail_cycle_via_unification(self):
        # λx. let y = {x with g=1}
        #     in if cond then get x.g（要求 x 含 g）else get y.g
        # x 既经 extend 获得 g，又被直接要求含 g：尾展开后 g 出现两次。
        expr = A.lambda_("x", A.let(
            "y", A.extend(A.var("x"), "g", A.int_(1)),
            A.if_(A.bool_(True), A.get(A.var("x"), "g"), A.get(A.var("y"), "g")),
        ))
        with self.assertRaises((RecordFieldError, InfiniteTypeError)):
            infer(expr, language="v2")

    def test_sound_open_row_passes(self):
        # 对照组：只经 y 访问新字段 g，不把 g 强塞回 x 的闭合视图。
        expr = A.lambda_("x", A.let(
            "y", A.extend(A.var("x"), "g", A.int_(1)),
            A.get(A.var("y"), "g"),
        ))
        r, _ = infer_ok(expr)
        self.assertEqual(r.principal_type, "{ ; r} -> Int")


if __name__ == "__main__":
    unittest.main()
