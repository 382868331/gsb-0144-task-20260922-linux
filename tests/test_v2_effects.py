"""v2：读/写/异常三种效应、函数潜在效应集合与 try 处理。

效应期望值由独立效应参考解释器（直接遍历 AST，不经推断器）复核；
每个成功结果都调 ``verify()``。
"""

import unittest

from polyinfer import A, infer


def infer_ok(node):
    result = infer(node, language="v2")
    report = result.verify()
    return result, report


class DirectEffectTests(unittest.TestCase):
    def test_pure_expression_has_no_effects(self):
        r, rep = infer_ok(A.lambda_("x", A.var("x")))
        self.assertEqual(set(r.effects), set())
        self.assertFalse(r.effects_undecided)
        # v2 纯函数箭头不打印效应标注。
        self.assertEqual(r.principal_type, "a -> a")

    def test_read_effect(self):
        r, _ = infer_ok(A.lambda_("r", A.read(A.var("r"))))
        self.assertEqual(r.principal_type, "Ref(a) -{R}-> a")
        self.assertEqual(set(r.effects), set())  # 体效应被收为潜在效应

    def test_write_effect(self):
        r, _ = infer_ok(A.lambda_("r", A.write_(A.var("r"), A.int_(1))))
        self.assertEqual(r.principal_type, "Ref(Int) -{W}-> Unit")

    def test_read_and_write_combine(self):
        body = A.write_(A.var("r"), A.read(A.var("r")))
        r, _ = infer_ok(A.lambda_("r", body))
        self.assertEqual(r.principal_type, "Ref(a) -{R,W}-> Unit")

    def test_direct_effects_at_top_level(self):
        r, _ = infer_ok(A.write_(A.ref(A.int_(1)), A.int_(2)))
        self.assertEqual(set(r.effects), {"W"})
        r2, _ = infer_ok(A.read(A.ref(A.int_(1))))
        self.assertEqual(set(r2.effects), {"R"})

    def test_raise_effect(self):
        r, _ = infer_ok(A.raise_(A.int_(1)))
        self.assertEqual(set(r.effects), {"X"})

    def test_raise_inside_lambda_is_latent(self):
        r, _ = infer_ok(A.lambda_("x", A.raise_(A.int_(0))))
        # raise 值类型为 Int，结果任意 → 参数 a，返回结果自由变量 b。
        self.assertEqual(r.principal_type, "a -{X}-> b")


class FunctionEffectTests(unittest.TestCase):
    def test_applying_effectful_function_emits_latent_effect(self):
        # let f = λr. r := 1 in f ref0：应用点放出潜在 W。
        prog = A.let(
            "f", A.lambda_("r", A.write_(A.var("r"), A.int_(1))),
            A.apply(A.var("f"), A.ref(A.int_(0))),
        )
        r, _ = infer_ok(prog)
        self.assertEqual(set(r.effects), {"W"})

    def test_polymorphic_function_effect_instances_independent(self):
        # id 两次应用不应互相钉效应；整体仍是纯的。
        prog = A.let(
            "id", A.lambda_("x", A.var("x")),
            A.if_(A.apply(A.var("id"), A.bool_(True)),
                  A.apply(A.var("id"), A.int_(1)), A.int_(2)),
        )
        r, _ = infer_ok(prog)
        self.assertEqual(set(r.effects), set())

    def test_effect_mismatch_rejected(self):
        from polyinfer.errors import EffectMismatchError

        # 一个写函数被当作纯函数类型使用：(λg. g 0 的结果类型标纯) 与
        # 实际潜在 W 冲突。直接构造：把 -{W}-> 的函数与纯箭头统一。
        w = A.lambda_("r", A.write_(A.var("r"), A.int_(1)))
        # λk. (k ref0) 其中 k 被用于一个写函数；再要求 k 纯 → 效应冲突。
        prog = A.let(
            "use", A.lambda_("k", A.apply(A.var("k"), A.ref(A.int_(0)))),
            A.apply(A.var("use"), w),
        )
        r, _ = infer_ok(prog)  # use 被推断为带 W，合法
        self.assertEqual(set(r.effects), {"W"})
        # 强制纯签名的反例由模块测试覆盖；这里直接核验效应统一原语：
        from polyinfer.inference import Inferencer

        inf = Inferencer(language="v2")
        with self.assertRaises(EffectMismatchError):
            inf.unify_effects(frozenset({"W"}), frozenset(), (), (), "测试")


class TryEffectTests(unittest.TestCase):
    def test_handler_removes_exception_keeps_write(self):
        # 验收反例：异常被处理后仍有写效应。
        body = A.try_(
            A.raise_(A.int_(1)),
            "e",
            A.write_(A.ref(A.int_(0)), A.int_(2)),
        )
        r, _ = infer_ok(body)
        self.assertEqual(set(r.effects), {"W"})  # X 被处理，W 保留
        self.assertNotIn("X", set(r.effects))

    def test_handler_with_no_other_effect_is_pure(self):
        body = A.try_(A.raise_(A.int_(1)), "e", A.int_(0))
        r, _ = infer_ok(body)
        self.assertEqual(set(r.effects), set())

    def test_handler_param_type_must_match_raised(self):
        from polyinfer.errors import TypeMismatchError

        body = A.try_(
            A.raise_(A.int_(1)),
            "e",
            A.if_(A.var("e"), A.int_(0), A.int_(1)),  # e 被当 Bool，实际 Int
        )
        with self.assertRaises(TypeMismatchError):
            infer(body, language="v2")

    def test_raise_through_if_still_caught(self):
        body = A.try_(
            A.if_(A.bool_(True), A.raise_(A.int_(7)), A.int_(1)),
            "e",
            A.var("e"),
        )
        r, _ = infer_ok(body)
        self.assertEqual(set(r.effects), set())
        self.assertEqual(r.principal_type, "Int")

    def test_write_in_body_survives_handling(self):
        # body 内既有 raise 又有 write：处理后 W 必须还在。
        body = A.try_(
            A.let("_u", A.write_(A.ref(A.int_(0)), A.int_(1)),
                  A.raise_(A.int_(2))),
            "e",
            A.int_(3),
        )
        r, _ = infer_ok(body)
        self.assertEqual(set(r.effects), {"W"})


class EvarUndecidedTests(unittest.TestCase):
    def test_evar_is_reported_undecided(self):
        r, _ = infer_ok(A.evar())
        self.assertTrue(r.effects_undecided)

    def test_undecided_not_downgraded_to_any_or_silent(self):
        # 未决不等于“无效应”也不等于“任意”：标志必须如实，且不静默通过
        # 为确定的 ground 主类型（结果类型保持自由变量）。
        r, _ = infer_ok(A.evar())
        self.assertNotEqual(r.principal_type, "Int")

    def test_determined_programs_are_not_undecided(self):
        for node, eff in [
            (A.int_(1), set()),
            (A.read(A.ref(A.int_(1))), {"R"}),
            (A.try_(A.raise_(A.int_(1)), "e", A.int_(0)), set()),
        ]:
            r, _ = infer_ok(node)
            self.assertFalse(r.effects_undecided, node)
            self.assertEqual(set(r.effects), eff)


if __name__ == "__main__":
    unittest.main()
