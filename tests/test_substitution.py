"""替换组合 / occurs / 渲染 / 规范命名的单元测试。"""

import unittest

from polyinfer.types import (
    BOOL,
    INT,
    UNIT,
    TFun,
    TRef,
    TVar,
    apply,
    bind,
    canonical_names,
    occurs,
    render_type,
)


class SubstitutionTests(unittest.TestCase):
    def setUp(self):
        self.a = TVar(0)
        self.b = TVar(1)
        self.c = TVar(2)

    def test_empty_subst_is_identity(self):
        self.assertEqual(apply({}, self.a), self.a)
        self.assertEqual(apply({}, TFun(self.a, self.b)), TFun(self.a, self.b))

    def test_simple_bind_and_apply(self):
        s = bind({}, self.a, INT)
        self.assertEqual(apply(s, self.a), INT)
        self.assertEqual(apply(s, TFun(self.a, self.b)), TFun(INT, self.b))

    def test_bind_composition_applies_new_binding_to_old_rhs(self):
        # s0 = {a -> b -> b}；再 bind {b -> Int}，旧右端必须同步推进。
        s0 = bind({}, self.a, TFun(self.b, self.b))
        s1 = bind(s0, self.b, INT)
        self.assertEqual(apply(s1, self.a), TFun(INT, INT))
        self.assertEqual(apply(s1, self.b), INT)
        self.assertEqual(apply(s1, TFun(self.a, self.b)), TFun(TFun(INT, INT), INT))

    def test_composition_order_semantics(self):
        # 先 a->Int，再 b->a：b 解析时应看到已有替换，得到 b->Int。
        s0 = bind({}, self.a, INT)
        s1 = bind(s0, self.b, self.a)
        self.assertEqual(apply(s1, self.b), INT)

    def test_idempotent_after_chained_binds(self):
        s = bind({}, self.a, TFun(self.b, self.c))
        s = bind(s, self.b, TRef(self.c))
        s = bind(s, self.c, BOOL)
        once = apply(s, TFun(self.a, self.b))
        twice = apply(s, once)
        self.assertEqual(once, twice)
        self.assertEqual(once, TFun(TFun(TRef(BOOL), BOOL), TRef(BOOL)))

    def test_occurs_basic(self):
        self.assertTrue(occurs(self.a, self.a))
        self.assertFalse(occurs(self.a, self.b))
        self.assertTrue(occurs(self.a, TFun(INT, self.a)))
        self.assertTrue(occurs(self.a, TRef(TFun(self.a, UNIT))))
        self.assertFalse(occurs(self.a, TFun(INT, TRef(BOOL))))


class RendererTests(unittest.TestCase):
    def test_basic_rendering(self):
        self.assertEqual(str(INT), "Int")
        self.assertEqual(str(BOOL), "Bool")
        self.assertEqual(str(UNIT), "Unit")
        self.assertEqual(str(TRef(INT)), "Ref(Int)")

    def test_function_arrow_right_associative(self):
        a, b, c = TVar(0), TVar(1), TVar(2)
        self.assertEqual(str(TFun(a, b)), "?0 -> ?1")
        # 高阶参数加括号
        self.assertEqual(str(TFun(TFun(a, b), c)), "(?0 -> ?1) -> ?2")
        # 返回函数不加多余括号
        self.assertEqual(str(TFun(a, TFun(b, c))), "?0 -> ?1 -> ?2")

    def test_ref_of_function_no_double_parens(self):
        a = TVar(0)
        self.assertEqual(str(TRef(TFun(a, a))), "Ref(?0 -> ?0)")

    def test_nested_ref_function_mix(self):
        a, b = TVar(0), TVar(1)
        t = TFun(TRef(TFun(a, b)), TRef(a))
        self.assertEqual(str(t), "Ref(?0 -> ?1) -> Ref(?0)")


class CanonicalNamingTests(unittest.TestCase):
    def test_identity_first_occurrence(self):
        a = TVar(7)  # 内部 id 不影响规范名
        names = canonical_names(TFun(a, a))
        self.assertEqual(names[a], "a")
        self.assertEqual(render_type(TFun(a, a), names), "a -> a")

    def test_const_order(self):
        x, y = TVar(3), TVar(9)
        t = TFun(x, TFun(y, x))
        names = canonical_names(t)
        self.assertEqual([names[x], names[y]], ["a", "b"])
        self.assertEqual(render_type(t, names), "a -> b -> a")

    def test_compose_order(self):
        f, g, x = TVar(0), TVar(1), TVar(2)
        # (f -> g -> x) 展开：(a -> b) -> (c -> a) -> c -> b
        a, b = TVar(10), TVar(11)
        t = TFun(TFun(a, b), TFun(TFun(x, a), TFun(x, b)))
        names = canonical_names(t)
        self.assertEqual(render_type(t, names), "(a -> b) -> (c -> a) -> c -> b")

    def test_names_stable_under_internal_renumbering(self):
        # 两棵结构相同、内部 id 不同的类型必须给出相同规范字符串。
        t1 = TFun(TVar(0), TVar(1))
        t2 = TFun(TVar(42), TVar(5))
        self.assertEqual(
            render_type(t1, canonical_names(t1)),
            render_type(t2, canonical_names(t2)),
        )


if __name__ == "__main__":
    unittest.main()
