"""polyinfer 的单元测试。

约束复核使用本文件内独立编写的替换器 _resolve（不调用被测核心的
apply_substitution / unify），对每个成功推导的全部约束逐一验证：
在最终替换下，每条约束两端结构相等。
"""

import unittest

from polyinfer import (
    Var, Int, Bool, Lam, App, Let, If, Ref, Read, Write,
    TVar, TFun, TRef, TInt, TBool, TUnit,
    INT, BOOL, UNIT,
    infer, unify, type_to_string, format_path,
    InferError, InputError, UnboundVariableError,
    UnifyError, OccursError, BudgetExhaustedError,
)


# ---------------------------------------------------------------------------
# 独立参考：替换器（与被测核心分开实现，仅复用公开类型数据结构）
# ---------------------------------------------------------------------------

def _resolve(subst, t):
    """独立实现的替换应用：反复解析变量直到不可再解析。"""
    if isinstance(t, TVar):
        if t.id in subst:
            return _resolve(subst, subst[t.id])
        return t
    if isinstance(t, TFun):
        return TFun(_resolve(subst, t.arg), _resolve(subst, t.ret))
    if isinstance(t, TRef):
        return TRef(_resolve(subst, t.inner))
    if isinstance(t, (TInt, TBool, TUnit)):
        return t
    raise AssertionError(f"unexpected type in constraint: {t!r}")


def check_all_constraints(testcase, result):
    """独立复核：最终替换必须满足推导记录的全部约束。"""
    testcase.assertGreater(len(result.constraints), 0,
                           "fixture should generate at least one constraint")
    for c in result.constraints:
        left = _resolve(result.substitution, c.left)
        right = _resolve(result.substitution, c.right)
        testcase.assertEqual(
            left, right,
            f"constraint not satisfied: {c.left!r} vs {c.right!r} "
            f"at {format_path(c.left_path)} vs {format_path(c.right_path)}")


# ---------------------------------------------------------------------------
# 程序构造辅助
# ---------------------------------------------------------------------------

def identity():
    return Lam("x", Var("x"))


class TestSuccessfulInference(unittest.TestCase):
    def test_identity_principal_type(self):
        r = infer(identity())
        self.assertEqual(r.type_str, "forall a. a -> a")

    def test_identity_two_instances(self):
        # 恒等函数的两种实例：Int 与 Bool
        r_int = infer(App(identity(), Int(3)))
        self.assertEqual(r_int.type, INT)
        self.assertEqual(r_int.type_str, "Int")
        check_all_constraints(self, r_int)

        r_bool = infer(App(identity(), Bool(True)))
        self.assertEqual(r_bool.type, BOOL)
        self.assertEqual(r_bool.type_str, "Bool")
        check_all_constraints(self, r_bool)

    def test_let_generalizes_identity_used_at_two_types(self):
        # let id = λx.x in let a = id 3 in id true  :  Bool
        prog = Let("id", identity(),
                   Let("a", App(Var("id"), Int(3)),
                       App(Var("id"), Bool(True))))
        r = infer(prog)
        self.assertEqual(r.type, BOOL)
        check_all_constraints(self, r)

    def test_const_function_canonical_naming(self):
        # λx.λy.x —— 变量按首次出现命名 a, b
        r = infer(Lam("x", Lam("y", Var("x"))))
        self.assertEqual(r.type_str, "forall a, b. a -> b -> a")

    def test_write_returns_unit(self):
        r = infer(Write(Ref(Int(1)), Int(2)))
        self.assertEqual(r.type, UNIT)
        self.assertEqual(r.type_str, "Unit")
        check_all_constraints(self, r)

    def test_read_type(self):
        # λr. read r  :  forall a. ref a -> a
        r = infer(Lam("r", Read(Var("r"))))
        self.assertEqual(r.type_str, "forall a. ref a -> a")

    def test_ref_of_identity(self):
        r = infer(Ref(identity()))
        self.assertEqual(r.type_str, "forall a. ref (a -> a)")

    def test_if_branches(self):
        r = infer(If(Bool(True), Int(1), Int(2)))
        self.assertEqual(r.type, INT)
        check_all_constraints(self, r)

    def test_let_shadowing(self):
        # let x = 1 in let x = true in x  :  Bool
        prog = Let("x", Int(1), Let("x", Bool(True), Var("x")))
        r = infer(prog)
        self.assertEqual(r.type, BOOL)

    def test_nested_let_value_restriction_positive_control(self):
        # f 绑定的是 lambda（语法值）→ 泛化 → 两种实例均可用
        prog = Let("f", Lam("y", Var("y")),
                   Let("a", App(Var("f"), Int(3)),
                       App(Var("f"), Bool(True))))
        r = infer(prog)
        self.assertEqual(r.type, BOOL)
        check_all_constraints(self, r)

    def test_write_read_roundtrip(self):
        # let r = ref 0 in let _ = write r 1 in read r  :  Int
        prog = Let("r", Ref(Int(0)),
                   Let("_", Write(Var("r"), Int(1)),
                       Read(Var("r"))))
        r = infer(prog)
        self.assertEqual(r.type, INT)
        check_all_constraints(self, r)


class TestRejections(unittest.TestCase):
    def test_lambda_param_is_monomorphic(self):
        # λf. let a = f 3 in f true —— f 单态，Int 与 Bool 冲突
        prog = Lam("f", Let("a", App(Var("f"), Int(3)),
                            App(Var("f"), Bool(True))))
        with self.assertRaises(UnifyError):
            infer(prog)

    def test_self_application_infinite_type(self):
        # λx. x x —— occurs 检查拒绝无限类型
        with self.assertRaises(OccursError):
            infer(Lam("x", App(Var("x"), Var("x"))))
        # OccursError 是 UnifyError 的子类
        self.assertTrue(issubclass(OccursError, UnifyError))

    def test_reference_polymorphism_hole_rejected(self):
        # 经典引用多态漏洞：若泛化 ref (λx.x)，则先写入 Int 函数、
        # 再以 Bool 读取会被错误接受。值限制下必须拒绝。
        # let r = ref (λx.x) in let _ = write r (λy.7) in (read r) true
        prog = Let("r", Ref(identity()),
                   Let("_", Write(Var("r"), Lam("y", Int(7))),
                       App(Read(Var("r")), Bool(True))))
        with self.assertRaises(UnifyError) as cm:
            infer(prog)
        err = cm.exception
        self.assertFalse(isinstance(err, OccursError))
        # 失败的是 Int 与 Bool 的冲突
        self.assertEqual({type(err.left), type(err.right)},
                         {TInt, TBool})

    def test_value_restriction_non_value_not_generalized(self):
        # f = (λx.x) (λy.y) 是 apply（非语法值）→ 不泛化 → 拒绝
        prog = Let("f", App(identity(), identity()),
                   Let("a", App(Var("f"), Int(3)),
                       App(Var("f"), Bool(True))))
        with self.assertRaises(UnifyError):
            infer(prog)

    def test_if_branch_mismatch(self):
        with self.assertRaises(UnifyError):
            infer(If(Bool(True), Int(1), Bool(False)))

    def test_if_cond_must_be_bool(self):
        with self.assertRaises(UnifyError) as cm:
            infer(If(Int(1), Int(2), Int(3)))
        self.assertEqual(cm.exception.left_path, ("if.cond",))
        self.assertEqual(cm.exception.right_path, ())

    def test_write_type_mismatch(self):
        # ref 0 写入 true —— 引用赋值拒绝
        prog = Let("r", Ref(Int(0)), Write(Var("r"), Bool(True)))
        with self.assertRaises(UnifyError):
            infer(prog)

    def test_unbound_variable(self):
        with self.assertRaises(UnboundVariableError) as cm:
            infer(App(Var("x"), Int(1)))
        self.assertEqual(cm.exception.name, "x")
        self.assertEqual(cm.exception.path, ("app.func",))

    def test_no_recursive_let(self):
        # let 绑定名在自身值中不可见（不支持递归 let）
        prog = Let("f", Lam("x", App(Var("f"), Var("x"))), Var("f"))
        with self.assertRaises(UnboundVariableError):
            infer(prog)

    def test_error_paths_locate_deep_failure(self):
        # let a = 1 in (λx.x) true 应用于深层位置，路径可定位
        prog = Let("a", Int(1),
                   If(Bool(True), App(identity(), Bool(True)), Int(2)))
        # if 分支 Int vs Bool 冲突，路径指向 if.then / if.else
        with self.assertRaises(UnifyError) as cm:
            infer(prog)
        err = cm.exception
        self.assertEqual(err.left_path, ("let.body", "if.then"))
        self.assertEqual(err.right_path, ("let.body", "if.else"))
        self.assertIn("let.body/if.then", str(err))


class TestInputValidation(unittest.TestCase):
    def test_int_rejects_bool(self):
        with self.assertRaises(InputError):
            infer(Int(True))

    def test_int_rejects_float(self):
        with self.assertRaises(InputError):
            infer(Int(1.5))

    def test_int_rejects_nan(self):
        with self.assertRaises(InputError) as cm:
            infer(Int(float("nan")))
        self.assertIn("NaN", str(cm.exception))

    def test_int_rejects_infinity(self):
        with self.assertRaises(InputError) as cm:
            infer(Int(float("inf")))
        self.assertIn("Infinity", str(cm.exception))

    def test_bool_rejects_int(self):
        with self.assertRaises(InputError):
            infer(Bool(1))

    def test_var_rejects_empty_name(self):
        with self.assertRaises(InputError):
            infer(Var(""))

    def test_non_node_rejected(self):
        with self.assertRaises(InputError):
            infer("λx.x")

    def test_error_path_locates_bad_field(self):
        prog = Let("a", Int(1), App(Var("a"), Int(False)))
        with self.assertRaises(InputError) as cm:
            infer(prog)
        self.assertIn("let.body/app.arg", str(cm.exception))

    def test_node_limit(self):
        # 501 个节点：Int(0) 外套 250 层 App(_, Int(1))
        node = Int(0)
        for _ in range(250):
            node = App(node, Int(1))
        with self.assertRaises(InputError) as cm:
            infer(node)
        self.assertIn("500", str(cm.exception))
        # 恰好 500 节点不触发上限（会在推断阶段失败，而非 InputError）
        node = Int(0)
        for _ in range(249):
            node = App(node, Int(1))  # 1 + 2*249 = 499，再加一层读
        node = Read(node)  # 500
        with self.assertRaises(InferError) as cm2:
            infer(node)
        self.assertNotIsInstance(cm2.exception, InputError)

    def test_limit_args_reject_bool(self):
        with self.assertRaises(InputError):
            infer(Int(1), max_nodes=True)
        with self.assertRaises(InputError):
            infer(Int(1), max_steps=False)

    def test_budget_exhausted_is_distinct_from_unify_error(self):
        # 该程序需要多次统一；预算 1 必然耗尽
        prog = If(Bool(True), App(identity(), Int(3)), Int(4))
        with self.assertRaises(BudgetExhaustedError) as cm:
            infer(prog, max_steps=1)
        self.assertNotIsInstance(cm.exception, UnifyError)
        self.assertNotEqual(BudgetExhaustedError, UnifyError)


class TestPublicUnify(unittest.TestCase):
    def test_unify_simple(self):
        s = unify(TFun(TVar(0), TVar(1)), TFun(INT, TVar(1)))
        self.assertEqual(_resolve(s, TVar(0)), INT)

    def test_unify_occurs(self):
        with self.assertRaises(OccursError):
            unify(TVar(0), TFun(TVar(0), INT))

    def test_unify_mismatch(self):
        with self.assertRaises(UnifyError):
            unify(INT, BOOL)

    def test_unify_refs(self):
        s = unify(TRef(TVar(0)), TRef(TFun(BOOL, TVar(1))))
        self.assertEqual(_resolve(s, TVar(0)), TFun(BOOL, TVar(1)))

    def test_unify_rejects_bad_args(self):
        with self.assertRaises(InputError):
            unify(INT, "not a type")
        with self.assertRaises(InputError):
            unify(INT, INT, max_steps=-1)


class TestCanonicalNaming(unittest.TestCase):
    def test_first_occurrence_order(self):
        # λf.λx. f x : forall a, b. (a -> b) -> a -> b
        r = infer(Lam("f", Lam("x", App(Var("f"), Var("x")))))
        self.assertEqual(r.type_str, "forall a, b. (a -> b) -> a -> b")

    def test_closed_type_no_quantifier(self):
        r = infer(Write(Ref(Bool(False)), Bool(True)))
        self.assertEqual(r.type_str, "Unit")


if __name__ == "__main__":
    unittest.main()
