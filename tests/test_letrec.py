"""互相递归绑定组 ``letrec`` 的行为测试。

覆盖：单函数自递归、互相调用、书写顺序无关、组内独立成员多态、
组内类型冲突定位、危险变量规则（引用不能借递归组多态逃逸）、
普通非递归 ``let`` 语义不变、结构校验与独立检查器篡改检测。
"""

import unittest
from dataclasses import replace

from polyinfer import A, infer
from polyinfer.errors import (
    InvalidAstError,
    TypeMismatchError,
    UnboundVariableError,
)
from polyinfer.reference import ReferenceCheckFailure, check_result


def infer_ok(node):
    result = infer(node)
    report = result.verify()
    return result, report


# 互相调用的两个函数（语言里没有算术，用 Bool 条件模拟奇偶递归）：
# f(x) = if x then false else g(x)
# g(x) = if x then true  else f(x)
F_BODY = A.if_(
    A.var("x"),
    A.bool_(False),
    A.apply(A.var("g"), A.var("x")),
)
G_BODY = A.if_(
    A.var("x"),
    A.bool_(True),
    A.apply(A.var("f"), A.var("x")),
)
F = A.lambda_("x", F_BODY)
G = A.lambda_("x", G_BODY)


class MutualRecursionTests(unittest.TestCase):
    def test_mutual_call_succeeds(self):
        program = A.letrec(
            [("f", F), ("g", G)], A.apply(A.var("f"), A.bool_(True))
        )
        result, report = infer_ok(program)
        self.assertEqual(result.principal_type, "Bool")
        # 两个成员各一条泛化留痕，且都被独立检查器认可。
        self.assertEqual(report.generalizations_checked, 2)
        names = {g.name for g in result.generalizations}
        self.assertEqual(names, {"f", "g"})

    def test_single_self_recursion_succeeds(self):
        loop = A.lambda_(
            "b",
            A.if_(A.var("b"), A.bool_(False),
                  A.apply(A.var("loop"), A.var("b"))),
        )
        result, _ = infer_ok(A.letrec([("loop", loop)], A.var("loop")))
        self.assertEqual(result.principal_type, "Bool -> Bool")

    def test_referencing_sibling_before_definition_works(self):
        """f 在书写顺序上先于 g，但 f 的体内已经可以引用 g。"""
        program = A.letrec(
            [("f", F), ("g", G)], A.apply(A.var("g"), A.bool_(False))
        )
        result, _ = infer_ok(program)
        self.assertEqual(result.principal_type, "Bool")

    def test_order_independence_same_principal_type(self):
        """交换绑定书写顺序不得改变推断结果。"""
        body_f = A.apply(A.var("f"), A.bool_(True))
        body_g = A.apply(A.var("g"), A.bool_(True))
        r1, _ = infer_ok(A.letrec([("f", F), ("g", G)], body_f))
        r2, _ = infer_ok(A.letrec([("g", G), ("f", F)], body_f))
        r3, _ = infer_ok(A.letrec([("f", F), ("g", G)], body_g))
        self.assertEqual(r1.principal_type, r2.principal_type)
        self.assertEqual(r2.principal_type, r3.principal_type)

    def test_independent_members_are_separately_polymorphic(self):
        """不共享类型变量的 lambda 成员必须各自独立泛化。"""
        idf = A.lambda_("x", A.var("x"))
        constf = A.lambda_("a", A.lambda_("b", A.var("a")))
        body = A.if_(
            A.apply(A.var("id"), A.bool_(True)),
            A.apply(A.var("id"), A.int_(1)),
            A.apply(A.apply(A.var("const"), A.int_(2)), A.bool_(False)),
        )
        program = A.letrec([("id", idf), ("const", constf)], body)
        result, _ = infer_ok(program)
        self.assertEqual(result.principal_type, "Int")


class RecursionGroupConflictTests(unittest.TestCase):
    def test_mutual_type_conflict_is_detected(self):
        """f : Bool -> Bool 与 g : Bool -> Int 互相要求同一单态，必须冲突。"""
        g_bad = A.lambda_(
            "y",
            A.if_(A.var("y"), A.int_(1), A.apply(A.var("f"), A.var("y"))),
        )
        program = A.letrec(
            [("f", F), ("g", g_bad)], A.apply(A.var("f"), A.bool_(True))
        )
        with self.assertRaises(TypeMismatchError) as cm:
            infer(program)
        err = cm.exception
        # 路径必须下钻到真正冲突的叶子，且位于某个成员的绑定树内。
        joined = "/".join(err.left_path + err.right_path)
        self.assertIn("binds", joined)
        self.assertEqual(str(err.left_type), "Int")
        self.assertEqual(str(err.right_type), "Bool")

    def test_self_recursion_conflict_path(self):
        # loop 既被当 Bool 用（条件）又被当 Bool -> ? 之外的 Int 返回值用。
        loop = A.lambda_(
            "b",
            A.if_(A.var("b"), A.int_(1),
                  A.if_(A.apply(A.var("loop"), A.var("b")),
                        A.int_(2), A.int_(3))),
        )
        # loop b 被当 Bool 条件，而 loop 的返回值被统一成 Int：冲突。
        with self.assertRaises(TypeMismatchError):
            infer(A.letrec([("loop", loop)], A.var("loop")))


class RecursionGroupSafetyTests(unittest.TestCase):
    def test_ref_member_cannot_be_generalized(self):
        rbind = A.ref(A.lambda_("z", A.var("z")))
        hbind = A.lambda_("v", A.write_(A.var("r"), A.var("v")))
        program = A.letrec(
            [("r", rbind), ("h", hbind)],
            A.apply(A.var("h"), A.lambda_("y", A.var("y"))),
        )
        result, _ = infer_ok(program)
        # 扩展式成员 r 不泛化；且因与 r 共享单元变量，lambda 成员 h
        # 也必须保持单态（危险变量规则）。
        quant = {g.name: g.generalized for g in result.generalizations}
        self.assertEqual(quant["r"], ())
        self.assertEqual(quant["h"], ())

    def test_ref_hole_through_group_is_blocked(self):
        """经典引用漏洞经 letrec 包装仍必须被拦住。

        r = ref(\u03bbz.z); h = \u03bbv. r := v；若 h 被错误泛化，
        先 h 1 再 h(\u03bby.y) 会让同一个单元先后装入 Int 与函数。
        """
        rbind = A.ref(A.lambda_("z", A.var("z")))
        hbind = A.lambda_("v", A.write_(A.var("r"), A.var("v")))
        body = A.let(
            "u1", A.apply(A.var("h"), A.int_(1)),
            A.let(
                "u2", A.apply(A.var("h"), A.lambda_("y", A.var("y"))),
                A.apply(A.read(A.var("r")), A.bool_(True)),
            ),
        )
        with self.assertRaises(TypeMismatchError):
            infer(A.letrec([("r", rbind), ("h", hbind)], body))

    def test_non_value_member_kept_monomorphic(self):
        # 组里直接放一个扩展式成员 k = (\u03bbi.i) 1（apply），其类型 Int；
        # 另一成员 id = \u03bbx.x 独立，仍可泛化；k 自身不泛化。
        idf = A.lambda_("x", A.var("x"))
        kbind = A.apply(A.lambda_("i", A.var("i")), A.int_(7))
        program = A.letrec(
            [("id", idf), ("k", kbind)],
            A.if_(A.apply(A.var("id"), A.bool_(False)),
                  A.var("k"), A.var("k")),
        )
        result, _ = infer_ok(program)
        self.assertEqual(result.principal_type, "Int")
        quant = {g.name: g.generalized for g in result.generalizations}
        self.assertEqual(quant["k"], ())
        self.assertNotEqual(quant["id"], ())


class LetrecStructureTests(unittest.TestCase):
    def test_let_stays_non_recursive(self):
        # 普通 let 行为不变：绑定里引用自身仍是未绑定变量。
        with self.assertRaises(UnboundVariableError) as cm:
            infer(A.let("f", A.var("f"), A.int_(1)))
        self.assertEqual(cm.exception.path, ("bound",))

    def test_unbound_sibling_reference_is_caught(self):
        with self.assertRaises(UnboundVariableError) as cm:
            infer(
                A.letrec(
                    [("f", A.lambda_("x", A.apply(A.var("ghost"), A.var("x"))))],
                    A.int_(0),
                )
            )
        self.assertIn("binds", cm.exception.path)

    def test_empty_binds_rejected(self):
        with self.assertRaises(InvalidAstError):
            infer(A.letrec([], A.int_(1)))

    def test_duplicate_names_rejected(self):
        lam = A.lambda_("x", A.var("x"))
        with self.assertRaises(InvalidAstError):
            infer(A.letrec([("f", lam), ("f", lam)], A.var("f")))

    def test_binds_must_be_list(self):
        with self.assertRaises(InvalidAstError):
            infer({"tag": "letrec", "binds": 42, "body": A.int_(1)})

    def test_node_count_includes_group(self):
        loop = A.lambda_(
            "b",
            A.if_(A.var("b"), A.bool_(False),
                  A.apply(A.var("loop"), A.var("b"))),
        )
        # letrec 容器 1 + 绑定内 7（lambda/if/var/bool/apply/var/var）
        # + body 的 var(loop) = 9；binds 列表与条目 dict 不是 AST 节点。
        result, _ = infer_ok(A.letrec([("loop", loop)], A.var("loop")))
        self.assertEqual(result.node_count, 9)


class LetrecReferenceAuditTests(unittest.TestCase):
    def _group_result(self):
        rbind = A.ref(A.lambda_("z", A.var("z")))
        hbind = A.lambda_("v", A.write_(A.var("r"), A.var("v")))
        return infer(
            A.letrec(
                [("r", rbind), ("h", hbind)],
                A.apply(A.var("h"), A.lambda_("y", A.var("y"))),
            )
        )

    def test_genuine_result_passes_audit(self):
        result = self._group_result()
        report = check_result(result)
        self.assertEqual(report.generalizations_checked, 2)

    def test_lie_about_group_over_generalization_is_caught(self):
        # 篡改：把与扩展式成员共享变量的 h 谎报为已泛化，
        # 独立检查器必须凭 group_members 自行重算并拒绝。
        result = self._group_result()
        gens = list(result.generalizations)
        h = next(g for g in gens if g.name == "h")
        from polyinfer.types import free_tvars

        stolen = next(v for v in free_tvars(h.bound_type))
        bad = replace(h, generalized=(stolen,))
        tampered = replace(result, generalizations=tuple(
            bad if g.name == "h" else g for g in gens
        ))
        with self.assertRaises(ReferenceCheckFailure):
            check_result(tampered)


if __name__ == "__main__":
    unittest.main()
