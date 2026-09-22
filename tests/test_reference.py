"""独立参考检查器测试 + 良构项随机（固定种子）补充测试。

篡改测试证明检查器确实独立审了核心产出：人为破坏结果数据，
检查器必须发现而不是放行。

补充测试用测试文件内自建的“按目标类型造项”生成器（stdlib ``random``
固定种子、不接触被测核心），批量保证：凡生成的良构程序推断成功，
且成功结果全部通过独立替换器的逐约束核验。往返一致只作补充，
主类型的期望值在其余测试中手写。
"""

import random
import unittest
from dataclasses import replace

from polyinfer import A, infer
from polyinfer.reference import ReferenceCheckFailure, check_result
from polyinfer.types import BOOL, INT, TFun, TVar


class ReferenceCheckerBasicTests(unittest.TestCase):
    def test_known_good_result_passes(self):
        result = infer(
            A.let(
                "id",
                A.lambda_("x", A.var("x")),
                A.apply(A.var("id"), A.int_(1)),
            )
        )
        report = check_result(result)
        self.assertGreater(report.constraints_checked, 0)
        self.assertEqual(report.generalizations_checked, 1)

    def test_verify_alias_works(self):
        result = infer(A.lambda_("x", A.var("x")))
        self.assertEqual(result.verify().constraints_checked, 0)


class TamperDetectionTests(unittest.TestCase):
    def _identity_let_result(self):
        return infer(
            A.let(
                "id",
                A.lambda_("x", A.var("x")),
                A.apply(A.var("id"), A.int_(1)),
            )
        )

    def test_tampered_constraint_is_caught(self):
        result = self._identity_let_result()
        constraints = list(result.constraints)
        idx = 0
        bad = constraints[idx]
        constraints[idx] = replace(bad, left=BOOL)  # 原为某函数/变量类型
        tampered = replace(result, constraints=tuple(constraints))
        with self.assertRaises(ReferenceCheckFailure):
            check_result(tampered)

    def test_tampered_substitution_is_caught(self):
        result = infer(
            A.apply(A.lambda_("x", A.var("x")), A.int_(1))
        )
        # 找出一个被绑定到 Int 的变量，把它的像改成 Bool，约束必然失配。
        subst = dict(result.subst)
        target_var = next(iter(subst))
        subst[target_var] = BOOL
        tampered = replace(result, subst=subst)
        with self.assertRaises(ReferenceCheckFailure):
            check_result(tampered)

    def test_cyclic_substitution_is_caught(self):
        result = infer(
            A.apply(A.lambda_("x", A.var("x")), A.int_(1))
        )
        subst = dict(result.subst)
        v1 = TVar(9001)
        v2 = TVar(9002)
        subst[v1] = v2
        subst[v2] = v1
        tampered = replace(result, subst=subst)
        with self.assertRaises(ReferenceCheckFailure):
            check_result(tampered)

    def test_occurs_violating_binding_is_caught(self):
        result = self._identity_let_result()
        subst = dict(result.subst)
        v = TVar(9003)
        subst[v] = TFun(v, INT)  # v ↦ v -> Int，违反 occurs
        tampered = replace(result, subst=subst)
        with self.assertRaises(ReferenceCheckFailure):
            check_result(tampered)

    def test_over_generalization_is_caught(self):
        result = self._identity_let_result()
        gens = list(result.generalizations)
        g = replace(gens[0], generalized=gens[0].generalized + (TVar(9004),))
        tampered = replace(result, generalizations=(g,))
        with self.assertRaises(ReferenceCheckFailure):
            check_result(tampered)

    def test_value_restriction_lie_is_caught(self):
        # 对真正的值绑定（λx.x）谎报 bound_tag='ref'，检查器独立重判后
        # 期望泛化集合应为空，与实际非空集合不符，必须报错。
        result = self._identity_let_result()
        gens = list(result.generalizations)
        self.assertNotEqual(gens[0].generalized, ())
        g = replace(gens[0], bound_tag="ref")
        tampered = replace(result, generalizations=(g,))
        with self.assertRaises(ReferenceCheckFailure):
            check_result(tampered)


# ------------------------------------------------------------ 良构项生成器

# 生成器自用的朴素类型表示（元组），与被测库的类型对象完全无关。
# ("i",) Int；("b",) Bool；("->", a, b) 函数类型。


def _gen_term(rng, target, depth, env):
    """按目标类型 ``target`` 造一个必然类型正确的 AST（无引用片段）。"""
    tag = target[0]

    # 可终止的叶子：环境中同类型变量
    same_vars = [name for name, t in env if t == target]

    if depth <= 0:
        if tag == "i":
            return A.int_(rng.randrange(0, 10))
        if tag == "b":
            return A.bool_(rng.choice([True, False]))
        # 函数：只能造 lambda
        return A.lambda_("p", _gen_term(rng, target[2], depth - 1,
                                        env + [("p", target[1])]))

    roll = rng.random()
    if same_vars and roll < 0.30:
        return A.var(rng.choice(same_vars))

    if tag == "i":
        if roll < 0.55:
            return A.int_(rng.randrange(0, 100))
        if roll < 0.80:
            # 应用环境中的 Int 返回函数
            funs = [(n, t) for n, t in env if t[0] == "->"]
            if funs:
                name, ft = rng.choice(funs)
                return A.apply(A.var(name), _gen_term(rng, ft[1], depth - 1, env))
        # if 表达式（条件/分支递归造）
        return A.if_(
            _gen_term(rng, ("b",), depth - 1, env),
            _gen_term(rng, ("i",), depth - 1, env),
            _gen_term(rng, ("i",), depth - 1, env),
        )

    if tag == "b":
        if roll < 0.55:
            return A.bool_(rng.choice([True, False]))
        return A.if_(
            _gen_term(rng, ("b",), depth - 1, env),
            _gen_term(rng, ("b",), depth - 1, env),
            _gen_term(rng, ("b",), depth - 1, env),
        )

    # 函数类型：lambda
    name = f"p{len(env)}"
    return A.lambda_(
        name,
        _gen_term(rng, target[2], depth - 1, env + [(name, target[1])]),
    )


class GeneratedWellTypedTests(unittest.TestCase):
    def test_many_generated_well_typed_terms(self):
        rng = random.Random(20260922)  # 固定种子：确定性、离线
        targets = [
            ("i",),
            ("b",),
            ("->", ("i",), ("i",)),
            ("->", ("b",), ("i",)),
            ("->", ("->", ("i",), ("i",)), ("i",)),
            ("->", ("i",), ("->", ("b",), ("i",))),
        ]
        count = 0
        for _ in range(40):
            target = rng.choice(targets)
            term = _gen_term(rng, target, rng.randrange(2, 6), [])
            result = infer(term)  # 生成器保证良构：必须推断成功
            report = check_result(result)  # 且独立检查器逐约束放行
            self.assertGreaterEqual(report.constraints_checked, 0)
            count += 1
        self.assertEqual(count, 40)

    def test_generated_terms_with_let_id_are_polymorphic(self):
        # 在生成项外面包一层 let id = λx.x in （项内用 id 的两种实例）。
        rng = random.Random(424242)
        for _ in range(10):
            n_int = rng.randrange(0, 50)
            n_bool = rng.randrange(0, 50)
            body = A.if_(
                A.apply(A.var("id"), A.bool_(True)),
                A.apply(A.var("id"), A.int_(n_int)),
                A.int_(n_bool),
            )
            term = A.let("id", A.lambda_("x", A.var("x")), body)
            result = infer(term)
            self.assertEqual(result.principal_type, "Int")
            check_result(result)


if __name__ == "__main__":
    unittest.main()
