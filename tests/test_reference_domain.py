"""确定性小参考域的完整核对。

用固定种子枚举一批 v2 小程序（限定在独立效应解释器可完整判定的子集：
无跨模块 xvar、无“未知高阶返回值被当函数调用”），逐条比较：

* 推断器报告的具体效应集合 == 独立效应解释器重算值；
* 未决标志一致（含 evar 即未决）；
* 成功结果全部通过独立检查器。

参考解释器**不调用**推断器；这里以它为期望值来源。
"""

import random
import unittest

from polyinfer import A, infer
from polyinfer.reference import ref_effects_of


# ------------------------------------------------------------ 项生成器
# 自用类型轨道：("i",) Int / ("b",) Bool / ("u",) Unit
# 生成器维护 env: 名字 -> (类型轨道, 是否带效应 lambda 标记)

class RefCaseGenerator:
    def __init__(self, rng):
        self.rng = rng

    def gen(self, target, depth, env, allow_evar):
        """造一个可类型化、效应可独立判定的项。"""
        rng = self.rng
        # 终止：同类型变量
        same = [n for n, (t, _e) in env if t == target]
        if depth <= 0:
            if same and rng.random() < 0.4:
                return A.var(rng.choice(same))
            return self._ground(target)

        roll = rng.random()
        if same and roll < 0.25:
            return A.var(rng.choice(same))
        if roll < 0.45:
            return self._ground(target)
        if roll < 0.6:
            # if：条件 Bool，两支同目标类型
            return A.if_(
                self.gen(("b",), depth - 1, env, allow_evar),
                self.gen(target, depth - 1, env, allow_evar),
                self.gen(target, depth - 1, env, allow_evar),
            )
        if roll < 0.72:
            # try：body 用 raise 造 X，handler 给目标类型（body 整体也归 target）
            return A.try_(
                self.gen(("i",), depth - 1, env, allow_evar)
                if rng.random() < 0.5
                else A.raise_(self.gen(("i",), depth - 1, env, allow_evar)),
                "e",
                self.gen(target, depth - 1, env + [("e", (("i",), False))], allow_evar),
            )
        if roll < 0.82:
            # 对同类型变量调用一个环境里的函数（beta 内联可判定）
            funs = [(n, t) for n, (t, _e) in env
                    if isinstance(t, tuple) and len(t) == 3 and t[0] == "->"
                    and t[2] == target]
            if funs:
                n, ft = rng.choice(funs)
                return A.apply(A.var(n), self.gen(ft[1], depth - 1, env, allow_evar))
            return self._ground(target)
        if roll < 0.9 and allow_evar:
            # 只在目标允许时放 evar（其类型自由，这里仅在非严格轨道出现）
            if target == ("i",):
                return A.evar()
        return self._ground(target)

    def _ground(self, target):
        rng = self.rng
        if target == ("i",):
            return A.int_(rng.randrange(0, 100))
        if target == ("b",):
            return A.bool_(rng.choice([True, False]))
        if target == ("u",):
            return A.write_(A.ref(A.int_(0)), A.int_(0))
        if isinstance(target, tuple) and len(target) == 3 and target[0] == "->":
            pname = f"p{len(self._param_seen)}"
            self._param_seen += 1
            body = self.gen(target[2], self._depth - 1,
                            self._env + [(pname, (target[1], False))],
                            self._allow_evar)
            return A.lambda_(pname, body)
        return A.int_(0)

    def gen_function(self, depth, env, argt, rett, allow_evar):
        """专门造一个 lambda（体内可含 R/W/X/try）。"""
        pname = f"fn{len(env)}"
        old = (self._param_seen, self._depth, self._env, self._allow_evar)
        self._param_seen += 1
        self._depth = depth
        self._env = env + [(pname, (argt, False))]
        self._allow_evar = allow_evar
        body = self.gen(rett, depth, self._env, allow_evar)
        self._param_seen, self._depth, self._env, self._allow_evar = old
        return A.lambda_(pname, body)

    def reset(self):
        self._param_seen = 0
        self._depth = 0
        self._env = []
        self._allow_evar = False


def _wrap_with_let_functions(body, fns):
    """把生成的 lambda 用非递归 let 串起来（内层先绑定）。"""
    node = body
    for name, lam in reversed(fns):
        node = A.let(name, lam, node)
    return node


class ReferenceDomainTests(unittest.TestCase):
    def test_effects_match_independent_reference_many_programs(self):
        rng = random.Random(20260924)
        gen = RefCaseGenerator(rng)
        checked = 0
        undecided_seen = 0
        for _ in range(300):
            gen.reset()
            # 造 0~2 个辅助函数（Int->Int / Int->Bool），再造主体
            fns = []
            env = []
            allow_evar = rng.random() < 0.25
            for k in range(rng.randrange(0, 3)):
                argt = rng.choice([("i",), ("b",)])
                rett = rng.choice([("i",), ("b",)])
                name = f"f{k}"
                lam = gen.gen_function(rng.randrange(1, 3), env, argt, rett, allow_evar)
                fns.append((name, lam))
                env.append((name, (("->", argt, rett), False)))
            target = rng.choice([("i",), ("b",)])
            gen._env = env
            body = gen.gen(target, rng.randrange(1, 4), env, allow_evar)
            program = _wrap_with_let_functions(body, fns)
            # 部分生成项可能不可类型化（生成器只尽力保证）；不可类型化的跳过，
            # 可类型化的才是参考域核对对象。
            try:
                result = infer(program, language="v2")
            except Exception:
                continue
            # 独立期望值（完全不经过推断器）。
            ref_labels, ref_und = ref_effects_of(program)
            self.assertEqual(
                set(result.effects), set(ref_labels),
                f"效应不一致：{result.principal_type} "
                f"核心={sorted(result.effects)} 参考={sorted(ref_labels)}",
            )
            self.assertEqual(
                bool(result.effects_undecided), bool(ref_und),
                f"未决标志不一致：核心={result.effects_undecided} 参考={ref_und}",
            )
            # 独立检查器（含效应、缺席、约束、泛化）必须全部放行。
            result.verify()
            checked += 1
            undecided_seen += int(ref_und)
        # 必须实际覆盖到一批良构项，且未决与确定两类都出现。
        self.assertGreaterEqual(checked, 100)
        self.assertGreater(undecided_seen, 0)

    def test_determined_small_domain_exact_values(self):
        """手写的确定性小参考例子：期望值逐一固定，不用 unknown 搪塞。"""
        cases = [
            (A.int_(1), frozenset(), False),
            (A.read(A.ref(A.int_(1))), frozenset({"R"}), False),
            (A.write_(A.ref(A.int_(1)), A.int_(2)), frozenset({"W"}), False),
            (A.raise_(A.int_(1)), frozenset({"X"}), False),
            (A.try_(A.raise_(A.int_(1)), "e", A.int_(0)), frozenset(), False),
            (A.try_(A.raise_(A.int_(1)), "e",
                     A.write_(A.ref(A.int_(0)), A.int_(1))),
             frozenset({"W"}), False),
            (A.evar(), frozenset(), True),
        ]
        for node, want_eff, want_und in cases:
            labels, und = ref_effects_of(node)
            self.assertEqual((labels, und), (want_eff, want_und))
            r = infer(node, language="v2")
            self.assertEqual((r.effects, r.effects_undecided),
                             (want_eff, want_und))


if __name__ == "__main__":
    unittest.main()
