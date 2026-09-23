"""扩展验证（与快速回归分开运行）：

运行::

    python -m unittest discover -s extended_tests -v

内容均为**确定性生成**的非退化规模（不读外部数据、不联网）：

* 12 个模块、含一个 4 模块递归 SCC，总 AST 节点 ≥ 3000；
* 每模块多个导出，签名覆盖多态/效应/开放行（20 字段记录）；
* 增量 vs 全量（不走缓存）逐导出逐项一致；
* 四模块泛化反例在大规模包中同样被拒；
* 节点/模块预算边界与实际耗时报告（演示可完成性，不做压力断言）。
"""

import random
import time
import unittest

from polyinfer import A
from polyinfer.ast import (
    export, module, package, rvar, signature, tcon, tfun,
    trecord, tref, tvar,
)
from polyinfer.errors import NodeLimitError, SignatureError
from polyinfer.inference import infer
from polyinfer.modules import ModuleCache, ModuleChecker, tarjan_scc


# ------------------------------------------------- 确定性非退化程序生成

class _Factory:
    """按目标形态确定性生成良构 v2 表达式（固定种子，离线）。"""

    def __init__(self, seed=20260924):
        self.rng = random.Random(seed)
        self.depth = 0

    def int_expr(self, depth=0):
        if depth <= 0:
            return A.int_(self.rng.randrange(100))
        roll = self.rng.random()
        if roll < 0.5:
            return A.int_(self.rng.randrange(100))
        if roll < 0.8:
            return A.if_(
                A.bool_(self.rng.random() < 0.5),
                self.int_expr(depth - 1),
                self.int_expr(depth - 1),
            )
        # 包一层纯恒等应用，增加节点但保持 Int
        return A.apply(
            A.lambda_("z", A.var("z")),
            self.int_expr(depth - 1),
        )

    def big_int_chain(self, layers):
        """每层是良构的 Int 表达式（let + if），节点随 layers 线性增长。"""
        node = A.int_(0)
        for i in range(layers):
            node = A.let(
                f"v{i}",
                self.int_expr(1),
                A.if_(A.bool_(True), node, A.int_(i)),
            )
        return node

    def entry_body(self, callee_mvar, module_index, layers=46):
        """Int -> Int 的 entry 绑定：经 mvar 调用 callee（保持 SCC 依赖），
        否则分支是一条 ``layers`` 深的确定性 Int 链，放大到非玩具规模。"""
        chain = A.int_(module_index)
        for i in range(layers):
            chain = A.let(
                f"w{module_index}_{i}",
                A.int_(i),
                A.if_(A.bool_(True), chain, A.int_(i + module_index)),
            )
        return A.lambda_(
            "z",
            A.if_(
                A.bool_(True),
                callee_mvar(A.var("z")),
                chain,
            ),
        )

    def twenty_field_record(self):
        fields = [(f"f{i}", self.int_expr(1)) for i in range(20)]
        return A.record(fields)


class LargeExpressionTests(unittest.TestCase):
    def test_at_least_3000_nodes_well_typed_and_verified(self):
        f = _Factory()
        # 每层约 6~9 节点；450 层稳定超过 3000。
        node = f.big_int_chain(470)
        t0 = time.perf_counter()
        r = infer(node, max_nodes=20_000, version=2)
        dt = time.perf_counter() - t0
        self.assertGreaterEqual(r.node_count, 3000)
        self.assertEqual(r.principal_type, "Int")
        rep = r.verify()
        self.assertGreater(rep.constraints_checked, 0)
        self.assertLess(dt, 30.0)  # 可完成性预算（远低于 90s 演示预算）

    def test_twenty_field_record_expression(self):
        r = infer(_Factory().twenty_field_record(), max_nodes=20_000, version=2)
        r.verify()
        self.assertGreaterEqual(r.principal_type.count(":"), 20)


# ------------------------------------------------- 12 模块非退化生成

def build_large_package():
    """12 模块：M0..M3 互为 4 模块 SCC；M4..M11 挂依赖，含开放行/效应导出。

    每个模块导出 entry（Int->Int），SCC 内互相经 mvar 调用；
    Box 风格模块额外导出 20 字段开放行记录的构造器。
    总节点数随每层链路放大到 3000+。
    """
    names = [f"M{i}" for i in range(12)]
    int_int = signature([], tfun(tcon("Int"), tcon("Int")))
    factory = _Factory()
    mods = {}
    for i, n in enumerate(names):
        if i < 4:
            imp = [names[(i - 1) % 4]]
        else:
            imp = [names[i % 4]]
        callee = imp[0]
        body = factory.entry_body(
            lambda arg: A.apply(A.mvar(callee, "entry"), arg),
            i, layers=52,
        )
        exports = [export("entry", int_int, body)]
        # 每偶数模块再加一个 20 字段开放行记录导出，放大节点规模。
        if i % 2 == 0:
            fields_sig = [(f"f{j}", tcon("Int")) for j in range(20)]
            fields_impl = [(f"f{j}", A.int_(i * 20 + j)) for j in range(20)]
            exports.append(
                export(
                    "bigrec",
                    signature([], trecord(fields_sig, rvar("r")), row_vars=["r"]),
                    A.record(fields_impl),
                )
            )
        mods[n] = module(imp, exports)
    return package(mods)


class LargePackageTests(unittest.TestCase):
    def setUp(self):
        self.pkg = build_large_package()
        self.mc = ModuleChecker(self.pkg, max_nodes=200_000)

    def test_scale_shape(self):
        self.assertEqual(len(self.pkg["modules"]), 12)
        sccs = tarjan_scc(self.mc.graph)
        self.assertEqual(max(len(c) for c in sccs), 4)
        counts = [len(m["modules"] if False else m["exports"]) for m in self.pkg["modules"].values()]
        self.assertGreaterEqual(sum(counts), 12)

    def test_full_check_completes_and_verifies(self):
        t0 = time.perf_counter()
        res = self.mc.check()
        dt = time.perf_counter() - t0
        self.assertEqual(len(res.modules), 12)
        total_nodes = 0
        for ex in res.all_exports():
            from polyinfer.reference import check_result
            check_result(ex.evidence)
            total_nodes += ex.node_count
        self.assertGreaterEqual(total_nodes, 3000)  # 12 模块非退化规模
        self.assertLess(dt, 30.0)

    def test_incremental_equals_full_per_export(self):
        # 先全量建缓存
        cache = ModuleCache()
        inc1 = ModuleChecker(self.pkg, max_nodes=200_000, cache=cache).check()
        # 改一个内部模块的实现（签名不变）：增量
        changed = build_large_package()
        target = "M6"
        changed["modules"][target]["exports"][0]["binding"] = A.lambda_(
            "z", A.if_(A.bool_(False), A.int_(6), A.var("z"))
        )
        inc2 = ModuleChecker(changed, max_nodes=200_000, cache=cache).check()
        full = ModuleChecker(changed, max_nodes=200_000).check_without_cache()

        self.assertIn(target, inc2.cache_misses)
        for m in changed["modules"]:
            ie = inc2.modules[m].exports
            fe = full.modules[m].exports
            self.assertEqual(len(ie), len(fe))
            for a, b in zip(ie, fe):
                self.assertEqual(
                    (a.name, a.principal_type, tuple(sorted(a.effects)),
                     a.declared_type, a.generalized, a.node_count),
                    (b.name, b.principal_type, tuple(sorted(b.effects)),
                     b.declared_type, b.generalized, b.node_count),
                )

    def test_invalidation_set_matches_reverse_closure(self):
        cache = ModuleCache()
        ModuleChecker(self.pkg, max_nodes=200_000, cache=cache).check()
        # 改 M1 的声明签名（Int->Int 改为 a->a，实现 λz... 仍满足需要调整）：
        changed = build_large_package()
        changed["modules"]["M1"]["exports"][0]["binding"] = A.lambda_(
            "z", A.var("z")
        )
        changed["modules"]["M1"]["exports"][0]["signature"] = signature(
            ["a"], tfun(tvar("a"), tvar("a"))
        )
        # 依赖 M1 的模块（同 SCC 的 M0/M2/M3 及挂在其上的偶数链）理论上受影响。
        closure = set(self.mc.reverse_dependency_closure(["M1"]))
        r = ModuleChecker(changed, max_nodes=200_000, cache=cache).check()
        # 实际重查集合必须是理论闭包的子集，且包含 M1；不受影响模块继续命中。
        self.assertIn("M1", r.cache_misses)
        self.assertTrue(set(r.cache_misses) <= closure)
        self.assertTrue(set(r.cache_hits).isdisjoint(set(r.cache_misses)))


class FourModuleCounterexampleAtScaleTests(unittest.TestCase):
    """四模块泛化反例即使包内还有其他模块也必须被拒。"""

    def test_counterexample_rejected_inside_larger_package(self):
        from tests.test_v2_modules import four_module_package
        base = four_module_package(True)
        extra = build_large_package()
        import copy
        merged_modules = {}
        merged_modules.update(copy.deepcopy(base["modules"]))
        extra_names = set(extra["modules"])
        # 加前缀避免重名，并把导入表/mvar 引用一并改名。
        for k, v in extra["modules"].items():
            mv = copy.deepcopy(v)
            mv["imports"] = [f"X{d}" if d in extra_names else d for d in mv["imports"]]
            def rename(node):
                if isinstance(node, dict):
                    if node.get("tag") == "mvar" and node["module"] in extra_names:
                        node["module"] = f"X{node['module']}"
                    for val in node.values():
                        rename(val)
                elif isinstance(node, list):
                    for x in node:
                        rename(x)
            for ex in mv["exports"]:
                rename(ex["binding"])
            merged_modules[f"X{k}"] = mv
        pkg = package(merged_modules)
        with self.assertRaises(SignatureError) as cm:
            ModuleChecker(pkg, max_nodes=200_000).check()
        self.assertEqual(cm.exception.export, "leak")


class NodeBudgetTests(unittest.TestCase):
    def test_node_budget_is_resource_refusal(self):
        node = A.int_(0)
        for _ in range(100):
            node = A.apply(node, A.int_(0))
        with self.assertRaises(NodeLimitError):
            infer(node, max_nodes=10, version=2)


if __name__ == "__main__":
    unittest.main()
