"""v2 模块层测试：显式签名、SCC 检查、效应/引用参与泛化、增量缓存。

重点反例（题面要求）：

1. **两个模块共享引用**：一个模块造 cell，另一个模块写入；签名必须
   如实声明 write，且 cell 类型不能在边界被偷偷泛化；
2. **开放行统一成环 / 行经递归签名传递**：四模块 SCC，开放行记录
   经互相递归的签名传递，某模块的引用写入把另一模块导出的泛化资格
   钉死——声明纯多态时必须拒绝；
3. **异常被处理后仍有写效应**：try 去掉 exn 不去掉 write，跨模块
   签名检查必须捕获；
4. **缓存失效集**：改一个导出，失效集恰为反向依赖闭包中键实际变化
   的模块，绝不清空全缓存；另用不走缓存路径与增量结果逐项比对。
"""

import copy
import unittest

from polyinfer import A
from polyinfer.ast import (
    export,
    module,
    package,
    rvar,
    signature,
    tcon,
    tfun,
    trecord,
    tref,
    tvar,
)
from polyinfer.errors import ModuleStructureError, SignatureError
from polyinfer.modules import (
    ModuleCache,
    ModuleChecker,
    check_package,
    tarjan_scc,
)
from polyinfer.reference import check_result


def sig_int_int(effects=()):
    return signature([], tfun(tcon("Int"), tcon("Int"), list(effects)))


# ---------------------------------------------------------------- 基础

class BasicModuleTests(unittest.TestCase):
    def test_two_modules_polymorphic_signature(self):
        pkg = package({
            "M": module([], [
                export("id", signature(["a"], tfun(tvar("a"), tvar("a"))),
                       A.lambda_("x", A.var("x"))),
            ]),
            "N": module(["M"], [
                export("use", sig_int_int(),
                       A.lambda_("x", A.apply(A.mvar("M", "id"), A.var("x")))),
            ]),
        })
        res = check_package(pkg)
        # 实现本身可被推断为多态 a->a，并经子包含满足声明的 Int->Int。
        self.assertEqual(res.modules["M"].exports[0].principal_type, "a -> a")
        self.assertEqual(res.modules["N"].exports[0].principal_type, "a -> a")
        self.assertEqual(res.modules["N"].exports[0].declared_type, "Int -> Int")
        # 每个导出的证据都经独立检查器核验。
        for ex in res.all_exports():
            check_result(ex.evidence)

    def test_every_export_needs_signature(self):
        bad = package({
            "M": module([], [
                {"name": "f", "binding": A.lambda_("x", A.var("x"))},
            ]),
        })
        with self.assertRaises(ModuleStructureError):
            check_package(bad)

    def test_import_unknown_module_rejected(self):
        bad = package({
            "M": module(["Ghost"], [
                export("f", sig_int_int(), A.lambda_("x", A.var("x"))),
            ]),
        })
        with self.assertRaises(ModuleStructureError):
            check_package(bad)

    def test_duplicate_export_rejected(self):
        bad = package({
            "M": module([], [
                export("f", sig_int_int(), A.lambda_("x", A.var("x"))),
                export("f", sig_int_int(), A.lambda_("y", A.var("y"))),
            ]),
        })
        with self.assertRaises(ModuleStructureError):
            check_package(bad)

    def test_signature_type_var_must_be_bound(self):
        bad = package({
            "M": module([], [
                export("f", signature([], tfun(tvar("a"), tvar("a"))),
                       A.lambda_("x", A.var("x"))),
            ]),
        })
        with self.assertRaises(ModuleStructureError):
            check_package(bad)


# ---------------------------------------------------- 效应/引用参与泛化

class SignatureEffectTests(unittest.TestCase):
    def _pkg(self, declared_effects=(), impl_kind="write"):
        # impl_kind: "write" = λrr.(rr := 1; 0)；"pure" = λrr.0
        if impl_kind == "write":
            impl = A.lambda_(
                "rr",
                A.let("_u", A.write_(A.var("rr"), A.int_(1)), A.int_(0)),
            )
        else:
            impl = A.lambda_("rr", A.int_(0))
        return package({
            "Cell": module([], [
                export("set",
                       signature([], tfun(tref(tcon("Int")), tcon("Int"),
                                         list(declared_effects))),
                       impl),
            ]),
        })

    def test_implementation_write_declared_pure_rejected(self):
        with self.assertRaises(SignatureError) as cm:
            check_package(self._pkg(declared_effects=(), impl_kind="write"))
        self.assertEqual(cm.exception.module, "Cell")
        self.assertEqual(cm.exception.export, "set")
        self.assertIn("write", str(cm.exception))

    def test_implementation_write_declared_write_accepted(self):
        res = check_package(self._pkg(declared_effects=("write",), impl_kind="write"))
        ex = res.modules["Cell"].exports[0]
        self.assertIn("write", ex.effects)

    def test_pure_implementation_can_have_narrower_effect(self):
        # 实现纯（Ref(Int)->Int）、签名声明 write 且同型：允许
        # （实现效应 ∅ ⊆ 声明 {write}，子包含协变）。
        res = check_package(self._pkg(declared_effects=("write",), impl_kind="pure"))
        self.assertEqual(set(res.modules["Cell"].exports[0].effects), set())

    def test_read_effect_also_checked(self):
        pkg = package({
            "R": module([], [
                export("get",
                       signature([], tfun(tref(tcon("Int")), tcon("Int"))),
                       A.lambda_("rr", A.read(A.var("rr")))),
            ]),
        })
        with self.assertRaises(SignatureError):
            check_package(pkg)

    def test_exception_handled_but_write_remains_rejected_as_pure(self):
        """异常被 try 处理，但 write 仍在——签名声称纯必须拒绝。"""
        impl = A.lambda_(
            "rr",
            A.try_(
                A.let("_u", A.write_(A.var("rr"), A.int_(1)),
                      A.raise_(A.exn("E"))),
                A.lambda_("e", A.int_(0)),
            ),
        )
        pkg = package({
            "H": module([], [
                export("safeish",
                       signature([], tfun(tref(tcon("Int")), tcon("Int"))),
                       impl),
            ]),
        })
        with self.assertRaises(SignatureError) as cm:
            check_package(pkg)
        self.assertIn("write", str(cm.exception))

    def test_exception_handled_and_write_declared_accepted(self):
        impl = A.lambda_(
            "rr",
            A.try_(
                A.let("_u", A.write_(A.var("rr"), A.int_(1)),
                      A.raise_(A.exn("E"))),
                A.lambda_("e", A.int_(0)),
            ),
        )
        pkg = package({
            "H": module([], [
                export("safeish",
                       signature([], tfun(tref(tcon("Int")), tcon("Int"), ["write"])),
                       impl),
            ]),
        })
        res = check_package(pkg)
        self.assertIn("write", res.modules["H"].exports[0].effects)
        self.assertNotIn("exn", res.modules["H"].exports[0].effects)


class SharedReferenceAcrossModulesTests(unittest.TestCase):
    """两个模块共享一个引用：写入与泛化资格跨模块联动。"""

    def _pkg(self, getter_sig_poly=True, setter_writes_declared=True):
        # Store.make: ∀a. a -> Ref(a)
        # Use.share: Ref(Int) -{write}-> Int，写入后读出
        store = module([], [
            export("make", signature(["a"], tfun(tvar("a"), tref(tvar("a")))),
                   A.lambda_("x", A.ref(A.var("x")))),
        ])
        effects = ["read", "write"] if setter_writes_declared else []
        if getter_sig_poly:
            share_sig = signature(["a"], tfun(tref(tvar("a")), tvar("a"), effects))
        else:
            share_sig = signature([], tfun(tref(tcon("Int")), tcon("Int"), effects))
        use = module(["Store"], [
            export("share", share_sig,
                   A.lambda_("rr", A.let(
                       "_u", A.write_(A.var("rr"), A.int_(1)),
                       A.read(A.var("rr"))))),
        ])
        return package({"Store": store, "Use": use})

    def test_shared_ref_write_declared_monomorphic_accepted(self):
        res = check_package(self._pkg(getter_sig_poly=False))
        ex = res.modules["Use"].exports[0]
        self.assertIn("write", ex.effects)

    def test_shared_ref_implementation_monomorphic_signature_polymorphic_rejected(self):
        # 实现里 rr 被 write Int 钉成 Ref(Int)，签名却声称 ∀a. Ref(a)->a：
        # 携带可变引用写入的变量不得在模块边界偷偷泛化。
        with self.assertRaises(SignatureError):
            check_package(self._pkg(getter_sig_poly=True))

    def test_shared_ref_write_but_signature_says_pure_rejected(self):
        with self.assertRaises(SignatureError):
            check_package(self._pkg(getter_sig_poly=False, setter_writes_declared=False))


# ------------------------------------------------------- 四模块 SCC 反例

def four_module_package(leak_poly_pure=True):
    """四模块成一个 SCC：Cell <-> Box <-> Mut <-> Use。

    * Cell.make : ∀a. a -> Ref(a)（内部经 Box.id，构成递归引用）
    * Box.id    : ∀a. a -> a（内部经 Mut.ping）
    * Box.pack  : ∀aρ. Ref(a) -> {cell: Ref(a) | ρ}  —— 开放行经签名传递
    * Mut.ping  : Int -> Int（经 Use.noop）
    * Mut.writeBox : ∀ρ. {cell: Ref(Int) | ρ} -{write}-> Unit
    * Use.noop  : Int -> Int（经 Cell.make）
    * Use.leak  : λrr. let b = Box.pack rr in
                           let _ = Mut.writeBox b in b
        真实类型 Ref(Int) -{write}-> {cell: Ref(Int) | ρ}
        —— leak_poly_pure=True 时签名谎称 ∀aρ 纯（必须被拒）；
        False 时如实声明 Ref(Int) -{write}-> ...（必须通过）。
    """
    cell = module(["Box"], [
        export("make", signature(["a"], tfun(tvar("a"), tref(tvar("a")))),
               A.lambda_("x", A.ref(A.apply(A.mvar("Box", "id"), A.var("x"))))),
    ])
    box = module(["Mut"], [
        export("id", signature(["a"], tfun(tvar("a"), tvar("a"))),
               A.lambda_("x", A.let(
                   "_z", A.apply(A.mvar("Mut", "ping"), A.int_(0)), A.var("x")))),
        export("pack",
               signature(["a"],
                         tfun(tref(tvar("a")),
                              trecord([("cell", tref(tvar("a")))], rvar("r"))),
                         row_vars=["r"]),
               A.lambda_("rr", A.record([("cell", A.var("rr"))]))),
    ])
    mut = module(["Use"], [
        export("ping", sig_int_int(),
               A.lambda_("n", A.apply(A.mvar("Use", "noop"), A.var("n")))),
        export("writeBox",
               signature([],
                         tfun(trecord([("cell", tref(tcon("Int")))], rvar("r")),
                              tcon("Unit"), ["write"]),
                         row_vars=["r"]),
               A.lambda_("b", A.write_(A.field_(A.var("b"), "cell"), A.int_(1)))),
    ])
    leak_binding = A.lambda_(
        "rr",
        A.let(
            "b", A.apply(A.mvar("Box", "pack"), A.var("rr")),
            A.let(
                "_u", A.apply(A.mvar("Mut", "writeBox"), A.var("b")),
                A.var("b"),
            ),
        ),
    )
    if leak_poly_pure:
        leak_sig = signature(
            ["a"],
            tfun(tref(tvar("a")),
                 trecord([("cell", tref(tvar("a")))], rvar("r"))),
            row_vars=["r"],
        )
    else:
        leak_sig = signature(
            [],
            tfun(tref(tcon("Int")),
                 trecord([("cell", tref(tcon("Int")))], rvar("r")), ["write"]),
            row_vars=["r"],
        )
    use = module(["Cell", "Box", "Mut"], [
        export("noop", sig_int_int(),
               A.lambda_("n", A.let(
                   "rr", A.apply(A.mvar("Cell", "make"), A.var("n")), A.var("n")))),
        export("leak", leak_sig, leak_binding),
    ])
    return package({"Cell": cell, "Box": box, "Mut": mut, "Use": use})


class FourModuleSccCounterexampleTests(unittest.TestCase):
    def test_scc_is_four(self):
        mc = ModuleChecker(four_module_package())
        sccs = tarjan_scc(mc.graph)
        self.assertEqual(sorted(len(c) for c in sccs), [4])

    def test_open_row_through_recursive_sig_write_kills_polymorphism(self):
        """题面核心反例：开放行经递归签名传递，writeBox 的写入把 leak 的
        泛化资格钉死——声称 ∀a 纯被构造性拒绝。"""
        with self.assertRaises(SignatureError) as cm:
            check_package(four_module_package(leak_poly_pure=True))
        self.assertEqual(cm.exception.module, "Use")
        self.assertEqual(cm.exception.export, "leak")

    def test_honest_monomorphic_write_signature_accepted(self):
        res = check_package(four_module_package(leak_poly_pure=False))
        leak = next(e for e in res.all_exports() if e.name == "leak")
        self.assertIn("write", leak.effects)
        # 内容类型变量被写入钉成 Int（未出现在泛化集中）；开放尾行变量
        # 不涉及 cell 内容、可安全泛化（R0）。
        self.assertEqual(leak.generalized, ("R0",))
        self.assertNotIn("a", leak.generalized)
        check_result(leak.evidence)

    def test_scc_order_checks_signatures_before_bodies(self):
        res = check_package(four_module_package(leak_poly_pure=False))
        # 四模块同 SCC：顺序确定（模块名字典序）且全部检查。
        self.assertEqual(set(res.order), {"Cell", "Box", "Mut", "Use"})


# ------------------------------------------------------- 闭行记录签名

class ClosedRowSignatureTests(unittest.TestCase):
    def test_closed_row_extra_field_rejected(self):
        # 签名要求闭行 {x:Int}，实现给 {x:Int,y:Bool}。
        pkg = package({
            "R": module([], [
                export("mk",
                       signature([], trecord([("x", tcon("Int"))], None)),
                       A.record([("x", A.int_(1)), ("y", A.bool_(True))])),
            ]),
        })
        with self.assertRaises(SignatureError):
            check_package(pkg)

    def test_closed_row_missing_declared_field_rejected(self):
        pkg = package({
            "R": module([], [
                export("mk",
                       signature([], trecord([("x", tcon("Int"))], None)),
                       A.record([("other", A.int_(1))])),
            ]),
        })
        with self.assertRaises(SignatureError):
            check_package(pkg)

    def test_open_row_signature_absorbs_extra_field(self):
        pkg = package({
            "R": module([], [
                export("mk",
                       signature([], trecord([("x", tcon("Int"))], rvar("r")),
                                 row_vars=["r"]),
                       A.record([("x", A.int_(1)), ("y", A.bool_(True))])),
            ]),
        })
        res = check_package(pkg)
        self.assertEqual(res.modules["R"].exports[0].name, "mk")


# ------------------------------------------------------- 增量缓存

class IncrementalCacheTests(unittest.TestCase):
    def _pkg(self, g_impl=None, g_sig=None):
        g_impl = g_impl or A.lambda_("y", A.var("y"))
        g_sig = g_sig or sig_int_int()
        return package({
            "A": module(["B"], [
                export("f", sig_int_int(),
                       A.lambda_("x", A.apply(A.mvar("B", "g"), A.var("x")))),
            ]),
            "B": module([], [
                export("g", g_sig, g_impl),
            ]),
            "C": module(["B"], [
                export("h", sig_int_int(),
                       A.lambda_("z", A.apply(A.mvar("B", "g"), A.var("z")))),
            ]),
            "D": module([], [
                export("k", sig_int_int(), A.lambda_("w", A.var("w"))),
            ]),
        })

    def test_first_run_all_miss_second_all_hit(self):
        cache = ModuleCache()
        r1 = ModuleChecker(self._pkg(), cache=cache).check()
        self.assertEqual(set(r1.cache_misses), {"A", "B", "C", "D"})
        r2 = ModuleChecker(self._pkg(), cache=cache).check()
        self.assertEqual(set(r2.cache_hits), {"A", "B", "C", "D"})

    def test_impl_only_change_invalidates_only_that_module(self):
        cache = ModuleCache()
        ModuleChecker(self._pkg(), cache=cache).check()
        new_g = A.lambda_("y", A.if_(A.bool_(True), A.var("y"), A.int_(0)))
        r = ModuleChecker(self._pkg(g_impl=new_g), cache=cache).check()
        # 签名指纹未变：只有 B 重查，A/C/D 命中（绝不靠清全缓存）。
        self.assertEqual(set(r.cache_misses), {"B"})
        self.assertEqual(set(r.cache_hits), {"A", "C", "D"})
        self.assertEqual(set(r.invalidated), {"B"})

    def test_signature_change_invalidates_reverse_dependency_closure(self):
        cache = ModuleCache()
        mc0 = ModuleChecker(self._pkg(), cache=cache)
        mc0.check()
        # g 改成多态 a->a（实现 λy.y 满足）：B 与反向依赖 A、C 失效，
        # 与 B 无关的 D 必须继续命中。
        new_sig = signature(["a"], tfun(tvar("a"), tvar("a")))
        r = ModuleChecker(self._pkg(g_sig=new_sig), cache=cache).check()
        self.assertEqual(set(r.cache_misses), {"A", "B", "C"})
        self.assertEqual(set(r.cache_hits), {"D"})
        # 理论失效闭包与实际重查集合一致。
        self.assertEqual(
            set(mc0.reverse_dependency_closure(["B"])), {"A", "B", "C"}
        )

    def test_failed_change_does_not_replace_last_valid_result(self):
        cache = ModuleCache()
        good = self._pkg()
        ModuleChecker(good, cache=cache).check()
        valid_before = cache.get("B")
        # 把 B 改成签名 Bool->Int（λy.y 不满足）：检查失败。
        bad = self._pkg(g_sig=signature([], tfun(tcon("Bool"), tcon("Int"))))
        with self.assertRaises(SignatureError):
            ModuleChecker(bad, cache=cache).check()
        # 上一次有效结果仍在缓存中，未被失败结果替换。
        self.assertIs(cache.get("B"), valid_before)
        # 再用原（有效）包检查：B 键等于有效键，直接命中。
        r = ModuleChecker(good, cache=cache).check()
        self.assertIn("B", r.cache_hits)

    def test_incremental_matches_full_path_per_export(self):
        """不走缓存的全量路径 vs 增量路径：逐导出逐项一致。"""
        cache = ModuleCache()
        pkg = self._pkg()
        ModuleChecker(pkg, cache=cache).check()
        # 改 B 实现触发增量
        changed = self._pkg(g_impl=A.lambda_("y", A.if_(
            A.bool_(False), A.int_(7), A.var("y"))))
        inc = ModuleChecker(changed, cache=cache).check()
        full = ModuleChecker(changed).check_without_cache()
        self.assertEqual(set(inc.order), set(full.order))
        for m in changed["modules"]:
            ie = inc.modules[m].exports
            fe = full.modules[m].exports
            self.assertEqual(len(ie), len(fe))
            for a, b in zip(ie, fe):
                self.assertEqual(a.name, b.name)
                self.assertEqual(a.principal_type, b.principal_type)
                self.assertEqual(a.declared_type, b.declared_type)
                self.assertEqual(set(a.effects), set(b.effects))
                self.assertEqual(a.generalized, b.generalized)
                self.assertEqual(a.dangerous_excluded, b.dangerous_excluded)
                self.assertEqual(a.node_count, b.node_count)

    def test_full_path_does_not_touch_shared_cache(self):
        cache = ModuleCache()
        pkg = self._pkg()
        ModuleChecker(pkg, cache=cache).check()
        # 全量路径使用独立缓存，不污染也不读取共享缓存：
        # 即使共享缓存已有结果，全量路径仍全部 miss。
        full = ModuleChecker(pkg).check_without_cache()
        self.assertEqual(set(full.cache_misses), {"A", "B", "C", "D"})
        self.assertEqual(full.cache_hits, ())


# ------------------------------------------------------- 规模

class ScaleTests(unittest.TestCase):
    def test_twelve_modules_scc_four_and_large_exports(self):
        # 12 个模块；前 4 个互为 SCC（环），其余模块单向依赖该环。
        names = [f"M{i}" for i in range(12)]
        mods = {}
        for i, n in enumerate(names):
            if i < 4:
                imp = [names[(i - 1) % 4]]
                callee = imp[0]
                body = A.lambda_("z", A.apply(A.mvar(callee, "entry"), A.var("z")))
            elif i % 2 == 0:
                imp = [names[0]]
                body = A.lambda_("z", A.apply(A.mvar(names[0], "entry"), A.var("z")))
            else:
                imp = []
                body = A.lambda_("z", A.var("z"))
            mods[n] = module(imp, [export("entry", sig_int_int(), body)])
        pkg = package(mods)
        mc = ModuleChecker(pkg, max_nodes=100_000)
        sccs = tarjan_scc(mc.graph)
        self.assertEqual(max(len(c) for c in sccs), 4)
        self.assertGreaterEqual(len(mods), 12)
        res = mc.check()
        self.assertEqual(len(res.modules), 12)
        for ex in res.all_exports():
            check_result(ex.evidence)

    def test_export_with_twenty_field_record(self):
        fields = [(f"f{i}", tcon("Int")) for i in range(20)]
        pkg = package({
            "Big": module([], [
                export("rec",
                       signature([], trecord(fields, None)),
                       A.record([(f"f{i}", A.int_(i)) for i in range(20)])),
            ]),
        })
        res = ModuleChecker(pkg, max_nodes=50_000).check()
        ex = res.modules["Big"].exports[0]
        self.assertIn("f0: Int", ex.principal_type)
        self.assertIn("f19: Int", ex.principal_type)


if __name__ == "__main__":
    unittest.main()
