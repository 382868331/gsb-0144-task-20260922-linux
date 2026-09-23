"""模块系统测试：递归 SCC、两阶段签名检查、效应参与泛化、
模块边界引用不逃逸、精确缓存失效与非缓存路径逐项比对。

所有成功结果都经独立模块检查器 ``check_module_report`` 审计。
"""

import copy
import unittest

from polyinfer import A
from polyinfer.errors import (
    EffectMismatchError,
    RecordFieldError,
    SkolemEscapeError,
    TypeMismatchError,
)
from polyinfer.modules import ModuleChecker, check_modules
from polyinfer.reference import check_module_report


def check_ok(modules, changed=None):
    checker, report = check_modules(modules, changed=changed)
    check_module_report(report)
    return checker, report


class SignatureParsingTests(unittest.TestCase):
    def test_basic_signatures(self):
        from polyinfer.signature import parse_signature

        cases = {
            "forall a. a -> a": "forall a. a -> a",
            "forall a r. {f: a ; r} -> a": "forall a, r. {f: a ; r} -> a",
            "Int -{W}-> Unit": "Int -{W}-> Unit",
            "forall a. Ref(a) -{R,W}-> a": "forall a. Ref(a) -{R,W}-> a",
            "{a: Int, b: Bool}": "{a: Int, b: Bool}",
        }
        for text, rendered in cases.items():
            self.assertEqual(parse_signature(text).render(), rendered)

    def test_free_var_must_be_quantified(self):
        from polyinfer.signature import SignatureError

        with self.assertRaises(SignatureError):
            check_ok({"M": A.make_module("M", {"f": "a -> a"},
                                         {"f": A.lambda_("x", A.var("x"))})})

    def test_bad_effect_label(self):
        from polyinfer.signature import SignatureError

        with self.assertRaises(SignatureError):
            check_ok({"M": A.make_module("M", {"f": "Int -{Z}-> Int"},
                                         {"f": A.lambda_("x", A.var("x"))})})


class BasicModuleTests(unittest.TestCase):
    def test_two_modules_polymorphic_import(self):
        M = A.make_module("M", {"id": "forall a. a -> a"},
                          {"id": A.lambda_("x", A.var("x"))})
        N = A.make_module("N", {"use": "Int -> Int"},
                          {"use": A.lambda_("n", A.apply(A.xvar("M", "id"), A.var("n")))},
                          imports=("M",))
        _, report = check_ok({"M": M, "N": N})
        self.assertEqual(report.sccs, (("M",), ("N",)))
        self.assertEqual(len(report.exports), 2)

    def test_every_export_needs_signature(self):
        from polyinfer.errors import InvalidAstError

        # 导出在 body 但没有签名。
        M = A.make_module("M", {"id": "forall a. a -> a"},
                          {"id": A.lambda_("x", A.var("x")),
                           "extra": A.int_(1)})
        M["exports"].clear()
        with self.assertRaises(InvalidAstError):
            check_ok({"M": M})

    def test_too_specific_implementation_rejected(self):
        M = A.make_module("M", {"f": "forall a. a -> a"},
                          {"f": A.lambda_("x", A.if_(A.var("x"), A.var("x"), A.var("x")))})
        with self.assertRaises(TypeMismatchError):
            check_ok({"M": M})

    def test_effect_in_signature_must_match_implementation(self):
        # 签名声称纯，实现有写效应：拒绝（写入参与泛化/一致性判断）。
        M = A.make_module("M", {"g": "Int -> Int"},
                          {"g": A.lambda_("x", A.let(
                              "_u", A.write_(A.ref(A.int_(0)), A.int_(1)),
                              A.var("x")))})
        with self.assertRaises(EffectMismatchError):
            check_ok({"M": M})

    def test_effectful_signature_accepted_when_matches(self):
        M = A.make_module("M", {"g": "Int -{W}-> Int"},
                          {"g": A.lambda_("x", A.let(
                              "_u", A.write_(A.ref(A.int_(0)), A.int_(1)),
                              A.var("x")))})
        _, report = check_ok({"M": M})
        self.assertEqual(report.exports[0].signature, "Int -{W}-> Int")

    def test_handled_exception_still_write_in_signature(self):
        # 异常被处理但写效应保留：实现签名必须带 W（不能按纯函数导出）。
        M = A.make_module("M", {"h": "Int -{W}-> Int"},
                          {"h": A.lambda_("x", A.try_(
                              A.raise_(A.int_(1)), "e",
                              A.let("_u", A.write_(A.ref(A.int_(0)), A.int_(2)),
                                    A.var("e"))))})
        _, report = check_ok({"M": M})
        check_module_report(report)
        # 反方向：去掉 W 的纯签名必须失败。
        M_bad = A.make_module("M", {"h": "Int -> Int"}, M["body"])
        with self.assertRaises(EffectMismatchError):
            check_ok({"M": M_bad})


class SharedReferenceTests(unittest.TestCase):
    def test_two_modules_share_one_reference(self):
        # A 内部持有一个顶层共享单元，导出 read/write 两个函数；
        # B 经这两个导出共享同一个引用（写入影响读取）。
        A_mod = A.make_module(
            "A",
            {"put": "Int -{W}-> Unit", "get": "Int -{R}-> Int"},
            {
                "cell": A.ref(A.int_(0)),
                "put": A.lambda_("_", A.write_(A.var("cell"), A.var("_"))),
                "get": A.lambda_("_", A.read(A.var("cell"))),
            },
        )
        B_mod = A.make_module(
            "B",
            {"run": "Int -{R,W}-> Int"},
            {"run": A.lambda_(
                "x",
                A.let("_u", A.apply(A.xvar("A", "put"), A.var("x")),
                      A.apply(A.xvar("A", "get"), A.int_(0))))},
            imports=("A",),
        )
        _, report = check_ok({"A": A_mod, "B": B_mod})
        self.assertEqual(report.sccs, (("A",), ("B",)))

    def test_shared_cell_not_generalized_across_boundary(self):
        # A 持有单态单元却试图导出多态访问：拒绝（携带可变引用的类型变量
        # 不能被模块边界泛化）。
        A_mod = A.make_module(
            "A",
            {"get": "forall a. Unit -{R}-> a"},
            {
                "cell": A.ref(A.evar()),
                "get": A.lambda_("_", A.read(A.var("cell"))),
            },
        )
        with self.assertRaises(SkolemEscapeError):
            check_ok({"A": A_mod})


class FourModuleCounterexampleTests(unittest.TestCase):
    """跨四模块反例：开放行经递归签名 A→B→C→D 传递，D 的引用写入若把
    多态开放行固定为闭合具体形状，就改变了 A 导出签名的泛化资格——
    必须在签名检查阶段拒绝。

    对开放行多态单元唯一健全的写入是“读后写回同一内容”（保留行尾）；
    写入闭合具体记录会同时把 a 钉成具体类型、把 r 钉成空尾。
    """

    SIG = "forall a r. Ref({f: a ; r}) -{R,W}-> Unit"

    def _bundle(self, d_act):
        return {
            "A": A.make_module("A", {"share": self.SIG},
                {"share": A.lambda_("r", A.if_(
                    A.bool_(True),
                    A.apply(A.xvar("B", "step"), A.var("r")),
                    A.apply(A.xvar("D", "act"), A.var("r"))))}),
            "B": A.make_module("B", {"step": self.SIG},
                # 内部纯值定义 pid 在模块内按 ∀a 泛化（产生可审计泛化事件）。
                {"pid": A.lambda_("z", A.var("z")),
                 "step": A.lambda_("r", A.apply(
                     A.var("pid"), A.apply(A.xvar("C", "go"), A.var("r"))))},
                imports=("C",)),
            "C": A.make_module("C", {"go": self.SIG},
                {"go": A.lambda_("r", A.apply(A.xvar("D", "act"), A.var("r")))},
                imports=("D",)),
            # D 回指 A：A→B→C→D→A 构成四模块 SCC。
            "D": A.make_module("D", {"act": self.SIG}, {"act": d_act},
                imports=("A",)),
        }

    def test_scc_has_four_modules(self):
        # 健全实现：读出原内容再写回（R,W），a 与 r 都保持多态；
        # 另一分支递归调用 A.share（同签名）。
        good_act = A.lambda_("r", A.if_(
            A.bool_(True),
            A.let("v", A.read(A.var("r")), A.write_(A.var("r"), A.var("v"))),
            A.apply(A.xvar("A", "share"), A.var("r"))))
        _, report = check_ok(self._bundle(good_act))
        self.assertEqual(report.sccs, (("A", "B", "C", "D"),))
        # 成功结果附可独立检查的统一约束与效应推导：四个导出潜在效应 {R,W}。
        mrep = check_module_report(report)
        self.assertGreater(mrep.constraints_checked, 0)
        self.assertEqual(mrep.module_exports_checked, 4)
        # B 的内部纯值定义 pid 产生至少一条泛化事件（模块层 ∀ 泛化可审计）。
        self.assertGreaterEqual(mrep.generalizations_checked, 1)
        for er in report.exports:
            self.assertEqual(set(er.effects), set())  # 体效应收为潜在效应
        for mt in report.traces:
            for es in mt.export_effects:
                self.assertEqual(set(es.signed_effects), {"R", "W"})
                self.assertEqual(set(es.observed_effects), {"R", "W"})
        # 同 SCC 内任一改动的精确失效集覆盖全部四个模块。
        self.assertEqual(
            set(ModuleChecker(self._bundle(good_act)).invalidation_set("D")),
            {"A", "B", "C", "D"},
        )

    def test_write_fixing_open_row_is_rejected(self):
        # D.act 读一下（保留 R 以匹配效应），却写入闭合的 {f: Int}：
        # a=Int、r=空，与刚性签名 $a/$r 冲突——引用写入改变了 A 的泛化资格。
        bad_act = A.lambda_("r", A.if_(
            A.bool_(True),
            A.let("_", A.read(A.var("r")),
                  A.write_(A.var("r"), A.record([("f", A.int_(1))]))),
            A.apply(A.xvar("A", "share"), A.var("r"))))
        with self.assertRaises((TypeMismatchError, RecordFieldError)):
            check_ok(self._bundle(bad_act))

    def test_write_extra_field_is_rejected(self):
        # 写入含额外字段 g 的闭合记录：同样破坏 r 的开放多态。
        bad_act = A.lambda_("r", A.if_(
            A.bool_(True),
            A.let("_", A.read(A.var("r")),
                  A.write_(A.var("r"),
                           A.record([("f", A.int_(1)), ("g", A.bool_(True))]))),
            A.apply(A.xvar("A", "share"), A.var("r"))))
        with self.assertRaises((TypeMismatchError, RecordFieldError)):
            check_ok(self._bundle(bad_act))

    def test_closed_cell_export_open_row_rejected(self):
        # 另一条路径：直接导出一个闭合单元却声称开放行多态引用。
        M = A.make_module("M", {"cell": "forall r. Ref({f: Int ; r})"},
                          {"cell": A.ref(A.record([("f", A.int_(1))]))})
        with self.assertRaises((RecordFieldError, SkolemEscapeError)):
            check_ok({"M": M})


class InternalRecursionTests(unittest.TestCase):
    def test_internal_def_cycle_rejected_without_signature(self):
        from polyinfer.modules import ModuleStructureError

        # 一个合法导出 + 两个相互递归、却没有签名的内部定义：必须拒绝。
        M = A.make_module(
            "M", {"k": "Int"},
            {
                "k": A.int_(0),
                "v": A.apply(A.var("w"), A.int_(1)),
                "w": A.lambda_("x", A.apply(A.var("v"), A.var("x"))),
            },
        )
        with self.assertRaises(ModuleStructureError):
            check_ok({"M": M})


# ---------------------------------------------------------------- 缓存

class CacheTests(unittest.TestCase):
    def _valid_bundle(self):
        return {
            "A": A.make_module("A", {"id": "forall a. a -> a"},
                               {"id": A.lambda_("x", A.var("x"))}),
            "B": A.make_module("B", {"f": "Int -> Int"},
                               {"f": A.lambda_("x", A.apply(A.xvar("A", "id"), A.var("x")))},
                               imports=("A",)),
            "C": A.make_module("C", {"g": "Int -> Int"},
                               {"g": A.lambda_("x", A.apply(A.xvar("B", "f"), A.var("x")))},
                               imports=("B",)),
            "D": A.make_module("D", {"k": "Bool -> Bool"},
                               {"k": A.lambda_("b", A.var("b"))}),
        }

    def test_first_check_stores_all_second_hits_all(self):
        checker, rep1 = check_ok(self._valid_bundle())
        self.assertEqual(set(rep1.cache_stored), {"A", "B", "C", "D"})
        rep2 = checker.check_all(changed=set())
        self.assertEqual(set(rep2.cache_hits), {"A", "B", "C", "D"})
        self.assertEqual(rep2.cache_stored, ())

    def test_invalidation_set_is_reverse_dependency_closure(self):
        checker = ModuleChecker(self._valid_bundle())
        checker._build_graph()
        # 改 A：A、B、C 失效，D 不失效（与实际约束依赖一致）。
        self.assertEqual(set(checker.invalidation_set("A")), {"A", "B", "C"})
        self.assertEqual(set(checker.invalidation_set("B")), {"B", "C"})
        self.assertEqual(set(checker.invalidation_set("D")), {"D"})

    def test_impl_change_invalidates_exact_set_not_all(self):
        bundle = self._valid_bundle()
        checker, _ = check_ok(bundle)
        bundle["A"]["body"]["id"] = A.lambda_("y", A.var("y"))  # 签名不变
        checker.modules = bundle
        rep = checker.check_all(changed={"A"})
        self.assertEqual(set(rep.cache_stored), {"A", "B", "C"})
        self.assertEqual(rep.cache_hits, ("D",))

    def test_signature_effect_change_invalidates_dependents(self):
        # 给 B 的签名加 R：B 与其依赖者 C 的键都因签名指纹变化失效；
        # 但 B 实现无 R，检查失败。
        bundle = self._valid_bundle()
        checker, _ = check_ok(bundle)
        bad = copy.deepcopy(bundle)
        bad["B"]["exports"]["f"] = "Int -{R}-> Int"
        checker.modules = bad
        with self.assertRaises(EffectMismatchError):
            checker.check_all(changed={"B"})

    def test_failed_edit_keeps_last_good_result(self):
        bundle = self._valid_bundle()
        checker, _ = check_ok(bundle)
        good_key = checker._cache["A"][0]
        bad = copy.deepcopy(bundle)
        # 实现具体（Bool 特化）却保留多态签名。
        bad["A"]["body"]["id"] = A.lambda_("x", A.if_(A.var("x"), A.var("x"), A.var("x")))
        checker.modules = bad
        with self.assertRaises(TypeMismatchError):
            checker.check_all(changed={"A"})
        # 上一次有效结果必须原样保留（不被失败替换）。
        self.assertEqual(checker._cache["A"][0], good_key)
        self.assertEqual(checker._cache["A"][1][0].principal_type, "a -> a")
        # 恢复合法包：全部命中，无需重算。
        checker.modules = bundle
        rep = checker.check_all(changed=set())
        self.assertEqual(set(rep.cache_hits), {"A", "B", "C", "D"})


class NonCacheParityTests(unittest.TestCase):
    """非缓存检查路径 vs 缓存增量路径：对小但完整的程序逐项比较。"""

    def _bundle(self):
        sig = "forall a r. {f: a ; r} -> a"
        mods = {
            "A": A.make_module(
                "A", {"pick": sig},
                {"pick": A.lambda_("x", A.if_(
                    A.bool_(True),
                    A.get(A.var("x"), "f"),
                    A.apply(A.xvar("C", "sel"), A.var("x"))))}),
            "B": A.make_module(
                "B", {"sel4": sig},
                {"sel4": A.lambda_("x", A.if_(
                    A.bool_(True),
                    A.apply(A.xvar("A", "pick"), A.var("x")),
                    A.get(A.var("x"), "f")))}),
            "C": A.make_module(
                "C", {"sel": sig},
                {"sel": A.lambda_("x", A.if_(
                    A.bool_(False),
                    A.get(A.var("x"), "f"),
                    A.apply(A.xvar("D", "tag"), A.var("x"))))}),
            "D": A.make_module(
                "D", {"tag": sig},
                {"tag": A.lambda_(
                    "x", A.apply(A.xvar("B", "sel4"), A.var("x")))}),
        }
        return mods

    def test_incremental_matches_full_item_by_item(self):
        bundle = self._bundle()
        # 全量（非缓存）路径。
        full = ModuleChecker(bundle).check_all(use_cache=False)
        # 增量缓存路径（先全量建立，再零改动重查）。
        checker = ModuleChecker(bundle)
        inc = checker.check_all(use_cache=True)
        again = checker.check_all(changed=set())

        self.assertEqual(full.sccs, inc.sccs, again.sccs)
        self.assertEqual(full.order, inc.order)
        for ef, ei, ea in zip(full.exports, inc.exports, again.exports):
            self.assertEqual(ef.module, ei.module, ea.module)
            self.assertEqual(ef.export, ei.export, ea.export)
            self.assertEqual(ef.signature, ei.signature, ea.signature)
            self.assertEqual(ef.principal_type, ei.principal_type,
                             ea.principal_type)
            self.assertEqual(ef.effects, ei.effects, ea.effects)
            self.assertEqual(ef.effects_undecided, ei.effects_undecided,
                             ea.effects_undecided)
            self.assertEqual(ef.node_count, ei.node_count, ea.node_count)
        # 零改动后缓存路径必须全部命中，且结果与全量逐项一致。
        self.assertEqual(set(again.cache_hits), set(bundle))

    def test_partial_invalidation_recheck_matches_full(self):
        bundle = self._bundle()
        checker = ModuleChecker(bundle)
        checker.check_all()
        # 改 D 实现（保持签名）：失效集 {A,B,C,D}（都在 SCC 内）。
        bundle["D"]["body"]["tag"] = A.lambda_(
            "x", A.apply(A.xvar("B", "sel4"), A.var("x"))
        )
        inc = checker.check_all(changed={"D"})
        full = ModuleChecker(copy.deepcopy(bundle)).check_all(use_cache=False)
        for ef, ei in zip(full.exports, inc.exports):
            self.assertEqual(ef.principal_type, ei.principal_type)
            self.assertEqual(ef.signature, ei.signature)


class EndToEndFeatureTests(unittest.TestCase):
    """一个端到端例子同时使用三组新能力：开放行记录（含字段访问/扩展）、
    R/W/X 三种效应（异常被处理后写效应保留）、跨模块引用与显式签名。"""

    def test_open_rows_effects_modules_together(self):
        store = A.make_module(
            "Store",
            {
                # 每次调用新建单元：开放行多态健全。
                "mk": "forall a r. {f: a ; r} -> Ref({f: a ; r})",
                # 读标志、读单元；标志为真时抛出记录（X 携带开放行记录）。
                "operate": "forall a r. Ref({f: a ; r}) -{R,X}-> {f: a ; r}",
                "bump": "Int -{W}-> Unit",
            },
            {
                "flag": A.ref(A.bool_(False)),
                "mk": A.lambda_("x", A.ref(A.var("x"))),
                "bump": A.lambda_("_", A.write_(A.var("flag"), A.bool_(True))),
                "operate": A.lambda_(
                    "c",
                    A.if_(
                        A.read(A.var("flag")),
                        A.raise_(A.read(A.var("c"))),
                        A.read(A.var("c")),
                    )
                ),
            },
        )
        app = A.make_module(
            "App",
            {
                # 字段投影（开放行）；扩展结果在 run 中使用。
                "peek": "forall a r. {f: a ; r} -> a",
                # bump 产生 W；operate 产生 R 并可能抛 X；try 处理 X 后
                # 处理器把记录写回新单元（W），再扩展出 ok 字段并投影 f。
                "run": "forall a r. {f: a ; r} -{R,W}-> a",
            },
            {
                "peek": A.lambda_("r", A.get(A.var("r"), "f")),
                "run": A.lambda_(
                    "rec",
                    A.let(
                        "c",
                        A.apply(A.xvar("Store", "mk"), A.var("rec")),
                        A.let(
                            "_u",
                            A.apply(A.xvar("Store", "bump"), A.int_(1)),
                            A.let(
                                "got",
                                A.try_(
                                    A.apply(A.xvar("Store", "operate"), A.var("c")),
                                    "e",
                                    # X 被处理：写回（W 保留）并返回记录。
                                    A.let(
                                        "_w",
                                        A.write_(A.var("c"), A.var("e")),
                                        A.var("e"),
                                    ),
                                ),
                                A.get(
                                    A.extend(A.var("got"), "ok", A.bool_(True)),
                                    "f",
                                ),
                            ),
                        ),
                    ),
                ),
            },
            imports=("Store",),
        )
        _, report = check_ok({"Store": store, "App": app})
        check_module_report(report)
        eff = {
            (er.module, er.export): set(es.signed_effects)
            for er in report.exports for es in [next(
                x for x in next(
                    mt.export_effects for mt in report.traces
                    if mt.module == er.module
                ) if x.export == er.export
            )]
        }
        self.assertEqual(eff[("Store", "operate")], {"R", "X"})
        self.assertEqual(eff[("Store", "bump")], {"W"})
        self.assertEqual(eff[("App", "run")], {"R", "W"})  # X 已被处理
        # 若 App.run 的签名漏掉 W（声称纯），必须失败——异常处理不洗白写。
        bad_app = A.make_module("App", dict(app["exports"],
                                            **{"run": "forall a r. {f: a ; r} -> a"}),
                                app["body"], imports=("Store",))
        with self.assertRaises(EffectMismatchError):
            check_ok({"Store": store, "App": bad_app})


class ScaleTests(unittest.TestCase):
    """主规模：≥12 模块、≥3000 AST 节点、每记录 20 字段、4 模块 SCC。"""

    def test_main_scale_completes_and_verifies(self):
        FIELDS = 20
        mods = {}
        sig = "forall a r. {" + ", ".join(f"f{i}: a" for i in range(FIELDS)) + " ; r} -> a"
        cyc = ["S0", "S1", "S2", "S3"]
        for i, m in enumerate(cyc):
            nxt = cyc[(i + 1) % 4]
            mods[m] = A.make_module(m, {"pick": sig},
                {"pick": A.lambda_("x", A.if_(A.bool_(True),
                    A.get(A.var("x"), f"f{i % FIELDS}"),
                    A.apply(A.xvar(nxt, "pick"), A.var("x"))))})

        def big(n):
            e = A.int_(1)
            for _ in range(n):
                e = A.if_(A.bool_(True), e, A.int_(2))
            return e

        prev = "S3"
        for j in range(8):
            m = f"L{j}"
            mods[m] = A.make_module(m, {"v": "Int"},
                {"v": big(130), "helper": big(1)}, imports=(prev,))
            prev = m

        def count(n):
            if isinstance(n, dict) and "tag" in n:
                return 1 + sum(count(v) for v in n.values())
            if isinstance(n, list):
                return sum(count(x) for x in n)
            return 0

        total = sum(c for m in mods.values() for e in m["body"].values() for c in [count(e)])
        checker, report = check_modules(mods)
        check_module_report(report)
        self.assertGreaterEqual(report.modules, 12)
        self.assertGreaterEqual(total, 3000)
        self.assertEqual(report.sccs[0], ("S0", "S1", "S2", "S3"))


if __name__ == "__main__":
    unittest.main()
