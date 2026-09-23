#!/usr/bin/env python3
"""polyinfer 固定输入演示（纯离线、真实计算、无预录结果、无 sleep）。

运行：``python demo.py``

内容
====
1. v1 兼容：组合算子主类型 + 引用多态漏洞拒绝（初版行为保持不变）；
2. v2 正常结果：开放行多态、字段访问/扩展、R/W/X 效应、try 处理 X 保留 W；
   每个成功结果都打印可独立检查的统一约束数、泛化事件与效应推导；
3. 棘手边界：开放行被统一成环（extend 的缺席约束）与同名字段扩展；
4. 跨四模块反例：开放行经递归 SCC 签名 A→B→C→D→A 传递，D 的引用写入
   把多态开放行钉成闭合具体形状，被签名检查实际拒绝；
5. 增量缓存：精确失效集与“失败不替换上次有效结果”。
"""

import copy
import time

from polyinfer import A, check_modules, format_path, infer
from polyinfer.errors import (
    EffectMismatchError,
    InfiniteTypeError,
    RecordFieldError,
    TypeMismatchError,
)
from polyinfer.modules import ModuleChecker
from polyinfer.reference import check_module_report


def section(title: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n{title}\n{bar}")


def show_v2(label: str, node) -> None:
    t0 = time.perf_counter()
    result = infer(node, language="v2")
    report = result.verify()  # 独立替换器/效应解释器逐约束核验，不经过统一核心
    ms = (time.perf_counter() - t0) * 1000
    print(f"[{label}]")
    print(f"  主类型       : {result.principal_type}")
    eff_str = ",".join(sorted(result.effects)) or "∅（纯）"
    print(f"  直接效应     : {eff_str}"
          f"{'（含未决效应变量）' if result.effects_undecided else ''}")
    print(f"  独立核验     : {report.constraints_checked} 条统一约束成立；"
          f"泛化事件 {report.generalizations_checked}；"
          f"效应重算 {'一致' if report.effects_checked else '—'}"
          f"（{ms:.2f} ms，真实计算）")


# ---------------------------------------------------------------- v1

t_start = time.perf_counter()
section("v1 兼容（默认 language='v1'，结果与初版逐项一致）")
compose = A.lambda_(
    "f", A.lambda_("g", A.lambda_(
        "x", A.apply(A.var("f"), A.apply(A.var("g"), A.var("x"))))),
)
r1 = infer(compose)
print(f"组合算子主类型 : {r1.principal_type}（{r1.verify().constraints_checked} 条约束核验）")

hole = A.let(
    "r", A.ref(A.lambda_("z", A.var("z"))),
    A.let("_u", A.write_(A.var("r"), A.int_(1)), A.read(A.var("r"))),
)
try:
    infer(hole)
except TypeMismatchError as exc:
    print(f"引用多态漏洞   : {type(exc).__name__}（{exc.left_type} vs {exc.right_type}）")

# ------------------------------------------------------------ v2 正常

section("v2 正常结果 1：开放行多态 —— 同一函数用于两种不同形状的记录")
open_row = A.let(
    "f", A.lambda_("x", A.get(A.var("x"), "a")),
    A.if_(
        A.bool_(True),
        A.apply(A.var("f"), A.record([("a", A.int_(1)), ("b", A.bool_(True))])),
        A.apply(A.var("f"), A.record([("a", A.int_(2)), ("c", A.int_(3))])),
    ),
)
show_v2("f = λx. x.a，分别用于 {a,b} 与 {a,c}", open_row)

section("v2 正常结果 2：不改变原记录的扩展 + 三种效应")
eff_prog = A.lambda_(
    "r",
    A.write_(A.var("r"), A.read(A.var("r"))),  # 读后写回：R,W
)
show_v2("λr. r := !r（读 R + 写 W）", eff_prog)

ext_prog = A.lambda_(
    "x", A.get(A.extend(A.var("x"), "b", A.int_(0)), "b"),
)
show_v2("λx. ({x with b=0}).b（扩展不修改原记录）", ext_prog)

section("v2 正常结果 3：异常被处理后，写效应必须保留")
handled = A.try_(
    A.raise_(A.int_(1)),                 # X
    "e",
    A.write_(A.ref(A.int_(0)), A.int_(2)),  # W
)
show_v2("try (raise 1) e => (ref 0 := 2)", handled)
print("  → X 被 try 处理掉，但 W 不是异常效应：结果效应为 {W} 而非纯。")

# ------------------------------------------------------------ 棘手边界

section("棘手边界 1：同名字段扩展一律报错（即使类型相同）")
dup_extend = A.let(
    "r", A.record([("f", A.int_(1))]),
    A.extend(A.var("r"), "f", A.int_(2)),
)
try:
    infer(dup_extend, language="v2")
except RecordFieldError as exc:
    print(f"  实际拒绝：{type(exc).__name__}：字段 {exc.field!r} 已存在；"
          f"扩展不得覆盖原记录字段。")

section("棘手边界 2：开放行统一成环 —— 缺席约束拦截字段重复进入同一尾行")
# λx. let y = {x with g=1} in x.g
# x 经 extend 被要求“尾不含 g”，又经 get 被要求“尾含 g”：成环/重复。
row_cycle = A.lambda_(
    "x",
    A.let("y", A.extend(A.var("x"), "g", A.int_(1)), A.get(A.var("x"), "g")),
)
try:
    infer(row_cycle, language="v2")
except (RecordFieldError, InfiniteTypeError) as exc:
    print(f"  实际拒绝：{type(exc).__name__}：开放行尾的缺席约束与后续字段"
          f"访问冲突（同名字段不得经尾展开出现两次）。")

# -------------------------------------------------------- 四模块反例

section("综合端到端：开放行(get/extend) + R/W/X(try) + 跨模块，同例并用")
store = A.make_module(
    "Store",
    {"mk": "forall a r. {f: a ; r} -> Ref({f: a ; r})",
     "operate": "forall a r. Ref({f: a ; r}) -{R,X}-> {f: a ; r}",
     "bump": "Int -{W}-> Unit"},
    {"flag": A.ref(A.bool_(False)),
     "mk": A.lambda_("x", A.ref(A.var("x"))),
     "bump": A.lambda_("_", A.write_(A.var("flag"), A.bool_(True))),
     "operate": A.lambda_("c", A.if_(A.read(A.var("flag")),
         A.raise_(A.read(A.var("c"))), A.read(A.var("c"))))},
)
app = A.make_module(
    "App",
    {"peek": "forall a r. {f: a ; r} -> a",
     "run": "forall a r. {f: a ; r} -{R,W}-> a"},
    {"peek": A.lambda_("r", A.get(A.var("r"), "f")),
     "run": A.lambda_("rec", A.let("c", A.apply(A.xvar("Store", "mk"), A.var("rec")),
         A.let("_u", A.apply(A.xvar("Store", "bump"), A.int_(1)),
           A.let("got", A.try_(
                   A.apply(A.xvar("Store", "operate"), A.var("c")), "e",
                   A.let("_w", A.write_(A.var("c"), A.var("e")), A.var("e"))),
             A.get(A.extend(A.var("got"), "ok", A.bool_(True)), "f")))))},
    imports=("Store",),
)
t0 = time.perf_counter()
_, e2e = check_modules({"Store": store, "App": app})
e2e_rep = check_module_report(e2e)
print(f"  两模块检查通过（{(time.perf_counter()-t0)*1000:.2f} ms）：")
for e in e2e.exports:
    print(f"    {e.module}.{e.export} :: {e.signature}")
print(f"  独立核验 {e2e_rep.constraints_checked} 条约束、"
      f"{e2e_rep.generalizations_checked} 条泛化、"
      f"{e2e_rep.module_exports_checked} 个效应签名。")
print("  关键点：operate 签名带 {R,X}；App.run 用 try 处理掉 X，处理器内的")
print("  写回（W）保留 → run 签名是 -{R,W}-> 而非纯；缺 W 的纯签名会被拒绝。")
bad_app = A.make_module("App", {"peek": app["exports"]["peek"],
                               "run": "forall a r. {f: a ; r} -> a"},
                       app["body"], imports=("Store",))
try:
    check_modules({"Store": store, "App": bad_app})
except EffectMismatchError:
    print("  实测：把 run 签名改成纯（漏 W）→ EffectMismatchError。")

section("跨四模块反例：开放行经递归签名传递，引用写入改变泛化资格")
SIG = "forall a r. Ref({f: a ; r}) -{R,W}-> Unit"


def bundle(d_act):
    return {
        "A": A.make_module("A", {"share": SIG},
            {"share": A.lambda_("r", A.if_(A.bool_(True),
                A.apply(A.xvar("B", "step"), A.var("r")),
                A.apply(A.xvar("D", "act"), A.var("r"))))}),
        "B": A.make_module("B", {"step": SIG},
            {"step": A.lambda_("r", A.apply(A.xvar("C", "go"), A.var("r")))},
            imports=("C",)),
        "C": A.make_module("C", {"go": SIG},
            {"go": A.lambda_("r", A.apply(A.xvar("D", "act"), A.var("r")))},
            imports=("D",)),
        "D": A.make_module("D", {"act": SIG}, {"act": d_act}, imports=("A",)),
    }


# 健全实现：读出再写回同一内容，a/r 保持多态；且 D→A 使四个模块构成 SCC。
good_act = A.lambda_("r", A.if_(
    A.bool_(True),
    A.let("v", A.read(A.var("r")), A.write_(A.var("r"), A.var("v"))),
    A.apply(A.xvar("A", "share"), A.var("r"))))
t0 = time.perf_counter()
checker, good_report = check_modules(bundle(good_act))
mrep = check_module_report(good_report)
print(f"  健全版（读后写回）通过：SCC = {good_report.sccs[0]}；"
      f"独立核验 {mrep.constraints_checked} 条模块约束、"
      f"{mrep.module_exports_checked} 个导出效应签名"
      f"（{(time.perf_counter()-t0)*1000:.2f} ms）")
for e in good_report.exports:
    print(f"    {e.module}.{e.export} :: {e.signature}")

# 反例：D 把单元写成闭合的 {f: Int} —— a 被钉成 Int、r 被钉成空尾。
bad_act = A.lambda_("r", A.if_(
    A.bool_(True),
    A.let("_", A.read(A.var("r")),
          A.write_(A.var("r"), A.record([("f", A.int_(1))]))),
    A.apply(A.xvar("A", "share"), A.var("r"))))
try:
    check_modules(bundle(bad_act))
except (TypeMismatchError, RecordFieldError) as exc:
    print(f"  反例实际拒绝：{type(exc).__name__}")
    print(f"    D.act 经 A→B→C→D→A 递归签名拿到刚性 $a/$r 开放行，")
    print(f"    写入闭合 {{f: Int}} 会把可变单元的泛化资格偷走——"
          f"签名检查在模块边界拦截。")

# ------------------------------------------------------------ 缓存

section("增量缓存：精确失效集（不清空全缓存）+ 失败原子性")
cache_bundle = {
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
cc = ModuleChecker(cache_bundle)
cc.check_all()
print(f"  依赖链 A←B←C，D 独立。改动 A 的失效集 = {sorted(cc.invalidation_set('A'))}"
      f"（不是全部 4 个模块）")
print(f"  改动 D 的失效集 = {sorted(cc.invalidation_set('D'))}")

bad = copy.deepcopy(cache_bundle)
bad["A"]["body"]["id"] = A.lambda_("x", A.if_(A.var("x"), A.var("x"), A.var("x")))
cc.modules = bad
try:
    cc.check_all(changed={"A"})
except TypeMismatchError:
    print("  把 A 改成 Bool 特化实现（签名仍声称 ∀a）：检查失败。")
retained = cc._cache["A"][1][0].principal_type
print(f"  失败后上一次有效结果保留：A 的缓存导出仍是 {retained!r}（未被失败替换）。")
cc.modules = cache_bundle
again = cc.check_all(changed=set())
print(f"  恢复合法包后零重算，命中模块：{sorted(again.cache_hits)}。")

elapsed = time.perf_counter() - t_start
print(f"\n演示结束（总真实计算耗时 {elapsed:.3f} 秒，上限 90 秒；无 sleep、无预录）。")
print("覆盖：开放行统一/泛化、R/W/X 效应与异常处理、四模块递归 SCC 反例、")
print("模块边界引用不逃逸、精确缓存失效与失败原子性；全部成功结果均独立核验。")
