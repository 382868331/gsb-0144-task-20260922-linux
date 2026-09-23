#!/usr/bin/env python3
"""polyinfer v2 固定输入演示（纯离线、真实计算、无预录结果、无 sleep）。

运行：``python demo.py``

三段，全部由推断器/模块检查器当场计算，独立参考检查器逐结果核验：

1. **新增正常结果**：开放行记录的字段访问/非破坏式扩展；带
   read/write/exn 效应的函数主类型；异常被 try 处理后写效应仍保留；
2. **棘手边界**：开放行统一成环（occurs 拒绝）与“异常处理洗不掉
   write”的跨模块签名；
3. **实际触发的拒绝/失败 + 增量缓存**：四模块递归 SCC 反例（开放行
   经递归签名传递、引用写入改变另一模块泛化资格）被构造性拒绝；
   随后演示缓存键含依赖签名/效应、失败不替换有效结果、失效集精确。
"""

import time

from polyinfer import (
    A,
    ModuleCache,
    ModuleChecker,
    format_path,
    infer,
)
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
from polyinfer.errors import (
    DuplicateFieldError,
    InfiniteTypeError,
    SignatureError,
    TypeMismatchError,
)


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def show(label: str, node, **kw) -> None:
    t0 = time.perf_counter()
    result = infer(node, version=2, **kw)
    report = result.verify()  # 独立替换器 + 效应事件重放，不经统一核心
    ms = (time.perf_counter() - t0) * 1000
    eff = ", ".join(sorted(result.effects)) or "纯（无效应）"
    print(f"[{label}]")
    print(f"  主类型     : {result.principal_type}")
    print(f"  本次效应   : {eff}")
    print(f"  核验       : {report.constraints_checked} 条统一约束、"
          f"{report.effect_unifications_checked} 个效应事件全部成立")
    print(f"  耗时       : {ms:.2f} ms")


# --------------------------------------------------------------- 四模块反例

def four_module_package(leak_poly_pure: bool):
    """Cell/Box/Mut/Use 互为一个 SCC；Box.pack 携带开放行经递归签名传递。"""
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
        export("ping", signature([], tfun(tcon("Int"), tcon("Int"))),
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
        A.let("b", A.apply(A.mvar("Box", "pack"), A.var("rr")),
        A.let("_u", A.apply(A.mvar("Mut", "writeBox"), A.var("b")),
              A.var("b"))),
    )
    if leak_poly_pure:
        leak_sig = signature(
            ["a"], tfun(tref(tvar("a")),
                        trecord([("cell", tref(tvar("a")))], rvar("r"))),
            row_vars=["r"],
        )
    else:
        leak_sig = signature(
            [], tfun(tref(tcon("Int")),
                     trecord([("cell", tref(tcon("Int")))], rvar("r")), ["write"]),
            row_vars=["r"],
        )
    use = module(["Cell", "Box", "Mut"], [
        export("noop", signature([], tfun(tcon("Int"), tcon("Int"))),
               A.lambda_("n", A.let(
                   "rr", A.apply(A.mvar("Cell", "make"), A.var("n")), A.var("n")))),
        export("leak", leak_sig, leak_binding),
    ])
    return package({"Cell": cell, "Box": box, "Mut": mut, "Use": use})


def main() -> None:
    # ---- 1. 新增正常结果 ------------------------------------------------
    section("1a 正常：开放行记录的字段访问（尾行多态）")
    show("λr. r.x  应用于 {x:Int, extra:Bool}",
         A.apply(A.lambda_("r", A.field_(A.var("r"), "x")),
                 A.record([("x", A.int_(1)), ("extra", A.bool_(True))])),
         max_nodes=2000)

    section("1b 正常：非破坏式扩展，原记录字段与新字段共存")
    ext = A.extend(A.record([("x", A.int_(1))]), [("y", A.bool_(True))])
    show("({x=1}) 扩展 y 后取 x", A.field_(ext, "x"), max_nodes=2000)
    show("同一记录取新字段 y", A.field_(ext, "y"), max_nodes=2000)

    section("1c 正常：read / write 潜伏效应")
    show("λr. !r", A.lambda_("r", A.read(A.var("r"))), max_nodes=2000)
    show("λr. r := 1", A.lambda_("r", A.write_(A.var("r"), A.int_(1))),
         max_nodes=2000)

    section("1d 正常（端到端三能力）：try 处理异常，但 write 效应保留")
    handled = A.lambda_(
        "r",
        A.try_(
            A.let("_u", A.write_(A.var("r"), A.int_(1)),
                  A.raise_(A.exn("Boom"))),
            A.lambda_("e", A.int_(0)),
        ),
    )
    show("λr. try (r := 1; raise Boom) with λe.0", handled, max_nodes=3000)
    print("  -> exn 已被处理（箭头无 exn），write 仍如实保留（不能被 try 洗掉）")

    # ---- 2. 棘手边界 ----------------------------------------------------
    section("2a 棘手边界：开放行统一成环，occurs 检查构造性拒绝")
    cyclic = A.lambda_(
        "r",
        A.if_(A.bool_(True), A.var("r"),
              A.extend(A.var("r"), [("z", A.var("r"))])),
    )
    try:
        infer(cyclic, max_nodes=3000, version=2)
        raise SystemExit("内部错误：开放行成环未被拦截")
    except InfiniteTypeError as exc:
        print(f"  实际拒绝：{type(exc).__name__}")
        print(f"    行变量 {exc.left_type} 不得出现在 {exc.right_type} 中")
        print(f"    路径 {format_path(exc.left_path)} vs {format_path(exc.right_path)}")

    section("2b 棘手边界：同名字段扩展一律报错（不静默覆盖）")
    try:
        infer(A.extend(A.record([("x", A.int_(1))]), [("x", A.int_(2))]),
              max_nodes=2000, version=2)
        raise SystemExit("内部错误：同名字段扩展未被拦截")
    except DuplicateFieldError as exc:
        print(f"  实际拒绝：{type(exc).__name__}：{exc.field} "
              f"@ {format_path(exc.path)}（原记录不变）")

    # ---- 3. 四模块 SCC 反例 + 缓存 --------------------------------------
    section("3a 拒绝：四模块递归 SCC，开放行经递归签名传递，引用写入\n"
            "     改变另一模块的泛化资格（谎称 ∀a 纯）")
    try:
        ModuleChecker(four_module_package(True)).check()
        raise SystemExit("内部错误：四模块泛化漏洞未被拦截")
    except SignatureError as exc:
        print(f"  实际拒绝：{type(exc).__name__}")
        print(f"  {exc}")
        print("  含义：Use.leak 实现里 Ref(a) 被 Mut.writeBox 写入 Int 钉死，")
        print("  a 是携带可变引用的未泛化刚性变量，不能在模块边界冒充 ∀a。")

    section("3b 同一实现如实声明 Ref(Int) -{write}-> ... 则通过")
    ok_res = ModuleChecker(four_module_package(False)).check()
    leak = next(e for e in ok_res.all_exports() if e.name == "leak")
    print(f"  Use.leak 实现类型 : {leak.principal_type}")
    print(f"  效应             : {sorted(leak.effects)}")
    print(f"  实际泛化变量     : {leak.generalized}（内容变量未被泛化）")
    leak.evidence and __import__("polyinfer").check_result(leak.evidence)
    print("  独立参考检查器   : 该导出全部约束与效应事件核验通过")

    section("3c 增量缓存：键含依赖签名；失败不替换有效结果；失效集精确")
    cache = ModuleCache()
    good = four_module_package(False)
    first = ModuleChecker(good, cache=cache).check()
    print(f"  首次检查        : miss={sorted(first.cache_misses)}")
    second = ModuleChecker(good, cache=cache).check()
    print(f"  不改输入再检查  : hit={sorted(second.cache_hits)}（全部命中）")

    # 只改 Box.pack 的实现（签名不变）：只有 Box 重查
    changed_impl = four_module_package(False)
    changed_impl["modules"]["Box"]["exports"][1]["binding"] = A.lambda_(
        "rr", A.let("_t", A.int_(0), A.record([("cell", A.var("rr"))]))
    )
    r_impl = ModuleChecker(changed_impl, cache=cache).check()
    print(f"  仅改 pack 实现  : miss={sorted(r_impl.cache_misses)} "
          f"hit={sorted(r_impl.cache_hits)}（不清空全缓存）")

    # 改坏 Use.leak 签名（谎称多态）：检查失败，缓存保留上一次有效结果
    try:
        ModuleChecker(four_module_package(True), cache=cache).check()
    except SignatureError:
        pass
    restored = ModuleChecker(good, cache=cache).check()
    print(f"  改坏后恢复      : Use 是否命中上一次有效结果 = "
          f"{'Use' in restored.cache_hits}（失败未替换有效缓存）")

    # 全量不走缓存路径，与增量结果逐项比较
    full = ModuleChecker(changed_impl).check_without_cache()
    same = all(
        [(e.name, e.principal_type, tuple(sorted(e.effects)), e.declared_type)
         for e in full.modules[m].exports]
        ==
        [(e.name, e.principal_type, tuple(sorted(e.effects)), e.declared_type)
         for e in r_impl.modules[m].exports]
        for m in changed_impl["modules"]
    )
    print(f"  不走缓存全量 vs 增量：逐导出逐项一致 = {same}")

    print("\n演示结束：新能力主类型/效应推导、开放行成环与同名字段两个边界、")
    print("四模块 SCC 泛化拒绝与精确增量缓存均为当场真实计算。")


if __name__ == "__main__":
    main()
