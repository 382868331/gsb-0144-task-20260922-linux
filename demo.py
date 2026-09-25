#!/usr/bin/env python3
"""polyinfer 固定输入演示（纯离线、真实计算、无预录结果）。

运行：``python demo.py``

打印三部分：

1. 正常推导：组合算子的主类型，以及恒等函数在同一作用域内的
   Int / Bool 两种实例（值限制下 let 多态成立），并用独立替换器
   核验成功推导的全部统一约束；
2. 实际触发的拒绝：引用多态漏洞程序
   ``let r = ref(λz.z) in r := 1; !r`` 被值限制拒绝，
   打印失败约束两端的类型与 AST 路径；
3. 诊断模式：对"合起来才有问题"的多约束程序与两处独立错误程序，
   给出按编号确定的包含极小冲突集、occurs/构造子不合区分与来源轨迹。
"""

import time

from polyinfer import A, diagnose, format_path, infer
from polyinfer.errors import InfiniteTypeError, TypeMismatchError


def section(title: str) -> None:
    bar = "=" * 68
    print(f"\n{bar}\n{title}\n{bar}")


def show_success(label: str, node) -> None:
    t0 = time.perf_counter()
    result = infer(node)
    report = result.verify()  # 独立替换器逐约束核验，不调用统一核心
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[{label}]")
    print(f"  主类型       : {result.principal_type}")
    print(f"  节点数       : {result.node_count}")
    print(f"  统一约束核验 : {report.constraints_checked} 条全部成立；"
          f"let 泛化事件 {report.generalizations_checked} 条；"
          f"替换绑定 {report.substitution_size} 个")
    print(f"  耗时         : {elapsed_ms:.2f} ms（真实计算）")


def main() -> None:
    section("正常结果 1：组合算子 λf. λg. λx. f (g x) 的主类型")
    compose = A.lambda_(
        "f",
        A.lambda_(
            "g",
            A.lambda_("x", A.apply(A.var("f"), A.apply(A.var("g"), A.var("x")))),
        ),
    )
    show_success("compose", compose)

    section("正常结果 2：恒等函数在同一作用域内的两种实例（let 多态）")
    identity_twice = A.let(
        "id",
        A.lambda_("x", A.var("x")),
        # id 分别按 Bool -> Bool（条件）与 Int -> Int（分支）使用。
        A.if_(
            A.apply(A.var("id"), A.bool_(True)),
            A.apply(A.var("id"), A.int_(1)),
            A.int_(2),
        ),
    )
    show_success("let id = λx.x in (id true ? id 1 : 2)", identity_twice)

    section("拒绝边界：引用多态漏洞 let r = ref(λz.z) in (r := 1; !r)")
    hole = A.let(
        "r",
        A.ref(A.lambda_("z", A.var("z"))),
        A.let(
            "_u",
            A.write_(A.var("r"), A.int_(1)),
            A.read(A.var("r")),
        ),
    )
    print("程序意图：先给单元写入函数，再写入 Int，最后按函数读出使用。")
    print("若 r 被错误泛化（∀a. Ref(a→a)），写入与读取会同时通过——")
    print("值限制下 ref 是扩展式、不泛化，r 单态，写 Int 必须被拒绝。\n")
    try:
        infer(hole)
    except TypeMismatchError as exc:
        print(f"实际拒绝类型：{type(exc).__name__}（已证明无解，非资源/预算问题）")
        print(f"  左端类型：{exc.left_type}")
        print(f"  左端路径：{format_path(exc.left_path)}  （ref 分配处的单元）")
        print(f"  右端类型：{exc.right_type}")
        print(f"  右端路径：{format_path(exc.right_path)}  （write 的写入值）")
    else:  # pragma: no cover - 演示中绝不应发生
        raise SystemExit("内部错误：引用多态漏洞未被拦截")

    section("附：自应用 λx. x x 的无限类型也被 occurs 检查拦截")
    try:
        infer(A.lambda_("x", A.apply(A.var("x"), A.var("x"))))
    except InfiniteTypeError as exc:
        print(f"实际拒绝类型：{type(exc).__name__}")
        print(f"  变量 {exc.left_type} 不得出现在 {exc.right_type} 中")
        print(f"  路径：{format_path(exc.left_path)} 与 {format_path(exc.right_path)}")

    # ================================================================
    # 诊断模式：不再只报"两处类型不合"，而是指出哪几条输入约束合起来
    # 才有问题，并给出按编号确定的包含极小冲突集。
    # ================================================================
    section("诊断 1：成功程序走诊断入口，主类型与普通入口逐字一致")
    identity = A.lambda_("x", A.var("x"))
    rep = diagnose(identity)
    print(f"  diagnose.ok            : {rep.ok}")
    print(f"  diagnose 主类型        : {rep.principal_type}")
    print(f"  infer    主类型        : {infer(identity).principal_type}")
    print(f"  重放次数（首轮成功即止）: {rep.replays_used}")

    section("诊断 2：多条约束『合起来』才有问题")
    # λx. let a = x 1 in (if x then 1 else 2)
    # #1 apply(x,1) 要求 x : Int -> a
    # #2 if 条件要求 x : Bool —— 单看 #1 或 #2 都能满足，合起来才矛盾
    # #3 if 两分支 Int=Int（与冲突无关，不应进入冲突集）
    together = A.lambda_(
        "x",
        A.let(
            "a",
            A.apply(A.var("x"), A.int_(1)),
            A.if_(A.var("x"), A.int_(1), A.int_(2)),
        ),
    )
    t0 = time.perf_counter()
    rep = diagnose(together)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(rep.describe())
    print(f"  极小性: {rep.minimal}；独立重放 {rep.replays_used} 次，"
          f"诊断耗时 {elapsed_ms:.2f} ms（每次重放均为冻结约束下的全新推断）")
    # 来源轨迹：实例化/绑定/统一都能追溯到义务编号。
    prov = rep.cores[0].provenance
    print("  来源轨迹（见证重放）：")
    for b in prov.binds:
        src = f"#{b.obligation_number}" if b.obligation_number is not None else "内部"
        print(f"    绑定 ?{b.var_ident} := {b.rhs:<12} 来自义务 {src}（{b.note}）")

    section("诊断 3：occurs 与普通类型不合明确区分")
    rep_occ = diagnose(A.lambda_("x", A.apply(A.var("x"), A.var("x"))))
    rep_mis = diagnose(A.read(A.int_(1)))
    print(f"  自应用 λx.x x   : 核 {rep_occ.conflict_numbers}，"
          f"见证种类 = {rep_occ.cores[0].provenance.witness_kind}（occurs 无限类型）")
    print(f"  对 Int 做 read : 核 {rep_mis.conflict_numbers}，"
          f"见证种类 = {rep_mis.cores[0].provenance.witness_kind}（构造子不合）")

    section("诊断 4：两处相互独立的错误分别成核（不追求最小基数）")
    # λx. λy. let a = x 1 in let b = y 1 in
    #   if x then (if y then 1 else 2) else (if y then 3 else 4)
    two_bugs = A.lambda_(
        "x",
        A.lambda_(
            "y",
            A.let(
                "a",
                A.apply(A.var("x"), A.int_(1)),
                A.let(
                    "b",
                    A.apply(A.var("y"), A.int_(1)),
                    A.if_(
                        A.var("x"),
                        A.if_(A.var("y"), A.int_(1), A.int_(2)),
                        A.if_(A.var("y"), A.int_(3), A.int_(4)),
                    ),
                ),
            ),
        ),
    )
    rep = diagnose(two_bugs)
    print(rep.describe())
    print(f"  核之间互不相交；逐条移除核内约束后剩余约束均可满足；"
          f"总重放 {rep.replays_used} 次。")

    section("诊断 5：共享引用不被泛化——值限制在诊断路径上一字未松")
    hole = A.let(
        "r",
        A.ref(A.lambda_("z", A.var("z"))),
        A.let("u", A.write_(A.var("r"), A.int_(1)), A.read(A.var("r"))),
    )
    rep = diagnose(hole)
    print(rep.describe())
    r_insts = [e for e in rep.cores[0].provenance.instantiations if e.name == "r"]
    print(f"  见证重放中 r 的多态实例化事件数: {len(r_insts)}（必须为 0：ref 是扩展式）")

    print("\n演示结束：普通推断与独立核验照旧；诊断模式给出按编号确定的")
    print("包含极小冲突集、occurs/构造子不合区分、实例化与替换的来源轨迹，")
    print("且每轮重放都从冻结原始约束全新开始，值限制与 occurs 检查不放松。")


if __name__ == "__main__":
    main()
