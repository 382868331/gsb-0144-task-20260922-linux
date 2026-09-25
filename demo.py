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
3. 诊断模式（:func:`polyinfer.diagnose`）：失败时不再只给两个位置，
   而是给出按稳定编号确定的包含极小冲突集——演示“多条约束合起来
   才有问题”、两处独立错误，以及 occurs 检查与普通类型不合的区分。
"""

import time

from polyinfer import A, diagnose, format_diagnosis, format_path, infer
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

    # ------------------------------------------------------------ 诊断
    section("诊断模式：λf. if f then (f 1) else 2 —— 哪几条约束合起来错")
    multi = A.lambda_(
        "f",
        A.if_(A.var("f"), A.apply(A.var("f"), A.int_(1)), A.int_(2)),
    )
    print("f 在条件里被当作 Bool，在应用里被当作 Int -> a；")
    print("两处单独都能成立，是 #0 与 #1 合起来把 f 逼成矛盾类型。\n")
    t0 = time.perf_counter()
    diag = diagnose(multi)
    ms = (time.perf_counter() - t0) * 1000
    print(format_diagnosis(diag))
    print(f"  诊断总耗时 : {ms:.2f} ms（真实冻结重放）")

    section("诊断模式：两处独立错误分别报出（if 条件错 + 分支不合）")
    two_errors = A.if_(A.int_(1), A.int_(2), A.bool_(False))
    diag2 = diagnose(two_errors)
    print(format_diagnosis(diag2))

    section("诊断模式：自应用的证据是 occurs 检查失败，不是构造子不合")
    diag3 = diagnose(A.lambda_("x", A.apply(A.var("x"), A.var("x"))))
    print(format_diagnosis(diag3))

    print("\n演示结束：正常主类型、引用漏洞/自应用拒绝、以及诊断模式的")
    print("稳定编号、包含极小冲突集（多条合起来才有问题）、两处独立错误")
    print("与 occurs/构造子不合的区分均为真实计算结果。")


if __name__ == "__main__":
    main()
