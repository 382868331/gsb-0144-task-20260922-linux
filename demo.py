#!/usr/bin/env python3
"""polyinfer 固定输入演示（纯离线、真实计算、无预录结果）。

运行：``python demo.py``

打印两部分：

1. 正常推导：组合算子的主类型，以及恒等函数在同一作用域内的
   Int / Bool 两种实例（值限制下 let 多态成立），并用独立替换器
   核验成功推导的全部统一约束；
2. 实际触发的拒绝：引用多态漏洞程序
   ``let r = ref(λz.z) in r := 1; !r`` 被值限制拒绝，
   打印失败约束两端的类型与 AST 路径。
"""

import time

from polyinfer import A, format_path, infer
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

    section("正常结果 3：互相递归组 letrec（书写顺序不影响结论）")
    # f(x) = x ? false : g(x)；g(x) = x ? true : f(x)
    # 两个成员彼此引用：分析 f 时 g 已在组内可见；f/g 共享同一单态
    # 参数类型，整组联立后各自得到 Bool -> Bool。
    f_def = A.lambda_(
        "x",
        A.if_(A.var("x"), A.bool_(False), A.apply(A.var("g"), A.var("x"))),
    )
    g_def = A.lambda_(
        "x",
        A.if_(A.var("x"), A.bool_(True), A.apply(A.var("f"), A.var("x"))),
    )
    mutual = A.letrec(
        [("f", f_def), ("g", g_def)], A.apply(A.var("f"), A.bool_(True))
    )
    show_success("letrec f,g 互相调用", mutual)

    section("递归组类型冲突：f 返回 Bool，g 却在同一递归里返回 Int")
    g_bad = A.lambda_(
        "y",
        A.if_(A.var("y"), A.int_(1), A.apply(A.var("f"), A.var("y"))),
    )
    conflict = A.letrec(
        [("f", f_def), ("g", g_bad)], A.apply(A.var("f"), A.bool_(True))
    )
    try:
        infer(conflict)
    except TypeMismatchError as exc:
        print(f"实际拒绝类型：{type(exc).__name__}（整组联立后构造性无解）")
        print(f"  左端类型：{exc.left_type}  路径：{format_path(exc.left_path)}")
        print(f"  右端类型：{exc.right_type}  路径：{format_path(exc.right_path)}")
    else:  # pragma: no cover
        raise SystemExit("内部错误：递归组类型冲突未被拦截")

    section("递归组也堵引用漏洞：r = ref(λz.z); h = λv. r := v")
    print("r 是扩展式成员，其单元自由变量对全组都是危险变量，")
    print("h 虽是 lambda 也不被泛化；先 h 1 再按函数使用单元必须被拒绝。\n")
    group_hole = A.letrec(
        [("r", A.ref(A.lambda_("z", A.var("z")))),
         ("h", A.lambda_("v", A.write_(A.var("r"), A.var("v"))))],
        A.let(
            "u1", A.apply(A.var("h"), A.int_(1)),
            A.apply(A.read(A.var("r")), A.bool_(True)),
        ),
    )
    try:
        infer(group_hole)
    except TypeMismatchError as exc:
        print(f"实际拒绝类型：{type(exc).__name__}（递归组未成为值限制后门）")
        print(f"  左端类型：{exc.left_type}  路径：{format_path(exc.left_path)}")
        print(f"  右端类型：{exc.right_type}  路径：{format_path(exc.right_path)}")
    else:  # pragma: no cover
        raise SystemExit("内部错误：递归组引用漏洞未被拦截")

    section("附：自应用 λx. x x 的无限类型也被 occurs 检查拦截")
    try:
        infer(A.lambda_("x", A.apply(A.var("x"), A.var("x"))))
    except InfiniteTypeError as exc:
        print(f"实际拒绝类型：{type(exc).__name__}")
        print(f"  变量 {exc.left_type} 不得出现在 {exc.right_type} 中")
        print(f"  路径：{format_path(exc.left_path)} 与 {format_path(exc.right_path)}")

    print("\n演示结束：组合算子/let 多态/互相递归组均真实推导成功，")
    print("递归组冲突、引用多态漏洞（含经递归组）与自应用均被真实拦截。")


if __name__ == "__main__":
    main()
