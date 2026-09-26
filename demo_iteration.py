#!/usr/bin/env python3
"""本轮迭代演示：rank-1 显式类型签名（纯离线、真实计算、无 sleep/预录）。

运行：``python demo_iteration.py``

全部用例真实调用 :func:`polyinfer.infer`，打印：

1. 正常：签名恒等函数 ``id : forall a. a -> a`` 在同一作用域内
   被按 Bool 与 Int **两次不同类型**调用；
2. 正常：签名比推断更具体（``Int -> Int``），以及相邻两个 let
   各自量化同名变量 ``a`` 但作用域互不穿透；
3. 边界拒绝：签名不匹配（实现 λx.true 配 ``Int -> Int``）、
   逃逸的刚性变量（闭包捕获外层单态参数却声明 ``forall a``）、
   值限制冲突（``ref id`` 配 ``forall a. Ref(a)``）。

每个成功结果都经过两道独立核验：库自带的 ``result.verify()``
（独立替换器复核全部约束与泛化）以及测试侧的小替换参照
``check_signatures``（不借用被测统一器）。
"""

import sys
import time

from polyinfer import A, infer
from polyinfer.errors import (
    EscapedRigidVariableError,
    SignatureMismatchError,
    ValueRestrictionConflictError,
)

sys.path.insert(0, "tests")
from signature_oracle import check_signatures  # noqa: E402


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def show_success(label: str, node) -> None:
    t0 = time.perf_counter()
    result = infer(node)
    report = result.verify()
    value_signed, expansive_signed = check_signatures(result)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[{label}] 接受，主类型: {result.principal_type}")
    print(
        f"  统一约束 {report.constraints_checked} 条、泛化事件 {report.generalizations_checked} 条；"
        f"签名绑定（语法值 {value_signed} / 扩展式单态 {expansive_signed}）；{elapsed_ms:.2f} ms"
    )


def show_rejection(label: str, node, expected) -> None:
    t0 = time.perf_counter()
    try:
        infer(node)
    except expected as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"[{label}] 拒绝: {type(exc).__name__}（{elapsed_ms:.2f} ms）")
        print(f"  {exc}")
        return
    raise SystemExit(f"内部错误：{label} 未被拦截")


def main() -> None:
    section("正常 1：let id : forall a. a -> a = λx.x，先用于 Bool 再用于 Int")
    twice = A.let(
        "id",
        A.lambda_("x", A.var("x")),
        A.if_(
            A.apply(A.var("id"), A.bool_(True)),
            A.apply(A.var("id"), A.int_(1)),
            A.int_(2),
        ),
        sig="forall a. a -> a",
    )
    show_success("id true; id 1", twice)

    section("正常 2：签名可以比推断主类型更具体（id : Int -> Int）")
    specific = A.let(
        "id",
        A.lambda_("x", A.var("x")),
        A.apply(A.var("id"), A.int_(7)),
        sig="Int -> Int",
    )
    show_success("monomorphic signature", specific)

    section("正常 3：相邻绑定的量化变量同名也互不穿透（两个 forall a）")
    adjacent = A.let(
        "f",
        A.lambda_("x", A.var("x")),
        A.let(
            "g",
            A.lambda_("y", A.var("y")),
            A.apply(A.var("g"), A.apply(A.var("f"), A.bool_(False))),
            sig="forall a. a -> a",
        ),
        sig="forall a. a -> a",
    )
    show_success("independent quantifiers", adjacent)

    section("边界 1：签名不匹配——实现 λx.true 却声明 Int -> Int")
    mismatch = A.let(
        "f",
        A.lambda_("x", A.bool_(True)),
        A.apply(A.var("f"), A.int_(1)),
        sig="Int -> Int",
    )
    show_rejection("signature mismatch", mismatch, SignatureMismatchError)

    section("边界 2：逃逸的刚性变量——闭包捕获外层单态参数")
    print("程序：λx. let f : forall a. a -> a = λy. x in (f true; f 1)")
    print("实现把参数 y 与外层单态 x 绑死；若放行，两次不同实例化会污染 x。")
    escape = A.lambda_(
        "x",
        A.let(
            "f",
            A.lambda_("y", A.var("x")),
            A.if_(
                A.apply(A.var("f"), A.bool_(True)),
                A.apply(A.var("f"), A.int_(1)),
                A.int_(0),
            ),
            sig="forall a. a -> a",
        ),
    )
    show_rejection("escaped rigid", escape, EscapedRigidVariableError)

    section("边界 3：值限制冲突——扩展式 ref 不能靠签名包装成 forall a. Ref(a)")
    print("程序：let id = λz.z in let r : forall a. Ref(a) = ref id in r := 1; !r")
    value_restriction = A.let(
        "id",
        A.lambda_("z", A.var("z")),
        A.let(
            "r",
            A.ref(A.var("id")),
            A.let(
                "u",
                A.write_(A.var("r"), A.int_(1)),
                A.read(A.var("r")),
            ),
            sig="forall a. Ref(a)",
        ),
    )
    show_rejection("value restriction", value_restriction, ValueRestrictionConflictError)

    section("对照：同样的扩展式写单态签名 Ref(Int) 则合法")
    mono_ref = A.let(
        "r",
        A.ref(A.int_(1)),
        A.read(A.var("r")),
        sig="Ref(Int)",
    )
    show_success("let r : Ref(Int) = ref 1", mono_ref)

    print("\n演示结束：3 个正常签名推导 + 3 类错误各 1 个对照拒绝，全部真实计算。")


if __name__ == "__main__":
    main()
