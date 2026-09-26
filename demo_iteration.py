#!/usr/bin/env python3
"""本轮迭代演示：let 绑定上的 rank-1 类型签名（纯离线、真实计算）。

运行：``python demo_iteration.py``

展示四部分，全部现场推断、无预录输出、无 sleep：

1. 正常：恒等函数带 ``forall a. a -> a`` 签名，在同一作用域内按
   Bool 与 Int 两种类型调用，独立参考检查器（不借用待测统一器的
   小型替换检查）逐约束核验；
2. 边界一：签名比实现更一般（``forall a b. a -> b`` 之于恒等函数），
   报“签名不匹配”；
3. 边界二：捕获外部单态变量的闭包配多态签名，刚性变量逃逸；
4. 边界三：给 ``ref`` 扩展式配多态签名，值限制冲突——单态引用
   不得包装成可任意实例化的多态值。
"""

import time

from polyinfer import A, format_path, infer
from polyinfer.errors import (
    RigidEscapeError,
    SignatureMismatchError,
    ValueRestrictionConflictError,
)

ID_SIG = A.sig(["a"], A.tfun(A.tvar("a"), A.tvar("a")))


def section(title: str) -> None:
    bar = "=" * 68
    print(f"\n{bar}\n{title}\n{bar}")


def main() -> None:
    t_all = time.perf_counter()

    section("正常：let id : forall a. a -> a = λx.x，按 Bool / Int 各调一次")
    prog = A.let(
        "id",
        A.lambda_("x", A.var("x")),
        A.if_(
            A.apply(A.var("id"), A.bool_(True)),
            A.apply(A.var("id"), A.int_(1)),
            A.int_(2),
        ),
        sig=ID_SIG,
    )
    t0 = time.perf_counter()
    result = infer(prog)
    report = result.verify()  # 独立替换检查，不经过统一核心
    ms = (time.perf_counter() - t0) * 1000
    print("程序：let id : forall a. a -> a = λx.x")
    print("      in if id true then id 1 else 2")
    print(f"  主类型       : {result.principal_type}")
    print(f"  签名事件     : 量化 {result.generalizations[0].sig_quantified}，"
          "实现类型与签名经独立检查结构同构")
    print(f"  独立核验     : {report.constraints_checked} 条约束、"
          f"{report.generalizations_checked} 条泛化事件全部成立")
    print(f"  耗时         : {ms:.2f} ms（真实计算）")

    section("边界一：签名不匹配 —— forall a b. a -> b 之于 λx.x")
    too_general = A.let(
        "f",
        A.lambda_("x", A.var("x")),
        A.var("f"),
        sig=A.sig(["a", "b"], A.tfun(A.tvar("a"), A.tvar("b"))),
    )
    try:
        infer(too_general)
    except SignatureMismatchError as exc:
        print(f"实际拒绝类型：{type(exc).__name__}")
        print(f"  let 位置：{format_path(exc.path)}，绑定名 {exc.name!r}")
        print(f"  冲突两端：{exc.left_type} @ {format_path(exc.left_path)}；"
              f"{exc.right_type} @ {format_path(exc.right_path)}")
    else:
        raise SystemExit("内部错误：过于一般的签名未被拦截")

    section("边界二：刚性变量逃逸 —— 闭包捕获外部单态变量")
    escape = A.lambda_(
        "y",
        A.let("f", A.lambda_("z", A.var("y")), A.int_(0), sig=ID_SIG),
    )
    print("程序：λy. let f : forall a. a -> a = λz. y in 0")
    print("f 的实际类型含外层单态参数 y 的类型变量；签名量化的 a")
    print("若与它统一，就会穿过外层绑定逃逸出签名作用域。\n")
    try:
        infer(escape)
    except RigidEscapeError as exc:
        print(f"实际拒绝类型：{type(exc).__name__}")
        print(f"  刚性变量 {exc.variable!r} 逃逸：{exc.where}")
        print(f"  let 位置：{format_path(exc.path)}")
    else:
        raise SystemExit("内部错误：刚性变量逃逸未被拦截")

    section("边界三：值限制冲突 —— 单态引用不得包装成多态")
    hole = A.let(
        "r",
        A.ref(A.lambda_("z", A.var("z"))),
        A.let("_u", A.write_(A.var("r"), A.int_(1)), A.read(A.var("r"))),
        sig=A.sig(["a"], A.tref(A.tfun(A.tvar("a"), A.tvar("a")))),
    )
    print("程序：let r : forall a. Ref(a -> a) = ref(λz.z) in r := 1; !r")
    print("ref 是扩展式，值限制下不得泛化；签名要求泛化即冲突。\n")
    try:
        infer(hole)
    except ValueRestrictionConflictError as exc:
        print(f"实际拒绝类型：{type(exc).__name__}")
        print(f"  绑定表达式 tag：{exc.bound_tag!r}，签名量化 {list(exc.quantified)}")
        print(f"  let 位置：{format_path(exc.path)}")
    else:
        raise SystemExit("内部错误：值限制冲突未被拦截")

    total_ms = (time.perf_counter() - t_all) * 1000
    print(f"\n演示结束：1 个正常签名推导 + 3 类签名错误均被真实计算拦截，"
          f"总耗时 {total_ms:.2f} ms。")


if __name__ == "__main__":
    main()

