"""demo.py — polyinfer 固定输入演示（真实计算，无 sleep / 预录结果）。

展示：
  1. 恒等函数的主类型及其两种实例（Int / Bool）；
  2. let 多态程序的主类型；
  3. 引用多态漏洞被值限制拒绝（实际触发的 UnifyError，含两端 AST 路径）。

运行：python demo.py
"""

from polyinfer import (
    Var, Int, Bool, Lam, App, Let, Ref, Read, Write,
    infer, UnifyError, format_path,
)


def main() -> None:
    identity = Lam("x", Var("x"))

    # 1. 恒等函数主类型 + 两种实例
    print("== 恒等函数 ==")
    print("  λx. x            :", infer(identity).type_str)
    print("  (λx. x) 3        :", infer(App(identity, Int(3))).type_str)
    print("  (λx. x) true     :", infer(App(identity, Bool(True))).type_str)

    # 2. let 多态：let id = λx.x in let a = id 3 in id true
    print("== let 多态（语法值泛化） ==")
    prog = Let("id", identity,
               Let("a", App(Var("id"), Int(3)),
                   App(Var("id"), Bool(True))))
    r = infer(prog)
    print("  let id = λx.x in let a = id 3 in id true")
    print("  主类型           :", r.type_str)
    print("  已满足约束数     :", len(r.constraints))

    # 3. 引用多态漏洞：值限制下被拒绝
    # let r = ref (λx.x) in let _ = write r (λy.7) in (read r) true
    print("== 引用多态漏洞（值限制拒绝） ==")
    hole = Let("r", Ref(identity),
               Let("_", Write(Var("r"), Lam("y", Int(7))),
                   App(Read(Var("r")), Bool(True))))
    print("  let r = ref (λx.x) in let _ = write r (λy.7) in (read r) true")
    try:
        result = infer(hole)
        print("  [意外] 被接受:", result.type_str)
    except UnifyError as e:
        print("  拒绝             :", e)
        print("  失败约束左端路径 :", format_path(e.left_path))
        print("  失败约束右端路径 :", format_path(e.right_path))


if __name__ == "__main__":
    main()
