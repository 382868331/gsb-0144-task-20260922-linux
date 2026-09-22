# polyinfer — 带值限制的多态类型推断

离线、纯标准库的 Hindley–Milner 风格类型推断模块（Algorithm W 的约束记录变体），
从零实现，未调用任何第三方类型推断库。

- **Python 版本**：3.14.7（仅标准库，安装 Python 后无需任何额外依赖）
- **平台**：Linux 原生；核心、测试、演示全部离线，无需网络/账号/数据库/Docker
- **输入上限**：AST 最多 **500 个节点**（接口边界，`infer(..., max_nodes=...)` 可调）

## 语言与规则

- 节点：`Var` `Int` `Bool` `Lam` `App` `Let` `If` `Ref` `Read` `Write`（纯数据，绝不 eval/exec 输入）
- 类型：`Int` `Bool` `Unit`、`TFun`（函数）、`TRef`（引用）、`TVar`（类型变量）
- **值限制**：`let` 仅对语法值（变量 / 字面量 / lambda）泛化
- **lambda 参数单态**；`write` 返回 `Unit`；**不含递归 let**（绑定名在自身值中不可见）
- 统一带 **occurs 检查**（拒绝无限类型）；替换的组合与应用由本模块自行实现

## 接口

```python
from polyinfer import (
    Var, Int, Bool, Lam, App, Let, If, Ref, Read, Write,   # AST 节点
    TInt, TBool, TUnit, TFun, TRef, TVar, INT, BOOL, UNIT, # 类型
    Scheme, Constraint, InferResult,
    infer, unify, apply_substitution, compose_substitutions,
    free_type_vars, type_to_string, format_path,
    InferError, InputError, UnboundVariableError,
    UnifyError, OccursError, BudgetExhaustedError,
)

result = infer(ast)            # -> InferResult
result.type_str                # 按首次出现规范命名的主类型，如 "forall a. a -> a"
result.type                    # 主类型（内部变量 id）
result.constraints             # 推导中生成并满足的全部约束（含两端 AST 路径）
result.substitution            # 最终替换 {var_id: Type}
```

### 错误语义

| 异常 | 含义 |
|---|---|
| `InputError` | 非法输入：非 AST 节点、`Int` 值为 bool/float（NaN、Infinity 单独报错）、`Bool` 值非 bool、空名字、超节点上限、非法 `max_nodes`/`max_steps`；消息含可定位的 AST 路径 |
| `UnboundVariableError` | 未绑定变量，含 `name` 与 `path` |
| `UnifyError` | 约束被**证明不可满足**，含失败约束两端的 `left_path` / `right_path`（`$/let.body/app.func` 形式） |
| `OccursError` | occurs 检查失败（无限类型），`UnifyError` 子类 |
| `BudgetExhaustedError` | 统一步数预算耗尽；与“证明无解”（`UnifyError`）**严格区分**，互不继承 |

非法输入一律显式报错，不静默纠正；推断为单次纯函数调用，失败不留下任何部分变更。

## 命令

```bash
python -m unittest discover -s tests -v   # 运行全部测试
python demo.py                            # 约 1 秒内完成：主类型 + 引用赋值拒绝演示
```

## 设计取舍

- **约束记录式求解**：遍历时每生成一条约束（含两端 AST 路径）立即求解，保证 `let`
  泛化时替换已就位；同时把约束快照存入 `InferResult.constraints`，测试用**独立编写的
  替换器**（`tests/test_polyinfer.py` 中的 `_resolve`，不调用核心 `apply_substitution`/`unify`）
  复核每个成功推导的全部约束在最终替换下两端结构相等。
- **值限制**采用 Wright 的语法值判定：仅 `Var`/`Int`/`Bool`/`Lam` 泛化，`Ref`、`App` 等
  一律单态，从而堵住“引用多态漏洞”（测试与 demo 均覆盖）。
- **规范命名**：主类型的变量按在类型中首次出现的顺序命名为 `a, b, ..., z, a1, ...`，
  含自由变量时输出 `forall` 前缀。
- **预算语义**：`max_steps` 是统一步数上限，耗尽抛 `BudgetExhaustedError`；
  在 500 节点上限内正常程序远不会触及，它只作为防御性边界存在。

## 已知限制

- 不支持递归 let、运算符、元组/记录等；语言仅限题面十种节点。
- 错误路径定位到**失败约束两端对应的 AST 子表达式**，不是词法行列号。
- 本模块不产生任何浮点结果，浮点误差口径不适用；`Int` 节点拒绝一切 float
  （NaN/Infinity 单独报错），整数参数拒绝 bool。
- 只实现本题契约的类型系统子集，不宣称兼容任何完整语言标准。
