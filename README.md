# polyinfer —— 带值限制与 rank-1 显式签名的多态类型推断库

为离线编程语言工程实现的可复用 Hindley–Milner 类型推断模块（Algorithm W）。
纯 Python 3.14.7 标准库，零第三方依赖，核心算法（统一、occurs 检查、
替换组合、泛化）全部从零自实现，不调用任何第三方同类实现。

## 运行环境与命令

- Python：**3.14.7**（Linux 原生；只用标准库，安装 Python 后无需其他依赖）
- 无需网络、账号、密钥、数据库、中间件、云 API 或 Docker

```bash
# 必要测试（在仓库根目录执行）
python -m unittest discover -s tests -v

# 固定输入演示：真实计算并打印 1 个主类型 + 1 个实际触发的拒绝
python demo.py

# 本轮迭代演示：rank-1 签名的 3 个正常用例 + 3 类对照拒绝
python demo_iteration.py
```

两个 demo 均不含 `sleep`、不读预录结果、不做压力测试，实测耗时
均约 0.05 秒（迭代演示要求上限 90 秒内）。建议用仓库自带 venv：

```bash
python3.14 -m venv .venv && . .venv/bin/activate
python --version   # Python 3.14.7
```

## 公开接口

```python
from polyinfer import infer, A
from polyinfer.errors import (
    PolytypeError,            # 本库所有错误的基类
    InvalidAstError,          # AST 结构/字面量非法（带 path）
    NodeLimitError,           # 节点数超上限：资源拒绝，不是类型无解
    UnboundVariableError,     # 未绑定变量（含递归 let）（带 path）
    TypeMismatchError,        # 统一失败：构造子冲突，已证明无解
    InfiniteTypeError,        # 统一失败：occurs 检查，已证明无限类型
    SignatureMismatchError,       # 显式签名与实现不兼容（带 let/绑定位置）
    EscapedRigidVariableError,    # 签名刚性变量逃逸到外层单态
    ValueRestrictionConflictError,# 扩展式绑定声明多态，违反值限制
    format_path,              # 路径元组 -> "<root>/body/fn"
)

result = infer(ast, max_nodes=500)
result.principal_type   # "a -> a"：按首次出现规范命名的主类型字符串
result.type             # 结构化类型对象（TVar/TCon/TFun/TRef）
result.names            # {TVar: "a"|"b"|...}
result.subst            # 最终幂等替换（{TVar: Type}）
result.constraints      # 全部统一等式留痕（含两端 AST 路径）
result.generalizations  # 全部 let 泛化事件留痕
result.node_count       # AST 节点数
result.verify()         # 独立替换器逐约束核验（不经过统一核心）
```

## rank-1 显式类型签名（本轮新增）

`let` 节点支持可选字符串字段 `sig`（构造器第 4 个参数）：

```python
A.let("id", A.lambda_("x", A.var("x")), body, sig="forall a. a -> a")
```

语法（只支持 rank-1）：

```text
sig   := [forall 变量 {变量} "."] 类型
类型  := 原子 "->" 类型 | 原子            # 箭头右结合
原子  := Int | Bool | Unit | "Ref(" 类型 ")" | 变量名 | "(" 类型 ")"
```

约定与语义：

- 类型变量是小写标识符，**必须全部**在 `forall` 列表中显式量化；
  重复量化、未使用量化、未限定自由变量、函数参数位置再写 `forall`
  （高阶多态）在 AST 校验阶段即报 `InvalidAstError`（path 含 `sig`）；
- 检查函数体时，每个量化变量绑定一个**刚性（skolem）变量**
  （`polyinfer.types.RigidTVar`），它不会被悄悄统一成具体类型，
  也不会与另一个不同的刚性变量合并；绑定检查通过后，该绑定按
  “量化变量 = 刚性变量”的类型方案进入函数体，每次使用独立实例化，
  因此可以像无签名时一样两次按不同类型调用；
- 无 `sig` 时推断行为完全不变；签名允许比推断主类型更具体
  （实现 `a -> a`、签名 `Int -> Int`），但不能把单态引用包装成
  可任意实例化的多态函数；相邻绑定的量化变量作用域互不穿透
  （两个 let 都写 `forall a. a -> a` 也互相独立）；
- 三类失败各自独立、携带表达式位置（let 位置 `path`、绑定位置
  `bound_path`、深入到冲突叶子的统一路径）：

| 错误 | 触发情形 |
|---|---|
| `SignatureMismatchError` | 实现类型与签名不兼容（构造子冲突、两个不同刚性变量被迫合并、签名检查期 occurs 失败） |
| `EscapedRigidVariableError` | 刚性变量被绑定外层的单态推断变量吞并（典型：`λx. let f: ∀a. a→a = λy. x in …` 捕获外层单态变量） |
| `ValueRestrictionConflictError` | 扩展式绑定（`ref`、`apply` 等）的签名仍含量化变量（如 `let r: ∀a. Ref(a) = ref id`），值限制禁止该泛化 |

解析与推断解耦：`polyinfer.signature.parse_signature` 产出纯语法树
`ParsedSignature`，`build_signature_type` 只做树构建（不调用统一器）。

### AST 数据结构（普通 dict，可 JSON 化）

| tag | 字段 | 含义 |
|---|---|---|
| `var` | `name: str` | 变量引用 |
| `int` | `value: int` | 整数字面量（**拒绝 bool 与 float**） |
| `bool` | `value: bool` | 布尔字面量 |
| `lambda` | `param: str`, `body` | 单参数函数 |
| `apply` | `fn`, `arg` | 函数应用 |
| `let` | `name: str`, `bound`, `body`, 可选 `sig: str` | 非递归 `let name = bound in body`，`sig` 为 rank-1 显式签名 |
| `if` | `cond`, `then`, `else_` | 条件表达式 |
| `ref` | `inner` | 创建引用（ML 的 `ref e`） |
| `read` | `ref` | 解引用（`!e`） |
| `write` | `ref`, `value` | 赋值（`e1 := e2`），类型为 `Unit` |

构造器见 `polyinfer.ast`（别名 `A`）：`A.var/int_/bool_/lambda_/apply/
let/if_/ref/read/write_`。手写同构 dict 亦可，入口会做完整结构校验。

### 类型

`Int`、`Bool`、`Unit`（零元构造子）、`a -> b`（函数）、`Ref(t)`（引用）、
类型变量。变量规范命名按主类型结构**先序遍历的首次出现顺序**取
`a, b, …, z, a1, b1, …`，与推断内部 id 无关：

```text
a -> a                         恒等函数
a -> b -> a                    常函数
(a -> b) -> (c -> a) -> c -> b 组合算子
Ref(a) -> Unit                 λr. r := !r
```

函数箭头右结合；作为函数参数出现的函数类型加括号；`Ref(...)` 内部
按顶层渲染（`Ref(a -> a)`，不出现双重括号）。

## 错误语义与路径

- **可定位**：结构错误带 `path`；统一失败带失败约束**两端**各自的
  AST 路径 `left_path` / `right_path`，并随函数参数/返回值、引用单元
  下钻（`[arg]` / `[ret]` / `[inner]`）到真正冲突的叶子，同时给出
  叶子两端类型（内部变量名 `?n`；规范名只用于最终主类型）。
- **不静默纠正**：`int` 节点只接受真正的 `int`（`bool` 虽是 `int`
  子类但显式拒绝）；`float` 不隐式截断；若以 float 提供，NaN 与
  ±Infinity 作为非有限值明确拒绝，其他 float 同样拒绝。
- **资源拒绝 vs 证明无解严格分开**：`NodeLimitError`（默认 >500 节点）
  表示“输入超出接口预算，未做推断”；`TypeMismatchError` 与
  `InfiniteTypeError` 是“已构造性证明该程序不可类型化”。推断过程确定、
  无回溯、无搜索，因此不存在“搜索预算耗尽”这一第三状态。
- **无部分变更**：`infer` 是纯函数，失败仅抛异常；为容纳 500 层深 AST
  临时提高的解释器递归上限在 `try/finally` 中恢复，成功失败都不留下
  全局变更。

## 设计取舍

1. **值限制（value restriction）**：`let` 仅当绑定表达式是非扩展的
   语法值（整数字面量、布尔字面量、lambda、变量）时才泛化；
   `ref`、`apply`、`read`、`write`、`if` 等扩展式得到单态方案。
   这堵住了经典引用多态漏洞：`let r = ref(λz.z) in r := 1; !r` 中
   `r` 不会被泛化为 `∀a. Ref(a→a)`，写入 `Int` 当场冲突。采用的是
   严格值限制，不是更宽松的弱化值限制/递归闭包变体。
2. **lambda 参数单态**：参数以空限定集的平凡类型方案入环境，函数体内
   共享同一个类型变量；只有 `let` 绑定能引入多态。
3. **显式替换线程**：不使用可变并查集，每个推断步骤返回
   `(subst, type)`；`bind(s, v, t)` 实现替换组合（新绑定右端与旧绑定
   右端都先经已有替换推进），产出替换幂等、无循环。
4. **引用不变式**：`Ref(t)` 按不变式逐点统一，不支持协变。
5. **约束留痕**：每次统一（含下钻子等式）都记录统一前两端类型与路径。
6. **非递归 let**：绑定表达式作用域不含被绑定名字；递归定义不在本
   契约内，绑定树中引用同名变量报 `UnboundVariableError`。
7. **rank-1 签名与刚性变量**：显式签名的量化变量在绑定检查期间是
   刚性 skolem，不与具体类型/彼此统一；逃逸（被外层单态吞并）与
   扩展式泛化（值限制）分别由专门错误拦截。不支持高阶多态
   （`forall` 只能出现在签名最外层）。

## 独立参考检查器（`polyinfer.reference`）

- 自带替换应用（带循环检测）、自由变量收集、结构同构判定，
  **不导入也不调用**核心的 `unify`/`infer` 来获得期望值；
- 对每个成功推导核验：替换满足 occurs 且幂等；最终替换作用于
  **全部**记录约束后两端结构同构；每条 let 的泛化集合恰为
  “绑定类型自由变量 − 环境自由变量”，并由检查器依据绑定表达式 tag
  独立重判值限制（不信任核心字段）。
- 测试包含对结果（约束、替换、循环、occurs、泛化集合、值限制标记）
  的**篡改检测**，证明检查器确实在独立审计；另有固定种子的良构项
  生成器做补充测试。主类型期望值均为手写，往返一致仅作补充。

## 输入上限

- AST 节点数默认上限 **500**（`infer(ast, max_nodes=...)` 可调，
  属接口边界）；超限抛 `NodeLimitError`。短演示不穷举上限规模；
  测试覆盖 499/500/501 边界语义。
- 整数支持 Python 任意精度大整数；本语言没有浮点类型与浮点运算，
  因此 README 无需浮点误差口径——不存在浮点结果，float 输入被拒绝。

## 模块布局

```text
polyinfer/
  __init__.py    公开接口
  ast.py         AST 构造器与结构/作用域/上限校验
  types.py       类型结构、occurs、apply、bind 替换组合、渲染、规范命名
  inference.py   Algorithm W 核心、值限制、约束/泛化留痕、infer() 入口
  reference.py   独立参考检查器（不经过统一核心）
  signature.py   rank-1 签名解析（纯语法层）与刚性变量实例化
  errors.py      错误类型体系与路径渲染
tests/           unittest 测试（替换原语、推断、四个验收场景、
                 输入契约、独立检查器与篡改检测、确定性生成测试、
                 rank-1 签名行为与不借用统一器的小替换参照）
demo.py          固定输入演示
demo_iteration.py 本轮迭代演示（rank-1 签名）
```

四个验收场景均有对应测试（`tests/test_inference.py`）：
恒等函数两种实例、lambda 参数不可多态（含 let 对照组）、
自应用无限类型、引用多态漏洞。

## 已知限制（只按本契约实现，不宣称行业标准完整兼容）

- 不含递归 let/递归函数、多态递归；类型标注仅限 let 绑定处的
  rank-1 前缀多态签名（不支持高阶/二级多态与类型别名）；
- 不含词法/语法解析器——输入是结构化 AST；
- 只有 `Int`/`Bool`/`Unit` 与函数、引用，不含元组、列表、变体、
  记录、ADT、kinding、行多态、高阶多态（`∀f. …`）等；
- 采用严格值限制：部分在弱化值限制下可泛化的良性程序在此被保守保留
  为单态（这是有意的安全取舍）；
- `Unit` 没有字面量，只作为 `write` 的结果类型出现；
- 无副作用求值语义：本库只做静态类型推断，不解释执行 AST，
  也绝不会用 Python 执行输入 AST。
