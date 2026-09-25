# polyinfer —— 带值限制的多态类型推断库

为离线编程语言工程实现的可复用 Hindley–Milner 类型推断模块（Algorithm W）。
纯 Python 3.14.7 标准库，零第三方依赖，核心算法（统一、occurs 检查、
替换组合、泛化）全部从零自实现，不调用任何第三方同类实现。

除普通推断入口 ``infer`` 外，本库在它旁边提供**诊断入口** ``diagnose``：
推断失败时不再只报“两个位置不合”，而是给出按稳定编号确定的
**包含极小冲突约束集**，并追踪实例化、替换与统一的来源。
诊断模式不改变任何类型系统规则（值限制、occurs、语法限制一律不放宽）。

## 运行环境与命令

- Python：**3.14.7**（Linux 原生；只用标准库，安装 Python 后无需其他依赖）
- 无需网络、账号、密钥、数据库、中间件、云 API 或 Docker

```bash
# 必要测试（在仓库根目录执行，含普通回归与诊断专项）
./.venv/bin/python -m unittest discover -s tests -v

# 固定输入演示：真实计算并打印主类型、实际触发的拒绝、以及诊断结果
./.venv/bin/python demo.py
```

demo 不含 `sleep`、不读预录结果、不做压力测试，实测耗时约 0.06 秒
（上限 90 秒内）。

## 公开接口

```python
from polyinfer import infer, diagnose, format_diagnosis, A
from polyinfer.errors import (
    PolytypeError,            # 本库所有错误的基类
    InvalidAstError,          # AST 结构/字面量非法（带 path）
    NodeLimitError,           # 节点数超上限：资源拒绝，不是类型无解
    UnboundVariableError,     # 未绑定变量（含递归 let）（带 path）
    TypeMismatchError,        # 统一失败：构造子冲突，已证明无解
    InfiniteTypeError,        # 统一失败：occurs 检查，已证明无限类型
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

diag = diagnose(ast, max_nodes=500, max_checks=512)   # 诊断模式
diag.ok                  # True 时 principal_type/result 与 infer 逐字一致
diag.minimal             # False 时（预算耗尽）不假称极小
diag.cores               # 一个或多个冲突核（多处独立错误 -> 多核）
diag.conflict_indices()  # 第一个核的稳定编号，如 (0, 1)
diag.evidence.kind       # "type-mismatch" | "infinite-type"
diag.constraints         # 全部带稳定编号的约束（位置/说明/形态/下钻条数）
diag.instantiation_trace # 使用点 -> let/lambda 来源、多态或单态共享
diag.provenance(var_id)  # 类型变量的替换来源链（指向编号约束）
diag.budget_exhausted    # 独立诊断预算是否耗尽
```


### AST 数据结构（普通 dict，可 JSON 化）

| tag | 字段 | 含义 |
|---|---|---|
| `var` | `name: str` | 变量引用 |
| `int` | `value: int` | 整数字面量（**拒绝 bool 与 float**） |
| `bool` | `value: bool` | 布尔字面量 |
| `lambda` | `param: str`, `body` | 单参数函数 |
| `apply` | `fn`, `arg` | 函数应用 |
| `let` | `name: str`, `bound`, `body` | 非递归 `let name = bound in body` |
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

## 诊断模式：极小冲突约束集（`polyinfer.diagnose`）

普通 `infer` 失败时只给“第一次统一失败”的两个位置。诊断入口
`diagnose(ast, max_nodes=500, max_checks=512)` 在它旁边，回答
“**哪几条输入约束合起来才有问题**”。

```python
from polyinfer import A, diagnose, format_diagnosis

# λf. if f then (f 1) else 2：
# f 既要 Bool（if 条件）又要 Int -> a（被应用），两条约束单看都成立，
# 合起来才矛盾。
prog = A.lambda_("f",
        A.if_(A.var("f"), A.apply(A.var("f"), A.int_(1)), A.int_(2)))
diag = diagnose(prog)
diag.ok                 # False
diag.minimal            # True：包含极小（逐条移除验证过）
diag.conflict_indices() # (0, 1)：#0 与 #1 合起来才冲突
diag.evidence.kind      # "type-mismatch"；自应用则是 "infinite-type"
print(format_diagnosis(diag))
```

### 稳定编号

诊断先做一次**容错遍历**（统一失败不抛出、记录后继续），枚举整棵 AST
诱导的全部顶层统一等式，按遍历首次出现顺序编号 `#0, #1, …`。每条等式
以“**两端 AST 路径对**”为稳定身份键——键只取决于 AST 结构，与替换内容
无关，因此同一程序每次诊断编号一致（测试对同程序多次诊断逐字比对）。
函数/引用下钻出的子等式（路径后缀 `[arg]`/`[ret]`/`[inner]`）共享所属
顶层等式的编号。`diag.constraints` 给出每条编号约束的来源位置、说明
与当时形态；`diag.instantiation_trace` 追踪每个变量使用点实例化自
哪个 let / lambda 形参（多态新实例还是单态共享）；
`diag.provenance(变量内部id)` 给出“某类型变量被哪条编号约束绑定成
什么”的替换来源链。

### 极小化算法（不要求最小基数）

对编号从大到小的每条约束，重放“当前候选集 − 该条”：**删除后仍冲突就
保留删除**，直到一轮结束。一轮结束后再对最终集合**逐条冻结重放验证**：
删去其中任意一条都必须可满足——“包含极小（irredundant）”是验证出来的，
不是口头声明；不声称最小基数（minimum cardinality）。

程序可能有**两处互相独立的错误**：取出一个冲突核后把它整体跳过再重放，
若仍冲突，则在剩余约束上再跑一轮逆序删除提取下一个核，直到剩余集合
可满足。例如 `if 1 then 2 else false`（条件错 + 两分支不合）报出两个
核 `(#0,)`、`(#1,)`。

### 冻结重放与“不放宽限制”

每一次试探都是：**深拷贝冻结的原始 AST** + **全新推断器**（空替换、
空类型变量计数器、空留痕）。绝不复用上一次失败遍历的并查集或替换；
“跳过一条约束”意味着该顶层统一及其全部下钻完全不发生。SAT/UNSAT
一律由**严格**重放判定（依据一阶语法合一定理：可合一则任意处理顺序
都成功），因此 let 泛化集合在成功前缀的部分最一般解下精确计算——
**值限制一条不放宽**：共享引用仍不泛化、lambda 形参仍单态，可满足只
能来自“删掉了冲突等式”，而不是来自更松的类型规则。

### occurs 与普通不合的区分

冲突证据 `diag.evidence.kind` 区分两类：自应用 `λx. x x` 是
`"infinite-type"`（occurs 检查失败、无限类型）；`if 1 ...`、
`read 1` 等是 `"type-mismatch"`（构造子不一致）。两者的叶子位置与
两端类型都保留。

### 独立预算与限制

- 诊断有**独立预算** `max_checks`（默认 512，诊断遍历次数；容错枚举、
  每次删除试探、逐核成员验证都计数；与 500 节点上限完全无关）。
  预算耗尽时返回已经找到的核与尚未缩减完的候选，
  `diag.minimal == False`、`diag.budget_exhausted == True`，
  并在 `format_diagnosis` 中明确“不宣称极小”——绝不会假称极小。
- 成功程序的诊断结果 `diag.ok == True`，`diag.principal_type` 与
  `result` 与普通 `infer` **逐字一致**，且同样通过独立参考检查器。
- AST 非法、节点超限、未绑定变量等“类型冲突之前”的错误与普通入口
  一样直接抛出，不会被诊断吞掉。
- 诊断只使用原有语法与类型系统，**不引入新类型构造、不改变规则**；
  它是对同一套 HM + 值限制推断的可解释性包装。

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
  __init__.py    公开接口（infer 与 diagnose 并列）
  ast.py         AST 构造器与结构/作用域/上限校验
  types.py       类型结构、occurs、apply、bind 替换组合、渲染、规范命名
  inference.py   Algorithm W 核心、值限制、约束/泛化/绑定/实例化留痕、infer()
  diagnose.py    诊断模式：稳定编号、冻结重放、逆序删除极小化、diagnose()
  reference.py   独立参考检查器（不经过统一核心）
  errors.py      错误类型体系与路径渲染
tests/           unittest 测试（替换原语、推断、四个验收场景、
                 输入契约、独立检查器与篡改检测、确定性生成测试、
                 诊断模式专项：极小核/独立多核/预算/冻结重放/追踪）
demo.py          固定输入演示（普通推断 + 拒绝边界 + 诊断模式）
```

四个验收场景均有对应测试（`tests/test_inference.py`）：
恒等函数两种实例、lambda 参数不可多态（含 let 对照组）、
自应用无限类型、引用多态漏洞。

## 已知限制（只按本契约实现，不宣称行业标准完整兼容）

- 不含递归 let/递归函数、多态递归、类型标注与用户给定类型方案；
- 不含词法/语法解析器——输入是结构化 AST；
- 只有 `Int`/`Bool`/`Unit` 与函数、引用，不含元组、列表、变体、
  记录、ADT、kinding、行多态、高阶多态（`∀f. …`）等；
- 采用严格值限制：部分在弱化值限制下可泛化的良性程序在此被保守保留
  为单态（这是有意的安全取舍）；
- `Unit` 没有字面量，只作为 `write` 的结果类型出现；
- 无副作用求值语义：本库只做静态类型推断，不解释执行 AST，
  也绝不会用 Python 执行输入 AST。

诊断模式的额外口径：

- 极小化需要至多“约束条数 + 核成员数 + 核数”量级的冻结重放，
  由独立预算 `max_checks` 封顶；预算耗尽时给未缩减完的证据而非极小核；
- 冲突核按“逆编号序单轮删除”确定，结果对同一 AST 确定、可复现，
  但**不保证最小基数**（契约明确只要求包含极小）；
- 诊断的可满足性结论只依赖严格重放与语法合一定理，容错遍历不参与
  任何 SAT/UNSAT 判定，因此追踪机制不可能改变推断结论。
