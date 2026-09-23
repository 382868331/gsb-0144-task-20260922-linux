# polyinfer —— 带值限制的多态类型推断库（HM/Algorithm W）+ 开放行、效应与递归模块

为离线编程语言工程实现的可复用类型推断库。纯 Python **3.14.7** 标准库，
零第三方依赖。核心算法（统一、occurs 检查、替换组合、泛化、行统一、
效应集合统一、SCC 分析）全部从零自实现，不调用任何第三方同类实现。

本版（2.x）在初版单表达式推断之上**迭代接入原有数据模型与公开入口**，
新增三组能力：

1. **开放行记录**：记录字面量、字段访问、不改变原记录的扩展；同名字段
   扩展一律报错；行变量做出现（occurs/缺席）检查，开放行统一成环被拒。
2. **读/写/异常效应**：`R`/`W`/`X` 三种效应与函数潜在效应集合；
   引用写入与异常处理都参与泛化/一致性判断；携带可变引用的类型变量
   不能被模块边界偷偷泛化。
3. **递归模块系统**：模块可互相递归引用；每个导出必须带显式签名；
   按依赖强连通分量（SCC）先查签名再查实现；缓存键含依赖签名与效应，
   失效集是精确的逆向依赖闭包；一次修改失败不替换上次有效结果。

## 运行环境与命令

- Python：**3.14.7**（Linux 原生；只用标准库）
- 无需网络、账号、密钥、数据库、中间件、云 API、容器或硬件。

```bash
# 全部测试（保留初版入口；v1 + v2 + 模块 + 参考域 + 篡改检测）
python -m unittest discover -s tests -v

# 仅快速回归（初版 v1：替换原语/推断/输入契约/独立检查器）
python -m unittest tests.test_substitution tests.test_inference \
                   tests.test_input_contract tests.test_reference -v

# 扩展验证（v2 记录、效应、模块 SCC/缓存、参考域、篡改检测）
python -m unittest tests.test_v2_records tests.test_v2_effects \
                   tests.test_modules tests.test_reference_domain \
                   tests.test_tamper_v2 -v

# 固定输入演示（v1 兼容 + v2 正常结果 + 两个棘手边界 + 四模块反例
# + 增量缓存；真实计算，实测 < 1 秒，硬上限 90 秒）
python demo.py
```

`demo.py` 不含 `sleep`、不读预录结果、不做压力测试。

## 版本与兼容策略

- `infer(ast)` 默认 **`language="v1"`**：初版 10 个节点、严格值限制、
  主类型渲染（含 `Ref(a) -> Unit`）、异常类型、确定性结果**逐项不变**；
  旧测试一个未删、断言未放松。
- 新语法通过**显式选项** `infer(ast, language="v2")` 启用；
  模块系统只在 v2 上工作。
- 版本号升到 `2.0.0`。v1 无需要迁移的破坏（见文末“相对初版的变更”）。

```python
from polyinfer import infer, A
infer(A.lambda_("x", A.var("x"))).principal_type          # "a -> a"（v1）
infer(node, language="v2").principal_type                 # v2
```

## 公开接口

```python
from polyinfer import infer, A
from polyinfer.modules import ModuleChecker, check_modules
from polyinfer.signature import parse_signature
from polyinfer.errors import (
    PolytypeError, InvalidAstError, NodeLimitError, UnboundVariableError,
    TypeMismatchError, InfiniteTypeError,
    RecordFieldError,        # 记录字段冲突/同名扩展/开放行成环（构造性无解）
    EffectMismatchError,     # 函数潜在效应集合与签名不一致
    SkolemEscapeError,       # 携带可变引用的类型变量越过模块边界泛化
)
from polyinfer.signature import SignatureError  # 导出签名串非法

result = infer(ast, max_nodes=500, language="v1")
result.principal_type        # 规范命名的主类型字符串
result.type / result.names / result.subst
result.constraints           # 全部统一等式留痕（两端 AST 路径）
result.generalizations       # 全部泛化事件留痕
result.verify()              # 独立检查器（不经过统一核心）
# v2 附加：
result.effects               # frozenset，子集 of {"R","W","X"}
result.effects_undecided     # 是否含未决效应变量（evar）
result.effect_traces         # 效应产生/处理留痕
result.row_absences          # 开放行缺席约束留痕（extend 防成环）
```

模块层：

```python
checker = ModuleChecker(modules, max_nodes=20000)
report = checker.check_all()                 # 增量（带缓存）
report = checker.check_all(use_cache=False)  # 非缓存检查路径（从零全查）
checker.invalidation_set("M")                # 精确缓存失效集
# 改完若干模块后：
report = checker.check_all(changed={"M"})
check_module_report(report)                  # 独立审计模块结果
```

## AST（普通 dict，可 JSON 化）

### v1 节点（默认）

| tag | 字段 |
|---|---|
| `var` | `name: str` |
| `int` | `value: int`（拒绝 bool/float，NaN/Infinity 拒绝） |
| `bool` | `value: bool` |
| `lambda` | `param: str`, `body` |
| `apply` | `fn`, `arg` |
| `let` | `name`, `bound`, `body`（非递归） |
| `if` | `cond`, `then`, `else_` |
| `ref` / `read` / `write` | `inner` / `ref` / `ref, value`（write 返回 Unit） |

### v2 新增节点（`language="v2"`）

| tag | 字段 | 语义 |
|---|---|---|
| `record` | `fields: [[名, 表达式], ...]` | 记录字面量；字面重名直接拒绝；字面行闭合 |
| `get` | `record`, `field: str` | 字段访问；用开放行 `{field: a ; r}` 约束，缺字段即冲突 |
| `extend` | `record`, `field`, `value`, `rest` | 不改变原记录的扩展 `{e with f=v}`；`rest` 缺省为 `record` 自身；扩展已存在字段一律报错 |
| `raise` | `value` | 抛出，效应 `X`，结果类型任意（与最近 try 的处理器参数一致） |
| `try` | `body`, `param`, `handler` | 移除 body 的 `X`（`R`/`W` 保留）；body 与 handler 类型一致 |
| `evar` | （无） | 未决定应占位：类型全新自由变量，效应全新效应变量，结果标 `undecided` |

构造器：`A.record/get/extend/raise_/try_/evar`（另见 `A.make_module`、`A.xvar`）。

## 类型与渲染

`Int`/`Bool`/`Unit`、`a -> b`、`Ref(t)`、类型变量；v2 增加：

- 记录 `{f1: t1, ...}`（闭合）与开放行 `{f: a ; r}`（`r` 是行变量）；
- 函数潜在效应：`a -{R,W}-> b`；纯函数仍渲染为 `a -> b`（v1 完全一致）；
- 签名检查时的刚性变量渲染为 `$名`（报告内部使用）。

规范命名：类型变量按主类型先序首次出现取 `a..z,a1..`；行变量独立取
`r, r1, ...`；函数箭头右结合，参数位置加括号，`Ref(...)` 内部不加双括号。

## 效应语义（并列规则与缺省固定）

- **直接效应**随表达式**并集**累积：`read` 加 `R`、`write` 加 `W`、
  `raise` 加 `X`；`if` 取三支并集；`let` 取绑定与体并集；
  `record/get/extend` 取部件并集。
- **lambda 是值**：体效应不直接外溢，而收为函数类型的**潜在效应集合**；
  `apply` 在调用点放出被调函数的潜在效应。多态函数每次使用点取全新
  效应变量，不同调用互不钉死。
- **try 只处理异常**：结果效应 = `(body − {X}) ∪ handler`。
  `R`/`W` 不被 try 移除——“异常被处理后仍有写效应”是核心反例之一。
- 效应统一为精确集合（具体标签必须逐点一致）+ 效应变量吸收
  （`{β}` 对 `{W}` 得 `β↦{W}`；`{W}` 对 `{}` 直接冲突）。
- **未决**：`evar` 引入效应变量；结果仍含未绑定效应变量时
  `effects_undecided=True`，并如实返回，既不当纯，也不降级成“任意”。

## 记录与开放行规则

- 记录行是具名字段序列 + 行尾（行变量 `RVar` 或空尾 `REmpty`）；
  字段按名比对、按名排序。
- 共有字段逐点统一；开放尾可吸收对方独有字段；两侧都开放时引入共享新尾。
- **同名字段扩展一律报错**：`extend` 要求被扩展行的尾**不含**新字段，
  该“缺席约束”结构化留痕（`row_absences`）；后续统一若把同名字段塞回
  同一尾（开放行被强制成环/字段出现两次），由替换绑定处拦截。
- **occurs 检查**同时覆盖类型变量与行变量；行变量绑定到含自身的行即
  `InfiniteTypeError`。
- 不可变记录在 v2 是结构性值（`record` 字段全为值、`get/extend` 部件
  全为值时按值参与值限制）；可变引用仍走严格值限制。

## 模块系统

### 输入（字段含义固定）

```python
A.make_module(name, exports, body, imports=())
# exports: {导出名: 签名串}  —— 每个导出都必须有签名
# body:    {定义名: v2 表达式 AST}
# imports: 依赖模块名元组（文档/校验用；依赖也从 xvar 推导）
A.xvar(module, name)  # 跨模块外部引用，只能指向别的模块的签名导出
```

- 一个模块的 `body` 定义是**相互递归的 let-rec 组**：可以引用本模块其他
  定义（含自身）与依赖模块的导出（`xvar`）。
- **内部（未导出）定义不允许相互成环**：成环定义必须列入 `exports` 并给
  签名，否则报 `ModuleStructureError`。
- 模块可互相递归引用；依赖图取 AST 中 `xvar` 与 `imports` 的并集。

### 签名串（完全显式，无推断、无通配）

```
forall a, r. {f: a ; r} -> a          # 限定变量逗号或空格分隔，必须带 '.'
Int -{W}-> Unit
forall a. Ref(a) -{R,W,X}-> a
{a: Int, b: Bool}
```

- 自由（未在 `forall` 列出的）类型/行变量一律 `SignatureError`；
  效应标签只允许 `R`/`W`/`X`，不支持效应变量或通配——**不存在“任意
  类型/任意效应”降级**。
- 非函数导出必须是纯值（直接效应为空）；函数的效应契约写在箭头标注里。

### 检查顺序（确定性）

1. 结构与签名校验全部通过后，Tarjan 求 SCC，按**被依赖在前**的拓扑序；
   SCC 内部按模块名排序，模块导出按名排序——同一输入永远同一结果。
2. **两阶段**：先解析并登记每个 SCC 的全部导出签名，再检查各导出实现。
   同 SCC 内模块按彼此声称的签名互相可见（支持递归）。
3. 实现检查把签名限定的普通变量实例化为**刚性 skolem**、行变量实例化为
   **刚性行变量**；实现体必须与刚性签名逐点一致（实现不够一般即失败）。
   自递归引用在本导出体检查期间看到同一组刚性变量，多态递归无法被特化。

### 模块边界的值限制（不可偷偷泛化）

- 推断器记录所有在**顶层扩展式**中产生的自由变量（`unsafe_vars`），
  既包括本模块字面 `ref` 分配的单元变量，也包括经外部多态函数应用产生
  的自由变量。该事实一旦在顶层成立，在任何嵌套层级统一时都有效。
- 这类变量若被绑定到刚性签名变量（含经替换链/效应/行尾传播），报
  `SkolemEscapeError`；闭合可变记录被宣称为开放行多态引用时，
  报 `RecordFieldError`。
- 每次调用新建单元的函数（如 `λx. ref x : a -> Ref(a)`）是健全多态，
  正常通过——防线针对的是**共享/形状固定**的可变状态。

### 缓存（键、失效集、失败原子性）

- **缓存键** = 本模块实现体的规范哈希 + 每个**直接依赖**的签名指纹
  （指纹含效应标注）+ 节点预算。依赖实现体变化不影响本模块键（只有签名
  契约变化才影响）；这与实际约束依赖一致。
- **失效集** = 被改模块本身 ∪ 依赖图上所有（传递）依赖它的模块
  （`invalidation_set`）。改动一个导出后，只有失效集内模块重算，
  其余命中；**不靠清空全缓存**。
- **失败原子性**：重新检查在临时状态上进行；一个模块的任一导出检查失败，
  该模块上一次的成功缓存条目原样保留，异常照常抛出。
- **非缓存路径**：`check_all(use_cache=False)` 从零全查；测试用它对小而
  完整的程序与增量路径**逐项比对**（主类型、签名、效应、未决标志、节点数
  全部一致，见 `tests/test_modules.py::NonCacheParityTests`）。

## 错误语义

- **可定位**：结构错误带 AST `path`；统一失败带两端路径，随函数
  `[arg]/[ret]/[eff]`、引用 `[inner]`、记录 `[field]/<名>` 下钻。
- **资源拒绝 vs 证明无解严格分开**：`NodeLimitError`（超节点预算，没做
  推断）不同于 `TypeMismatchError`/`InfiniteTypeError`/`RecordFieldError`/
  `EffectMismatchError`（构造性无解）与 `SkolemEscapeError`（模块边界
  泛化逃逸）。推断确定、无回溯、无搜索，不存在“搜索预算耗尽”第三态。
- `evar` 的未决是**如实返回的结果状态**，不是异常。

## 独立参考检查器（`polyinfer.reference`）

- 自带替换应用（循环检测）、自由变量/刚性变量收集、结构同构（含行字段、
  潜在效应集合）、行尾展平；**不导入也不调用**核心的 `unify`/`infer`/
  模块检查器来取期望值。
- 核验：替换幂等无循环、**全部**记录约束在最终替换下两端同构（效应变量
  先经独立解析）、泛化集合（值限制由检查器从**源 AST** 独立重判）、
  开放行缺席约束（在最终替换上复核尾行不含缺席字段）。
- **效应独立重算**：`RefEffectInterpreter` 直接遍历源 AST（支持一层
  beta 内联：直接 lambda、let-lambda、形参实参映射），与结果的具体效应
  集合和未决标志逐条比对；它对真正跨不到的外部函数保守报未决，绝不拿
  unknown 当确定值。
- 模块报告 `check_module_report` 独立复核约束、泛化、缺席约束与
  **导出效应签名**（不信任求解器的任何布尔标志）。
- 篡改检测（`tests/test_tamper_v2.py`、`tests/test_reference.py`）：
  人为篡改效应集合、未决标志、替换（往缺席尾塞字段）、泛化集合、
  模块导出效应、约束两端，检查器都必须发现。

## 输入上限与确定性

- 单表达式 `infer(..., max_nodes=500)`（v1 默认与初版一致；可调）。
- 模块包预算 `ModuleChecker(modules, max_nodes=20000)`，按全包 AST 节点
  计（含 `xvar`）；超限 `NodeLimitError`。
- 整数为 Python 任意精度；本语言无浮点，float 输入（含 NaN/±Infinity）
  在 v1/v2 都被拒绝。
- 支持规模已验证：**≥12 个模块、≥3000 AST 节点、每记录 20 字段、
  4 个模块的递归 SCC**（`tests/test_modules.py::ScaleTests`，非退化输入，
  含大 if 链与跨模块开放行递归调用，实测约 0.05 秒）。

## 模块布局

```text
polyinfer/
  __init__.py    公开接口
  ast.py         AST 构造器、单表达式/模块结构与作用域校验
  types.py       类型（含开放行 TRow/RVar/REmpty、刚性 TSkol/RSkol、
                 带效应 TFun）、occurs/apply/bind、渲染、规范命名
  signature.py   导出签名串词法/语法解析与方案实例化
  inference.py   Algorithm W 核心：行统一、效应、值限制、skolem 逃逸防线
  modules.py     递归模块、Tarjan SCC、两阶段签名检查、增量缓存
  reference.py   独立参考检查器 + 独立效应解释器（不经过统一核心）
  errors.py      错误类型体系与路径渲染
tests/           v1 四件套（未删未放松）+ v2 记录/效应/模块/参考域/篡改
demo.py          固定输入演示
```

## 相对初版（1.x）的变更与迁移

逐条列出；没有需要旧调用改写的破坏：

1. **`TFun` 增加带默认值的效应字段**：`TFun(arg, ret)` 二参构造、相等性、
   纯函数渲染全部不变；效应只在 v2 非空时渲染为 `-{...}->`。
2. **`infer` 增加关键字参数 `language="v1"`**：缺省即旧行为；
   v2 语法在 v1 下仍按“未知 tag”拒绝（显式选项，不隐式启用）。
3. **新增错误类型** `RecordFieldError`/`EffectMismatchError`/
   `SkolemEscapeError`/`SignatureError` 与模块结构错误，都是
   `PolytypeError` 子类；旧的异常类型与抛出条件保持。
4. **新增类型节点** `TRec/TRow/RVar/REmpty/TSkol/RSkol` 与新 AST 节点，
   旧 AST/类型代码不受影响。
5. **`InferenceResult` 只新增带缺省的字段**（`language/effects/...`），
   旧字段语义与类型不变；v1 结果效应为空、`effects_checked` 在 v1 不启用。
6. 初版“不含记录/效应/模块”的已知限制在本版解除；v1 的严格值限制
   （而非弱化值限制）仍是有意取舍，v2 在其基础上增加结构性不可变值。

迁移办法：旧调用无需改动；要使用新能力时传 `language="v2"`，
模块场景用 `ModuleChecker`/`check_modules` 并为每个导出提供签名串。

## 已知限制（只按本契约实现，不宣称行业标准完整兼容）

- 不含词法/语法解析器——输入是结构化 AST；无多态递归推断（递归定义必须
  给显式签名，这正是模块契约）；签名效应只支持具体 `R/W/X` 集合，
  不含用户级效应变量声明（`evar` 仅作表达式级未决占位）。
- 行多态采用“缺席约束 + occurs”方案保证本契约内的健全性，未实现完整
  行约束求解器（如行合并子的全部化简变体）；不含行泛型高阶抽象的
  全部行业扩展。
- 无求值语义：本库只做静态推断，绝不执行输入 AST；`evar` 的未决是
  效应分析的诚实状态，不代表运行时行为。
