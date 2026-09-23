# polyinfer —— 带值限制的多态类型推断库（HM / 行多态 / 效应 / 模块）

为离线编程语言工程实现的可复用类型推断库（Algorithm W）。
纯 Python **3.14.7** 标准库，零第三方依赖；核心算法（统一、occurs、
替换组合、泛化、效应并查集、Tarjan SCC、skolem 子包含）全部从零
自实现，不调用任何第三方同类实现，也**不会用 Python 执行输入 AST**。

- **v1（缺省，完全向后兼容）**：变量、整数、布尔、lambda、apply、
  非递归 let、if、ref/read/write；值限制 + 引用不变式。
- **v2（显式 `version=2` 或模块入口）**：开放行记录（`record` /
  `field` / `extend`）、`read`/`write`/`exn` 效应、异常 `raise`/`try`、
  带显式签名、可互相递归的多模块包与增量检查缓存。

## 运行环境与命令

- Python **3.14.7**（Linux 原生；只用标准库）。
- 无需网络、账号、密钥、数据库、中间件、云 API 或 Docker。

```bash
# 快速回归（v1 全部旧测试 + v2 单元测试；不含规模用例）
python -m unittest discover -s tests -v

# 扩展验证（3000+ 节点表达式、12 模块/4 模块 SCC、20 字段记录、
# 增量 vs 全量逐项比对；确定性生成，约 1 秒）
python -m unittest discover -s extended_tests -v

# 固定输入演示：新能力主类型/效应、两个棘手边界、一次实际触发的
# 四模块 SCC 拒绝，以及增量缓存（实测约 0.1 秒，上限 90 秒内）
python demo.py
```

demo 不含 `sleep`、不读预录结果、不做压力测试。

## 公开接口

```python
from polyinfer import infer, A

# v1（旧入口，行为/异常/主类型渲染完全不变）
r1 = infer(A.lambda_("x", A.var("x")))
r1.principal_type            # "a -> a"

# v2：开放行记录 / 效应 / 异常
r2 = infer(node, version=2)
r2.principal_type            # 如 "Ref(a) -{read, write}-> Unit"
r2.effects                   # frozenset 根表达式本次求值的效应标签
r2.verify()                  # 独立参考检查器（替换/约束/泛化/效应事件）

# 模块包
from polyinfer import ModuleChecker, ModuleCache, check_package
pkg_result = ModuleChecker(package_node).check()
pkg_result.cache_hits / cache_misses / invalidated / order / modules
```

## 表达式 AST（普通 dict，可 JSON 化）

| tag | 字段 | 版本 |
|---|---|---|
| `var` | `name` | v1 |
| `int` / `bool` | `value`（int 拒绝 bool/float/NaN/Inf） | v1 |
| `lambda` / `apply` / `let`（非递归）/ `if` | 见 v1 | v1 |
| `ref` / `read` / `write` | 可变引用与读写 | v1 |
| `record` | `fields: [[name, expr], ...]`，**开放行** | v2 |
| `field` | `record`, `name` 字段访问 | v2 |
| `extend` | `record`, `fields` 非破坏式扩展 | v2 |
| `exn` | `name` 异常值，类型 `Exn` | v2 |
| `raise` | `value`（须 `Exn`），引入 `exn` 效应 | v2 |
| `try` | `body`, `handler`（须 `Exn -> t`） | v2 |
| `mvar` | `module`, `name` 跨模块导出引用（仅模块内） | v2 |

构造器见 `polyinfer.ast`（别名 `A`）：`A.record / A.field_ / A.extend /
A.raise_ / A.try_ / A.exn / A.mvar`。

## 模块包与签名

```python
{"modules": {
    "M": {"imports": ["N"], "exports": [
        {"name": "f",
         "signature": {"vars": ["a"], "row_vars": ["r"],
                       "type": <类型表达式>},
         "binding": <表达式 AST>}]}}}
```

- 每个导出**必须**有显式签名；多态递归只经签名发生。
- 类型表达式 AST：`tcon`（Int/Bool/Unit/Exn）、`tvar`、`rvar`、
  `tfun`（`effects: ["read","write","exn"]` 子集列表）、`tref`、
  `trecord`（`fields` 序列；`tail=None` 闭行，或 `rvar` 开放行）。
  构造器：`signature / export / module / package / tcon / tvar / rvar /
  tfun / tref / trecord`。
- 模块可互相递归：检查器对导入图求 Tarjan SCC，**先**确认 SCC 签名
  齐备，**再**按分量检查实现；SCC 内经彼此声明签名互相可见。

### 输入字段、边界含义与缺省选择（并列规则固定如下）

- **记录字面量默认开放行**：`{x:1}` 类型为 `{x:Int | R0}`，可传给只
  读 `x` 的函数，也可携带任意额外字段；要封闭性在**模块签名**里写
  闭行 `trecord(..., tail=None)`，签名检查对多余/缺失字段构造性拒绝。
- **同名字段扩展一律报错**：`extend` 不覆盖。确定的同名（字面量字段、
  扩展点已知字段、经行替换暴露出的确定字段、以及“先 extend 出 y 的
  记录再经尾行绑定暴露出同名字段”）都会被拒绝；原记录不被修改
  （非破坏式，结果记录与源记录共享尾行但字段列表独立）。
- **简单行多态（不做 absence 多态）**：在类型仍为抽象开放行时访问其
  字段，字段类型作为多态变量推断（喂给具体记录后由尾行吸收其余字段）；
  这是标准简单行多态的选择，确定性的缺字段在闭行签名处拒绝。
- **效应是潜伏标签集合**：函数类型 `a -{read, write}-> b`；纯函数仍是
  `a -> b`（v1 渲染不变）。应用累积被调/参数/潜伏效应的并集。
- **try 精确移除 `exn`**：`try body (λe.…)` 只去掉 body 已确定的
  `exn` 标签；`read`/`write` 与 handler 自身效应**保留**。
- **效应不参与量化**：签名里的效应只能是具体标签；实现效应必须是
  声明效应的子集（纯实现满足声明带效应，反之拒绝）。

## 类型与命名

- `Int`/`Bool`/`Unit`/`Exn`、`a -> b`、`a -{…}-> b`、`Ref(t)`、
  `{f: t, … | Rn}`（开放行）/ `{f: t, …}`（闭行）。
- 普通变量按类型结构先序首次出现命名 `a..z,a1..`；行变量独立计数
  `R0,R1,…`；内部未解析名仅出现在错误对象中（`?n` / `R?n` / `E?n`）。

## 边界泛化与三个跨能力反例

推断在 `let` 与模块边界做值限制泛化，并额外排除 **Tofte 危险变量**
（出现在任意 `Ref(...)` 内部、且不源自 lambda 参数的自由类型变量）。
携带可变引用的变量因此不能在模块边界被偷偷泛化。验收级反例（均有
测试与演示）：

1. **两个模块共享一个引用**：一个模块造 `Ref(a)`，另一模块写入
   `Int` 后读取；实现类型被钉成 `Ref(Int) -{read,write}-> Int`，
   签名若谎称 `∀a. Ref(a) -> a` 或漏报 write，均被
   `SignatureError` 拒绝；如实声明单态+write 才通过。
2. **开放行统一成环**：`λr. if true then r else extend(r,[z=r])`
   的行变量出现在自身展开式中，被 occurs 检查报 `InfiniteTypeError`。
3. **异常被处理后仍有写效应**：`try (r:=1; raise E) (λe.0)` 的类型
   是 `Ref(Int) -{write}-> Int`（exn 消失、write 保留）；跨模块签名
   若声称纯，被效应定向包含检查拒绝。

另有一个**跨四模块递归 SCC** 的端到端反例：`Cell/Box/Mut/Use` 互为
一个 SCC，`Box.pack : ∀aρ. Ref(a) -> {cell:Ref(a)|ρ}` 让开放行经递归
签名传递，`Mut.writeBox` 的引用写入把 `Use.leak` 的内容变量泛化资格
钉死——谎称 `∀aρ` 纯被拒绝；改成如实的 `Ref(Int) -{write}-> …`
（尾行变量可安全泛化为 `R0`）才通过。

## 增量缓存

`ModuleChecker(pkg, cache=ModuleCache()).check()` 以模块为粒度缓存：

- 缓存键 = 本模块完整源码（签名+绑定+导入表）指纹 ∪ 各依赖模块
  **声明签名**指纹；键与检查顺序、缓存状态无关（SCC 内也稳定）。
- 只改实现（对外签名不变）→ 仅本模块重查，依赖继续命中；
  改签名 → 仅沿反向依赖闭包、键实际变化的模块重查；
  与改动无依赖关系的模块始终命中——**绝不清空全缓存**。
- 一次检查失败**不替换**上一次有效结果（旧条目保留）。
- `check_without_cache()` 是完全独立的全量路径；测试对小程序与
  12 模块非退化包逐项（主类型/声明类型/效应/泛化/节点数）比较
  增量与全量，结果一致。`reverse_dependency_closure([m])` 返回
  理论失效闭包，测试核对实际重查集是其子集且含被改模块。

## 独立参考检查器（`polyinfer.reference`）

不导入也不调用核心的 `unify`/`infer`/模块检查器，只把结果当待审数据：

1. 替换良构（独立 occurs、循环检测、幂等，含行展开）；
2. 最终替换作用于**全部**记录约束两端后结构同构（函数逐点、引用
   不变式、开放行字段按名对齐、尾行同构）；
3. 每条泛化的集合恰为“绑定自由变量 − 环境自由变量 − 危险变量”，
   值/扩展式与 `record` 全值字段性由检查器据留痕 tag **独立重判**，
   危险变量豁免（lambda 参数）按 `exempt_vars` 留痕复核；
4. **效应**：独立并查集按序重放 `effect_log` 的 union/remove 事件，
   精确重建每个效应变量的标签集，与核心声称的 `eff_solution` 逐项
   相等（不是只看下界）；`try` 的 write 保留由此被独立验证。

期望值均手写或由测试内独立枚举规则产生；另有对约束、替换、循环、
泛化集合、值限制标记、效应解、效应日志、field_tags 留痕的**篡改
检测**测试，以及 `{read 否/是}×{write 否/是}` 完整效应格（4 元素，
非抽样）的穷举核对。

## 输入上限与资源语义

- 单表达式默认 `max_nodes=500`（`infer(ast, max_nodes=...)` 可调）；
  超限抛 `NodeLimitError`——**资源拒绝，不是类型无解**。
- 模块包默认 `max_nodes=5000`、`max_modules=64`
  （`ModuleChecker(pkg, max_nodes=..., max_modules=...)`）。
- 支持范围已实测：**3000+ 节点表达式、12 模块（含 4 模块 SCC）、
  每记录 20 字段**；扩展测试中 12 模块 / 3330 表达式节点全量检查约
  0.05 秒，3700+ 节点表达式约 2.6 秒（本机，仅说明可完成性）。
- 整数支持 Python 任意精度大整数；无浮点类型，float 输入被拒绝；
  本库只做静态推断，不解释执行 AST。
- 推断确定、无回溯、无搜索，因此没有“搜索预算耗尽”的第三状态；
  统一失败是构造性无解证明，资源超限是独立异常类。

## 与 v1 的兼容性 / 迁移

- 缺省 `version=1`：语法、错误类型与路径、`principal_type` 渲染、
  `InferenceResult` 前 7 个字段全部保持；旧的 78 个测试一字未改仍通过。
- v2 能力通过显式 `version=2` 或模块包入口启用（新增格式，不改旧格式）。
- 旧的严格值限制在 v1 维持；v2 的值形式额外包括“全值字段的 record”，
  并叠加危险变量规则。若原行为与新规则有表面差异，只在 `version=2`
  出现，v1 调用方零影响。

## 模块布局

```text
polyinfer/
  __init__.py    公开接口（v1 + v2）
  ast.py         AST/签名/模块包构造器与结构、作用域、上限校验
  types.py       类型（含 TRecord/TRow/TEffect、三类 TVar）、替换、渲染
  inference.py   Algorithm W：值限制+危险变量、开放行统一、效应并查集
  modules.py     签名解析、Tarjan SCC、skolem 子包含、增量缓存
  reference.py   独立参考检查器（替换/约束/泛化/效应事件）
  errors.py      错误体系与路径渲染
tests/           快速回归（unittest discover -s tests）
extended_tests/  规模与增量扩展验证（discover -s extended_tests）
demo.py          固定输入演示
```

## 已知限制（只按本契约实现）

- 不含词法/语法解析器，输入是结构化 AST；不含递归 let/多态递归定义
  （多态递归仅在模块间经显式签名发生）、高阶多态（`∀f.…`）、
  变体/ADT/元组/列表、kinding 系统、子类型（记录是 HM 开放行不变式，
  不是宽度子类型）。
- 效应是 `{read,write,exn}` 三标签的扁平集合，不含效应多态/效应行；
  `try` 只移除确定的 `exn` 标签，不拆分未决效应变量。
- 简单行多态：不跟踪字段缺失（absence）约束；抽象开放行上访问未知
  字段按多态变量处理，确定性缺字段在闭行签名处拒绝。
- 严格/语法值限制（v2 放宽到全值字段 record）：部分在弱化值限制下
  可泛化的良性扩展式在此保守单态，这是有意的安全取舍。
- `Unit` 无字面量，只作为 `write` 结果出现；`Exn` 是单一抽象异常
  类型（异常值靠 `exn` 字面量的名字区分，不做异常变体分类）。
