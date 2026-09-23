"""错误类型体系。

所有错误都可定位：

* 结构/输入错误带 AST 路径 (``path``)；
* 统一失败带约束两端各自的 AST 路径 (``left_path``/``right_path``)。

资源上限 (:class:`NodeLimitError`) 与“已证明无解”
(:class:`UnificationError`、:class:`UnboundVariableError`) 是不同的类，
调用方可以据此区分“输入太大没推”与“程序确实不可类型化”。
本推断器是确定性的 Algorithm W，不存在搜索预算耗尽这一状态——
每次统一失败都是该约束无解的构造性证据。

v2 新增（记录/行、效应、模块）的错误同样是确定的构造性拒绝：
:class:`DuplicateFieldError`（同名扩展）、
:class:`SignatureError`（导出签名与实现不一致，含 skolem 逸出）、
:class:`ModuleStructureError`（模块结构/依赖/导出缺失）。
"""

from __future__ import annotations

from typing import Any, Tuple

Path = Tuple[Any, ...]


def format_path(path: Path) -> str:
    """把 AST 路径渲染成 ``/root/body/fn`` 形式。"""
    if not path:
        return "<root>"
    return "<root>/" + "/".join(str(step) for step in path)


class PolytypeError(Exception):
    """本库所有错误的公共基类。"""


class InvalidAstError(PolytypeError):
    """AST 结构不合法（非节点、字段缺失/多余、名字非字符串、字面量非法等）。

    永远不做静默纠正：bool 不接受为 int、float（含 NaN/Infinity）不隐式转 int。
    """

    def __init__(self, path: Path, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"非法 AST（位置 {format_path(path)}）：{message}")


class NodeLimitError(PolytypeError):
    """节点数超过接口上限（v1 单表达式默认 500）。这是资源拒绝，不是类型无解。"""

    def __init__(self, count: int, limit: int) -> None:
        self.count = count
        self.limit = limit
        super().__init__(f"AST 节点数 {count} 超过上限 {limit}")


class UnboundVariableError(PolytypeError):
    """使用了未在作用域中绑定的变量（非递归 let，lambda 参数为单态绑定）。"""

    def __init__(self, path: Path, name: str) -> None:
        self.path = path
        self.name = name
        super().__init__(f"未绑定变量 {name!r}（位置 {format_path(path)}）")


class UnificationError(PolytypeError):
    """统一失败的公共基类，携带失败约束两端的 AST 路径与类型。"""

    def __init__(
        self,
        left_path: Path,
        right_path: Path,
        left_type: Any,
        right_type: Any,
        message: str,
    ) -> None:
        self.left_path = left_path
        self.right_path = right_path
        # 叶子冲突位置上的两个类型（函数与函数统一时，深入到真正冲突的那对）。
        self.left_type = left_type
        self.right_type = right_type
        super().__init__(
            f"{message}：左端 {left_type} @ {format_path(left_path)}；"
            f"右端 {right_type} @ {format_path(right_path)}"
        )


class TypeMismatchError(UnificationError):
    """两个不同外构造子的类型无法统一（已证明无解）。"""

    def __init__(self, left_path: Path, right_path: Path, left_type: Any, right_type: Any) -> None:
        super().__init__(left_path, right_path, left_type, right_type, "类型不一致")


class InfiniteTypeError(UnificationError):
    """occurs 检查失败：类型/行变量出现在自身展开式中，构成无限类型（已证明无解）。"""

    def __init__(
        self,
        left_path: Path,
        right_path: Path,
        variable: Any,
        expanded: Any,
    ) -> None:
        self.variable = variable
        self.expanded = expanded
        super().__init__(
            left_path,
            right_path,
            variable,
            expanded,
            f"无限类型：{variable} 出现在 {expanded} 中",
        )


class EffectMismatchError(UnificationError):
    """效应冲突：把实际带效应的函数当作纯函数使用等（已证明无解）。

    效应统一本身是并集（只会累积，不冲突）；本错误用于函数类型逐点
    比较中效应集合不相容的定位场景（如签名检查里声明纯、实现有写效应）。
    """

    def __init__(self, left_path: Path, right_path: Path, left_eff: Any, right_eff: Any) -> None:
        super().__init__(
            left_path,
            right_path,
            left_eff,
            right_eff,
            "效应集合不相容",
        )


class DuplicateFieldError(PolytypeError):
    """记录扩展时同名字段已存在（含开放行经替换暴露出的同名冲突）。"""

    def __init__(self, path: Path, field: str) -> None:
        self.path = path
        self.field = field
        super().__init__(
            f"记录字段 {field!r} 重复（位置 {format_path(path)}）："
            "扩展不允许覆盖已有字段，原记录不变"
        )


class MissingFieldError(UnificationError):
    """字段访问时闭行确定缺少该字段（已证明无解）。"""

    def __init__(self, left_path: Path, right_path: Path, left_type: Any, right_type: Any) -> None:
        super().__init__(
            left_path, right_path, left_type, right_type, "记录缺少被访问字段"
        )


# ---------------------------------------------------------------- 模块层

class ModuleError(PolytypeError):
    """模块结构/依赖类错误的公共基类。"""


class ModuleStructureError(ModuleError):
    """模块包结构非法：模块缺失、import 未知、导出无绑定/无签名、字段非法等。"""

    def __init__(self, message: str, module: str | None = None) -> None:
        self.module = module
        where = f"（模块 {module!r}）" if module is not None else ""
        super().__init__(f"模块结构非法{where}：{message}")


class SignatureError(ModuleError):
    """实现与其导出签名不一致（含被引用写入污染的变量越界泛化、skolem 逸出）。"""

    def __init__(self, module: str, export: str, detail: str) -> None:
        self.module = module
        self.export = export
        self.detail = detail
        super().__init__(
            f"模块 {module!r} 的导出 {export!r} 未通过签名检查：{detail}"
        )
