"""错误类型体系。

所有错误都可定位：

* 结构/输入错误带 AST 路径 (``path``)；
* 统一失败带约束两端各自的 AST 路径 (``left_path``/``right_path``)。

资源上限 (:class:`NodeLimitError`) 与“已证明无解”
(:class:`UnificationError`、:class:`UnboundVariableError`) 是不同的类，
调用方可以据此区分“输入太大没推”与“程序确实不可类型化”。
本推断器是确定性的 Algorithm W，不存在搜索预算耗尽这一状态——
每次统一失败都是该约束无解的构造性证据。
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
    """节点数超过接口上限（默认 500）。这是资源拒绝，不是类型无解。"""

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
    """occurs 检查失败：类型变量出现在自身展开式中，会构成无限类型（已证明无解）。"""

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


class RecordFieldError(PolytypeError):
    """记录字段相关的不可满足：缺字段、同名扩展、开放行展开后字段重复。

    与 :class:`TypeMismatchError` 分开，便于调用方区分“行/字段契约”冲突
    与普通构造子冲突；它同样是构造性无解，不是资源拒绝。
    """

    def __init__(self, left_path: Path, right_path: Path, field: Any, message: str) -> None:
        self.left_path = left_path
        self.right_path = right_path
        self.field = field
        super().__init__(
            f"记录字段冲突（字段 {field!r}）：{message}；"
            f"左端 {format_path(left_path)}，右端 {format_path(right_path)}"
        )


class EffectMismatchError(PolytypeError):
    """函数潜在效应集合无法与签名/目标一致（具体效应标签冲突）。"""

    def __init__(
        self,
        left_path: Path,
        right_path: Path,
        left_eff: frozenset,
        right_eff: frozenset,
        note: str,
    ) -> None:
        self.left_path = left_path
        self.right_path = right_path
        self.left_eff = left_eff
        self.right_eff = right_eff

        def fmt(e):
            return "{" + ",".join(sorted(e)) + "}" if e else "{}"

        super().__init__(
            f"效应不一致（{note}）：{fmt(left_eff)} ≠ {fmt(right_eff)}；"
            f"左端 {format_path(left_path)}，右端 {format_path(right_path)}"
        )


class SkolemEscapeError(PolytypeError):
    """模块边界泛化逃逸：0 层可变引用单元中的类型变量被绑定到刚性签名变量。

    这正是“携带可变引用的类型变量不能被模块边界偷偷泛化”的构造性拒绝。
    """

    def __init__(self, cell_var: Any, target: Any, skolems: list) -> None:
        self.cell_var = cell_var
        self.target = target
        self.skolems = skolems
        super().__init__(
            f"模块边界逃逸：顶层可变引用单元变量 {cell_var} 被要求为 "
            f"{target}（含刚性签名变量 {', '.join(str(s) for s in skolems)}）；"
            f"携带可变引用的类型变量不得经导出签名泛化"
        )
