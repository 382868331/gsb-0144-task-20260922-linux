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
    """使用了未在作用域中绑定的变量。

    普通 ``let`` 非递归（绑定里引用自身报此错）；自递归/互相递归须用
    ``letrec`` 组，组内未声明的名字同样报此错。
    """

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
