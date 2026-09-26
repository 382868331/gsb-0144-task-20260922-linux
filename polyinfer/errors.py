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


# ------------------------------------------------------------ 签名检查错误


class SignatureError(PolytypeError):
    """let 绑定上的 rank-1 类型签名相关错误的公共基类。"""

    def __init__(self, path: Path, name: str, message: str) -> None:
        self.path = path          # let 节点位置
        self.name = name          # 被绑定的名字
        self.message = message
        super().__init__(f"签名错误（let {name!r}，位置 {format_path(path)}）：{message}")


class SignatureMismatchError(SignatureError):
    """绑定表达式的类型与签名声称的类型对不上。"""

    def __init__(
        self,
        path: Path,
        name: str,
        left_path: Path,
        right_path: Path,
        left_type: Any,
        right_type: Any,
        detail: str,
    ) -> None:
        self.left_path = left_path
        self.right_path = right_path
        self.left_type = left_type
        self.right_type = right_type
        super().__init__(
            path,
            name,
            f"签名与实现不匹配（{detail}）：左端 {left_type} @ "
            f"{format_path(left_path)}；右端 {right_type} @ {format_path(right_path)}",
        )


class RigidEscapeError(SignatureError):
    """签名量化的刚性变量逃逸到签名作用域之外。"""

    def __init__(self, path: Path, name: str, variable: str, where: str) -> None:
        self.variable = variable
        self.where = where
        super().__init__(
            path,
            name,
            f"刚性变量 {variable!r} 逃逸出签名作用域（出现在{where}）；"
            "签名量化的类型变量不能穿过相邻/外层绑定",
        )


class ValueRestrictionConflictError(SignatureError):
    """签名要求泛化（含量化变量），但绑定表达式是扩展式。"""

    def __init__(self, path: Path, name: str, bound_tag: str, quantified: Tuple[str, ...]) -> None:
        self.bound_tag = bound_tag
        self.quantified = tuple(quantified)
        super().__init__(
            path,
            name,
            f"签名量化了 {list(quantified)}，但绑定表达式是扩展式 "
            f"{bound_tag!r}；值限制不允许对其泛化",
        )
