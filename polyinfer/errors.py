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


class SignatureMismatchError(PolytypeError):
    """显式 rank-1 签名与绑定表达式实际类型不兼容。

    签名检查时，绑定体在刚性类型变量下被推导，最后把推导出的类型与
    签名类型统一；该统一失败即签名过窄/形状不符（构造子冲突、
    无限类型或两个不同刚性变量被迫合并）。携带 let 绑定位置与
    失败统一深入到的两端路径和叶子类型。
    """

    def __init__(
        self,
        let_path: Path,
        name: str,
        signature_text: str,
        left_path: Path,
        right_path: Path,
        left_type: Any,
        right_type: Any,
        reason: str,
    ) -> None:
        self.path = let_path
        self.name = name
        self.signature = signature_text
        self.left_path = left_path
        self.right_path = right_path
        self.left_type = left_type
        self.right_type = right_type
        self.reason = reason
        super().__init__(
            f"let {name!r} 的显式签名 {signature_text!r} 与其实现不兼容（{reason}）："
            f"实现类型 {left_type} @ {format_path(left_path)}；"
            f"签名类型 {right_type} @ {format_path(right_path)}"
        )


class EscapedRigidVariableError(PolytypeError):
    """签名的刚性类型变量逃逸到了绑定之外的外层环境。

    典型反例：签名比实现更“宽”，而实现捕获了外层的单态变量，
    例如 ``λx. let f : forall a. a -> a = λy. x in ...``——实现体
    要求参数类型等于外层已被固定的单态变量，刚性变量 ``a`` 会被
    外层推断变量吞并，若允许泛化，调用两次不同类型就会污染外层。
    """

    def __init__(self, let_path: Path, name: str, signature_text: str, variable: Any, bound_path: Path) -> None:
        self.path = let_path
        self.name = name
        self.signature = signature_text
        self.variable = variable
        self.bound_path = bound_path
        super().__init__(
            f"let {name!r} 的签名 {signature_text!r} 中刚性变量 {variable} 逃逸出绑定："
            f"它被绑定表达式（{format_path(bound_path)}）中来自外层作用域的单态变量约束，"
            f"该绑定不能获得声明的多态方案"
        )


class ValueRestrictionConflictError(PolytypeError):
    """扩展式绑定的显式签名要求多态，与值限制冲突。

    只有语法值（字面量、lambda、变量）才能承载量化变量；
    ``ref e``、函数应用等扩展式即便写了含 ``forall`` 的签名，
    也不能包装成可任意实例化的多态值（引用多态漏洞的显式版本）。
    """

    def __init__(self, let_path: Path, name: str, signature_text: str, bound_tag: str, bound_path: Path) -> None:
        self.path = let_path
        self.name = name
        self.signature = signature_text
        self.bound_tag = bound_tag
        self.bound_path = bound_path
        super().__init__(
            f"let {name!r} 的签名 {signature_text!r} 要求多态，但绑定表达式 "
            f"{bound_tag!r}（{format_path(bound_path)}）是扩展式而非语法值："
            f"值限制下不能把单态引用/计算结果包装成可任意实例化的多态值"
        )
