"""AST 定义、构造器与结构校验。

公开 AST 是普通 dict（可 JSON 化），固定 ``tag`` 字段区分类别：

=================  =========================================================
tag                字段
=================  =========================================================
``"var"``          ``name: str``
``"int"``          ``value: int``（bool 被拒绝）
``"bool"``         ``value: bool``
``"lambda"``       ``param: str``, ``body: Node``
``"apply"``        ``fn: Node``, ``arg: Node``
``"let"``          ``name: str``, ``bound: Node``, ``body: Node``
``"if"``           ``cond: Node``, ``then: Node``, ``else_: Node``
``"ref"``          ``inner: Node``       —— 创建引用（等价 ML 的 ref e）
``"read"``         ``ref: Node``         —— 解引用（!e）
``"write"``        ``ref: Node``, ``value: Node`` —— 赋值（e1 := e2），返回 Unit
=================  =========================================================

构造器 :func:`var` ... :func:`write_` 只做最薄的一层封装，便于手写
示例与测试；所有硬校验集中在 :class:`AstValidator`，``infer`` 入口会调用。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

from .errors import InvalidAstError

# 节点种类 -> 必需字段（顺序即文档与错误路径使用的顺序）。
NODE_FIELDS: Dict[str, Tuple[str, ...]] = {
    "var": ("name",),
    "int": ("value",),
    "bool": ("value",),
    "lambda": ("param", "body"),
    "apply": ("fn", "arg"),
    "let": ("name", "bound", "body"),
    "if": ("cond", "then", "else_"),
    "ref": ("inner",),
    "read": ("ref",),
    "write": ("ref", "value"),
}

Node = Dict[str, Any]
Path = Tuple[str, ...]

Node = Dict[str, Any]
Path = Tuple[str, ...]
ObligationId = Tuple[str, Path]

# 顶层统一等式（"义务"）的种类与"挂点"路径（挂点就是该义务由哪个 AST
# 节点诱发）。结构下钻（[arg]/[ret]/[inner]）不产生新义务。
# 诊断模式据此给输入约束稳定编号；顺序与种类表只取决于 AST 本身，
# 不取决于推断状态。
OBLIGATION_KINDS = ("apply", "if_cond", "if_branches", "read", "write")


def obligation_id(kind: str, path: Path) -> ObligationId:
    """一条顶层统一义务的稳定身份：(种类, 诱发节点的 AST 路径)。"""
    return (kind, path)


# ---------------------------------------------------------------- 构造器

def var(name: str) -> Node:
    return {"tag": "var", "name": name}


def int_(value: int) -> Node:
    return {"tag": "int", "value": value}


def bool_(value: bool) -> Node:
    return {"tag": "bool", "value": value}


def lambda_(param: str, body: Node) -> Node:
    return {"tag": "lambda", "param": param, "body": body}


def apply(fn: Node, arg: Node) -> Node:
    return {"tag": "apply", "fn": fn, "arg": arg}


def let(name: str, bound: Node, body: Node) -> Node:
    return {"tag": "let", "name": name, "bound": bound, "body": body}


def if_(cond: Node, then: Node, else_: Node) -> Node:
    return {"tag": "if", "cond": cond, "then": then, "else_": else_}


def ref(inner: Node) -> Node:
    return {"tag": "ref", "inner": inner}


def read(ref_node: Node) -> Node:
    return {"tag": "read", "ref": ref_node}


def write_(ref_node: Node, value: Node) -> Node:
    return {"tag": "write", "ref": ref_node, "value": value}


# ---------------------------------------------------------------- 校验

class AstValidator:
    """深度校验整棵 AST。

    任何不合法都抛 :class:`InvalidAstError`，路径定位到最小相关子树。
    规则要点：

    * 节点必须是 ``dict``，含且仅含 ``tag`` 与声明字段；
    * 标识符必须是非空 ``str``；
    * ``int.value`` 必须是真正的 ``int``——``bool`` 是 ``int`` 子类，显式拒绝；
      ``float`` 不隐式转换，NaN/Infinity（若经由 float 出现）同样拒绝；
    * ``bool.value`` 必须是真正的 ``bool``；
    * 不支持递归 let：``let`` 的绑定表达式中引用同名变量即报错
      （这是语法级契约，由作用域检查顺带完成）。
    """

    def __init__(self, max_nodes: int) -> None:
        self.max_nodes = max_nodes
        self.count = 0

    def validate(self, node: Any) -> None:
        self._check(node, (), scope=())

    # -- 内部 -------------------------------------------------------------

    def _fail(self, path: Path, message: str) -> None:
        raise InvalidAstError(path, message)

    def _check(self, node: Any, path: Path, scope: Tuple[str, ...]) -> None:
        if not isinstance(node, dict):
            self._fail(path, f"节点必须是 dict，实际得到 {type(node).__name__}")
        tag = node.get("tag")
        if not isinstance(tag, str):
            self._fail(path, "缺少字符串字段 'tag' 或其值不是字符串")
        fields = NODE_FIELDS.get(tag)
        if fields is None:
            self._fail(path, f"未知节点 tag {tag!r}")
        extra = set(node) - {"tag", *fields}
        if extra:
            self._fail(path, f"存在多余字段 {sorted(extra)}；{tag} 节点只允许 {list(fields)}")
        missing = [f for f in fields if f not in node]
        if missing:
            self._fail(path, f"缺少字段 {missing}")

        self.count += 1
        if self.count > self.max_nodes:
            # 直接向上抛资源错误；这是纯输入校验，尚未产生任何外部状态变更。
            from .errors import NodeLimitError

            raise NodeLimitError(self.count, self.max_nodes)

        if tag == "var":
            name = node["name"]
            self._check_name(path, name)
            if name not in scope:
                from .errors import UnboundVariableError

                raise UnboundVariableError(path, name)
        elif tag == "int":
            self._check_int(path, node["value"])
        elif tag == "bool":
            value = node["value"]
            if not isinstance(value, bool):
                self._fail(path, f"'bool' 节点的 value 必须是 bool，实际得到 {type(value).__name__}")
        elif tag in ("lambda", "let"):
            name_field = "param" if tag == "lambda" else "name"
            self._check_name(path, node[name_field])
        # let 的非递归检查在绑定子树的 scope 处理里完成。

        # 递归校验子节点
        if tag == "lambda":
            self._check(node["body"], path + ("body",), scope + (node["param"],))
        elif tag == "apply":
            self._check(node["fn"], path + ("fn",), scope)
            self._check(node["arg"], path + ("arg",), scope)
        elif tag == "let":
            name = node["name"]
            # 非递归 let：绑定表达式的作用域不含 name 自身，
            # 因此绑定树中引用同名变量会以 UnboundVariableError 被拒绝。
            self._check(node["bound"], path + ("bound",), scope)
            self._check(node["body"], path + ("body",), scope + (name,))
        elif tag == "if":
            self._check(node["cond"], path + ("cond",), scope)
            self._check(node["then"], path + ("then",), scope)
            self._check(node["else_"], path + ("else_",), scope)
        elif tag == "ref":
            self._check(node["inner"], path + ("inner",), scope)
        elif tag == "read":
            self._check(node["ref"], path + ("ref",), scope)
        elif tag == "write":
            self._check(node["ref"], path + ("ref",), scope)
            self._check(node["value"], path + ("value",), scope)

    def _check_name(self, path: Path, name: Any) -> None:
        if not isinstance(name, str):
            self._fail(path, f"标识符必须是 str，实际得到 {type(name).__name__}")
        if not name:
            self._fail(path, "标识符不能为空字符串")

    def _check_int(self, path: Path, value: Any) -> None:
        # bool 是 int 的子类型，按契约显式拒绝。
        if isinstance(value, bool):
            self._fail(path, "'int' 节点不接受 bool（bool 必须使用 'bool' 节点）")
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                self._fail(path, f"拒绝非有限数值 {value!r}，且 float 不会隐式转为 int")
            self._fail(path, f"'int' 节点只接受 int，float 不会被隐式截断（实际 {value!r}）")
        if not isinstance(value, int):
            self._fail(path, f"'int' 节点的 value 必须是 int，实际得到 {type(value).__name__}")
