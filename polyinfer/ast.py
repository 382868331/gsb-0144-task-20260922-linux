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

类型签名（``let`` 的可选 ``sig`` 字段）也是普通 dict::

    {"forall": ["a", "b"], "type": <SigType>}

其中 ``<SigType>`` 递归定义为：

    - ``tvar``: ``name: str``（必须出现在 ``forall`` 列表里）
    - ``tcon``: ``name: str``（``Int`` / ``Bool`` / ``Unit``）
    - ``tfun``: ``arg``, ``ret``
    - ``tref``: ``inner``

签名是 rank-1 的：量化只发生在最外层，签名类型内部不出现 ``forall``。
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

# 签名类型（SigType）节点种类 -> 必需字段。
SIG_TYPE_FIELDS: Dict[str, Tuple[str, ...]] = {
    "tvar": ("name",),
    "tcon": ("name",),
    "tfun": ("arg", "ret"),
    "tref": ("inner",),
}

SIG_CON_NAMES = frozenset({"Int", "Bool", "Unit"})


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


def let(name: str, bound: Node, body: Node, sig: "Node | None" = None) -> Node:
    node: Node = {"tag": "let", "name": name, "bound": bound, "body": body}
    if sig is not None:
        node["sig"] = sig
    return node


def if_(cond: Node, then: Node, else_: Node) -> Node:
    return {"tag": "if", "cond": cond, "then": then, "else_": else_}


def ref(inner: Node) -> Node:
    return {"tag": "ref", "inner": inner}


def read(ref_node: Node) -> Node:
    return {"tag": "read", "ref": ref_node}


def write_(ref_node: Node, value: Node) -> Node:
    return {"tag": "write", "ref": ref_node, "value": value}


# ------------------------------------------------------- 签名构造器

def sig(quantified, sig_type: Node) -> Node:
    """rank-1 签名：forall quantified . sig_type。"""
    return {"forall": list(quantified), "type": sig_type}


def tvar(name: str) -> Node:
    return {"tag": "tvar", "name": name}


def tcon(name: str) -> Node:
    return {"tag": "tcon", "name": name}


def tfun(arg: Node, ret: Node) -> Node:
    return {"tag": "tfun", "arg": arg, "ret": ret}


def tref(inner: Node) -> Node:
    return {"tag": "tref", "inner": inner}


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
        allowed = {"tag", *fields}
        if tag == "let":
            allowed.add("sig")  # let 可携带可选的 rank-1 类型签名
        extra = set(node) - allowed
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
        if tag == "let" and "sig" in node:
            self._check_sig(node["sig"], path + ("sig",))
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

    # -- 类型签名 -----------------------------------------------------------

    def _check_sig(self, sig_node: Any, path: Path) -> None:
        """校验 rank-1 签名结构：{"forall": [...], "type": <SigType>}。

        签名里出现的每个类型变量名都必须被本签名的 forall 量化——
        类型变量的作用域就是这一处签名，不能穿过相邻绑定。
        """
        if not isinstance(sig_node, dict):
            self._fail(path, f"签名必须是 dict，实际得到 {type(sig_node).__name__}")
        extra = set(sig_node) - {"forall", "type"}
        if extra:
            self._fail(path, f"签名存在多余字段 {sorted(extra)}；只允许 'forall' 与 'type'")
        for field in ("forall", "type"):
            if field not in sig_node:
                self._fail(path, f"签名缺少字段 {field!r}")
        quantified = sig_node["forall"]
        if not isinstance(quantified, (list, tuple)):
            self._fail(path + ("forall",), f"'forall' 必须是名字列表，实际得到 {type(quantified).__name__}")
        seen = set()
        for qname in quantified:
            self._check_name(path + ("forall",), qname)
            if qname in seen:
                self._fail(path + ("forall",), f"量化变量 {qname!r} 重复")
            seen.add(qname)
        self._check_sig_type(sig_node["type"], path + ("type",), seen)

    def _check_sig_type(self, node: Any, path: Path, quantified: set) -> None:
        if not isinstance(node, dict):
            self._fail(path, f"签名类型节点必须是 dict，实际得到 {type(node).__name__}")
        tag = node.get("tag")
        fields = SIG_TYPE_FIELDS.get(tag)
        if fields is None:
            self._fail(path, f"未知签名类型 tag {tag!r}（签名是 rank-1，类型内不允许嵌套 forall）")
        extra = set(node) - {"tag", *fields}
        if extra:
            self._fail(path, f"存在多余字段 {sorted(extra)}；{tag} 节点只允许 {list(fields)}")
        missing = [f for f in fields if f not in node]
        if missing:
            self._fail(path, f"缺少字段 {missing}")
        self.count += 1
        if self.count > self.max_nodes:
            from .errors import NodeLimitError

            raise NodeLimitError(self.count, self.max_nodes)
        if tag == "tvar":
            name = node["name"]
            self._check_name(path, name)
            if name not in quantified:
                self._fail(
                    path,
                    f"签名类型变量 {name!r} 未被本签名的 'forall' 量化；"
                    "类型变量作用域仅限本签名，不能引用外层绑定的变量",
                )
        elif tag == "tcon":
            name = node["name"]
            if name not in SIG_CON_NAMES:
                self._fail(path, f"未知基础类型 {name!r}，只允许 {sorted(SIG_CON_NAMES)}")
        elif tag == "tfun":
            self._check_sig_type(node["arg"], path + ("arg",), quantified)
            self._check_sig_type(node["ret"], path + ("ret",), quantified)
        elif tag == "tref":
            self._check_sig_type(node["inner"], path + ("inner",), quantified)
