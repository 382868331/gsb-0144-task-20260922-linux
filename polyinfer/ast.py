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
``"letrec"``       ``binds: [{name, bound}, ...]``, ``body: Node``
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
    "letrec": ("binds", "body"),
    "if": ("cond", "then", "else_"),
    "ref": ("inner",),
    "read": ("ref",),
    "write": ("ref", "value"),
}

Node = Dict[str, Any]
Path = Tuple[str, ...]


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


def letrec(binds, body: Node) -> Node:
    """互相递归（含单函数自递归）绑定组。

    ``binds`` 接受 ``(name, bound)`` 二元组或 ``{"name", "bound"}`` dict，
    至少一个绑定，名字不得重复；所有绑定彼此可见、也对 ``body`` 可见。
    """
    entries = []
    for index, item in enumerate(binds):
        if isinstance(item, tuple) and len(item) == 2:
            bname, bound = item
            entry = {"name": bname, "bound": bound}
        elif isinstance(item, dict) and "name" in item and "bound" in item:
            entry = {"name": item["name"], "bound": item["bound"]}
        else:
            raise TypeError(
                f"letrec 第 {index} 个绑定必须是 (name, bound) 或 "
                "{'name', 'bound'}，实际得到 "
                f"{type(item).__name__ if not isinstance(item, dict) else sorted(item)}"
            )
        entries.append(entry)
    return {"tag": "letrec", "binds": entries, "body": body}


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
    * ``let`` 保持非递归：绑定表达式作用域不含同名变量，引用同名即报错；
      互相递归/自递归必须显式使用 ``letrec`` 绑定组。
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

        if tag == "letrec":
            self._check_letrec_binds(node, path, scope)

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
        # letrec 的子树校验集中在 _check_letrec_binds，避免重复计数。
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

    def _check_letrec_binds(self, node: Node, path: Path, scope: Tuple[str, ...]) -> None:
        binds = node["binds"]
        if not isinstance(binds, list):
            self._fail(path, f"'letrec' 节点的 binds 必须是 list，实际得到 {type(binds).__name__}")
        if not binds:
            self._fail(path, "'letrec' 至少需要一个绑定（自递归也是单元素组）")

        names = []
        binds_path = path + ("binds",)
        for index, entry in enumerate(binds):
            entry_path = binds_path + (f"[{index}]",)
            if not isinstance(entry, dict):
                self._fail(entry_path, f"绑定必须是 dict，实际得到 {type(entry).__name__}")
            extra = set(entry) - {"name", "bound"}
            if extra:
                self._fail(entry_path, f"存在多余字段 {sorted(extra)}；绑定只允许 name/bound")
            if "name" not in entry or "bound" not in entry:
                self._fail(entry_path, "绑定缺少 name 或 bound 字段")
            self._check_name(entry_path, entry["name"])
            names.append(entry["name"])

        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            self._fail(binds_path, f"'letrec' 绑定名不得重复：{dupes}")

        # 组内每个名字在组内 lambda 未引入遮蔽前对全部绑定可见；
        # lambda 自身的参数遮蔽交给递归检查时的 scope 叠加处理。
        group_scope = scope + tuple(names)
        for index, entry in enumerate(binds):
            self._check(
                entry["bound"],
                binds_path + (f"[{index}]", "bound"),
                group_scope,
            )
        self._check(node["body"], path + ("body",), group_scope)

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
