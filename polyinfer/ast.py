"""AST 定义、构造器与结构校验。

公开 AST 是普通 dict（可 JSON 化），固定 ``tag`` 字段区分类别。

v1 节点（语言版本 1，``infer(...)`` 的默认行为，与初版完全一致）
================================================================

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
``"write"``        ``ref: Node``, ``value: Node`` —— 赋值（e1 := e2），Unit
=================  =========================================================

v2 新增节点（``infer(..., language="v2")`` 或模块推断时启用）
==============================================================

=================  =========================================================
tag                字段
=================  =========================================================
``"record"``       ``fields: [[name, Node], ...]``  —— 记录（默认开放）
``"get"``          ``record: Node``, ``field: str`` —— 字段访问
``"extend"``       ``record: Node``, ``field: str``, ``value: Node``,
                   ``rest: Node | None`` —— 不改变原记录的扩展；
                   ``rest`` 缺省为被扩展记录本身（``{e with f = v}``）
``"raise"``        ``value: Node``        —— 抛出（效应 X；结果任意类型）
``"try"``          ``body: Node``, ``param: str``, ``handler: Node``
                   —— 异常处理：移除 body 的 X 效应（W/R 保留）
``"evar"``         —— 零字段：未决定应的占位（参考域中按未知处理）
=================  =========================================================

模块（v2 模块系统）：见 :class:`ModuleBundle` 与 :func:`make_module`。
模块可互相递归引用；每个导出都必须带显式签名。

构造器只做最薄的一层封装；所有硬校验集中在 :class:`AstValidator`
（单表达式）与 :class:`ModuleValidator`（模块包）。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

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
    # ---- v2 ----
    "record": ("fields",),
    "get": ("record", "field"),
    "extend": ("record", "field", "value", "rest"),
    "raise": ("value",),
    "try": ("body", "param", "handler"),
    "evar": (),
}

# 各语言版本允许的节点 tag。
V1_TAGS = frozenset(
    {"var", "int", "bool", "lambda", "apply", "let", "if", "ref", "read", "write"}
)
V2_EXTRA_TAGS = frozenset({"record", "get", "extend", "raise", "try", "evar"})
V2_TAGS = V1_TAGS | V2_EXTRA_TAGS

# 各版本的构造子/字面量“值”集合（值限制判定）。
V1_VALUE_TAGS = frozenset({"int", "bool", "lambda", "var"})
# v2 下记录字面量也是非扩展式（其所有字段值若是语法值则可安全泛化；
# 可泛化性最终由“泛化变量是否可达可变引用”的逃逸检查决定）。
V2_VALUE_TAGS = V1_VALUE_TAGS | frozenset({"record"})

Node = Dict[str, Any]
Path = Tuple[str, ...]


# ---------------------------------------------------------------- 构造器 v1

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


# ---------------------------------------------------------------- 构造器 v2

def record(fields: List[Tuple[str, Node]]) -> Node:
    """记录字面量：``fields`` 为 ``(字段名, 表达式)`` 序列。

    同名字段（含经开放行带入的）在推断期统一时也会再次报错；
    字面量内部的字面重名在这里直接拒绝。
    """
    return {"tag": "record", "fields": [list(pair) for pair in fields]}


def get(record_node: Node, field: str) -> Node:
    return {"tag": "get", "record": record_node, "field": field}


def extend(record_node: Node, field: str, value: Node, rest: Node | None = None) -> Node:
    """不改变原记录的扩展：``{e with field = value}``。

    结果行是“被扩展记录的行 + 新字段”；``rest`` 用于显式指定结果行的
    开放尾来源，缺省为被扩展记录本身。扩展一个已含同名字段的记录必错。
    """
    return {"tag": "extend", "record": record_node, "field": field,
            "value": value, "rest": rest if rest is not None else record_node}


def raise_(value: Node) -> Node:
    return {"tag": "raise", "value": value}


def try_(body: Node, param: str, handler: Node) -> Node:
    return {"tag": "try", "body": body, "param": param, "handler": handler}


def evar() -> Node:
    """未决定应的占位表达式：参考域中代表“此处效应未指定”。"""
    return {"tag": "evar"}


# ---------------------------------------------------------------- 模块

def make_module(
    name: str,
    exports: Dict[str, str],
    body: Dict[str, Node],
    imports: Tuple[str, ...] = (),
) -> Dict[str, Any]:
    """构造一个模块描述（普通 dict）。

    * ``exports``：``{导出名: 签名串}``，每个导出都必须有签名；
      内部定义（在 ``body`` 但不在 ``exports``）可被本模块其他定义引用，
      但不进入模块环境。
    * ``body``：``{定义名: 表达式 AST}``；定义按 let-rec 成组，可相互递归
      （包括跨模块的导出引用）。
    * ``imports``：直接依赖的模块名（仅用于文档/校验；依赖也会从 AST 中
      形如 ``{"tag": "xvar", "module": m, "name": n}`` 的外部引用推导）。
    """
    return {
        "name": name,
        "exports": dict(exports),
        "body": dict(body),
        "imports": tuple(imports),
    }


def xvar(module: str, name: str) -> Node:
    """跨模块外部引用节点（仅在模块推断中合法）。"""
    return {"tag": "xvar", "module": module, "name": name}


# ---------------------------------------------------------------- 校验

class AstValidator:
    """深度校验单表达式 AST。

    任何不合法都抛 :class:`InvalidAstError`，路径定位到最小相关子树。
    """

    def __init__(self, max_nodes: int, language: str = "v1") -> None:
        self.max_nodes = max_nodes
        self.language = language
        self.allowed = V2_TAGS if language == "v2" else V1_TAGS
        self.count = 0

    def validate(self, node: Any) -> None:
        self._check(node, (), scope=())

    def _fail(self, path: Path, message: str) -> None:
        raise InvalidAstError(path, message)

    def _check(self, node: Any, path: Path, scope: Tuple[str, ...]) -> None:
        if not isinstance(node, dict):
            self._fail(path, f"节点必须是 dict，实际得到 {type(node).__name__}")
        tag = node.get("tag")
        if not isinstance(tag, str):
            self._fail(path, "缺少字符串字段 'tag' 或其值不是字符串")
        if tag == "xvar":
            # 跨模块引用只允许出现在模块推断里，由 ModuleValidator 单独受理。
            self._fail(path, "xvar 节点只能出现在模块定义中（单表达式推断不受理）")
        fields = NODE_FIELDS.get(tag)
        if fields is None:
            self._fail(path, f"未知节点 tag {tag!r}")
        if tag not in self.allowed:
            self._fail(
                path,
                f"节点 tag {tag!r} 在语言版本 {self.language} 中不可用"
                f"（单表达式 v2 需要 infer(..., language='v2')）",
            )
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
        elif tag == "get":
            self._check_name(path, node["field"])
        elif tag == "extend":
            self._check_name(path, node["field"])
        elif tag == "try":
            self._check_name(path, node["param"])
        elif tag == "record":
            self._check_fields(path, node["fields"])

        # 递归校验子节点
        if tag == "lambda":
            self._check(node["body"], path + ("body",), scope + (node["param"],))
        elif tag == "apply":
            self._check(node["fn"], path + ("fn",), scope)
            self._check(node["arg"], path + ("arg",), scope)
        elif tag == "let":
            name = node["name"]
            # 非递归 let：绑定表达式的作用域不含 name 自身。
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
        elif tag == "record":
            seen = set()
            for i, pair in enumerate(node["fields"]):
                fname = pair[0]
                if fname in seen:
                    self._fail(path + ("fields", i), f"记录字面量字段重名 {fname!r}")
                seen.add(fname)
                self._check(pair[1], path + ("fields", i, 1), scope)
        elif tag == "get":
            self._check(node["record"], path + ("record",), scope)
        elif tag == "extend":
            self._check(node["record"], path + ("record",), scope)
            self._check(node["value"], path + ("value",), scope)
            self._check(node["rest"], path + ("rest",), scope)
        elif tag == "raise":
            self._check(node["value"], path + ("value",), scope)
        elif tag == "try":
            self._check(node["body"], path + ("body",), scope)
            self._check(
                node["handler"], path + ("handler",),
                scope + (node["param"],),
            )
        # evar：无子节点

    def _check_fields(self, path: Path, fields: Any) -> None:
        if not isinstance(fields, list):
            self._fail(path, f"'record' 的 fields 必须是 list，实际得到 {type(fields).__name__}")
        for i, pair in enumerate(fields):
            if not (isinstance(pair, list) and len(pair) == 2):
                self._fail(path + ("fields", i), "每个字段必须是 [字段名, 表达式] 二元列表")
            self._check_name(path + ("fields", i), pair[0])

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


class ModuleValidator:
    """校验模块包：结构、命名、签名串、跨模块引用目标存在性。"""

    def __init__(self, modules: Dict[str, Dict[str, Any]], max_nodes: int) -> None:
        self.modules = modules
        self.max_nodes = max_nodes
        self.total_nodes = 0

    def validate(self) -> None:
        if not isinstance(self.modules, dict) or not self.modules:
            raise InvalidAstError((), "模块包必须是非空 dict：{模块名: 模块描述}")
        for mname, mod in self.modules.items():
            self._check_module(mname, mod)

    def _fail(self, path: Path, message: str) -> None:
        raise InvalidAstError(path, message)

    def _check_module(self, mname: str, mod: Any) -> None:
        base: Path = (mname,)
        if not isinstance(mname, str) or not mname:
            self._fail((), f"模块名必须是非空 str，实际 {mname!r}")
        if not isinstance(mod, dict):
            self._fail(base, f"模块必须是 dict，实际得到 {type(mod).__name__}")
        for key in ("name", "exports", "body"):
            if key not in mod:
                self._fail(base, f"模块缺少字段 {key!r}")
        if mod["name"] != mname:
            self._fail(base + ("name",), f"模块名字段 {mod['name']!r} 与键 {mname!r} 不一致")
        exports = mod["exports"]
        body = mod["body"]
        if not isinstance(exports, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in exports.items()
        ):
            self._fail(base + ("exports",), "exports 必须是 {str: 签名字符串}")
        if not isinstance(body, dict) or not all(isinstance(k, str) for k in body):
            self._fail(base + ("body",), "body 必须是 {定义名: AST 节点}")
        if not exports:
            self._fail(base + ("exports",), "每个模块至少要有一个带签名的导出")
        for ename, sig in exports.items():
            if ename not in body:
                self._fail(base + ("exports", ename), f"导出 {ename!r} 在 body 中没有对应定义")
        imports = mod.get("imports", ())
        if not isinstance(imports, tuple | list):
            self._fail(base + ("imports",), "imports 必须是模块名元组/列表")
        for dep in imports:
            if not isinstance(dep, str) or dep not in self.modules:
                self._fail(base + ("imports",), f"依赖的模块 {dep!r} 不在模块包中")

        expr_validator = AstValidator(self.max_nodes, language="v2")
        for dname, expr in body.items():
            self._validate_def_expr(mname, dname, expr, expr_validator)
        self.total_nodes += expr_validator.count

    def _validate_def_expr(
        self, mname: str, dname: str, node: Any, ev: AstValidator
    ) -> None:
        """校验一个定义表达式；作用域为整个模块的定义名（递归组），
        xvar 节点在局部由这里受理。"""
        # 复用 AstValidator 的字段校验逻辑，但替换入口以处理 xvar 与
        # 模块级递归作用域：直接内联一个轻量遍历。
        local_names = tuple(self.modules[mname]["body"])
        self._walk(mname, dname, node, (), local_names, ev)

    def _walk(
        self,
        mname: str,
        dname: str,
        node: Any,
        path: Path,
        scope: Tuple[str, ...],
        ev: AstValidator,
    ) -> None:
        base: Path = (mname, "body", dname) + path
        if not isinstance(node, dict):
            raise InvalidAstError(base, f"节点必须是 dict，实际得到 {type(node).__name__}")
        tag = node.get("tag")
        if not isinstance(tag, str):
            raise InvalidAstError(base, "缺少字符串字段 'tag'")
        if tag == "xvar":
            mod_ref = node.get("module")
            name_ref = node.get("name")
            if set(node) != {"tag", "module", "name"}:
                raise InvalidAstError(base, "xvar 节点只允许 tag/module/name 字段")
            if not isinstance(mod_ref, str) or not isinstance(name_ref, str):
                raise InvalidAstError(base, "xvar 的 module/name 必须是 str")
            if mod_ref not in self.modules:
                raise InvalidAstError(base, f"xvar 引用了不存在的模块 {mod_ref!r}")
            if name_ref not in self.modules[mod_ref]["exports"]:
                raise InvalidAstError(
                    base, f"模块 {mod_ref!r} 没有导出 {name_ref!r}（每个外部引用必须指向签名导出）"
                )
            ev.count += 1
            if ev.count > self.max_nodes:
                from .errors import NodeLimitError

                raise NodeLimitError(ev.count, self.max_nodes)
            return
        fields = NODE_FIELDS.get(tag)
        if fields is None:
            raise InvalidAstError(base, f"未知节点 tag {tag!r}")
        if tag not in V2_TAGS:
            raise InvalidAstError(base, f"节点 tag {tag!r} 在 v2 中不可用")
        extra = set(node) - {"tag", *fields}
        if extra:
            raise InvalidAstError(base, f"存在多余字段 {sorted(extra)}；{tag} 节点只允许 {list(fields)}")
        missing = [f for f in fields if f not in node]
        if missing:
            raise InvalidAstError(base, f"缺少字段 {missing}")

        ev.count += 1
        if ev.count > self.max_nodes:
            from .errors import NodeLimitError

            raise NodeLimitError(ev.count, self.max_nodes)

        if tag == "var":
            ev._check_name(base, node["name"])
            if node["name"] not in scope:
                from .errors import UnboundVariableError

                raise UnboundVariableError(base, node["name"])
        elif tag == "int":
            ev._check_int(base, node["value"])
        elif tag == "bool":
            if not isinstance(node["value"], bool):
                raise InvalidAstError(base, "'bool' 节点的 value 必须是 bool")
        elif tag in ("lambda", "let"):
            ev._check_name(base, node["param" if tag == "lambda" else "name"])
        elif tag in ("get", "extend"):
            ev._check_name(base, node["field"])
        elif tag == "try":
            ev._check_name(base, node["param"])
        elif tag == "record":
            ev._check_fields(base, node["fields"])

        # 子节点
        if tag == "lambda":
            self._walk(mname, dname, node["body"], path + ("body",), scope + (node["param"],), ev)
        elif tag == "apply":
            self._walk(mname, dname, node["fn"], path + ("fn",), scope, ev)
            self._walk(mname, dname, node["arg"], path + ("arg",), scope, ev)
        elif tag == "let":
            name = node["name"]
            self._walk(mname, dname, node["bound"], path + ("bound",), scope, ev)
            self._walk(mname, dname, node["body"], path + ("body",), scope + (name,), ev)
        elif tag == "if":
            self._walk(mname, dname, node["cond"], path + ("cond",), scope, ev)
            self._walk(mname, dname, node["then"], path + ("then",), scope, ev)
            self._walk(mname, dname, node["else_"], path + ("else_",), scope, ev)
        elif tag == "ref":
            self._walk(mname, dname, node["inner"], path + ("inner",), scope, ev)
        elif tag == "read":
            self._walk(mname, dname, node["ref"], path + ("ref",), scope, ev)
        elif tag == "write":
            self._walk(mname, dname, node["ref"], path + ("ref",), scope, ev)
            self._walk(mname, dname, node["value"], path + ("value",), scope, ev)
        elif tag == "record":
            seen = set()
            for i, pair in enumerate(node["fields"]):
                if pair[0] in seen:
                    raise InvalidAstError(base + ("fields", i), f"记录字面量字段重名 {pair[0]!r}")
                seen.add(pair[0])
                self._walk(mname, dname, pair[1], path + ("fields", i, 1), scope, ev)
        elif tag == "get":
            self._walk(mname, dname, node["record"], path + ("record",), scope, ev)
        elif tag == "extend":
            self._walk(mname, dname, node["record"], path + ("record",), scope, ev)
            self._walk(mname, dname, node["value"], path + ("value",), scope, ev)
            self._walk(mname, dname, node["rest"], path + ("rest",), scope, ev)
        elif tag == "raise":
            self._walk(mname, dname, node["value"], path + ("value",), scope, ev)
        elif tag == "try":
            self._walk(mname, dname, node["body"], path + ("body",), scope, ev)
            self._walk(
                mname, dname, node["handler"], path + ("handler",),
                scope + (node["param"],), ev,
            )
        # evar：无
