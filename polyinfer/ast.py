"""AST 定义、构造器与结构校验。

公开 AST 是普通 dict（可 JSON 化），固定 ``tag`` 字段区分类别。

v1 表达式（``version=1``，:func:`polyinfer.inference.infer` 的缺省模式）：

=================  =========================================================
``"var"``          ``name: str``
``"int"``          ``value: int``（bool 被拒绝）
``"bool"``         ``value: bool``
``"lambda"``       ``param: str``, ``body: Node``
``"apply"``        ``fn: Node``, ``arg: Node``
``"let"``          ``name: str``, ``bound: Node``, ``body: Node``
``"if"``           ``cond: Node``, ``then: Node``, ``else_: Node``
``"ref"``          ``inner: Node``
``"read"``         ``ref: Node``
``"write"``        ``ref: Node``, ``value: Node``
=================  =========================================================

v2 新增表达式（需显式 ``version=2`` 或模块检查入口）：

=================  =========================================================
``"record"``       ``fields: [[name, Node], ...]`` 记录字面量（开放行）
``"field"``        ``record: Node``, ``name: str`` 字段访问
``"extend"``       ``record: Node``, ``fields: [[name, Node], ...]``
                   非破坏式扩展（原记录不变；同名字段一律报错）
``"raise"``        ``value: Node``（要求 ``Exn``），引入 ``exn`` 效应
``"try"``          ``body: Node``, ``handler: Node``（handler 须为
                   ``Exn -> t`` 函数）；body 的 ``exn`` 效应被处理移除，
                   其余效应（如 write）保留
``"exn"``          ``name: str`` 异常值字面量，类型为 ``Exn``
``"mvar"``         ``module: str``, ``name: str`` 跨模块导出引用
=================  =========================================================

模块包（v2）结构::

    {"modules": {
        "M": {"imports": ["N"], "exports": [
            {"name": "f",
             "signature": {"vars": ["a"], "type": <类型表达式 AST>},
             "binding": <表达式 AST>}]}}}

类型表达式 AST：``tcon``（Int/Bool/Unit/Exn）、``tvar``（普通类型变量）、
``rvar``（行变量）、``tfun``（含 ``effects`` 标签列表）、``tref``、
``trecord``（``fields`` 为 [名字, 类型表达式] 序列，``tail`` 为
``rvar`` 或 ``None``，缺省闭行）。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

from .errors import InvalidAstError

Path = Tuple[str, ...]

# v1 节点种类 -> 必需字段。
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

# v2 新增表达式节点。
V2_NODE_FIELDS: Dict[str, Tuple[str, ...]] = {
    "record": ("fields",),
    "field": ("record", "name"),
    "extend": ("record", "fields"),
    "raise": ("value",),
    "try": ("body", "handler"),
    "exn": ("name",),
    "mvar": ("module", "name"),
}

EFFECT_TAG_NAMES = frozenset({"read", "write", "exn"})
TYPE_CON_NAMES = frozenset({"Int", "Bool", "Unit", "Exn"})

Node = Dict[str, Any]


# ---------------------------------------------------------------- v1 构造器

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


# ---------------------------------------------------------------- v2 构造器

def record(fields: list[tuple[str, Node]]) -> Node:
    """记录字面量；字段以 ``[[name, expr], ...]`` 给出，行保持开放。"""
    return {"tag": "record", "fields": [[n, v] for n, v in fields]}


def field_(record_node: Node, name: str) -> Node:
    return {"tag": "field", "record": record_node, "name": name}


def extend(record_node: Node, fields: list[tuple[str, Node]]) -> Node:
    """非破坏式扩展：返回含新字段的新记录，原记录不变。"""
    return {"tag": "extend", "record": record_node, "fields": [[n, v] for n, v in fields]}


def raise_(value: Node) -> Node:
    return {"tag": "raise", "value": value}


def try_(body: Node, handler: Node) -> Node:
    return {"tag": "try", "body": body, "handler": handler}


def exn(name: str) -> Node:
    return {"tag": "exn", "name": name}


def mvar(module: str, name: str) -> Node:
    return {"tag": "mvar", "module": module, "name": name}


# ------------------------------------------------------- 签名/模块构造器

def tcon(name: str) -> Node:
    return {"tag": "tcon", "name": name}


def tvar(name: str) -> Node:
    return {"tag": "tvar", "name": name}


def rvar(name: str) -> Node:
    return {"tag": "rvar", "name": name}


def tfun(arg: Node, ret: Node, effects: list[str] | None = None) -> Node:
    return {"tag": "tfun", "arg": arg, "ret": ret, "effects": list(effects or [])}


def tref(inner: Node) -> Node:
    return {"tag": "tref", "inner": inner}


def trecord(fields: list[tuple[str, Node]], tail: Node | None = None) -> Node:
    """闭行记录传 ``tail=None``；开放行传 :func:`rvar`。"""
    return {"tag": "trecord", "fields": [[n, t] for n, t in fields], "tail": tail}


def signature(vars_: list[str], type_expr: Node, row_vars: list[str] | None = None) -> Node:
    """导出签名：显式 forall 变量表（普通变量 + 行变量分别列出，顺序确定）。

    效应不做量化：签名里的效应只能是具体标签，写在各 ``tfun`` 的
    ``effects`` 列表中。
    """
    return {
        "vars": list(vars_),
        "row_vars": list(row_vars or []),
        "type": type_expr,
    }


def export(name: str, sig: Node, binding: Node) -> Node:
    return {"name": name, "signature": sig, "binding": binding}


def module(imports: list[str], exports: list[Node]) -> Node:
    return {"imports": list(imports), "exports": list(exports)}


def package(modules: dict[str, Node]) -> Node:
    return {"modules": dict(modules)}


# ---------------------------------------------------------------- 表达式校验

class AstValidator:
    """深度校验整棵 v1 表达式 AST（行为与 v1 完全一致）。"""

    def __init__(self, max_nodes: int) -> None:
        self.max_nodes = max_nodes
        self.count = 0

    def validate(self, node: Any) -> None:
        checker = ExprValidator(self.max_nodes, version=1, counter=self)
        checker.check(node, (), scope=())
        self.count = checker.count


class ExprValidator:
    """v1/v2 表达式统一校验器。

    ``version=1`` 时遇到 v2 节点按“未知 tag”拒绝，保持旧入口契约不变；
    ``version=2`` 放行全部表达式节点。``module_context`` 给出
    (当前模块名, 可引用的导入模块集合)，用于 ``mvar`` 校验；
    非模块上下文（独立 v2 表达式）中出现 ``mvar`` 直接拒绝。
    """

    def __init__(
        self,
        max_nodes: int,
        version: int = 1,
        counter: "ExprValidator | AstValidator | None" = None,
        module_context: Tuple[str | None, frozenset] | None = None,
    ) -> None:
        self.max_nodes = max_nodes
        self.version = version
        if counter is not None:
            self.count = counter.count
        else:
            self.count = 0
        self.module_context = module_context

    def _fail(self, path: Path, message: str) -> None:
        raise InvalidAstError(path, message)

    def check(self, node: Any, path: Path, scope: Tuple[str, ...]) -> None:
        if not isinstance(node, dict):
            self._fail(path, f"节点必须是 dict，实际得到 {type(node).__name__}")
        tag = node.get("tag")
        if not isinstance(tag, str):
            self._fail(path, "缺少字符串字段 'tag' 或其值不是字符串")
        fields = NODE_FIELDS.get(tag)
        if fields is None:
            if self.version >= 2 and tag in V2_NODE_FIELDS:
                fields = V2_NODE_FIELDS[tag]
            else:
                self._fail(path, f"未知节点 tag {tag!r}")
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
            self._check_name(path, node["name"])
            if node["name"] not in scope:
                from .errors import UnboundVariableError

                raise UnboundVariableError(path, node["name"])
        elif tag == "int":
            self._check_int(path, node["value"])
        elif tag == "bool":
            value = node["value"]
            if not isinstance(value, bool):
                self._fail(path, f"'bool' 节点的 value 必须是 bool，实际得到 {type(value).__name__}")
        elif tag in ("lambda", "let"):
            self._check_name(path, node["param" if tag == "lambda" else "name"])
        elif tag in ("field",):
            self._check_name(path, node["name"])
        elif tag in ("exn", "mvar"):
            self._check_name(path, node["name"])
            if tag == "mvar":
                self._check_mvar(path, node)
        elif tag in ("record", "extend"):
            self._check_field_list(path, node["fields"])

        # 递归子节点
        if tag == "lambda":
            self.check(node["body"], path + ("body",), scope + (node["param"],))
        elif tag == "apply":
            self.check(node["fn"], path + ("fn",), scope)
            self.check(node["arg"], path + ("arg",), scope)
        elif tag == "let":
            name = node["name"]
            # 非递归 let：绑定表达式作用域不含 name 自身。
            self.check(node["bound"], path + ("bound",), scope)
            self.check(node["body"], path + ("body",), scope + (name,))
        elif tag == "if":
            self.check(node["cond"], path + ("cond",), scope)
            self.check(node["then"], path + ("then",), scope)
            self.check(node["else_"], path + ("else_",), scope)
        elif tag == "ref":
            self.check(node["inner"], path + ("inner",), scope)
        elif tag == "read":
            self.check(node["ref"], path + ("ref",), scope)
        elif tag == "write":
            self.check(node["ref"], path + ("ref",), scope)
            self.check(node["value"], path + ("value",), scope)
        elif tag == "record":
            for i, pair in enumerate(node["fields"]):
                self.check(pair[1], path + ("fields", f"[{i}]", "value"), scope)
        elif tag == "field":
            self.check(node["record"], path + ("record",), scope)
        elif tag == "extend":
            self.check(node["record"], path + ("record",), scope)
            for i, pair in enumerate(node["fields"]):
                self.check(pair[1], path + ("fields", f"[{i}]", "value"), scope)
        elif tag == "raise":
            self.check(node["value"], path + ("value",), scope)
        elif tag == "try":
            self.check(node["body"], path + ("body",), scope)
            self.check(node["handler"], path + ("handler",), scope)

    def _check_mvar(self, path: Path, node: Node) -> None:
        mod = node["module"]
        if not isinstance(mod, str) or not mod:
            self._fail(path, "'mvar' 的 module 必须是非空 str")
        if self.module_context is None:
            self._fail(path, "独立表达式上下文中不允许跨模块引用 mvar")
        current, allowed = self.module_context
        # mvar 只用于显式导入的模块；引用本模块更早导出用普通 var。
        if mod not in allowed:
            self._fail(
                path,
                f"mvar 引用的模块 {mod!r} 不在当前模块的导入表中"
                "（引用本模块导出请用 var）",
            )

    def _check_field_list(self, path: Path, fields: Any) -> None:
        if not isinstance(fields, list) or not fields:
            self._fail(path, "字段表必须是非空 list，形如 [[name, expr], ...]")
        seen = set()
        for i, pair in enumerate(fields):
            sub = path + ("fields", f"[{i}]")
            if not isinstance(pair, list) or len(pair) != 2:
                self._fail(sub, "字段项必须是且仅是 [name, expr]")
            name, _expr = pair
            if not isinstance(name, str) or not name:
                self._fail(sub, "字段名必须是非空 str")
            if name in seen:
                # 同一字面量/扩展内的同名字段在语法期即拒绝（不静默覆盖）。
                from .errors import DuplicateFieldError

                raise DuplicateFieldError(sub, name)
            seen.add(name)

    def _check_name(self, path: Path, name: Any) -> None:
        if not isinstance(name, str):
            self._fail(path, f"标识符必须是 str，实际得到 {type(name).__name__}")
        if not name:
            self._fail(path, "标识符不能为空字符串")

    def _check_int(self, path: Path, value: Any) -> None:
        if isinstance(value, bool):
            self._fail(path, "'int' 节点不接受 bool（bool 必须使用 'bool' 节点）")
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                self._fail(path, f"拒绝非有限数值 {value!r}，且 float 不会隐式转为 int")
            self._fail(path, f"'int' 节点只接受 int，float 不会被隐式截断（实际 {value!r}）")
        if not isinstance(value, int):
            self._fail(path, f"'int' 节点的 value 必须是 int，实际得到 {type(value).__name__}")


# ---------------------------------------------------------------- 模块包校验

class PackageValidator:
    """模块包结构校验：模块表、导入、导出三元组（名字+签名+绑定）与签名 AST。

    本类只做结构校验（含表达式作用域校验，``mvar`` 限定到导入表）；
    依赖图 SCC 与签名/实现一致性在 :mod:`polyinfer.modules` 中处理。
    """

    def __init__(self, max_nodes: int, max_modules: int) -> None:
        self.max_nodes = max_nodes
        self.max_modules = max_modules
        self.count = 0

    def validate(self, package_node: Any) -> Dict[str, dict]:
        if not isinstance(package_node, dict) or set(package_node) != {"modules"}:
            raise InvalidAstError((), "模块包必须是 {'modules': {名字: 模块}} 且无多余字段")
        modules = package_node["modules"]
        if not isinstance(modules, dict) or not modules:
            raise InvalidAstError(("modules",), "modules 必须是非空 dict")
        if len(modules) > self.max_modules:
            from .errors import ModuleStructureError

            raise ModuleStructureError(
                f"模块数 {len(modules)} 超过上限 {self.max_modules}"
            )
        for name in modules:
            if not isinstance(name, str) or not name:
                raise InvalidAstError(("modules",), "模块名必须是非空 str")
        for name, mod in modules.items():
            self._check_module(name, mod, modules)
        return modules

    def _check_module(self, name: str, mod: Any, modules: dict) -> None:
        from .errors import ModuleStructureError

        path = ("modules", name)
        if not isinstance(mod, dict) or set(mod) != {"imports", "exports"}:
            raise ModuleStructureError(
                "模块必须是 {'imports': [...], 'exports': [...]} 且无多余字段", name
            )
        imports, exports = mod["imports"], mod["exports"]
        if not isinstance(imports, list) or not all(isinstance(i, str) and i for i in imports):
            raise ModuleStructureError("imports 必须是 str 列表", name)
        if len(set(imports)) != len(imports):
            raise ModuleStructureError("imports 中存在重复模块", name)
        for dep in imports:
            if dep not in modules:
                raise ModuleStructureError(f"导入了包内不存在的模块 {dep!r}", name)
        if not isinstance(exports, list) or not exports:
            raise ModuleStructureError("每个模块至少有一个导出", name)

        seen = set()
        ordered: list[str] = []
        for i, ex in enumerate(exports):
            epath = path + ("exports", f"[{i}]")
            if not isinstance(ex, dict) or set(ex) != {"name", "signature", "binding"}:
                raise ModuleStructureError(
                    f"导出项必须只含 name/signature/binding", name
                )
            ename = ex["name"]
            if not isinstance(ename, str) or not ename:
                raise ModuleStructureError("导出名必须是非空 str", name)
            if ename in seen:
                raise ModuleStructureError(f"导出 {ename!r} 重复", name)
            seen.add(ename)
            # 先校验签名 AST。
            self._check_signature(ex["signature"], epath + ("signature",), name)
            # 表达式作用域：同模块**更早**导出按 var 可见（顺序与实现
            # 检查一致，不做前向引用）；导入模块与本模块的导出按 mvar 可见。
            expr_validator = ExprValidator(
                self.max_nodes,
                version=2,
                counter=self,
                module_context=(name, frozenset(imports)),
            )
            expr_validator.check(
                ex["binding"], epath + ("binding",), tuple(ordered)
            )
            self.count = expr_validator.count
            ordered.append(ename)

    def _check_signature(self, sig: Any, path: Path, module_name: str) -> None:
        from .errors import ModuleStructureError

        if not isinstance(sig, dict) or set(sig) != {"vars", "row_vars", "type"}:
            raise ModuleStructureError(
                "签名必须是 {vars, row_vars, type}", module_name
            )
        for key in ("vars", "row_vars"):
            vals = sig[key]
            if not isinstance(vals, list) or not all(isinstance(v, str) and v for v in vals):
                raise ModuleStructureError(f"签名 {key} 必须是 str 列表", module_name)
            if len(set(vals)) != len(vals):
                raise ModuleStructureError(f"签名 {key} 中有重复变量", module_name)
        if set(sig["vars"]) & set(sig["row_vars"]):
            raise ModuleStructureError("普通类型变量与行变量不得同名", module_name)
        self._check_type_expr(
            sig["type"],
            path + ("type",),
            frozenset(sig["vars"]),
            frozenset(sig["row_vars"]),
            module_name,
        )

    def _check_type_expr(self, node, path, tvars, rvars, module_name,
                         row_position: bool = False) -> None:
        from .errors import ModuleStructureError

        if not isinstance(node, dict):
            raise InvalidAstError(path, f"类型表达式必须是 dict，实际 {type(node).__name__}")
        tag = node.get("tag")
        allowed = {"tcon", "tvar", "rvar", "tfun", "tref", "trecord"}
        if tag not in allowed:
            raise ModuleStructureError(f"签名中出现未知类型表达式 {tag!r}", module_name)
        extra = set(node) - {
            "tag", "name", "arg", "ret", "effects", "inner", "fields", "tail"
        }
        if extra:
            raise ModuleStructureError(f"类型表达式存在多余字段 {sorted(extra)}", module_name)
        if tag == "tcon":
            if node.get("name") not in TYPE_CON_NAMES:
                raise ModuleStructureError(
                    f"未知基础类型 {node.get('name')!r}，允许 {sorted(TYPE_CON_NAMES)}",
                    module_name,
                )
        elif tag == "tvar":
            if node["name"] not in tvars:
                raise ModuleStructureError(
                    f"类型变量 {node['name']!r} 未在签名 vars 中绑定", module_name
                )
        elif tag == "rvar":
            # 行变量只允许出现在 trecord 的尾行位置（kind=行，不能当普通类型）。
            if not row_position:
                raise ModuleStructureError(
                    f"行变量 {node['name']!r} 只能作为 trecord 的尾行（tail），"
                    "不得出现在字段/参数/返回/Ref 内部",
                    module_name,
                )
            if node["name"] not in rvars:
                raise ModuleStructureError(
                    f"行变量 {node['name']!r} 未在签名 row_vars 中绑定", module_name
                )
        elif tag == "tfun":
            effects = node["effects"]
            if not isinstance(effects, list) or any(e not in EFFECT_TAG_NAMES for e in effects):
                raise ModuleStructureError(
                    f"effects 必须是 {sorted(EFFECT_TAG_NAMES)} 的子集列表", module_name
                )
            if len(set(effects)) != len(effects):
                raise ModuleStructureError("effects 中存在重复标签", module_name)
            self._check_type_expr(node["arg"], path + ("arg",), tvars, rvars, module_name)
            self._check_type_expr(node["ret"], path + ("ret",), tvars, rvars, module_name)
        elif tag == "tref":
            self._check_type_expr(node["inner"], path + ("inner",), tvars, rvars, module_name)
        elif tag == "trecord":
            fields = node["fields"]
            if not isinstance(fields, list):
                raise ModuleStructureError("trecord.fields 必须是列表", module_name)
            names = set()
            for i, pair in enumerate(fields):
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ModuleStructureError("trecord 字段必须是 [name, 类型]", module_name)
                fname, ftype = pair
                if not isinstance(fname, str) or not fname:
                    raise ModuleStructureError("trecord 字段名必须是非空 str", module_name)
                if fname in names:
                    raise ModuleStructureError(f"签名记录中字段 {fname!r} 重复", module_name)
                names.add(fname)
                self._check_type_expr(
                    ftype, path + ("fields", f"[{i}]"), tvars, rvars, module_name
                )
            tail = node["tail"]
            if tail is None:
                pass
            elif isinstance(tail, dict) and tail.get("tag") == "rvar":
                if tail["name"] not in rvars:
                    raise ModuleStructureError(
                        f"尾行变量 {tail['name']!r} 未在 row_vars 中绑定", module_name
                    )
            else:
                raise ModuleStructureError(
                    "trecord.tail 必须是 None（闭行）或 rvar（开放行）", module_name
                )
