"""rank-1 显式类型签名的解析（纯语法层，不依赖推断/统一核心）。

签名写在 ``let`` 节点的可选字符串字段 ``sig`` 上，语法（rank-1）::

    sig   := [forall var {var} "."] type
    type  := atom "->" type | atom
    atom  := "Int" | "Bool" | "Unit" | "Ref(" type ")" | varname

* 箭头右结合；只允许 rank 1：``forall`` 只能出现在最前面，函数参数
  位置不允许再写 ``forall``（不支持高阶多态）；
* 构造子名固定为 ``Int`` / ``Bool`` / ``Unit``（大写开头）；
* 类型变量名是小写字母开头的标识符；**所有**类型变量都必须在
  ``forall`` 列表中显式量化（不允许未限定自由变量），量化变量
  不允许重复或从未出现；
* 解析产出 :class:`ParsedSignature`（语法树 + 量化名，按在类型中
  首次出现的顺序排列），与推断完全解耦；推断器再用
  :func:`build_signature_type` 把语法树实例化成带
  :class:`~polyinfer.types.RigidTVar` 的真实类型。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple, Union

from .errors import InvalidAstError
from .types import BOOL, INT, UNIT, TCon, TFun, TRef, TVar, Type


# ---------------------------------------------------------------- 语法树

@dataclass(frozen=True)
class SVar:
    name: str


@dataclass(frozen=True)
class SCon:
    name: str


@dataclass(frozen=True)
class SFun:
    arg: "SigNode"
    ret: "SigNode"


@dataclass(frozen=True)
class SRef:
    inner: "SigNode"


SigNode = Union[SVar, SCon, SFun, SRef]


@dataclass(frozen=True)
class ParsedSignature:
    """签名解析结果：量化变量名（按类型中首次出现排序）+ 类型语法树。"""

    quantified: Tuple[str, ...]
    body: SigNode


_CONS = {"Int": INT, "Bool": BOOL, "Unit": UNIT}


# ---------------------------------------------------------------- 分词

class _Tokenizer:
    _SYMS = ("->", ".", "(", ")", ",")

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def error(self, message: str) -> None:
        raise InvalidAstError(("sig",), f"类型签名解析失败（列 {self.pos + 1}）：{message}")

    def skip(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def at_end(self) -> bool:
        self.skip()
        return self.pos >= len(self.text)

    def peek(self) -> str:
        self.skip()
        return self.text[self.pos : self.pos + 1]

    def consume_keyword(self, word: str) -> bool:
        self.skip()
        if self.text.startswith(word, self.pos):
            after = self.pos + len(word)
            if after >= len(self.text) or not (self.text[after].isalnum() or self.text[after] == "_"):
                self.pos = after
                return True
        return False

    def consume_punct(self, punct: str) -> bool:
        self.skip()
        if self.text.startswith(punct, self.pos):
            self.pos += len(punct)
            return True
        return False

    def expect_punct(self, punct: str) -> None:
        if not self.consume_punct(punct):
            self.error(f"期望 {punct!r}")

    def read_ident(self) -> str:
        self.skip()
        start = self.pos
        if self.pos >= len(self.text) or not (self.text[self.pos].isalpha() or self.text[self.pos] == "_"):
            self.error("期望标识符")
        while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos] == "_"):
            self.pos += 1
        return self.text[start : self.pos]

    def consume_arrow(self) -> bool:
        return self.consume_punct("->")


# ---------------------------------------------------------------- 解析

def parse_signature(text: object) -> ParsedSignature:
    """解析签名文本；非法文本一律抛 :class:`InvalidAstError`（path 为 ``('sig',)``）。"""
    if not isinstance(text, str) or not text.strip():
        raise InvalidAstError(("sig",), "类型签名必须是非空字符串")
    tk = _Tokenizer(text)

    quantified: list[str] = []
    if tk.consume_keyword("forall"):
        first = tk.read_ident()
        names = [first]
        while True:
            tk.skip()
            if tk.consume_punct("."):
                break
            if tk.peek() == ",":
                tk.expect_punct(",")
            names.append(tk.read_ident())
        seen: set[str] = set()
        for nm in names:
            if not nm[0].islower():
                tk.error(f"量化类型变量必须以小写字母开头：{nm!r}")
            if nm in seen:
                tk.error(f"量化变量重复出现：{nm!r}")
            seen.add(nm)
            quantified.append(nm)

    body = _parse_type(tk, top_rank=True)
    if not tk.at_end():
        tk.error(f"意外的尾部字符 {tk.text[tk.pos:]!r}")

    used: list[str] = []

    def collect(node: SigNode) -> None:
        if isinstance(node, SVar):
            if node.name not in used:
                used.append(node.name)
        elif isinstance(node, SFun):
            collect(node.arg)
            collect(node.ret)
        elif isinstance(node, SRef):
            collect(node.inner)

    collect(body)

    used_set = set(used)
    declared = set(quantified)
    unquantified = [nm for nm in used if nm not in declared]
    if unquantified:
        tk.error(f"类型变量 {unquantified} 未在 forall 中量化（不允许未限定自由变量）")
    unused = [nm for nm in quantified if nm not in used_set]
    if unused:
        tk.error(f"量化变量 {unused} 从未在类型体中出现")

    order = [nm for nm in used if nm in declared]
    return ParsedSignature(tuple(order), body)


def _parse_type(tk: _Tokenizer, top_rank: bool) -> SigNode:
    arg = _parse_atom(tk)
    if tk.consume_arrow():
        ret = _parse_type(tk, top_rank=False)
        return SFun(arg, ret)
    return arg


def _parse_atom(tk: _Tokenizer) -> SigNode:
    tk.skip()
    if tk.consume_keyword("Ref"):
        tk.expect_punct("(")
        inner = _parse_type(tk, top_rank=False)
        tk.expect_punct(")")
        return SRef(inner)
    if tk.consume_keyword("forall"):
        tk.error("forall 只能出现在签名最外层（仅支持 rank-1，不支持高阶多态）")
    if tk.peek() == "(":
        # 括号仅用于分组（Ref 之外也允许 (a -> b) -> c 这类写法）。
        tk.expect_punct("(")
        inner = _parse_type(tk, top_rank=False)
        tk.expect_punct(")")
        return inner
    ident = tk.read_ident()
    if ident in _CONS:
        return SCon(ident)
    if ident[0].isupper():
        tk.error(f"未知类型构造子 {ident!r}（仅支持 Int/Bool/Unit/Ref）")
    return SVar(ident)


# ---------------------------------------------------------------- 实例化

def build_signature_type(
    parsed: ParsedSignature,
    make_rigid: Callable[[str], TVar],
) -> Tuple[Tuple[TVar, ...], Type]:
    """把解析结果实例化成真实类型。

    每个量化名通过 ``make_rigid`` 取得一个刚性变量（通常是
    ``Inferencer`` 自己计数器产出的 :class:`RigidTVar`），
    返回 ``(按首次出现排序的刚性变量, 类型)``。该函数只做树构建，
    不调用统一器。
    """
    mapping: dict[str, TVar] = {}

    def build(node: SigNode) -> Type:
        if isinstance(node, SVar):
            tv = mapping.get(node.name)
            if tv is None:  # 解析阶段已保证变量都被量化
                tv = make_rigid(node.name)
                mapping[node.name] = tv
            return tv
        if isinstance(node, SCon):
            return _CONS[node.name]
        if isinstance(node, SFun):
            return TFun(build(node.arg), build(node.ret))
        return TRef(build(node.inner))

    tp = build(parsed.body)
    rigids = tuple(mapping[nm] for nm in parsed.quantified)
    return rigids, tp
