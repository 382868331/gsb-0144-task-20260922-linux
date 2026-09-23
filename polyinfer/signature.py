"""导出签名串的词法/语法解析与方案实例化。

签名文法（签名串是**显式**接口，不做推断；每个模块导出都必须带一个）::

    scheme   := ["forall" NAME ("," NAME)* "."] monotype
    monotype := atom ["->"]? 完整箭头：
    monotype := atom ("-{" LABEL ("," LABEL)* "}->")? monotype     （右结合）
               | atom ("->" monotype)?
    atom     := "Int" | "Bool" | "Unit"
               | "Ref(" monotype ")"
               | "{" [field ("," field)* [";" NAME]] "}"
               | "(" monotype ")"
               | NAME          （仅允许出现在 forall 列表中的限定名）
    field    := NAME ":" monotype
    LABEL    := "R" | "W" | "X"

语义：

* 限定类型变量在**使用处**实例化为全新柔性变量（普通多态实例化），
  在**实现检查处**实例化为刚性 skolem；自由（未限定）名字一律拒绝。
* 行尾变量（``{...; r}`` 中的 ``r``）若出现也必须在 forall 列表中。
* 效应是 R/W/X 的具体集合，不做效应变量；纯箭头写作 ``->``。
* 解析器不使用核心的任何推断状态；负 id 占位变量在实例化时整体换名。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .errors import PolytypeError
from .types import (
    ALL_EFFECTS,
    BOOL,
    INT,
    UNIT,
    REmpty,
    RVar,
    TFun,
    TRec,
    TRef,
    TRow,
    TVar,
    Type,
    render_type,
)


class SignatureError(PolytypeError):
    """签名串非法（语法错、未知类型名、效应标签非法、自由变量未限定等）。"""

    def __init__(self, module: str, export: str, text: str, message: str, pos: int = -1) -> None:
        self.module = module
        self.export = export
        self.text = text
        self.pos = pos
        self.raw_message = message
        where = f"{pos}:" if pos >= 0 else ""
        if module or export:
            super().__init__(
                f"模块 {module!r} 导出 {export!r} 的签名 {text!r} 非法（位置 {where}）：{message}"
            )
        else:
            super().__init__(f"签名 {text!r} 非法（位置 {where}）：{message}")


# ---------------------------------------------------------------- 词法

_KEYWORDS = {"Int", "Bool", "Unit", "Ref", "forall"}


def _tokenize(text: str) -> List[Tuple[str, str]]:
    """返回 [(kind, text)]；kind ∈ name / lname / punct。"""
    tokens: List[Tuple[str, str]] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if c.isalpha() or c == "_":
            j = i + 1
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            word = text[i:j]
            tokens.append(("name", word))
            i = j
            continue
        if text.startswith("-{", i):
            tokens.append(("effopen", "-{"))
            i += 2
            continue
        two = text[i:i + 2]
        if two == "->":
            tokens.append(("arrow", "->"))
            i += 2
            continue
        if c in "{}(),;:.":
            tokens.append(("punct", c))
            i += 1
            continue
        raise ValueError(f"非法字符 {c!r}")
    return tokens


# ---------------------------------------------------------------- 解析结果

@dataclass(frozen=True)
class ParsedSignature:
    """解析后的签名：``forall names . t``。

    占位变量：被限定的普通变量是负 id 的 :class:`TVar`，行变量是负 id 的
    :class:`RVar`；``qnames`` 给出每个占位对象的源名。
    """

    t: Type
    qnames: Dict[Type, str]
    text: str

    def quantified(self) -> Tuple[Type, ...]:
        return tuple(self.qnames)

    def render(self) -> str:
        """重新渲染（用于缓存指纹与展示）。"""
        names = {v: nm for v, nm in self.qnames.items()}
        body = render_type(self.t, names)
        if self.qnames:
            qs = ", ".join(self.qnames[v] for v in self.qnames)
            return f"forall {qs}. {body}"
        return body

    # ------------------------------------------------ 实例化

    def instantiate_flex(self, fresh_tvar, fresh_rvar) -> Type:
        """使用处实例化：限定变量全部换全新柔性变量。"""
        mapping: Dict[Type, Type] = {}
        for placeholder, name in self.qnames.items():
            if isinstance(placeholder, RVar):
                mapping[placeholder] = fresh_rvar()
            else:
                mapping[placeholder] = fresh_tvar()
        return _rename(self.t, mapping)

    def instantiate_skol(self, fresh_skol, fresh_rvar, fresh_rskol=None) -> Type:
        """实现检查处实例化：普通限定变量换刚性 skolem，行变量换**刚性行
        变量**（``fresh_rskol``；缺省时回退为柔性行变量，仅内部测试用）。

        行变量即使在 Ref 内也必须刚性：实现不得把形状固定（闭合）的可变
        记录宣称为开放行多态。
        """
        mapping: Dict[Type, Type] = {}
        for placeholder, name in self.qnames.items():
            if isinstance(placeholder, RVar):
                mapping[placeholder] = (
                    fresh_rskol(name) if fresh_rskol is not None else fresh_rvar()
                )
            else:
                mapping[placeholder] = fresh_skol(name)
        return _rename(self.t, mapping)

    def skolems(self, t: Type) -> Dict[Type, str]:
        """返回实例化类型中各占位对象对应的源名（供报告）。"""
        return dict(self.qnames)


def _rename(tp: Type, mapping: Dict[Type, Type]) -> Type:
    if isinstance(tp, (TVar, RVar)):
        return mapping.get(tp, tp)
    if isinstance(tp, TFun):
        return TFun(_rename(tp.arg, mapping), _rename(tp.ret, mapping), tp.eff)
    if isinstance(tp, TRef):
        return TRef(_rename(tp.inner, mapping))
    if isinstance(tp, TRow):
        return TRow(
            tuple((n, _rename(ft, mapping)) for n, ft in tp.fields),
            _rename(tp.rest, mapping),
        )
    if isinstance(tp, TRec):
        return TRec(_rename(tp.row, mapping))
    return tp


# ---------------------------------------------------------------- 解析器

class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        try:
            self.tokens = _tokenize(text)
        except ValueError as exc:
            raise SignatureError("", "", text, str(exc)) from exc
        self.pos = 0
        self.t_counter = 0
        self.r_counter = 0
        self.bound: Dict[str, Type] = {}

    def fail(self, message: str) -> None:
        raise SignatureError("", "", self.text, message, self.pos)

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else (None, None)

    def next(self):
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect_punct(self, c: str) -> None:
        kind, txt = self.peek()
        if kind != "punct" or txt != c:
            self.fail(f"期望 {c!r}，实际得到 {txt!r}")
        self.next()

    def parse(self) -> ParsedSignature:
        kind, txt = self.peek()
        if kind == "name" and txt == "forall":
            self.next()
            names: List[str] = []
            while True:
                k2, n2 = self.peek()
                if k2 != "name" or n2 in _KEYWORDS:
                    self.fail("forall 后必须是变量名列表（空格或逗号分隔）")
                self.next()
                if n2 in names:
                    self.fail(f"forall 列表中变量 {n2!r} 重复")
                names.append(n2)
                k3, t3 = self.peek()
                if k3 == "punct" and t3 == ",":
                    self.next()
                    continue
                k4, t4 = self.peek()
                if k4 == "name" and t4 not in _KEYWORDS:
                    continue  # 空格分隔的下一个限定名
                break
            k4, t4 = self.peek()
            if k4 != "punct" or t4 != ".":
                self.fail("forall 列表后必须有 '.'")
            self.next()
            for nm in names:
                # 预留绑定；普通变量还是行变量取决于语法使用位置，
                # 遇到时再决定占位种类。
                self.bound[nm] = None  # type: ignore[assignment]
        t = self.parse_mono()
        if self.pos != len(self.tokens):
            self.fail(f"签名尾部有多余记号 {self.tokens[self.pos][1]!r}")
        qnames: Dict[Type, str] = {}
        for nm, placeholder in self.bound.items():
            if placeholder is None:
                self.fail(f"限定变量 {nm!r} 在签名体中没有出现")
            qnames[placeholder] = nm
        return ParsedSignature(t=t, qnames=qnames, text=self.text)

    def parse_mono(self) -> Type:
        left = self.parse_atom()
        kind, txt = self.peek()
        if kind == "arrow":
            self.next()
            right = self.parse_mono()
            return TFun(left, right, frozenset())
        if kind == "effopen":
            eff = self.parse_eff_label()
            right = self.parse_mono()
            return TFun(left, right, eff)
        return left

    def parse_eff_label(self) -> frozenset:
        self.next()  # -{
        labels: List[str] = []
        while True:
            k, t = self.peek()
            if k != "name" or t not in ALL_EFFECTS:
                self.fail(f"效应标签只能是 R/W/X，实际得到 {t!r}")
            self.next()
            if t in labels:
                self.fail(f"效应标签 {t} 重复")
            labels.append(t)
            k2, t2 = self.peek()
            if k2 == "punct" and t2 == ",":
                self.next()
                continue
            break
        # 期望 }->
        k3, t3 = self.peek()
        if k3 != "punct" or t3 != "}":
            self.fail(f"效应集合必须以 '}}->' 结束，实际得到 {t3!r}")
        self.next()
        k4, t4 = self.peek()
        if k4 != "arrow":
            self.fail(f"效应集合后期望 '->'，实际得到 {t4!r}")
        self.next()
        return frozenset(labels)

    def parse_atom(self) -> Type:
        kind, txt = self.peek()
        if kind == "name":
            self.next()
            if txt == "Int":
                return INT
            if txt == "Bool":
                return BOOL
            if txt == "Unit":
                return UNIT
            if txt == "Ref":
                self.expect_punct("(")
                inner = self.parse_mono()
                self.expect_punct(")")
                return TRef(inner)
            if txt in _KEYWORDS:
                self.fail(f"{txt!r} 不是合法的原子类型")
            return self.resolve_var(txt)
        if kind == "punct" and txt == "{":
            return self.parse_record()
        if kind == "punct" and txt == "(":
            self.next()
            t = self.parse_mono()
            self.expect_punct(")")
            return t
        self.fail(f"期望类型，实际得到 {txt!r}")

    def resolve_var(self, name: str) -> Type:
        if name not in self.bound:
            self.fail(
                f"自由类型变量 {name!r} 未在 forall 中限定"
                f"（签名必须完全显式：forall {name}. ...）"
            )
        placeholder = self.bound[name]
        if placeholder is None:
            placeholder = TVar(-1 - self.t_counter)
            self.t_counter += 1
            self.bound[name] = placeholder
        return placeholder

    def resolve_row_var(self, name: str) -> RVar:
        if name not in self.bound:
            self.fail(f"自由行变量 {name!r} 未在 forall 中限定")
        placeholder = self.bound[name]
        if placeholder is None:
            placeholder = RVar(-1 - self.r_counter)
            self.r_counter += 1
            self.bound[name] = placeholder
            return placeholder
        if not isinstance(placeholder, RVar):
            self.fail(f"{name!r} 已作为普通类型变量使用，不能再用作行变量")
        return placeholder

    def parse_record(self) -> Type:
        self.next()  # {
        fields: List[Tuple[str, Type]] = []
        rest: Type = REmpty()
        k, t = self.peek()
        if k == "punct" and t == "}":
            self.next()
            return TRec(TRow((), REmpty()))
        while True:
            # 两种入口：字段名 或 行变量 ;
            k1, t1 = self.peek()
            if k1 == "punct" and t1 == ";":
                self.next()
                k2, t2 = self.peek()
                if k2 != "name":
                    self.fail("';' 后必须是行变量名")
                self.next()
                rest = self.resolve_row_var(t2)
                self.expect_punct("}")
                break
            if k1 != "name" or t1 in _KEYWORDS:
                self.fail(f"记录字段必须是 名字: 类型，实际得到 {t1!r}")
            self.next()
            fname = t1
            self.expect_punct(":")
            ft = self.parse_mono()
            if any(fn == fname for fn, _ in fields):
                self.fail(f"记录签名中字段 {fname!r} 重名")
            fields.append((fname, ft))
            k3, t3 = self.peek()
            if k3 == "punct" and t3 == ",":
                self.next()
                continue
            if k3 == "punct" and t3 == "}":
                self.next()
                break
            if k3 == "punct" and t3 == ";":
                continue  # 回到循环顶吃 ;r
            self.fail(f"字段之间必须是 ','，记录以 '}}' 结束，实际得到 {t3!r}")
        names = tuple(n for n, _ in fields)
        if len(set(names)) != len(names):
            self.fail("记录字段重名")
        return TRec(TRow(tuple(fields), rest))


def parse_signature(text: str, module: str = "", export: str = "") -> ParsedSignature:
    """解析签名串。``module``/``export`` 仅用于错误定位。"""
    if not isinstance(text, str) or not text.strip():
        raise SignatureError(module, export, text, "签名必须是非空字符串")
    try:
        sig = _Parser(text).parse()
    except SignatureError as exc:
        raise SignatureError(module, export, text, exc.raw_message, exc.pos) from None
    return sig
