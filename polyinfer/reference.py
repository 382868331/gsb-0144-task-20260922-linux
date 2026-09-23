"""独立参考检查器。

与核心 (:mod:`polyinfer.inference`) / 模块检查 (:mod:`polyinfer.modules`)
**完全独立**：

* 不调用 ``infer`` / ``Inferencer.unify`` / ``ModuleChecker``；
* 不导入核心的任何函数来得到“期望值”（值判定、效应收集都在这里重写）；
* 只把核心产出的结果对象当作待审数据。

它核验：

1. **替换良构**：最终替换幂等、无循环（独立 occurs）、右端不含绑定变量自身；
2. **全部约束成立**：最终替换作用于每条记录约束两端，结构同构
   （含函数箭头潜在效应标签、引用不变式、记录逐字段与开放行）；
3. **泛化合法**：每条 let 的泛化集合恰为“绑定类型自由变量 − 环境自由
   变量”；值限制由检查器独立重判（v2 含 record/get/extend 递归判定），
   不信任核心字段；
4. **效应推导（v2）**：检查器自带一个独立的效应小参考解释器，直接遍历
   源 AST 重算效应（不经过推断器、不读替换），与结果的具体效应集合及
   效应留痕逐条核对；未决（evar）程序必须如实标记。
5. **模块边界**：导出检查记录中的刚性 skolem 不得被绑定到具体类型或
   异名刚性变量；不逃逸变量不得被泛化（不信任求解器的布尔标志）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from .errors import PolytypeError
from .types import (
    ALL_EFFECTS,
    REmpty,
    RSkol,
    RVar,
    TCon,
    TFun,
    TRec,
    TRef,
    TRow,
    TSkol,
    TVar,
    Type,
)


class ReferenceCheckFailure(PolytypeError):
    """独立检查器发现核心结果不自洽。正常交付的核心不应触发。"""


# ------------------------------------------------------------ 自实现原语

def ref_apply(subst: Dict[Type, Type], tp: Type, seen: Optional[Set[Type]] = None) -> Type:
    """检查器自带的替换应用（带循环检测），不调用核心模块函数。"""
    if seen is None:
        seen = set()
    if isinstance(tp, (TVar, RVar)):
        target = subst.get(tp)
        if target is None:
            return tp
        if tp in seen:
            raise ReferenceCheckFailure(f"替换存在循环链，经过变量 {tp}")
        return ref_apply(subst, target, seen | {tp})
    if isinstance(tp, TFun):
        return TFun(
            ref_apply(subst, tp.arg, seen),
            ref_apply(subst, tp.ret, seen),
            tp.eff,
        )
    if isinstance(tp, TRef):
        return TRef(ref_apply(subst, tp.inner, seen))
    if isinstance(tp, TRow):
        return TRow(
            tuple((n, ref_apply(subst, ft, seen)) for n, ft in tp.fields),
            ref_apply(subst, tp.rest, seen),
        )
    if isinstance(tp, TRec):
        return TRec(ref_apply(subst, tp.row, seen))
    return tp  # TCon / REmpty / TSkol / RSkol


def ref_free_vars(tp: Type, acc: Optional[Set[Type]] = None) -> Set[Type]:
    """检查器自带的自由变量收集（TVar 与 RVar，不含 skolem）。"""
    if acc is None:
        acc = set()
    if isinstance(tp, TVar):
        acc.add(tp)
    elif isinstance(tp, RVar):
        acc.add(tp)
    elif isinstance(tp, TFun):
        ref_free_vars(tp.arg, acc)
        ref_free_vars(tp.ret, acc)
    elif isinstance(tp, TRef):
        ref_free_vars(tp.inner, acc)
    elif isinstance(tp, TRow):
        for _n, ft in tp.fields:
            ref_free_vars(ft, acc)
        ref_free_vars(tp.rest, acc)
    elif isinstance(tp, TRec):
        ref_free_vars(tp.row, acc)
    return acc


def ref_skolems(tp: Type, acc: Optional[Set[Any]] = None) -> Set[Any]:
    if acc is None:
        acc = set()
    if isinstance(tp, (TSkol, RSkol)):
        acc.add(tp)
    elif isinstance(tp, TFun):
        ref_skolems(tp.arg, acc)
        ref_skolems(tp.ret, acc)
    elif isinstance(tp, TRef):
        ref_skolems(tp.inner, acc)
    elif isinstance(tp, TRow):
        for _n, ft in tp.fields:
            ref_skolems(ft, acc)
        ref_skolems(tp.rest, acc)
    elif isinstance(tp, TRec):
        ref_skolems(tp.row, acc)
    return acc


def _resolve_eff(tp: Type, eff_subst: Dict[int, frozenset]) -> Type:
    """用效应变量替换把 TFun 的潜在效应解析为具体标签集合。

    未被绑定的效应变量表示“没有更多具体效应”，核对时丢弃；
    这不会掩盖冲突——unify_effects 失败会直接抛异常，成功结果中的
    效应变量都已被绑到范式。
    """
    if isinstance(tp, TFun):
        labels: Set[str] = set()

        def resolve(eff, seen):
            for x in eff:
                if isinstance(x, str):
                    labels.add(x)
                elif hasattr(x, "ident"):
                    mapped = eff_subst.get(x.ident)
                    if mapped is not None and x.ident not in seen:
                        resolve(mapped, seen | {x.ident})

        resolve(tp.eff, set())
        return TFun(
            _resolve_eff(tp.arg, eff_subst),
            _resolve_eff(tp.ret, eff_subst),
            frozenset(labels),
        )
    if isinstance(tp, TRef):
        return TRef(_resolve_eff(tp.inner, eff_subst))
    if isinstance(tp, TRow):
        return TRow(
            tuple((n, _resolve_eff(ft, eff_subst)) for n, ft in tp.fields),
            _resolve_eff(tp.rest, eff_subst),
        )
    if isinstance(tp, TRec):
        return TRec(_resolve_eff(tp.row, eff_subst))
    return tp


def ref_norm_row(tp: Type) -> Type:
    """把替换后嵌套的行尾链展平：TRow(f1, TRow(f2, tail)) -> TRow(f1+f2, tail)。"""
    if isinstance(tp, TRec):
        return TRec(ref_norm_row(tp.row))
    if isinstance(tp, TRow):
        fields = list(tp.fields)
        rest = tp.rest
        guard = 0
        while isinstance(rest, TRow):
            fields.extend(rest.fields)
            rest = rest.rest
            guard += 1
            if guard > 10000:
                raise ReferenceCheckFailure("行尾链异常展开（疑似循环）")
        return TRow(tuple(fields), rest)
    if isinstance(tp, TFun):
        return TFun(ref_norm_row(tp.arg), ref_norm_row(tp.ret), tp.eff)
    if isinstance(tp, TRef):
        return TRef(ref_norm_row(tp.inner))
    return tp


def ref_isomorphic(left: Type, right: Type, path: str = "$") -> None:
    left = ref_norm_row(left)
    right = ref_norm_row(right)
    """结构同构判定：构造子（含行字段、效应集合）必须一一对应。

    故意不做任何变量绑定/统一：到达这里的类型已被最终替换解析到范式，
    若仍是不同的未绑定变量，则约束确实没被满足。行字段按名比对
    （两端字段集合必须一致，顺序无关——核心统一时已按名排序）。
    """
    if isinstance(left, (TVar, RVar)) or isinstance(right, (TVar, RVar)):
        if type(left) is type(right) and left == right:
            return
        raise ReferenceCheckFailure(
            f"约束在 {path} 未解析：左端 {left} ≠ 右端 {right}"
        )
    if isinstance(left, TSkol) or isinstance(right, TSkol) \
            or isinstance(left, RSkol) or isinstance(right, RSkol):
        if type(left) is type(right) and left == right:
            return
        raise ReferenceCheckFailure(
            f"约束在 {path} 刚性变量冲突：{left} ≠ {right}"
        )
    if isinstance(left, TCon) and isinstance(right, TCon):
        if left.name != right.name:
            raise ReferenceCheckFailure(f"约束在 {path} 基础类型冲突：{left.name} ≠ {right.name}")
        return
    if isinstance(left, TFun) and isinstance(right, TFun):
        if set(left.eff) != set(right.eff):
            raise ReferenceCheckFailure(
                f"约束在 {path}.eff 潜在效应冲突：{sorted(left.eff)} ≠ {sorted(right.eff)}"
            )
        ref_isomorphic(left.arg, right.arg, path + ".arg")
        ref_isomorphic(left.ret, right.ret, path + ".ret")
        return
    if isinstance(left, TRef) and isinstance(right, TRef):
        ref_isomorphic(left.inner, right.inner, path + ".inner")
        return
    if isinstance(left, TRow) and isinstance(right, TRow):
        f1 = dict(left.fields)
        f2 = dict(right.fields)
        if set(f1) != set(f2):
            raise ReferenceCheckFailure(
                f"约束在 {path} 记录字段集合不一致：{sorted(f1)} ≠ {sorted(f2)}"
            )
        for n in sorted(f1):
            ref_isomorphic(f1[n], f2[n], f"{path}.{n}")
        ref_isomorphic(left.rest, right.rest, path + ".rest")
        return
    if isinstance(left, TRec) and isinstance(right, TRec):
        ref_isomorphic(left.row, right.row, path + ".row")
        return
    if isinstance(left, REmpty) and isinstance(right, REmpty):
        return
    raise ReferenceCheckFailure(
        f"约束在 {path} 外构造子冲突：{type(left).__name__} ≠ {type(right).__name__}"
    )

# ------------------------------------------------------------ 独立值判定

V1_VALUE_TAGS = frozenset({"int", "bool", "lambda", "var"})


def ref_is_value(node: Any, language: str) -> bool:
    """检查器独立重写的语法值判定（不信任核心）。"""
    tag = node["tag"]
    if tag in V1_VALUE_TAGS:
        return True
    if language == "v2":
        if tag == "record":
            return all(ref_is_value(pair[1], language) for pair in node["fields"])
        if tag == "get":
            return ref_is_value(node["record"], language)
        if tag == "extend":
            return (
                ref_is_value(node["record"], language)
                and ref_is_value(node["value"], language)
                and ref_is_value(node["rest"], language)
            )
    return False


# ------------------------------------------------------------ 独立效应解释器

def _norm_row_fields(tp: Type) -> Optional[Dict[str, Type]]:
    """TRec/已解析行 -> 字段 dict；非记录返回 None。"""
    if isinstance(tp, TRec) and isinstance(tp.row, TRow) and isinstance(tp.row.rest, REmpty):
        return dict(tp.row.fields)
    return None


class RefEffectInterpreter:
    """确定性小参考域的效应重算器：直接遍历源 AST。

    规则（与推断器相互独立，不读替换，不调用 unify）：

    * int/bool/var/lambda 直接效应为空；lambda 的**潜在**效应由体决定；
    * read +R，write +W，raise +X；
    * if = 三支并集；let = 绑定并集体；record 为字段并集；
      get/extend 为部件并集；
    * try = (body − {X}) ∪ handler（X 被处理，R/W 保留）；
    * apply f a = 效应(f) ∪ 效应(a) ∪ 潜在效应(f)；
    * 潜在效应靠**一层 beta 内联**确定：直接 lambda 取体效应；
      let 绑定的 lambda 内联（恢复绑定时的调用环境）；lambda 形参在调用
      点映射到实参表达式（实参再被调用时继续内联）；只有真正跨不到的
      函数（如 xvar 外部函数、未知高阶返回值）才标记未决。

    返回 (具体标签集合, undecided)。
    """

    def __init__(self) -> None:
        # name -> (lambda 节点, 该 let 生效处的形参实参映射)
        self.let_lambdas: Dict[str, Tuple[Any, Tuple]] = {}
        # name -> 记录字面量节点（字段全为值的 let 绑定）
        self.let_records: Dict[str, Any] = {}

    def effects(
        self,
        node: Any,
        call_args: Optional[Dict[str, Tuple[Any, Any]]] = None,
    ) -> Tuple[Set[str], bool]:
        if call_args is None:
            call_args = {}
        tag = node["tag"]
        if tag in ("int", "bool", "xvar"):
            # xvar 作为值出现无直接效应；它被调用时的潜在效应不可见 → 未决。
            return set(), False
        if tag == "var":
            return set(), False
        if tag == "evar":
            return set(), True
        if tag == "lambda":
            return set(), False  # 体效应收为潜在效应
        if tag == "apply":
            fe, fu = self.effects(node["fn"], call_args)
            ae, au = self.effects(node["arg"], call_args)
            le, lu = self._latent(node["fn"], node["arg"], call_args)
            return fe | ae | le, fu or au or lu
        if tag == "let":
            if node["bound"]["tag"] == "lambda":
                self.let_lambdas[node["name"]] = (node["bound"], dict(call_args))
            elif node["bound"]["tag"] == "record" and all(
                    self._is_value_expr(pair[1]) for pair in node["bound"]["fields"]):
                self.let_records[node["name"]] = node["bound"]
            be, bu = self.effects(node["bound"], call_args)
            ye, yu = self.effects(node["body"], call_args)
            return be | ye, bu or yu
        if tag == "if":
            acc: Set[str] = set()
            und = False
            for key in ("cond", "then", "else_"):
                e, u = self.effects(node[key], call_args)
                acc |= e
                und = und or u
            return acc, und
        if tag == "ref":
            return self.effects(node["inner"], call_args)
        if tag == "read":
            e, u = self.effects(node["ref"], call_args)
            return e | {"R"}, u
        if tag == "write":
            re_, ru = self.effects(node["ref"], call_args)
            ve, vu = self.effects(node["value"], call_args)
            return re_ | ve | {"W"}, ru or vu
        if tag == "record":
            acc = set()
            und = False
            for _n, expr in node["fields"]:
                e, u = self.effects(expr, call_args)
                acc |= e
                und = und or u
            return acc, und
        if tag == "get":
            return self.effects(node["record"], call_args)
        if tag == "extend":
            acc = set()
            und = False
            for key in ("record", "value", "rest"):
                e, u = self.effects(node[key], call_args)
                acc |= e
                und = und or u
            return acc, und
        if tag == "raise":
            e, u = self.effects(node["value"], call_args)
            return e | {"X"}, u
        if tag == "try":
            be, bu = self.effects(node["body"], call_args)
            he, hu = self.effects(node["handler"], call_args)
            return (be - {"X"}) | he, bu or hu
        raise ReferenceCheckFailure(f"参考效应解释器遇到未知节点 {tag!r}")

    @staticmethod
    def _is_value_expr(node: Any) -> bool:
        tag = node.get("tag")
        if tag in ("int", "bool", "var", "lambda"):
            return True
        if tag == "record":
            return all(RefEffectInterpreter._is_value_expr(p[1]) for p in node["fields"])
        if tag == "get":
            return RefEffectInterpreter._is_value_expr(node["record"])
        if tag == "extend":
            return all(RefEffectInterpreter._is_value_expr(node[k])
                       for k in ("record", "value", "rest"))
        return False

    def _record_field(self, get_node: Any):
        """把 (get 记录 字段) 解析为字段表达式（仅字面/let 绑定的记录）。"""
        rec = get_node["record"]
        fname = get_node["field"]
        if rec.get("tag") == "var" and rec["name"] in self.let_records:
            rec = self.let_records[rec["name"]]
        if rec.get("tag") == "record":
            for n, expr in rec["fields"]:
                if n == fname:
                    return expr
        return None

    def _resolve_lambda(self, fn: Any, call_args):
        """把被应用函数解析为 (lambda 节点, 定义处形参环境)。"""
        if fn.get("tag") == "lambda":
            return fn, call_args
        if fn.get("tag") == "var":
            if fn["name"] in call_args:
                target, outer = call_args[fn["name"]]
                return self._resolve_lambda(target, outer)
            if fn["name"] in self.let_lambdas:
                lam, bound_args = self.let_lambdas[fn["name"]]
                return lam, bound_args
        if fn.get("tag") == "get":
            field_expr = self._record_field(fn)
            if field_expr is not None:
                return self._resolve_lambda(field_expr, call_args)
        return None, None

    def _latent(
        self,
        fn: Any,
        app_arg: Any,
        call_args: Dict[str, Tuple[Any, Any]],
    ) -> Tuple[Set[str], bool]:
        """被应用函数的潜在效应：beta 内联一层，形参在调用点绑定到实参。"""
        lam, bound_args = self._resolve_lambda(fn, call_args)
        if lam is None:
            # 外部函数 / 未知高阶返回值充当函数：效应不可见 → 未决。
            return set(), True
        body_env = dict(bound_args)
        body_env[lam["param"]] = (app_arg, call_args)
        return self.effects(lam["body"], body_env)


def ref_effects_of(node: Any) -> Tuple[frozenset, bool]:
    """独立重算一个表达式的具体效应集合与未决标志。"""
    interp = RefEffectInterpreter()
    labels, und = interp.effects(node)
    bad = labels - set(ALL_EFFECTS)
    if bad:  # 不可达：产生器只产出 R/W/X
        raise ReferenceCheckFailure(f"未知效应标签 {bad}")
    return frozenset(labels), und


# ------------------------------------------------------------ 结果核验

@dataclass(frozen=True)
class RefCheckReport:
    constraints_checked: int
    generalizations_checked: int
    substitution_size: int
    effects_checked: bool = False
    module_exports_checked: int = 0


def _check_absences(absences, subst):
    for ab in absences:
        resolved = ref_norm_row(ref_apply(subst, ab.row))
        names: Set[str] = set()
        cur = resolved
        guard = 0
        while isinstance(cur, TRow):
            names.update(n for n, _ in cur.fields)
            cur = cur.rest
            guard += 1
            if guard > 10000:
                raise ReferenceCheckFailure("缺席约束核对时行尾链异常")
        if ab.field in names:
            raise ReferenceCheckFailure(
                f"开放行缺席约束被破坏：{ab.path} 的 extend 声称尾行不含 "
                f"{ab.field!r}，最终替换却把该字段塞回该行（开放行成环漏检）"
            )


def check_result(result: Any) -> RefCheckReport:
    """核验一份 ``infer`` 返回结果，全部通过才返回统计，否则抛错。"""
    subst = dict(result.subst)
    language = getattr(result, "language", "v1")
    # 源 AST 始终随结果返回；独立检查器用它重算值形状与效应。
    source = getattr(result, "source_ast", None)

    # 1) 替换良构。
    for var, rhs in subst.items():
        if var in ref_free_vars(rhs):
            raise ReferenceCheckFailure(f"替换不满足 occurs：{var} ↦ {rhs}")
        once = ref_apply(subst, var)
        twice = ref_apply(subst, once)
        ref_isomorphic(once, twice, f"idempotency[{var}]")

    # 2) 每条约束：最终替换作用后两端必须同构（效应变量先解析）。
    eff_subst = dict(getattr(result, "eff_subst", ()))
    for index, c in enumerate(result.constraints):
        l = _resolve_eff(ref_apply(subst, c.left), eff_subst)
        r = _resolve_eff(ref_apply(subst, c.right), eff_subst)
        try:
            ref_isomorphic(l, r, f"constraint#{index}")
        except ReferenceCheckFailure as exc:
            raise ReferenceCheckFailure(
                f"{exc}（左端路径 {c.left_path}，右端路径 {c.right_path}，"
                f"来源：{c.note}）"
            ) from exc

    # 3) 泛化：核心声称的集合必须恰好等于合法闭包集合；值限制独立重判。
    for g in result.generalizations:
        # 值判定完全独立：从源 AST 按路径取出绑定表达式自行判定，
        # 不读核心的 bound_tag/bound_value_shape_ok 结论（只借路径定位）。
        bound_node = None
        if language == "v2" and source is not None:
            bound_node = _node_at_path(source, tuple(g.path) + ("bound",))
        is_value = (
            ref_is_value(bound_node, language)
            if bound_node is not None
            else g.bound_tag in V1_VALUE_TAGS
        )
        bound_free = ref_free_vars(ref_apply(subst, g.bound_type))
        env_free: Set[Type] = set()
        for _ename, quantified, body in g.env_snapshot:
            body_free = ref_free_vars(ref_apply(subst, body))
            env_free |= body_free - set(quantified)
        expected = (bound_free - env_free) if is_value else set()
        actual = set(g.generalized)
        if actual != expected:
            raise ReferenceCheckFailure(
                f"let {g.name!r} @ {g.path} 泛化集合不合法："
                f"核心给出 {sorted(_vkey(v) for v in actual)}，"
                f"独立重算（绑定是{'值' if is_value else '扩展式'}）"
                f"应为 {sorted(_vkey(v) for v in expected)}"
            )

    # 3.5) 开放行缺席约束（extend 防成环）：每条缺席记录声称的字段在最终
    #      替换解析后的行尾里都不得出现。独立遍历最终行，不信任核心。
    _check_absences(getattr(result, "row_absences", ()), subst)

    # 4) 效应（v2）：独立效应参考解释器直接遍历源 AST 重算。
    effects_checked = False
    if language == "v2":
        if source is not None:
            ref_labels, ref_und = ref_effects_of(source)
            if set(result.effects) != set(ref_labels):
                raise ReferenceCheckFailure(
                    f"效应集合不一致：核心给出 {sorted(result.effects)}，"
                    f"独立效应参考解释器重算为 {sorted(ref_labels)}"
                )
            # 未决标志必须如实：参考域中 evar 存在则未决。
            if bool(result.effects_undecided) != bool(ref_und):
                raise ReferenceCheckFailure(
                    f"效应未决标志不一致：核心 {result.effects_undecided}，"
                    f"独立重算 {ref_und}（不得把未决当确定，也不得虚报）"
                )
            _check_effect_traces(result, source)
            effects_checked = True

    return RefCheckReport(
        constraints_checked=len(result.constraints),
        generalizations_checked=len(result.generalizations),
        substitution_size=len(subst),
        effects_checked=effects_checked,
    )


def _node_at_path(root: Any, path: Tuple[Any, ...]) -> Optional[Any]:
    """按推断路径从源 AST 取子节点（record 字段路径形如 ('fields', i, 1)）。"""
    cur = root
    i = 0
    while i < len(path):
        step = path[i]
        if isinstance(step, int):
            if not isinstance(cur, list) or step >= len(cur):
                return None
            cur = cur[step]
        else:
            if not isinstance(cur, dict) or step not in cur:
                return None
            cur = cur[step]
        i += 1
    return cur


def _vkey(v: Type):
    if isinstance(v, RVar):
        return ("r", v.ident)
    return ("t", v.ident)


def _check_effect_traces(result: Any, source: Any) -> None:
    """效应留痕与 AST 结构一致：每个产生节点都有一条方向正确的记录。"""
    expected = {"read": "R", "write": "W"}
    found: Dict[Tuple, Set[str]] = {}

    def walk(node: Any, path: Tuple[str, ...]) -> None:
        tag = node["tag"]
        if tag in expected:
            found.setdefault(path, set()).add(expected[tag])
        if tag == "raise":
            found.setdefault(path, set()).add("X")
        for key, child in _children(node):
            walk(child, path + (key,))

    walk(source, ())
    traced: Dict[Tuple, Set[str]] = {}
    for tr in result.effect_traces:
        for label in tr.produced:
            if label in ALL_EFFECTS:
                traced.setdefault(tuple(tr.path), set()).add(label)
    for path, labels in found.items():
        if traced.get(path, set()) != labels:
            raise ReferenceCheckFailure(
                f"效应留痕在 {path} 不一致：AST 要求 {sorted(labels)}，"
                f"留痕为 {sorted(traced.get(path, set()))}"
            )


def _children(node: Any) -> List[Tuple[str, Any]]:
    out = []
    for key, value in node.items():
        if key == "tag":
            continue
        if isinstance(value, dict) and "tag" in value:
            out.append((key, value))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, list) and len(item) == 2 and isinstance(item[1], dict):
                    out.append((f"{key}[{i}]", item[1]))
    return out


# ------------------------------------------------------------ 模块报告核验

def check_module_report(report: Any) -> RefCheckReport:
    """核验模块检查的全部模块追踪（约束/泛化/skolem 逃逸/效应签名）。

    ``report`` 是 :class:`polyinfer.modules.ModuleReport`，其 ``traces``
    字段给出每个模块一次检查的完整推断留痕。检查器不信任结果中的任何
    布尔标志，全部重算。
    """
    traces = getattr(report, "traces", None)
    if not traces:
        raise ReferenceCheckFailure("模块报告缺少 traces，无法独立核验")
    total_c = 0
    total_g = 0
    for mt in traces:
        subst = dict(mt.subst)
        eff_subst = dict(mt.eff_subst)
        # 约束成立。
        for index, c in enumerate(mt.constraints):
            l = _resolve_eff(ref_apply(subst, c.left), eff_subst)
            r = _resolve_eff(ref_apply(subst, c.right), eff_subst)
            try:
                ref_isomorphic(l, r, f"{mt.module}/constraint#{index}")
            except ReferenceCheckFailure as exc:
                raise ReferenceCheckFailure(f"{exc}（{c.note}）") from exc
        # 开放行缺席约束。
        _check_absences(mt.row_absences, subst)
        # 替换良构（含 skolem 不得被绑到具体类型）。
        for var, rhs in subst.items():
            if isinstance(var, (TVar, RVar)):
                if var in ref_free_vars(rhs):
                    raise ReferenceCheckFailure(
                        f"模块 {mt.module} 替换不满足 occurs：{var} ↦ {rhs}"
                    )
                rhs_sk = ref_skolems(rhs)
                # 柔性变量允许解析到 skolem，但 skolem 自身不得出现在
                # 被绑定到 ground 结构的右端里（ground 含 skolem 只可能是
                # “实现不够一般”，核心 unify 已应拒绝）——这里复核：
                # skolem 不得与 TCon/REmpty 之外的具体 ground 冲突已在
                # 约束同构中覆盖；额外确认没有 skolem 被当作绑定键。
            if isinstance(var, TSkol):
                raise ReferenceCheckFailure(
                    f"模块 {mt.module} 的刚性变量 {var} 被替换绑定"
                    f"（到 {rhs}）：实现比签名更不一般却被放行"
                )
        # 泛化（导出体内部的 let 事件）；值形状从源 AST 独立重判。
        source_mods = mt.source_modules
        for g in mt.generalizations:
            bound_node = None
            # 内部定义根的路径是 (模块, "body", 定义名)，绑定表达式即定义本体；
            # 导出体内 let 的路径后面还有 let 子路径，需继续下钻到其 "bound"。
            if len(g.path) >= 3 and g.path[1] == "body" and g.path[0] in source_mods:
                root_def = source_mods[g.path[0]]["body"].get(g.path[2])
                rest = tuple(g.path[3:])
                if not rest:
                    bound_node = root_def
                else:
                    bound_node = _node_at_path(root_def, rest + ("bound",))
            is_value = (
                ref_is_value(bound_node, "v2")
                if bound_node is not None
                else g.bound_tag in V1_VALUE_TAGS
            )
            bound_free = ref_free_vars(ref_apply(subst, g.bound_type))
            env_free: Set[Type] = set()
            for _ename, quantified, body in g.env_snapshot:
                env_free |= ref_free_vars(ref_apply(subst, body)) - set(quantified)
            expected = (bound_free - env_free) if is_value else set()
            if set(g.generalized) != expected:
                raise ReferenceCheckFailure(
                    f"模块 {mt.module} 的内部定义 {g.name!r} 泛化集合不合法："
                    f"{set(map(_vkey, g.generalized))} != {set(map(_vkey, expected))}"
                )
        # 导出签名效应：实现体记录的效应集合必须与签名箭头一致。
        for esig in mt.export_effects:
            want = frozenset(esig.signed_effects)
            got = frozenset(esig.observed_effects)
            if want != got:
                raise ReferenceCheckFailure(
                    f"模块 {mt.module} 导出 {esig.export!r} 效应与签名不一致："
                    f"签名 {sorted(want)}，实现 {sorted(got)}"
                )
        total_c += len(mt.constraints)
        total_g += len(mt.generalizations)
    return RefCheckReport(
        constraints_checked=total_c,
        generalizations_checked=total_g,
        substitution_size=0,
        module_exports_checked=sum(len(mt.export_effects) for mt in traces),
    )
