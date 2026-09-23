"""模块系统：递归模块、签名两阶段检查、依赖 SCC 与增量缓存。

职责
====

1. **结构/签名先行**：按依赖图的强连通分量（Tarjan）推进；每个 SCC 先
   解析并登记其中**全部**导出签名（签名阶段），再检查其中每个导出的
   实现体（实现阶段）。模块可互相递归引用（同一 SCC 内的模块按各自
   声称的签名互见）。
2. **实现一致性**：导出签名的限定变量在实现检查处实例化为刚性 skolem；
   实现体的类型与效应必须与签名逐点一致。0 层分配的可变引用单元内的
   类型变量若被绑定到 skolem，即拒绝（模块边界不得偷偷泛化可变引用）。
3. **缓存**：缓存键为（本模块实现体的规范哈希，各**直接依赖**的签名
   指纹），签名指纹包含效应标注；因此依赖签名（或效应）变化才失效，
   且失效集是依赖图上的精确逆向可达闭包，不是全量清空。
4. **失败原子性**：一次重新检查失败时，上一次的成功结果保留不动。
5. **非缓存路径**：:meth:`ModuleChecker.check_all` 可在
   ``use_cache=False`` 下从零重查；测试用它与增量路径逐项比对。
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from . import ast as ast_mod
from .errors import PolytypeError
from .inference import Inferencer, TypeScheme, _all_free
from .signature import ParsedSignature, parse_signature
from .types import (
    TFun,
    TSkol,
    TVar,
    Type,
    apply,
    canonical_names,
    render_type,
)


class ModuleStructureError(PolytypeError):
    """模块包结构不满足契约（内部定义成环、SCC 内签名缺失等）。"""


# ---------------------------------------------------------------- 结果

@dataclass(frozen=True)
class ExportResult:
    module: str
    export: str
    signature: str
    principal_type: str           # 实现体推导出的类型（规范命名渲染）
    names: Dict[Any, str]
    effects: frozenset
    effects_undecided: bool
    constraints: int
    node_count: int


@dataclass(frozen=True)
class ExportEffectSig:
    """一个导出的效应签名核对项：签名声称的潜在效应 vs 实现观察值。"""

    export: str
    signed_effects: Tuple[str, ...]
    observed_effects: Tuple[str, ...]


@dataclass(frozen=True)
class ModuleTrace:
    """一个模块一次检查的完整留痕，供独立检查器审计。"""

    module: str
    subst: Dict[Any, Any]
    eff_subst: Dict[int, frozenset]
    constraints: Tuple[Any, ...]
    generalizations: Tuple[Any, ...]
    row_absences: Tuple[Any, ...]
    export_effects: Tuple[ExportEffectSig, ...]
    source_modules: Dict[str, Dict[str, Any]]


@dataclass(frozen=True)
class ModuleReport:
    modules: int
    sccs: Tuple[Tuple[str, ...], ...]
    order: Tuple[str, ...]
    exports: Tuple[ExportResult, ...]
    cache_hits: Tuple[str, ...]
    cache_stored: Tuple[str, ...]
    invalidated: Tuple[str, ...]
    from_cache: bool
    traces: Tuple[ModuleTrace, ...] = ()


# ---------------------------------------------------------------- 工具

def _canonical_json(obj: Any) -> str:
    """确定性 JSON（字典键排序），用于实现体哈希。"""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _module_impl_hash(mod: Dict[str, Any]) -> str:
    return hashlib.sha256(
        _canonical_json({"body": mod["body"]}).encode("utf-8")
    ).hexdigest()[:16]


def _tarjan(graph: Dict[str, Set[str]]) -> List[List[str]]:
    """标准 Tarjan SCC；返回的 SCC 按依赖拓扑排序（被依赖者在前）。"""
    index = {}
    low = {}
    stack: List[str] = []
    on_stack: Set[str] = set()
    counter = [0]
    result: List[List[str]] = []

    def strong(v: str) -> None:
        index[v] = counter[0]
        low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in sorted(graph.get(v, ())):
            if w not in index:
                strong(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                comp.append(w)
                if w == v:
                    break
            result.append(sorted(comp))

    for v in sorted(graph):
        if v not in index:
            strong(v)
    return result


# ---------------------------------------------------------------- 检查器

class ModuleChecker:
    def __init__(self, modules: Dict[str, Dict[str, Any]], max_nodes: int = 20000) -> None:
        self.modules = modules
        self.max_nodes = max_nodes
        # 模块名 -> 依赖集（AST xvar 与 imports 字段的并集）。
        self.deps: Dict[str, Set[str]] = {}
        # 模块名 -> {导出: 解析后的签名}。
        self.sigs: Dict[str, Dict[str, ParsedSignature]] = {}
        # 模块名 -> {内部定义: TypeScheme}（推断一次，供导出检查复用）。
        self.internal_schemes: Dict[str, Dict[str, TypeScheme]] = {}
        # 缓存：模块名 -> (缓存键, ExportResult 元组)。
        self._cache: Dict[str, Tuple[str, Tuple[ExportResult, ...]]] = {}
        self._last_invalidated: Set[str] = set()
        # (模块, 导出, 限定名) -> 共享 skolem。
        self._skol_cache: Dict[Tuple[str, str, str], Any] = {}

    # ------------------------------------------------------------ 图

    def _build_graph(self) -> None:
        for mname, mod in self.modules.items():
            deps: Set[str] = set(mod.get("imports", ()))

            def walk(n: Any) -> None:
                if isinstance(n, dict):
                    if n.get("tag") == "xvar":
                        deps.add(n["module"])
                    # 不能按键名跳过：定义名（如 "tag"）也可能叫 tag，
                    # 其值是需要继续遍历的 AST。
                    for v in n.values():
                        walk(v)
                elif isinstance(n, list):
                    for item in n:
                        walk(item)

            walk(mod["body"])
            deps.discard(mname)
            self.deps[mname] = deps

    def _reverse_deps(self) -> Dict[str, Set[str]]:
        rev: Dict[str, Set[str]] = {m: set() for m in self.modules}
        for m, ds in self.deps.items():
            for d in ds:
                rev.setdefault(d, set()).add(m)
        return rev

    def invalidation_set(self, changed: str) -> Tuple[str, ...]:
        """改变模块 ``changed`` 后的精确缓存失效集：

        它本身 + 依赖图上所有（传递）依赖它的模块。与“缓存键实际引用
        的依赖签名”一致，不需要清空全缓存。
        """
        if not self.deps:
            self._build_graph()
        rev = self._reverse_deps()
        hit: Set[str] = {changed}
        stack = [changed]
        while stack:
            cur = stack.pop()
            for up in rev.get(cur, ()):
                if up not in hit:
                    hit.add(up)
                    stack.append(up)
        return tuple(sorted(hit))

    # ------------------------------------------------------------ 签名

    def _signature_fingerprint(self, mname: str) -> str:
        """模块对外签名指纹：导出（名 :: 规范签名串）排序后哈希。

        包含效应标注（签名串本身含 ``-{R,W,X}->``），因此效应契约变化
        会改变所有依赖者的缓存键。
        """
        parts = [
            f"{name} :: {sig.render()}"
            for name, sig in sorted(self.sigs[mname].items())
        ]
        return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]

    def _cache_key(self, mname: str) -> str:
        own = _module_impl_hash(self.modules[mname])
        dep_fps = sorted(
            (d, self._signature_fingerprint(d)) for d in self.deps[mname]
        )
        return _canonical_json({"own": own, "deps": dep_fps, "max_nodes": self.max_nodes})

    def _parse_all_signatures(self) -> None:
        for mname, mod in self.modules.items():
            parsed: Dict[str, ParsedSignature] = {}
            for ename, text in sorted(mod["exports"].items()):
                parsed[ename] = parse_signature(text, mname, ename)
            self.sigs[mname] = parsed

    # ------------------------------------------------------------ 实现

    def _asserted_scheme(self, mname: str, ename: str) -> TypeScheme:
        """从签名构造（未实例化的）方案；限定占位是签名里的负 id 变量，
        每次 :meth:`Inferencer.instantiate` 都会换新变量。"""
        psig = self.sigs[mname][ename]
        return TypeScheme(frozenset(psig.quantified()), psig.t)

    def _infer_internals(self, mname: str, checker: Inferencer,
                         env: Dict[str, TypeScheme]) -> Dict[str, TypeScheme]:
        """推断模块的内部（非导出）定义（在调用方提供的共享 checker 上）。

        内部定义不允许成环（成环的定义必须导出并给签名）：先在局部调用
        图上拓扑排序，再按顺序像 let 绑定一样推断/泛化
        （0 层可变单元变量不泛化）。
        """
        mod = self.modules[mname]
        internal = [n for n in mod["body"] if n not in mod["exports"]]
        calls: Dict[str, Set[str]] = {n: set() for n in internal}

        def collect(name: str, node: Any) -> None:
            if isinstance(node, dict):
                if node.get("tag") == "var" and node["name"] in calls:
                    calls[name].add(node["name"])
                for k, v in node.items():
                    if k != "tag":
                        collect(name, v)

        for n in internal:
            collect(n, mod["body"][n])
        # 内部定义间的环检测（DFS 三色）。
        color: Dict[str, int] = {n: 0 for n in internal}
        order: List[str] = []

        def visit(n: str, stack: Tuple[str, ...]) -> None:
            if color[n] == 1:
                cycle = stack[stack.index(n):] + (n,)
                raise ModuleStructureError(
                    f"模块 {mname!r} 的内部定义形成递归环 {' -> '.join(cycle)}："
                    f"递归定义必须列入 exports 并提供签名"
                )
            if color[n] == 2:
                return
            color[n] = 1
            for d in sorted(calls[n]):
                visit(d, stack + (n,))
            color[n] = 2
            order.append(n)

        for n in sorted(internal):
            visit(n, ())

        schemes: Dict[str, TypeScheme] = {}
        subst: Dict[Any, Any] = {}
        for name in order:
            s, t, eff = checker.ti(mod["body"][name], env, subst, (mname, "body", name))
            subst = s
            t = apply(s, t)
            from .inference import Generalization, is_syntactic_value

            node = mod["body"][name]
            if is_syntactic_value(node):
                env_free = _env_free(env, s)
                quantified = tuple(
                    v for v in _all_free(t)
                    if v not in env_free and v not in checker.unsafe_vars
                    and not isinstance(v, TSkol)
                )
                schemes[name] = TypeScheme(frozenset(quantified), t)
            else:
                quantified = ()
                schemes[name] = TypeScheme(frozenset(), t)
            # 内部定义泛化事件同样留痕，供独立检查器按源 AST 复核值限制。
            checker.generalizations.append(
                Generalization(
                    path=(mname, "body", name),
                    name=name,
                    bound_tag=node["tag"],
                    bound_type=t,
                    generalized=quantified,
                    env_snapshot=tuple(
                        (en, es.quantified, apply(s, es.t)) for en, es in env.items()
                    ),
                    bound_fields=(
                        tuple(p[0] for p in node["fields"]) if node["tag"] == "record" else ()
                    ),
                    bound_value_shape_ok=is_syntactic_value(node),
                )
            )
            env[name] = schemes[name]
        self.internal_schemes[mname] = schemes
        # 返回累积替换，供随后的导出检查在同一状态上继续。
        return schemes, subst

    def _new_checker(self) -> Inferencer:
        checker = Inferencer(language="v2")
        # 外部引用表由各次检查的环境决定；xvar 经 external_env 查方案。
        checker.external_env = {}
        return checker

    def _ambient_env(
        self, mname: str, checker: Inferencer, exclude_self: bool
    ) -> Dict[str, TypeScheme]:
        """构造本模块的局部名字环境：同模块全部导出的声称方案
        （exclude_self=True 时排除自身，用于非递归对照）+ 已推断内部定义。"""
        env: Dict[str, TypeScheme] = {}
        for ename in self.modules[mname]["exports"]:
            if exclude_self and ename == mname:
                continue
            env[ename] = self._asserted_scheme(mname, ename)
        for iname, sch in self.internal_schemes.get(mname, {}).items():
            env[iname] = sch
        return env

    def _check_one_export(
        self,
        mname: str,
        ename: str,
        checker: Inferencer,
        env: Dict[str, TypeScheme],
        subst: Dict[Any, Any],
    ) -> Tuple[ExportResult, ExportEffectSig, Dict[Any, Any]]:
        mod = self.modules[mname]
        node = mod["body"][ename]
        psig = self.sigs[mname][ename]

        # 显式签名契约不接受未决效应：导出体中若含 evar（效应未知占位），
        # 无法与完全具体的签名效应逐点核对，必须如实拒绝，而不是把未决
        # 悄悄绑成空集洗白为纯函数。
        if _contains_evar(node):
            from .errors import EffectMismatchError

            raise EffectMismatchError(
                (mname, "body", ename), (mname, "exports", ename),
                frozenset(), frozenset(),
                "导出实现含未决效应占位 evar：显式签名只接受确定的 R/W/X 效应",
            )

        n_constraints_before = len(checker.constraints)
        # 声称签名在“实现检查处”实例化：普通限定变量 -> 刚性 skolem，
        # 行变量 -> 柔性新行变量。同一导出的同名限定变量在整个模块检查中
        # 复用同一个 skolem（递归自调用因此按签名语义保持一致）。
        skol_t = psig.instantiate_skol(
            lambda nm: self._skol(checker, mname, ename, nm),
            checker.fresh_rvar,
            lambda nm: self._rskol(checker, mname, ename, nm),
        )
        # 自递归引用：在本导出体的检查期间，把自身名字绑定为“同一组
        # skolem 的单态方案”——体中每次递归调用都看到同一刚性签名，
        # 多态递归无法被悄悄特化；其他导出仍按柔性多态方案使用本导出。
        prev_self = env.get(ename)
        env[ename] = TypeScheme(frozenset(), skol_t)
        try:
            s, body_t, body_eff = checker.ti(node, env, subst, (mname, "body", ename))
        finally:
            if prev_self is None:
                env.pop(ename, None)
            else:
                env[ename] = prev_self
        body_t0 = apply(s, body_t)

        s2 = checker.unify(
            s, body_t0, skol_t,
            (mname, "body", ename), (mname, "exports", ename),
            f"导出 {mname}.{ename} 的实现必须符合签名",
        )
        # 非函数导出：直接效应也必须为空（签名只能给纯值，异常/写效应
        # 不能藏在返回值里）。
        checker.unify_effects(
            checker.eff_resolve(body_eff), frozenset(),
            (mname, "body", ename), (mname, "exports", ename),
            f"导出 {mname}.{ename} 的直接效应必须为空",
        )
        final_t = apply(s2, body_t)
        # 用签名里的限定源名渲染（成功时实现类型已与刚性签名一致）。
        display_names: Dict[Any, str] = {}
        for qname in psig.qnames.values():
            sk = self._skol_cache.get((mname, ename, qname))
            if sk is not None:
                display_names[sk] = qname        # 残余柔性变量（不应在合法结果中出现）仍按首次出现补名。
        for v, nm in canonical_names(final_t).items():
            display_names.setdefault(v, nm)
        # 外层箭头效应：签名声称 vs 实现观察（嵌套箭头已在 unify 中逐点核过）。
        signed_outer = _outer_effects(psig.t)
        observed_outer = _outer_effect_labels(apply(s2, body_t))
        eff_sig = ExportEffectSig(
            export=ename,
            signed_effects=tuple(sorted(signed_outer)),
            observed_effects=tuple(sorted(observed_outer)),
        )
        result = ExportResult(
            module=mname,
            export=ename,
            signature=psig.render(),
            principal_type=render_type(final_t, display_names),
            names=dict(display_names),
            effects=frozenset(x for x in checker.eff_resolve(body_eff) if isinstance(x, str)),
            effects_undecided=any(
                not isinstance(x, str) for x in checker.eff_resolve(body_eff)
            ),
            constraints=len(checker.constraints) - n_constraints_before,
            node_count=sum(1 for _ in _iter_nodes(node)),
        )
        return result, eff_sig, s2

    def _skol(self, checker: Inferencer, mname: str, ename: str, qname: str) -> TSkol:
        """同一 (模块, 导出, 限定名) 在本次模块检查内复用同一刚性变量。"""
        key = (mname, ename, qname)
        cached = self._skol_cache.get(key)
        if cached is None:
            cached = checker.fresh_skol(qname)
            self._skol_cache[key] = cached
        return cached

    def _rskol(self, checker: Inferencer, mname: str, ename: str, qname: str):
        """行限定名对应的刚性行变量（同样按 (模块, 导出, 名) 复用）。"""
        from .types import RSkol

        key = (mname, ename, qname)
        cached = self._skol_cache.get(key)
        if cached is None:
            cached = checker.fresh_rskol(qname)
            self._skol_cache[key] = cached
        if not isinstance(cached, RSkol):
            # 同一限定名既被用作普通变量又被用作行变量：解析阶段已分开占位，
            # 正常不会撞键；撞了说明签名构造异常，按结构错误拒绝。
            raise ModuleStructureError(
                f"模块 {mname!r} 导出 {ename!r} 的限定名 {qname!r} 种类冲突"
            )
        return cached

    def _check_module(self, mname: str) -> Tuple[Tuple[ExportResult, ...], ModuleTrace]:
        # 整个模块共用一个 Inferencer：内部定义（含 0 层可变单元）与各导出
        # 在同一替换/计数器状态中检查，跨导出共享的单态单元因此真正共享，
        # 也避免了不同检查间 TVar 内部 id 相撞。
        checker = self._new_checker()
        self._skol_cache = {}
        n_c0 = len(checker.constraints)
        ext: Dict[Tuple[str, str], TypeScheme] = {}
        for dep in self.deps[mname]:
            for dname in self.sigs[dep]:
                ext[(dep, dname)] = self._asserted_scheme(dep, dname)
        checker.external_env = ext

        env: Dict[str, TypeScheme] = {}
        for ename in self.modules[mname]["exports"]:
            env[ename] = self._asserted_scheme(mname, ename)
        _schemes, init_subst = self._infer_internals(mname, checker, env)

        out = []
        eff_sigs: List[ExportEffectSig] = []
        subst: Dict[Any, Any] = init_subst
        for ename in sorted(self.modules[mname]["exports"]):
            er, es, subst = self._check_one_export(mname, ename, checker, env, subst)
            out.append(er)
            eff_sigs.append(es)
        trace = ModuleTrace(
            module=mname,
            subst=dict(subst),
            eff_subst=dict(checker.eff_subst),
            constraints=tuple(checker.constraints[n_c0:]),
            generalizations=tuple(checker.generalizations),
            row_absences=tuple(checker.row_absences),
            export_effects=tuple(eff_sigs),
            source_modules=dict(self.modules),
        )
        return tuple(out), trace

    # ------------------------------------------------------------ 入口

    def check_all(
        self,
        use_cache: bool = True,
        changed: Optional[Set[str]] = None,
    ) -> ModuleReport:
        """检查全部模块。

        * ``use_cache=False``：从零重查（非缓存检查路径），结果可与
          缓存路径逐项比较；
        * ``changed``：声明本次发生修改的模块集；缓存只失效
          :meth:`invalidation_set` 给出的精确集合；未失效且键未变的
          模块直接命中。一次模块检查失败时，旧缓存条目保持不变。
        """
        if isinstance(self.max_nodes, bool) or not isinstance(self.max_nodes, int):
            raise TypeError(
                f"max_nodes 必须是 int，实际为 {type(self.max_nodes).__name__}"
            )
        if self.max_nodes < 1:
            raise ValueError(f"max_nodes 必须为正整数，实际为 {self.max_nodes}")
        validator = ast_mod.ModuleValidator(self.modules, self.max_nodes)
        old_limit = sys.getrecursionlimit()
        needed = 200 * self.max_nodes + 1000
        if needed > old_limit:
            sys.setrecursionlimit(needed)
        try:
            validator.validate()
            self._build_graph()
            self._parse_all_signatures()
            sccs = _tarjan(self.deps)
            order = [m for comp in sccs for m in comp]

            if not use_cache:
                self._cache = {}

            # 计算失效集（仅在增量路径）；非缓存路径等价于全部失效但不写回语义。
            invalidated: Set[str] = set()
            if use_cache and changed:
                for ch in changed:
                    if ch not in self.modules:
                        raise ModuleStructureError(f"被修改的模块 {ch!r} 不在模块包中")
                    invalidated.update(self.invalidation_set(ch))
            self._last_invalidated = set(invalidated)

            hits: List[str] = []
            stored: List[str] = []
            export_results: Dict[str, Tuple[ExportResult, ...]] = {}
            traces_by_name: Dict[str, ModuleTrace] = {}

            # 两阶段：先登记本 SCC 全部签名（上面已统一解析），再按 SCC 顺序
            # 检查实现；SCC 内部模块按名排序，结果确定。
            for mname in order:
                key = self._cache_key(mname)
                cached = self._cache.get(mname)
                if use_cache and mname not in invalidated and cached and cached[0] == key:
                    export_results[mname] = cached[1]
                    traces_by_name[mname] = cached[2]
                    hits.append(mname)
                    continue
                # 检查；失败时不替换上一次有效结果（缓存原子性）。
                old = self._cache.get(mname)
                try:
                    results, trace = self._check_module(mname)
                except Exception:
                    if old is not None:
                        self._cache[mname] = old  # 显式保留旧结果
                    raise
                self._cache[mname] = (key, results, trace)
                stored.append(mname)
                export_results[mname] = results
                traces_by_name[mname] = trace
        finally:
            sys.setrecursionlimit(old_limit)

        all_exports = tuple(
            er
            for mname in order
            for er in export_results[mname]
        )
        return ModuleReport(
            modules=len(self.modules),
            sccs=tuple(tuple(comp) for comp in sccs),
            order=tuple(order),
            exports=all_exports,
            cache_hits=tuple(hits),
            cache_stored=tuple(stored),
            invalidated=tuple(sorted(invalidated)),
            from_cache=use_cache,
            traces=tuple(traces_by_name[m] for m in order),
        )


def _env_free(env, s):  # 与 inference._env_free_vars 同语义的本地副本
    free = set()
    for scheme in env.values():
        body = apply(s, scheme.t)
        for v in _all_free(body):
            if v not in scheme.quantified:
                free.add(v)
    return free


def _outer_effects(t: Type) -> frozenset:
    """签名类型最外层箭头的效应集合（非函数为空集）。"""
    return t.eff if isinstance(t, TFun) else frozenset()


def _outer_effect_labels(t: Type) -> frozenset:
    """实现类型最外层箭头的**具体**效应标签（去掉未决效应变量）。"""
    if not isinstance(t, TFun):
        return frozenset()
    return frozenset(x for x in t.eff if isinstance(x, str))


def _iter_nodes(node: Any):
    """遍历 AST 节点（含 xvar），仅用于计数。"""
    if isinstance(node, dict) and "tag" in node:
        yield node
        for k, v in node.items():
            if k == "tag":
                continue
            if isinstance(v, list):
                for item in v:
                    yield from _iter_nodes(item)
            else:
                yield from _iter_nodes(v)


def _contains_evar(node: Any) -> bool:
    """AST 中是否含未决效应占位 evar。"""
    return any(n.get("tag") == "evar" for n in _iter_nodes(node))


def check_modules(
    modules: Dict[str, Dict[str, Any]],
    max_nodes: int = 5000,
    use_cache: bool = True,
    changed: Optional[Set[str]] = None,
) -> Tuple[ModuleChecker, ModuleReport]:
    """便捷入口：构造 :class:`ModuleChecker` 并执行一次全量/增量检查。"""
    checker = ModuleChecker(modules, max_nodes=max_nodes)
    report = checker.check_all(use_cache=use_cache, changed=changed)
    return checker, report
