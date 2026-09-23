"""模块层：显式签名、SCC 依赖检查与增量缓存（v2）。

检查流程
========

1. :class:`~polyinfer.ast.PackageValidator` 完成结构校验（含签名 AST、
   导入表、``mvar`` 可见性）；
2. 对模块导入图求强连通分量（Tarjan），按“被依赖者在前”的顺序：
   每个 SCC 的签名在结构阶段已齐备，随后逐模块检查实现；SCC 内模块
   经彼此的**声明签名**互相可见，故支持模块间递归（多态递归只经
   显式签名发生）；
3. 每个导出绑定独立推断，环境只含签名：导入导出经 ``mvar`` 按声明
   方案多态实例化，同模块更早导出按声明方案绑定；
4. 实现类型先做**边界泛化**（值形式才泛化、排除 Tofte 危险变量），
   再与签名做 prenex skolem 子包含：

   * 实现侧合法泛化变量 → 柔性 meta；
   * 实现侧未泛化的残留自由变量（典型：被引用写入污染的危险变量）
     → **刚性 weak skolem**；
   * 签名侧量化变量 → 刚性 skolem。

   刚性只能与柔性或自身统一，两个不同刚性相撞即拒绝；效应做定向
   包含 ``实现效应 ⊆ 声明效应``；引用按不变式双向严格一致；开放行
   字段按名对齐。由此：实现偷偷写入却声称纯、异常被处理但写效应仍在、
   携带可变引用的变量想在模块边界被泛化，都会被构造性拒绝。

缓存
====

:class:`ModuleCache` 以模块为粒度，键 = 本模块完整源码（签名+绑定+
导入表）指纹 ∪ 各依赖模块“已验证签名+效应”指纹。改动一个导出：

* 签名与效应不变 → 依赖模块键不变，继续命中（实现变化对外不可见）；
* 签名/效应变化 → 仅沿反向依赖闭包、键实际变化的模块重新检查。

一次检查失败**不替换**上一次有效结果（旧条目保留，异常上抛）。
:meth:`ModuleChecker.check_without_cache` 提供完全不走缓存的独立
全量路径，测试用它与增量结果逐项比对。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Tuple

from . import ast as ast_mod
from .errors import ModuleStructureError, SignatureError
from .inference import (
    Inferencer,
    TypeScheme,
    _is_value,
    _rename,
    dangerous_vars,
)
from .types import (
    BOOL,
    EXN,
    INT,
    KIND_ROW,
    KIND_TYPE,
    UNIT,
    ROW_EMPTY,
    RowEmpty,
    TCon,
    TFun,
    TRecord,
    TRef,
    TRow,
    TVar,
    Type,
    apply,
    canonical_names,
    free_tvars,
    render_type,
)

_TCONS = {"Int": INT, "Bool": BOOL, "Unit": UNIT, "Exn": EXN}


# ---------------------------------------------------------------- 签名解析

@dataclass(frozen=True)
class ParsedSignature:
    name: str
    quantified: frozenset
    type: Type
    fingerprint: str


def parse_signature(export_node: dict, ident_start: int = 0) -> Tuple[ParsedSignature, int]:
    sig = export_node["signature"]
    counter = ident_start
    tmap: Dict[str, TVar] = {}

    def fresh(kind: int) -> TVar:
        nonlocal counter
        v = TVar(counter, kind)
        counter += 1
        return v

    for n in sig["vars"]:
        tmap[n] = fresh(KIND_TYPE)
    for n in sig["row_vars"]:
        tmap[n] = fresh(KIND_ROW)

    def parse(texpr: dict) -> Type:
        tag = texpr["tag"]
        if tag == "tcon":
            return _TCONS[texpr["name"]]
        if tag in ("tvar", "rvar"):
            return tmap[texpr["name"]]
        if tag == "tfun":
            return TFun(parse(texpr["arg"]), parse(texpr["ret"]),
                        frozenset(texpr["effects"]))
        if tag == "tref":
            return TRef(parse(texpr["inner"]))
        if tag == "trecord":
            fields = tuple((p[0], parse(p[1])) for p in texpr["fields"])
            tail = ROW_EMPTY if texpr["tail"] is None else tmap[texpr["tail"]["name"]]
            return TRecord(fields, tail)
        raise ModuleStructureError(f"未知类型表达式 {tag!r}")

    tp = parse(sig["type"])
    names = canonical_names(tp, tuple(tmap.values()))
    return ParsedSignature(
        name=export_node["name"],
        quantified=frozenset(tmap.values()),
        type=tp,
        fingerprint=render_type(tp, names),
    ), counter


# ---------------------------------------------------------------- 检查结果

@dataclass(frozen=True)
class ExportResult:
    module: str
    name: str
    principal_type: str
    declared_type: str
    effects: FrozenSet[str]
    generalized: Tuple[str, ...]
    dangerous_excluded: Tuple[str, ...]
    node_count: int
    evidence: Any = None  # 供独立参考检查器审计的 InferenceResult 形留痕


@dataclass(frozen=True)
class ModuleResult:
    module: str
    exports: Tuple[ExportResult, ...]
    key: str


@dataclass(frozen=True)
class PackageResult:
    order: Tuple[str, ...]
    modules: Dict[str, ModuleResult]
    cache_hits: Tuple[str, ...]
    cache_misses: Tuple[str, ...]
    invalidated: Tuple[str, ...]

    def all_exports(self) -> Tuple[ExportResult, ...]:
        out: list[ExportResult] = []
        for m in self.order:
            out.extend(self.modules[m].exports)
        return tuple(out)


# ---------------------------------------------------------------- SCC

def tarjan_scc(graph: Dict[str, List[str]]) -> List[List[str]]:
    """Tarjan SCC；边 ``v -> w`` 表示 v 依赖 w。产出顺序汇点（被依赖）在前。"""
    index: Dict[str, int] = {}
    low: Dict[str, int] = {}
    stack: List[str] = []
    on_stack: set = set()
    counter = [0]
    sccs: List[List[str]] = []

    def strong(v: str) -> None:
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in graph.get(v, ()):
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
            sccs.append(comp)

    for v in sorted(graph):
        if v not in index:
            strong(v)
    return [sorted(comp) for comp in sccs]


# ---------------------------------------------------------------- skolem 子包含

class _SubFail(Exception):
    pass


@dataclass
class _Sub:
    """prenex 单向匹配状态：柔性可绑定，刚性只与柔性/自身相容。"""

    rigids: set
    flex: Dict[TVar, Type] = field(default_factory=dict)
    # 签名开放行 skolem -> 实现侧为其选定的多余字段见证（重复出现须一致）。
    row_witness: Dict[TVar, TRecord] = field(default_factory=dict)
    path: str = ""

    def bind_flex(self, v: TVar, tp: Type, where: str) -> None:
        if v in self.rigids:
            if isinstance(tp, TVar) and tp == v:
                return
            raise _SubFail(f"{where}: 刚性变量 {v} 不得填充为 {tp}")
        old = self.flex.get(v)
        if old is None:
            self.flex[v] = tp
        else:
            self.match(old, tp, where)

    def match(self, impl: Type, decl: Type, where: str) -> None:
        """实现类型 impl 满足声明类型 decl（协变视角；函数参数逆变）。"""
        if isinstance(impl, TVar):
            if isinstance(decl, TVar) and decl == impl:
                return
            if impl in self.rigids:
                if isinstance(decl, TVar) and decl not in self.rigids:
                    self.bind_flex(decl, impl, where)
                    return
                raise _SubFail(
                    f"{where}: 实现的刚性/未泛化变量 {impl} 不得与 {decl} 统一"
                    "（模块边界不得偷偷泛化携带可变引用的变量）"
                )
            self.bind_flex(impl, decl, where)
            return
        if isinstance(decl, TVar):
            if decl in self.rigids:
                raise _SubFail(f"{where}: 具体类型 {impl} 不得填入签名 skolem {decl}")
            self.bind_flex(decl, impl, where)
            return
        if isinstance(impl, TCon) and isinstance(decl, TCon):
            if impl.name != decl.name:
                raise _SubFail(f"{where}: {impl.name} ≠ {decl.name}")
            return
        if isinstance(impl, TRef) and isinstance(decl, TRef):
            # 引用不变式：双向严格一致（既不协变也不逆变）。
            self.match(impl.inner, decl.inner, where + ".Ref")
            self.match(decl.inner, impl.inner, where + ".Ref(反向)")
            return
        if isinstance(impl, TFun) and isinstance(decl, TFun):
            self.match(decl.arg, impl.arg, where + ".参数(逆变)")
            self.match(impl.ret, decl.ret, where + ".返回")
            self._check_effects(impl.eff, decl.eff, where)
            return
        if isinstance(impl, TRecord) and isinstance(decl, TRecord):
            self._match_records(impl, decl, where)
            return
        raise _SubFail(
            f"{where}: 外构造子不相容 {type(impl).__name__} vs {type(decl).__name__}"
        )

    @staticmethod
    def _check_effects(impl_eff: frozenset, decl_eff: frozenset, where: str) -> None:
        il = {x for x in impl_eff if isinstance(x, str)}
        iv = {x for x in impl_eff if not isinstance(x, str)}
        dv = {x for x in decl_eff if not isinstance(x, str)}
        if iv or dv:
            raise _SubFail(f"{where}: 效应在签名检查时必须已全部解析为具体标签")
        extra = il - set(decl_eff)
        if extra:
            raise _SubFail(
                f"{where}: 实现效应 {sorted(il)} 超出声明 {sorted(decl_eff)}"
                f"（未声明 {sorted(extra)}；try 只处理 exn，不消除 write/read）"
            )

    def _match_records(self, impl: TRecord, decl: TRecord, where: str) -> None:
        """记录的 prenex 匹配（HM 开放行，字段本身按不变式）：

        * 声明要求的每个字段实现必须提供，且字段类型双向严格一致；
        * 实现独有字段交给尾行：

          - 声明闭行：拒绝（实现多字段）；实现的开放尾行 flex 绑空行；
          - 声明开放（ρ 为刚性 skolem）：实现的开放尾行 flex σ 绑成
            ``{独有字段 | ρ}``（σ 是实现侧可自由选择的 meta，合法）；
            实现闭行时 ρ 的见证为该确定字段集。同一 ρ 多次出现时
            字段见证必须一致。
        """
        im = dict(impl.fields)
        dm = dict(decl.fields)
        for name in sorted(dm):
            if name not in im:
                raise _SubFail(f"{where}: 实现缺少声明字段 {name!r}")
            self.match(im[name], dm[name], f"{where}.字段 {name}")
            self.match(dm[name], im[name], f"{where}.字段 {name}(不变反向)")
        extra = sorted(set(im) - set(dm))

        it, dt = impl.tail, decl.tail
        decl_closed = isinstance(dt, RowEmpty)
        if decl_closed:
            if extra:
                raise _SubFail(
                    f"{where}: 签名为闭行，实现多出字段 {extra}"
                )
            if isinstance(it, TVar):
                # 实现尾行 flex：约束它不得再含字段（绑空闭行）。
                if it not in self.rigids:
                    self.bind_flex(it, TRow((), ROW_EMPTY), where + ".尾行")
            return

        # 声明开放行 skolem ρ。
        witness = TRecord(tuple((n, im[n]) for n in extra), ROW_EMPTY)
        if isinstance(it, RowEmpty):
            # 实现闭行：ρ 见证为确定的多余字段集。
            self._row_witness(dt, witness, where)
        elif isinstance(it, TVar):
            if it in self.rigids:
                # 实现尾行也是刚性（实现侧未泛化的行变量）：它必须
                # 就是 ρ，且两边字段集一致（不允许 flex 吸收）。
                if (it.ident, it.kind) != (dt.ident, dt.kind) and extra:
                    raise _SubFail(
                        f"{where}: 实现开放尾行 {it} 与签名行 skolem {dt} 不一致"
                        "（实现侧该行未在边界泛化，不能吸收进签名变量）"
                    )
                self._row_witness(dt, witness, where)
            else:
                # 实现尾行柔性：σ := {独有字段 | ρ}。
                self.bind_flex(it, TRow(tuple((n, im[n]) for n in extra), dt),
                               where + ".尾行")
                self._row_witness(dt, witness, where)

    def _row_witness(self, row_skolem: TVar, witness: TRecord, where: str) -> None:
        prev = self.row_witness.get(row_skolem)
        if prev is None:
            self.row_witness[row_skolem] = witness
            return
        if {n for n, _ in prev.fields} != {n for n, _ in witness.fields}:
            raise _SubFail(
                f"{where}: 开放行 skolem {row_skolem} 的字段见证不一致："
                f"{sorted(n for n, _ in prev.fields)} vs "
                f"{sorted(n for n, _ in witness.fields)}"
            )
        pd, wd = dict(prev.fields), dict(witness.fields)
        for n in sorted(pd):
            self.match(wd[n], pd[n], f"{where}.行见证 {n}")
            self.match(pd[n], wd[n], f"{where}.行见证 {n}(反向)")


def _collect_latent_effects(tp: Type) -> frozenset:
    """收集类型（含嵌套函数）中所有已烘焙的潜伏效应标签。"""
    out: set = set()

    def walk(t: Type) -> None:
        if isinstance(t, TFun):
            out.update(x for x in t.eff if isinstance(x, str))
            walk(t.arg)
            walk(t.ret)
        elif isinstance(t, TRef):
            walk(t.inner)
        elif isinstance(t, (TRecord, TRow)):
            for _n, ft in t.fields:
                walk(ft)

    walk(tp)
    return frozenset(out)


# ---------------------------------------------------------------- 缓存

def _fingerprint(node: Any) -> str:
    import json

    return json.dumps(node, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class _CacheEntry:
    key: str
    result: ModuleResult
    sig_fingerprint: str


class ModuleCache:
    def __init__(self) -> None:
        self._entries: Dict[str, _CacheEntry] = {}

    def get(self, module: str) -> _CacheEntry | None:
        return self._entries.get(module)

    def put(self, module: str, entry: _CacheEntry) -> None:
        self._entries[module] = entry

    def verified_signature(self, module: str) -> str | None:
        e = self._entries.get(module)
        return None if e is None else e.sig_fingerprint

    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._entries))


# ---------------------------------------------------------------- 检查器

class ModuleChecker:
    def __init__(
        self,
        package_node: dict,
        max_nodes: int = 5000,
        max_modules: int = 64,
        cache: ModuleCache | None = None,
    ) -> None:
        self.max_nodes = max_nodes
        validator = ast_mod.PackageValidator(max_nodes, max_modules)
        self.modules: Dict[str, dict] = validator.validate(package_node)
        self.graph: Dict[str, List[str]] = {
            name: list(mod["imports"]) for name, mod in self.modules.items()
        }
        self.reverse: Dict[str, List[str]] = {name: [] for name in self.modules}
        for name, deps in self.graph.items():
            for d in deps:
                self.reverse[d].append(name)
        self.cache = cache if cache is not None else ModuleCache()

    # ------------------------------------------------------ 键

    def _module_key(self, name: str) -> str:
        """模块缓存键：自身完整源码 + 各依赖模块的**声明签名**指纹。

        依赖键只取自包源码中的声明签名（含 effects/row_vars），与检查
        顺序、缓存状态无关——因此同一 SCC 内模块在任意一次检查里键都
        稳定。依赖的声明签名改变（或自身源码改变）才使键变化；
        依赖实现改变（签名不变）不影响本模块键。
        """
        mod = self.modules[name]
        parts = [f"self={_fingerprint(mod)}"]
        for dep in sorted(mod["imports"]):
            sigs = [
                f"{ex['name']}:{_fingerprint(ex['signature'])}"
                for ex in self.modules[dep]["exports"]
            ]
            parts.append(f"dep[{dep}]={';'.join(sigs)}")
        return "|".join(parts)

    # ------------------------------------------------------ 实现检查

    def _scc_of(self, name: str, plan: List[List[str]]) -> frozenset:
        for comp in plan:
            if name in comp:
                return frozenset(comp)
        return frozenset((name,))

    def _check_module(self, name: str, plan: List[List[str]]) -> Tuple[ExportResult, ...]:
        mod = self.modules[name]
        same_scc = self._scc_of(name, plan)
        env: Dict[Any, TypeScheme] = {}
        idc = 10_000

        # 导入（含同 SCC）：按声明签名方案可见。
        for dep in sorted(mod["imports"]):
            for ex in self.modules[dep]["exports"]:
                parsed, idc = parse_signature(ex, idc)
                env[("mvar", dep, ex["name"])] = TypeScheme(parsed.quantified, parsed.type)

        # 同 SCC 内未导入但互相递归引用？契约要求显式 import；mvar 校验
        # 已保证引用模块在 imports 中，故无需额外放行。

        results: List[ExportResult] = []
        local: Dict[str, TypeScheme] = {}
        for ex in mod["exports"]:
            parsed, idc = parse_signature(ex, idc)
            r = self._check_export(name, ex, parsed, env, local)
            results.append(r)
            local[ex["name"]] = TypeScheme(parsed.quantified, parsed.type)
        return tuple(results)

    def _check_export(self, module_name, ex, parsed, env, local) -> ExportResult:
        checker = Inferencer(version=2, start=1_000_000)
        full_env: Dict[Any, TypeScheme] = dict(env)
        full_env.update(local)
        subst, impl_raw, eff_raw = checker.ti(ex["binding"], full_env, {}, ("binding",))
        impl_t0 = apply(subst, impl_raw)
        impl_t = checker.bake_effects(impl_t0)
        root_labels, _ = checker.eff_resolve(eff_raw)
        # 导出绑定通常是函数：其效应是潜伏在函数类型里的标签，需要把
        # 顶层求值效应与全部潜伏效应标签合并，才是该导出的效应集合。
        latent_labels = _collect_latent_effects(impl_t)
        root_labels = frozenset(root_labels) | latent_labels

        # 边界泛化：与表达式 let 相同的值限制 + 危险变量规则。
        env_free: set = set()
        for scheme in full_env.values():
            env_free |= {
                v for v in free_tvars(apply(subst, scheme.t))
                if v not in scheme.quantified
            }
        free = [v for v in free_tvars(impl_t) if v not in env_free]
        # λ 参数豁免按替换后的等价类算（参数变量可能已被别名重定向）。
        param_reprs = {apply(subst, p) for p in checker.lambda_params}
        param_reprs = {v for v in param_reprs if isinstance(v, TVar)}
        dangerous = set(dangerous_vars(impl_t, param_reprs))
        is_value = _is_value(ex["binding"])
        generalized = [v for v in free if is_value and v not in dangerous]
        gen_set = set(generalized)
        residual = [v for v in free if v not in gen_set]  # 未泛化 → 刚性

        # 签名 skolem。
        sk_ren: Dict[TVar, TVar] = {}
        rigids: set = set(residual)
        nxt = [5_000_000]

        def rigid(v: TVar) -> TVar:
            if v not in sk_ren:
                w = TVar(nxt[0], v.kind)
                nxt[0] += 1
                sk_ren[v] = w
                rigids.add(w)
            return sk_ren[v]

        decl_t = _rename(parsed.type, {v: rigid(v) for v in free_tvars(parsed.type)})
        flex = {v: TVar(6_000_000 + i, v.kind) for i, v in enumerate(generalized)}
        impl_inst = _rename(impl_t, flex)
        # 残留自由变量同样重命名到刚性域（保持与 residual 集合同一对象）。
        resid_map = {}
        for v in residual:
            resid_map[v] = v  # 本身即刚性（已在 rigids）
        impl_inst = _rename(impl_inst, resid_map)

        sub = _Sub(rigids=rigids)
        try:
            sub.match(impl_inst, decl_t, f"{module_name}.{ex['name']}")
        except _SubFail as fail:
            raise SignatureError(module_name, ex["name"], str(fail)) from None

        names = canonical_names(impl_t)
        evidence = _Evidence(
            subst=subst,
            constraints=tuple(checker.constraints),
            generalizations=tuple(checker.generalizations),
            effect_log=tuple(checker.effect_log),
            eff_solution=checker.eff_full_solution(),
            version=2,
        )
        return ExportResult(
            module=module_name,
            name=ex["name"],
            principal_type=render_type(
                impl_t, names, checker.eff_solution_for(impl_t)
            ),
            declared_type=parsed.fingerprint,
            effects=frozenset(root_labels),
            generalized=tuple(sorted(names[v] for v in generalized if v in names)),
            dangerous_excluded=tuple(
                sorted(names[v] for v in dangerous if v in names and v in free)
            ),
            node_count=checker.nodes_seen,
            evidence=evidence,
        )

    # ------------------------------------------------------ 驱动

    def check(self) -> PackageResult:
        """增量检查：按 SCC 先签名后实现，模块粒度命中/失效缓存。"""
        plan = tarjan_scc(self.graph)
        hits: List[str] = []
        misses: List[str] = []
        invalidated: List[str] = []
        order: List[str] = []
        results: Dict[str, ModuleResult] = {}

        for comp in plan:
            # 阶段一：SCC 签名齐备（结构校验已保证，此处是显式语义检查点）。
            for m in comp:
                if not self.modules[m]["exports"]:
                    raise ModuleStructureError("SCC 内模块无导出", m)
            # 阶段二：实现（同 SCC 经签名互相可见）。
            for m in comp:
                order.append(m)
                key = self._module_key(m)
                cached = self.cache.get(m)
                if cached is not None and cached.key == key:
                    hits.append(m)
                    results[m] = cached.result
                    continue
                if cached is not None:
                    invalidated.append(m)
                misses.append(m)
                exports = self._check_module(m, plan)  # 失败则直接上抛
                entry = _CacheEntry(
                    key=key,
                    result=ModuleResult(m, exports, key),
                    sig_fingerprint=self._sig_fp(m, exports),
                )
                # 成功才写入：失败（上面已抛异常）不会到达这里，
                # 旧条目自然保留——“一次修改失败不替换上一次有效结果”。
                self.cache.put(m, entry)
                results[m] = entry.result
        return PackageResult(tuple(order), results, tuple(hits), tuple(misses),
                             tuple(invalidated))

    def _sig_fp(self, m: str, exports: Tuple[ExportResult, ...]) -> str:
        # 依赖取键只看“已验证的**声明**签名”（含声明效应）：依赖方只能
        # 经签名观察本模块；实现效应在签名范围内的变化（如由 write 变纯）
        # 不改变对外契约，不应额外让依赖失效。
        parts = []
        index = {e["name"]: e for e in self.modules[m]["exports"]}
        for exr in exports:
            parts.append(f"{exr.name}:{_fingerprint(index[exr.name]['signature'])}")
        return ";".join(parts)

    def check_without_cache(self) -> PackageResult:
        """全量不走缓存：用全新的一次性缓存跑同一条 SCC 路径。"""
        twin = ModuleChecker.__new__(ModuleChecker)
        twin.max_nodes = self.max_nodes
        twin.modules = self.modules
        twin.graph = self.graph
        twin.reverse = self.reverse
        twin.cache = ModuleCache()
        return twin.check()

    def reverse_dependency_closure(self, roots: List[str]) -> List[str]:
        """改动 roots 后理论上可能受影响的模块（反向依赖闭包）。"""
        seen: set = set()
        stack = list(roots)
        while stack:
            m = stack.pop()
            if m in seen:
                continue
            seen.add(m)
            stack.extend(self.reverse.get(m, ()))
        return sorted(seen)


@dataclass(frozen=True)
class _Evidence:
    """独立参考检查器需要的 InferenceResult 形字段（模块导出留痕）。"""

    subst: Any
    constraints: tuple
    generalizations: tuple
    effect_log: tuple
    eff_solution: Any
    version: int = 2


# ---------------------------------------------------------------- 便捷入口

def check_package(
    package_node: dict,
    max_nodes: int = 5000,
    max_modules: int = 64,
    cache: ModuleCache | None = None,
) -> PackageResult:
    """一次性检查整个模块包（可用外部缓存复用结果）。"""
    return ModuleChecker(package_node, max_nodes, max_modules, cache).check()
