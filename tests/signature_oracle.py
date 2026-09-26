"""测试专用：rank-1 签名行为的独立小替换参照检查。

这个参照**不导入、不调用**被测统一器（``Inferencer.unify``/``bind``），
只消费 ``infer`` 成功结果里的最终替换与约束留痕，自行实现：

1. 替换应用（带循环检测）；
2. “多次使用各自实例化”的核对：对每个带签名的 let 泛化记录，
   确认其量化集合恰为存活刚性变量、不含任何普通推断变量，
   即该绑定确实可被任意实例化，而不是被某个单态变量冒充；
3. 刚性变量存活核对：扩展式绑定的记录里不得残留刚性变量
   （值限制），语法值绑定里每个声明刚性变量在最终替换下仍是刚性变量。
"""

from __future__ import annotations

from polyinfer.types import RigidTVar, TFun, TRef, TVar, Type


def oracle_apply(subst, tp, seen=None):
    if seen is None:
        seen = frozenset()
    if isinstance(tp, TVar):
        target = subst.get(tp)
        if target is None:
            return tp
        if tp.ident in seen:
            raise AssertionError(f"参照检查发现替换循环：{tp}")
        return oracle_apply(subst, target, seen | {tp.ident})
    if isinstance(tp, TFun):
        return TFun(oracle_apply(subst, tp.arg, seen), oracle_apply(subst, tp.ret, seen))
    if isinstance(tp, TRef):
        return TRef(oracle_apply(subst, tp.inner, seen))
    return tp


VALUE_TAGS = frozenset({"int", "bool", "lambda", "var"})


def check_signatures(result):
    """核对所有带签名 let 记录，返回 ((rigid_value_lets, expansive_lets), ...)。"""
    subst = dict(result.subst)
    value_signed = 0
    expansive_signed = 0
    for g in result.generalizations:
        if not getattr(g, "signed", False):
            continue
        # 签名绑定的普通推断泛化集合必须为空：方案只能由声明的刚性变量构成。
        non_rigid_generalized = [v for v in g.generalized if not isinstance(v, RigidTVar)]
        assert not non_rigid_generalized, (
            f"{g.name}: 签名绑定混入了 {len(non_rigid_generalized)} 个普通推断泛化变量"
        )
        # 泛化集合里绝不能混入普通推断变量（否则不是真正可任意实例化）。
        for v in g.generalized:
            assert not isinstance(v, RigidTVar) or v in set(getattr(g, "rigid_vars", ())), \
                f"{g.name}: 泛化集合混入非声明变量 {v}"
        surviving = set()
        for r in getattr(g, "rigid_vars", ()):
            resolved = oracle_apply(subst, r)
            if isinstance(resolved, RigidTVar):
                surviving.add(resolved)
            else:
                raise AssertionError(
                    f"{g.name}: 声明的刚性变量被最终替换解析成 {resolved}，方案不成立"
                )
        if g.bound_tag in VALUE_TAGS:
            # 单态签名（Int -> Int）本来就没有刚性变量，合法。
            declared = set(getattr(g, "rigid_vars", ()))
            if declared:
                assert surviving, f"{g.name}: 语法值签名绑定丢失了全部刚性变量"
            value_signed += 1
        else:
            assert not surviving, f"{g.name}: 扩展式绑定残留刚性变量，值限制被绕过"
            expansive_signed += 1
    return value_signed, expansive_signed
