"""polyinfer —— 带值限制的多态类型推断（HM / Algorithm W）可复用库。

v1（缺省）公开接口::

    from polyinfer import infer, A
    result = infer(A.lambda_("x", A.var("x")))
    result.principal_type        # "a -> a"
    result.verify()              # 独立替换器核验（不经过统一核心）

v2（开放行记录、read/write/exn 效应、raise/try 异常、模块边界）::

    result = infer(node, version=2)     # 单表达式 v2
    result.principal_type, result.effects
    from polyinfer import ModuleChecker, check_package, ModuleCache
    pkg_result = check_package(package_node)

节点用 :mod:`polyinfer.ast` 构造器（``A`` 是该模块别名）或同构 dict 构造。
"""

from . import ast as A
from .errors import (
    DuplicateFieldError,
    EffectMismatchError,
    InfiniteTypeError,
    InvalidAstError,
    MissingFieldError,
    ModuleError,
    ModuleStructureError,
    NodeLimitError,
    PolytypeError,
    SignatureError,
    TypeMismatchError,
    UnboundVariableError,
    format_path,
)
from .inference import (
    Constraint,
    Generalization,
    InferenceResult,
    TypeScheme,
    infer,
)
from .modules import (
    ExportResult,
    ModuleCache,
    ModuleChecker,
    ModuleResult,
    PackageResult,
    check_package,
    parse_signature,
    tarjan_scc,
)
from .reference import RefCheckReport, ReferenceCheckFailure, check_result
from .types import (
    BOOL,
    EXN,
    INT,
    UNIT,
    ROW_EMPTY,
    KIND_EFF,
    KIND_ROW,
    KIND_TYPE,
    TCon,
    TEffect,
    TFun,
    TRecord,
    TRef,
    TRow,
    TVar,
    Type,
    RowEmpty,
    canonical_names,
    render_type,
)

__all__ = [
    "A",
    "infer",
    "InferenceResult",
    "Constraint",
    "Generalization",
    "TypeScheme",
    # 模块层
    "ModuleChecker", "ModuleCache", "check_package",
    "ModuleResult", "PackageResult", "ExportResult",
    "parse_signature", "tarjan_scc",
    # 参考检查
    "RefCheckReport",
    "check_result",
    "ReferenceCheckFailure",
    # 错误
    "PolytypeError",
    "InvalidAstError",
    "NodeLimitError",
    "UnboundVariableError",
    "TypeMismatchError",
    "InfiniteTypeError",
    "DuplicateFieldError",
    "MissingFieldError",
    "EffectMismatchError",
    "ModuleError",
    "ModuleStructureError",
    "SignatureError",
    "format_path",
    # 类型
    "Type",
    "TVar",
    "TCon",
    "TFun",
    "TRef",
    "TRecord",
    "TRow",
    "TEffect",
    "RowEmpty",
    "INT",
    "BOOL",
    "UNIT",
    "EXN",
    "ROW_EMPTY",
    "KIND_TYPE",
    "KIND_ROW",
    "KIND_EFF",
    "render_type",
    "canonical_names",
]

__version__ = "2.0.0"
