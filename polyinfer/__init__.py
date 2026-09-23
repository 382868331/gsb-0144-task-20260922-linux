"""polyinfer —— 带值限制的多态类型推断库（HM/Algorithm W）+ 开放行、
效应系统与递归模块检查。

公开接口::

    # v1（默认，与初版完全兼容）
    from polyinfer import infer, A
    result = infer(A.lambda_("x", A.var("x")))
    result.principal_type        # "a -> a"

    # v2（显式选项）：开放行记录 + R/W/X 效应 + evar
    result = infer(node, language="v2")

    # 模块系统：递归模块、签名两阶段检查、SCC、增量缓存
    from polyinfer.modules import ModuleChecker, check_modules
    checker, report = check_modules(modules)
    checker.invalidation_set("M")      # 精确缓存失效集
    checker.check_all(use_cache=False) # 非缓存检查路径

节点用 :mod:`polyinfer.ast` 里的构造器（``A`` 是该模块的别名）或同构 dict
构造；详见 README。
"""

from . import ast as A
from .errors import (
    EffectMismatchError,
    InfiniteTypeError,
    InvalidAstError,
    NodeLimitError,
    PolytypeError,
    RecordFieldError,
    SkolemEscapeError,
    TypeMismatchError,
    UnboundVariableError,
    format_path,
)
from .inference import (
    Constraint,
    EffVar,
    EffectTrace,
    Generalization,
    InferenceResult,
    TypeScheme,
    infer,
)
from .modules import (
    ExportEffectSig,
    ExportResult,
    ModuleChecker,
    ModuleReport,
    ModuleTrace,
    check_modules,
)
from .reference import (
    RefCheckReport,
    ReferenceCheckFailure,
    check_module_report,
    check_result,
)
from .signature import ParsedSignature, SignatureError, parse_signature
from .types import (
    BOOL,
    INT,
    UNIT,
    REmpty,
    RVar,
    TCon,
    TFun,
    TRec,
    TRef,
    TRow,
    TSkol,
    TVar,
    Type,
    canonical_names,
    render_type,
)

__all__ = [
    "A",
    "infer",
    # 模块系统
    "ModuleChecker",
    "check_modules",
    "ModuleReport",
    "ModuleTrace",
    "ExportResult",
    "ExportEffectSig",
    # 签名
    "parse_signature",
    "ParsedSignature",
    "SignatureError",
    # 结果/留痕
    "InferenceResult",
    "Constraint",
    "Generalization",
    "TypeScheme",
    "EffectTrace",
    "EffVar",
    "RefCheckReport",
    "check_result",
    "check_module_report",
    "ReferenceCheckFailure",
    # 错误
    "PolytypeError",
    "InvalidAstError",
    "NodeLimitError",
    "UnboundVariableError",
    "TypeMismatchError",
    "InfiniteTypeError",
    "RecordFieldError",
    "EffectMismatchError",
    "SkolemEscapeError",
    "format_path",
    # 类型
    "Type",
    "TVar",
    "RVar",
    "REmpty",
    "TCon",
    "TFun",
    "TRec",
    "TRow",
    "TRef",
    "TSkol",
    "INT",
    "BOOL",
    "UNIT",
    "render_type",
    "canonical_names",
]

__version__ = "2.0.0"
