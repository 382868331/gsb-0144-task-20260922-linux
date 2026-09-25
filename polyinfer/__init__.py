"""polyinfer —— 带值限制的多态类型推断（HM / Algorithm W）可复用库。

公开接口::

    from polyinfer import infer, diagnose, A
    from polyinfer.errors import (
        PolytypeError, InvalidAstError, NodeLimitError,
        UnboundVariableError, TypeMismatchError, InfiniteTypeError,
        format_path,
    )

    result = infer(A.lambda_("x", A.var("x")))
    result.principal_type        # "a -> a"
    result.verify()              # 独立替换器核验（不经过统一核心）

    diag = diagnose(ast)         # 诊断模式：失败时给按编号确定的极小冲突集
    diag.minimal, diag.conflict_indices()

节点用 :mod:`polyinfer.ast` 里的构造器（``A`` 是该模块的别名）或同构 dict
构造；详见 README。
"""

from . import ast as A
from .diagnose import (
    BindingHop,
    ConflictCore,
    ConflictEvidence,
    ConstraintInfo,
    DiagnosisResult,
    diagnose,
    format_diagnosis,
)
from .errors import (
    InfiniteTypeError,
    InvalidAstError,
    NodeLimitError,
    PolytypeError,
    TypeMismatchError,
    UnboundVariableError,
    format_path,
)
from .inference import (
    BindTrace,
    Constraint,
    Generalization,
    InferenceResult,
    InstantiationTrace,
    SchemeOrigin,
    TypeScheme,
    infer,
)
from .reference import RefCheckReport, ReferenceCheckFailure, check_result
from .types import (
    BOOL,
    INT,
    UNIT,
    TCon,
    TFun,
    TRef,
    TVar,
    Type,
    canonical_names,
    render_type,
)

__all__ = [
    "A",
    "infer",
    "diagnose",
    "format_diagnosis",
    "DiagnosisResult",
    "ConflictCore",
    "ConflictEvidence",
    "ConstraintInfo",
    "BindingHop",
    "InferenceResult",
    "Constraint",
    "Generalization",
    "TypeScheme",
    "SchemeOrigin",
    "BindTrace",
    "InstantiationTrace",
    "RefCheckReport",
    "check_result",
    "ReferenceCheckFailure",
    "PolytypeError",
    "InvalidAstError",
    "NodeLimitError",
    "UnboundVariableError",
    "TypeMismatchError",
    "InfiniteTypeError",
    "format_path",
    "Type",
    "TVar",
    "TCon",
    "TFun",
    "TRef",
    "INT",
    "BOOL",
    "UNIT",
    "render_type",
    "canonical_names",
]

__version__ = "1.1.0"
