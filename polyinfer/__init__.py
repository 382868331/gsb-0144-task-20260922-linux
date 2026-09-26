"""polyinfer —— 带值限制的多态类型推断（HM / Algorithm W）可复用库。

公开接口::

    from polyinfer import infer, A
    from polyinfer.errors import (
        PolytypeError, InvalidAstError, NodeLimitError,
        UnboundVariableError, TypeMismatchError, InfiniteTypeError,
        format_path,
    )

    result = infer(A.lambda_("x", A.var("x")))
    result.principal_type        # "a -> a"
    result.verify()             # 独立替换器核验（不经过统一核心）

节点用 :mod:`polyinfer.ast` 里的构造器（``A`` 是该模块的别名）或同构 dict
构造；详见 README。
"""

from . import ast as A
from .errors import (
    EscapedRigidVariableError,
    InfiniteTypeError,
    InvalidAstError,
    NodeLimitError,
    PolytypeError,
    SignatureMismatchError,
    TypeMismatchError,
    UnboundVariableError,
    ValueRestrictionConflictError,
    format_path,
)
from .inference import (
    Constraint,
    Generalization,
    InferenceResult,
    TypeScheme,
    infer,
)
from .reference import RefCheckReport, ReferenceCheckFailure, check_result
from .signature import ParsedSignature, build_signature_type, parse_signature
from .types import (
    BOOL,
    INT,
    UNIT,
    TCon,
    TFun,
    TRef,
    TVar,
    RigidTVar,
    Type,
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
    "RefCheckReport",
    "check_result",
    "ReferenceCheckFailure",
    "PolytypeError",
    "InvalidAstError",
    "NodeLimitError",
    "UnboundVariableError",
    "TypeMismatchError",
    "InfiniteTypeError",
    "SignatureMismatchError",
    "EscapedRigidVariableError",
    "ValueRestrictionConflictError",
    "format_path",
    "Type",
    "TVar",
    "RigidTVar",
    "TCon",
    "TFun",
    "TRef",
    "INT",
    "BOOL",
    "UNIT",
    "render_type",
    "canonical_names",
    "parse_signature",
    "build_signature_type",
    "ParsedSignature",
]

__version__ = "1.1.0"
