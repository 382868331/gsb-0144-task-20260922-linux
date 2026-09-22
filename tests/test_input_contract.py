"""输入契约与错误语义测试：定位性、bool 拒绝、NaN/Infinity、节点上限。"""

import math
import unittest

from polyinfer import A, infer
from polyinfer.errors import (
    InvalidAstError,
    NodeLimitError,
    UnboundVariableError,
    format_path,
)


class InputValidationTests(unittest.TestCase):
    def test_node_must_be_dict(self):
        with self.assertRaises(InvalidAstError) as cm:
            infer(42)
        self.assertEqual(cm.exception.path, ())

    def test_unknown_tag(self):
        with self.assertRaises(InvalidAstError) as cm:
            infer({"tag": "wat"})
        self.assertIn("wat", cm.exception.message)

    def test_missing_tag(self):
        with self.assertRaises(InvalidAstError):
            infer({"name": "x"})

    def test_missing_field_is_located(self):
        bad = {"tag": "lambda", "param": "x"}  # 缺 body
        with self.assertRaises(InvalidAstError) as cm:
            infer(bad)
        self.assertEqual(cm.exception.path, ())
        self.assertIn("body", cm.exception.message)

    def test_extra_field_rejected(self):
        with self.assertRaises(InvalidAstError) as cm:
            infer({"tag": "int", "value": 1, "note": "nope"})
        self.assertIn("note", cm.exception.message)

    def test_int_rejects_bool(self):
        with self.assertRaises(InvalidAstError) as cm:
            infer({"tag": "int", "value": True})
        self.assertIn("bool", cm.exception.message)

    def test_int_rejects_float_without_silent_truncation(self):
        with self.assertRaises(InvalidAstError):
            infer({"tag": "int", "value": 2.5})

    def test_int_rejects_nan_and_infinity(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidAstError):
                    infer({"tag": "int", "value": bad})

    def test_int_accepts_bigint(self):
        result = infer(A.int_(10 ** 40))
        self.assertEqual(result.principal_type, "Int")

    def test_bool_value_must_be_bool(self):
        with self.assertRaises(InvalidAstError):
            infer({"tag": "bool", "value": 1})

    def test_identifier_must_be_nonempty_str(self):
        with self.assertRaises(InvalidAstError):
            infer({"tag": "var", "name": ""})
        with self.assertRaises(InvalidAstError):
            infer({"tag": "var", "name": 3})

    def test_nested_error_path(self):
        # apply(fn=int, arg=...)：fn 本身合法；错误发生在推断阶段。
        # 这里测校验路径：if 的 cond 内的 var 名字类型错误。
        bad = A.if_({"tag": "var", "name": 0}, A.int_(1), A.int_(2))
        with self.assertRaises(InvalidAstError) as cm:
            infer(bad)
        self.assertEqual(cm.exception.path, ("cond",))

    def test_deeply_nested_invalid_path(self):
        bad = A.lambda_(
            "x",
            A.apply(A.var("x"), {"tag": "write", "ref": A.var("x")}),  # 缺 value
        )
        with self.assertRaises(InvalidAstError) as cm:
            infer(bad)
        self.assertEqual(cm.exception.path, ("body", "arg"))


class ScopingTests(unittest.TestCase):
    def test_unbound_variable_at_root(self):
        with self.assertRaises(UnboundVariableError) as cm:
            infer(A.var("oops"))
        self.assertEqual(cm.exception.path, ())
        self.assertEqual(cm.exception.name, "oops")

    def test_unbound_nested_path(self):
        program = A.lambda_("x", A.apply(A.var("y"), A.var("x")))
        with self.assertRaises(UnboundVariableError) as cm:
            infer(program)
        self.assertEqual(cm.exception.path, ("body", "fn"))

    def test_no_recursive_let(self):
        with self.assertRaises(UnboundVariableError) as cm:
            infer(A.let("f", A.var("f"), A.int_(1)))
        self.assertEqual(cm.exception.path, ("bound",))
        self.assertEqual(cm.exception.name, "f")

    def test_shadowing_is_fine(self):
        result = infer(
            A.let("x", A.int_(1), A.let("x", A.bool_(True), A.var("x")))
        )
        self.assertEqual(result.principal_type, "Bool")


class NodeLimitTests(unittest.TestCase):
    @staticmethod
    def _chain(applies):
        """``applies`` 个 apply 的左链，每节点 2 个子位，
        节点总数为 2*applies + 1（(((0 0) 0) ... 0)）。"""
        node = A.int_(0)
        for _ in range(applies):
            node = A.apply(node, A.int_(0))
        return node

    def test_at_limit_accepted_as_size(self):
        result = infer(A.lambda_("x", A.var("x")))  # lambda + var 共 2 节点
        self.assertEqual(result.node_count, 2)

    def test_over_default_limit_rejected_as_resource(self):
        node = self._chain(250)  # 2*250+1 = 501 节点
        with self.assertRaises(NodeLimitError) as cm:
            infer(node)
        self.assertEqual(cm.exception.limit, 500)
        self.assertGreaterEqual(cm.exception.count, 501)

    def test_limit_boundary_is_interface_parameter(self):
        node = self._chain(25)  # 2*25+1 = 51 节点
        with self.assertRaises(NodeLimitError):
            infer(node, max_nodes=50)
        # 放宽到 51 后越过资源检查，随后因 int 不可应用而类型失败——
        # 这是与资源拒绝不同的另一类错误。
        from polyinfer.errors import TypeMismatchError

        with self.assertRaises(TypeMismatchError):
            infer(node, max_nodes=51)

    def test_exactly_499_and_500_well_typed_nodes_accepted(self):
        # if 链每层 3 节点 + 末端 1：166 层恰好 499 节点。
        node = A.int_(0)
        for _ in range(166):
            node = A.if_(A.bool_(True), node, A.int_(0))
        r499 = infer(node)
        self.assertEqual((r499.node_count, r499.principal_type), (499, "Int"))
        r499.verify()

        # 499 层嵌套 lambda + 末端 int：恰好 500 节点的良构深程序。
        node = A.int_(0)
        for i in range(499):
            node = A.lambda_(f"p{i}", node)
        r500 = infer(node)
        self.assertEqual(r500.node_count, 500)
        r500.verify()

    def test_max_nodes_parameter_itself_validated(self):
        with self.assertRaises(TypeError):
            infer(A.int_(1), max_nodes=True)  # bool 不接受为 int
        with self.assertRaises(TypeError):
            infer(A.int_(1), max_nodes=10.0)
        with self.assertRaises(ValueError):
            infer(A.int_(1), max_nodes=0)

    def test_resource_limit_distinct_from_unprovable(self):
        """预算/资源拒绝必须与“证明无解”是不同异常类。"""
        self.assertTrue(issubclass(NodeLimitError, Exception))
        from polyinfer.errors import InfiniteTypeError, TypeMismatchError

        for unprovable in (TypeMismatchError, InfiniteTypeError):
            self.assertFalse(issubclass(NodeLimitError, unprovable))
            self.assertFalse(issubclass(unprovable, NodeLimitError))


class PathFormatTests(unittest.TestCase):
    def test_format_root_and_steps(self):
        self.assertEqual(format_path(()), "<root>")
        self.assertEqual(format_path(("body", "fn")), "<root>/body/fn")


if __name__ == "__main__":
    unittest.main()
