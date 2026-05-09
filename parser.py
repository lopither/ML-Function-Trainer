from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


class FunctionParseError(ValueError):
    """Raised when a user supplied function cannot be parsed safely."""


@dataclass(frozen=True)
class ParsedFunction:
    original_text: str
    expression: sp.Expr
    numpy_function: Callable[[np.ndarray], np.ndarray]


class SafeFunctionParser:
    """Parse a restricted mathematical expression of x with SymPy."""

    def __init__(self) -> None:
        self.x = sp.Symbol("x")
        self._allowed_functions = {
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "exp": sp.exp,
            "log": sp.log,
            "sqrt": sp.sqrt,
            "abs": sp.Abs,
            "Abs": sp.Abs,
        }
        self._allowed_constants = {
            "pi": sp.pi,
            "e": sp.E,
            "E": sp.E,
        }
        self._local_dict = {
            "x": self.x,
            **self._allowed_functions,
            **self._allowed_constants,
        }
        self._function_classes = {fn for fn in self._allowed_functions.values()}
        self._global_dict = {
            "__builtins__": {},
            "Add": sp.Add,
            "Mul": sp.Mul,
            "Pow": sp.Pow,
            "Integer": sp.Integer,
            "Float": sp.Float,
            "Rational": sp.Rational,
            "Symbol": sp.Symbol,
        }
        self._transformations = standard_transformations + (
            implicit_multiplication_application,
            convert_xor,
        )

    def parse(self, text: str) -> ParsedFunction:
        expression_text = (text or "").strip()
        if not expression_text:
            raise FunctionParseError("Enter a mathematical function of x.")

        try:
            expression = parse_expr(
                expression_text,
                local_dict=self._local_dict,
                global_dict=self._global_dict,
                transformations=self._transformations,
                evaluate=True,
            )
        except Exception as exc:
            raise FunctionParseError(f"Could not parse function: {exc}") from exc

        if not isinstance(expression, sp.Expr):
            raise FunctionParseError("The function must be a scalar expression of x.")

        unknown_symbols = expression.free_symbols - {self.x}
        if unknown_symbols:
            names = ", ".join(sorted(str(symbol) for symbol in unknown_symbols))
            raise FunctionParseError(f"Unknown symbol(s): {names}. Only x is allowed.")

        unsupported_functions = [
            str(function)
            for function in expression.atoms(sp.Function)
            if function.func not in self._function_classes
        ]
        if unsupported_functions:
            names = ", ".join(sorted(unsupported_functions))
            raise FunctionParseError(f"Unsupported function(s): {names}.")

        if expression.has(sp.zoo, sp.oo, -sp.oo, sp.nan):
            raise FunctionParseError("The expression is undefined or infinite.")

        try:
            numpy_function = sp.lambdify(
                self.x,
                expression,
                modules=[{"Abs": np.abs}, "numpy"],
            )
        except Exception as exc:
            raise FunctionParseError(f"Could not convert function to NumPy: {exc}") from exc

        return ParsedFunction(
            original_text=expression_text,
            expression=expression,
            numpy_function=numpy_function,
        )
