"""Blocks for filtering molecules."""

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Literal

from rdkit import Chem
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from cmxflow.operators.base import MoleculeBlock

logger = logging.getLogger(__name__)

Operator = Literal["<", ">", "<=", ">=", "==", "!="]

OPERATORS: dict[Operator, Callable] = {
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}

# Pattern for simple comparison: property op value OR value op property
# Matches: MW>200, logP<=5, 200<MW, 5>=logP
SIMPLE_PATTERN = re.compile(
    r"^\s*"
    r"([a-zA-Z_][a-zA-Z0-9_]*|[-+]?\d*\.?\d+)"  # property name or number
    r"\s*(<=|>=|==|!=|<|>)\s*"  # operator
    r"([a-zA-Z_][a-zA-Z0-9_]*|[-+]?\d*\.?\d+)"  # property name or number
    r"\s*$"
)

# Pattern for range expression: value op property op value
# Matches: 200<MW<500, 0<=logP<=5
RANGE_PATTERN = re.compile(
    r"^\s*"
    r"([-+]?\d*\.?\d+)"  # lower value
    r"\s*(<=?)\s*"  # lower operator (< or <=)
    r"([a-zA-Z_][a-zA-Z0-9_]*)"  # property name
    r"\s*(<=?)\s*"  # upper operator (< or <=)
    r"([-+]?\d*\.?\d+)"  # upper value
    r"\s*$"
)


class FilterExpressionError(ValueError):
    """Raised when a filter expression cannot be parsed."""

    pass


class SubstructureFilterError(ValueError):
    """Raised when substructure filter configuration is invalid."""

    pass


@dataclass
class FilterCondition:
    """A single filter condition.

    Attributes:
        property_name: Name of the molecule property to check.
        operator: Comparison operator.
        value: Value to compare against.
    """

    property_name: str
    operator: Operator
    value: float


def _flip_operator(op: str) -> Operator:
    """Flip a comparison operator for reversed expressions.

    Args:
        op: Original operator.

    Returns:
        Flipped operator (e.g., '<' becomes '>').
    """
    flip_map: dict[str, Operator] = {
        "<": ">",
        ">": "<",
        "<=": ">=",
        ">=": "<=",
        "==": "==",
        "!=": "!=",
    }
    return flip_map[op]


def _is_number(s: str) -> bool:
    """Check if a string represents a number.

    Args:
        s: String to check.

    Returns:
        True if the string can be parsed as a float.
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def _parse_simple_expression(expr: str) -> list[FilterCondition]:
    """Parse a simple comparison expression.

    Args:
        expr: Expression like 'MW>200' or '200<MW'.

    Returns:
        List containing a single FilterCondition.

    Raises:
        FilterExpressionError: If the expression is invalid.
    """
    match = SIMPLE_PATTERN.match(expr)
    if not match:
        raise FilterExpressionError(f"Invalid filter expression: '{expr}'")

    left, op, right = match.groups()
    left_is_num = _is_number(left)
    right_is_num = _is_number(right)

    if left_is_num and right_is_num:
        raise FilterExpressionError(
            f"Invalid filter expression: '{expr}' - both sides are numbers"
        )
    if not left_is_num and not right_is_num:
        raise FilterExpressionError(
            f"Invalid filter expression: '{expr}' - both sides are property names"
        )

    if left_is_num:
        # value op property -> flip to property flipped_op value
        return [
            FilterCondition(
                property_name=right,
                operator=_flip_operator(op),
                value=float(left),
            )
        ]
    else:
        # property op value
        return [
            FilterCondition(
                property_name=left,
                operator=op,  # type: ignore[arg-type]
                value=float(right),
            )
        ]


def _parse_range_expression(expr: str) -> list[FilterCondition]:
    """Parse a range expression.

    Args:
        expr: Expression like '200<MW<500'.

    Returns:
        List of two FilterConditions.

    Raises:
        FilterExpressionError: If the expression is invalid.
    """
    match = RANGE_PATTERN.match(expr)
    if not match:
        raise FilterExpressionError(f"Invalid range expression: '{expr}'")

    lower_val, lower_op, prop, upper_op, upper_val = match.groups()

    return [
        FilterCondition(
            property_name=prop,
            operator=_flip_operator(lower_op),
            value=float(lower_val),
        ),
        FilterCondition(
            property_name=prop,
            operator=upper_op,  # type: ignore[arg-type]
            value=float(upper_val),
        ),
    ]


def parse_filter_expression(expression: str) -> list[FilterCondition]:
    """Parse a filter expression into conditions.

    Supports:
    - Simple comparisons: MW>200, logP<=5
    - Reverse comparisons: 200<MW (flipped to MW>200)
    - Range expressions: 200<MW<500 (parsed into two conditions)
    - Multiple conditions: MW>200, logP>0 (AND logic)

    Args:
        expression: Filter expression string.

    Returns:
        List of FilterConditions (AND logic).

    Raises:
        FilterExpressionError: If any part of the expression is invalid.
    """
    if not expression or not expression.strip():
        return []

    conditions: list[FilterCondition] = []

    for part in expression.split(","):
        part = part.strip()
        if not part:
            continue

        # Try range pattern first (more specific)
        if RANGE_PATTERN.match(part):
            conditions.extend(_parse_range_expression(part))
        else:
            conditions.extend(_parse_simple_expression(part))

    return conditions


class PropertyFilterBlock(MoleculeBlock):
    """Block that filters molecules based on property conditions.

    Molecules are filtered based on conditions specified in the 'filters'
    input text. Conditions use AND logic - molecules must satisfy all
    conditions to pass through.

    Supported filter syntax:
    - Simple comparisons: MW>200, logP<=5
    - Reverse comparisons: 200<MW (equivalent to MW>200)
    - Range expressions: 200<MW<500
    - Multiple conditions: MW>200, logP>0

    Operators: <, >, <=, >=, ==, !=

    Example:
        workflow.add(PropertyFilterBlock())
        workflow.set_required_input({
            "1.text@filters": "200<MolWt<500, logP>0"
        })
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the PropertyFilterBlock."""
        super().__init__(name="PropertyFilter", input_text=["filters"])
        self.set_inputs(**kwargs)
        self._parsed_conditions: list[FilterCondition] | None = None

    def _get_conditions(self) -> list[FilterCondition]:
        """Get parsed conditions, parsing from input_text if needed.

        Returns:
            List of FilterConditions.

        Raises:
            FilterExpressionError: If the filter expression is invalid.
        """
        if self._parsed_conditions is None:
            expression = self.input_text.get("filters", "")
            self._parsed_conditions = parse_filter_expression(expression)
        return self._parsed_conditions

    def _get_property_value(self, mol: Chem.Mol, property_name: str) -> float | None:
        """Get a property value from a molecule.

        Args:
            mol: RDKit Mol object.
            property_name: Name of the property to retrieve.

        Returns:
            Property value as float, or None if not found or not numeric.
        """
        if not mol.HasProp(property_name):
            raise KeyError(f"Molecule missing property: {property_name}")

        try:
            # Try to get as double first (most common for numeric props)
            return float(mol.GetDoubleProp(property_name))
        except (KeyError, RuntimeError, ValueError):
            pass

        try:
            return float(mol.GetIntProp(property_name))
        except (KeyError, RuntimeError, ValueError):
            pass

        try:
            return float(mol.GetProp(property_name))
        except (KeyError, RuntimeError, ValueError):
            pass

        logger.debug(f"Could not convert property '{property_name}' to numeric value")
        return None

    def _evaluate_condition(
        self, mol: Chem.Mol, condition: FilterCondition
    ) -> bool | None:
        """Evaluate a single condition against a molecule.

        Args:
            mol: RDKit Mol object.
            condition: FilterCondition to evaluate.

        Returns:
            True if condition passes, False if fails, None if property missing.
        """
        value = self._get_property_value(mol, condition.property_name)
        if value is None:
            return None

        op_func = OPERATORS[condition.operator]
        return bool(op_func(value, condition.value))

    def _forward(self, mol: Chem.Mol) -> Chem.Mol | None:
        """Filter a molecule based on property conditions.

        Args:
            mol: Input RDKit Mol object.

        Returns:
            The molecule if all conditions pass, None otherwise.
        """
        conditions = self._get_conditions()

        # Empty filter passes all molecules
        if not conditions:
            return mol

        for condition in conditions:
            result = self._evaluate_condition(mol, condition)
            if result is None:
                # Missing property - filter out
                return None
            if not result:
                # Condition failed
                return None

        return mol

    def check_output(self, arg: Any) -> bool:
        """Validate that output is a valid molecule.

        Args:
            arg: Output to validate.

        Returns:
            True if the output is a valid molecule, False otherwise.
        """
        return isinstance(arg, Chem.Mol)


class SubstructureFilterBlock(MoleculeBlock):
    """Block that filters molecules based on substructure matches.

    Molecules can be filtered using SMARTS patterns and/or built-in RDKit
    filter catalogs (e.g., PAINS, BRENK, NIH, ZINC). The filter uses OR logic:
    a molecule is flagged if it matches any pattern or catalog.

    Inputs:
        query: Space-separated list of catalog names and/or SMARTS patterns.
            Catalog names (e.g., PAINS, BRENK, NIH, ZINC) are detected automatically.
            Everything else is treated as a SMARTS pattern.
        mode: "remove" (default) filters out matches, "keep" keeps only matches.

    Example:
        workflow.add(SubstructureFilterBlock())
        workflow.set_required_input({
            "1.text@query": "PAINS BRENK [OH]",
            "1.text@mode": "remove"
        })
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the SubstructureFilterBlock."""
        super().__init__(
            name="SubstructureFilter",
            input_text=["query", "mode"],
        )
        if "mode" not in kwargs:
            kwargs["mode"] = "remove"
        self.set_inputs(**kwargs)
        self._compiled_patterns: list[Chem.Mol] = []
        self._filter_catalog: FilterCatalog | None = None
        self._query_parsed: bool = False
        self._mode: str | None = None
        self._get_mode()

    def _get_mode(self) -> str:
        """Get the filter mode, parsing and caching if needed.

        Returns:
            Filter mode: "remove" or "keep".

        Raises:
            SubstructureFilterError: If mode is invalid.
        """
        if self._mode is None:
            mode = self.input_text.get("mode", "").strip().lower()
            if not mode:
                mode = "remove"
            if mode not in ("remove", "keep"):
                raise SubstructureFilterError(
                    f"Invalid mode: '{mode}'. Must be 'remove' or 'keep'."
                )
            self._mode = mode
        return self._mode

    def _parse_query(self) -> None:
        """Parse the query string into patterns and catalogs.

        Splits the query on whitespace. Tokens matching catalog names
        (case-insensitive) are loaded as catalogs; all others are compiled
        as SMARTS patterns.

        Raises:
            SubstructureFilterError: If a SMARTS pattern is invalid.
        """
        if self._query_parsed:
            return

        query_str = self.input_text.get("query", "").strip()
        if query_str:
            catalog_names: list[str] = []
            for token in query_str.split():
                if token.upper() in FilterCatalogParams.FilterCatalogs.names:
                    catalog_names.append(token.upper())
                else:
                    pattern = Chem.MolFromSmarts(token)
                    if pattern is None:
                        raise SubstructureFilterError(
                            f"Invalid SMARTS pattern: '{token}'"
                        )
                    self._compiled_patterns.append(pattern)

            if catalog_names:
                params = FilterCatalogParams()
                for name in catalog_names:
                    params.AddCatalog(FilterCatalogParams.FilterCatalogs.names[name])
                self._filter_catalog = FilterCatalog(params)

        self._query_parsed = True

    def _check_pattern_match(self, mol: Chem.Mol) -> bool:
        """Check if molecule matches any SMARTS pattern.

        Args:
            mol: RDKit Mol object.

        Returns:
            True if molecule matches any pattern, False otherwise.
        """
        self._parse_query()
        return any(mol.HasSubstructMatch(p) for p in self._compiled_patterns)

    def _check_catalog_matches(self, mol: Chem.Mol) -> bool:
        """Check if molecule matches any catalog filters.

        Args:
            mol: RDKit Mol object.

        Returns:
            True if molecule matches any catalog filter, False otherwise.
        """
        self._parse_query()
        if self._filter_catalog is None:
            return False
        return bool(self._filter_catalog.HasMatch(mol))

    def _forward(self, mol: Chem.Mol) -> Chem.Mol | None:
        """Filter a molecule based on substructure matches.

        Args:
            mol: Input RDKit Mol object.

        Returns:
            The molecule if it passes the filter, None otherwise.
        """
        mode = self._get_mode()
        has_match = self._check_pattern_match(mol) or self._check_catalog_matches(mol)

        if mode == "remove":
            return None if has_match else mol
        else:  # mode == "keep"
            return mol if has_match else None

    def check_output(self, arg: Any) -> bool:
        """Validate that output is a valid molecule.

        Args:
            arg: Output to validate.

        Returns:
            True if the output is a valid molecule, False otherwise.
        """
        return isinstance(arg, Chem.Mol)
