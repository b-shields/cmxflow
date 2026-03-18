"""Parameter types for workflow optimization."""

from abc import ABC, abstractmethod
from typing import Any, Sequence


class Parameter(ABC):
    """Base class for all parameter types.

    Attributes:
        name: Name of the parameter.
    """

    def __init__(self, name: str, default: Any) -> None:
        """Initialize the parameter.

        Args:
            name: Name of the parameter.
            default: Default value for the parameter.
        """
        self.name = name
        self.set(default)

    def get(self) -> Any:
        """Get the current parameter value.

        Returns:
            Current value.
        """
        return self._value

    def set(self, value: Any) -> None:
        """Set the parameter value.

        Args:
            value: New value to set.

        Raises:
            ValueError: If the value fails validation via check().
        """
        if not self.check(value):
            raise ValueError(f"Value {value} is not allowed for {self.name}")
        self._value = value

    @property
    @abstractmethod
    def options(self) -> Any:
        """Return the allowed options/range for this parameter.

        Returns:
            For Continuous/Integer: tuple of (low, high) bounds.
            For Categorical: list of allowed values.
        """
        ...

    @abstractmethod
    def check(self, value: Any) -> bool:
        """Check if a value is within the specified constraints."""
        ...

    def __call__(self) -> dict[str, Any]:
        return {self.name: self._value}

    def __repr__(self) -> str:
        if isinstance(self._value, float):
            return f"{self._value:.6f}"
        return str(self._value)


class Continuous(Parameter):
    """Continuous parameter with float values in a range.

    Attributes:
        name: Name of the parameter.
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).
    """

    def __init__(self, name: str, default: float, low: float, high: float) -> None:
        """Initialize a continuous parameter.

        Args:
            name: Name of the parameter.
            default: Default parameter value.
            low: Lower bound (inclusive).
            high: Upper bound (inclusive).
        """
        self.low = low
        self.high = high
        super().__init__(name, default)

    @property
    def options(self) -> tuple[float, float]:
        """Return the (low, high) range.

        Returns:
            Tuple of (low, high) bounds.
        """
        return (self.low, self.high)

    def check(self, value: float) -> bool:
        """Check if a value is within the continuous range.

        Args:
            value: Value to validate.

        Returns:
            True if value is a number within [low, high], False otherwise.
        """
        if isinstance(value, (float, int)):
            return value >= self.low and value <= self.high
        return False


class Integer(Parameter):
    """Integer parameter with values in a range.

    Attributes:
        name: Name of the parameter.
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).
    """

    def __init__(self, name: str, default: int, low: int, high: int) -> None:
        """Initialize an integer parameter.

        Args:
            name: Name of the parameter.
            default: Default parameter value.
            low: Lower bound (inclusive).
            high: Upper bound (inclusive).
        """
        self.low = low
        self.high = high
        super().__init__(name, default)

    @property
    def options(self) -> tuple[int, int]:
        """Return the (low, high) range.

        Returns:
            Tuple of (low, high) bounds.
        """
        return (self.low, self.high)

    def check(self, value: int) -> bool:
        """Check if a value is within the integer range.

        Args:
            value: Value to validate.

        Returns:
            True if value is an integer within [low, high], False otherwise.
        """
        if isinstance(value, int):
            return value >= self.low and value <= self.high
        return False


class Categorical(Parameter):
    """Categorical parameter with discrete choices.

    Attributes:
        name: Name of the parameter.
        choices: Sequence of allowed values.
    """

    def __init__(self, name: str, default: Any, choices: Sequence[Any]) -> None:
        """Initialize a categorical parameter.

        Args:
            name: Name of the parameter.
            default: Default parameter value.
            choices: Sequence of allowed values.
        """
        self.choices = list(choices)
        super().__init__(name, default)

    @property
    def options(self) -> list[Any]:
        """Return the list of allowed choices.

        Returns:
            List of allowed values.
        """
        return self.choices

    def check(self, value: Any) -> bool:
        """Check if a value is one of the allowed choices.

        Args:
            value: Value to validate.

        Returns:
            True if value is in the choices list, False otherwise.
        """
        return value in self.choices
