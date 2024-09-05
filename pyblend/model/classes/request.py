from pyblend.model.classes.quality import Quality


class Request(Quality):
    """
    Represents a Quality Request parameter.

    The attributes indicate the parameter name, its maximum and minium percentage,
    the goal percentage, and its importance on the request.
    """

    def __init__(
        self: 'Request', 
        parameter: str,
        minimum: float,
        maximum: float,
        goal: float,
        importance: int
    ):
        """Instantiates a new Quality Request.

        Parameters
        ----------
        parameter : str
            The quality parameter name.
        minimum : float
            The minimum percentage of the quality parameter.
        maximum : float
            The maximum percentage of the quality parameter.
        goal : float
            The goal percentage of the quality parameter.
        importance : int
            The importance of the quality parameter.
        """

        super().__init__(parameter, 0)
        self._minimum: float = minimum
        self._maximum: float = maximum
        self._goal: float = goal
        self._importance: int = importance

    def __repr__(self: 'Request') -> str:
        """String representation of a Quality Request parameter.

        This method provides a formatted string that includes the key attributes
        of the Quality Request parameter, such as the parameter name, minimum
        and maximum values, goal, and importance. It is useful for debugging and
        logging purposes, allowing for a quick overview of the object's state.

        Returns:
            str: The string representation of this class, detailing its attributes.
        """

        return (
            f'parameter: {self._parameter}\n'
            + f'minimum: {self._minimum}\n'
            + f'maximum: {self._maximum}\n'
            + f'goal: {self._goal}\n'
            + f'importance: {self._importance}\n'
        )

    @property
    def minimum(self: 'Request') -> float:
        """float: The minimum percentage of the quality parameter."""
        return self._minimum

    @minimum.setter
    def minimum(self: 'Request', value: float) -> None:
        self._minimum = value

    @property
    def maximum(self: 'Request') -> float:
        """float: The maximum percentage of the quality parameter."""
        return self._maximum

    @maximum.setter
    def maximum(self: 'Request', value: float) -> None:
        self._maximum = value

    @property
    def goal(self: 'Request') -> float:
        """float: The goal percentage of the quality parameter."""
        return self._goal

    @goal.setter
    def goal(self: 'Request', value: float) -> None:
        self._goal = value

    @property
    def importance(self: 'Request') -> int:
        """int: The importance of the quality parameter."""
        return self._importance

    @importance.setter
    def importance(self: 'Request', value: int) -> None:
        self._importance = value
