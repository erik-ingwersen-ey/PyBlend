class Quality:
    """
    This class represents the ore quality parameter.

    The attributes indicate the quality parameter name and its percentage.
    """

    def __init__(self: 'Quality', parameter: str, value: float):
        """
        Instantiate a new `Quality` parameter.

        Parameters
        ----------
        parameter : str
            The quality parameter name.
        value : float
            The percentage of the quality parameter.
        """
        self._parameter: str = parameter
        self._value: float = value

    def __repr__(self: 'Quality') -> str:
        """
        String representation of the `Quality` parameter.
        
        Returns
        -------
        str
            The string representation of this class.
        """
        return (
            f"parameter: {self._parameter}\n"
            + f"value: {self._value}\n"
        )

    @property
    def parameter(self: 'Quality') -> str:
        """The quality parameter name."""
        return self._parameter

    @parameter.setter
    def parameter(self: 'Quality', value: str) -> None:
        self._parameter = value

    @property
    def value(self: 'Quality') -> float:
        """The percentage of the quality parameter."""
        return self._value

    @value.setter
    def value(self: 'Quality', value: float) -> None:
        self._value = value
