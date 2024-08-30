from typing import List

from pyblend.model.classes.quality import Quality


class Input:
    """Ore Input.

    The attributes indicate the weight of ore available in the input,
    the quality parameters of these ores and when the input is available.
    """

    def __init__(
        self: 'Input',
        id: int,
        weight: float,
        quality: List[Quality],
        time: float
    ):
        """Instantiates a new `Input` class.

        Parameters
        ----------
        id : int
            The input identifier.
        weight : float
            The input ore weight.
        quality : List[Quality]
            List with the quality parameters.
        time : float
            Time when input is available.
        """
        self._id: int = id
        self._weight: float = weight
        self._quality: List[Quality] = quality
        self._time: float = time

    def __repr__(self: 'Input') -> str:
        """Return a string representation of the Input instance.

        This method provides a detailed string representation of the Input
        object, including its id, weight, quality, and time attributes. It is
        useful for debugging and logging purposes to quickly understand the
        state of an Input object.

        Returns:
            str: A formatted string representing the Input instance.
        """
        return (
            f'id: {self._id}\n'
            + f'weight: {self._weight}\n'
            + f'quality: {self._quality}\n'
            + f'time: {self._time}\n'
        )

    @property
    def id(self: 'Input') -> int:
        """The input identifier."""
        return self._id

    @id.setter
    def id(self: 'Input', value: int) -> None:
        self._id = value

    @property
    def weight(self: 'Input') -> float:
        """The input ore weight."""
        return self._weight

    @weight.setter
    def weight(self: 'Input', value: float) -> None:
        self._weight = value

    @property
    def quality(self: 'Input') -> List[Quality]:
        """List with the quality parameters."""
        return self._quality

    @quality.setter
    def quality(self: 'Input', value: List[Quality]) -> None:
        self._quality = value

    @property
    def time(self: 'Input') -> float:
        """Time when input is available."""
        return self._time

    @time.setter
    def time(self: 'Input', value: float) -> None:
        self._time = value
