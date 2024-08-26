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
        """String representation of an Input.

        This method returns a formatted string that includes the attributes of
        the Input class, such as id, weight, quality, and time. It is useful for
        debugging and logging purposes to quickly visualize the state of an
        Input object.

        Returns:
            str: The string representation of this class.
        """

        return (
            f'id: {self._id}\n'
            + f'weight: {self._weight}\n'
            + f'quality: {self._quality}\n'
            + f'time: {self._time}\n'
        )

    @property
    def id(self: 'Input') -> int:
        """Retrieve the input identifier.

        This method returns the unique identifier associated with the input
        instance. The identifier is stored as a private attribute and can be
        used to distinguish between different input objects.

        Returns:
            int: The unique identifier of the input instance.
        """
        return self._id

    @id.setter
    def id(self: 'Input', value: int) -> None:
        self._id = value

    @property
    def weight(self: 'Input') -> float:
        """Retrieve the weight of the input ore.

        This method returns the weight attribute of the input object. It is
        expected to be a floating-point number representing the weight of the
        ore.

        Returns:
            float: The weight of the input ore.
        """
        return self._weight

    @weight.setter
    def weight(self: 'Input', value: float) -> None:
        self._weight = value

    @property
    def quality(self: 'Input') -> List[Quality]:
        """Retrieve the quality parameters.

        This method returns a list of quality parameters associated with the
        instance. The quality parameters are stored in a private attribute and
        are intended to provide insights into the quality metrics of the input
        data.

        Returns:
            List[Quality]: A list containing the quality parameters.
        """
        return self._quality

    @quality.setter
    def quality(self: 'Input', value: List[Quality]) -> None:
        self._quality = value

    @property
    def time(self: 'Input') -> float:
        """Retrieve the time when the input is available.

        This method returns the time associated with the input instance. It
        accesses the private attribute `_time`, which stores the time value.
        This function is typically used to determine when the input was last
        updated or made available for processing.

        Returns:
            float: The time when the input is available.
        """
        return self._time

    @time.setter
    def time(self: 'Input', value: float) -> None:
        self._time = value
