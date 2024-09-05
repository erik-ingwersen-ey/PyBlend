from typing import List

from pyblend.model.classes.quality import Quality


class Stockpile:
    """
    Ore Stockpile.

    The attributes indicate the stockpile position, the yard where it's located,
    the rails that have access to it, its ore capacity, initial weight,
    and quality parameters.
    """

    def __init__(
        self: 'Stockpile',
        id: int,
        position: int,
        yard: int,
        rails: List[int],
        capacity: float,
        weight_ini: float,
        quality_ini: List[Quality]
    ):
        """Instantiates a new `Stockpile`.

        Parameters
        ----------
        id : int
            The stockpile identifier.
        position : int
            The stockpile position.
        yard : int
            The yard where the stockpile is located.
        rails : List[int]
            List of rails that have access to the stockpile.
        capacity : float
            The stockpile ore capacity.
        weight_ini : float
            The stockpile initial weight.
        quality_ini : List[Quality]
            List of quality parameters presents in the stockpile.
        """
        self._id: int = id
        self._position: int = position
        self._yard: int = yard
        self._rails: List[int] = rails
        self._capacity: float = capacity
        self._weight_ini: float = weight_ini
        self._quality_ini: List[Quality] = quality_ini

    def __repr__(self: 'Stockpile') -> str:
        """Return a string representation of the `Stockpile` instance.

        This method provides a detailed string representation of the `Stockpile`
        class instance, including its attributes such as id, position, yard,
        rails, capacity, weightIni, and qualityIni. This representation is
        useful for debugging and logging purposes, allowing for a quick overview
        of the instance's state.

        Returns:
            str: The string representation of the `Stockpile` class instance.
        """

        return (
            f'id: {self._id}\n'
            + f'position: {self._position}\n'
            + f'yard: {self._yard}\n'
            + f'rails: {self._rails}\n'
            + f'capacity: {self._capacity}\n'
            + f'weightIni: {self._weight_ini}\n'
            + f'qualityIni: {self._quality_ini}\n'
        )

    @property
    def id(self: 'Stockpile') -> int:
        """int: The stockpile identifier."""
        return self._id

    @id.setter
    def id(self: 'Stockpile', value: int) -> None:
        self._id = value

    @property
    def position(self: 'Stockpile') -> int:
        """int: The stockpile position."""
        return self._position

    @position.setter
    def position(self: 'Stockpile', value: int) -> None:
        self._position = value

    @property
    def yard(self: 'Stockpile') -> int:
        """Get the yard where the stockpile is located.

        This method returns the yard attribute of the Stockpile instance. The
        yard represents a specific location associated with the stockpile.

        Returns:
            int: The yard number or identifier for the stockpile.
        """
        return self._yard

    @yard.setter
    def yard(self: 'Stockpile', value: int) -> None:
        self._yard = value

    @property
    def rails(self: 'Stockpile') -> List[int]:
        """Retrieve a list of rails that have access to the stockpile.

        This method returns the internal list of rails associated with the
        stockpile instance. The rails represent connections or pathways that can
        access the stockpile, and this method provides a way to obtain that
        information for further processing or analysis.

        Returns:
            List[int]: A list of integers representing the rails that have
            access to the stockpile.
        """
        return self._rails

    @rails.setter
    def rails(self: 'Stockpile', value: List[int]) -> None:
        self._rails = value

    @property
    def capacity(self: 'Stockpile') -> float:
        """Get the ore capacity of the stockpile.

        This method retrieves the current capacity of the stockpile, which
        represents the maximum amount of ore that can be held. The capacity is
        stored as a private attribute and can be accessed through this method.

        Returns:
            float: The ore capacity of the stockpile.
        """
        return self._capacity

    @capacity.setter
    def capacity(self: 'Stockpile', value: float) -> None:
        self._capacity = value

    @property
    def weight_ini(self: 'Stockpile') -> float:
        """Retrieve the initial weight of the stockpile.

        This method accesses the private attribute that stores the initial
        weight of the stockpile and returns its value. The initial weight is
        typically set during the initialization of the stockpile object and
        represents the starting weight before any modifications or operations
        are performed.

        Returns:
            float: The initial weight of the stockpile.
        """
        return self._weight_ini

    @weight_ini.setter
    def weight_ini(self: 'Stockpile', value: float) -> None:
        self._weight_ini = value

    @property
    def quality_ini(self: 'Stockpile') -> List[Quality]:
        """Retrieve the list of quality parameters present in the stockpile.

        This method accesses the internal attribute that stores the quality
        parameters associated with the stockpile. It returns a list of quality
        objects that represent the various quality metrics or characteristics
        defined for the stockpile.

        Returns:
            List[Quality]: A list containing the quality parameters of the stockpile.
        """
        return self._quality_ini

    @quality_ini.setter
    def quality_ini(self: 'Stockpile', value: List[Quality]) -> None:
        self._quality_ini = value
