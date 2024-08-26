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
        """String representation of a `Stockpile`.
        
        Returns
        -------
        str
            The string representation of the `Stockpile` class instance.
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
        """The yard where the stockpile is located."""
        return self._yard

    @yard.setter
    def yard(self: 'Stockpile', value: int) -> None:
        self._yard = value

    @property
    def rails(self: 'Stockpile') -> List[int]:
        """List of rails that have access to the stockpile."""
        return self._rails

    @rails.setter
    def rails(self: 'Stockpile', value: List[int]) -> None:
        self._rails = value

    @property
    def capacity(self: 'Stockpile') -> float:
        """The stockpile ore capacity."""
        return self._capacity

    @capacity.setter
    def capacity(self: 'Stockpile', value: float) -> None:
        self._capacity = value

    @property
    def weight_ini(self: 'Stockpile') -> float:
        """The stockpile initial weight."""
        return self._weight_ini

    @weight_ini.setter
    def weight_ini(self: 'Stockpile', value: float) -> None:
        self._weight_ini = value

    @property
    def quality_ini(self: 'Stockpile') -> List[Quality]:
        """List of quality parameters presents in the stockpile.
        """
        return self._quality_ini

    @quality_ini.setter
    def quality_ini(self: 'Stockpile', value: List[Quality]) -> None:
        self._quality_ini = value
