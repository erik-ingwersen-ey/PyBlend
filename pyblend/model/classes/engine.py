from typing import List


class Engine:
    """This class represents an Engine or a Machine. The attributes indicate 
    which rail the equipment is on, the yards to which it has access and its
    working configuration, whether as a stacker, reclaimer or both.
    """

    def __init__(
        self: 'Engine', 
        id: int,
        speed_stack: float,
        speed_reclaim: float,
        pos_ini: int,
        rail: int,
        yards: List[int]
    ):
        """Instantiates a new Engine.
        
        Args:
            id (int): The Engine identifier.
            speed_stack (float): The stacking speed of the engine. If this 
                attribute is different from zero, then the equipment can 
                perform the stacking function.
            speed_reclaim (float): The reclaiming speed of the engine. If 
                this attribute is different from zero, then the equipment 
                can perform the reclaiming function.
            pos_ini (int): The starting position of the engine.
            rail (int): The rail to which the engine is attached.
            yards (List[int]): List with the ore yards that the engine has 
                access to.
        """

        self._id: int = id
        self._speed_stack: float = speed_stack
        self._speed_reclaim: float = speed_reclaim
        self._pos_ini: int = pos_ini
        self._rail: int = rail
        self._yards: List[int] = yards

    def __repr__(self: 'Engine') -> str:
        """This method returns the string representation of an Engine.
        
        Returns:
            str: The string representation of this class.
        """

        return f'id: {self._id}\n' + \
               f'speedStack: {self._speed_stack}\n' + \
               f'speedReclaim: {self._speed_reclaim}\n' + \
               f'posIni: {self._pos_ini}\n' + \
               f'rail: {self._rail}\n' + \
               f'yards: {self._yards}\n'

    @property
    def id(self: 'Engine') -> int:
        """int: The Engine identifier."""
        return self._id

    @id.setter
    def id(self: 'Engine', value: int) -> None:
        self._id = value

    @property
    def speed_stack(self: 'Engine') -> float:
        """float: The stacking speed of the engine. If this attribute is 
        different from zero, then the equipment can perform the stacking 
        function.
        """
        return self._speed_stack

    @speed_stack.setter
    def speed_stack(self: 'Engine', value: float) -> None:
        self._speed_stack = value

    @property
    def speed_reclaim(self: 'Engine') -> float:
        """float: The reclaiming speed of the engine. If this attribute is 
        different from zero, then the equipment can perform the reclaiming 
        function.
        """
        return self._speed_reclaim

    @speed_reclaim.setter
    def speed_reclaim(self: 'Engine', value: float) -> None:
        self._speed_reclaim = value

    @property
    def pos_ini(self: 'Engine') -> int:
        """int: The starting position of the engine."""
        return self._pos_ini

    @pos_ini.setter
    def pos_ini(self: 'Engine', value: int) -> None:
        self._pos_ini = value

    @property
    def rail(self: 'Engine') -> int:
        """int: The rail to which the engine is attached."""
        return self._rail

    @rail.setter
    def rail(self: 'Engine', value: int) -> None:
        self._rail = value

    @property
    def yards(self: 'Engine') -> List[int]:
        """List[int]: List with the ore yards that the engine has access to."""
        return self._yards

    @yards.setter
    def yards(self: 'Engine', value: List[int]) -> None:
        self._yards = value
