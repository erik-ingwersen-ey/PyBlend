from typing import List


class Engine:
    """
    Class representing an `Engine` or a Machine.

    The attributes indicate which rail the equipment is on, the yards to
    which it has access, and its working configuration, whether as a stacker,
    reclaimer, or both.

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
        """Instantiates a new `Engine` class.
        
        Parameters
        ----------
        id : int
            The Engine identifier.
        speed_stack : float
            The stacking speed of the engine. If this attribute is different from zero,
            then the equipment can perform the stacking function.
        speed_reclaim : float
            The reclaiming speed of the engine. If this attribute is different from zero,
            then the equipment can perform the reclaiming function.
        pos_ini : int
            The starting position of the engine.
        rail : int
            The rail to which the engine is attached.
        yards : List[int]
            List with the ore yards that the engine has access to.
        """

        self._id: int = id
        self._speed_stack: float = speed_stack
        self._speed_reclaim: float = speed_reclaim
        self._pos_ini: int = pos_ini
        self._rail: int = rail
        self._yards: List[int] = yards

    def __repr__(self: 'Engine') -> str:
        """Return a string representation of the Engine instance.

        This method provides a detailed string representation of the Engine
        object, including its unique identifier, speed stack, speed reclaim,
        initial position, rail, and yards. This is useful for debugging and
        logging purposes, allowing for a clear view of the Engine's current
        state.

        Returns:
            str: A formatted string containing the attributes of the
            Engine instance.
        """

        return (
            f'id: {self._id}\n'
            + f'speedStack: {self._speed_stack}\n'
            + f'speedReclaim: {self._speed_reclaim}\n'
            + f'posIni: {self._pos_ini}\n'
            + f'rail: {self._rail}\n'
            + f'yards: {self._yards}\n'
        )

    @property
    def id(self: 'Engine') -> int:
        """Retrieve the identifier of the Engine.

        This method returns the unique identifier associated with the Engine
        instance. The identifier is an integer that can be used to distinguish
        this Engine from others.

        Returns:
            int: The unique identifier of the Engine.
        """
        return self._id

    @id.setter
    def id(self: 'Engine', value: int) -> None:
        self._id = value

    @property
    def speed_stack(self: 'Engine') -> float:
        """Retrieve the stacking speed of the engine.

        This method returns the stacking speed of the engine, which indicates
        whether the equipment can perform the stacking function. If the stacking
        speed is different from zero, it implies that the engine is capable of
        executing the stacking operation.

        Returns:
            float: The stacking speed of the engine.
        """
        return self._speed_stack

    @speed_stack.setter
    def speed_stack(self: 'Engine', value: float) -> None:
        self._speed_stack = value

    @property
    def speed_reclaim(self: 'Engine') -> float:
        """Retrieve the reclaiming speed of the engine.

        This method returns the reclaiming speed of the engine, which indicates
        whether the equipment can perform the reclaiming function. If the speed
        is different from zero, it implies that the equipment is capable of
        performing the reclaiming operation.

        Returns:
            float: The reclaiming speed of the engine.
        """
        return self._speed_reclaim

    @speed_reclaim.setter
    def speed_reclaim(self: 'Engine', value: float) -> None:
        self._speed_reclaim = value

    @property
    def pos_ini(self: 'Engine') -> int:
        """Retrieve the starting position of the engine.

        This method returns the initial position of the engine, which is stored
        as an internal attribute. It is useful for understanding the engine's
        configuration and state at the beginning of its operation.

        Returns:
            int: The starting position of the engine.
        """
        return self._pos_ini

    @pos_ini.setter
    def pos_ini(self: 'Engine', value: int) -> None:
        self._pos_ini = value

    @property
    def rail(self: 'Engine') -> int:
        """Get the rail to which the engine is attached.

        This method retrieves the rail identifier associated with the engine
        instance. It accesses the private attribute `_rail` and returns its
        value, which represents the specific rail that the engine is connected
        to.

        Returns:
            int: The rail identifier of the engine.
        """
        return self._rail

    @rail.setter
    def rail(self: 'Engine', value: int) -> None:
        self._rail = value

    @property
    def yards(self: 'Engine') -> List[int]:
        """Retrieve the list of ore yards accessible by the engine.

        This method returns a list containing the ore yards that the engine has
        access to. The information is stored in a private attribute, ensuring
        encapsulation and maintaining the integrity of the data.

        Returns:
            List[int]: A list of integers representing the ore yards.
        """
        return self._yards

    @yards.setter
    def yards(self: 'Engine', value: List[int]) -> None:
        self._yards = value
