import math
from typing import Any

import httpx


class Compass:
    def __init__(self, abbreviation_expansion_endpoint: str = "http://127.0.0.1:8000/abbreviation-expansion"):
        self.FULL_ALPHABET = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        # Synserver endpoint configuration
        self.abbreviation_expansion_endpoint = abbreviation_expansion_endpoint

        # Text accumulation state
        self.text_buffer = ""

        # Navigation state
        self.current_group: list[str] = list(self.FULL_ALPHABET)
        self.history_stack: list[list[str]] = []
        self.mapped_zones: dict[str, list[str]] = {"up": [], "right": [], "down": [], "left": []}

        # Initial calculation
        self._calculate_zones()

    def clear(self) -> None:
        """Resets the accumulated text state."""
        self.text_buffer = ""
        self.reset_compass()

    def reset_compass(self) -> None:
        """Resets the navigation state to the full alphabet."""
        self.current_group = list(self.FULL_ALPHABET)
        self.history_stack = []
        self._calculate_zones()

    def process_input(self, direction: str) -> None:
        """
        Handle directional input ('up', 'right', 'down', 'left').
        """
        if direction not in self.mapped_zones:
            return

        target_group = self.mapped_zones[direction]

        # 1. Validate: Is the zone empty?
        if not target_group:
            return

        # 2. CHECK: Is there only 1 item in the target?
        if len(target_group) == 1:
            self._type_character(target_group[0])
            self.reset_compass()
        else:
            # 3. Drill Down
            # Save current state for backspace
            self.history_stack.append(list(self.current_group))
            # Update current group
            self.current_group = target_group
            self._calculate_zones()

    def back(self) -> None:
        """
        Handle backspace/navigate up logic.
        """
        # If we are deep in the menu, go up one level
        if self.history_stack:
            self.current_group = self.history_stack.pop()
            self._calculate_zones()
        else:
            # If we are at top level, delete a character
            if self.text_buffer:
                self.text_buffer = self.text_buffer[:-1]

    async def expand_abbreviation(
        self,
        *,
        timeout_s: float = 10.0,
    ) -> Any:
        """
        Asynchronously POST to synserver abbreviation expansion endpoint.

        Equivalent to:

            curl -sS -X POST "http://127.0.0.1:8000/abbreviation-expansion" \
              -H "Content-Type: application/json" \
              -d '{"text": "SFUIS"}'

        Args:
            timeout_s: Request timeout in seconds.

        Returns:
            Parsed JSON response (dict/list/etc), depending on server response.
        """
        payload_text = self.text_buffer

        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(
                self.abbreviation_expansion_endpoint,
                json={"text": payload_text},
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            return resp.json()

    def _type_character(self, char: str) -> None:
        self.text_buffer += char

    def _calculate_zones(self) -> None:
        """
        Splits the current group into 4 zones (up, right, down, left).
        """
        self.mapped_zones = {"up": [], "right": [], "down": [], "left": []}

        total = len(self.current_group)
        if total == 0:
            return

        quarter = math.ceil(total / 4)
        start_index = 0

        # Slice for 'up'
        end_index = start_index + quarter
        self.mapped_zones["up"] = self.current_group[start_index:end_index]
        start_index = end_index

        # Slice for 'right'
        end_index = start_index + quarter
        self.mapped_zones["right"] = self.current_group[start_index:end_index]
        start_index = end_index

        # Slice for 'down'
        end_index = start_index + quarter
        self.mapped_zones["down"] = self.current_group[start_index:end_index]
        start_index = end_index

        # Slice for 'left' (remainder)
        self.mapped_zones["left"] = self.current_group[start_index:]

    @property
    def current_text(self) -> str:
        return self.text_buffer

    @property
    def zones(self) -> dict[str, list[str]]:
        return self.mapped_zones
