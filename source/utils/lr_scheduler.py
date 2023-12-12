# Experiment resources related to the MuLMS corpus (WIESP 2023).
# Copyright (c) 2023 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
This module contains scheduler functions for Pytorch training
loops.
"""


class WarmupSchedule:
    """Wrapper for LR schedule with warmup."""

    def __init__(self, warmup_steps: int) -> None:
        """
        Initializes the LR scheduler.

        Args:
            warmup_steps (int): Number of steps for linear warmup.
        """
        self.warmup_steps: int = warmup_steps

    def __call__(self, step: int) -> float:
        """
        Executes a step of the scheduler.

        Args:
            step (int): Step number

        Returns:
            float: Value used for scheduling
        """
        if step == 0:
            return 0
        elif 1 <= step <= self.warmup_steps:
            return step / self.warmup_steps
        else:
            return 1


class TriangularSchedule:
    """Wrapper for LR schedule with linear warmup and linear decay."""

    def __init__(self, warmup_steps: int, decay_steps: int) -> None:
        """
        Initializes the LR scheduler.

        Args:
            warmup_steps (int): Number of steps for linear warmup.
            decay_steps (int): Number of steps for decay.
        """
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps

    def __call__(self, step: int) -> float:
        """
        Executes a step of the scheduler.

        Args:
            step (int): Step number

        Returns:
            float: Value used for scheduling
        """
        if step == 0:
            return 0
        elif 1 <= step <= self.warmup_steps:
            return step / self.warmup_steps
        else:
            return max(0, 1 - (step - self.warmup_steps) / self.decay_steps)


class SqrtSchedule:
    """Wrapper for Noam LR schedule."""

    def __init__(self, warmup_steps):
        """
        Initializes the LR scheduler.

        Args:
            warmup_steps (int): Number of steps for linear warmup.
        """
        self.warmup_steps = warmup_steps
        self.sqrt_warmup_steps = warmup_steps**0.5
        self.inv_warmup_steps = warmup_steps ** (-1.5)

    def __call__(self, step: int) -> float:
        """
        Executes a step of the scheduler.

        Args:
            step (int): Step number

        Returns:
            float: Value used for scheduling
        """
        if step == 0:
            return 0
        else:
            return self.sqrt_warmup_steps * min(step ** (-0.5), step * self.inv_warmup_steps)
