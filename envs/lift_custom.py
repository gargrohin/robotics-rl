"""
Custom Lift environment with modified reward function.

The default robosuite Lift reward has issues:
- Reaching reward dominates (~0.4 per step × 500 steps = 200)
- Grasp bonus is tiny (0.25 one-time)
- Success bonus is tiny (2.25 one-time)

Your task: Create a better reward function that encourages grasping and lifting.
"""

import numpy as np
from robosuite.environments.manipulation.lift import Lift


class LiftCustomReward(Lift):
    """
    Lift environment with custom reward shaping.

    This class inherits from robosuite's Lift environment.
    You only need to override the reward() method.
    """

    def __init__(
        self,
        reaching_weight: float = 0.1,
        reaching_coeff: float = 3.0,
        grasp_reward: float = 0.5,
        lift_reward: float = 1.0,
        success_reward: float = 100.0,
        time_penalty: float = 0.5,
        **kwargs
    ):
        self.reaching_weight = reaching_weight
        self.reaching_coeff = reaching_coeff
        self.grasp_reward = grasp_reward
        self.lift_reward = lift_reward
        self.success_reward = success_reward
        self.time_penalty = time_penalty

        # Ensure reward_shaping is True (we're doing our own shaping)
        kwargs['reward_shaping'] = True

        super().__init__(**kwargs)

        # Cache for table height (set in _reset_internal)
        self._table_height = None

    def _reset_internal(self):
        """Reset and cache table height."""
        super()._reset_internal()
        # Store table height for calculating lift progress
        self._table_height = self.model.mujoco_arena.table_offset[2]

    def reward(self, action=None):
        """
        Custom reward function - IMPLEMENT THIS!

        Available helper methods you can use:

        1. Check if task is complete (cube lifted 4cm above table):
           self._check_success()  -> returns bool

        2. Check if gripper is grasping the cube:
           self._check_grasp(
               gripper=self.robots[0].gripper,
               object_geoms=self.cube
           )  -> returns bool

        3. Get distance from gripper to cube:
           self._gripper_to_target(
               gripper=self.robots[0].gripper,
               target=self.cube.root_body,
               target_type="body",
               return_distance=True
           )  -> returns float (distance in meters)

        4. Get cube position (x, y, z):
           self.sim.data.body_xpos[self.cube_body_id]  -> np.array of shape (3,)

        5. Table height (cached in self._table_height after reset)

        Suggested reward components:
        - Reaching: reward for being close to cube (but scaled DOWN)
        - Grasping: reward for successfully grasping (scaled UP)
        - Lifting: reward based on cube height above table (NEW)
        - Success: large bonus for completing the task (scaled UP)

        Returns:
            float: reward value
        """
        reward = 0.0
        dist = self._gripper_to_target(
            gripper=self.robots[0].gripper,
            target=self.cube.root_body,
            target_type="body",
            return_distance=True
        )

        dist_reward = 1 - np.tanh(self.reaching_coeff * dist)
        dist_reward *= self.reaching_weight
        reward += dist_reward

        if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube):
            grasp_reward = self.grasp_reward
            reward += grasp_reward
            
            height_above_table = self.sim.data.body_xpos[self.cube_body_id][2] - self._table_height
            lift_reward = min(height_above_table, 1.0) * self.lift_reward
            reward += lift_reward

        if self._check_success():
            success_reward = self.success_reward
            reward += success_reward

        # Time penalty to encourage faster completion
        reward -= self.time_penalty

        return reward

