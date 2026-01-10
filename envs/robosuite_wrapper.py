"""
Gymnasium-compatible wrapper for robosuite environments.

This wrapper converts robosuite's dict-based observations and API
to the standard Gymnasium interface expected by RL algorithms.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import robosuite as suite
from robosuite.controllers import load_composite_controller_config

class RobosuiteGymWrapper(gym.Env):
    """
    Wraps a robosuite environment to be compatible with Gymnasium.

    Key responsibilities:
    1. Convert dict observations to flat numpy arrays
    2. Define proper observation_space and action_space
    3. Handle reset() and step() return formats
    """

    def __init__(
        self,
        env_name: str = "Lift",
        robots: str = "Panda",
        # controller_type: str = "OSC_POSE",
        obs_keys: list = None,
        max_episode_steps: int = 500,
        reward_scale: float = 1.0,
        render_mode: str = None,
    ):
        """
        Initialize the wrapper.

        Args:
            env_name: robosuite task name ("Lift", "Stack", "PickPlace", etc.)
            robots: robot type ("Panda", "Sawyer", etc.)
            controller_type: controller to use ("OSC_POSE", "OSC_POSITION", etc.)
            obs_keys: list of observation keys to include in flattened obs
            max_episode_steps: horizon for the episode
            reward_scale: multiply rewards by this factor
        """
        super().__init__()

        # Default observation keys if not specified
        if obs_keys is None:
            obs_keys = [
                "robot0_eef_pos",
                "robot0_gripper_qpos",
                "gripper_to_cube_pos",
                "cube_pos",
            ]

        self.obs_keys = obs_keys
        self.reward_scale = reward_scale
        self.max_episode_steps = max_episode_steps
        self._step_count = 0

        self.render_mode = render_mode

        # controller_config = load_composite_controller_config(controller="BASIC")
        # controller_config["conf"]["arm"]["type"] = controller_type

        if self.render_mode:
            has_offscreen_renderer = True
            camera_names = "agentview"
            camera_heights = 256
            camera_widths = 256
        else:
            has_offscreen_renderer = False
            camera_names = None
            camera_heights = None
            camera_widths = None
        
        self.env = suite.make(
            env_name=env_name,
            robots=robots,
            # controller_config=controller_config,
            has_renderer=False,
            has_offscreen_renderer=has_offscreen_renderer,
            use_camera_obs=True if self.render_mode else False,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            reward_shaping=True,
        )
        sample_obs = self.env.reset()
        obs_dim = sum(sample_obs[key].flatten().shape[0] for key in self.obs_keys)

        # Define observation_space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Define action_space
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.env.action_dim,),
            dtype=np.float32,
        )

    def _get_obs(self, robosuite_obs: dict) -> np.ndarray:
        """
        Convert robosuite's dict observation to a flat numpy array.

        Args:
            robosuite_obs: dictionary of observations from robosuite

        Returns:
            Flattened numpy array containing selected observations
        """

        return np.concatenate([robosuite_obs[key].flatten() for key in self.obs_keys])

    def reset(self, seed=None, options=None):
        """
        Reset the environment.

        Returns:
            observation: flattened numpy array
            info: dictionary with additional information
        """
        super().reset(seed=seed)
        self._step_count = 0

        robosuite_obs = self.env.reset()
        obs = self._get_obs(robosuite_obs)

        if self.render_mode:
            self._last_frame = robosuite_obs["agentview_image"]

        return obs, {}

    def step(self, action: np.ndarray):
        """
        Take a step in the environment.

        Args:
            action: numpy array of shape (action_dim,)

        Returns:
            observation: flattened numpy array
            reward: scalar reward
            terminated: whether episode ended due to task completion/failure
            truncated: whether episode ended due to time limit
            info: dictionary with additional information
        """
        self._step_count += 1

        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Step the robosuite environment
        robosuite_obs, reward, done, info = self.env.step(action)

        if self.render_mode:
            self._last_frame = robosuite_obs["agentview_image"]

        # Convert observation
        obs = self._get_obs(robosuite_obs)

        # Handle termination vs truncation
        terminated = self._check_success()  # Task Success
        # Robosuite completed or Time limit
        truncated = (done or self._step_count >= self.max_episode_steps) and not terminated

        # Scale reward if needed
        reward = reward * self.reward_scale

        return obs, reward, terminated, truncated, info

    def _check_success(self) -> bool:
        """Check if the task was completed successfully."""

        return self.env._check_success()

    def render(self):
        """Render is not supported in headless mode."""
        if self.render_mode:
            return self._last_frame
        else:
            return None

    def close(self):
        """Clean up the environment."""
        if self.env is not None:
            self.env.close()
