from typing import Any, Dict

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.utils import distance


class CustomReach1(Task): # Reach Task - Without any other object
    def __init__(
        self,
        sim,
        get_ee_position,
        reward_type="sparse",
        distance_threshold=0.05,
        goal_range=0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_sphere(
            body_name="target",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        return np.array([])

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        return ee_position

    def reset(self) -> None:
        self.goal = self._sample_goal()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        # goal = np.random.uniform(self.goal_range_low, self.goal_range_high)
        # print(goal)
        # goal = np.array([0.11546597, -0.07790424,  0.11437155])
        goal = np.array([0.1222784, -0.14232362, 0.11468627])
        return goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        dist_goal = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(dist_goal > self.distance_threshold, dtype=np.float32)
        else:
            return -dist_goal.astype(np.float32)
        

class CustomReach2(Task): # Reach Task - Penalty when touching Red Ball
    def __init__(
        self,
        sim,
        get_ee_position,
        reward_type="sparse",
        distance_threshold=0.05,
        goal_range=0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_sphere(
            body_name="target",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )
        self.sim.create_sphere(
            body_name="object",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.9, 0.1, 0.1, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        return np.array([])

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        return ee_position

    def reset(self) -> None:
        self.goal = self._sample_goal()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        self.object = self._sample_object()
        self.sim.set_base_pose("object", self.object, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        # goal = np.random.uniform(self.goal_range_low, self.goal_range_high)
        goal = np.array([0.1222784, -0.14232362, 0.11468627])
        return goal
    
    def _sample_object(self) -> np.ndarray:
        """Randomize goal."""
        # goal = np.random.uniform(self.goal_range_low, self.goal_range_high)
        # print(goal)
        goal = np.array([0.01464405, 0.06455681, 0.18082901])
        return goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        dist_goal = distance(achieved_goal, desired_goal)
        dist_obj = distance(achieved_goal, self.object)
        if self.reward_type == "sparse":
            if dist_obj < self.distance_threshold:
                return -np.array(dist_goal > self.distance_threshold, dtype=np.float32) - 0.5
            else:
                return -np.array(dist_goal > self.distance_threshold, dtype=np.float32)
        else:
            return -dist_goal.astype(np.float32)
        

class CustomReach3(Task): # Reach Task - Rewarding when touching Blue Ball
    def __init__(
        self,
        sim,
        get_ee_position,
        reward_type="sparse",
        distance_threshold=0.05,
        goal_range=0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_sphere(
            body_name="target",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )
        self.sim.create_sphere(
            body_name="object",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.1, 0.9, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        return np.array([])

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        return ee_position

    def reset(self) -> None:
        self.goal = self._sample_goal()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        self.object = self._sample_object()
        self.sim.set_base_pose("object", self.object, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        # goal = np.random.uniform(self.goal_range_low, self.goal_range_high)
        goal = np.array([0.1222784, -0.14232362, 0.11468627])
        return goal
    
    def _sample_object(self) -> np.ndarray:
        """Randomize goal."""
        # goal = np.random.uniform(self.goal_range_low, self.goal_range_high)
        # print(goal)
        goal = np.array([0.01464405, 0.06455681, 0.13082901])
        return goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        dist_goal = distance(achieved_goal, desired_goal)
        dist_obj = distance(achieved_goal, self.object)
        if self.reward_type == "sparse":
            if dist_obj < self.distance_threshold:
                return -np.array(dist_goal > self.distance_threshold, dtype=np.float32) + 0.5
            else:
                return -np.array(dist_goal > self.distance_threshold, dtype=np.float32)
        else:
            return -dist_goal.astype(np.float32)