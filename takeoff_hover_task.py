import numpy as np
from physics_sim import PhysicsSim
from math import exp


class TakeoffHoverTask:
    """Task (environment) that defines the goal and provides feedback to the agent."""

    def __init__(self, init_height, target_height, runtime=10):
        """Initialize a Task object.
        Params
        ======
            init_height: initial height at which the quadcopter starts
            target_height: the goal height of quadcopter
            runtime: time limit for each episode
        """
        self.init_height = init_height
        # Simulation
        init_pose = np.array([0.0, 0.0, init_height, 0.0, 0.0, 0.0])
        init_velocities = np.array([0.0, 0.0, 0.0])
        init_angle_velocities = np.array([0.0, 0.0, 0.0])
        self.sim = PhysicsSim(
            init_pose, init_velocities, init_angle_velocities, runtime
        )

        self.state_size = len(self.create_state())
        self.action_low = 0
        self.action_high = 900
        self.action_size = 1

        # Goal
        self.target_height = target_height
        self.previous_height = init_height

    def get_reward(self, current_height):
        """Uses current pose of sim to return reward."""
        # height_diff = abs(current_height - self.previous_height)
        # reward = current_height + exp(-1 * height_diff) * 1000
        # self.previous_height = current_height
        # if current_height < self.init_height or current_height > self.target_height:
        #     return -reward * 10
        # return reward
        if (
            self.previous_height < current_height <= self.target_height
            or self.previous_height == self.target_height == current_height
        ):
            reward = 1
        else:
            reward = -1
        self.previous_height = current_height
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        done = self.sim.next_timestep(rotor_speeds * 4)
        return self.create_state(), self.get_reward(self.sim.pose[2]), done
        # reward = 0
        # pose_all = []
        # for _ in range(self.action_repeat):
        #     done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
        #     reward += self.get_reward()
        #     pose_all.append(self.sim.pose)
        # next_state = np.concatenate(pose_all)
        # return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = [self.create_state()]
        return state

    def create_state(self):
        return np.array([self.sim.pose[2], self.sim.v[2], self.sim.linear_accel[2]])
