import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        obs, total_reward, done, trunk, info = None, 0.0, False, None, None
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', render_mode='rgb_array', apply_api_compatibility=True)
env = JoypadSpace(env, COMPLEX_MOVEMENT)
env = SkipFrame(env, skip=4)