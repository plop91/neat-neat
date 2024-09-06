import cv2

import gym


def write_on_img(img, text, pos, font_scale=1, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255), line_type=cv2.LINE_4):
    """
    Write text on an image with a shadow effect
    """
    cv2.putText(img, text, (pos[0]+1, pos[1]+1), font, font_scale,
                (0, 0, 0), thickness, line_type)
    cv2.putText(img, text, pos, font, font_scale,
                color, thickness, line_type)
    return img


def write_neat_info_on_img(frame, generation, genome, fitness):
    frame = write_on_img(frame,
                         f"Generation:{generation}",
                         pos=(10, 50),
                         font=cv2.FONT_HERSHEY_TRIPLEX,
                         font_scale=.45,
                         color=(0, 150, 255),
                         thickness=1,
                         line_type=cv2.LINE_4)
    frame = write_on_img(frame,
                         f"Genome:{genome}",
                         pos=(10, 70),
                         font=cv2.FONT_HERSHEY_TRIPLEX,
                         font_scale=.45,
                         color=(0, 150, 255),
                         thickness=1,
                         line_type=cv2.LINE_4)
    frame = write_on_img(frame,
                         f"Fitness:{fitness}",
                         pos=(10, 90),
                         font=cv2.FONT_HERSHEY_TRIPLEX,
                         font_scale=.45,
                         color=(0, 150, 255),
                         thickness=1,
                         line_type=cv2.LINE_4)

    return frame


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