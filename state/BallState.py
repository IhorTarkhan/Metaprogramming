import math
import random

from state.ScreenState import ScreenState
from state.WallsState import WallsState


class BallState:
    def __init__(self, screen: ScreenState, walls: WallsState):
        ball_position_radius = random.random() * walls.radius * 0.8
        ball_position_angle = random.random() * 2 * math.pi
        self.x = int(screen.width / 2 + ball_position_radius * math.sin(ball_position_angle))
        self.y = int(screen.height / 2 + ball_position_radius * math.cos(ball_position_angle))
        self.velocity_x = 0
        self.velocity_y = 0

    def position(self):
        return self.x, self.y
