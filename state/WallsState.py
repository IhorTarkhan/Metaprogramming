import math
import random

from state.ScreenState import ScreenState


class WallsState:
    def __init__(self, screen: ScreenState):
        self.radius = min(screen.center) * 0.9
        hole_angle = random.random() * 2 * math.pi
        hole_angle_length = math.pi / 10
        self.hole_start_angle = hole_angle - hole_angle_length / 2
        self.hole_stop_angle = hole_angle + hole_angle_length / 2
