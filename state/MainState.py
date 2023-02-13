from state.BallState import BallState
from state.Direction import Direction
from state.ScreenState import ScreenState
from state.WallsState import WallsState


class MainState:
    def __init__(self):
        self.direction: Direction = Direction.DOWN
        self.screen = ScreenState()
        self.walls = WallsState(self.screen)
        self.ball = BallState(self.screen, self.walls)
