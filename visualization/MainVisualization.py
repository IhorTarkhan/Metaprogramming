import pygame as pg

from state.MainState import MainState
from visualization.BallVisualization import BallVisualization
from visualization.DirectionArrowVisualization import DirectionArrowVisualization
from visualization.WallsVisualization import WallsVisualization


class MainVisualization:
    def __init__(self, state: MainState):
        self.state = state
        self.screen = pg.display.set_mode([self.state.screen.width, self.state.screen.height])
        self.walls_visualization = WallsVisualization(self.screen, self.state)
        self.ball_visualization = BallVisualization(self.screen)
        self.direction_arrow_visualization = DirectionArrowVisualization(self.screen)

    def visualize(self):
        self.screen.fill((255, 255, 255))
        self.walls_visualization.visualize()
        self.ball_visualization.visualize(self.state)
        self.direction_arrow_visualization.visualize(self.state)
