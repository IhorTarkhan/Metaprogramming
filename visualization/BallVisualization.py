import pygame as pg
from pygame.surface import Surface

from state.MainState import MainState


class BallVisualization:
    def __init__(self, screen: Surface):
        self.screen = screen

    def visualize(self, state: MainState):
        pg.draw.circle(self.screen, (0, 0, 255), state.ball.position(), 3)
