import math

import pygame as pg
from pygame import Surface

from state.Direction import Direction
from state.MainState import MainState


class DirectionArrowVisualization:
    def __init__(self, screen: Surface):
        self.screen = screen
        self.points = [(0, -20), (15, 0), (5, 0), (5, 20), (-5, 20), (-5, 0), (-15, 0)]

    def visualize(self, state: MainState):
        if state.direction == Direction.UP:
            rotate_angle = 0
        elif state.direction == Direction.DOWN:
            rotate_angle = math.pi
        elif state.direction == Direction.LEFT:
            rotate_angle = math.pi / 2
        elif state.direction == Direction.RIGHT:
            rotate_angle = 3 * math.pi / 2

        def rotate(point):
            x = point[0]
            y = point[1]
            new_x = x * math.cos(rotate_angle) + y * math.sin(rotate_angle)
            new_y = -x * math.sin(rotate_angle) + y * math.cos(rotate_angle)
            return new_x, new_y

        def move(point):
            x = point[0]
            y = point[1]
            return state.screen.width / 2 + state.walls.radius + 100 + 2.5 * x, state.screen.height - 100 + 2.5 * y

        pg.draw.polygon(self.screen, (0, 0, 0), list(map(move, list(map(rotate, self.points)))))
