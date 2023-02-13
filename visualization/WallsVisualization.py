import pygame as pg
from pygame import Surface

from state.MainState import MainState


class WallsVisualization:
    def __init__(self, screen: Surface, state: MainState):
        self.screen = screen
        self.screen_center = state.screen.center
        self.radius = state.walls.radius
        screen_width = state.screen.width
        screen_height = state.screen.height
        self.circumscribed_rect = (
            screen_width / 2 - self.radius, screen_height / 2 - self.radius, 2 * self.radius, 2 * self.radius)
        self.hole_start_angle = state.walls.hole_start_angle
        self.hole_stop_angle = state.walls.hole_stop_angle

    def visualize(self):
        pg.draw.circle(self.screen, (0, 255, 0), self.screen_center, self.radius, width=3)
        pg.draw.arc(self.screen, (0, 0, 0), self.circumscribed_rect, self.hole_start_angle, self.hole_stop_angle, 3)
