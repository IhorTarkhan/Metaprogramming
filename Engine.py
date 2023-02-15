import math
import random

import pygame
import pymunk
from pygame import Vector2 as VectorPG, Color
from pygame.event import Event
from pymunk import Vec2d as VectorPM

from Direction import Direction

g = 98.1


class Engine:
    def __init__(self):
        self.screen = pygame.display.set_mode((16 * 60, 9 * 60))
        self.clock = pygame.time.Clock()

        self.space = pymunk.Space()
        self.space.gravity = VectorPM(0.0, -g)

        self.ball = self.create_ball()
        self.walls = self.create_walls()
        self.direction = Direction.DOWN

    def to_pm(self, point: VectorPG) -> VectorPM:
        return VectorPM(point.x, self.screen.get_height() - point.y)

    def to_pg(self, point: VectorPM) -> VectorPG:
        return VectorPG(point.x, self.screen.get_height() - point.y)

    def create_ball(self) -> pymunk.Shape:
        center = VectorPG(self.screen.get_width() / 2, self.screen.get_height() / 2)
        rand_r = min(center) * 0.7 * random.random()
        rand_angle = 2 * math.pi * random.random()
        position = center + VectorPG(rand_r * math.sin(rand_angle), rand_r * math.cos(rand_angle))

        body = pymunk.Body(1, 1)
        body.position = self.to_pm(position)
        shape = pymunk.Circle(body, 10)
        shape.elasticity = 1
        self.space.add(body, shape)
        return shape

    def create_walls(self) -> list[pymunk.Segment]:
        walls = []

        center = VectorPG(self.screen.get_width() / 2, self.screen.get_height() / 2)
        rand_angle = 2 * math.pi * random.random()
        r = min(center) * 0.9
        n = 30
        min_angle = 2 * math.pi / n
        for i in range(n - 2):
            angle1 = min_angle * i + rand_angle
            angle_2 = min_angle * (i + 1) + rand_angle
            p1 = VectorPG(r * math.sin(angle1), r * math.cos(angle1)) + center
            p2 = VectorPG(r * math.sin(angle_2), r * math.cos(angle_2)) + center
            shape = pymunk.Segment(self.space.static_body, self.to_pm(p1), self.to_pm(p2), 0.0)
            shape.elasticity = 1
            self.space.add(shape)
            walls.append(shape)
        return walls

    def run(self) -> None:
        pygame.init()
        while True:
            events = pygame.event.get()
            if any(e.type == pygame.QUIT for e in events):
                break
            self.handle_frame(events)
            pygame.display.set_caption("FPS: " + str(int(self.clock.get_fps() * 10) / 10))
            pygame.display.update()
            self.space.step(1 / 60)
            self.clock.tick(60)
        pygame.quit()

    def handle_frame(self, events: list[Event]):
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP or event.key == pygame.K_w:
                    self.direction = Direction.UP
                    self.space.gravity = VectorPM(0, g)
                elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    self.direction = Direction.DOWN
                    self.space.gravity = VectorPM(0, -g)
                elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    self.direction = Direction.LEFT
                    self.space.gravity = VectorPM(-g, 0)
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    self.direction = Direction.RIGHT
                    self.space.gravity = VectorPM(g, 0)

        self.screen.fill(Color("WHITE"))
        pygame.draw.circle(self.screen, Color("BLUE"), self.to_pg(self.ball.body.position), self.ball.radius)
        for w in self.walls:
            pygame.draw.line(self.screen, Color("BLACK"), self.to_pg(w.a), self.to_pg(w.b))
        self.draw_direction_arrow()

    def draw_direction_arrow(self):
        points = [(0, -20), (15, 0), (5, 0), (5, 20), (-5, 20), (-5, 0), (-15, 0)]
        if self.direction == Direction.UP:
            rotate_angle = 0
        elif self.direction == Direction.DOWN:
            rotate_angle = math.pi
        elif self.direction == Direction.LEFT:
            rotate_angle = math.pi / 2
        elif self.direction == Direction.RIGHT:
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
            screen_width = self.screen.get_width()
            screen_height = self.screen.get_height()
            return screen_width / 2 + min(screen_width, screen_height) / 2 + 2.5 * x, screen_height - 100 + 2.5 * y

        pygame.draw.polygon(self.screen, (0, 0, 0), list(map(move, list(map(rotate, points)))))
