import math
import random
import sys
from enum import Enum
from time import time
from typing import Optional

import gym
import numpy as np
import pygame
import pymunk
from gym import spaces
from gym.utils import seeding
from pygame import Vector2 as VectorPG, Color
from pymunk import Vec2d as VectorPM


class Direction(Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class Engine(gym.Env):
    def __init__(self):
        self.g = 98.1
        self.width = 16 * 60
        self.height = 9 * 60
        self.screen = None
        self.clock = pygame.time.Clock()

        self.center = VectorPG(self.width / 2, self.height / 2)
        self.r = min(self.center) * 0.9

        self.space = pymunk.Space()
        self.space.gravity = VectorPM(0.0, -self.g)

        self.center_hole = None
        self.ball = self.create_ball()
        self.walls = self.create_walls()
        self.direction = Direction.DOWN

        self.steps_count = 0

        self.action_space = spaces.Discrete(4)
        high = np.array([1, 1, 1, 1])
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def to_pm(self, point: VectorPG) -> VectorPM:
        return VectorPM(point.x, self.height - point.y)

    def to_pg(self, point: VectorPM) -> VectorPG:
        return VectorPG(point.x, self.screen.get_height() - point.y)

    def create_ball(self) -> pymunk.Shape:

        rand_r = min(self.center) * (0.95 * random.random())
        rand_angle = 2 * math.pi * random.random()
        position = self.center + VectorPG(rand_r * math.sin(rand_angle), rand_r * math.cos(rand_angle))

        body = pymunk.Body(1, 1)
        body.position = self.to_pm(position)
        shape = pymunk.Circle(body, 10)
        shape.elasticity = 1
        self.space.add(body, shape)
        return shape

    def create_walls(self) -> list[pymunk.Segment]:
        walls = []

        n = 30
        min_angle = 2 * math.pi / n
        rand_angle = math.pi / 2 + min_angle
        for i in range(n - 2):
            angle1 = min_angle * i + rand_angle
            angle_2 = min_angle * (i + 1) + rand_angle
            p1 = VectorPG(self.r * math.sin(angle1), self.r * math.cos(angle1)) + self.center
            p2 = VectorPG(self.r * math.sin(angle_2), self.r * math.cos(angle_2)) + self.center
            shape = pymunk.Segment(self.space.static_body, self.to_pm(p1), self.to_pm(p2), 8)
            shape.elasticity = 1
            self.space.add(shape)
            walls.append(shape)
        angle_c = min_angle * (n - 1) + rand_angle
        self.center_hole = VectorPG(self.r * math.sin(angle_c), self.r * math.cos(angle_c)) + self.center

        return walls

    def run(self) -> None:

        action = 1
        self.render()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP or event.key == pygame.K_w:
                        action = 0
                    elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                        action = 1
                    elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                        action = 2
                    elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                        action = 3
                    elif event.key == pygame.K_r:
                        self.reset()
            _, _, done, _ = self.step(action)
            self.render()
            if done:
                break

        pygame.quit()

    def step(self, action):
        self.steps_count += 1
        if action == 0:
            self.direction = Direction.UP
            self.space.gravity = VectorPM(0, self.g)
        elif action == 1:
            self.direction = Direction.DOWN
            self.space.gravity = VectorPM(0, -self.g)
        elif action == 2:
            self.direction = Direction.LEFT
            self.space.gravity = VectorPM(-self.g, 0)
        elif action == 3:
            self.direction = Direction.RIGHT
            self.space.gravity = VectorPM(self.g, 0)
        self.space.step(1 / 60)

        obs = self._get_observation()

        done = np.sqrt(
            (self.ball.body.position.x - self.center.x) ** 2 + (self.ball.body.position.y - self.center.y) ** 2) \
               > self.r
        if done:
            return obs, 1 - (self.steps_count / 5_000), done, {}
            # return obs, 1, done, {}
        if self.steps_count > 250000:
            return obs, -1, done, {}
        return obs, 0, done, {}

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.width, self.height)
            )
        self.screen.fill(Color("WHITE"))
        pygame.draw.circle(self.screen, Color("BLUE"), self.to_pg(self.ball.body.position), self.ball.radius)
        # pygame.draw.circle(self.screen, Color("RED"), self.to_pg(self.center_hole), self.ball.radius)
        for w in self.walls:
            pygame.draw.line(self.screen, Color("BLACK"), self.to_pg(w.a), self.to_pg(w.b))
        self.render_direction_arrow()
        pygame.display.set_caption("Frame: " + str(self.steps_count))
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                sys.exit(0)
        pygame.display.update()
        self.clock.tick(60)

    def render_direction_arrow(self):
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

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        self.steps_count = 0
        self.space.gravity = VectorPM(0.0, -self.g)
        self.ball.body.velocity = VectorPM(0, 0)

        rand_r = min(self.center) * 0.7 * random.random()
        rand_angle = 2 * math.pi * random.random()
        position = self.center + VectorPG(rand_r * math.sin(rand_angle), rand_r * math.cos(rand_angle))
        self.ball.body.position = self.to_pm(position)
        return self._get_observation()

    def _get_observation(self):
        return np.array([
            self.ball.body.position.x / self.width,
            self.ball.body.position.y / self.height,
            self.ball.body.velocity.x / 1000,
            self.ball.body.velocity.y / 1000
        ], dtype=np.float32)

    def seed(self, seed=int(time())):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
