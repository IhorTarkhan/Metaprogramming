import math
import random

import pygame as pg

from Ball import Ball


class Visualization:
    def __init__(self):
        width = 16 * 60
        height = 9 * 60
        hole_angle = random.random() * 2 * math.pi
        hole_angle_length = math.pi / 10

        self.center = (width / 2, height / 2)
        self.radius = min(self.center) * 0.9
        self.circumscribed_rect = (width / 2 - self.radius, height / 2 - self.radius, 2 * self.radius, 2 * self.radius)
        self.hole_start_angle = hole_angle - hole_angle_length / 2
        self.hole_stop_angle = hole_angle + hole_angle_length / 2

        self.screen = pg.display.set_mode([width, height])
        ball_position_radius = random.random() * self.radius * 0.8
        ball_position_angle = random.random() * 2 * math.pi
        ball_x = int(width / 2 + ball_position_radius * math.sin(ball_position_angle))
        ball_y = int(height / 2 + ball_position_radius * math.cos(ball_position_angle))
        self.ball = Ball(ball_x, ball_y, acceleration_y=1)

    def update(self) -> bool:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return True
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_UP or event.key == pg.K_w:
                    self.ball.gravity_up()
                if event.key == pg.K_DOWN or event.key == pg.K_s:
                    self.ball.gravity_down()
                if event.key == pg.K_LEFT or event.key == pg.K_a:
                    self.ball.gravity_left()
                if event.key == pg.K_RIGHT or event.key == pg.K_d:
                    self.ball.gravity_right()

        self.ball.atomic_move()

        self.screen.fill((255, 255, 255))
        pg.draw.circle(self.screen, (0, 255, 0), self.center, self.radius, width=3)
        pg.draw.arc(self.screen, (0, 0, 0), self.circumscribed_rect, self.hole_start_angle, self.hole_stop_angle, 3)
        pg.draw.circle(self.screen, (0, 0, 255), self.ball.position(), 3)

    def run(self) -> None:
        pg.init()
        while True:
            pg.time.delay(100)
            if self.update():
                pg.quit()
                return
            pg.display.update()
