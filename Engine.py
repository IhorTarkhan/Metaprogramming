import pygame as pg
from pygame.event import Event
from sympy import Point

from state.Direction import Direction
from state.MainState import MainState
from visualization.MainVisualization import MainVisualization


class Engine:
    def __init__(self):
        self.state = MainState()
        self.visualization = MainVisualization(self.state)

    def run(self) -> None:
        pg.init()
        while True:
            events = pg.event.get()
            if any(e.type == pg.QUIT for e in events):
                break
            self.handle_frame(events)
            self.visualization.visualize()
            pg.display.update()
            pg.time.delay(300)
        pg.quit()

    def handle_frame(self, events: list[Event]):
        for event in events:
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_UP or event.key == pg.K_w:
                    self.state.direction = Direction.UP
                elif event.key == pg.K_DOWN or event.key == pg.K_s:
                    self.state.direction = Direction.DOWN
                elif event.key == pg.K_LEFT or event.key == pg.K_a:
                    self.state.direction = Direction.LEFT
                elif event.key == pg.K_RIGHT or event.key == pg.K_d:
                    self.state.direction = Direction.RIGHT

        predict_position, predict_velocity = self.free_fall_prediction()
        distance = Point(predict_position).distance(self.state.screen.center)
        print(distance)
        self.state.ball.x = predict_position[0]
        self.state.ball.y = predict_position[1]
        self.state.ball.velocity_x = predict_velocity[0]
        self.state.ball.velocity_y = predict_velocity[1]

    def free_fall_prediction(self):
        ball = self.state.ball
        direction = self.state.direction

        new_x = ball.x + ball.velocity_x
        new_y = ball.y + ball.velocity_y

        if direction == Direction.UP:
            new_velocity_x = ball.velocity_x
            new_velocity_y = ball.velocity_y - 1
        elif direction == Direction.DOWN:
            new_velocity_x = ball.velocity_x
            new_velocity_y = ball.velocity_y + 1
        elif direction == Direction.LEFT:
            new_velocity_x = ball.velocity_x - 1
            new_velocity_y = ball.velocity_y
        else:
            new_velocity_x = ball.velocity_x + 1
            new_velocity_y = ball.velocity_y
        return (new_x, new_y), (new_velocity_x, new_velocity_y)
