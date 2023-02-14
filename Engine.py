import pygame as pg
from pygame.event import Event

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

        ball_state = self.state.ball
        direction = self.state.direction

        ball_state.x += ball_state.velocity_x
        ball_state.y += ball_state.velocity_y

        if direction == Direction.UP:
            ball_state.velocity_y -= 1
        elif direction == Direction.DOWN:
            ball_state.velocity_y += 1
        elif direction == Direction.LEFT:
            ball_state.velocity_x -= 1
        elif direction == Direction.RIGHT:
            ball_state.velocity_x += 1
