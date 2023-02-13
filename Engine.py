import pygame as pg
from pygame.event import Event

from state.Direction import Direction
from state.MainState import MainState
from visualization.BallVisualization import BallVisualization
from visualization.WallsVisualization import WallsVisualization


class Engine:
    def __init__(self):
        self.state = MainState()
        self.screen = pg.display.set_mode([self.state.screen.width, self.state.screen.height])
        self.walls_visualization = WallsVisualization(self.screen, self.state)
        self.ball_visualization = BallVisualization(self.screen)

    def handle_frame(self, events: list[Event]):
        for event in events:
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_UP or event.key == pg.K_w:
                    self.state.direction = Direction.UP
                if event.key == pg.K_DOWN or event.key == pg.K_s:
                    self.state.direction = Direction.DOWN
                if event.key == pg.K_LEFT or event.key == pg.K_a:
                    self.state.direction = Direction.LEFT
                if event.key == pg.K_RIGHT or event.key == pg.K_d:
                    self.state.direction = Direction.RIGHT

        self.update_state_per_frame()

        self.screen.fill((255, 255, 255))
        self.walls_visualization.visualize()
        self.ball_visualization.visualize(self.state)
        # pg.draw.polygon(self.screen, (0, 0, 0),
        #                 ((0, 100), (0, 200), (200, 200), (200, 300), (300, 150), (200, 0), (200, 100)))

    def update_state_per_frame(self):
        self.ball.atomic_move()

    def run(self) -> None:
        pg.init()
        while True:
            events = pg.event.get()
            if any(e.type == pg.QUIT for e in events):
                break
            self.handle_frame(events)
            pg.display.update()
            pg.time.delay(100)
        pg.quit()
