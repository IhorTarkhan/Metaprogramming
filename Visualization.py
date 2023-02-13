import pygame as pg

from Ball import Ball


class Visualization:
    def __init__(self):
        self.screen = pg.display.set_mode([16 * 60, 9 * 60])
        self.ball = Ball(250, 250, acceleration_y=1)

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
        self.ball.draw(self.screen)

    def run(self) -> None:
        pg.init()
        while True:
            pg.time.delay(100)
            if self.update():
                pg.quit()
                return
            pg.display.update()
