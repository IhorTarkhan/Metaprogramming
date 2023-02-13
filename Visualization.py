import pygame as pg


class Visualization:
    def __init__(self):
        self.screen = pg.display.set_mode([500, 500])
        self.position_x = 250
        self.position_y = 250

    def update(self) -> bool:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return True
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    print(1)

        self.screen.fill((255, 255, 255))
        self.position_x -= 1
        pg.draw.circle(self.screen, (0, 0, 255), (self.position_x, self.position_y), 75, width=4)

    def run(self):
        pg.init()
        while True:
            pg.time.delay(30)
            if self.update():
                pg.quit()
                break
            pg.display.update()
