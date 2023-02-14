import pygame

import pymunk
from pymunk import Vec2d

X, Y = 0, 1


def flipy(y):
    return -y + 600


def main():
    pygame.init()
    screen = pygame.display.set_mode((900, 600))
    clock = pygame.time.Clock()
    running = True

    space = pymunk.Space()
    space.gravity = 0.0, -900.0

    balls = []

    line_point1 = None
    static_lines = []
    run_physics = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                p = event.pos[X], flipy(event.pos[Y])
                body = pymunk.Body(10, 10)
                body.position = p
                shape = pymunk.Circle(body, 10)
                shape.friction = 0.5
                shape.collision_type = 2
                space.add(body, shape)
                balls.append(shape)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                if line_point1 is None:
                    line_point1 = Vec2d(event.pos[X], flipy(event.pos[Y]))
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 3:
                if line_point1 is not None:
                    line_point2 = Vec2d(event.pos[X], flipy(event.pos[Y]))
                    shape = pymunk.Segment(space.static_body, line_point1, line_point2, 0.0)
                    shape.friction = 0.99
                    space.add(shape)
                    static_lines.append(shape)
                    line_point1 = None

        if run_physics:
            dt = 1.0 / 60.0
            for x in range(1):
                space.step(dt)

        screen.fill(pygame.Color("white"))

        for ball in balls:
            r = ball.radius
            v = ball.body.position
            rot = ball.body.rotation_vector
            p = int(v.x), int(flipy(v.y))
            p2 = p + Vec2d(rot.x, -rot.y) * r * 0.9
            p2 = int(p2.x), int(p2.y)
            pygame.draw.circle(screen, pygame.Color("blue"), p, int(r), 2)
            pygame.draw.line(screen, pygame.Color("red"), p, p2)

        if line_point1 is not None:
            p1 = int(line_point1.x), int(flipy(line_point1.y))
            p2 = pygame.mouse.get_pos()
            pygame.draw.lines(screen, pygame.Color("black"), False, [p1, p2])

        for line in static_lines:
            body = line.body
            pv1 = body.position + line.a.rotated(body.angle)
            pv2 = body.position + line.b.rotated(body.angle)
            p1 = int(pv1.x), int(flipy(pv1.y))
            p2 = int(pv2.x), int(flipy(pv2.y))
            pygame.draw.lines(screen, pygame.Color("lightgray"), False, [p1, p2])

        pygame.display.flip()
        clock.tick(60)
        pygame.display.set_caption("fps: " + str(int(clock.get_fps())))


if __name__ == "__main__":
    main()
