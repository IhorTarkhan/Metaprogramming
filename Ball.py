class Ball:
    def __init__(self,
                 x: int, y: int,
                 velocity_x: int = 0, velocity_y: int = 0,
                 acceleration_x: int = 0, acceleration_y: int = 0):
        self.x = x
        self.y = y
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.acceleration_x = acceleration_x
        self.acceleration_y = acceleration_y

    def atomic_move(self):
        self.x += self.velocity_x
        self.velocity_x += self.acceleration_x
        self.y += self.velocity_y
        self.velocity_y += self.acceleration_y

    def gravity_up(self):
        self.acceleration_x = 0
        self.acceleration_y = -1

    def gravity_down(self):
        self.acceleration_x = 0
        self.acceleration_y = 1

    def gravity_left(self):
        self.acceleration_x = -1
        self.acceleration_y = 0

    def gravity_right(self):
        self.acceleration_x = 1
        self.acceleration_y = 0

    def position(self):
        return self.x, self.y
