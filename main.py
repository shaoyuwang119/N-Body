import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class NBody:
    G = 1
    mass = pos = vel = None
    rng = np.random.default_rng()
    scat = fig = ax = ani = None
    N = 0

    def __init__(self):
        self.mass = np.array([])
        self.pos = np.array([[]])
        self.vel = np.array([[]])

    def init_plot(self):
        plt.style.use('dark_background')

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.scat = plt.scatter(self.pos[:, 0], self.pos[:, 1], s=np.log10(self.mass), c=self.rng.random((self.N, 3)))
        #self.scat = plt.scatter(self.pos[:, 0], self.pos[:, 1], s=1, c=self.rng.random((self.N, 3)))

        self.ax.set_axis_off()
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-100, 100)
        self.ax.set_ylim(-100, 100)

    def step(self, t):
        for i in range(0, self.mass.size):
            net_force = np.zeros(2)
            m1 = self.mass[i]
            for j in range(0, self.mass.size):
                if i == j: continue
                m2 = self.mass[j]
                r = self.pos[j] - self.pos[i]
                r_mag = np.linalg.norm(r) + 0.5
                net_force += r * (self.G * m1 * m2 / r_mag ** 3)
            a = net_force / m1
            self.vel[i] += a * t
            self.pos[i] += self.vel[i] * t

    def update(self, frame):
        self.step(0.01)
        self.scat.set_offsets(self.pos)
        return [self.scat]

    def create_body(self, mass, pos, vel):
        self.N += 1
        pos = np.array([pos])
        vel = np.array([vel])
        self.mass = np.append(self.mass, mass)
        if self.pos.size == 0:
            self.pos = np.array(pos)
            self.vel = np.array(vel)
            return
        self.pos = np.append(self.pos, pos, axis=0)
        self.vel = np.append(self.vel, vel, axis=0)

    def create_galaxy(self, n, rad):
        center = 1000000
        self.create_body(center, np.array([0,0]), np.array([0,0]))

        for i in range(n - 1):
            mass = self.rng.uniform(low=200, high=500)

            r = self.rng.random() * rad
            theta = self.rng.uniform(0, 2 * np.pi)
            x, y = r * math.cos(theta), r * math.sin(theta)
            pos = np.array([x,y])

            vel = np.array([math.cos(theta+np.pi/2),math.sin(theta+np.pi/2)]) * math.sqrt(self.G * center / r)

            self.create_body(mass, pos, vel)


    def main(self):
        self.init_plot()
        self.ani = FuncAnimation(plt.gcf(), func=self.update, frames=1000, interval=1, blit=True)
        plt.show()

nbody = NBody()
nbody.create_galaxy(100, 30)
print(nbody.pos)
nbody.main()