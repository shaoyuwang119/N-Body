import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#import Quad


class NBody:

    def __init__(self):
        self.mass = np.array([])
        self.pos = np.array([[]])
        self.vel = np.array([[]])

        self.scat = self.fig = self.ax = self.ani = None
        self.rng = np.random.default_rng()

        self.N = 0
        self.G = 1

    def init_plot(self):
        plt.style.use('dark_background')

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.scat = plt.scatter(self.pos[:, 0], self.pos[:, 1], s=np.log10(self.mass), c=self.rng.random((self.N, 3)))
        #self.scat = plt.scatter(self.pos[:, 0], self.pos[:, 1], s=1, c=self.rng.random((self.N, 3)))

        self.ax.set_axis_off()
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-128, 128)
        self.ax.set_ylim(-128, 128)

    def step(self, t):
        r = self.pos[None, :, :] - self.pos[:, None, :]  # 3d (N,N,2)

        dist2 = np.sum(r**2, axis=2) + 0.3 # 2d (N,N)
        inv_dist3 = dist2 ** (-1.5) # 2d (N,N)

        np.fill_diagonal(inv_dist3, 0)

        mprod = self.mass[:, None] * self.mass[None, :]
        f = self.G * (r * (mprod * inv_dist3)[:, :, None])
        net_f = np.sum(f, axis=1) # 2d (N,2)
        self.vel += net_f / self.mass[:, None] * t
        self.pos += self.vel * t

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

    def create_galaxy(self, n, rad, center, c_mass):
        self.create_body(c_mass, center, np.array([0,0]))

        for i in range(n - 1):
            mass = self.rng.uniform(low=200, high=400)

            r = self.rng.random() * rad
            theta = self.rng.uniform(0, 2 * np.pi)
            x, y = r * math.cos(theta), r * math.sin(theta)
            pos = np.array(center + [x,y])

            v = math.sqrt(self.G * c_mass / r)
            vel = np.array([math.cos(theta+np.pi/2),math.sin(theta+np.pi/2)]) * v

            self.create_body(mass, pos, vel)


    def main(self):
        self.init_plot()
        self.ani = FuncAnimation(plt.gcf(), func=self.update, frames=1000, interval=10, blit=True)
        plt.show()

nbody = NBody()
nbody.create_galaxy(100, 60, np.array([0,0]), 200000)
#nbody.create_galaxy(40, 30, np.array([0, 20]), 50000)
nbody.main()