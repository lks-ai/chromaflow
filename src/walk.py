from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.special import comb

class WalkGenerator(ABC):
    def __init__(self, num_frames, num_orbits):
        self.num_frames = num_frames
        self.num_orbits = num_orbits

    @abstractmethod
    def generate_walk(self, centers):
        pass

class LorentzAttractorWalk(WalkGenerator):
    def __init__(self, num_frames, num_orbits, sigma=10, rho=28, beta=8/3, dt=0.01):
        super().__init__(num_frames, num_orbits)
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt

    def generate_walk(self, centers):
        dt = self.dt
        num_steps = self.num_frames * self.num_orbits
        x, y, z = 1.0, 1.0, 1.0

        walk = []
        for _ in range(num_steps):
            dx = self.sigma * (y - x) * dt
            dy = (x * (self.rho - z) - y) * dt
            dz = (x * y - self.beta * z) * dt
            x += dx
            y += dy
            z += dz
            walk.append([x, y, z])

        walk = np.array(walk)
        step_size = len(walk) // self.num_frames
        walk = walk[::step_size][:self.num_frames]

        walk_min = np.min(walk, axis=0)
        walk_max = np.max(walk, axis=0)
        walk = (walk - walk_min) / (walk_max - walk_min) * (np.max(centers, axis=0) - np.min(centers, axis=0)) + np.min(centers, axis=0)
        return walk

class EllipticalWalk(WalkGenerator):
    def generate_walk(self, centers):
        num_steps = self.num_frames * self.num_orbits
        if len(centers) < 2:
            raise ValueError("You must have at least 2 clusters for elliptical walks")
        center1, center2 = centers[0], centers[1]
        t = np.linspace(0, 2 * np.pi * self.num_orbits, num_steps)
        a = (center2 - center1) / 2
        b = np.array([a[1], -a[0], 0])
        walk = center1 + a * np.cos(t)[:, np.newaxis] + b * np.sin(t)[:, np.newaxis]
        
        step_size = len(walk) // self.num_frames
        return walk[::step_size][:self.num_frames]

class LinearWalk(WalkGenerator):
    def generate_walk(self, centers):
        num_steps = self.num_frames * self.num_orbits
        walk = []
        for i in range(len(centers)):
            start = centers[i]
            end = centers[(i + 1) % len(centers)]
            t = np.linspace(0, 1, num_steps // len(centers))
            segment = start + t[:, np.newaxis] * (end - start)
            walk.extend(segment)
        walk = np.array(walk)
        
        step_size = len(walk) // self.num_frames
        return walk[::step_size][:self.num_frames]

class SplineWalk(WalkGenerator):
    def generate_walk(self, centers):
        num_steps = self.num_frames * self.num_orbits
        centers = np.vstack([centers, centers[0]])  # Closing the loop
        tck, _ = splprep([centers[:, 0], centers[:, 1], centers[:, 2]], s=0, per=True)
        t = np.linspace(0, 1, num_steps)
        spline = splev(t, tck)
        walk = np.array(spline).T
        
        step_size = len(walk) // self.num_frames
        return walk[::step_size][:self.num_frames]

class RandomWalk(WalkGenerator):
    def generate_walk(self, centers):
        num_steps = self.num_frames * self.num_orbits
        walk = [centers[0]]
        for _ in range(num_steps):
            next_point = centers[np.random.randint(0, len(centers))]
            walk.append(next_point)
        walk = np.array(walk)
        
        step_size = len(walk) // self.num_frames
        return walk[::step_size][:self.num_frames]

class SinusoidalWalk(WalkGenerator):
    def generate_walk(self, centers):
        num_steps = self.num_frames * self.num_orbits
        t = np.linspace(0, 2 * np.pi * self.num_orbits, num_steps)
        a = (np.max(centers, axis=0) - np.min(centers, axis=0)) / 2
        b = np.array([a[1], -a[0], 0])
        phase_shift = np.pi / 4  # Adding a phase shift for differentiation
        walk = centers[0] + a * np.sin(t + phase_shift)[:, np.newaxis] + b * np.cos(t)[:, np.newaxis]
        
        step_size = len(walk) // self.num_frames
        return walk[::step_size][:self.num_frames]

class BezierWalk(WalkGenerator):
    def generate_walk(self, centers):
        num_steps = self.num_frames * self.num_orbits
        centers = np.vstack([centers, centers[0]])  # Closing the loop
        n = len(centers) - 1

        def bernstein_poly(i, n, t):
            return comb(n, i) * (t**(n - i)) * ((1 - t)**i)

        t = np.linspace(0, 1, num_steps)
        polynomial_array = np.array([bernstein_poly(i, n, t) for i in range(n + 1)])
        walk = np.dot(centers.T, polynomial_array).T
        
        step_size = len(walk) // self.num_frames
        return walk[::step_size][:self.num_frames]

class WalkGeneratorFactory:
    @staticmethod
    def create_walk_generator(walk_type, num_frames, num_orbits, **kwargs):
        if walk_type == "lorentz":
            return LorentzAttractorWalk(num_frames, num_orbits, **kwargs)
        elif walk_type == "elliptical":
            return EllipticalWalk(num_frames, num_orbits, **kwargs)
        elif walk_type == "linear":
            return LinearWalk(num_frames, num_orbits, **kwargs)
        elif walk_type == "spline":
            return SplineWalk(num_frames, num_orbits, **kwargs)
        elif walk_type == "random":
            return RandomWalk(num_frames, num_orbits, **kwargs)
        elif walk_type == "sinusoidal":
            return SinusoidalWalk(num_frames, num_orbits, **kwargs)
        elif walk_type == "bezier":
            return BezierWalk(num_frames, num_orbits, **kwargs)
        else:
            raise ValueError(f"Unknown walk type: {walk_type}")