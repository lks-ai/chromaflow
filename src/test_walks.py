import numpy as np
import matplotlib.pyplot as plt
from walk import (
    LorentzAttractorWalk,
    EllipticalWalk,
    LinearWalk,
    SplineWalk,
    RandomWalk,
    SinusoidalWalk,
    BezierWalk
)

def plot_walk(walk, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(walk[:, 0], walk[:, 1], walk[:, 2])
    ax.scatter(walk[:, 0], walk[:, 1], walk[:, 2], c='r', marker='o')
    ax.set_title(title)
    plt.show()

def main():
    num_frames = 100
    num_orbits = 3

    # Generate random cluster centers for testing
    np.random.seed(42)
    centers = np.random.rand(5, 3) * 2

    walk_generators = [
        (LorentzAttractorWalk(num_frames, num_orbits), "Lorentz Attractor Walk"),
        (EllipticalWalk(num_frames, num_orbits), "Elliptical Walk"),
        (LinearWalk(num_frames, num_orbits), "Linear Walk"),
        (SplineWalk(num_frames, num_orbits), "Spline Walk"),
        (RandomWalk(num_frames, num_orbits), "Random Walk"),
        (SinusoidalWalk(num_frames, num_orbits), "Sinusoidal Walk"),
        (BezierWalk(num_frames, num_orbits), "Bezier Walk")
    ]

    for generator, title in walk_generators:
        walk = generator.generate_walk(centers)
        plot_walk(walk, title)

if __name__ == "__main__":
    main()
