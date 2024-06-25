from sklearn.cluster import KMeans
import numpy as np
import torch
import random
from .walk import WalkGenerator

class ColorMotionGenerator:
    """Generates natural motion masks from images or video"""
    def __init__(self, input_image_tensor, walk_generator: WalkGenerator, num_clusters=5, random_state=42, max_iter=100, tol=1e-2, N_ratio=0.3):
        self.input_image_tensor = input_image_tensor # The Image batch tensor shape (batch_size, image_width, image_height, rgb_channels)
        self.walk_generator = walk_generator # The Generator to use (Lorentz, Elliptical, Linear, etc.)
        self.num_clusters = num_clusters # K-Means number of clusters: changes the walk space and affects smoothness of color interpolation
        self.random_state = random_state # K-Means random seed starting point (integer)
        self.max_iter = max_iter # K-Means Maximum number of Iterations
        self.tol = tol # K-Means tolerance: changes how tight clusters become
        self.N_ratio = N_ratio # The number of pixels to output

    def create_clusters(self, image_array):
        pixels = image_array.reshape(-1, 3)
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=self.random_state, max_iter=self.max_iter, tol=self.tol)
        kmeans.fit(pixels)
        return kmeans

    def generate_animation(self, kmeans, walk, image_array):
        width, height, _ = image_array.shape
        animation = []
        pixels = image_array.reshape(-1, 3)
        N = int(image_array.shape[0] * image_array.shape[1] * self.N_ratio)

        for point in walk:
            distances = np.linalg.norm(pixels - point, axis=1)
            nearest_indices = np.argsort(distances)[:N]
            greyscale_values = (1 - distances[nearest_indices] / np.max(distances[nearest_indices])) * 255
            
            frame_array = np.zeros((width, height, 3), dtype=np.uint8)
            for idx, val in zip(nearest_indices, greyscale_values):
                coord = np.unravel_index(idx, (width, height))
                frame_array[coord] = [val, val, val]

            animation.append(torch.tensor(frame_array, dtype=torch.float32) / 255.0)

        return torch.stack(animation)

    def generate(self):
        input_image_array = self.input_image_tensor.squeeze(0).numpy()
        kmeans = self.create_clusters(input_image_array)
        walk = self.walk_generator.generate_walk(kmeans.cluster_centers_)
        animation = self.generate_animation(kmeans, walk, input_image_array)
        return torch.clamp(animation, 0.0, 1.0)

# Example usage:
# input_image_tensor = <some_tensor>
# walk_generator = LorentzAttractorWalk(num_frames=64, num_orbits=3)
# generator = ColorMotionGenerator(input_image_tensor, walk_generator, num_clusters=5)
# animation = generator.generate()
