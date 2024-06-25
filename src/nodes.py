"""ChromaFlow pack for ComfyUI"""

import os, sys
import numpy as np

# Comfy libs
def add_comfy_path():
    current_path = os.path.dirname(os.path.abspath(__file__))
    comfy_path = os.path.abspath(os.path.join(current_path, '../../../comfy'))
    if comfy_path not in sys.path:
        sys.path.insert(0, comfy_path)


add_comfy_path()

from comfy.utils import ProgressBar # type: ignore
import folder_paths # type: ignore

from .motion import ColorMotionGenerator
from .util import apply_gaussian_blur, increase_contrast, downscale_image
from .walk import (
    WalkGenerator,
    LorentzAttractorWalk,
    EllipticalWalk,
    LinearWalk,
    SplineWalk,
    RandomWalk,
    SinusoidalWalk,
    BezierWalk,
    WalkGeneratorFactory,
)

walk_types = ["elliptical", "linear", "spline", "random", "sinusoidal", "bezier"]
MAX_FP32 = np.iinfo(np.int32).max

class MotionMaskGenerator:
    """Takes an input image, outputs an animated mask based on a walk through clustered color space"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"forceInput": True}),
                "walk": ("WALK", {"forceInput": True}),
                "num_frames": ("INT", {"default": 16, "min": 1, "max": 1000000}),
                "orbits": ("INT", {"default": 1, "min": 1, "max": 1024}),
                "clusters": ("INT", {"default": 8, "min": 1, "max": 256}),
                "max_iter": ("INT", {"default": 100, "min": 1, "max": 1000}),
                "seed": ("INT", {"default": 42, "min": 0, "max": MAX_FP32}),
                "fill": ("FLOAT", {"default": 0.3, "min": 0.01, "max": 0.99, "step": 0.01}),
                "grow": ("INT", {"default": 32, "min": 0, "max": 1000}),
                "bw_thresh": ("FLOAT", {"default": 0.2, "min": 0.01, "max": 0.99, "step": 0.01}),
                "image_resize": ("INT", {"default": 512, "min": 128, "max": 8092, "step": 8}),
            },
        }

    # outputs: images_depth, images_
    RETURN_TYPES = ("IMAGE", "MASK", "MASK",)
    RETURN_NAMES = ("image_depth", "attn_mask", "invert_mask")
    FUNCTION = "generate"
    OUTPUT_NODE = True

    CATEGORY = "mask/motion"

    def generate(self, image, walk, num_frames, orbits, clusters, max_iter, seed, fill, grow, bw_thresh, image_resize):
        """Use the random walk data to walk through color space, returns controlnet animation and ipadapter attention mask"""
        # Make sure tensor is resized and video output shape is divisible by 2
        scaled = downscale_image(image, image_resize)

        np.random.seed(seed)
        # TODO Add phase offset to the walk generator
        walk_generator = WalkGeneratorFactory.create_walk_generator(walk['name'], num_frames, orbits, **walk['kwargs'])
        generator = ColorMotionGenerator(scaled, walk_generator, num_clusters=clusters, max_iter=max_iter, N_ratio=fill)
        animation = generator.generate()

        # Get the IPAdapter attention masks
        blurred = apply_gaussian_blur(animation, blur_amount=grow)
        contrasted = increase_contrast(blurred, threshold=bw_thresh)
        
        # Apply the Masks
        channels = ["red", "green", "blue", "alpha"]
        mask = contrasted[:, :, :, channels.index("red")]
        inverted = 1.0 - mask

        del walk_generator, generator, blurred, contrasted
        return (animation, mask, inverted)
    
class LorentzColorWalk:
    """Select walk settings"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigma": ("INT", {"default": 10, "min": 8, "max": 12}),
                "rho": ("INT", {"default": 28, "min": 20, "max": 40}),
                "beta": ("FLOAT", {"default": 8/3, "min": 2.0, "max": 4.0, "step": 0.01}),
                "dt": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 0.01, "step": 0.001}),
            },
        }

    RETURN_TYPES = ("WALK",)
    RETURN_NAMES = ("walk",)
    FUNCTION = "init"
    OUTPUT_NODE = True

    CATEGORY = "mask/motion"

    def init(self, sigma, rho, beta, dt):
        return ({"name": "lorentz", "kwargs": {
            "sigma": sigma,
            "rho": rho,
            "beta": beta,
            "dt": dt,
        }},)
        
class ColorWalk:
    """Select walk type"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "type": (walk_types, {}),
            },
        }

    RETURN_TYPES = ("WALK",)
    RETURN_NAMES = ("walk",)
    FUNCTION = "init"
    OUTPUT_NODE = True

    CATEGORY = "mask/motion"

    def init(self, type):
        return ({"name": type, "kwargs": {}},)
    

from .util import calculate_frame_count

class BPMConfig:
    """Configures the number of frames and orbits based on frame rate and BPM"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "BPM": ("FLOAT", {"default": 120.0, "min": 1.0, "max": 1024.0, "step": 0.01}),
                "num_beats": ("INT", {"default": 4, "min": 1, "max": 1024}),
                "FPS": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 1024.0, "step": 0.01}), 
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("num_frames", "orbits")
    FUNCTION = "config"
    OUTPUT_NODE = True

    CATEGORY = "mask/motion"

    def config(self, BPM, num_beats, FPS):
        frame_count = calculate_frame_count(BPM, FPS, num_beats)
        return (frame_count, num_beats)
    
