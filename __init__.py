"""
@author: lks-ai
@title: ChromaFlow
@nickname: chromaflow
@description: Quickly make beat-matched animated masks and depth maps from any input image
"""

from .src.nodes import MotionMaskGenerator, LorentzColorWalk, ColorWalk, BPMConfig

NODE_CLASS_MAPPINGS = {
    "MotionMaskGenerator": MotionMaskGenerator,
    "LorentzColorWalk": LorentzColorWalk,
    "ColorWalk": ColorWalk,
    "BPMConfig": BPMConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MotionMaskGenerator": "ChromaFlow Motion Mask 🪃",
    "LorentzColorWalk": "Lorentz Walk 🪃",
    "ColorWalk": "Walk 🪃",
    "BPMConfig": "Configure from BPM 🪃",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
#WEB_DIRECTORY = "./web"