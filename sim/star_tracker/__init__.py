"""
Star tracker sensor simulation.

Simulates a star tracker camera mounted on a satellite, generating 256x256 pixel images
with visible satellites rendered as dots.
"""

from .star_tracker import render_star_tracker_image, render_star_tracker_sequence

__all__ = ['render_star_tracker_image', 'render_star_tracker_sequence']
