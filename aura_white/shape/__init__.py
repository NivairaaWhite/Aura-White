"""Aura Shape - our own conditional flow-matching Diffusion Transformer for 3-D shape completion.

Given the part of a shape that a single view can see (a truncated signed-distance volume + an "observed" mask) it
generates the rest.  Trainable from scratch on procedural shapes in minutes (the built-in dataset) or on your own mesh
collection (Objaverse, ...) with `aura-white shape-train`; used by `aura-white shape-complete` and importable as
`ShapeCompleter`.  It does NOT replace the large image-to-3D models - it is the piece that lets Aura White reason about
volume that no picture shows, with weights you own."""
from .sdf import mesh_to_sdf, sdf_to_mesh, observed_mask, random_shape
from .model import ShapeDiT, FlowMatching
from .complete import ShapeCompleter

__all__ = ["mesh_to_sdf", "sdf_to_mesh", "observed_mask", "random_shape", "ShapeDiT", "FlowMatching", "ShapeCompleter"]
