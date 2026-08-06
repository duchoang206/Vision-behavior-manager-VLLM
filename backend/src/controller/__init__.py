"""
Controller Package: Target Registry, Re-ID Matching & Rack Association.
"""

from src.controller.registry import TargetRegistry, target_registry
from src.controller.rack_association import RackAssociationEngine, rack_association_engine

__all__ = [
    "TargetRegistry",
    "target_registry",
    "RackAssociationEngine",
    "rack_association_engine"
]
