"""Hybrid CV + AI inspection of 1 m x 1 m floor cells (Building view ``inspection`` rules).

Layers (plan §6): image_quality → illumination_normalizer → homography_rectifier
→ geometry_cv_inspector → ai_defect_inspector → arbitration_gate →
temporal_inspection_manager.  ``pipeline`` chains them for one frame,
``runtime`` runs them continuously and reports to the FMS.
"""
