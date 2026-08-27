"""Public entry point for the spine proofreading workflow.

The notebook-facing API accepts the mesh first and the morphology second.
The existing validation implementation uses the reverse keyword order, so
this module provides the stable facade without duplicating the workflow.
"""

if __package__:
    from . import proofreading_interface
else:  # Support notebooks that add ``lib`` directly to ``sys.path``.
    import proofreading_interface


def run(mesh_path, morphology_path, validation_csv_path=None):
    """Launch proofreading for the supplied dataset.

    Parameters
    ----------
    mesh_path : path-like
        Path to the segmented mesh used for visualization.
    morphology_path : path-like
        Path to the morphology containing the spine annotations.
    validation_csv_path : path-like, optional
        Explicit validation output path. Real datasets otherwise use the
        legacy mesh-adjacent path; mock mode remains non-persistent by default.

    Returns
    -------
    None
        The interface is displayed directly; no widget is returned so a
        notebook's implicit result display cannot render it a second time.
    """
    return proofreading_interface.display_preview(
        morphology_path=morphology_path,
        mesh_path=mesh_path,
        validation_csv_path=validation_csv_path,
    )
