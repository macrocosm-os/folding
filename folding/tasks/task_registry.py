class TaskRegistry:
    """
    Handles the organization of all tasks that we want inside of SN25, which includes:
        - Molecular Dynamics (MD)
        - ML Inference

    It also attaches its corresponding reward pipelines.
    """

    def __init__(self):
        self.tasks = {
            "md_synthetic": None,
            "md_organic": None,
            "ml_synthetic": None,
            "ml_organic": None,
        }
