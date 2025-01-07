import numpy as np
from datetime import datetime, timezone
import pandas as pd


def get_epistula_body(job: "Job") -> dict:
    """Obtain the body of information needed to upload or update a job to the GJP server.

    Args:
        job (Job): a Job object containing the job details.

    Returns:
        body (dict): The body of information needed to upload or update a job to the GJP server.
    """

    body = job.to_dict()
    body["pdb_id"] = body.pop("pdb")
    body["system_config"] = {
        "ff": body.pop("ff"),
        "box": body.pop("box"),
        "water": body.pop("water"),
        "system_kwargs": body.pop("system_kwargs"),
    }
    body["em_s3_link"] = body.get("em_s3_link", "s3://path/to/em")
    body["priority"] = body.get("priority", 1)
    body["is_organic"] = body.get("is_organic", False)
    body["update_interval"] = body.pop("update_interval").total_seconds()
    body["max_time_no_improvement"] = body.pop("max_time_no_improvement").total_seconds()
    body["best_loss_at"] = (
        datetime.now(timezone.utc).isoformat() if body["best_loss_at"] == pd.NaT else body["best_loss_at"]
    )
    body["event"] = str(body.pop("event"))
    body["best_hotkey"] = "" if body["best_hotkey"] is None else body["best_hotkey"]
    body["best_loss"] = 0 if body["best_loss"] == np.inf else body["best_loss"]

    return body
