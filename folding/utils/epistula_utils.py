import numpy as np
from datetime import datetime, timezone
import pandas as pd
import json


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
    body["s3_links"] = json.dumps(body.pop("s3_links"))
    body["priority"] = body.get("priority", 1)
    body["is_organic"] = body.get("is_organic", False)
    body["update_interval"] = body.pop("update_interval").total_seconds()
    body["max_time_no_improvement"] = body.pop("max_time_no_improvement").total_seconds()
    body["best_loss_at"] = (
         body["best_loss_at"] if pd.notna(body["best_loss_at"]) else datetime.now(timezone.utc).isoformat()
    )
    body["event"] = json.dumps(body.pop("event"))
    body["best_hotkey"] = "" if body["best_hotkey"] is None else body["best_hotkey"]
    body["best_loss"] = 0 if body["best_loss"] == np.inf else body["best_loss"]

    return body
