import requests
import pandas as pd
from termcolor import colored
from tabulate import tabulate
from datetime import datetime, timedelta
import argparse
from collections import defaultdict
import json
import bittensor as bt


def load_metagraph(netuid: int = 25):
    metagraph = bt.metagraph(netuid)
    return metagraph


def get_incentive_data(m: bt.metagraph, num_uids=10):
    incentive_data = sorted(
        zip(m.incentive, m.uids, m.hotkeys), key=lambda x: x[0], reverse=True
    )
    return incentive_data[:num_uids]


def response_to_dict(response):
    response = response.json()["results"][0]
    if "error" in response.keys():
        raise ValueError(f"Failed to get all PDBs: {response['error']}")
    elif "values" not in response.keys():
        return {}
    columns = response["columns"]
    values = response["values"]
    data = [dict(zip(columns, row)) for row in values]
    return data


def get_inactive_jobs(hours_back: int = 48):
    time_window = (datetime.now() - timedelta(hours=hours_back)).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )

    url = "174.138.3.61:4001"
    query = f"SELECT * FROM jobs WHERE active = 0 AND updated_at >= '{time_window}'"
    response = requests.get(
        f"http://{url}/db/query", params={"q": query, "consistency": "strong"}
    )
    return response_to_dict(response=response)


def get_active_jobs():
    url = "174.138.3.61:4001"
    query = "SELECT * FROM jobs WHERE active = 1"
    response = requests.get(
        f"http://{url}/db/query", params={"q": query, "consistency": "strong"}
    )
    return response_to_dict(response=response)


def get_hotkey_wins(df: pd.DataFrame, hotkey: str):
    if "best_hotkey" in df.columns:
        return sum(df.best_hotkey == hotkey)
    return 0


def get_number_of_submissions(df: pd.DataFrame):
    results = defaultdict(int)
    for _, row in df.iterrows():
        event = json.loads(row.event)
        for hk, status_code in zip(eval(row.hotkeys), event["response_status_codes"]):
            results[hk] += int(status_code) == 200

    return results


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Display job leaderboard")
    parser.add_argument(
        "--status",
        choices=["active", "inactive", "leaderboard"],
        default="leaderboard",
        help="Show active jobs, inactive jobs, or miner leaderboard (default: active)",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=48,
        help="For inactive jobs, show jobs from the last N hours (default: 48)",
    )
    parser.add_argument(
        "--num_uids",
        type=int,
        default=10,
        help="Number of top UIDs to display in leaderboard (default: 10)",
    )
    parser.add_argument(
        "--netuid", type=int, default=25, help="Network UID to use (default: 25)"
    )
    args = parser.parse_args()

    if args.status == "leaderboard":
        # Get miner leaderboard data
        print(
            colored(
                f"\n=== Loading metagraph for netuid {args.netuid}... ===\n", "green"
            )
        )
        metagraph = load_metagraph(netuid=args.netuid)

        print(
            colored(
                f"=== Getting top {args.num_uids} miners by incentive... ===\n", "green"
            )
        )
        incentive_data = get_incentive_data(metagraph, num_uids=args.num_uids)

        print(
            colored(
                f"=== Getting inactive jobs for the last {args.hours} hours... ===\n",
                "green",
            )
        )
        jobs = get_inactive_jobs(hours_back=args.hours)
        jobs_df = pd.DataFrame(jobs)

        # Get number of submissions for each hotkey
        submissions = get_number_of_submissions(jobs_df)

        # Prepare data for display
        leaderboard_data = []
        for incentive, uid, hotkey in incentive_data:
            wins = get_hotkey_wins(jobs_df, hotkey)
            leaderboard_data.append(
                {
                    "UID": int(uid),
                    "Hotkey": hotkey[:10] + "...",  # Truncate for display
                    "Incentive": round(float(incentive), 6),
                    "Wins": wins,
                    "Submissions": submissions[hotkey],
                    "Win/Submission %": 100 * round(wins / submissions[hotkey], 2),
                }
            )

        # Create DataFrame for display
        leaderboard_df = pd.DataFrame(leaderboard_data)

        # Calculate total wins
        total_wins = leaderboard_df["Wins"].sum()

        # Calculate median Win/Submission Percentage
        median_win_submission_percentage = leaderboard_df["Win/Submission %"].median()

        # Add a summary row
        summary_row = {
            "UID": "TOTAL",
            "Hotkey": "",
            "Incentive": "",
            "Wins": total_wins,
            "Submissions": "",
            "Win/Submission %": str(median_win_submission_percentage) + "%",
        }

        # Append summary row to the DataFrame
        leaderboard_df = pd.concat(
            [leaderboard_df, pd.DataFrame([summary_row])], ignore_index=True
        )

        # Format and display the table with green color
        table = tabulate(
            leaderboard_df, headers="keys", tablefmt="grid", showindex=False
        )
        TITLE = f"TOP MINERS LEADERBOARD (Last {args.hours} hours)"
        print(colored(f"\n=== {TITLE} ===\n", "green", attrs=["bold"]))
        print(colored(table, "green"))

        # Add additional statistics
        total_jobs = len(jobs_df) if not jobs_df.empty else 0
        print(colored(f"\nTotal Completed Jobs: {total_jobs}", "green"))
        print(
            colored(
                f"Percentage of Submitted Jobs Won by Top {args.num_uids} Miners: {round(total_wins/total_jobs*100, 2)}%",
                "green",
            )
        )

    else:
        # Get jobs based on the specified status
        print(
            colored(
                f"\n=== Extracting {args.status} jobs... ===\n", "green", attrs=["bold"]
            )
        )
        if args.status == "active":
            jobs = get_active_jobs()
            TITLE = "ACTIVE JOBS LEADERBOARD"
        else:
            jobs = get_inactive_jobs(hours_back=args.hours)
            TITLE = f"INACTIVE JOBS LEADERBOARD (Last {args.hours} hours)"

        df = pd.DataFrame(jobs)

        # Extract just the PDB IDs
        if "pdb_id" in df.columns and not df.empty:
            pdb_ids = df["pdb_id"].tolist()

            # Create a simple DataFrame with just PDB IDs for display
            display_df = pd.DataFrame({"PDB ID": pdb_ids})

            # Format and display the table with green color
            table = tabulate(
                display_df, headers="keys", tablefmt="grid", showindex=True
            )
            print(colored(f"\n=== {TITLE} ===\n", "green", attrs=["bold"]))
            print(colored(table, "green"))
            print(
                colored(
                    f"\nTotal {args.status.capitalize()} Jobs: {len(pdb_ids)}", "green"
                )
            )
        else:
            print(
                colored(f"No {args.status} jobs found or no PDB IDs in the data", "red")
            )
