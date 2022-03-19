from pathlib import Path
from datetime import datetime


def path_with_date(path, mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    date = datetime.utcnow()
    date_str = f"{date.year}_{date.month}_{date.day}_{date.hour}_{date.minute}_{date.second}"
    path = path.joinpath(date_str)
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path
