#!/usr/bin/env python3
"""
run_historical_days.py

Runner to execute co-optimization, reserve subproblem and sequential energy for a series of dates.
Specifically loops May 1 2020 to May 10 2020, determines a day_type for each date,
and calls the existing scripts in /scripts.
"""
import subprocess
import os
from datetime import datetime, timedelta

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

def date_to_day_type(date_obj):
    # All May weekdays mapped to SpringWD; adjust if differentiating weekends
    return 'SpringWD'

if __name__ == '__main__':
    start = datetime(2020, 5, 1)
    end   = datetime(2020, 5, 10)
    current = start

    while current <= end:
        date_str = current.strftime('%Y-%m-%d')
        day_type = date_to_day_type(current)
        print(f"\n=== {date_str}  ({day_type}) ===")

        # 1) Co-optimization (runs all weekday/season variants inside)
        cmd_coopt = ['python3', os.path.join(SCRIPTS_DIR, 'run_and_report.py')]
        subprocess.run(cmd_coopt, check=True)

        # 2) Reserve subproblem for this season
        cmd_res = [
            'python3', os.path.join(SCRIPTS_DIR, 'run_reserve_subproblem.py'),
            '--season', day_type
        ]
        subprocess.run(cmd_res, check=True)

        # 3) Sequential energy step (fixed commitment)
        cmd_energy = [
            'python3', os.path.join(SCRIPTS_DIR, 'run_energy.py'),
            '--day-types', day_type
        ]
        subprocess.run(cmd_energy, check=True)

        current += timedelta(days=1)

    print('\nAll dates processed.')
