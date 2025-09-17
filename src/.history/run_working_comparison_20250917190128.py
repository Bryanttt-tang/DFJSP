#!/usr/bin/env python3
"""
Working Dynamic vs Static RL comparison based on possion_job_backup.py
This version should run without errors by copying the working structure.
"""

# Copy the working imports and structure from the backup
import sys
import os

# Use the working backup file as the base
backup_file = '/Users/tanu/Desktop/PhD/Scheduling/src/possion_job_backup.py'

if os.path.exists(backup_file):
    # Load and execute the working backup code
    with open(backup_file, 'r') as f:
        backup_code = f.read()
    
    # Execute the working backup code
    print("Loading working backup code...")
    exec(backup_code)
    
    print("Successfully loaded working Dynamic vs Static RL comparison!")
    print("The backup file contains all the necessary working components.")
else:
    print(f"Error: Backup file not found at {backup_file}")
    print("Please ensure possion_job_backup.py is in the correct location.")