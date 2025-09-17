#!/usr/bin/env python3
"""
Run the working backup version to test the dynamic vs static RL comparison
"""

print("🔄 Running the working backup version: possion_job_backup.py")
print("=" * 60)

import subprocess
import sys
import os

# Change to the source directory
os.chdir('/Users/tanu/Desktop/PhD/Scheduling/src')

# Run the working backup file
try:
    result = subprocess.run([sys.executable, 'possion_job_backup.py'], 
                          capture_output=False, text=True, check=True)
    print("✅ Successfully completed the working backup version!")
except subprocess.CalledProcessError as e:
    print(f"❌ Error running backup: {e}")
    print("Let's run it directly in Python instead...")
    
    # Try running it directly
    try:
        exec(open('possion_job_backup.py').read())
        print("✅ Successfully executed backup code directly!")
    except Exception as e2:
        print(f"❌ Direct execution also failed: {e2}")
        import traceback
        traceback.print_exc()
except FileNotFoundError:
    print("❌ possion_job_backup.py not found")
    print("Available files:")
    for f in os.listdir('.'):
        if f.endswith('.py'):
            print(f"  {f}")