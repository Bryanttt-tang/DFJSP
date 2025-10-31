#!/usr/bin/env python3
"""
Script to rename all instances of 'Dynamic RL' to 'Reactive RL' in proactive_sche.py
"""

# Read the file
with open('proactive_sche.py', 'r') as f:
    content = f.read()

# Count occurrences before replacement
count_dynamic = content.count('Dynamic RL')
count_dynamic_lower = content.count('dynamic RL')
count_dynamic_upper = content.count('DYNAMIC RL')

print(f"Found {count_dynamic} instances of 'Dynamic RL'")
print(f"Found {count_dynamic_lower} instances of 'dynamic RL'")
print(f"Found {count_dynamic_upper} instances of 'DYNAMIC RL'")
print(f"Total: {count_dynamic + count_dynamic_lower + count_dynamic_upper} instances")

# Perform replacements (case-sensitive)
content = content.replace('Dynamic RL', 'Reactive RL')
content = content.replace('dynamic RL', 'reactive RL')
content = content.replace('DYNAMIC RL', 'REACTIVE RL')

# Write back
with open('proactive_sche.py', 'w') as f:
    f.write(content)

# Verify
with open('proactive_sche.py', 'r') as f:
    new_content = f.read()
    
count_reactive = new_content.count('Reactive RL')
count_reactive_lower = new_content.count('reactive RL')
count_reactive_upper = new_content.count('REACTIVE RL')

print(f"\n✅ Replacement completed!")
print(f"Now have {count_reactive} instances of 'Reactive RL'")
print(f"Now have {count_reactive_lower} instances of 'reactive RL'")
print(f"Now have {count_reactive_upper} instances of 'REACTIVE RL'")
print(f"Total: {count_reactive + count_reactive_lower + count_reactive_upper} instances")
