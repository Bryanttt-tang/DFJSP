# Event-Time Advancement: Rule-Based RL vs Best Heuristic

## Summary

Both methods use event-driven simulation with automatic time advancement, but they serve different purposes at different points in the execution.

---

## Rule-Based RL: TWO Event-Time Advancements

### 1. **BEFORE Action Selection** (Lines ~1576-1590)
```python
# At START of step(), advance event_time if no ready ops
ready_ops = self._get_ready_operations()
while not ready_ops and self.operations_scheduled < self.total_operations:
    next_event_time = self._get_next_event_time()
    self._update_event_time_and_arrivals(next_event_time)
    ready_ops = self._get_ready_operations()
```

**Purpose**: Prevents agent from getting stuck with no valid actions to select.

**When**: Before agent selects a dispatching rule.

**Why necessary**: Ensures agent always has at least one schedulable operation when asked to choose an action.

---

### 2. **AFTER Scheduling** (Lines ~1636-1644)
```python
# After scheduling operation, advance event_time if no ready ops remain
ready_ops_after = self._get_ready_operations()
if not ready_ops_after and self.operations_scheduled < self.total_operations:
    next_event_time = self._get_next_event_time()
    if next_event_time > self.event_time:
        self._update_event_time_and_arrivals(next_event_time)
```

**Purpose**: Controls the "frontier" of when future jobs become visible to the agent.

**When**: After scheduling an operation that cleared all ready operations.

**Why necessary**: Ensures sequential decision-making with limited foresight. Without this, the agent would see all future jobs immediately after they arrive, violating the event-driven constraint.

---

### Why BOTH Are Needed

These serve **different purposes**:

| **Before Action** | **After Scheduling** |
|---|---|
| Ensures agent never sees empty action space | Controls when future jobs are revealed |
| Prevents "stuck" state | Maintains sequential decision-making |
| Happens at start of step() | Happens after scheduling |
| Guarantees valid actions exist | Controls visibility frontier |

**Example scenario**:
1. Agent schedules last ready operation at time t=10
2. **AFTER advancement**: event_time advances from 10→15 (next arrival)
3. Next step() begins
4. **BEFORE advancement**: Not needed (jobs already visible from previous advancement)
5. Agent selects action, schedules operation
6. **AFTER advancement**: If this cleared all ops, advance again to next event

Without the AFTER advancement, after scheduling at step 2, the agent would still see event_time=10, so future jobs arriving at t=15 wouldn't be visible yet. The BEFORE advancement would then have to do all the work on the next step.

The key insight: **AFTER advancement is proactive** (reveals jobs as soon as all current work is done), while **BEFORE advancement is reactive** (only advances when absolutely necessary to avoid stuck state).

---

## Best Heuristic: Event-Time Advancement

### Main Loop Structure (Lines ~4935-4960)

```python
while completed_operations < total_operations:
    # Discover new arrivals at current sim_time
    for job_id, arr_time in arrival_times.items():
        if job_id not in arrived_jobs and arr_time <= sim_time:
            arrived_jobs.add(job_id)
    
    # Find ready operations
    ready_operations = [...]
    
    if not ready_operations:
        # AUTOMATIC EVENT ADVANCEMENT
        next_time = float('inf')
        
        # 1. Check for next NEW job arrival
        for job_id, arr_time in arrival_times.items():
            if job_id not in arrived_jobs and arr_time > sim_time:
                next_time = min(next_time, arr_time)
        
        # 2. Check for next machine completion
        # (Also handles precedence-blocked operations)
        for machine, free_time in machine_next_free.items():
            if free_time > sim_time:
                next_time = min(next_time, free_time)
        
        if next_time == float('inf'):
            break
        sim_time = next_time
        continue  # Re-check for ready ops at new time
    
    # Select and schedule operation...
```

---

### Why No Check for Arrived Jobs?

**Question**: Why don't we check arrival times for **already-arrived** jobs?

**Answer**: It's **unnecessary** because:

#### Case 1: Job Fully Scheduled
If a job has arrived and all its operations are done:
- `job_next_op[job_id] >= len(jobs_data[job_id])`
- No more operations to schedule
- No need to check its arrival time

#### Case 2: Job Has Remaining Ops But Not Ready
If a job has arrived and has remaining operations but they're not in `ready_operations`:
- The operation is blocked by **precedence constraint**
- Previous operation hasn't finished yet
- We need to wait for **machine completion**, not arrival time
- The machine completion check already covers this!

**Example**:
- Job J0 arrived at t=0, has 3 operations
- Op1 finishes at t=10, Op2 starts immediately
- Op2 finishes at t=20
- At t=15, when checking what's next:
  - J0 is already arrived (no need to check arrival_times[J0])
  - J0's Op3 is not ready (waiting for Op2 to finish at t=20)
  - Machine completion check finds t=20 as next event
  - At t=20, Op3 becomes ready

#### Case 3: Job Has Ready Ops
If a job has arrived and has ready operations:
- They're already in `ready_operations` list
- We wouldn't be in the `if not ready_operations` block
- No advancement needed

---

### What the Two Checks Cover

```python
# 1. New job arrivals - discovers jobs dynamically
for job_id, arr_time in arrival_times.items():
    if job_id not in arrived_jobs and arr_time > sim_time:
        next_time = min(next_time, arr_time)

# 2. Machine completions - handles:
#    - Machines becoming free for new operations
#    - Precedence constraints being satisfied
for machine, free_time in machine_next_free.items():
    if free_time > sim_time:
        next_time = min(next_time, free_time)
```

These two checks are **sufficient** because:
- New arrivals: Discovers future jobs
- Machine completions: Unblocks precedence-constrained operations in arrived jobs

---

## Key Difference: Gantt Chart Building

The best heuristic is like **building a Gantt chart** by placing operation blocks:

1. Start at sim_time = 0
2. Find all operations that can be placed now (ready_operations)
3. Select one operation using dispatching rule
4. Place it on the Gantt chart (schedule it)
5. Update sim_time to next event if no more blocks can be placed now
6. Repeat until all operations are placed

The sim_time acts as the "current time cursor" moving along the timeline, and we can only place operations that are:
- From arrived jobs
- With precedence constraints satisfied
- At or after sim_time

This matches exactly how a human scheduler would build a Gantt chart!

---

## Conclusion

✅ **Rule-Based RL**: Needs BOTH advancement points (before action + after scheduling) to:
- Prevent stuck states (before)
- Control visibility frontier (after)

✅ **Best Heuristic**: Only needs to check:
- New job arrivals
- Machine completions

❌ **Don't need**: Checking arrival times for already-arrived jobs (redundant with machine completion check)
