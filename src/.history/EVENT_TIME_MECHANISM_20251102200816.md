# Event Time Mechanism: Preventing Past Scheduling

## Fixed Bug ✅

**Error:** `TypeError: predict_next_arrivals() got an unexpected keyword argument 'num_predictions'`

**Fix:** Changed parameter name from `num_predictions` to `num_jobs_to_predict` to match the method signature.

---

## Event Time (`self.event_time`) - The Time Frontier

### Core Concept

`self.event_time` represents the **"current knowledge frontier"** in the environment. It's the point in time up to which the agent has complete information about:
- Which jobs have arrived
- Which machines are idle or busy
- Current system state

**Critical Invariant:** The agent can NEVER schedule operations to start before `self.event_time`.

---

## How Event Time Updates

### 1. Initialization
```python
def reset(self, seed=None):
    self.event_time = 0.0  # Start at time zero
    # Initial jobs arrive at t=0
    self._check_arrivals()  # Jobs with arrival_time <= 0.0 become available
```

### 2. Wait Actions (6 durations: 1, 2, 3, 5, 10, ∞)

**When:** Agent explicitly chooses to wait

**Update Logic:**
```python
def _execute_wait_action_with_predictor_guidance(self, wait_duration):
    # Determine target wait time
    if wait_duration == float('inf'):
        # Wait to next event (arrival or machine completion)
        next_event_time = self._get_next_event_time()
        target_time = next_event_time
    else:
        # Wait for specific duration (but not beyond next event)
        target_time = self.event_time + wait_duration
        next_event_time = self._get_next_event_time()
        target_time = min(target_time, next_event_time)  # Cap at next event
    
    # CRITICAL: Update event_time
    self.event_time = target_time
    
    # Check for new arrivals at updated time
    self._check_arrivals()
```

**Example:**
```
Current: event_time = 10.0
Agent: wait_5 (wait 5 units)
Next event: t = 18.0 (job arrival)

Execution:
  target_time = 10 + 5 = 15.0
  target_time = min(15.0, 18.0) = 15.0 ✓
  
  event_time: 10.0 → 15.0
  
  _check_arrivals():
    Check if any jobs with arrival_time <= 15.0
    If yes, add to self.arrived_jobs
```

### 3. Scheduling Actions (when all idle machines used)

**When:** Agent schedules an operation on the LAST idle machine

**Update Logic:**
```python
def step(self, action):
    # ... schedule operation ...
    
    machine_free_before_scheduling = self.machine_end_times[machine]
    
    # Calculate start time (CANNOT be before event_time)
    start_time = max(machine_ready, job_ready, self.event_time)
    
    # After scheduling, check if we should advance time
    if machine_free_before_scheduling <= self.event_time:
        # This machine was IDLE
        
        # Check if any OTHER idle machines remain
        other_idle_machines_exist = any(
            free_time <= self.event_time 
            for m, free_time in self.machine_end_times.items()
            if m != machine
        )
        
        if not other_idle_machines_exist:
            # NO MORE IDLE MACHINES - advance to next event
            next_event_time = self._get_next_event_time()
            if next_event_time > self.event_time:
                self.event_time = next_event_time
                self._check_arrivals()
```

**Example:**
```
State at t=10:
  M0: idle (free_time = 10)
  M1: busy until 15 (free_time = 15)
  M2: idle (free_time = 10)
  
Agent: Schedule J3→M0

After scheduling:
  M0: busy until 20 (free_time = 20)
  M1: busy until 15
  M2: idle (free_time = 10)  ← Still idle!
  
Decision: Do NOT advance event_time (other idle machine exists)
event_time stays at 10.0
```

```
State at t=10:
  M0: idle (free_time = 10)
  M1: busy until 15 (free_time = 15)
  M2: busy until 18 (free_time = 18)
  
Agent: Schedule J3→M0

After scheduling:
  M0: busy until 20 (free_time = 20)
  M1: busy until 15
  M2: busy until 18
  
Decision: NO MORE idle machines
next_event = min(15, 18, next_arrival) = 15
event_time: 10 → 15 ✓

At t=15:
  M1 becomes idle
  Check for new arrivals with arrival_time <= 15
```

---

## Preventing Past Scheduling

### The Guard: `max(machine_ready, job_ready, self.event_time)`

Every scheduling operation calculates start time as:

```python
# Calculate earliest start time - CANNOT schedule in the past
machine_ready = self.machine_end_times[machine]  # When machine free
job_ready = max(self.job_end_times[job_id], actual_arrival)  # When job ready
start_time = max(machine_ready, job_ready, self.event_time)  # ← THE GUARD
```

**Why this works:**

1. **`machine_ready`**: Can't start before machine is free
2. **`job_ready`**: Can't start before job arrives AND previous operation completes
3. **`self.event_time`**: **Can't start before current knowledge frontier**

The third constraint is CRITICAL - it ensures:
```
start_time >= self.event_time (ALWAYS)
```

### Example Scenarios

**Scenario 1: Normal Scheduling**
```
event_time = 20
M0 free at 18 (< event_time, so M0 is idle)
J5 arrived at 15, previous op finished at 22

start_time = max(18, 22, 20) = 22 ✓
```

**Scenario 2: After Wait Action**
```
event_time = 10
Agent: wait_5
event_time: 10 → 15

M0 free at 12 (< 15, so M0 is idle now)
J7 arrived at 14, no previous ops

start_time = max(12, 14, 15) = 15 ✓
Operation scheduled at t=15, not at t=14 (past)
```

**Scenario 3: Machine Busy in Future**
```
event_time = 20
M0 free at 25 (> event_time, so M0 is busy)
J8 arrived at 18, no previous ops

start_time = max(25, 18, 20) = 25 ✓
Operation scheduled at t=25 (when machine becomes free)
```

---

## Event Time Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ EPISODE START                                                    │
├─────────────────────────────────────────────────────────────────┤
│ event_time = 0.0                                                │
│ Initial jobs arrive (arrival_time <= 0)                         │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ AGENT DECISION POINT                                            │
├─────────────────────────────────────────────────────────────────┤
│ Observation includes:                                            │
│   - event_time (current frontier)                               │
│   - arrived_jobs (arrival_time <= event_time)                   │
│   - machine states at event_time                                │
│                                                                  │
│ Agent chooses:                                                   │
│   - Scheduling action (arrived job → machine)                   │
│   - Wait action (1, 2, 3, 5, 10, or ∞ time units)              │
└─────────────────┬───────────────────────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
┌───────────────┐    ┌──────────────────┐
│ SCHEDULE      │    │ WAIT             │
├───────────────┤    ├──────────────────┤
│ 1. Calculate  │    │ 1. Advance time  │
│    start_time │    │    event_time += │
│    = max(..., │    │    duration      │
│    event_time)│    │                  │
│                │    │ 2. Check arrivals│
│ 2. Record op   │    │    (arrival_time │
│                │    │    <= event_time)│
│ 3. If last    │    │                  │
│    idle machine│    │ 3. Calculate     │
│    used:       │    │    reward        │
│    - Advance   │    │    (predictor-   │
│      event_time│    │    guided)       │
│    - Check     │    │                  │
│      arrivals  │    │                  │
└────────┬──────┘    └────────┬─────────┘
         │                    │
         └──────────┬─────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ UPDATE STATE          │
        ├───────────────────────┤
        │ - machine_end_times   │
        │ - job_progress        │
        │ - current_makespan    │
        │ - arrived_jobs        │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ DONE?                 │
        ├───────────────────────┤
        │ All jobs completed?   │
        └───────┬───────────────┘
                │
        ┌───────┴───────┐
        │ NO            │ YES
        ▼               ▼
┌──────────────┐   ┌──────────────┐
│ Next step    │   │ Episode end  │
│ (loop back)  │   │ (reset)      │
└──────────────┘   └──────────────┘
```

---

## Critical Invariants Maintained

### Invariant 1: Time Monotonicity
```python
# event_time NEVER decreases
assert new_event_time >= old_event_time
```

**Enforced by:**
- Wait actions: `event_time += positive_duration`
- Scheduling: `event_time = max(event_time, next_event)`

### Invariant 2: No Past Scheduling
```python
# All operation start times >= event_time
assert start_time >= self.event_time
```

**Enforced by:**
```python
start_time = max(machine_ready, job_ready, self.event_time)
```

### Invariant 3: Arrival Visibility
```python
# Agent can only see jobs that have arrived
assert all(self.job_arrival_times[j] <= self.event_time 
          for j in self.arrived_jobs)
```

**Enforced by:**
```python
def _check_arrivals(self):
    for job_id, arrival_time in self.job_arrival_times.items():
        if job_id not in self.arrived_jobs and arrival_time <= self.event_time:
            self.arrived_jobs.add(job_id)
```

### Invariant 4: Idle Machine Definition
```python
# Machine is idle if free_time <= event_time
is_idle = (self.machine_end_times[machine] <= self.event_time)
```

**Used in:**
- Action masking (only idle machines can be assigned)
- Event time advancement decision
- Observation space (machine states)

---

## Wait Action Impact on Event Time

### Flexible Wait Durations

```python
self.wait_durations = [1.0, 2.0, 3.0, 5.0, 10.0, float('inf')]
```

**Duration 1-10 (Fixed):**
- Agent commits to waiting specific time
- `event_time += duration`
- Checks arrivals at new time
- Can reassess situation after

**Duration ∞ (Adaptive):**
- Wait until next event (arrival or machine completion)
- `event_time = _get_next_event_time()`
- Largest time jump
- Used when no immediate work available

### Example: Strategic Wait

```
State at t=20:
  M0: idle, M1: idle, M2: busy until 28
  Arrived: J3 (SHORT job)
  Predictor: J5 (LONG job) arriving at t=25 (confidence 0.85)
  
Agent decision tree:
  IF wait_5:
    event_time: 20 → 25
    At t=25: J5 arrives (if prediction correct)
    Can schedule J5 on fast M0 (better makespan)
    
  IF schedule_J3_now:
    event_time: stays at 20 (M1 still idle)
    M0 becomes busy
    When J5 arrives at t=25, only M1 available (slower)
    
  Strategic choice: wait_5 if confident LONG job coming
```

---

## Debugging Event Time Issues

### Common Problems

**Problem 1: Time doesn't advance**
```python
# Symptom: event_time stuck, environment seems frozen

# Check 1: Are there idle machines?
idle_count = sum(1 for t in self.machine_end_times.values() 
                if t <= self.event_time)
print(f"Idle machines: {idle_count}")

# If idle_count > 0, event_time should NOT advance
# Agent needs to schedule or explicitly wait

# Check 2: Is next event time valid?
next_event = self._get_next_event_time()
print(f"Next event: {next_event}")

# If next_event == inf, all jobs arrived and machines idle
# Episode should terminate
```

**Problem 2: Operations scheduled before event_time**
```python
# This should NEVER happen due to max() guard

# If you see this, check:
start_time = max(machine_ready, job_ready, self.event_time)
assert start_time >= self.event_time, f"Violation: {start_time} < {self.event_time}"
```

**Problem 3: Jobs arrive before event_time**
```python
# Symptom: Job appears in arrived_jobs but arrival_time > event_time

# This is a bug! Check _check_arrivals():
def _check_arrivals(self):
    for job_id, arrival_time in self.job_arrival_times.items():
        if job_id not in self.arrived_jobs and arrival_time <= self.event_time:
            # ^ Must have arrival_time <= event_time
            self.arrived_jobs.add(job_id)
```

### Logging Event Time

```python
# Add this to step() for debugging
print(f"Step {self.steps}: event_time={self.event_time:.1f}, "
      f"action={'wait' if is_wait else f'schedule J{job_id}'}, "
      f"arrived={len(self.arrived_jobs)}, "
      f"idle_machines={sum(1 for t in self.machine_end_times.values() if t <= self.event_time)}")
```

---

## Summary

**Event Time (`self.event_time`):**
- Current knowledge frontier
- Updated by: wait actions (always) + scheduling actions (conditionally)
- NEVER decreases (monotonic)

**Preventing Past Scheduling:**
- Guard: `start_time = max(machine_ready, job_ready, event_time)`
- Ensures: `start_time >= event_time` ALWAYS
- Result: Operations can only be scheduled at or after current time

**Wait Actions:**
- Explicitly advance event_time
- Check for new arrivals at updated time
- Enable strategic temporal reasoning
- 6 durations allow nuanced waiting policies

**The system maintains temporal consistency while enabling sophisticated proactive scheduling through predictor-guided wait decisions.**
