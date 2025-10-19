# System Architecture: FJSP Dataset Generation

```
┌─────────────────────────────────────────────────────────────────────┐
│                     FJSP DATASET GENERATION SYSTEM                   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                            CORE UTILITIES                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                       utils.py                              │   │
│  ├────────────────────────────────────────────────────────────┤   │
│  │                                                             │   │
│  │  generate_fjsp_dataset(                                    │   │
│  │      num_initial_jobs, num_future_jobs,                    │   │
│  │      total_num_machines, seed                              │   │
│  │  ) → (jobs_data, machine_list)                            │   │
│  │                                                             │   │
│  │  generate_arrival_times(                                   │   │
│  │      num_initial_jobs, num_future_jobs,                    │   │
│  │      arrival_mode, arrival_rate, seed                      │   │
│  │  ) → arrival_times                                         │   │
│  │                                                             │   │
│  │  print_dataset_info(                                       │   │
│  │      jobs_data, machine_list, arrival_times                │   │
│  │  )                                                         │   │
│  │                                                             │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ imports
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         MAIN APPLICATION FILES                       │
├──────────────────────┬──────────────────────┬───────────────────────┤
│                      │                      │                       │
│  backup_no_wait.py   │  proactive_sche.py  │  reactive_scheduling  │
│                      │                      │          .py          │
│  ┌────────────────┐ │  ┌────────────────┐ │  ┌────────────────┐  │
│  │ Import utils   │ │  │ Import utils   │ │  │ Import utils   │  │
│  │                │ │  │                │ │  │                │  │
│  │ Use generated  │ │  │ Use generated  │ │  │ Use generated  │  │
│  │ OR predefined  │ │  │ OR predefined  │ │  │ OR predefined  │  │
│  │ datasets       │ │  │ datasets       │ │  │ datasets       │  │
│  └────────────────┘ │  └────────────────┘ │  └────────────────┘  │
│                      │                      │                       │
└──────────────────────┴──────────────────────┴───────────────────────┘
                                    │
                                    │ imports
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        EXAMPLES & TESTING                            │
├──────────────────┬──────────────────────┬──────────────────────────┤
│                  │                      │                          │
│  test_utils.py   │ advanced_examples.py │  visualize_datasets.py  │
│                  │                      │                          │
│  ┌────────────┐ │  ┌────────────────┐ │  ┌───────────────────┐  │
│  │  5 basic   │ │  │  7 research    │ │  │  Charts &         │  │
│  │  examples  │ │  │  scenarios     │ │  │  statistics       │  │
│  │            │ │  │                │ │  │                   │  │
│  │  Simple    │ │  │  Advanced      │ │  │  Visual           │  │
│  │  demos     │ │  │  use cases     │ │  │  analysis         │  │
│  └────────────┘ │  └────────────────┘ │  └───────────────────┘  │
│                  │                      │                          │
└──────────────────┴──────────────────────┴──────────────────────────┘
                                    │
                                    │ references
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           DOCUMENTATION                              │
├──────────────────┬──────────────────────┬──────────────────────────┤
│                  │                      │                          │
│  README.md       │  QUICK_REFERENCE.md │  SUMMARY.md              │
│                  │                      │                          │
│  ┌────────────┐ │  ┌────────────────┐ │  ┌───────────────────┐  │
│  │  Full      │ │  │  Quick lookup  │ │  │  Change summary   │  │
│  │  tutorial  │ │  │  syntax guide  │ │  │  & migration      │  │
│  │            │ │  │                │ │  │                   │  │
│  │  Examples  │ │  │  Common tasks  │ │  │  Benefits         │  │
│  │  & guides  │ │  │  & tips        │ │  │  & next steps     │  │
│  └────────────┘ │  └────────────────┘ │  └───────────────────┘  │
│                  │                      │                          │
└──────────────────┴──────────────────────┴──────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                         DATA FLOW DIAGRAM                            │
└─────────────────────────────────────────────────────────────────────┘

  Input Parameters                     Generation                  Output
  ────────────────                     ──────────                  ──────

  num_initial_jobs ────┐
  num_future_jobs  ────┼──> generate_fjsp_dataset() ──> jobs_data
  total_num_machines ──┤                                 machine_list
  seed ────────────────┘

  num_initial_jobs ────┐
  num_future_jobs  ────┼──> generate_arrival_times() ──> arrival_times
  arrival_mode ────────┤
  arrival_rate ────────┤
  seed ────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                     GENERATION RULES SUMMARY                         │
└─────────────────────────────────────────────────────────────────────┘

  ┌───────────────────────────────────────────────────────────────┐
  │ Operations per Job        │ Uniform[1, 5]                     │
  ├───────────────────────────────────────────────────────────────┤
  │ Machines per Operation    │ Uniform[1, M]                     │
  ├───────────────────────────────────────────────────────────────┤
  │ Processing Time           │ Uniform[1, 10]                    │
  ├───────────────────────────────────────────────────────────────┤
  │ Initial Job Arrivals      │ t = 0                             │
  ├───────────────────────────────────────────────────────────────┤
  │ Deterministic Arrivals    │ t = 4, 8, 12, 16, ...            │
  ├───────────────────────────────────────────────────────────────┤
  │ Poisson Arrivals          │ Exponential(λ) inter-arrivals    │
  └───────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                          USAGE WORKFLOW                              │
└─────────────────────────────────────────────────────────────────────┘

  Step 1: Read Documentation
  ──────────────────────────
    QUICK_REFERENCE.md  ──┐
    README.md           ──┼──> Understand API
    SUMMARY.md          ──┘

  Step 2: Run Examples
  ────────────────────
    test_utils.py          ──┐
    advanced_examples.py   ──┼──> See demonstrations
    visualize_datasets.py  ──┘

  Step 3: Generate Data
  ─────────────────────
    from utils import generate_fjsp_dataset
    
    jobs, machines = generate_fjsp_dataset(
        num_initial_jobs=5,
        num_future_jobs=5,
        total_num_machines=3,
        seed=42
    )

  Step 4: Use in Environment
  ──────────────────────────
    env = PoissonDynamicFJSPEnv(
        jobs_data=jobs,
        machine_list=machines,
        ...
    )

  Step 5: Train/Evaluate
  ──────────────────────
    model.learn(...)
    evaluate_dynamic_on_dynamic(...)


┌─────────────────────────────────────────────────────────────────────┐
│                      COMMON CONFIGURATIONS                           │
└─────────────────────────────────────────────────────────────────────┘

  Small (Debugging)
  ─────────────────
    generate_fjsp_dataset(2, 2, 2, seed=42)
    → 4 jobs, 2 machines

  Default
  ───────
    generate_fjsp_dataset(3, 4, 3, seed=42)
    → 7 jobs, 3 machines

  Medium
  ──────
    generate_fjsp_dataset(5, 5, 4, seed=42)
    → 10 jobs, 4 machines

  Large
  ─────
    generate_fjsp_dataset(10, 10, 5, seed=42)
    → 20 jobs, 5 machines

  Very Large (Benchmarking)
  ─────────────────────────
    generate_fjsp_dataset(20, 20, 8, seed=42)
    → 40 jobs, 8 machines


┌─────────────────────────────────────────────────────────────────────┐
│                        KEY BENEFITS                                  │
└─────────────────────────────────────────────────────────────────────┘

  ✓ Flexibility       Generate any problem size
  ✓ Reproducibility   Same seed → same dataset
  ✓ Variety          Different seeds → different instances
  ✓ Scalability      Small to large problems
  ✓ Standardization  Consistent generation rules
  ✓ Easy Testing     Quick scenario generation
  ✓ Documentation    Comprehensive guides
  ✓ Examples         Multiple demonstrations
  ✓ Visualization    Built-in analysis tools
  ✓ Compatibility    Works with existing code


┌─────────────────────────────────────────────────────────────────────┐
│                         FILE REFERENCE                               │
└─────────────────────────────────────────────────────────────────────┘

  Core          │  utils.py
  ──────────────┼────────────────────────────────────────
  Examples      │  test_utils.py
                │  advanced_examples.py
                │  visualize_datasets.py
  ──────────────┼────────────────────────────────────────
  Docs          │  DATASET_GENERATION_README.md
                │  DATASET_GENERATION_SUMMARY.md
                │  QUICK_REFERENCE.md
                │  FILE_SUMMARY.md
                │  ARCHITECTURE.md (this file)
  ──────────────┼────────────────────────────────────────
  Modified      │  backup_no_wait.py
                │  proactive_sche.py
                │  reactive_scheduling.py
```
