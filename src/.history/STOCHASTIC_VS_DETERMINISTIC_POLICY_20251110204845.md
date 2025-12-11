# Stochastic Training vs Deterministic Evaluation in RL

## Your Observation

> "If Perfect RL trains on exact arrival times and evaluates on the same arrivals, shouldn't the final training reward match the evaluation reward?"

**Short answer:** No, they will differ because **training uses stochastic policy** but **evaluation uses deterministic policy**.

## The Key Difference

### During Training (Stochastic Rollouts)
```python
# PPO samples actions from policy distribution π(a|s)
action = sample_from_distribution(policy(state))  # STOCHASTIC

# This introduces randomness:
# - Sometimes picks suboptimal actions (exploration)
# - Action probabilities like: {action1: 0.6, action2: 0.3, action3: 0.1}
# - Samples randomly: could pick action1, action2, or action3
```

**Mean Episode Reward during training:** Average reward over many **stochastic** rollouts.

### During Evaluation (Deterministic Policy)
```python
# Takes argmax of policy distribution
action, _ = model.predict(obs, deterministic=True)  # DETERMINISTIC

# Always picks best action:
# - No exploration
# - Always picks action1 (highest probability)
# - Repeatable: same state → same action
```

**Evaluation Reward:** Reward from **deterministic** rollout (best actions only).

## Example Scenario

Imagine a scheduling state where the policy learned:
- Schedule Job A on Machine 1: **70% probability** (best)
- Schedule Job A on Machine 2: **20% probability** (okay)
- Wait: **10% probability** (bad)

### Training (Stochastic):
- **70% of episodes:** Choose optimal action → makespan = 40 → reward = -40
- **20% of episodes:** Choose suboptimal action → makespan = 45 → reward = -45
- **10% of episodes:** Choose bad action → makespan = 50 → reward = -50

**Mean training reward:** -40 × 0.7 + -45 × 0.2 + -50 × 0.1 = **-42.5**

### Evaluation (Deterministic):
- **100% of episodes:** Always choose optimal action → makespan = 40 → reward = **-40**

**Deterministic evaluation reward:** **-40** (better than training average!)

## Why This Is Actually Good

This is **by design** and **expected behavior**:

1. **Exploration during training** helps the agent discover better policies
2. **Exploitation during evaluation** uses the learned policy without randomness
3. **Deterministic evaluation is fairer** for comparison (removes luck factor)

## What You Should See

For Perfect Knowledge RL trained on exact arrivals:

```
Training Progress:
  Rollout 1: mean_eps_reward = -55.0  (early, poor policy + exploration)
  Rollout 10: mean_eps_reward = -48.0  (learning, better policy + exploration)
  Rollout 50: mean_eps_reward = -42.5  (good policy + some exploration)
  Rollout 100: mean_eps_reward = -41.2  (great policy + minimal exploration)

Final Deterministic Evaluation:
  Evaluation makespan: 40.0 → reward = -40.0  (best possible!)
```

**Key insight:** Deterministic evaluation reward should be **≥** final training reward (less negative = better).

## Why They Should Be Close (But Not Identical)

For **Perfect Knowledge RL** specifically:

1. **Low entropy coefficient** (0.0001) → policy becomes nearly deterministic over time
2. **Exact arrivals** → no stochasticity in environment
3. **Converged policy** → argmax distribution becomes peaked (e.g., 99.9% vs 0.1%)

So after sufficient training:
- Training reward (stochastic): **-40.2** (mostly optimal actions + 0.2% exploration)
- Evaluation reward (deterministic): **-40.0** (100% optimal actions)

**Expected gap:** Small (< 2-3%), but deterministic should always be better or equal.

## When To Worry

🚨 **Red flags:**
```python
Training: mean_eps_reward = -40.0
Evaluation: makespan = 50.0 → reward = -50.0  # WORSE than training!
```

This indicates:
- **Bug in evaluation:** Not using the same scenario
- **Environment mismatch:** Training and evaluation use different settings
- **Policy overfitting:** Learned something specific to training randomness

✅ **Expected behavior:**
```python
Training: mean_eps_reward = -42.5  (stochastic)
Evaluation: makespan = 40.0 → reward = -40.0  (deterministic, better)
```

## Verification

The code now prints both:
```
Init 1 makespan: 295.2 | Train reward: -295.8 | Eval reward: -295.2 (gap: +1.6%)
Init 2 makespan: 291.8 | Train reward: -292.4 | Eval reward: -291.8 (gap: +0.4%)
```

**What to look for:**
- ✅ Eval reward should be **≥** train reward (less negative)
- ✅ Gap should be small (< 5%) for well-trained policy
- ✅ Both should improve over training iterations

## Advanced: Why Deterministic Is Fairer

Imagine comparing two methods:

**Method A (trained with high entropy):**
- Training: -45 (lots of exploration)
- Evaluation: -40 (deterministic)

**Method B (trained with low entropy):**
- Training: -42 (less exploration)
- Evaluation: -41 (deterministic)

If we compared **training** rewards, Method B looks better (-42 > -45).
If we compared **evaluation** rewards, Method A is actually better (-40 > -41)!

**Lesson:** Always use **deterministic evaluation** for fair comparison.

## Summary

1. **Training reward ≠ Evaluation reward** is expected and correct
2. **Deterministic evaluation** should be **≥** stochastic training (less negative)
3. **Perfect Knowledge RL** should have small gap due to low entropy
4. **Your intuition was right** about same arrivals → comparable performance
5. **But you need to account for** stochastic training vs deterministic evaluation

The code now explicitly shows both rewards so you can verify this!
