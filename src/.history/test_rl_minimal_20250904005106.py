import gym
import numpy as np
from gym import spaces
import random
import collections

# Test imports
try:
    from stable_baselines3.common.vec_env import DummyVecEnv
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.monitor import Monitor
    print("All SB3 imports successful")
except Exception as e:
    print(f"SB3 import error: {e}")
    exit(1)

# Extended 7-Job Instance Data
EXTENDED_JOBS_DATA = collections.OrderedDict({
    0: [{'proc_times': {'M0': 4, 'M1': 6}}, {'proc_times': {'M1': 5, 'M2': 3}}, {'proc_times': {'M0': 2}}],
    1: [{'proc_times': {'M1': 7, 'M2': 5}}, {'proc_times': {'M0': 4}}, {'proc_times': {'M1': 3, 'M2': 4}}],
    2: [{'proc_times': {'M0': 5, 'M2': 6}}, {'proc_times': {'M0': 3, 'M1': 4}}, {'proc_times': {'M2': 7}}]
})
EXTENDED_MACHINE_LIST = ['M0', 'M1', 'M2']
EXTENDED_ARRIVAL_TIMES = {0: 0, 1: 0, 2: 0}

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.action_masks()

class SimpleDynamicFJSPEnv(gym.Env):
    def __init__(self, jobs_data, machine_list, job_arrival_times=None):
        super().__init__()
        self.jobs = jobs_data
        self.machines = machine_list
        self.job_ids = list(self.jobs.keys())
        self.num_jobs = len(self.job_ids)
        self.max_ops_per_job = max(len(ops) for ops in self.jobs.values()) if self.num_jobs > 0 else 1
        self.total_operations = sum(len(ops) for ops in self.jobs.values())
        
        if job_arrival_times is None:
            self.job_arrival_times = {job_id: 0 for job_id in self.job_ids}
        else:
            self.job_arrival_times = job_arrival_times

        self.action_space = spaces.Discrete(50)  # Simplified action space
        
        obs_size = 20  # Simplified observation space
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.current_step = 0
        return np.ones(20, dtype=np.float32), {}

    def step(self, action):
        self.current_step += 1
        obs = np.ones(20, dtype=np.float32) * 0.5
        reward = 1.0
        done = self.current_step >= 10  # Simple termination
        info = {"makespan": 50.0}
        return obs, reward, done, False, info

    def action_masks(self):
        # Simple mask - first 10 actions are valid
        mask = np.zeros(50, dtype=bool)
        mask[:10] = True
        return mask

def test_environment():
    print("Testing environment creation...")
    try:
        env = SimpleDynamicFJSPEnv(EXTENDED_JOBS_DATA, EXTENDED_MACHINE_LIST, EXTENDED_ARRIVAL_TIMES)
        print("Environment created successfully")
        
        obs, info = env.reset()
        print(f"Reset successful, obs shape: {obs.shape}")
        
        mask = env.action_masks()
        print(f"Action mask shape: {mask.shape}, valid actions: {np.sum(mask)}")
        
        obs, reward, done, truncated, info = env.step(0)
        print(f"Step successful, reward: {reward}")
        
        return True
    except Exception as e:
        print(f"Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training():
    print("\nTesting RL training...")
    try:
        def make_env():
            env = SimpleDynamicFJSPEnv(EXTENDED_JOBS_DATA, EXTENDED_MACHINE_LIST, EXTENDED_ARRIVAL_TIMES)
            env = ActionMasker(env, mask_fn)
            return Monitor(env)

        vec_env = DummyVecEnv([make_env])
        print("Vectorized environment created")
        
        model = MaskablePPO(
            MaskableActorCriticPolicy, 
            vec_env, 
            verbose=1,
            n_steps=64,  # Very small for testing
            batch_size=16,
            n_epochs=2,
            policy_kwargs=dict(net_arch=[32, 32])
        )
        print("Model created successfully")
        
        model.learn(total_timesteps=100)
        print("Training completed successfully!")
        
        return True
    except Exception as e:
        print(f"Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Running RL diagnostic tests...")
    
    if test_environment():
        print("✓ Environment test passed")
        if test_training():
            print("✓ Training test passed")
            print("\n✓ All tests passed! The issue is elsewhere.")
        else:
            print("✗ Training test failed")
    else:
        print("✗ Environment test failed")
