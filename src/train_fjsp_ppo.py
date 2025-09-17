import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from dynamic_fjsp_env import FJSPEnv


def train_fjsp():
    """Train PPO agent on FJSP environment"""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError:
        print("Please install stable-baselines3: pip install stable-baselines3")
        return

    def make_env():
        return FJSPEnv(reward_mode="dense")

    # Create environment
    env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])

    # Create PPO model with better hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.95,
        ent_coef=0.01,  # Small entropy bonus for exploration
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[128, 128])  # Smaller network for this simple problem
    )

    # Set up evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_fjsp_model/",
        log_path="./logs/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    print("Starting training...")
    model.learn(total_timesteps=20000, callback=eval_callback)
    model.save("ppo_fjsp_final")

    print("\nTraining completed! Testing the trained model...")
    
    # Test the trained model
    test_env = FJSPEnv(reward_mode="dense")
    obs = test_env.reset()
    total_reward = 0
    steps = 0
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        total_reward += reward
        steps += 1
        
        if done:
            break
    
    print(f"\nTest Episode Results:")
    print(f"Total steps: {steps}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final makespan: {test_env.current_makespan:.2f}")
    print(f"Average reward per step: {total_reward/steps:.2f}")
    
    # Render final result
    test_env.render()
    
    return model, test_env

if __name__ == "__main__":
    # Run a simple test first
    print("Testing environment...")
    env = FJSPEnv(reward_mode="dense")
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test a few random actions
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {i+1}: Action {action}, Reward {reward:.2f}, Done {done}")
        if done:
            break
    
    env.render()
    print("\nEnvironment test completed!")
    
    # Uncomment to run training
    model, test_env = train_fjsp()