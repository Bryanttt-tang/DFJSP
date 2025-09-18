import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import torch
import matplotlib.pyplot as plt

# Custom callback to log episode length and KL divergence
class LoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_lengths = []
        self.kl_divergences = []

    def _on_step(self) -> bool:
        # Log episode lengths as before
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_lengths.append(info["episode"]["l"])
                print(f"Episode length: {info['episode']['l']}")
        return True


    def _on_rollout_end(self):
        kl = self.model.logger.name_to_value.get("train/approx_kl")
        if kl is not None:
            self.kl_divergences.append(kl)
            print(f"[Callback] Approx. KL from PPO logger: {kl:.6f}")


    def _on_training_end(self) -> None:
        # Plot once at the very end
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
        if self.episode_lengths:
            ax1.plot(self.episode_lengths)
            ax1.set_title("Episode Length")
            ax1.set_xlabel("Episode")
            ax1.set_ylabel("Length")
        if self.kl_divergences:
            ax2.plot(self.kl_divergences)
            ax2.set_title("Approx. KL Divergence")
            ax2.set_xlabel("Rollout #")
            ax2.set_ylabel("KL")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # 🧠 Parallelized environment: runs 8 environments at once, Faster rollout collection and Stable on-policy learning
    # each of the 8 envs collects 2048 steps
    env = make_vec_env("CartPole-v1", n_envs=2, vec_env_cls=SubprocVecEnv)

    # 📐 Define network architecture
    policy_kwargs = dict(
        net_arch=[dict(pi=[64, 64], vf=[64, 64])]  # separate networks for policy and value
    )

    # 🏋️‍♂️ Initialize PPO model
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        device="cpu",
        verbose=1,
        n_steps=2048,
        n_epochs=50,
        learning_rate=0.0003,
        ent_coef=0.01
    )

    # 📊 Train and log with callback
    logging_callback = LoggingCallback()
    model.learn(total_timesteps=250000, callback=logging_callback)
    model.save("ppo_cartpole")
    print("Saved trained policy.")

    del model  # Remove model to demonstrate loading

    # 🔁 Load saved model
    model = PPO.load("ppo_cartpole")

    # 🎥 Run trained model and visualize
    eval_env = gym.make("CartPole-v1", render_mode="human")
    for ep in range(1):
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"Finished evaluation with reward: {total_reward}")
    eval_env.close()
