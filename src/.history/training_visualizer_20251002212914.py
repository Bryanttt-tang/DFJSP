import matplotlib.pyplot as plt
import numpy as np
from collections import deque

class TrainingVisualizer:
    def __init__(self, window_size=100):
        self.losses = []
        self.rewards = []
        self.episodes = []
        self.window_size = window_size
        
    def log_episode(self, episode, loss, reward):
        """Log training metrics for an episode"""
        self.episodes.append(episode)
        self.losses.append(loss)
        self.rewards.append(reward)
        
    def plot_training_progress(self, save_path=None):
        """Plot loss and reward trends"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot loss
        ax1.plot(self.episodes, self.losses, 'b-', alpha=0.3, label='Raw Loss')
        if len(self.losses) > self.window_size:
            smoothed_loss = self._moving_average(self.losses, self.window_size)
            ax1.plot(self.episodes[self.window_size-1:], smoothed_loss, 'b-', linewidth=2, label='Smoothed Loss')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss Over Episodes')
        ax1.legend()
        ax1.grid(True)
        
        # Plot reward
        ax2.plot(self.episodes, self.rewards, 'g-', alpha=0.3, label='Raw Reward')
        if len(self.rewards) > self.window_size:
            smoothed_reward = self._moving_average(self.rewards, self.window_size)
            ax2.plot(self.episodes[self.window_size-1:], smoothed_reward, 'g-', linewidth=2, label='Smoothed Reward')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Expected Reward')
        ax2.set_title('Expected Reward Over Episodes')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def _moving_average(self, data, window):
        """Calculate moving average"""
        return np.convolve(data, np.ones(window)/window, mode='valid')
