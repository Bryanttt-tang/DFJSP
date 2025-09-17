from email import policy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np
import itertools
import gymnasium as gym
import matplotlib.pyplot as plt
import time
# for i in gym.envs.registry.keys():
# 	print(i)

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x=self.fc1(x)
        x=self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
def calculate_stepwise_returns(rewards, discount_factor):
    returns = []

    R = 0
    for r in reversed(rewards):
        R = r + R * discount_factor # R_t=r_t + gamma * R_{t+1}
        returns.insert(0, R)
    print("rewards: ", rewards)
    print("returns: ", returns)    
    returns = torch.tensor(returns)
    print("returns in tensor: ", returns)
    normalized_returns = (returns - returns.mean()) / returns.std()
    return normalized_returns
    
    # Simulates one full episode in the environment
def forward_pass(env, policy, discount_factor):
    log_prob_actions = []
    rewards = []
    done = False
    episode_return = 0 # total return (sum of all rewards in the episode).

    policy.train()
    # Resets the environment to start a new episode，get initial state.
    observation, info = env.reset()

    while not done:
        # unsqueeze(0) adds a batch dimension — turns shape (n,) into (1, n)
        observation = torch.FloatTensor(observation).unsqueeze(0)
        action_pred = policy(observation)
        action_prob = F.softmax(action_pred, dim = -1)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        # log probability of the sampled action — used later in computing gradients.
        log_prob_action = dist.log_prob(action)
        # print("log_prob_action: ", log_prob_action)
        # print("action: ", action.item())
        observation, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated

        log_prob_actions.append(log_prob_action)
        rewards.append(reward)
        episode_return += reward
    # Concatenates all log probability tensors into one tensor.
    log_prob_actions = torch.cat(log_prob_actions)
    stepwise_returns = calculate_stepwise_returns(rewards, discount_factor)
    print("stepwise_returns: ", len(stepwise_returns))
    print('rewards: ', len(rewards))
    return episode_return, stepwise_returns, log_prob_actions

def calculate_loss(stepwise_returns, log_prob_actions):
    loss = -(stepwise_returns * log_prob_actions).sum()
    return loss

def update_policy(stepwise_returns, log_prob_actions, optimizer):
    stepwise_returns = stepwise_returns.detach()
    loss = calculate_loss(stepwise_returns, log_prob_actions)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def visualize_policy(env, policy, episodes=3, render_delay=0.02):
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            env.render()
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            # Use deterministic (greedy) action:
            with torch.no_grad():
                logits = policy(obs_tensor)
                action = torch.argmax(logits, dim=1).item()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

            time.sleep(render_delay)  # slow down for viewability

        print(f"Episode {ep+1}: survived {steps} steps, reward {total_reward}")
    env.close()

    # for ep in range(episodes):
    #     obs = env.reset()[0]  # for gym >= 0.26
    #     done = False
    #     total_reward = 0
    #     while not done:
    #         env.render()  # display the environment
    #         obs_tensor = torch.tensor(obs).float().unsqueeze(0)
    #         with torch.no_grad():
    #             logits = policy(obs_tensor)
    #             action = torch.argmax(logits, dim=1).item()
    #         obs, reward, done, _, _ = env.step(action)
    #         total_reward += reward
    #     print(f"Episode {ep+1} finished with reward {total_reward}")
    # env.close()    

def main(): 
    env = gym.make("CartPole-v1")
    print("observation space: ", env.observation_space)
    observation, info = env.reset()
    print("observation: ", observation.shape)
    print("action space: ", env.action_space.n)
    SEED=1111
    env.reset(seed=SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    MAX_EPOCHS = 500
    DISCOUNT_FACTOR = 0.99
    N_TRIALS = 25
    REWARD_THRESHOLD = 475
    PRINT_INTERVAL = 10
    INPUT_DIM = env.observation_space.shape[0]
    HIDDEN_DIM = 128
    OUTPUT_DIM = env.action_space.n
    DROPOUT = 0.5

    episode_returns = []
    episode_lengths = []
    policy = PolicyNetwork(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT)

    # visualize_policy(env, policy)

    LEARNING_RATE = 0.0001
    optimizer = optim.Adam(policy.parameters(), lr = LEARNING_RATE)

    for episode in range(1, MAX_EPOCHS+1):
        obs = env.reset()[0]
        episode_return, stepwise_returns, log_prob_actions = forward_pass(env, policy, DISCOUNT_FACTOR)
        _ = update_policy(stepwise_returns, log_prob_actions, optimizer)

        episode_returns.append(episode_return)
        mean_episode_return = np.mean(episode_returns[-N_TRIALS:])
        episode_lengths.append(len(stepwise_returns))
        if episode % PRINT_INTERVAL == 0:
            print(f'| Episode: {episode:3} | Mean Rewards: {mean_episode_return:5.1f} |')

        if mean_episode_return >= REWARD_THRESHOLD:
            print(f'Reached reward threshold in {episode} episodes')
            break
    env.close()

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(episode_returns)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Return per Episode")

    plt.subplot(1,2,2)
    plt.plot(episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length (Steps)")
    plt.title("Survival Length During Training")

    plt.tight_layout()
    plt.show()
    env_test = gym.make("CartPole-v1", render_mode="human")  # use 'human' render mode
    visualize_policy(env_test, policy)

if __name__ == "__main__":
    main()
