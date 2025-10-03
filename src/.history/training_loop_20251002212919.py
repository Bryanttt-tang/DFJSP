from training_visualizer import TrainingVisualizer
from job_generator import JobGenerator

class Trainer:
    def __init__(self):
        self.visualizer = TrainingVisualizer()
        self.job_gen = JobGenerator(arrival_rate=1.0, max_arrival_time=200)
        
    def train(self, num_episodes=1000):
        for episode in range(num_episodes):
            # Your training logic here...
            loss = self.train_episode()  # Your training function
            reward = self.evaluate_episode()  # Your evaluation function
            
            # Log metrics
            self.visualizer.log_episode(episode, loss, reward)
            
            # Plot every 100 episodes
            if episode % 100 == 0:
                self.visualizer.plot_training_progress()
                
    def train_episode(self):
        # Generate jobs with bounded arrival times
        arrivals = self.job_gen.generate_poisson_arrivals(num_jobs=50)
        # Your training logic...
        return loss
        
    def evaluate_episode(self):
        # Your evaluation logic...
        return reward
