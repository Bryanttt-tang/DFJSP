import numpy as np

class JobGenerator:
    def __init__(self, arrival_rate=1.0, max_arrival_time=200):
        self.arrival_rate = arrival_rate
        self.max_arrival_time = max_arrival_time
    
    def generate_poisson_arrivals(self, num_jobs):
        """Generate Poisson arrival times with max bound"""
        arrivals = []
        current_time = 0
        
        for _ in range(num_jobs):
            inter_arrival = np.random.exponential(1/self.arrival_rate)
            current_time += inter_arrival
            
            if current_time > self.max_arrival_time:
                break
                
            arrivals.append(current_time)
        
        return np.array(arrivals)
