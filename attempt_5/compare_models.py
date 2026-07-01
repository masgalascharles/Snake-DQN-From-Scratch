import numpy as np
import os
from src.agent import Agent
from src.environment import Environment
from multiprocessing import Pool


def run_episodes(args):
    model_path, number_of_episodes, training_start = args
    environment = Environment()
    input_size = environment.get_state().shape[0]
    agent = Agent(input_size, 0.001, 0.99)
    
    agent.load(model_path)
    
    score = 0
    number_of_moves = 0

    for episode in range(1, number_of_episodes + 1):
        environment.reset(for_training=training_start)
        
        done = False
        moves_without_apple = 0
        
        while not done:
            state = environment.get_state()
            timeout = min(environment.size ** 2 + environment.size, environment.size ** 2 * 0.2 + len(environment.snake) * 1.25)
            prediction = agent.predict(state.reshape(1, -1))
            direction = np.argmax(prediction)
            apple_eaten, done = environment.move(direction)
            number_of_moves += 1

            if apple_eaten:
                moves_without_apple = 0
                score += 1
            elif moves_without_apple >= timeout:
                done = True
            else:
                moves_without_apple += 1

    return (score, number_of_moves)

def compare_models(model_paths, number_of_episodes):
    print(f"\nRunning for {number_of_episodes} episodes.\n")
    
    for i in range(0, 2):
        if i == 0:
            print(f"\nTask #{i + 1}: starting state is always the default\n")
        elif i == 1:
            print(f"\nTask #{i + 1}: starting state is picked randomly\n")
        
        for model_path in model_paths:
            number_of_cores = max(1, os.cpu_count() // 2)
            episodes_per_worker = number_of_episodes // number_of_cores
            remaining = number_of_episodes % number_of_cores
            jobs = []

            for j in range(number_of_cores):
                episodes = episodes_per_worker + (1 if j < remaining else 0)
                jobs.append((model_path, episodes, bool(i)))

            with Pool(number_of_cores) as pool:
                results = pool.map(run_episodes, jobs)

            score = sum([result[0] for result in results])
            number_of_moves = sum([result[1] for result in results])
            
            print(f"{model_path} | Score: {score}, Total moves: {number_of_moves}, Apples eaten per 100 moves: {score / number_of_moves * 100}")


if __name__ == "__main__":
    model_paths = [
        "model_11_step_300000",
        "model_11_step_600000",
        "model_11_step_900000",
        "model_11_step_1200000",
        "model_11_step_1500000",
        "model_11_step_1800000",
        "model_11"
    ]

    compare_models(model_paths, 500)
