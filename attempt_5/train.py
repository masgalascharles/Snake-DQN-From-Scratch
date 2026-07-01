import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import os
import json
from src.agent import Agent
from src.environment import Environment
from src.environment import moves
from collections import deque


rng = np.random.default_rng()


environment = Environment()

steps = 2000000
epsilon = 0.005
min_epsilon = 0.0005
epsilon_decay_rate = (epsilon - min_epsilon) / 2000000
target_network_update_interval = 10000
print_interval = 5000
save_graph_interval = 300000
save_model_interval = 300000
eval_length = 20
moves_survived_per_episode = []
apples_eaten_per_episode = []
losses = []

model_name = "model_11"
input_size = environment.get_state().shape[0]
learning_rate = 0.0003
gamma = 0.99
memory = deque(maxlen=350000)
batch_size = 128
agent = Agent(input_size, learning_rate, gamma)

config = {
    "model_name": model_name,
    
    "steps": steps,
    "epsilon_start": epsilon,
    "min_epsilon": min_epsilon,
    "epsilon_decay_rate": epsilon_decay_rate,
    "target_network_update_interval": target_network_update_interval,
    "learning_rate": learning_rate,
    "gamma": gamma,
    "memory_length": memory.maxlen,
    "batch_size": batch_size,
    "input_size": input_size,
    
    "timeout": "min(environment.size ** 2 + environment.size, environment.size ** 2 * 0.2 + len(environment.snake) * 1.25)",
    "starting_lengths_used": "33% 2-75, 33% 76-150, 33% 151-300",
    "other_notes": "Just an experiment to see what happens if I train model 10 for even longer."
}


with open(f"{model_name}_config.json", "w") as f:
    json.dump(config, f, indent=4)


def save_graphs(step):
    plt.figure(figsize=(10, 5))
    plt.plot(moves_survived_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Moves Survived")
    plt.title("Moves Survived Per Episode Over Time")
    plt.grid(True)
    plt.savefig(f"{model_name}_moves_survived_per_episode_over_time_step_{step}.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(apples_eaten_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Apples Eaten")
    plt.title("Apples Eaten Per Episode Over Time")
    plt.grid(True)
    plt.savefig(f"{model_name}_apples_eaten_per_episode_over_time_step_{step}.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss Over Time")
    plt.grid(True)
    plt.savefig(f"{model_name}_loss_over_time_step_{step}.png", dpi=300, bbox_inches="tight")
    plt.close()


steps_without_apple = 0
moves_survived = 0
apples_eaten = 0

agent.load("models/model_10/model_10")
environment.reset(for_training=True)

for step in range(1, steps + 1):
    timeout = min(environment.size ** 2 + environment.size, environment.size ** 2 * 0.2 + len(environment.snake) * 1.25)
    state = environment.get_state()

    if rng.random() >= epsilon:
        prediction = agent.predict(state.reshape(1, -1))
        direction = np.argmax(prediction)
    else:
        direction = rng.integers(0, 4)

    apple_eaten, done = environment.move(direction)
    reward = -0.001

    if done:
        reward = -1
    elif apple_eaten:
        reward = 0.9
        apples_eaten += 1
        steps_without_apple = 0
    else:
        steps_without_apple += 1
        
        if steps_without_apple > timeout:
            done = True
            reward = -1

    next_state = environment.get_state()

    memory.append([state, direction, reward, next_state, done])

    if len(memory) > batch_size:
        loss = agent.train(memory, batch_size)
        
        losses.append(loss)

    if step % target_network_update_interval == 0:
        agent.update_target_network()

    if step % save_model_interval == 0:
        agent.save(f"{model_name}_step_{step}")
    
    if epsilon > min_epsilon:
        epsilon -= epsilon_decay_rate

        if epsilon < min_epsilon:
            epsilon = min_epsilon

    if step % print_interval == 0 and \
       len(moves_survived_per_episode) >= eval_length and \
       len(apples_eaten_per_episode) >= eval_length and \
       len(losses) >= 5000:
        print("\n")
        print("-" * 50)
        print(f"Step: {step}")
        print(f"Epsilon: {epsilon}")
        print(f"Most moves survived: {np.max(moves_survived_per_episode)}")
        print(f"Average moves survived over last {eval_length} episodes: {np.mean(moves_survived_per_episode[-eval_length:])}")
        print(f"Most apples eaten: {np.max(apples_eaten_per_episode)}")
        print(f"Average apples eaten over last {eval_length} episodes: {np.mean(apples_eaten_per_episode[-eval_length:])}")
        print(f"Average loss over last 5,000 steps: {np.mean(losses[-5000:])}")
        print("-" * 50)
        print("\n")

    if step % save_graph_interval == 0:
        save_graphs(step)

    if done:
        environment.reset(for_training=True)
        moves_survived_per_episode.append(moves_survived)
        apples_eaten_per_episode.append(apples_eaten)

        steps_without_apple = 0
        moves_survived = 0
        apples_eaten = 0
    else:
        moves_survived += 1


agent.save(model_name)
save_graphs("done")
