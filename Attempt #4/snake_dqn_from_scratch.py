import numpy as np
import cupy as cp
import pygame
import sys
import matplotlib.pyplot as plt
import copy
import os
import json
from collections import deque


rng = np.random.default_rng()

moves = {
    0: np.array([0, -1]),
    1: np.array([0, 1]),
    2: np.array([-1, 0]),
    3: np.array([1, 0])
}


class Environment:
    def __init__(self):
        self.size = 21
        self.score = 0
        self.hamiltonian_path = []

        x = 1
        y = 0
        direction = 1

        while len(self.hamiltonian_path) < self.size ** 2:
            self.hamiltonian_path.append([x, y])

            x += direction

            if x < 1 or x >= self.size:
                x -= direction
                y += 1
                direction *= -1

        self.reset()

    def get_state(self):
        state = np.array([
            len(self.snake) / self.size ** 2,
            (self.apple[0] - self.snake[0][0]) / self.size,
            (self.apple[1] - self.snake[0][1]) / self.size,
            self.snake[0][1] / self.size,
            abs(self.snake[0][1] - self.size + 1) / self.size,
            self.snake[0][0] / self.size,
            abs(self.snake[0][0] - self.size + 1) / self.size
        ])
        surroundings = self.get_surroundings()

        return np.concatenate((state, surroundings))

    def reset(self, for_training=False):
        self.score = 0

        if for_training:
            self.randomize_snake_position()
        else:
            self.snake = np.array([[10, 10], [10, 11], [10, 12]])
        
        self.randomize_apple_position()

    def get_surroundings(self):
        danger_id = -1
        apple_id = 1
        directions = {
            0: np.array([0, -1]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([1, 0]),
            4: np.array([-1, -1]),
            5: np.array([1, -1]),
            6: np.array([-1, 1]),
            7: np.array([1, 1])
        }

        snake_head = self.snake[0].copy()
        surroundings = []

        for key in directions:
            temp = snake_head + directions[key]
            nearest = None
            nearest_x_distance = None
            nearest_y_distance = None

            while True:
                if temp[0] < 0 or temp[0] >= self.size or \
                   temp[1] < 0 or temp[1] >= self.size or \
                   np.any(np.all(self.snake[1:] == temp, axis=1)):
                    nearest = danger_id
                    nearest_x_distance = (temp[0] - snake_head[0]) / self.size
                    nearest_y_distance = (temp[1] - snake_head[1]) / self.size

                    break
                elif np.array_equal(temp, self.apple):
                    nearest = apple_id
                    nearest_x_distance = (temp[0] - snake_head[0]) / self.size
                    nearest_y_distance = (temp[1] - snake_head[1]) / self.size

                    break

                temp += directions[key]

            surroundings.append(nearest)
            surroundings.append(nearest_x_distance)
            surroundings.append(nearest_y_distance)

        return np.array(surroundings)

    def get_snake_head_distance_from_apple(self):
        x_distance = self.snake[0][0] - self.apple[0]
        y_distance = self.snake[0][1] - self.apple[1]

        return np.sqrt(x_distance ** 2 + y_distance ** 2)

    def randomize_snake_position(self):
        random_number = rng.integers(0, 3)

        if random_number == 0:
            starting_length = rng.integers(2, 40)
        elif random_number == 1:
            starting_length = rng.integers(41, 100)
        elif random_number == 2:
            starting_length = rng.integers(101, 200)

        max_start = len(self.hamiltonian_path) - starting_length
        start_index = rng.integers(0, max_start + 1)
        self.snake = np.array(self.hamiltonian_path[start_index:start_index + starting_length])

    def randomize_apple_position(self):
        self.apple = rng.integers(0, self.size, size=2)

        while np.any(np.all(self.snake[:] == self.apple, axis=1)):
            self.apple = rng.integers(0, self.size, size=2)

    def move(self, direction):
        new_head = self.snake[0] + moves[direction]
        self.snake = np.insert(self.snake, 0, new_head, axis=0)

        if new_head[0] < 0 or new_head[0] >= self.size or \
           new_head[1] < 0 or new_head[1] >= self.size or \
           np.any(np.all(self.snake[1:] == new_head, axis=1)):
            apple_eaten = False
            done = True
        elif np.array_equal(new_head, self.apple):
            self.score += 1
            
            self.randomize_apple_position()

            apple_eaten = True
            done = False
        else:
            self.snake = np.delete(self.snake, -1, axis=0)
            apple_eaten = False
            done = False

        return apple_eaten, done


class Layer:
    def __init__(self, number_of_inputs, number_of_neurons):
        self.weights = cp.random.normal(
            loc=0,
            scale=np.sqrt(2 / number_of_inputs),
            size=(number_of_neurons, number_of_inputs)
        )
        self.biases = cp.zeros((1, number_of_neurons))


class Agent:
    def __init__(self, input_size, learning_rate, gamma):
        self.network = [
            Layer(input_size, 512),
            Layer(512, 256),
            Layer(256, 64),
            Layer(64, 4)
        ]
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.update_target_network()

    def leaky_relu(self, x):
        return cp.maximum(0.01 * x, x)

    def d_leaky_relu(self, x):
        return cp.where(x > 0, 1, 0.01)

    def forward(self, x, using_target_network=False):
        if using_target_network:
            layers = self.target_network
        else:
            layers = self.network

        x = cp.asarray(x)
        z1 = x @ layers[0].weights.T + layers[0].biases
        a1 = self.leaky_relu(z1)
        z2 = a1 @ layers[1].weights.T + layers[1].biases
        a2 = self.leaky_relu(z2)
        z3 = a2 @ layers[2].weights.T + layers[2].biases
        a3 = self.leaky_relu(z3)
        z4 = a3 @ layers[3].weights.T + layers[3].biases

        return x, z1, a1, z2, a2, z3, a3, z4

    def predict(self, x, using_target_network=False):
        _, _, _, _, _, _, _, prediction = self.forward(x, using_target_network=using_target_network)

        return prediction.get()

    def train(self, memory, batch_size):
        indices = rng.choice(len(memory), size=batch_size, replace=False)

        state = np.array([memory[i][0] for i in indices])
        x, z1, a1, z2, a2, z3, a3, z4 = self.forward(state)
        direction = cp.array([memory[i][1] for i in indices])
        reward = cp.array([memory[i][2] for i in indices])
        next_state = cp.array([memory[i][3] for i in indices])
        done = cp.array([memory[i][4] for i in indices], dtype=cp.float64)

        next_q_values = cp.asarray(self.predict(next_state, using_target_network=True))
        max_next_q_value = cp.max(next_q_values, axis=1)
        target_q_value = reward + self.gamma * max_next_q_value * (1 - done)

        q_value = z4[cp.arange(batch_size), direction]
        
        error = cp.zeros_like(z4)
        error[cp.arange(batch_size), direction] = q_value - target_q_value

        delta4 = error
        dw4 = delta4.T @ a3 / batch_size
        db4 = cp.sum(delta4, axis=0, keepdims=True) / batch_size

        delta3 = delta4 @ self.network[3].weights * self.d_leaky_relu(z3)
        dw3 = delta3.T @ a2 / batch_size
        db3 = cp.sum(delta3, axis=0, keepdims=True) / batch_size

        delta2 = delta3 @ self.network[2].weights * self.d_leaky_relu(z2)
        dw2 = delta2.T @ a1 / batch_size
        db2 = cp.sum(delta2, axis=0, keepdims=True) / batch_size

        delta1 = delta2 @ self.network[1].weights * self.d_leaky_relu(z1)
        dw1 = delta1.T @ x / batch_size
        db1 = cp.sum(delta1, axis=0, keepdims=True) / batch_size

        self.network[3].weights -= dw4 * self.learning_rate
        self.network[3].biases -= db4 * self.learning_rate

        self.network[2].weights -= dw3 * self.learning_rate
        self.network[2].biases -= db3 * self.learning_rate

        self.network[1].weights -= dw2 * self.learning_rate
        self.network[1].biases -= db2 * self.learning_rate

        self.network[0].weights -= dw1 * self.learning_rate
        self.network[0].biases -= db1 * self.learning_rate

        return cp.mean(0.5 * (q_value - target_q_value) ** 2).item()

    def update_target_network(self):
        self.target_network = copy.deepcopy(self.network)

    def save(self, name):
        np.savez(
            f"{name}.npz",
             l0w=self.network[0].weights.get(),
             l0b=self.network[0].biases.get(),
             l1w=self.network[1].weights.get(),
             l1b=self.network[1].biases.get(),
             l2w=self.network[2].weights.get(),
             l2b=self.network[2].biases.get(),
             l3w=self.network[3].weights.get(),
             l3b=self.network[3].biases.get()
        )

    def load(self, name):
        model = np.load(f"{name}.npz")

        self.network[0].weights = cp.asarray(model["l0w"])
        self.network[0].biases  = cp.asarray(model["l0b"])
        self.network[1].weights = cp.asarray(model["l1w"])
        self.network[1].biases  = cp.asarray(model["l1b"])
        self.network[2].weights = cp.asarray(model["l2w"])
        self.network[2].biases  = cp.asarray(model["l2b"])
        self.network[3].weights = cp.asarray(model["l3w"])
        self.network[3].biases  = cp.asarray(model["l3b"])

        self.update_target_network()


environment = Environment()

steps = 5000000
timeout = environment.size ** 2 // 2.75
epsilon = 0.08
min_epsilon = 0.02
epsilon_decay_rate = (epsilon - min_epsilon) / (steps - (steps - 500000))
target_network_update_interval = 10000
print_interval = 5000
save_graph_interval = 200000
save_model_interval = 200000
moves_survived_per_episode = []
apples_eaten_per_episode = []
losses = []

model_name = "model_6"
input_size = environment.get_state().shape[0]
learning_rate = 0.001
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
    "timeout": timeout,
    
    "state_includes_snake_length": True,
    "starting_lengths_used": "33% 2-39, 41-99, 101-199",
    "other_notes": "Experiment to see what happens if I continued training model 5 for a long time."
}

os.makedirs(f"saved_runs/{model_name}", exist_ok=True)

with open(f"saved_runs/{model_name}/config.json", "w") as f:
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


##############
## TRAINING ##
##############

steps_without_apple = 0
moves_survived = 0
apples_eaten = 0

agent.load("saved_runs/model_5/model_5")
environment.reset(for_training=True)

for step in range(1, steps + 1):
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
       len(moves_survived_per_episode) >= 20 and \
       len(apples_eaten_per_episode) >= 20 and \
       len(losses) >= 5000:
        print("\n")
        print("-" * 50)
        print(f"Step: {step}")
        print(f"Epsilon: {epsilon}")
        print(f"Most moves survived: {np.max(moves_survived_per_episode)}")
        print(f"Average moves survived over last 20 episodes: {np.mean(moves_survived_per_episode[-20:])}")
        print(f"Most apples eaten: {np.max(apples_eaten_per_episode)}")
        print(f"Average apples eaten over last 20 episodes: {np.mean(apples_eaten_per_episode[-20:])}")
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


################
## EVALUATION ##
################
"""
def compare_models(model_paths):
    for i in range(0, 2):
        if i == 0:
            print(f"\nTask #{i + 1}: starting state is always the default\n")
        elif i == 1:
            print(f"\nTask #{i + 1}: starting state is picked randomly\n")
        
        for model_path in model_paths:
            input_size = environment.get_state().shape[0]
            learning_rate = 0.001
            gamma = 0.99
            agent = Agent(input_size, learning_rate, gamma)
            
            agent.load(model_path)
            
            episodes = 500
            score = 0
            number_of_moves = 0

            for episode in range(1, episodes + 1):
                environment.reset(for_training=bool(i))
                
                done = False
                moves_without_apple = 0
                
                while not done:
                    state = environment.get_state()
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

            print(f"{model_path} | Score: {score}, Total moves: {number_of_moves}")


model_paths = [
    "saved_runs/model_5/model_5",
    "model_6",
    "model_6_step_4000000",
    "model_6_step_3000000",
    "model_6_step_2000000",
    "model_6_step_1000000",
]

compare_models(model_paths)
"""

"""
pygame.init()
environment.reset()

width, height = 840, 840
cell = 40
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
done = False

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    if done:
        environment.reset()

        done = False

    screen.fill((0, 0, 0))

    font = pygame.font.SysFont("impact", 420)
    text_surface = font.render(f"{environment.score}", True, (255, 255, 255))
    text_rect = text_surface.get_rect()
    text_rect.center = (width / 2, height / 2)

    text_surface.set_alpha(120)
    screen.blit(text_surface, text_rect)

    snake = environment.snake
    apple = environment.apple

    pygame.draw.rect(screen, (0, 210, 0), (snake[0][0] * cell, snake[0][1] * cell, cell, cell))

    for x, y in snake[1:]:
        pygame.draw.rect(screen, (0, 255, 0), (x * cell, y * cell, cell, cell))

    pygame.draw.rect(screen, (255, 0, 0), (apple[0] * cell, apple[1] * cell, cell, cell))

    state = environment.get_state()
    prediction = agent.predict(state.reshape(1, -1))
    _, done = environment.move(np.argmax(prediction))

    pygame.display.flip()
    clock.tick(15)
"""
