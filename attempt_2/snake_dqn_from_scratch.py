import numpy as np
import cupy as cp
import pygame
import sys
import matplotlib.pyplot as plt
import copy
from collections import deque


rng = np.random.default_rng()

state_vector_size = 4
empty_vector = np.array([0, 0, 0, 0])
new_row_vector = np.array([1, 0, 0, 0])
snake_head_vector = np.array([0, 1, 0, 0])
snake_body_vector = np.array([0, 0, 1, 0])
apple_vector = np.array([0, 0, 0, 1])

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

        self.reset()

    def get_state(self):
        state = np.full((self.size, self.size + 1, state_vector_size), empty_vector)
        
        for y in range(0, len(state)):
            for x in range(0, len(state[y])):
                cell_position = np.array([x, y])

                if x == 0:
                    state[y][x] = new_row_vector
                elif np.array_equal(self.snake[0], cell_position):
                    state[y][x] = snake_head_vector
                elif np.any(np.all(self.snake[1:] == cell_position, axis=1)):
                    state[y][x] = snake_body_vector
                elif np.array_equal(self.apple, cell_position):
                    state[y][x] = apple_vector

        return state.reshape(-1)

    def reset(self, for_training=False):
        self.score = 0

        if for_training:
            self.randomize_snake_position()
        else:
            self.snake = np.array([[10, 10], [10, 11], [10, 12]])
        
        self.randomize_apple_position()

    def randomize_snake_position(self):
        starting_x = rng.integers(0, self.size)
        starting_y = rng.integers(0, self.size)
        starting_length = rng.integers(2, self.size ** 2 / 2)
        self.snake = np.zeros((starting_length, 2))
        self.snake[:, 0] = starting_x
        self.snake[:, 1] = np.arange(starting_y, starting_y + starting_length)

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

    def save(self):
        np.savez(
            "model.npz",
             l0w=self.network[0].weights.get(),
             l0b=self.network[0].biases.get(),
             l1w=self.network[1].weights.get(),
             l1b=self.network[1].biases.get(),
             l2w=self.network[2].weights.get(),
             l2b=self.network[2].biases.get(),
             l3w=self.network[3].weights.get(),
             l3b=self.network[3].biases.get()
        )

    def load(self):
        model = np.load("model.npz")

        self.network[0].weights = cp.asarray(model["l0w"])
        self.network[0].biases  = cp.asarray(model["l0b"])
        self.network[1].weights = cp.asarray(model["l1w"])
        self.network[1].biases  = cp.asarray(model["l1b"])
        self.network[2].weights = cp.asarray(model["l2w"])
        self.network[2].biases  = cp.asarray(model["l2b"])
        self.network[3].weights = cp.asarray(model["l3w"])
        self.network[3].biases  = cp.asarray(model["l3b"])


environment = Environment()

episodes = 250000
epsilon = 1
min_epsilon = 0.05
epsilon_decay_rate = (epsilon - min_epsilon) / (episodes - 100000)
target_network_update_interval = 1000
print_interval = 250
save_graph_interval = 25000
moves_survived_per_episode = []
apples_eaten_per_episode = []
losses = []

input_size = environment.get_state().shape[0]
learning_rate = 0.001
gamma = 0.99
agent = Agent(input_size, learning_rate, gamma)
memory = deque(maxlen=10000)
batch_size = 512


##############
## TRAINING ##
##############

for episode in range(1, episodes + 1):
    environment.reset(for_training=True)

    moves_survived = 0
    apples_eaten = 0
    episode_losses = []
    done = False
    
    while not done and moves_survived < 150:
        state = environment.get_state()
        prediction = agent.predict(state.reshape(1, -1))

        if rng.random() >= epsilon:
            direction = np.argmax(prediction)
        else:
            direction = rng.integers(0, 4)

        apple_eaten, done = environment.move(direction)
        reward = -0.001

        if done:
            reward = -1
        elif apple_eaten:
            reward = 0.8
            apples_eaten += 1

        next_state = environment.get_state()

        memory.append([state, direction, reward, next_state, done])

        if len(memory) > batch_size:
            loss = agent.train(memory, batch_size)

            episode_losses.append(loss)
        
        moves_survived += 1

    moves_survived_per_episode.append(moves_survived)
    apples_eaten_per_episode.append(apples_eaten)

    if len(episode_losses) > 0:
        losses.append(np.mean(episode_losses))
    
    if epsilon > min_epsilon:
        epsilon -= epsilon_decay_rate

        if epsilon < min_epsilon:
            epsilon = min_epsilon

    if episode % target_network_update_interval == 0:
        agent.update_target_network()

    if episode % print_interval == 0:
        print("\n")
        print("-" * 50)
        print(f"Episode: {episode}")
        print(f"Epsilon: {epsilon}")
        print(f"Most moves survived: {np.max(moves_survived_per_episode)}")
        print(f"Average moves survived per episode: {np.mean(moves_survived_per_episode)}")
        print(f"Most apples eaten: {np.max(apples_eaten_per_episode)}")
        print(f"Average apples eaten per episode: {np.mean(apples_eaten_per_episode)}")
        print(f"Average loss: {np.mean(losses)}")
        print("-" * 50)
        print("\n")

    if episode % save_graph_interval == 0:
        plt.figure(figsize=(10, 5))
        plt.plot(moves_survived_per_episode)
        plt.xlabel("Episode")
        plt.ylabel("Moves Survived")
        plt.title("Moves Survived Per Episode Over Time")
        plt.grid(True)
        plt.savefig(f"moves_survived_per_episode_over_time_episode_{episode}.png", dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(apples_eaten_per_episode)
        plt.xlabel("Episode")
        plt.ylabel("Apples Eaten")
        plt.title("Apples Eaten Per Episode Over Time")
        plt.grid(True)
        plt.savefig(f"apples_eaten_per_episode_over_time_episode_{episode}.png", dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.title("Loss Over Time")
        plt.grid(True)
        plt.savefig(f"loss_over_time_episode_{episode}.png", dpi=300, bbox_inches="tight")
        plt.close()


agent.save()


plt.figure(figsize=(10, 5))
plt.plot(moves_survived_per_episode)
plt.xlabel("Episode")
plt.ylabel("Moves Survived")
plt.title("Moves Survived Per Episode Over Time")
plt.grid(True)
plt.savefig("moves_survived_per_episode_over_time.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(apples_eaten_per_episode)
plt.xlabel("Episode")
plt.ylabel("Apples Eaten")
plt.title("Apples Eaten Per Episode Over Time")
plt.grid(True)
plt.savefig("apples_eaten_per_episode_over_time.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.title("Loss Over Time")
plt.grid(True)
plt.savefig("loss_over_time.png", dpi=300, bbox_inches="tight")
plt.close()

################
## EVALUATION ##
################
"""
pygame.init()

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
    clock.tick(2)
"""
