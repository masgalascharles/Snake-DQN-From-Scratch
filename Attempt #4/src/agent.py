import numpy as np
import cupy as cp
import copy


rng = np.random.default_rng()


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
