import numpy as np


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
