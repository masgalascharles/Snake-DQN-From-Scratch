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

        while len(self.hamiltonian_path) < self.size ** 2 - self.size:
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
            (self.apple[0] - self.snake[0, 0]) / self.size,
            (self.apple[1] - self.snake[0, 1]) / self.size,
            self.snake[0, 1] / self.size,
            abs(self.snake[0, 1] - self.size + 1) / self.size,
            self.snake[0, 0] / self.size,
            abs(self.snake[0, 0] - self.size + 1) / self.size
        ])
        surroundings = self.get_surroundings()
        available_space, can_reach_tail = self.get_available_space()
        available_space /= self.size ** 2

        return np.concatenate((state, surroundings, available_space, can_reach_tail))

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

    def get_available_space(self):
        available_space_after_each_move = []
        can_reach_tail_after_each_move = []
        base_grid = np.zeros((self.size, self.size), dtype=bool)
        
        for snake_body in self.snake[:-1]:
            if snake_body[0] >= 0 and snake_body[0] < self.size and \
               snake_body[1] >= 0 and snake_body[1] < self.size:
                base_grid[snake_body[0], snake_body[1]] = True

        for move in moves.values():
            new_head = self.snake[0] + move
            new_snake = np.insert(self.snake, 0, new_head, axis=0)

            if new_head[0] < 0 or new_head[0] >= self.size or \
               new_head[1] < 0 or new_head[1] >= self.size:
                available_space_after_each_move.append(0)
                can_reach_tail_after_each_move.append(0)
                
                continue
            
            grid = base_grid.copy()

            if np.array_equal(new_head, self.apple):
                tail_tip = tuple(self.snake[-1])
                grid[tail_tip[0], tail_tip[1]] = True
            else:
                new_snake = np.delete(new_snake, -1, axis=0)
                tail_tip = tuple(new_snake[-1])

            if grid[new_head[0], new_head[1]]:
                available_space_after_each_move.append(0)
                can_reach_tail_after_each_move.append(0)
                
                continue
            
            grid[new_head[0], new_head[1]] = True
            queue = [tuple(new_head)]
            head_index = 0
            available_space = 0
            can_reach_tail = 0
            
            while head_index < len(queue):
                x, y = queue[head_index]
                head_index += 1
                
                for bfs_move in moves.values():
                    next_x = x + bfs_move[0]
                    next_y = y + bfs_move[1]
                    
                    if next_x >= 0 and next_x < self.size and \
                       next_y >= 0 and next_y < self.size:
                        if not grid[next_x, next_y]:
                            grid[next_x, next_y] = True
                            available_space += 1
                            
                            queue.append((next_x, next_y))
                        elif can_reach_tail == 0 and tail_tip == (next_x, next_y):
                            can_reach_tail = 1
                            
            available_space_after_each_move.append(available_space)
            can_reach_tail_after_each_move.append(can_reach_tail)

        return (np.array(available_space_after_each_move, dtype=np.float64), np.array(can_reach_tail_after_each_move))

    def get_snake_head_distance_from_apple(self):
        x_distance = self.snake[0, 0] - self.apple[0]
        y_distance = self.snake[0, 1] - self.apple[1]

        return np.sqrt(x_distance ** 2 + y_distance ** 2)

    def randomize_snake_position(self):
        random_number = rng.integers(0, 3)

        if random_number == 0:
            starting_length = rng.integers(2, 76)
        elif random_number == 1:
            starting_length = rng.integers(76, 151)
        else:
            starting_length = rng.integers(151, 301)

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
        done = False
        
        if np.array_equal(new_head, self.apple):
            self.score += 1
            apple_eaten = True
            
            self.randomize_apple_position()
        else:
            apple_eaten = False
            self.snake = np.delete(self.snake, -1, axis=0)

        if new_head[0] < 0 or new_head[0] >= self.size or \
           new_head[1] < 0 or new_head[1] >= self.size or \
           np.any(np.all(self.snake[1:] == new_head, axis=1)):
            apple_eaten = False
            done = True

        return apple_eaten, done
