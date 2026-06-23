# Snake-DQN-From-Scratch
This is a Deep Q Network built from scratch and trained to play Snake, using only NumPy and CuPy. The goal of this project was to learn how DQNs work at the lowest level by implementing everything manually rather than relying on deep learning frameworks.

Includes:
- Custom neural network
- Experience replay
- Target network
- GPU acceleration with CuPy
- Epsilon-greedy exploration

## Attempt #4
This attempt includes many small changes between similar models. Here is a brief summary of all of them:
- **Model 1** | Trained 500,000 steps. Training length is now in steps instead of episodes. This allows precise control of training length and is just cleaner overall. Batch size is now 128 to speed up training time. Rewards are +0.9 for apples, -1 for death, and -0.001 for each step. Maximum move limit has been removed, because I realized it was limiting the agent from exploring later game states. State representation improved so that the agent can easily see the geometric relationships between objects. The state representation is:
  ```python
  state = [
    x_distance_to_apple,
    y_distance_to_apple,
    y_distance_to_upper_wall,
    y_distance_to_lower_wall,
    x_distance_to_left_wall,
    x_distance_to_right_wall,
    first_object_type,
    object_x_distance,
    object_y_distance,
    ...objects gotten and added to the state 8 times, gotten by a ray cast in each direction from the snake's head. Object types are -1 for any danger (wall, snake body) and 1 for an apple.
  }
  ```
- **Model 2** | Exact same setup as model 1, but trained for 1,000,000 steps instead of 500,000. Also increased memory length from 100,000 to 300,000. This produced significantly better results. Lacked late game skills.
- **Model 3** | Continued training of model 2, but only bigger snakes were used as the starting length. This was an attempt to improve late game performance. Trained for 600,000 steps.
- **Model 4** | Added snake length to the beginning of the state representation. I also started preventing the snake tail from starting outside of the game area by using a modified Hamiltonian path. I generate a zig-zag like path through the game area, but exclude the first column to make sure the snake has area to escape and can't get trapped from the start. Trained for 600,000 steps.
- **Model 41** | Changed the way starting lengths are randomly selected. Now, there is a 1/3 chance to get an early game snake, middle game snake, or late game snake. I started training with model 4, and trained for another 1,000,000 steps.
- **Model 5** | Continued training of model 41, but I added a stalling penalty of -1 if the agent doesn't eat an apple for `environment.size ** 2 // 2.75` steps. Trained for 1,000,000 steps.

## Attempt #3
**Network:**\
(892, 512) → (512, 256) → (256, 64) → (64, 4)\
Leaky ReLU activation

**Significant changes:**
- State representation
- Reward system

**State representation:**
- When the game is reset, the snake now starts with a random length between 2 and 40, instead of 2 and 221. This change was made to give the agent exposure to more early game states. It still starts in a random position.
- Flattened vector of the game display with shape (1, 892). The new state size comes from `environment.size ** 2 * 2 + 10`.
- The new states look like this:
  ```python
  state = [
    # The following features are normalized by dividing by environment.size
    the x distance between the snake head and the apple,
    the y distance between the snake head and the apple,
    the distance between the snake head and the upper wall,
    the distance between the snake head and the lower wall,
    the distance between the snake head and the left wall,
    the distance between the snake head and the right wall,
    
    what is up,       # These 4 values can
    what is down,     # be an integer from -1 to 1,
    what is left,     # where -1 = danger, 0 = empty,
    what is right,    # and 1 = apple
    
    the rest of the state is the coordinates for each part of the snake, # This is also normalized by dividing by environment.size
    unused indices are filled with padding, which is -2
  ]
  ```
  The goal of this state representation was to make it easier for the agent to get the information it needs to make decisions. The previous one-hot encoded game state required the network to learn geometric relationships from a very large, flattened grid. Now, it is given these relationships, like the distance to the apple and the distances to the walls.

**Rewards:**
- -1 for death
- +0.8 for apple
- -0.005 if agent moves away from the apple, but +0.004 if the agent moves towards the apple

**Training:**\
I lowered the total number of episodes, because based on the graphs of the previous attempt, it doesn't need 250,000.
- 200,000 total episodes
- 100,000 "exploration" episodes (higher epsilon)
- 100,000 "fine-tuning" episodes (constant epsilon of 0.05)

**Hyperparameters:**
- 0.001 learning rate
- 0.99 gamma
- 10,000 memory length
- 512 batch size
- 1,000 target network update interval
- 0.05 minimum epsilon
- 100 maximum moves per episode

**RESULTS:**
<img width="2539" height="1407" alt="loss_over_time_episode_200000" src="https://github.com/user-attachments/assets/728bf8c6-3885-4632-91bf-83a94b4377ca" />
<img width="2552" height="1407" alt="moves_survived_per_episode_over_time_episode_200000" src="https://github.com/user-attachments/assets/e58003c8-9e2f-4bad-9dc6-20e44af48ad0" />
<img width="2499" height="1407" alt="apples_eaten_per_episode_over_time_episode_200000" src="https://github.com/user-attachments/assets/0eb198a8-9336-48e3-9cce-7feced7bdd5f" />
https://github.com/user-attachments/assets/52707c94-4108-4a8f-b885-5705b23490be

While the graphs made it look like this run had potential, during evaluation, the agent might eat a few apples but then it goes in an infinite loop. I think this is caused by my new reward system of giving a reward for getting closer to the apple. It seems like it learned that going in circles and getting that small reward is better than risking going after the apples and dying. One improvement though, is that unlike the last attempt, this agent doesn't try to kill itself and actually learned to stay alive very well.

## Attempt #2
I discovered I could use CuPy to speed up the training process. CuPy is a very simple transition from NumPy, except it allows operations on the GPU for significantly faster results.

**Significant changes:**
- Target network
- Experience replay
- State representation
- GPU acceleration
- Normalized rewards

**Network:**\
(1848, 512) → (512, 256) → (256, 64) → (64, 4)\
Leaky ReLU activation

**State representation:**
- When the game is reset, the snake now starts with a random length from 2 to 221 (I got 221 from `environment.size ** 2 / 2`, so the snake could start at a length from 2 to half of the environment's area). It also starts in a random position, not just the center. This is intended to give the agent a wide variety of states to explore, preventing it from overfitting to the usual starting state.
- Vector of the game display with shape (1, 1848). The new state size comes from `(environment.size ** 2 + environment.size) * state_vector_size`.
- One hot encoded vectors now represent each of the things in the game:
  ```python
  empty_vector = np.array([0, 0, 0, 0])
  new_row_vector = np.array([1, 0, 0, 0])
  snake_head_vector = np.array([0, 1, 0, 0])
  snake_body_vector = np.array([0, 0, 1, 0])
  apple_vector = np.array([0, 0, 0, 1])
  ```

**Rewards:**\
Now between -1 and 1 as an attempt to prevent training instability.
- +0.8 for apple
- -1 for death
- -0.001 for each step

**Training:**\
I had the idea to let the agent explore random states at first, and then choose its moves to gain experience in later stages of the game.
- 250,000 total episodes
- 150,000 "exploration" episodes (higher epsilon)
- 100,000 "fine-tuning" episodes (constant epsilon of 0.05)

**Hyperparameters:**
- 0.001 learning rate
- 0.99 gamma
- 10,000 memory length
- 512 batch size
- 1,000 target network update interval
- 0.05 minimum epsilon
- 150 maximum moves per episode

**RESULTS:**
<img width="2565" height="1407" alt="loss_over_time" src="https://github.com/user-attachments/assets/ba2faf05-2e9b-48ce-86dd-bc14a047cec4" />
<img width="2552" height="1407" alt="moves_survived_per_episode_over_time" src="https://github.com/user-attachments/assets/ad912861-99f1-439c-819d-d588fb2553cf" />
<img width="2539" height="1407" alt="apples_eaten_per_episode_over_time" src="https://github.com/user-attachments/assets/d1f02e9a-5a55-4d58-95e7-522a75a78527" />
https://github.com/user-attachments/assets/badadc8a-e37a-45f5-a502-cac1be74d1be

The agent seemed to have learned that all moves are bad, which I suspect is because of the constant negative reward with no immediate positive, except when it might've randomly stumbled upon an apple during training. I'm unsure as to why during the end of training the agent was able to consistently survive for so long, but during testing it was killing itself quickly. I also think my state representation is a limiting factor. Even with my new row vector, it is likely hard for my agent to learn spatial relationships. My plan is to feed the network a state representation with more given information (like distance from the snake head to the apple) and modify my reward system to include rewards for moving towards the apple.

## Attempt #1
Network:
(462, 256) → (256, 128) → (128, 64) → (64, 4)
Leaky ReLU activation

State representation:
- Vector of the game display with shape (1, 462)
- Token added at the start of each row to keep them separated, even in a flattened vector
- 0 is an empty space, 1 is a new row, 2 is the snake head, 3 is part of the snake body, 4 is an apple

Rewards:
- +5 for apple
- -10 for death
- -0.01 for each step to prevent stalling

Training:
- 150,000 episodes
- 0.01 learning rate
- 0.95 gamma
- 0.1 minimum epsilon

For the first attempt, I wanted to see what would happen without a target network or experience replay. This resulted in the network being severely unstable, with the loss exploding to infinity. I also suspected this could be because of my rewards and state representation, in addition to having no target network or experience replay. But even after I fixed the exploding loss by experimenting with the rewards a bit, the loss appeared to never drop and it remained high throughout training. This proved the reward system wasn't the only issue.
