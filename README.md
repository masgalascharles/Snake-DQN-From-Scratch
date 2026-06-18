# Snake-DQN-From-Scratch
This is a Deep Q Network built from scratch and trained to play Snake, using only NumPy and CuPy.

Includes:
- Custom neural network
- Experience replay
- Target network
- GPU acceleration with CuPy
- Epsilon-greedy exploration

## Attempt #2
I discovered I could use CuPy to speed up the training process. CuPy is a very simple transition from NumPy, except it allows operations on the GPU for significantly faster results.

Network:

(1848, 512) → (512, 256) → (256, 64) → (64, 4)
Leaky ReLU activation

State representation:
- When the game is reset, the snake now starts with a random length from 2 to 221 (I got 221 from `environment.size ** 2 / 2`, so the snake could start at a length from 2 to half of the environment's area). It also starts in a random position, not just the center. This is intended to give the agent a wide variety of states to explore, preventing it from overfitting to the usual starting state.
- Flattened vector of the game display with shape (1, 1848)
- The new state size comes from `(environment.size ** 2 + environment.size) * state_vector_size`
- One hot encoded vectors now represent each of the things in the game:
  ```python
  empty_vector = np.array([0, 0, 0, 0])
  new_row_vector = np.array([1, 0, 0, 0])
  snake_head_vector = np.array([0, 1, 0, 0])
  snake_body_vector = np.array([0, 0, 1, 0])
  apple_vector = np.array([0, 0, 0, 1])
  ```

Rewards are now between -1 and 1 as an attempt to prevent training instability:
- +0.8 for apple
- -1 for death
- -0.001 for each step

Training:
I had the idea to let the agent explore random states at first, and then choose its moves to gain experience in later stages of the game.
- 250,000 total episodes
- 150,000 "exploration" episodes (higher epsilon)
- 100,000 "fine-tuning" episodes (constant epsilon of 0.05)

Hyperparameters:
- 0.001 learning rate
- 0.99 gamma
- 10,000 memory length
- 512 batch size
- 1,000 target network update interval
- 0.05 minimum epsilon
- 150 maximum moves per episode

RESULTS:
<img width="2565" height="1407" alt="loss_over_time" src="https://github.com/user-attachments/assets/ba2faf05-2e9b-48ce-86dd-bc14a047cec4" />
<img width="2552" height="1407" alt="moves_survived_per_episode_over_time" src="https://github.com/user-attachments/assets/ad912861-99f1-439c-819d-d588fb2553cf" />
<img width="2539" height="1407" alt="apples_eaten_per_episode_over_time" src="https://github.com/user-attachments/assets/d1f02e9a-5a55-4d58-95e7-522a75a78527" />
https://github.com/user-attachments/assets/badadc8a-e37a-45f5-a502-cac1be74d1be

The agent seemed to have learned that all moves are bad, which I suspect is because of the constant negative reward with no immediate positive, except when it might've randomly stumbled upon an apple during training. I'm unsure as to why during the end of training the agent was able to consistently survive for so long, but during testing it was killing itself quickly. I also think my state representation is a limiting factor. Even with my new row vector, it is likely hard for my agent to learn spatial relationships. My plan is to feed the network a state representation with more given information (like distance to apple) and modify my reward system to include positives for moving towards the apple.

## Attempt #1
Network:
(462, 256) → (256, 128) → (128, 64) → (64, 4)
Leaky ReLU activation

State representation:
- Flattened vector of the game display with shape (1, 462)
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
