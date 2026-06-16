# Snake-DQN-From-Scratch
I've been studying neural networks in my free time, so I figured the best way to test my understanding was to build one without PyTorch, TensorFlow, or any other deep learning framework.


## Attempt #1
Network:
(462, 256)
Leaky ReLU
(256, 128)
Leaky ReLU
(128, 64)
Leaky ReLU
(64, 4)

State representation:
- Flatten vector of the game display with shape (1, 462)
- Token added at the start of each row to keep them separated, even in a flattened vector

Rewards:
- +5 for apple
- -10 for death
- -0.01 for each step to prevent stalling

Training:
- 150,000 episodes
- 0.01 learning rate
- 0.95 gamma
- 0.1 minimum epsilon

For the first attempt, I wanted to see what would happen without a target network or experience replay.
This resulted in the network being severely unstable, with the loss exploding to infinity.
