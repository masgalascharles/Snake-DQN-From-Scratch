import numpy as np
import pygame
import sys
from src.agent import Agent
from src.environment import Environment


environment = Environment()
timeout = environment.size ** 2 // 2.75

"""
def compare_models(model_paths):
    for i in range(0, 2):
        if i == 0:
            print(f"\nTask #{i + 1}: starting state is always the default\n")
        elif i == 1:
            print(f"\nTask #{i + 1}: starting state is picked randomly\n")
        
        for model_path in model_paths:
            input_size = environment.get_state().shape[0] - (0 if model_paths.index(model_path) >= 3 else 1) # Models 1 through 3 don't include snake length in the input, so the input size for them is 1 less
            learning_rate = 0.001
            gamma = 0.99
            agent = Agent(input_size, learning_rate, gamma)
            
            agent.load(model_path)
            
            episodes = 10
            score = 0
            number_of_moves = 0

            for episode in range(1, episodes + 1):
                environment.reset(for_training=bool(i))
                
                done = False
                moves_without_apple = 0
                
                while not done:
                    state = environment.get_state()

                    if input_size == state.shape[0] - 1:
                        state = np.delete(state, 0)
                    
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

            print(f"{model_path} | Score: {score}, Total moves: {number_of_moves}, Apples eaten per 100 moves: {score / number_of_moves * 100}")


model_paths = [
    "models/model_1/model_1",
    "models/model_2/model_2",
    "models/model_3/model_3",
    "models/model_4/model_4",
    "models/model_4/model_41",
    "models/model_5/model_5",
    "models/model_6/model_6"
]

compare_models(model_paths)
"""

input_size = environment.get_state().shape[0]
learning_rate = 0.001
gamma = 0.99
agent = Agent(input_size, learning_rate, gamma)

agent.load("models/model_5/model_5")
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
    clock.tick(40)
