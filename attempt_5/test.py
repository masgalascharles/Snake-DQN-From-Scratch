import numpy as np
import pygame
import sys
import time
from src.agent import Agent
from src.environment import Environment


environment = Environment()
input_size = environment.get_state().shape[0]
learning_rate = 0.001
gamma = 0.99
agent = Agent(input_size, learning_rate, gamma)

agent.load("models/model_11/model_11_step_1800000")
environment.reset()
pygame.init()

width, height = 840, 840
cell = 40
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
done = False
tick_speed = 25
min_tick_speed = 1
max_tick_speed = 50


def draw(done=False):
    screen.fill((0, 0, 0))

    font = pygame.font.SysFont("impact", 420)
    text_surface = font.render(f"{environment.score}", True, (255, 255, 255))
    text_rect = text_surface.get_rect()
    text_rect.center = (width / 2, height / 2)

    text_surface.set_alpha(120)
    screen.blit(text_surface, text_rect)

    snake = environment.snake
    apple = environment.apple

    pygame.draw.rect(screen, (255, 0, 0), (apple[0] * cell, apple[1] * cell, cell, cell))

    for x, y in snake[1:]:
        pygame.draw.rect(screen, (0, 255, 0), (x * cell, y * cell, cell, cell))
    
    pygame.draw.rect(screen, ((0, 210, 0) if not done else (255, 0, 0)), (snake[0][0] * cell, snake[0][1] * cell, cell, cell))

    font = pygame.font.SysFont("impact", 15)
    text_surface = font.render(f"tick_speed: {tick_speed}. Use up/down arrows to adjust. Min {min_tick_speed}, max {max_tick_speed}.", True, (255, 255, 255))

    screen.blit(text_surface, (0, 0))


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and tick_speed < max_tick_speed:
                tick_speed += 1
            elif event.key == pygame.K_DOWN and tick_speed > min_tick_speed:
                tick_speed -= 1

    draw()

    state = environment.get_state()
    prediction = agent.predict(state.reshape(1, -1))
    _, done = environment.move(np.argmax(prediction))

    if done:
        draw(done=True)
        pygame.display.flip()
        time.sleep(10)
        environment.reset()

        done = False

        continue

    pygame.display.flip()
    clock.tick(tick_speed)
