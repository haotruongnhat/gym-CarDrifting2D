import gym
import gym_Drifting2D
import random
import gym_CarRacing2D

import numpy as np

env = gym.make("CarDrifting2D-v0")

# env = gym.make("CarDrifting2D-v0", drag=0.9, power=1, turnSpeed=0.04, angularDrag=0.6, multiInputs=False, showGates=False, constantAccel=False)
# Parameter Definitions:
# Drag, how much the car skids, the higher the more skid
# power, how fast the car accelerates
# turnSpeed, how fast the car turns
# angularDrag, how much the car spins, the higher the more spin
# Multi Inputs means the agent can go both forward/backward AND left or right simultaneously
# Show Gates is to show the reward gates
# constant accel is to accelerate constantly
import pygame
a = np.array([0.0, 0.0, 0.0])

def register_input():
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                a[0] = +1.0
            if event.key == pygame.K_RIGHT:
                a[0] = -1.0
            if event.key == pygame.K_UP:
                a[1] = +1.0
            if event.key == pygame.K_DOWN:
                a[2] = +0.8
            if event.key == pygame.K_RETURN:
                global restart
                restart = True

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                a[0] = 0
            if event.key == pygame.K_RIGHT:
                a[0] = 0
            if event.key == pygame.K_UP:
                a[1] = 0
            if event.key == pygame.K_DOWN:
                a[2] = 0

env.render()

isopen = True
while isopen:
    env.reset()
    total_reward = 0.0
    steps = 0
    restart = False
    while True:
        register_input()
        s, r, done, info = env.step(a)
        total_reward += r
        if steps % 200 == 0 or done:
            print("\naction " + str([f"{x:+0.2f}" for x in a]))
            print(f"step {steps} total_reward {total_reward:+0.2f}")
        steps += 1
        isopen = env.render()
        if done or restart or isopen is False:
            break
env.close()
