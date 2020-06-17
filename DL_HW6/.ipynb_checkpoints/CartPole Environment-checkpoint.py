import gym

game = gym.make("CartPole-v0").env
obs = game.reset()  # Initial observations showing cart's status
print(obs)  # [Position, Velocity, Angle, Angular Velocity]
print(game.action_space)  # Possible actions
# In this case there are two possible actions
# Accelerating left (0) or right (1)

game_over = False
action = 1
while not game_over:
    image = game.render(mode='rgb_array')
    obs, reward, game_over, info = game.step(action)  # Tells the agent to take action (1)
    # obs is environment's new state after taking the action
    # reward of taking this action
    action = 0 if action == 1 else 1

game.close()
