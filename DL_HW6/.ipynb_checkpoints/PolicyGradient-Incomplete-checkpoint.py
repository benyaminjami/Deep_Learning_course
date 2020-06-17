import tensorflow as tf
import gym


# You must add few lines of code and change all -1s

class Agent:
    def __init__(self, learning_rate):
        # Build the network to predict the correct action
        tf.reset_default_graph()
        input_dimension = 4
        hidden_dimension = -1
        self.input = tf.placeholder(dtype=tf.float32, shape=[1, input_dimension], name='X')
        hidden_layer = -1
        logits = -1

        # Sample an action according to network's output
        # use tf.multinomial and sample one action from network's output
        self.action = -1

        # Optimization according to policy gradient algorithm
        cross_entropy = -1
        self.optimizer = -1  # use one of tensorflow optimizers
        grads_vars = self.optimizer.compute_gradients(
            cross_entropy)  # gradient of current action w.r.t. network's variables
        self.gradients = [grad for grad, var in grads_vars]

        # get rewards from the environment and evaluate rewarded gradients
        #  and feed it to agent and then call train operation
        self.rewarded_grads_placeholders_list = []
        rewarded_grads_and_vars = []
        for grad, var in grads_vars:
            rewarded_grad_placeholder = tf.placeholder(dtype=tf.float32, shape=grad.shape)
            self.rewarded_grads_placeholders_list.append(rewarded_grad_placeholder)
            rewarded_grads_and_vars.append((rewarded_grad_placeholder, var))

        self.train_operation = self.optimizer.apply_gradients(rewarded_grads_and_vars)

        self.saver = tf.train.Saver()

        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )

        self.ses = tf.Session(config=config)
        self.ses.run(tf.global_variables_initializer())

    def get_action_and_gradients(self, obs):
        # compute network's action and gradients given the observations
        return action, gradients

    def train(self, rewarded_gradients):
        feed_dict = {}
        # feed gradients into the placeholder and call train operation

    def save(self):
        self.saver.save(self.ses, "SavedModel/")

    def load(self):
        self.saver.restore(self.ses, "SavedModel/")


epochs = -1
max_steps_per_game = 1000
games_per_epoch = -1
discount_factor = -1
learning_rate = 0.01

agent = Agent(learning_rate)
game = gym.make("CartPole-v0").env
for epoch in range(epochs):
    epoch_rewards = []
    epoch_gradients = []
    epoch_average_reward = 0
    for episode in range(games_per_epoch):
        obs = game.reset()
        step = 0
        single_episode_rewards = []
        single_episode_gradients = []
        game_over = False
        while not game_over and step < max_steps_per_game:
            step += 1
            # image = game.render(mode='rgb_array') # Call this to render game and show visual
            action, gradients = agent.get_action_and_gradients(obs)
            obs, reward, game_over, info = game.step(action)
            single_episode_rewards.append(reward)
            single_episode_gradients.append(gradients)

        epoch_rewards.append(single_episode_rewards)
        epoch_gradients.append(single_episode_gradients)
        epoch_average_reward += sum(single_episode_rewards)

    epoch_average_reward /= games_per_epoch
    print("Epoch = {}, , Average reward = {}".format(epoch, epoch_average_reward))

    normalized_rewards = -1
    mean_rewarded_gradients = -1

    agent.train(mean_rewarded_gradients)

agent.save()
game.close()

# Run this part after training the network
# agent = Agent(0)
# game = gym.make("CartPole-v0").env
# agent.load()
# score = 0
# for i in range(10):
#     obs = game.reset()
#     game_over = False
#     while not game_over:
#         score += 1
#         image = game.render(mode='rgb_array')  # Call this to render game and show visual
#         action, _ = agent.get_action_and_gradients(obs)
#         obs, reward, game_over, info = game.step(action)
#
# print("Average Score = ", score / 10)
