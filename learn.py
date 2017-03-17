import numpy as np
import tensorflow as tf
import perceptron
import matplotlib.pyplot as plt
import gym
import sys
import subprocess

# Prevent sleeping
if 'darwin' in sys.platform:
    print('Running \'caffeinate\' on MacOSX to prevent the system from sleeping')
    subprocess.Popen('caffeinate')

# Import CartPole environment
env = gym.make('CartPole-v0')

# Start with a clean slate
tf.reset_default_graph()

# Parameters
# Needs to be large enough to make progress, but small enough for stability
learning_rate = 0.005
num_episodes = 1501
# Encourage exploring initially
initial_eps = 0.5
# Needs to be large to make progress
discount_factor = 0.99
batch_size = 100
# Need big enough buffer for IID, but not so big that it triggers catastrophic forgetting
buffer_size = 500
log_freq = 10
visualisation_freq = 10
visualise = False
num_runs = 12
# Don't set this too high, or else outliers skew the success
max_episode_length = 200
target_performance = 195

# Network Parameters
# Needs high enough capacity, but small enough to permit efficient fitting
n_hidden_1 = 30 # 1st layer number of nodes
n_hidden_2 = 30 # 2nd layer number of nodes
n_input = 4 # input layer (state)
n_output = 2 # output layer (q values for actions)

# Create computation graph

model = perceptron.MultilayerPerceptron(n_input, n_hidden_1, n_hidden_2, n_output)
inputs = tf.placeholder(shape=[1, n_input], dtype=tf.float32)

current_output, reg = model.evaluate(inputs)
target_output = tf.placeholder(shape=[1, n_output],dtype=tf.float32)

loss = tf.reduce_sum(tf.square(target_output - current_output)) + reg
trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)

minimize = trainer.minimize(loss)

init = tf.initialize_all_variables()

rewardToPlot = []

for alpha in range(num_runs):
    print("Run number {} of {}".format(str(alpha) + 1, str(num_runs)))
    # Create lists to contain total rewards per episode
    rList = []
    rAggList = []

    # Stores tuples of (S, A, R, S')
    # For experience replay
    replay = []
    h = 0
    eps = initial_eps

    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_episodes):
            # Anneal epsilon
            eps = max(eps - (initial_eps / 500 ), 0.05)
            # Reset environment and get first new observation
            s = env.reset()
            rTotal = 0
            lossTotal = 0
            j = 0
            # The q-network
            while j < max_episode_length:
                if i % visualisation_freq == 0 and visualise:
                    env.render()
                j+=1
                # Choose an action greedily from the Q-network
                q = sess.run(current_output, feed_dict={inputs: [s]})[0]
                a = np.argmax(q)
                if np.random.rand(1) < eps:
                    a = env.action_space.sample()
                # Get new state and reward from environment
                sPrime, r, d, _ = env.step(a)
                if d:
                    # Nowhere to go in terminal state
                    sPrime = None
                # If buffer not filled, add to it
                if len(replay) < buffer_size:
                    replay.append((s, a, r, sPrime))
                # Buffer is full - overwrite values
                else:
                    if h < (buffer_size - 1):
                        h += 1
                    else:
                        h = 0
                    replay[h] = (s, a, r, sPrime)
                # Prepare for next iteration
                if d:
                    break
                s = sPrime
                rTotal += r
            # Update only once per episode for stability
            if batch_size < len(replay):
                # Randomly sample our experience replay memory
                minibatch_indices = np.random.choice(range(len(replay)), batch_size)
                for index in minibatch_indices:
                    s_mem, a_mem, r_mem, sPrime_mem = replay[index]
                    target_q = sess.run(current_output, feed_dict={inputs: [s_mem]})[0]
                    # Expected return in the terminal state is known to be 0
                    max_q_prime = 0
                    if sPrime_mem is not None:
                        # Obtain the Q' values by feeding the new state through our network
                        q_prime = sess.run(current_output, feed_dict={inputs: [sPrime_mem]})[0]
                        # Obtain max Q' and set our target value for chosen action
                        max_q_prime = np.max(q_prime)
                    # Q-learning
                    target_q[a_mem] = r_mem + discount_factor * max_q_prime
                    sess.run(minimize, feed_dict={inputs: [s_mem], target_output: [target_q]})
            rList.append(rTotal)
            if i % log_freq == 0:
                rAggList.append(np.mean(rList[-log_freq:]))
                print("Average reward after " + str(i) + " episodes = " + str(np.mean(rList[-log_freq:])))
            if i > 100 and np.mean(rList[-100:]) > target_performance:
                print("Mission complete after {} episodes! The average over the last 100 was {}.".\
                    format(i, np.mean(rList[-100:])))
                s = env.reset()
                j = 0
                while j < max_episode_length:
                    if visualise:
                        env.render()
                    j += 1
                    if j == max_episode_length:
                        print("Pole did not fall in first {} steps".format(max_episode_length))
                    # Choose an action greedily from the Q-network
                    q = sess.run(current_output, feed_dict={inputs: [s]})[0]
                    a = np.argmax(q)
                    # Get new state and reward from environment
                    sPrime, r, d, _ = env.step(a)
                    if d:
                        print("Pole fell on step {}".format(j))
                        break
                    s = sPrime
                break
            if i == num_episodes - 1:
                print("*** Mission failed! ***")
                s = env.reset()
                j = 0
                while j < max_episode_length:
                    if visualise:
                        env.render()
                    j += 1
                    if j == max_episode_length:
                        print("Pole did not fall in first {} steps".format(max_episode_length))
                    # Choose an action greedily from the Q-network
                    q = sess.run(current_output, feed_dict={inputs: [s]})[0]
                    a = np.argmax(q)
                    # Get new state and reward from environment
                    sPrime, r, d, _ = env.step(a)
                    if d:
                        print("Pole fell on step {}".format(j))
                        break
                    s = sPrime
    rewardToPlot.append(rAggList)

# Thanks http://stackoverflow.com/questions/18296755/python-max-function-using-key-and-lambda-expression
plt.plot([np.mean([x[i] for x in rewardToPlot if len(x) > i]) for i in range(len(max(rewardToPlot, key=len)))])
plt.savefig( 'reward.png' )

plt.clf()