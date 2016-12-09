import numpy as np
import tensorflow as tf
import env as environment
import perceptron
import lstm_network
import matplotlib.pyplot as plt
import random
import copy
import gym


def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))

# env = environment.ShoprEnv()
env = gym.make('CartPole-v0')
reset = tf.reset_default_graph()

# Parameters
learning_rate = 0.2
num_episodes = 501
initial_eps = 0.5
discount = 0.9
batch_size = 5
buffer = 30
log_freq = 10
num_runs = 1

# Network Parameters
n_hidden_1 = 20 # 1st layer number of nodes
n_hidden_2 = 20 # 2nd layer number of nodes
n_input = 4 # input layer (basket state)
n_output = 5 # output layer (q values for actions)

# TO DO | implement more occasional updating of weights
#       | check this is okay with OpenAI gym
#       | work out what's wrong with this environment
#       | implement multi-agent RL (with noisy backprop)

model = perceptron.MultilayerPerceptron(n_input, n_hidden_1, n_hidden_2, n_output)
inputs = [tf.placeholder(shape=[1, n_input], dtype=tf.float32)]

# Construct output
current_output = model.evaluate(inputs[0])

# current_output = lstm_network.lstm_network(inputs, weights, biases)
target_output = tf.placeholder(shape=[1,n_output],dtype=tf.float32)

# Define loss and trainer
loss = tf.reduce_sum(tf.square(target_output - current_output))
trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
updateModel = trainer.minimize(loss)

init = tf.initialize_all_variables()
rewardToPlot = []
lossToPlot = []

for alpha in range(num_runs):
    print("ALPHA = " + str(alpha))
    # Create lists to contain total rewards and steps per episode
    jList = []
    rList = []
    sList = []
    rAggList = []
    lossAggList = []
    lossList = []
    # Stores tuples of (S, A, R, S')
    # For prioritized experience replay
    replay = []
    priority = []
    h = 0
    eps = initial_eps
    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_episodes):
            # learning_rate = learning_rate - learning_rate / (num_episodes)
            # Reset environment and get first new observation
            s = copy.copy(env.reset())
            rTotal = 0
            lossTotal = 0
            d = False
            j = 0
            # The q-network
            while j < 99:
                j+=1
                # Choose an action greedily (with e chance of random action) from the Q-network
                q = sess.run([current_output], feed_dict={inputs[0]: [s]})[0]
                a = np.argmax(q)
                if np.random.rand(1) < eps:
                    a = env.sample_action()
                # Get new state and reward from environment
                s1,r,d,_ = env.step(a)
                if d == True:
                    # Reduce chance of random action as we train the model
                    # rTotal += r
                    eps -= initial_eps / (num_episodes - 50)
                    eps = max(eps, 0)
                    break
                if len(replay) < buffer:  # if buffer not filled, add to it
                    replay.append((s, a, r, s1))
                    priority.append(0.001)
                    rTotal += r
                    s = copy.copy(s1)
                    continue
                # buffer is full - overwrite values
                if (h < (buffer - 1)):
                    h += 1
                else:
                    h = 0
                replay[h] = (s, a, r, s1)
                priority[h] = np.max(priority)
                s = copy.copy(s1)
                rTotal += r
                # randomly sample our experience replay memory
                minibatch_indices = np.random.choice(range(len(replay)), batch_size,
                                                     p=[np.sqrt(a)/sum([np.sqrt(b) for b in priority]) for a in priority])
                for index in minibatch_indices:
                    memory = replay[index]
                    s_mem, a_mem, r_mem, s1_mem = memory
                    q = sess.run(current_output, feed_dict={inputs[0]: [s_mem]})[0]
                    # Obtain the Q' values by feeding the new state through our network
                    q_prime = sess.run(current_output, feed_dict={inputs[0]: [s1_mem]})[0]
                    # Obtain max Q' and set our target value for chosen action
                    max_q_prime = np.max(q_prime)
                    target_q = q
                    # Q-learning
                    # target_q[a_mem] = np.clip(r_mem + discount * max_q_prime,-1,1)
                    # print(target_q)
                    # SARSA
                    target_q[a_mem] = np.clip(r_mem + discount * q_prime[a_mem],-1,1)
                    # Train our network using target and predicted Q values
                    _, loss_val = sess.run([updateModel,loss],feed_dict={inputs[0]: [s_mem],target_output: [target_q]})
                    priority[index] = abs(loss_val)
                    lossTotal += loss_val
                lossTotal /= batch_size
            sList.append(s)
            jList.append(j)
            rList.append(rTotal)
            lossList.append(lossTotal)
            if i % log_freq == 0:
                rAggList.append(np.mean(rList[-log_freq:]))
                lossAggList.append(np.mean(lossList[-log_freq:]))
                print("Average reward after " + str(i) + " episodes = " + str(np.mean(rList[-log_freq:])))
                print("Average loss after " + str(i) + " episodes = " + str(np.mean(lossList[-log_freq:])))
                print("Average basket after " + str(i) + " episodes = " + str(np.mean(sList[-log_freq:],axis=0)))
                # print("Epsilon after " + str(i) + " episodes = " + str(eps))
        # Evaluation
        print('Evaluating model...')
        s = copy.copy(env.reset())
        while j < 99:
            j += 1
            q = sess.run([current_output], feed_dict={inputs[0]: [s]})[0]
            print(q)
            a = np.argmax(q)
            s1, r, d, _ = env.step(a)
            print(str(s) +',' + str(a) + '->' + str(s1))
            if d:
                break
            s = copy.copy(s1)
    rewardToPlot.append(rAggList)
    lossToPlot.append(lossAggList)

plt.plot(np.mean(rewardToPlot,axis=0))
plt.savefig( 'reward.png' )

plt.clf()

plt.plot(np.mean(lossToPlot,axis=0))
plt.savefig( 'loss.png' )
