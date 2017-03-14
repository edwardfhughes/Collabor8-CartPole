import numpy as np
import tensorflow as tf
import env as environment
import perceptron
import lstm_network
import matplotlib.pyplot as plt
import random
import copy
import gym

# env = environment.ShoprEnv()
env = gym.make('CartPole-v0')
reset = tf.reset_default_graph()

np.random.seed(0)

# Parameters
learning_rate = 0.0001
num_episodes = 501
initial_eps = 0.5
eps_decay = 0.999
discount = 0.7
batch_size = 40
buffer = 3000
log_freq = 10
num_runs = 1
reg_constant = 0.01
grad_update_freq = 50

# Network Parameters
n_hidden_1 = 30 # 1st layer number of nodes
n_hidden_2 = 30 # 2nd layer number of nodes
n_input = 4 # input layer (state)
n_output = 2 # output layer (q values for actions)

model = perceptron.MultilayerPerceptron(n_input, n_hidden_1, n_hidden_2, n_output)
inputs = [tf.placeholder(shape=[1, n_input], dtype=tf.float32)]

# Construct output
current_output = model.evaluate(inputs[0])

# current_output = lstm_network.lstm_network(inputs, weights, biases)
target_output = tf.placeholder(shape=[1,n_output],dtype=tf.float32)

# Define loss (with regularization) and trainer
tvars = tf.trainable_variables()
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.reduce_sum(tf.square(target_output - current_output)) + reg_constant * sum(reg_losses)
trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# frozen network for stability
W1Grad = tf.placeholder(tf.float32,name="batch_grad1")
W2Grad = tf.placeholder(tf.float32,name="batch_grad2")
WoutGrad = tf.placeholder(tf.float32,name="batch_grad3")
batchGrad = [W1Grad,W2Grad,WoutGrad]
# ignore hidden weights which it somehow knows we defined
newGrads = tf.gradients(loss, tvars)[0:3]
updateGrads = trainer.apply_gradients(zip(batchGrad, tvars))
# updateModel = trainer.minimize(loss)

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
    stepUntilDoneList = []
    # Stores tuples of (S, A, R, S')
    # For prioritized experience replay
    replay = []
    priority = []
    h = 0
    eps = initial_eps
    grad_count = 0
    with tf.Session() as sess:
        sess.run(init)
        gradBuffer = sess.run(tvars)
        for ix, grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0
        for i in range(num_episodes):
            # learning_rate = learning_rate - learning_rate / (num_episodes)
            # Reset environment and get first new observation
            s = copy.copy(env.reset())
            rTotal = 0
            lossTotal = 0
            d = False
            j = 0
            # The q-network
            while j < 1000:
                # env.render()
                j+=1
                # Choose an action greedily (with e chance of random action) from the Q-network
                q = sess.run([current_output], feed_dict={inputs[0]: [s]})[0]
                a = np.argmax(q)
                if np.random.rand(1) < eps:
                    # a = env.sample_action()
                    a = env.action_space.sample()
                # Get new state and reward from environment
                s1,r,d,_ = env.step(a)
                if d:
                    # Penalise failure (should we do this?)
                    if r < 200:
                        r -= 50
                if len(replay) < buffer:  # if buffer not filled, add to it
                    replay.append((s, a, r, s1))
                    priority.append(0.001)
                    if d:
                        rTotal += r
                        break
                # buffer is full - overwrite values
                else:
                    if (h < (buffer - 1)):
                        h += 1
                    else:
                        h = 0
                    replay[h] = (s, a, r, s1)
                    priority[h] = 0.001 # np.max(priority)
                s = copy.copy(s1)
                rTotal += r
                # randomly sample our experience replay memory
                # minibatch_indices = np.random.choice(range(len(replay)), batch_size,
                                                     # p=[np.sqrt(a)/sum([np.sqrt(b) for b in priority]) for a in priority])
                if not (batch_size < len(replay)):
                    break
                minibatch_indices = np.random.choice(range(len(replay)), batch_size)
                for index in minibatch_indices:
                    temp_loss = 0
                    memory = replay[index]
                    s_mem, a_mem, r_mem, s1_mem = memory
                    target_q = sess.run(current_output, feed_dict={inputs[0]: [s_mem]})[0]
                    # Obtain the Q' values by feeding the new state through our network
                    q_prime = sess.run(current_output, feed_dict={inputs[0]: [s1_mem]})[0]
                    # Obtain max Q' and set our target value for chosen action
                    max_q_prime = np.max(q_prime)
                    # Q-learning
                    target_q[a_mem] = np.clip(r_mem + discount * max_q_prime,-200,300)
                    # print(target_q[a_mem])
                    # print(target_q)
                    # SARSA
                    # target_q[a_mem] = np.clip(r_mem + discount * q_prime[a_mem],-1,1)
                    # Train our network using target and predicted Q values
                    # why do we have to do these sequentially?
                    tGrad = sess.run(newGrads, feed_dict={inputs[0]: [s_mem], target_output: [target_q]})
                    loss_val = sess.run(loss, feed_dict={inputs[0]: [s_mem], target_output: [target_q]})
                    # priority[index] = abs(loss_val)
                    temp_loss += loss_val
                    for ix, grad in enumerate(tGrad):
                        # accumulate the gradients
                        gradBuffer[ix] += grad
                    grad_count += 1
                    if grad_count % grad_update_freq == 0:
                        sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0], W2Grad: gradBuffer[1], WoutGrad: gradBuffer[2]})
                        for ix, grad in enumerate(gradBuffer):
                            gradBuffer[ix] = grad * 0
                lossTotal += temp_loss / batch_size
                if d:
                    # Reduce chance of random action as we train the model
                    # eps -= initial_eps / (num_episodes)
                    eps = eps * eps_decay
                    eps = max(eps, 1e-5)
                    # get loss per iteration, a better measure
                    lossTotal = lossTotal / j
                    # replay.append((s, a, r, s1))
                    # priority.append(0.001)
                    stepUntilDoneList.append(j)
                    break
            # sList.append(s)
            # jList.append(j)
            rList.append(rTotal)
            lossList.append(lossTotal)
            if i % log_freq == 0:
                rAggList.append(np.mean(rList[-log_freq:]))
                lossAggList.append(np.mean(lossList[-log_freq:]))
                print("Average reward after " + str(i) + " episodes = " + str(np.mean(rList[-log_freq:])))
                print("Average loss after " + str(i) + " episodes = " + str(np.mean(lossList[-log_freq:])))
                # print("Average basket after " + str(i) + " episodes = " + str(np.mean(sList[-log_freq:],axis=0)))
                # print("Average steps until done after " + str(i) + " episodes = " + str(np.mean(stepUntilDoneList[-log_freq:])))
                # print("Epsilon after " + str(i) + " episodes = " + str(eps))
        # Evaluation
        print('Evaluating model...')
        s = copy.copy(env.reset())
        j = 0
        while j < 10000:
            # env.render()
            j += 1
            q = sess.run([current_output], feed_dict={inputs[0]: [s]})[0]
            # print(q)
            a = np.argmax(q)
            s1, r, d, _ = env.step(a)
            # print(str(s) +',' + str(a) + '->' + str(s1))
            if d:
                print('failed after ' + str(j) + ' timesteps')
                break
            s = copy.copy(s1)
    rewardToPlot.append(rAggList)
    lossToPlot.append(lossAggList)

plt.plot(np.mean(rewardToPlot,axis=0))
plt.savefig( 'reward.png' )

plt.clf()

plt.plot(np.mean(lossToPlot,axis=0))
plt.savefig( 'loss.png' )
