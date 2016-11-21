import numpy as np
import tensorflow as tf
import env as environment
import perceptron

env = environment.ShoprEnv()
tf.reset_default_graph()

# Parameters
learning_rate = 0.1
num_episodes = 1000
eps = 0.1
discount = 0.99

# Network Parameters
n_hidden_1 = 20 # 1st layer number of nodes
n_hidden_2 = 20 # 2nd layer number of nodes
n_input = 8 # input layer (basket state)
n_output = 9 # output layer (q values for actions)

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_uniform([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_uniform([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_uniform([n_hidden_2, n_output]))
}
biases = {
    'b1': tf.Variable(tf.random_uniform([n_hidden_1])),
    'b2': tf.Variable(tf.random_uniform([n_hidden_2])),
    'out': tf.Variable(tf.random_uniform([n_output]))
}

# Graph input
inputs = tf.placeholder(shape=[1,n_input],dtype=tf.float32)

# Construct output
current_output = perceptron.multilayer_perceptron(inputs, weights, biases)
qmax_action = tf.argmax(current_output,1)
target_output = tf.placeholder(shape=[1,n_output],dtype=tf.float32)

# Define loss and trainer
loss = tf.reduce_sum(tf.square(target_output - current_output))
trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
updateModel = trainer.minimize(loss)

init = tf.initialize_all_variables()

# Create lists to contain total rewards and steps per episode
jList = []
rList = []
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        # Reset environment and get first new observation
        s = env.reset()
        rTotal = 0
        d = False
        j = 0
        # The q-network
        while j < 20:
            j+=1
            # Choose an action greedily (with e chance of random action) from the Q-network
            a,q = sess.run([qmax_action,current_output],feed_dict={inputs:[s]})
            if np.random.rand(1) < eps:
                a[0] = env.sample_action()
            # Get new state and reward from environment
            s1,r,d,_ = env.step(a[0])
            # print(s,a[0],s1)
            # Obtain the Q' values by feeding the new state through our network
            q_prime = sess.run(current_output,feed_dict={inputs:[s1]})
            # Obtain max Q' and set our target value for chosen action
            max_q_prime = np.max(q_prime)
            target_q = q
            target_q[0,a[0]] = r + discount * max_q_prime
            # Train our network using target and predicted Q values
            _ = sess.run([updateModel],feed_dict={inputs:[s],target_output:target_q})
            rTotal += r
            s = s1
            if d == True:
                # Reduce chance of random action as we train the model.
                eps = 1./((i/50) + 10)
                break
        jList.append(j)
        rList.append(rTotal)
        if i % 50 == 0:
            print("Total reward after " + str(i) + " episodes = " + str(rTotal))
            print("When hit done after " + str(i) + " episodes = " + str(j))
            print("Basket after " + str(i) + " episodes = " + str(s1))