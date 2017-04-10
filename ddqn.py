import numpy as np
import tensorflow as tf
import random
import os 

CHECKPOINT_PATH = os.path.dirname(os.path.realpath(__file__)) + '/checkpoint/ddqn'
MEMORY_SIZE = 100
BATCH_SIZE = 32
GAMMA = 0.999
INIT_EPSILON = 1.0
FIN_EPSILON = 0.01
OBSERVE = 1000
EXPLORE = 1200

class DDQN():
    def __init__(self):
        self.session = tf.Session()

        self.epsilon = 1.0

        # Forward pass network
        self.input_tensor = tf.placeholder("float", [None, 1, 3, 2])

        with tf.variable_scope("q"):
            self.action_dist = self.forward_pass()

        # Target network
        with tf.variable_scope("target_q"):
            self.target_action_dist = self.forward_pass()

        # Experience relay
        self.memory = list()

        # Train
        self.action_input = tf.placeholder("float", [None, 3])
        self.y_input = tf.placeholder("float", [None])
        self.q_action = tf.reduce_sum(tf.multiply(self.action_dist, self.action_input), reduction_indices = 1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_action))
        self.train_step = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(self.cost)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        # Summary
        self.summary_ops, self.summary_vars = self.build_summaries()
        self.writer = tf.summary.FileWriter(CHECKPOINT_PATH, self.session.graph, flush_secs=10)
        self.cost_to_print = 0.0
        self.q_to_print = 0.0
    
    def build_summaries(self): 
        win_ratio = tf.Variable(0.)
        tf.summary.scalar("Win Ratio", win_ratio)

        cost = tf.Variable(0.)
        tf.summary.scalar("Cost", cost)

        q = tf.Variable(0.)
        tf.summary.scalar("Q Value", q)

        summary_vars = [win_ratio, cost, q]
        summary_ops = tf.summary.merge_all()

        return summary_ops, summary_vars

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.0, shape=shape)
        return tf.Variable(initial)

    def _fc_variable(self, weight_shape):
        input_channels  = weight_shape[0]
        output_channels = weight_shape[1]
        d = 1.0 / np.sqrt(input_channels)
        bias_shape = [output_channels]
        weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
        bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
        return weight, bias

    def forward_pass(self):
        h_flat = tf.reshape(self.input_tensor, [-1, 6])
        
        W_fc1, b_fc1 = self._fc_variable([6, 6])
        W_fc2, b_fc2 = self._fc_variable([6, 3])

        h_fc = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

        output = tf.nn.softmax(tf.matmul(h_fc, W_fc2) + b_fc2)
        return output

    def get_action(self, state):
        # Get action
        if self.epsilon > FIN_EPSILON and state.t > OBSERVE:
            self.epsilon -= (INIT_EPSILON - FIN_EPSILON) / EXPLORE

        if np.random.rand(1) < self.epsilon:
            return random.randrange(3)
        else:
            action_dist = self.session.run(self.action_dist, feed_dict={self.input_tensor: [state.s_t]})

            card_prob = np.zeros([len(state.my_cards[0])])

            # Select best action probability among my cards
            prob_sum = 0.0
            for i in range(0, len(state.my_cards[0])):
                if state.my_cards[0][i][0] == 1:
                    card_prob[i] = action_dist[0][i]
                    prob_sum += card_prob[i]
                else:
                    card_prob[i] = 0.0
            
            card_prob /= prob_sum

            return np.random.choice(range(len(card_prob)), p=card_prob)

    def store_experience(self, current_state, action_to_store, reward, next_state, done):
        self.memory.append((current_state, action_to_store, reward, next_state, done))

        if len(self.memory) > MEMORY_SIZE:
            self.memory.pop(0)

    def train(self, state, current_state, action_to_store, reward, next_state, done):
        # Store first
        self.store_experience(current_state, action_to_store, reward, next_state, done)

        if state.t < OBSERVE:
            return

        if len(self.memory) < BATCH_SIZE:
            batch_size = len(self.memory)
        else:
            batch_size = BATCH_SIZE

        minibatch = random.sample(self.memory, batch_size)

        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        terminal_batch = [data[4] for data in minibatch]

        y_batch = []
        q_batch = self.session.run(self.action_dist, feed_dict={self.input_tensor: next_state_batch})
        best_actions = np.argmax(q_batch, axis=1)
        
        target_q_batch = self.session.run(self.target_action_dist, feed_dict={self.input_tensor: next_state_batch})

        # for i in range(0, batch_size):
        #     terminal = minibatch[i][4]
        #     if terminal:
        #         y_batch.append(reward_batch[i])
        #     else:
        #         #y_batch.append(reward_batch[i] + GAMMA * np.max(q_batch[i]))
        #         y_batch.append(reward_batch[i] + GAMMA * np.max(target_q_batch[i]))

        y_batch = reward_batch + np.invert(terminal_batch).astype(np.float32) * GAMMA * target_q_batch[np.arange(batch_size), best_actions]

        self.q_to_print = np.sum(y_batch)

        _, self.cost_to_print = self.session.run([self.train_step, self.cost], feed_dict={self.y_input: y_batch, self.action_input: action_batch, self.input_tensor: state_batch})

        if state.t % 1000 == 0:
            #print('copying target network................................')
            self.copy_model_parameters(self.session, self.action_dist, self.target_action_dist)

    def write_summary(self, win_ratio, episode):
        summary_str = self.session.run(self.summary_ops, feed_dict={
                    self.summary_vars[0]: win_ratio,
                    self.summary_vars[1]: self.cost_to_print,
                    self.summary_vars[2]: self.q_to_print
                })

        self.writer.add_summary(summary_str, episode)

    def copy_model_parameters(self, sess, estimator1, estimator2):
        """
        Copies the model parameters of one estimator to another.

        Args:
        sess: Tensorflow session instance
        estimator1: Estimator to copy the paramters from
        estimator2: Estimator to copy the parameters to
        """
        e1_params = [t for t in tf.trainable_variables() if t.name.startswith("q")]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables() if t.name.startswith("target_q")]
        e2_params = sorted(e2_params, key=lambda v: v.name)

        update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            update_ops.append(op)

        sess.run(update_ops)

