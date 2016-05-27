import tensorflow as tf

class MLP():

    def __init__(self, topology, learning_rate=0.01, optimizer_algorithm='AdamOptimizer', reducer_function='reduce_mean', error_function=None):
        self.topology = topology
        self.learning_rate = learning_rate
        self.optimizer_algorithm = optimizer_algorithm
        self.reducer_function = reducer_function
        self.error_function = error_function

        optimizer_algorithm = getattr(tf.train, self.optimizer_algorithm)
        reducer_function = getattr(tf, self.reducer_function)

        input_size = self.topology[0][0]
        output_size = self.topology[-1][0]

        def lookup_activation_function(a):
            if a == 'none':
                return lambda x: x
            if hasattr(tf, a):
                return getattr(tf, a)
            if hasattr(tf.nn, a):
                return getattr(tf.nn, a)
            raise 'Unknown activation function'

        layer_definitions = list( map(lambda x: (x[0], lookup_activation_function(x[1])), self.topology) )

        input_activation_function = layer_definitions[0][1]
        output_activation_function = layer_definitions[-1][1]

        # Construct the input layer
        n_input = tf.placeholder(tf.float32, shape=[None, input_size], name="n_input")
        input_layer = input_activation_function(n_input)

        # Construct the hidden layers
        previous_size = input_size
        previous_layer = input_layer
        for i, layer_def in enumerate(layer_definitions[1:-1]):
            hidden_size = layer_def[0]
            activation_function = layer_def[1]

            b_hidden = tf.Variable(tf.zeros([hidden_size]), name="hidden_%d_bias" % i)
            W_hidden = tf.Variable(tf.zeros([previous_size, hidden_size]), name="hidden_%d_weights" % i)
            hidden = activation_function(tf.matmul(previous_layer, W_hidden) + b_hidden)

            previous_size = hidden_size
            previous_layer = hidden

        # Construct the output layer
        n_output = tf.placeholder(tf.float32, shape=[None, output_size], name="n_output")

        W_output = tf.Variable(tf.zeros([previous_size, output_size]), name="output_weights")
        output = output_activation_function(tf.matmul(previous_layer, W_output))

        # Define the learning ops
        if self.error_function is None:
            cross_entropy = -tf.reduce_sum( n_output * tf.log( tf.clip_by_value( output, 1e-10,1.0) ) )
        else:
            cross_entropy = self.error_function(n_output, output)

        loss = reducer_function(cross_entropy)
        optimizer = optimizer_algorithm(self.learning_rate)
        trainer = optimizer.minimize(loss)

        # Store the necessary ops
        self.n_input = n_input
        self.n_output = n_output
        self.output = output
        self.trainer = trainer
        self.loss = loss

        # Initialize a session
        self.session = tf.Session()
        init = tf.initialize_all_variables()
        self.session.run(init)

    def train(self, input_data, output_data, epochs, print_interval=None):
        for epoch in range(1, epochs+1):
            _, err = self.session.run([self.trainer, self.loss],
                               feed_dict={self.n_input: input_data, self.n_output: output_data})

            if print_interval is not None and epoch % print_interval == 0:
                print(self.apply([[1.0, 1.0], [1.0, 0.0]]))
                print("%d: %f" % (epoch, err))

    def apply(self, input_data):
        return self.session.run(self.output, feed_dict={self.n_input: input_data})


if __name__ == '__main__':
    #input_data = [[0.], [1.]]
    #output_data = [[1.], [0.]]

    #mlp = MLP([(1, 'none'), (5, 'sigmoid'), (1, 'sigmoid')])
    #mlp.train(input_data, output_data, 10000, print_interval=1000)

    input_data = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
    output_data = [[0.], [1.], [1.], [0.]]

    mlp = MLP([(2, 'none'), (5, 'sigmoid'), (1, 'sigmoid')])
    mlp.train(input_data, output_data, 1000, print_interval=100)

    result = mlp.apply(input_data)
    for i in range(len(input_data)):
        print('%s -> %s' % (input_data[i], result[i]))
