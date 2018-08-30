import tensorflow as tf
tf.set_random_seed(77999)
from sklearn.base import BaseEstimator


class DLRegression(BaseEstimator):
    def __init__(self,
                 hidden_layers,
                 batch_size,
                 dropout,
                 l2_regularizer,
                 learning_rate,
                 beta1,
                 beta2,
                 epsilon,
                 epochs,
                 save_path,
                 logs_path,
                 print_epoch=True,
                 activation=tf.nn.relu):
        self.hidden_layers =hidden_layers
        self.batch_size = batch_size
        self.dropout = dropout
        self.l2_regularizer = l2_regularizer
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.epochs = epochs
        self.save_path = save_path
        self.logs_path = logs_path
        self.print_epoch = print_epoch
        self.activation = activation

        tf.set_random_seed(77999)
        tf.reset_default_graph()
        super().__init__()

    def _set_placeholders(self):
        """
        Placeholders for input features and output
        """
        self.X = tf.placeholder(tf.float32, shape=[None, self.x_shape])
        self.y_true = tf.placeholder(tf.float32, shape=[None, self.y_shape])

    def _hidden_layers(self):
        """
        Create hidden Layers
        """
        hidden_lst = [self.X]
        for i, layer in enumerate(self.hidden_layers):
            hidden_layer_i = tf.layers.dense(inputs=hidden_lst[i],
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer
                                                                                 (scale=self.l2_regularizer),
                                             units=layer,
                                             activation=self.activation,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True,
                                                                                                     seed=77999))
            hidden_layer_i = tf.layers.dropout(inputs=hidden_layer_i, rate=self.dropout, seed=77999)
            hidden_lst.append(hidden_layer_i)

        self.hidden_output = hidden_lst[-1]
        print(f'\nHidden layers: {hidden_lst}\n')

    def _output_layer(self):
        """
        Create Output layer
        """
        self.output_layer = tf.layers.dense(inputs=self.hidden_output, units=self.y_shape, activation=None)
        print(f'Output layer: {self.output_layer}\n')

    def _loss(self):
        self.loss_train = tf.losses.mean_squared_error(self.output_layer, self.y_true) + tf.losses.get_regularization_loss()
        self.loss_val = tf.losses.mean_squared_error(self.output_layer, self.y_true) + tf.losses.get_regularization_loss()

    def _optimizer(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                               beta1=self.beta1,
                                               beta2=self.beta2,
                                               epsilon=self.epsilon)
        self.train = optimizer.minimize(self.loss_train)

    def fit(self, X, y, valid_data=(), **fit_params):
        """
        Fit Deep Learning regression model weights to  self.save_path+'/trained_model_weight'
        :param X: Train X
        :param y: Train Y
        :param valid_data: Optional Validation data
        :param fit_params:
        :return: None
        """
        tf.set_random_seed(77999)
        self.x_shape = X.shape[1]
        self.y_shape = y.shape[1]
        self._set_placeholders()
        self._hidden_layers()
        self._output_layer()
        self._loss()
        self._optimizer()

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            tf.set_random_seed(77999)
            sess.run(init)

            for i in range(self.epochs):
                _, loss_training = sess.run([self.train, self.loss_train], feed_dict={self.X: X, self.y_true: y})
                if len(valid_data) == 2:
                    loss_valid = sess.run(sess.loss_val, feed_dict={self.X: valid_data[0], self.y_true: valid_data[1]})
                    if self.print_epoch:
                        print(f'Epoch {i}: Train loss = {loss_training:.4f}, Valid loss = {loss_valid:,4f}')
                else:
                    if self.print_epoch:
                        print(f'Epoch {i}: Train loss = {loss_training:.4f}')

            #save session
            saver.save(sess, self.save_path+'/trained_model_weight')
            sess.close()

    def predict(self, X, **fit_params):
        """
        Load tf weigths and predict
        :param X: input test data (numpy array)
        :param fit_params:
        :return: predicted output (numpy array)
        """

        tf.reset_default_graph()  # reset graph to avoid using currently loaded weights.
        self.x_shape = X.shape[1]
        self._set_placeholders()
        self._hidden_layers()
        self._output_layer()
        self._loss()
        self._optimizer()

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, self.save_path + '/trained_model_weight')
            y_test_pred = self.output_layer.eval(feed_dict={self.X: X})
            sess.close()

        return y_test_pred