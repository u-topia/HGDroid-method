import numpy as np
import tensorflow as tf
import sys

# 实现玻尔兹曼机类
class RBM():

	def __init__(self, num_visible, num_hidden, learning_rate=0.01, momentum = 0.95, xavier_const=1.0, err_function = 'mse'):

		if err_function not in {'mse', 'cosine'}:
			raise ValueError('err_function should be either \'mse\' or \'cosine\'')
		self.nv = num_visible  # 可见层节点数 
		self.nh = num_hidden  # 隐藏层节点数
		self.learning_rate = learning_rate  # 学习率
		self.momentum = mumentum  # 动量

		# tf.placeholder()在模型中占位，分配必要的内存
		self.x = tf.placeholder(tf.float32, [None, self.nv])
		self.y = tf.placeholder(tf.float32, [None, self.nh])

		# tf.Variable()变量初始化
		self.w = tf.Variable(tf_xavier_init(self.nv, self.nh, const=xavier_const), dtype=tf.float32)
		self.visible_bias = tf.Variable(tf.zeros([self.nv]), dtype=tf.float32)
		self.hidden_bias = tf.Variable(tf.zeros([self.nh]), dtype=tf.float32)

		self.delta_w = tf.Variable(tf.zeros([self.nv, self.nh]), dtype=tf.float32)
		self.delta_visible_bias = tf.Variable(tf.zeros([self.nv]), dtype=tf.float32)
		self.delta_hidden_bias = tf.Variable(tf.zeros([self.nh]),dtype=tf.float32)

		self.update_weights = None
		self.update_deltas = None
		self.compute_hidden = None
		self.computee_visible = None
		self.compute_visible_from_hidden = None

		if err_function == 'cosine':
			x1_norm = tf.nn.l2_normalize(self.x, 1)  # 实现l2范化
			x2_norm = tf.nn.l2_normalize(self.compute_visible, 1)
			cos_val = tf.reduce_mean(tf.reduce_sum(tf.multiply(x1_norm, x2_norm), 1))
			self.compute_err = tf.acos(cos_val)/tf.constant(np.pi)
		else:
			self.compute_err = tf.reduce_mean(tf.square(self.x - self.compute_visible))

		# 将所有的图变量进行集体初始化
		init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init)

	def get_err(self, batch_x):
		return self.sess.run(self.computer_err, feed_dict = {self.x: batch_x})

	def reconstruct(self, batch_x):
		return self.sess.run(self.compute_visible, feed_dict = {self.x: batch_x})

	def partial_fit(self, batch_x):
		self.sess.run(self.update_weights + self.update_deltas, feed_dict = {self.x: batch_x})

	def fit(self, data_x, n_epoches = 10, batch_size = 10, shuffle = True, verbose = True):
		# data_x表示数据形状，n_epoches表示迭代次数，batch_size表示每次输入的大小，shuffle数据，verbose表示输出到标准输出
		assert n_epoches > 0

		n_data = data_x.shape[0]

		if batch_size > 0:
			n_batches = n_data/batch_size + (0 if n_data % batch_size == 0 else 1)
		else:
			n_batches = 10

		if shuffle:
			data_x_cpy = data_x.copy()
			inds = np.arange(n_data)
		else:
			data_x_cpy = data_x_cpy

		errs = []

		for e in range(n_epoches):
			print('Epoch: {:d}'.format(e))

			epoch_errs = np.zeros((n_batches, ))
			epoch_errs_ptr = 0 

			if shuffle:
				np.random.shuffle(inds)
				data_x_cpy = data_x_cpy[inds]

			r_batches = range(n_batches)

			for b in r_batches:
				batch_x = data_x_cpy[b * batch_size: (b + 1) * batch_size]
				self.partial_fit(batch_x)
				batch_err = self.get_err(batch_x)
				epoch_errs[epoch_errs_ptr] = batch_err
				epoch_errs_ptr += 1

			if verbose:
				err_mean = epoch_errs.mean()
				print('Train error:{:.4f}'.format(err_mean))
				sys.stdout.flush()

			errs = np.hstack([errs, epoch_errs])

		return errs


	def sigmoid(self, z):
		return 1.0/(1.0 + np.exp(-z))

# 二进制-二进制玻尔兹曼机（BBRBM）, binary = True
class BBRBM(RBM):
	def __init__(self, *args, **kwargs):
		RBM.__init__(self, *arg, **kwargs)

	def _initialize_vars(self):
		hidden_p = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hidden_bias)
		visible_recon_p = tf.nn.sigmoid(tf.matmul(sample_bernoulli(hippen_p), tf.transpose(self.w)) + self.visible_bias)
		hidden_recon_p = tf.nn.sigmoid(tf.matmul(visible_recon_p, self.w) + self.hidden_bias)

		positive_grad = tf.matmul(tf.transpose(self.x), hidden_p)
		negative_grad = tf.malmul(tf.transpose(visible_recon_p), hidden_recon_p)

		def f(x_old, x_new):
			return self.momentum * x_old + self.learning_rate * x_new * (1 - self.momentum) / tf.to_float(tf.shape(x_new)[0])

		delta_w_new = f(self.delta_w, positive_grad - negative_grad)
		delta_visible_bias_new = f(self.delta_visible_bias, tf.reduce_mean(self.x - visible_recon_p, 0))
		delta_hidden_bias_new = f(self.delta_hidden_bias, tf.reduce_mean(hidden_p - hidden_recon_p, 0))

		# assign()对变量进行更新
		update_delat_w = self.delta_w.assign(delta_w_new)
		update_delta_visible_bias = self.delta_visible_bias.assign(delta_visible_bias_new)
		update_delta_hidden_bias = self.delta_hidden_bias.assign(delta_hidden_bias_new)

		update_w = self.w.assign(self.w + delta_w_new)
		update_visible_bias = self.visible_bias.assign(self.visible_bias + delta_visible_bias_new)
		update_delta_hidden_bias = self.hidden_bias.assign(self.hidden_bias + delta_hidden_bias_new)

		self.update_deltas = [update_delta_w, update_delta_visible_bias, update_delta_hidden_bias]
		self.update_weights = [update_w, update_visible_bias, update_hidden_bias]

		self.compute_hidden = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hidden_bias)
		self.compute_visible = tf.nn.sigmoid(tf.matmul(self.compute_hidden, tf.transpose(self.w)) + self.visible_bias)
		self.compute_visible_from_hidden = tf.nn.sigmoid(tf.matmul(self.y, tf.transpose(self.w)) + self.visible_bias)



# 高斯-二进制玻尔兹曼机（GRBM）, 与伯努利玻尔兹曼机（BBRBM）的区别在于可见层的激活函数不同
class GRBM(RBM):
	def __init__(self, *args, **kwargs):
		RBM.__init__(self, *args, **kwargs)
