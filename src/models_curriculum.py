import tf_util as U
import tensorflow as tf
import numpy as np

# Check max-pooling on proj, deconv net



class mymodel_curr(object):
	def __init__(self, name, *args, **kwargs):
		with tf.variable_scope(name):
			self._init(*args, **kwargs)
			self.scope = tf.get_variable_scope().name

	def _init(self, img_shape, latent_dim):
		sequence_length = None

		disentangle_feat_sz = 4
		entangle_feat_sz = latent_dim - disentangle_feat_sz

		img1 = U.get_placeholder(name="img1", dtype=tf.float32, shape=[sequence_length, img_shape[0], img_shape[1], img_shape[2]])
		img2 = U.get_placeholder(name="img2", dtype=tf.float32, shape=[sequence_length, img_shape[0], img_shape[1], img_shape[2]])

		cond1 = U.get_placeholder(name="cond1", dtype=tf.bool, shape=())
		cond2 = U.get_placeholder(name="cond2", dtype=tf.bool, shape=())
		cond3 = U.get_placeholder(name="cond3", dtype=tf.bool, shape=())
		cond4 = U.get_placeholder(name="cond4", dtype=tf.bool, shape=())
		cond5 = U.get_placeholder(name="cond5", dtype=tf.bool, shape=())
		cond6 = U.get_placeholder(name="cond6", dtype=tf.bool, shape=())


		# [testing -> ]
		img_test = U.get_placeholder(name="img_test", dtype=tf.float32, shape=[sequence_length, img_shape[0], img_shape[1], img_shape[2]])
		reconst_tp = U.get_placeholder(name="reconst_tp", dtype=tf.float32, shape=[sequence_length, latent_dim])
		# [testing <- ]


		img1_scaled = img1
		img2_scaled = img2

		[mu1, logvar1, mu2, logvar2] = self.siamese_encoder(img1_scaled, img2_scaled, latent_dim)

		print np.shape(mu1)
		print np.shape(logvar1)

		latent_z1 = self.sample_latent_var(mu1, logvar1)
		latent_z2 = self.sample_latent_var(mu2, logvar2)


		self.latent_z1 = latent_z1
		self.latent_z2 = latent_z2

		print np.shape(latent_z1)
		print np.shape(latent_z2)

		[reconst1, reconst2] = self.siamese_decoder(latent_z1, latent_z2)
		self.reconst1 = reconst1
		self.reconst2 = reconst2

		# [testing -> ]
		selected_feature = 28
		latent_v_range = np.arange(-1, 1, 0.5)

		[mu_test, logvar_test] = self.encoder(img_test, latent_dim)
		self.latent_z_test = self.sample_latent_var(mu_test, logvar_test)
		print ("----")
		print np.shape(self.latent_z_test)
		self.reconst_test = self.decoder(reconst_tp)

		# [testing <- ]

		sh_mu1_1 = mu1[:, 0:]
		sh_logvar1_1 = logvar1[:, 0:]
		sh_mu1_2 = mu2[:, 0:]
		sh_logvar1_2 = logvar2[:, 0:]

		sh_mu2_1 = mu1[:, 0:-1]
		sh_logvar2_1 = logvar1[:, 0:-1]
		sh_mu2_2 = mu2[:, 0:-1]
		sh_logvar2_2 = logvar2[:, 0:-1]

		sh_mu3_1 = mu1[:, 0:-2]
		sh_logvar3_1 = logvar1[:, 0:-2]
		sh_mu3_2 = mu2[:, 0:-2]
		sh_logvar3_2 = logvar2[:, 0:-2]

		sh_mu4_1 = mu1[:, 0:-3]
		sh_logvar4_1 = logvar1[:, 0:-3]
		sh_mu4_2 = mu2[:, 0:-3]
		sh_logvar4_2 = logvar2[:, 0:-3]

		sh_mu5_1 = mu1[:, 0:-4]
		sh_logvar5_1 = logvar1[:, 0:-4]
		sh_mu5_2 = mu2[:, 0:-4]
		sh_logvar5_2 = logvar2[:, 0:-4]

		sh_mu6_1 = mu1[:, 0:-5]
		sh_logvar6_1 = logvar1[:, 0:-5]
		sh_mu6_2 = mu2[:, 0:-5]
		sh_logvar6_2 = logvar2[:, 0:-5]

		sh_mu7_1 = mu1[:, 0:-6]
		sh_logvar7_1 = logvar1[:, 0:-6]
		sh_mu7_2 = mu2[:, 0:-6]
		sh_logvar7_2 = logvar2[:, 0:-6]											


		siam_loss1 = U.sum(tf.square(sh_mu1_1 - sh_mu1_2), axis = 1) + U.sum(tf.square(sh_logvar1_1 - sh_logvar1_2), axis = 1)
		siam_loss2 = U.sum(tf.square(sh_mu2_1 - sh_mu2_2), axis = 1) + U.sum(tf.square(sh_logvar2_1 - sh_logvar2_2), axis = 1)
		siam_loss3 = U.sum(tf.square(sh_mu3_1 - sh_mu3_2), axis = 1) + U.sum(tf.square(sh_logvar3_1 - sh_logvar3_2), axis = 1)
		siam_loss4 = U.sum(tf.square(sh_mu4_1 - sh_mu4_2), axis = 1) + U.sum(tf.square(sh_logvar4_1 - sh_logvar4_2), axis = 1)
		siam_loss5 = U.sum(tf.square(sh_mu5_1 - sh_mu5_2), axis = 1) + U.sum(tf.square(sh_logvar5_1 - sh_logvar5_2), axis = 1)
		siam_loss6 = U.sum(tf.square(sh_mu6_1 - sh_mu6_2), axis = 1) + U.sum(tf.square(sh_logvar6_1 - sh_logvar6_2), axis = 1)
		siam_loss7 = U.sum(tf.square(sh_mu7_1 - sh_mu7_2), axis = 1) + U.sum(tf.square(sh_logvar7_1 - sh_logvar7_2), axis = 1)



		# If cond1 true, then we get self.siam_loss1, else then we search again with cond2.
		# If cond2 true, then we get self.siam_loss2, else then we search again with cond3 and so on.

		self.siam_loss6 = U.switch(cond6, siam_loss6, siam_loss7)
		self.siam_loss5 = U.switch(cond5, siam_loss5, self.siam_loss6)
		self.siam_loss4 = U.switch(cond4, siam_loss4, self.siam_loss5)
		self.siam_loss3 = U.switch(cond3, siam_loss3, self.siam_loss4)
		self.siam_loss2 = U.switch(cond2, siam_loss2, self.siam_loss3)
		self.siam_loss = U.switch(cond1, siam_loss1, self.siam_loss2)


		self.kl_loss1 = 0.5 * U.sum((tf.exp(logvar1) + mu1**2 - 1. - logvar1), axis = 1)
		self.kl_loss2 = 0.5 * U.sum((tf.exp(logvar2) + mu2**2 - 1. - logvar2), axis = 1)

		reconst_error1 = tf.square(reconst1 - img1_scaled)
		self.reconst_error1 = tf.reduce_sum(reconst_error1, [1, 2, 3])
		reconst_error2 = tf.square(reconst2 - img2_scaled)
		self.reconst_error2 = tf.reduce_sum(reconst_error2, [1, 2, 3])		


		self.vaeloss = 10000.0*self.siam_loss + 30000.0*self.kl_loss1 + 30000.0*self.kl_loss2 + self.reconst_error1 + self.reconst_error2



	def encoder_net(self, img, latent_dim):
		x = img
		x = tf.nn.relu(U.conv2d(x, 32, "c1", [4, 4], [2, 2], pad = "SAME")) # [32, 32, 32]
		x = tf.nn.relu(U.conv2d(x, 32, "c2", [4, 4], [2, 2], pad = "SAME")) # [16, 16, 32]
		x = tf.nn.relu(U.conv2d(x, 64, "c3", [4, 4], [2, 2], pad = "SAME")) # [8, 8, 64]
		x = tf.nn.relu(U.conv2d(x, 64, "c4", [4, 4], [2, 2], pad = "SAME")) # [4, 4, 64]
		x = U.flattenallbut0(x) # [1024]
		x = tf.nn.relu(U.dense(x, 256, 'l1', U.normc_initializer(1.0))) # 1024
		mu = U.dense(x, latent_dim, 'l1_1', U.normc_initializer(1.0)) # 32
		logvar = U.dense(x, latent_dim, 'l1_2', U.normc_initializer(1.0)) # 32
		return mu, logvar

	def decoder_net(self, latent_variable):
		x = latent_variable
		x = U.dense(x, 256, 'l2', U.normc_initializer(1.0))
		x = tf.nn.relu(U.dense(x, 1024, 'l3', U.normc_initializer(1.0)))
		x = tf.reshape(x, [tf.shape(x)[0], 4,4,64]) # Unflatten [4, 4, 64]
		x = tf.nn.relu(U.conv2d_transpose(x, [4,4,64,64], [tf.shape(x)[0], 8,8,64], "uc1", [2, 2], pad="SAME")) # [8, 8, 64]
		x = tf.nn.relu(U.conv2d_transpose(x, [4,4,32,64], [tf.shape(x)[0], 16,16,32], "uc2", [2, 2], pad="SAME")) # [16, 16, 32]
		x = tf.nn.relu(U.conv2d_transpose(x, [4,4,32,32], [tf.shape(x)[0], 32,32,32], "uc3", [2, 2], pad="SAME")) # [32, 32, 32]
		x = U.conv2d_transpose(x, [4,4,1,32], [tf.shape(x)[0], 64,64,1], "uc4", [2, 2], pad="SAME") # [64, 64, 1]
		return x

	def sample_latent_var(self, mu, logvar):
		eps = tf.random_normal(shape=tf.shape(mu))
		return mu + tf.exp(logvar / 2) * eps

	def siamese_encoder(self, s1, s2, latent_dim):
		with tf.variable_scope("siamese_encoder") as scope:
			[mu1, logvar1] = self.encoder_net(img = s1, latent_dim = latent_dim)
			scope.reuse_variables()
			[mu2, logvar2] = self.encoder_net(img = s2, latent_dim = latent_dim)
		return [mu1, logvar1, mu2, logvar2]

	# [testing -> ]
	def encoder(self, s1, latent_dim):
		with tf.variable_scope("siamese_encoder") as scope:
			scope.reuse_variables()
			[mu1, logvar1] = self.encoder_net(img = s1, latent_dim = latent_dim)
		return [mu1, logvar1]

	def decoder(self, l1):
		with tf.variable_scope("siamese_decoder") as scope:
			scope.reuse_variables()
			reconst_test = self.decoder_net(latent_variable = l1)
		return reconst_test
	# [testing <- ]

	def siamese_decoder(self, l1, l2):
		with tf.variable_scope("siamese_decoder") as scope:
			reconst1 = self.decoder_net(latent_variable = l1)
			scope.reuse_variables()
			reconst2 = self.decoder_net(latent_variable = l2)
		return [reconst1, reconst2]

	def get_loss(self):
		return [self.siam_loss, self.kl_loss1, self.kl_loss2, self.reconst_error1, self.reconst_error2]

	def get_variables(self):
		return tf.get_collection(tf.GraphKeys.VARIABLES, self.scope)
	def get_trainable_variables(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
	def get_initial_state(self):
		return []