# Distance Metric Learning dmlearn.py

import math
import os
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from utils import ulogger as logger
import config

def stop_reached(datum, tensor):
	if datum.node_name == 't_for_stop': # and tensor > 100:
		return True
	return False

t_for_stop = tf.constant(5.0, name='t_for_stop')


def build_nn(var_scope, input_dim, b_reuse):
	# num_inputs = tf.shape(t_nn_x)[0]
	weight_factor = 1.0 / tf.cast(input_dim * config.c_key_dim, dtype=tf.float32)
	# t_shape = tf.constant([2], dtype=tf.int32)
	with tf.variable_scope(var_scope, reuse=b_reuse):
		# v_W = tf.Variable(tf.random_normal(shape=[num_inputs, c_key_dim], mean=0.0, stddev=weight_factor), dtype=tf.float32)
		v_W = tf.get_variable('v_W', shape=[input_dim, config.c_key_dim], dtype=tf.float32,
							  initializer=tf.random_normal_initializer(mean=0.0, stddev=weight_factor))
	return v_W

def build_nn_run(name_scope, t_nn_x, v_W):

	with tf.name_scope(name_scope):
		t_y = tf.nn.l2_normalize(tf.matmul(t_nn_x, v_W), dim=1, name='t_y')

	return t_y

norm = lambda vec: vec / np.linalg.norm(vec, axis=1, keepdims=True)

def init_output_tensors(ivec_arr, name_scope_prefix, l_W):
	l_y = []
	for iivec, one_ivec in enumerate(ivec_arr):
		if one_ivec == None:
			# This will happen when there were no records for that input_vec_dim
			continue
		# norm in the following line works on 2D arrays too, one sum for each row
		ivec_normed = norm(one_ivec)
		v_x = tf.Variable(tf.constant(ivec_normed.astype(np.float32)), dtype=tf.float32, trainable=False, name='v_x')
		l_y.append(build_nn_run(name_scope=name_scope_prefix+'_main', t_nn_x=v_x, v_W=l_W[iivec]))

	t_y = tf.concat(l_y, axis=0, name='t_y')
	return t_y

def init_train_tensors(ivec_dim_dict, ivec_arr, name_scope_prefix):

	l_W = [None] * len(ivec_dim_dict)
	for ivec_dim, dim_pos in ivec_dim_dict.iteritems():
		l_W[dim_pos] = (build_nn(var_scope=name_scope_prefix+'_nn_'+str(dim_pos), input_dim=ivec_dim, b_reuse=False))

	t_y = init_output_tensors(ivec_arr, name_scope_prefix, l_W)

	return t_y, l_W

def create_selector_tensors(ovec, numrecs, t_y):
	ovec_norm = norm(ovec)
	v_o = tf.Variable(tf.constant(ovec_norm.astype(np.float32)), dtype=tf.float32, trainable=False, name='v_o')
	v_r1 = tf.Variable(	tf.random_uniform([config.c_rsize], minval=0, maxval=numrecs-1, dtype=tf.int32),
						trainable=False, name='v_r1')
	v_r2 = tf.Variable(	tf.random_uniform([config.c_rsize], minval=0, maxval=numrecs-1, dtype=tf.int32),
						trainable=False, name='v_r2')

	t_o1 = tf.gather(v_o, v_r1, name='t_o1')
	t_o2 = tf.gather(v_o, v_r2, name='t_o2')

	t_y1 = tf.gather(t_y, v_r1, name='t_y1')
	t_y2 = tf.gather(t_y, v_r2, name='t_y2')

	t_cdo = tf.reduce_sum(tf.multiply(t_o1, t_o2), axis=1, name='t_cdo')
	t_cdy = tf.reduce_sum(tf.multiply(t_y1, t_y2), axis=1, name='t_cdy')
	t_err = tf.reduce_mean((t_cdo - t_cdy) ** 2, name='t_err')
	op_train_step = tf.train.AdamOptimizer(config.FLAGS.nn_lrn_rate).minimize(t_err, name='op_train_step')

	return op_train_step, t_err, v_r1, v_r2

def create_pair_tensors(match_pairs, mismatch_pairs, t_y_db, t_y_q, ):
	num_match_pairs = len(match_pairs)
	num_mismatch_pairs = len(mismatch_pairs)

	v_match_pairs = tf.Variable(tf.constant(match_pairs), trainable=True, name='v_match_pairs')
	v_mismatch_pairs = tf.Variable(tf.constant(mismatch_pairs), trainable=True, name='v_mismatch_pairs')
	v_all_pairs = tf.concat([v_match_pairs, v_mismatch_pairs], axis=0)

	v_r1 = tf.Variable(tf.zeros([config.c_match_batch_size], dtype=tf.int32), dtype=tf.int32)
	op_r1_assign = tf.assign(v_r1, tf.random_uniform([config.c_match_batch_size], dtype=tf.int32,
													 minval=0, maxval=num_match_pairs-1))
	v_r2 = tf.Variable(tf.zeros([config.c_mismatch_batch_size], dtype=tf.int32), dtype=tf.int32)
	op_r2_assign = tf.assign(v_r2, tf.random_uniform([config.c_mismatch_batch_size], dtype=tf.int32,
													 minval=num_match_pairs,
													 maxval=num_match_pairs+num_mismatch_pairs-1))
	l_batch_assigns = [op_r1_assign, op_r2_assign]

	t_r_pairs = tf.gather(v_all_pairs, tf.concat([v_r1, v_r2], axis=0), name='t_r_pairs')

	batch_size = config.c_match_batch_size + config.c_mismatch_batch_size
	t_q_of_pair = tf.squeeze(tf.slice(t_r_pairs, [0, 0], [batch_size, 1]), 1, name='t_q_of_pair')
	t_db_of_pair = tf.squeeze(tf.slice(t_r_pairs, [0, 1], [batch_size, 1], name='temp'), 1, name='t_db_of_pair')

	t_y_db_batch = tf.gather(t_y_db, t_db_of_pair)
	t_y_q_batch = tf.gather(t_y_q, t_q_of_pair)

	t_targets = tf.concat([tf.ones([config.c_match_batch_size], dtype=tf.float32),
						   tf.zeros([config.c_mismatch_batch_size], dtype=tf.float32)], axis=0)

	return l_batch_assigns, t_y_db_batch, t_y_q_batch, t_targets

def create_learn_tensors(t_y_db_batch, t_y_q_batch, t_targets):
	t_cdy = tf.reduce_sum(tf.multiply(t_y_q_batch, t_y_db_batch), axis=1, name='t_cdy')
	t_err = tf.reduce_mean((t_targets - t_cdy) ** 2, name='t_err')
	op_train_step = tf.train.AdamOptimizer(config.FLAGS.nn_lrn_rate).minimize(t_err, name='op_train_step')

	return t_err, op_train_step

def prep_learn(ivec_dim_dict_db, ivec_dim_dict_q, ivec_arr_db, ivec_arr_q, match_pairs, mismatch_pairs):
	t_y_db, l_W_db = init_train_tensors(ivec_dim_dict_db, ivec_arr_db, 'db')
	t_y_q, l_W_q = init_train_tensors(ivec_dim_dict_q, ivec_arr_q, 'q')
	l_batch_assigns, t_y_db_batch, t_y_q_batch, t_targets = \
		create_pair_tensors(match_pairs, mismatch_pairs, t_y_db=t_y_db, t_y_q=t_y_q)
	t_err, op_train_step = create_learn_tensors(t_y_db_batch, t_y_q_batch, t_targets)

	return t_y_db, l_W_db, l_W_q, l_batch_assigns, t_err, op_train_step

def prep_eval(ivec_arr_eval, t_y_db, l_W_q):
	t_y_eval = init_output_tensors(ivec_arr_eval, 'eval', l_W_q)
	t_cdy_eval = tf.matmul(t_y_eval, t_y_db, transpose_b=True, name='t_cyy_eval')
	t_top_cds, t_top_idxs = tf.nn.top_k(t_cdy_eval, k=config.c_num_k_eval, sorted=True, name='t_top_cds')

	return t_top_cds, t_top_idxs

def init_learn(l_W_all):
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	saver_dict = {'W_'+str(i): W for i, W in enumerate(l_W_all)}

	saver = tf.train.Saver(saver_dict, max_to_keep=3)
	ckpt = None
	if config.FLAGS.save_dir:
		ckpt = tf.train.get_checkpoint_state(config.FLAGS.save_dir)
	if ckpt and ckpt.model_checkpoint_path:
		logger.info('restoring from %s', ckpt.model_checkpoint_path)
		saver.restore(sess, ckpt.model_checkpoint_path)

	if config.FLAGS.debug:
		sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type="curses")
		sess.add_tensor_filter("stop_reached", stop_reached)

	return sess, saver

# def create_eval_tensors(db_size, q_size, t_y):
# 	db_size = int(float(numrecs/ config.c_eval_db_factor) * (1.0 - config.c_test_pct))
# 	test_size = numrecs - db_size # int(float(numrecs / c_eval_db_factor) * c_test_pct)
# 	v_r_eval = tf.Variable(tf.random_shuffle(tf.range(numrecs)), name='v_r_eval')
# 	t_r_db = tf.slice(input_=v_r_eval, begin=[0], size=[db_size], name='t_r_db')
# 	t_r_test = tf.slice(input_=v_r_eval, begin=[db_size], size=[test_size], name='t_r_test')
# 	v_y = tf.Variable(tf.zeros([numrecs, config.c_key_dim], dtype=tf.float32), name='v_y')
# 	op_y = tf.assign(v_y, t_y, name='op_y')
#
# 	t_key_db = tf.gather(v_y, t_r_db, name='t_key_db')
# 	t_key_test = tf.gather(v_y, t_r_test, name='t_key_test')
#
# 	t_eval_key_cds =  tf.matmul(t_key_test, t_key_db, transpose_b=True, name='t_eval_key_cds')
# 	t_key_cds, t_r_key_idxs = tf.nn.top_k(t_eval_key_cds, config.c_key_num_ks, sorted=True, name='t_keys')
# 	t_key_idxs = tf.gather(t_r_db, t_r_key_idxs, name='t_key_idxs')
# 	return t_r_test, t_key_cds, t_key_idxs, op_y, v_r_eval

def run_learning(sess, l_batch_assigns, t_err, saver, op_train_step):

	losses = []

	for step in range(config.c_num_steps):
		sess.run(l_batch_assigns)
		if step == 0:
			errval = math.sqrt(sess.run(t_err))
			logger.info('Starting error: %f', errval)
		elif step % (config.c_num_steps / 100) == 0:
			errval = np.mean(losses)
			losses = []
			logger.info('step: %d: error: %f', step, errval)
		if step % (config.c_num_steps / 10) == 0:
			if saver and config.FLAGS.save_dir:
				logger.info('saving v_W to %s', config.FLAGS.save_dir)
				saved_file = saver.save(sess,
										os.path.join(config.FLAGS.save_dir, 'model.ckpt'),
										step)
		outputs = sess.run([t_err, op_train_step])
		losses.append(math.sqrt(outputs[0]))

def run_eval(sess, t_top_cds, t_top_idxs):
	return sess.run([t_top_cds, t_top_idxs])







