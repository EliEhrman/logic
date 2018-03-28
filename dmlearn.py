# Distance Metric Learning dmlearn.py
from __future__ import print_function

import math
import os
# import time
import random
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from utils import ulogger as logger
import config
import makerecs as mr

def stop_reached(datum, tensor):
	if datum.node_name == 't_for_stop': # and tensor > 100:
		return True
	return False

t_for_stop = tf.constant(5.0, name='t_for_stop')

def learn_reset():
	tf.reset_default_graph()

def build_templ_nn(var_scope, input_dim, b_reuse):
	weight_factor = 1.0 / (input_dim * config.c_key_dim)
	# weight_factor = 1.0 / 150 * 15
	# t_shape = tf.constant([2], dtype=tf.int32)
	with tf.variable_scope(var_scope, reuse=b_reuse):
		# v_W = tf.Variable(tf.random_normal(shape=[num_inputs, c_key_dim], mean=0.0, stddev=weight_factor), dtype=tf.float32)
		v_W = tf.get_variable('v_W', shape=[input_dim, config.c_key_dim], dtype=tf.float32,
							  initializer=tf.random_normal_initializer(mean=0.0, stddev=weight_factor) )


	with tf.name_scope(var_scope):
		ph_input = tf.placeholder(tf.float32, shape=(None, input_dim), name='input')
		t_y = tf.nn.l2_normalize(tf.matmul(ph_input, v_W), dim=1, name='t_y_')

	return ph_input, v_W, t_y

def create_tmpl_dml_tensors(t_y, var_scope):
	# output and errors should depend on the list of igg
	ph_numrecs = tf.placeholder(tf.int32, shape=(), name='ph_numrecs_'+var_scope)
	# ph_o should be shape [numrecs, num_ggs] where num_ggs is the number of graduated ggs for the template
	# ph_o = tf.placeholder(tf.float32, shape=([None, None]), name='ph_o_'+var_scope)
	ph_o = tf.placeholder(tf.float32, shape=([None]), name='ph_o_'+var_scope)
	# v_o = tf.Variable(tf.constant(ovec_norm.astype(np.float32)), dtype=tf.float32, trainable=False, name='v_o')
	v_r1 = tf.Variable(	tf.zeros([config.c_rsize], dtype=tf.int32),
						trainable=False, name='v_r1')
	v_r2 = tf.Variable(	tf.zeros([config.c_rsize], dtype=tf.int32),
						trainable=False, name='v_r2')
	op_r1 = tf.assign(	v_r1, tf.random_uniform([config.c_rsize], minval=0, maxval=ph_numrecs, dtype=tf.int32),
						name='op_r1')
	op_r2 = tf.assign(v_r2, tf.random_uniform([config.c_rsize], minval=0, maxval=ph_numrecs, dtype=tf.int32),
						name='op_r2')

	t_o1 = tf.gather(ph_o, v_r1, name='t_o1')
	t_o2 = tf.gather(ph_o, v_r2, name='t_o2')

	t_y1 = tf.gather(t_y, v_r1, name='t_y1')
	t_y2 = tf.gather(t_y, v_r2, name='t_y2')

	t_cdo = tf.where(tf.equal(t_o1, t_o2), tf.ones([config.c_rsize], dtype=tf.float32), tf.zeros([config.c_rsize], dtype=tf.float32), name='t_cdo')
	# t_cdo = tf.reduce_sum(tf.multiply(t_o1, t_o2), axis=1, name='t_cdo')
	t_cdy = tf.reduce_sum(tf.multiply(t_y1, t_y2), axis=1, name='t_cdy')
	t_err = tf.reduce_mean((t_cdo - t_cdy) ** 2, name='t_err')
	op_train_step = tf.train.GradientDescentOptimizer(config.FLAGS.nn_lrn_rate).minimize(t_err, name='op_train_step')

	return op_train_step, t_err, v_r1, v_r2, op_r1, op_r2, ph_numrecs, ph_o

def init_templ_learn():
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	# new proposed saver:
	# saver_dict = dict()
	# saver = tf.train.Saver(saver_dict, max_to_keep=3)

	# saver_dict = {'W_'+str(i): W for i, W in enumerate(l_W_all)}

	# # saver = tf.train.Saver(saver_dict, max_to_keep=3)
	# ckpt = None
	# if config.FLAGS.save_dir:
	# 	ckpt = tf.train.get_checkpoint_state(config.FLAGS.save_dir)
	# if ckpt and ckpt.model_checkpoint_path:
	# 	logger.info('restoring from %s', ckpt.model_checkpoint_path)
	# 	saver.restore(sess, ckpt.model_checkpoint_path)

	if config.FLAGS.debug:
		sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type="curses")
		sess.add_tensor_filter("stop_reached", stop_reached)

	return sess #, saver_dict, saver

def create_db_from_W(nd_W, templ_perm_arr, igg_arr):
	numrecs = len(igg_arr)
	if numrecs < 1:
		return None

	if config.c_b_nbns:
		new_arr = [modify_vec_for_success(templ_perm_arr[irec]) if igg_arr[irec] else templ_perm_arr[irec] for irec in range(numrecs)]
		nd_perm_arr = np.stack(new_arr, axis=0)
	else:
		nd_perm_arr = np.stack(templ_perm_arr, axis=0)
	# t_y = tf.nn.l2_normalize(tf.matmul(ph_input, v_W), dim=1, name='t_y_')
	nd_y = np.matmul(nd_perm_arr, nd_W)
	nd_norms = np.linalg.norm(nd_y, axis=1)
	nd_db = nd_y / nd_norms[:, None]
	return nd_db

def do_gg_learn(sess, learn_params, perm_arr, igg_arr, b_must_learn):
	b_success = True
	ph_input, v_W, t_y, op_train_step, t_err, v_r1, v_r2, op_r1, op_r2, ph_numrecs, ph_o = learn_params
	numrecs = len(igg_arr)
	# print('numrecs:', numrecs, 'igg_arr:', igg_arr)
	# time.sleep(1)
	sess.run(tf.global_variables_initializer())
	# new_arr = []
	# for irec in range(numrecs):
	# 	if igg_arr[irec]:
	# 		new_arr.append(modify_vec_for_success(perm_arr[irec]))
	# 	else:
	# 		new_arr.append(perm_arr[irec])
	if config.c_b_nbns:
		new_arr = [modify_vec_for_success(perm_arr[irec]) if igg_arr[irec] else perm_arr[irec] for irec in range(numrecs)]
		nd_perm_arr = np.stack(new_arr, axis=0)
	else:
		nd_perm_arr = np.stack(perm_arr, axis=0)


	# sess.run(t_for_stop)
	sess.run([op_r1, op_r2], feed_dict={ph_numrecs: numrecs})
	losses = [[sess.run(t_err, feed_dict={ph_numrecs: numrecs, ph_input: nd_perm_arr, ph_o: igg_arr})]]
	# if must learn don't give up till almost the end
	give_up_count =  (config.c_gg_num_learn_steps - (2 * config.c_gg_learn_test_every)) if b_must_learn else config.c_gg_learn_give_up_at
	for step in range(config.c_gg_num_learn_steps):
		sess.run([op_r1, op_r2], feed_dict={ph_numrecs: numrecs})
		if step % config.c_gg_learn_test_every == 0:
			# err =
			err = np.mean(losses)
			losses = []
			print('lrn step ', step, err)
			if err < config.c_gg_learn_good_thresh:
				break
			# if err > config.c_gg_learn_give_up_thresh and step > give_up_count:
			# 	print('do_templ_learn: Giving up on learning!')
			# 	assert False
			# 	b_success = False
			# 	break
		nn_outputs = sess.run([t_err, op_train_step], feed_dict={ph_numrecs: numrecs, ph_input: nd_perm_arr, ph_o: igg_arr})
		losses.append(nn_outputs[0])

	nd_W, nd_y =  sess.run([v_W, t_y], feed_dict={ph_input:nd_perm_arr, ph_o:igg_arr})
	nd_norms = np.linalg.norm(nd_y, axis=1)
	nd_db = nd_y / nd_norms[:, None]
	return nd_W, nd_db, b_success


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
		if one_ivec == []:
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

"""
looks like no longer in use
def get_templ_cds(perm_vec, nd_W, nd_db):
	perm_embed = np.matmul(perm_vec, nd_W)
	en = np.linalg.norm(perm_embed)
	perm_embed = perm_embed / en
	nd_cd = np.matmul(nd_db, perm_embed )
	return nd_cd
"""

# The return value indicaates a match on the gg not whether the match succeeded in matching a result
def get_gg_score(perm_rec, perm_vec, perm_phrases, nd_W, nd_db, igg, igg_arr, thresh_cd,
				 gens_rec, event_result_list, b_blocking,
				 event_result_score_list, templ_len, templ_scvo, gg_blocking,
				 b_gg_confirmed, result_confirmed_list,
				 gg_confirmed_list, success_score, b_score_valid):
	perm_vec = modify_vec_for_success(perm_vec)
	perm_embed = np.matmul(perm_vec, nd_W)
	en = np.linalg.norm(perm_embed)
	perm_embed = perm_embed / en
	nd_cd = np.matmul(nd_db, perm_embed )
	# print('nd_cd', nd_cd)

	max_match_cd, min_match_cd = 0.0, 1.0
	for imatch, one_match in enumerate(igg_arr):
		# second test required because not all perms may have participated in the last learn
		if one_match and imatch < len(nd_cd):
			cd = nd_cd[imatch]
			if cd  > max_match_cd:
				max_match_cd = cd
			if cd < min_match_cd:
				min_match_cd = cd

	print('min cd:', min_match_cd, 'max cd:', max_match_cd, 'threshold:', thresh_cd)

	if min_match_cd < thresh_cd:
		return False, False, []

	# if event_result_list is empty, every gg match means a problem, so find a way to report
	# under all circumstances, the list of unsuccessful matches needs to be created to learn blocking
	generated_result = mr.replace_vars_in_phrase(perm_rec, gens_rec)
	expected_result = generated_result[1:-1]

	if event_result_list == []:
		print('gg_score for null results expected some result.')
		return True, False, [perm_phrases, expected_result]

	b_one_result_matched = False
	if gg_blocking == b_blocking:
		for iresult, event_result in enumerate(event_result_list):
			if mr.match_rec_exact(expected_result, event_result):
				if b_score_valid:
					gg_sig = [success_score, templ_len, templ_scvo, igg, b_blocking]
					event_result_score_list[iresult].append(gg_sig)
					print('Adding score ', event_result_score_list[iresult][-1], 'for event ', event_result)
				else:
					print('Will not add score yet, until we get results for enough eids for ', templ_scvo)
				b_one_result_matched = True
				if b_gg_confirmed and b_score_valid:
					result_confirmed_list[iresult] = True
					if gg_confirmed_list[iresult] == []:
						gg_confirmed_list[iresult] = gg_sig

	if not b_one_result_matched:
		print('Strange. gg matched but no corresponding event in event list')
		if b_gg_confirmed:
			print('Even stranger! The gg is actually confirmed')
		return True, False, [perm_phrases, expected_result]

	return True, True, [perm_phrases, expected_result]

"""
looks like he is not used
def get_score(perm_rec, perm_vec, nd_W, nd_db, gg_list, igg_arr, eid_arr, event_result_list, event_result_score_list, templ_len, templ_scvo):
	perm_embed = np.matmul(perm_vec, nd_W)
	en = np.linalg.norm(perm_embed)
	perm_embed = perm_embed / en
	nd_cd = np.matmul(nd_db, perm_embed )
	print('nd_cd', nd_cd)
	ind = np.argpartition(nd_cd, -config.c_num_k_eval)[-config.c_num_k_eval:]

	igg_sums = np.zeros([len(igg_arr[0])], np.float32)
	for iind, one_ind in enumerate(ind):
		igg_sums += igg_arr[one_ind]

	igg_sums /= float(len(ind))

	for igg, gg in enumerate(gg_list):
		if igg == 0:
			continue
		if igg_sums[igg] > 0.3:
			generated_result = mr.replace_vars_in_phrase(perm_rec, gg.get_gens_rec())
			for iresult, event_result in enumerate(event_result_list):
				if mr.match_rec_exact(generated_result[1:-1], event_result):
					event_result_score_list[iresult].append([igg_sums[igg], templ_len, templ_scvo, igg])

	return event_result_score_list
"""
"""
	# score = 0.0
	success_set, fail_set = set(), set()
	for iind, one_ind in enumerate(ind):
		igg = igg_arr[one_ind]
		if igg == 0:
			b_match_success = False
		else:
			generated_result = mr.get_result_for_cvo_and_rec(perm_rec, gg_list[igg].get_gens_rec())
			b_match_success =  mr.match_rec_exact(generated_result[1:-1], event_result)
		if b_match_success:
			success_set.add(eid_arr[one_ind])
			print('event result match on:', one_ind)
			# score += 1.0
		else:
			fail_set.add(eid_arr[one_ind])

	num_success, num_fail = float(len(success_set)), float(len(fail_set))
	return num_success / (num_success + num_fail)
"""

def l2_norm_arr(nd_arr):
	en = np.linalg.norm(nd_arr)
	return nd_arr / en


def get_score_stats(templ_iperm, perm_vec, nd_W, nd_db, igg_arr, b_always_print=False):
	rnd_for_print = random.random()
	perm_embed = np.matmul(perm_vec, nd_W)
	en = np.linalg.norm(perm_embed)
	perm_embed = perm_embed / en
	nd_cd = np.matmul(nd_db, perm_embed )
	if rnd_for_print < 0.02:
		print('#', templ_iperm, ': nd_cd', nd_cd)
	# ind = np.argpartition(nd_cd, -config.c_num_k_eval)[-config.c_num_k_eval:]
	max_match_cd, min_match_cd = 0.0, 1.0
	for imatch, one_match in enumerate(igg_arr):
		if one_match:
			cd = nd_cd[imatch]
			if cd  > max_match_cd:
				max_match_cd = cd
			if cd < min_match_cd:
				min_match_cd = cd

	if b_always_print or rnd_for_print < 0.02:
		print('min cd:', min_match_cd, 'max cd:', max_match_cd)
	return min_match_cd

	# igg_sums = np.zeros([len(igg_arr[0])], np.float32)
	# cd_sum = 0.0
	# for iind, one_ind in enumerate(ind):
	# 	igg_sums += igg_arr[one_ind]
	# 	cd_sum += nd_cd[one_ind]
	#
	# igg_sums /= float(len(ind))
	# cd_sum /= float(len(ind))
	# print('igg_sums:', igg_sums, 'cd_sum:', cd_sum)
	#
	# return igg_sums, cd_sum

def modify_vec_for_success(vec):
	vec = np.multiply(vec, np.absolute(vec))
	en = np.linalg.norm(vec)
	vec = vec / en
	return vec

rtesters = []
g_b_eval_compare_numrecs = -1

def eval_compare_conts_learn(nd_W, nd_data, nd_matches, numrecs):
	global g_b_eval_compare_numrecs
	global rtesters

	if g_b_eval_compare_numrecs != numrecs:
		g_b_eval_compare_numrecs = numrecs
		rtesters = random.sample(xrange(numrecs), config.c_cont_lrn_num_testers)

	nd_keys = np.dot(nd_data, nd_W)
	en = np.linalg.norm(nd_keys, axis=1)
	nd_keys = nd_keys / en[:, None]

	nd_test_keys = nd_keys[rtesters]

	score = 0.0
	num_scoring = 0.0
	for tester in rtesters:
		test_vec = nd_keys[tester]
		cd = [[np.dot(test_vec, one_vec), ione] for ione, one_vec in enumerate(nd_keys)]
		cands = sorted(cd, key=lambda x: x[0], reverse=True)[1:config.c_cont_lrn_num_cd_winners+1]
		cd_winners = [cand[1] for cand in cands]
		winner_matches = (nd_matches[cd_winners]).astype(np.float32)
		prediction = np.average(winner_matches)
		real_result = nd_matches[tester].astype(np.float32)
		num_scoring += 1.0
		score += abs(prediction - real_result)

	score = 1.0 - (score/num_scoring)
	print('Eval of cont learn:', score)
	return score

def do_compare_conts_learn(mgr, stat_list):
	reclen = len(stat_list)
	mlist = mgr.get_match_list()
	# mlist = [b == 'True'for b in mlist]
	numrecs = len(mlist)
	if numrecs < config.c_cont_lrn_num_testers:
		print('do_compare_conts_learn: Too few recs to learn')
		return

	data = np.ndarray(shape=[reclen, numrecs], dtype=np.float32)
	for icont, cont_stat in enumerate(stat_list):
		data[icont, :] = [1.0 if b else 0.0 for b in cont_stat.get_match_list()]
	matches = np.ndarray(shape=[numrecs], dtype=np.bool)
	matches[:] = mlist

	data = np.transpose(data)
	en = np.linalg.norm(data, axis=1)
	nz = np.nonzero(en)
	z = np.where(en == 0.0)
	zero_matches = matches[z]
	en, data, matches = en[nz], data[nz], matches[nz]
	numrecs = data.shape[0]
	data = data / en[:, None]

	t_data = tf.constant(data, dtype=tf.float32)
	t_matches = tf.constant(matches, dtype=tf.bool)

	v_r1 = tf.Variable(tf.zeros([config.c_cont_lrn_batch_size], dtype=tf.int32), dtype=tf.int32)
	op_r1_assign = tf.assign(v_r1, tf.random_uniform([config.c_cont_lrn_batch_size], dtype=tf.int32,
													 minval=0, maxval=numrecs-1))
	v_r2 = tf.Variable(tf.zeros([config.c_cont_lrn_batch_size], dtype=tf.int32), dtype=tf.int32)
	op_r2_assign = tf.assign(v_r2, tf.random_uniform([config.c_cont_lrn_batch_size], dtype=tf.int32,
													 minval=0, maxval=numrecs - 1))

	t_m1 = tf.gather(t_matches, v_r1, name='t_m1')
	t_m2 = tf.gather(t_matches, v_r2, name='t_m2')

	weight_factor = 1.0 / tf.cast(reclen * config.c_cont_lrn_key_dim, dtype=tf.float32)
	# t_shape = tf.constant([2], dtype=tf.int32)
	with tf.variable_scope('cont_lrn', reuse=False):
		# v_W = tf.Variable(tf.random_normal(shape=[num_inputs, c_key_dim], mean=0.0, stddev=weight_factor), dtype=tf.float32)
		v_W = tf.get_variable('v_W', shape=[reclen, config.c_cont_lrn_key_dim], dtype=tf.float32,
							  initializer=tf.random_normal_initializer(mean=0.0, stddev=weight_factor))

	t_y = tf.nn.l2_normalize(tf.matmul(t_data, v_W), dim=1, name='t_y')

	t_y1 = tf.gather(t_y, v_r1, name='t_y1')
	t_y2 = tf.gather(t_y, v_r2, name='t_y2')

	t_cdo = tf.where(tf.equal(t_m1, t_m2), tf.ones([config.c_cont_lrn_batch_size], dtype=tf.float32), tf.zeros([config.c_cont_lrn_batch_size], dtype=tf.float32), name='t_cdo')
	t_cdy = tf.reduce_sum(tf.multiply(t_y1, t_y2), axis=1, name='t_cdy')
	t_err = tf.reduce_mean((t_cdo - t_cdy) ** 2, name='t_err')
	op_train_step = tf.train.AdamOptimizer(config.FLAGS.cont_nn_lrn_rate).minimize(t_err, name='op_train_step')

	l_batch_assigns = [op_r1_assign, op_r2_assign]

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	eval_compare_conts_learn(sess.run(v_W), data, matches, numrecs)
	losses = []
	meta_losses = []
	num_stalled = 0
	for step in range(config.c_num_steps):
		sess.run(l_batch_assigns)
		if step == 0:
			errval = math.sqrt(sess.run(t_err))
			logger.info('Starting error: %f', errval)
		elif step % (config.c_num_steps / 100) == 0:
			errval = np.mean(losses)
			meta_losses.append(errval)
			losses = []
			logger.info('step: %d: error: %f', step, errval)
			eval_compare_conts_learn(sess.run(v_W), data, matches, numrecs)
			# if errval < config.c_cont_lrn_stop_thresh:
			# 	break
			if len(meta_losses) > 10:
				meta_losses = meta_losses[1:]
				meta_err = np.mean(meta_losses)
				if errval > meta_err * 0.99:
					num_stalled += 1
				else:
					num_stalled = 0
				if num_stalled > 2:
					break

		outputs = sess.run([t_err, op_train_step])
		losses.append(math.sqrt(outputs[0]))

	mgr.set_W(sess.run(v_W))

	sess.close()
	learn_reset()

	return