"""
This module implements a simple oracle plus learner duo for learning how to modify stories
based on inference from sentences as they come in.

There is no story as yet. The oracle simply feeds the stories to the learner together with their
inferences

Examples are either subject-verb-object or two such connected by an AND

"""
import os
import math
import collections
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import rules
from rules import df_type
from rules import dm_type
from rules import conn_type
import story
import utils
from utils import ulogger as logger
import config
import els

# df_type_bool = 0
# df_type_var = 1
# df_type_obj = 2
# df_type_varobj = 3
num_df_types = len(df_type)
num_dm_types = len(dm_type)
num_conn_types = len(conn_type)

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



# A gen rule is how to generate records
# A fld or fld_def (input_flds or output_flds) defines a record of fields for manipulating and learning records
def gen_phrases(gen_rules, els_dict, els_arr, max_phrases_per_rule):
	global logger
	curr_flds_id = -1

	input_flds_arr = []
	output_flds_arr = []
	fld_def_arr = []
	ivec_arr = []
	ivec_pos_list = []
	ivec_dim_dict = {}
	ivec_dim_by_rule = []

	input = []
	output = []
	for igen, i_and_o_rule in enumerate(gen_rules):
		src_recs, recs = \
			rules.gen_for_rule(b_gen_for_learn=True, rule=i_and_o_rule)
		curr_flds_id += 1
		# ivec = make_vec(src_recs, i_and_o_rule.preconds, els_arr)
		ivec = els.make_vec(src_recs, els_dict)
		fld_def_arr.extend([curr_flds_id] * len(src_recs))
		# len_so_far = len(ivec_arr)
		ivec_dim = ivec.shape[1]
		dict_id = ivec_dim_dict.get(ivec_dim, None)
		if dict_id == None:
			dict_id = len(ivec_dim_dict)
			ivec_dim_dict[ivec_dim] = dict_id
		ivec_pos_list.extend([dict_id for i in recs])
		ivec_dim_by_rule.append(dict_id)
		input += src_recs
		ivec_arr.append(ivec)
		del dict_id, ivec, src_recs

		# ovec = make_vec(recs, i_and_o_rule.gens, els_arr)
		ovec = els.make_vec(recs, els_dict)
		ovec = els.pad_ovec(ovec)
		output += recs
		del recs

		if curr_flds_id == 0:
			# all_ivecs = ivec
			all_ovecs = ovec
		else:
			# all_ivecs = np.concatenate((all_ivecs, ivec), axis=0)
			all_ovecs = np.concatenate((all_ovecs, ovec), axis=0)

		input_flds_arr.append(i_and_o_rule.preconds)
		output_flds_arr.append(i_and_o_rule.gens)

		# end loop over gen rules

	return 	input_flds_arr, output_flds_arr, fld_def_arr, input, output, \
			ivec_pos_list, all_ovecs, ivec_arr, ivec_dim_dict, ivec_dim_by_rule


def init_train_tensors(ovec, ivec_dim_dict, ivec_arr, ivec_dim_by_rule, numrecs):
	# num_ivec_types = len(ivec_dim_dict)
	# input_dim = ivec.shape[1]
	# extra = np.asarray([[1.0, 0.0] if output[i][3] else [0.0, 1.0] for i in range(numrecs)])
	# ovec = np.concatenate((ovec, extra), axis=1)

	norm = lambda vec: vec / np.linalg.norm(vec, axis=1, keepdims=True)


	ovec_norm = norm(ovec)
	l_W = [None] * len(ivec_dim_dict)
	for ivec_dim, dim_pos in ivec_dim_dict.iteritems():
		l_W[dim_pos] = (build_nn(var_scope='nn_'+str(dim_pos), input_dim=ivec_dim, b_reuse=False))

	l_y = []
	for iivec, one_ivec in enumerate(ivec_arr):
		ivec_normed = norm(one_ivec)
		v_x = tf.Variable(tf.constant(ivec_normed.astype(np.float32)), dtype=tf.float32, trainable=False, name='v_x')
		l_y.append(build_nn_run(name_scope='main', t_nn_x=v_x, v_W=l_W[ivec_dim_by_rule[iivec]]))

	t_y =tf.concat(l_y, axis=0, name='t_y')

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

	return op_train_step, t_y, t_err, v_r1, v_r2, l_W


def init_eval_tensors(numrecs, t_y):
	db_size = int(float(numrecs/ config.c_eval_db_factor) * (1.0 - config.c_test_pct))
	test_size = numrecs - db_size # int(float(numrecs / c_eval_db_factor) * c_test_pct)
	v_r_eval = tf.Variable(tf.random_shuffle(tf.range(numrecs)), name='v_r_eval')
	t_r_db = tf.slice(input_=v_r_eval, begin=[0], size=[db_size], name='t_r_db')
	t_r_test = tf.slice(input_=v_r_eval, begin=[db_size], size=[test_size], name='t_r_test')
	v_y = tf.Variable(tf.zeros([numrecs, config.c_key_dim], dtype=tf.float32), name='v_y')
	op_y = tf.assign(v_y, t_y, name='op_y')

	t_key_db = tf.gather(v_y, t_r_db, name='t_key_db')
	t_key_test = tf.gather(v_y, t_r_test, name='t_key_test')

	t_eval_key_cds =  tf.matmul(t_key_test, t_key_db, transpose_b=True, name='t_eval_key_cds')
	t_key_cds, t_r_key_idxs = tf.nn.top_k(t_eval_key_cds, config.c_key_num_ks, sorted=True, name='t_keys')
	t_key_idxs = tf.gather(t_r_db, t_r_key_idxs, name='t_key_idxs')
	return t_r_test, t_key_cds, t_key_idxs, op_y, v_r_eval


def init_learn(l_W):
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	saver_dict = {'W_'+str(i): W for i, W in enumerate(l_W)}

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


def do_init():
	# utils.init_logging()
	# logger = utils.ulogger

	els_arr, els_dict, def_article, num_els, name_set, object_set, place_set, action_set = els.init_objects()
	gen_rules = rules.init_rules(name_set, object_set, place_set, action_set, els_dict)
	# for rule in gen_rules:
	# 	out_str = 'rule print: \n'
	# 	out_str = rules.print_rule(rule, out_str)
	# 	print(out_str)
	story_rules = rules.init_story_rules(name_set, object_set, place_set, action_set)
	query_rules = rules.init_query_rules(name_set, object_set, place_set, els_dict)
	blocking_rules = rules.init_blocking_rules(name_set, object_set, place_set, action_set, els_dict)
	del name_set, object_set, place_set, action_set

	input_flds_arr, output_flds_arr, fld_def_arr, \
	input, output, ivec_pos_list, ovec, ivec_arr, ivec_dim_dict, ivec_dim_by_rule = \
		gen_phrases(gen_rules, els_dict=els_dict, els_arr=els_arr, max_phrases_per_rule=config.c_max_phrases_per_rule)
	story_arr = story.create_story(els_dict, def_article, els_arr, story_rules, query_rules, gen_rules, blocking_rules)
	del els_arr, gen_rules, story_rules

	numrecs = len(input)
	op_train_step, t_y, t_err, v_r1, v_r2, l_W \
		= init_train_tensors(ovec, ivec_dim_dict, ivec_arr, ivec_dim_by_rule, numrecs)
	del ovec, ivec_dim_dict, ivec_arr, ivec_dim_by_rule

	t_r_test, t_key_cds, t_key_idxs, op_y, v_r_eval \
		= init_eval_tensors(numrecs, t_y)
	del t_y, numrecs

	sess, saver = init_learn(l_W)
	del l_W

	return sess, v_r1, v_r2, t_err, saver, op_train_step, \
		   op_y, v_r_eval, t_r_test, t_key_cds, t_key_idxs, \
		   fld_def_arr, input_flds_arr, input, output, output_flds_arr, \
		   def_article, els_dict

def do_learn(sess, v_r1, v_r2, t_err, saver, op_train_step):

	losses = []

	for step in range(config.c_num_steps):
		sess.run(tf.variables_initializer([v_r1, v_r2]))
		if step == 0:
			errval = math.sqrt(sess.run(t_err))
			logger.info('Starting error: %f', errval)
		elif step % (config.c_num_steps / 1000) == 0:
			errval = np.mean(losses)
			losses = []
			logger.info('step: %d: error: %f', step, errval)
		if step % (config.c_num_steps / 100) == 0:
			if saver and config.FLAGS.save_dir:
				logger.info('saving v_W to %s', config.FLAGS.save_dir)
				saved_file = saver.save(sess,
										os.path.join(config.FLAGS.save_dir, 'model.ckpt'),
										step)
		outputs = sess.run([t_err, op_train_step])
		losses.append(math.sqrt(outputs[0]))

def do_eval(sess, op_y, v_r_eval, t_r_test, t_key_cds, t_key_idxs,
			fld_def_arr, input_flds_arr, input, output,
			def_article, els_dict):
	sess.run(op_y)
	sess.run(tf.variables_initializer([v_r_eval]))
	sess.run(t_for_stop)
	r_r_test, r_key_cds, r_key_idxs = sess.run([t_r_test, t_key_cds, t_key_idxs])
	logger.info([r_r_test, r_key_cds, r_key_idxs])


	for itest, iiphrase in enumerate(r_r_test):
		phrase = input[iiphrase]
		fld_def = fld_def_arr[iiphrase]
		input_flds = input_flds_arr[fld_def]
		out_phrase = output[r_key_idxs[itest][0]]
		out_str = "input: "
		out_str = els.print_phrase(phrase, phrase, out_str, def_article, els_dict)
		logger.info(out_str)
		for iout in range(config.c_key_num_ks-1):
			oiphrase = r_key_idxs[itest][iout]
			o_fld_def = fld_def_arr[oiphrase]
			out_str = els.print_phrase(	phrase, output[oiphrase],
									out_str, def_article, els_dict)
			logger.info(out_str)
			del oiphrase

sess, v_r1, v_r2, t_err, saver, op_train_step, \
		op_y, v_r_eval, t_r_test, t_key_cds, t_key_idxs, \
		fld_def_arr, input_flds_arr, input, output, output_flds_arr, \
		def_article, els_dict \
	= do_init()

do_learn(sess, v_r1, v_r2, t_err, saver, op_train_step)

do_eval(sess, op_y, v_r_eval, t_r_test, t_key_cds,
		t_key_idxs, fld_def_arr, input_flds_arr, input, output,
		def_article, els_dict)

sess.close()

logger.info('done.')
