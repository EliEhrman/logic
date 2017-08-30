from enum import Enum
from time import gmtime, strftime
import sys
import os
import logging
import csv
import random
import math
import collections
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_bool('debug', False,
					 'call tfdbg ')
tf.flags.DEFINE_bool('heavy', True,
					 'run on a serious GPGPU ')
tf.flags.DEFINE_bool('use_log_file', False,
					 'send output to pylog.txt ')
tf.flags.DEFINE_float('nn_lrn_rate', 0.003,
					 'base learning rate for nn ')
tf.flags.DEFINE_string('save_dir', '/tmp/logicmodels',
					   'directory to save model to. If empty, dont save')
tf.flags.DEFINE_float('fine_thresh', 0.00,
					 'once error drops below this switch to 50% like pairs ')

batch_size = 256
max_names = 5
max_objects = 5
max_countries = 5
c_num_inputs = 3
c_num_outputs = 3
c_rsize = ((max_names*max_objects)**2) / 8
c_num_steps = 0 # 100000
c_eval_db_factor = 1 # fraction of database to consider
c_test_pct = 0.1
c_key_num_ks = 5
c_max_vars = 10
c_key_dim = 15

logger = None

# varmod is a var whose object was in the input and is the only value to be modified
# mod is a field, normally seen in output, that tells you how to modify the database
# conn is a connective such as AND or OR
# var says this is a var, actual value defined in varobj
# varobj is a field with both a var id and an object, this providing the data and giving it a var name
# obj is just an object. It will not be repeated or used in a var

# mod is one of dm_type and says how to change the database
df_type = Enum('df_type', 'bool var obj varobj varmod mod conn')
actions = ['picked up', 'put down', 'has', 'went to', 'is located at']
dm_type = Enum('dm_type', 'Insert Remove Modify')
conn_type = Enum('conn_type', 'AND OR')

# df_type_bool = 0
# df_type_var = 1
# df_type_obj = 2
# df_type_varobj = 3
num_df_types = len(df_type)
num_dm_types = len(dm_type)
num_conn_types = len(conn_type)

SDataField = collections.namedtuple('SDataField', 'df_type, var_num')
SDataField.__new__.__defaults__ = (None, None, None)

def stop_reached(datum, tensor):
	if datum.node_name == 't_for_stop': # and tensor > 100:
		return True
	return False

t_for_stop = tf.constant(5.0, name='t_for_stop')

def make_vec(arr, data_flds, els_arr):
	global logger
	numrecs = len(arr)
	num_els = len(els_arr)
	field_id = 0  # what field are we up to
	for el, fld in enumerate(data_flds):
		field_id = min(len(arr[0])-1, field_id)
		# dfa = [arr[i][el] for i in range(numrecs)]
		subvec0 = np.zeros((numrecs, num_df_types))
		subvec0[:, data_flds[el].df_type.value - 1] = 1
		# get_val = lambda f, t: f.value if t else f
		# a = np.asarray([get_val(arr[i][field_id], (data_flds[el].df_type == df_type.mod)) for i in range(numrecs)])
		a = np.asarray([arr[i][field_id] for i in range(numrecs)])
		if data_flds[el].df_type == df_type.varobj:
			subvec1 = np.zeros((numrecs, c_max_vars))
			subvec1[:, data_flds[el].var_num] = 1
			subvec2 = np.zeros((numrecs, num_els))
			subvec2[np.arange(numrecs), a] = 1
			subvec = np.concatenate((subvec0, subvec1, subvec2), axis=1)
			field_id += 1
		elif data_flds[el].df_type == df_type.obj:
			subvec2 = np.zeros((numrecs, num_els))
			subvec2[np.arange(numrecs), a] = 1
			subvec = np.concatenate((subvec0, subvec2), axis=1)
			field_id += 1
		elif data_flds[el].df_type == df_type.bool:
			subvec1 = np.zeros((numrecs, 2))
			subvec1[np.arange(numrecs), a.astype(np.int64)] = 1
			subvec = np.concatenate((subvec0, subvec1), axis=1)
			field_id += 1
		elif data_flds[el].df_type == df_type.mod:
			subvec1 = np.zeros((numrecs, num_dm_types))
			subvec1[np.arange(numrecs), a.astype(np.int64)] = 1
			subvec = np.concatenate((subvec0, subvec1), axis=1)
			field_id += 1
		elif data_flds[el].df_type == df_type.var:
			subvec1 = np.zeros((numrecs, c_max_vars))
			subvec1[:, data_flds[el].var_num] = 1
			subvec = np.concatenate((subvec0, subvec1), axis=1)
			# note, no increment of field_id
		elif data_flds[el].df_type == df_type.varmod:
			# same as var but indicates a search that must match all other fields
			subvec1 = np.zeros((numrecs, c_max_vars))
			subvec1[:, data_flds[el].var_num] = 1
			subvec = np.concatenate((subvec0, subvec1), axis=1)
			# note, no increment of field_id
		elif data_flds[el].df_type == df_type.conn:
			subvec1 = np.zeros((numrecs, num_conn_types))
			subvec1[np.arange(numrecs), a] = 1
			subvec = np.concatenate((subvec0, subvec1), axis=1)
			field_id += 1
		else:
			logger.error('Invalid field ID. Exiting')
			exit()

		if el == 0:
			vec = subvec
		else:
			vec = np.concatenate((vec, subvec), axis=1)
	return vec

def build_nn(name_scope, t_nn_x, input_dim, b_reuse):
	# num_inputs = tf.shape(t_nn_x)[0]
	weight_factor = 1.0 / tf.cast(input_dim * c_key_dim, dtype=tf.float32)
	# t_shape = tf.constant([2], dtype=tf.int32)
	with tf.variable_scope('nn', reuse=b_reuse):
		# v_W = tf.Variable(tf.random_normal(shape=[num_inputs, c_key_dim], mean=0.0, stddev=weight_factor), dtype=tf.float32)
		v_W = tf.get_variable('v_W', shape=[input_dim, c_key_dim], dtype=tf.float32,
							  initializer=tf.random_normal_initializer(mean=0.0, stddev=weight_factor))

	with tf.name_scope(name_scope):
		t_y = tf.nn.l2_normalize(tf.matmul(t_nn_x, v_W), dim=1, name='t_y')

	return v_W, t_y


def init_logging():
	logger = logging.getLogger('logic')
	ch = logging.StreamHandler(stream=sys.stdout)
	ch.setLevel(logging.DEBUG)
	logger.addHandler(ch)
	logger.setLevel(logging.DEBUG)
	logger.info('Starting at: %s', strftime("%Y-%m-%d %H:%M:%S", gmtime()))
	return logger

# distinguishing between objs and objects. The former is generic. The latter are things you pick up, take etc

def init_els(els_dict, els_arr, def_article, fname=None, alist=None, new_def_article=False, cap_first=False, max_new=5):
	num_els = len(els_arr)
	if fname != None:
		fh_names = open(fname, 'rb')
		fr_names = csv.reader(fh_names, delimiter=',')
		if cap_first:
			new_els = [new_id.lower().title() for lname in fr_names for new_id in lname]
		else:
			new_els = [new_id.lower() for lname in fr_names for new_id in lname]

		random.shuffle(new_els)
		new_els = new_els[0:min(max_new, len(new_els))]
	else:
		new_els = alist

	els_arr += new_els
	for iel, el_name in enumerate(new_els):
		els_dict[el_name] = num_els + iel
	def_article += [new_def_article for id in new_els]
	num_new_els = len(new_els)
	new_els_range = range(num_els, num_els+num_new_els)
	num_els += num_new_els
	return [new_els_range, num_new_els, new_els], num_els

def gen_phrases(gen_rules, els_dict, els_arr):
	curr_flds_id = -1

	input_flds_arr = []
	output_flds_arr = []
	vars_dict_arr = []
	fld_def_arr = []

	input = []
	output = []
	for i_and_o_rules in gen_rules:
		curr_flds_id += 1
		vars_dict = {}
		numrecs = 1
		for ii, rule in enumerate(i_and_o_rules):
			flds = []
			for fld_rule in rule:
				els_set , df_type, sel_el, var_id = fld_rule
				flds.append(SDataField(df_type=df_type, var_num=var_id))
				if ii == 0: # i rule
					if sel_el != None or els_set == [] or els_set[1] == None or els_set[1] <= 1:
						numrecs *= 1
					else:
						numrecs *= els_set[1]
			if ii == 0: # i rule
				field_id = 0
				for el, fld in enumerate(flds):
					type = flds[el].df_type
					if type == df_type.varobj:
						vars_dict[flds[el].var_num] = field_id
						field_id += 1
					elif type == df_type.obj or type == df_type.bool or type == df_type.mod or type == df_type.conn:
						field_id += 1
				input_flds = flds
			else:
				output_flds = flds

		for ii, rule in enumerate(i_and_o_rules):
			recs = [[] for i in range(numrecs)]
			for ifrule, fld_rule in enumerate(rule):
				els_set, df_type, sel_el, var_id = fld_rule
				if ii == 0 and sel_el == None:
						numrecdiv = 1
						for ifrcont in range(ifrule+1, len(rule)):
							els_set2, df_type2, sel_el2, var_id2 = rule[ifrcont]
							if sel_el2 == None:
								numrecdiv *= els_set2[1]
						recval = -1
						for irec in range(numrecs):
							# recval = (irec % numrecmod) / numrecdiv
							if irec % numrecdiv == 0:
								recval += 1
							if recval == els_set[1]:
								recval = 0
							recs[irec].append(els_set[0][recval])
				# following if applies to both input ad output
				elif df_type == df_type.obj:
					for irec in range(numrecs):
						recs[irec].append(els_dict[sel_el])
				elif df_type == df_type.bool:
					for irec in range(numrecs):
						recs[irec].append(int(sel_el))
				elif df_type == df_type.mod or df_type == df_type.conn:
					for irec in range(numrecs):
						recs[irec].append(sel_el.value-1)
			if ii == 0:
				ivec = make_vec(recs, input_flds, els_arr)
				fld_def_arr.extend([curr_flds_id] * len(recs))
				input += recs
			else:
				ovec = make_vec(recs, output_flds, els_arr)
				output += recs

		# add up ivec, ovec, input and output. Makes sure everybody gets the shuffle stick
		if curr_flds_id == 0:
			all_ivecs = ivec
			all_ovecs = ovec
		else:
			all_ivecs = np.concatenate((all_ivecs, ivec), axis=0)
			all_ovecs = np.concatenate((all_ovecs, ovec), axis=0)

		input_flds_arr.append(input_flds)
		output_flds_arr.append(output_flds)
		vars_dict_arr.append(vars_dict)


	numallrecs = len(input)
	shuffle_stick = range(numallrecs)
	random.shuffle(shuffle_stick)
	input = [input[i] for i in shuffle_stick]
	output = [output[i] for i in shuffle_stick]
	fld_def_arr = [fld_def_arr[i] for i in shuffle_stick]
	all_ivecs = np.take(all_ivecs, shuffle_stick, axis=0)
	all_ovecs = np.take(all_ovecs, shuffle_stick, axis=0)

	return 	input_flds_arr, output_flds_arr, vars_dict_arr, fld_def_arr, input, output, all_ivecs, all_ovecs


els_arr = []
els_dict = {}
def_article = []
# name_ids, num_els, num_names, names = \
name_set, num_els = \
	init_els(	fname='names.txt', cap_first=True,
				max_new=max_names, els_dict=els_dict, els_arr=els_arr, def_article=def_article)
# object_ids, num_els, num_objects, objects = \
object_set, num_els = \
	init_els(	fname='objects.txt', new_def_article=True,
				max_new=max_objects, els_dict=els_dict, els_arr=els_arr, def_article=def_article)
place_set, num_els =\
	init_els(	fname='countries.txt', cap_first=True,
				max_new=max_countries, els_dict=els_dict, els_arr=els_arr, def_article=def_article)

action_set, num_els =\
	init_els(	alist=actions,
				max_new=max_objects, els_dict=els_dict, els_arr=els_arr, def_article=def_article)

"""
def make_pickups(input_flds_arr, output_flds_arr, vars_dict_arr, fld_def_arr):
	if fld_def_arr == []:
		curr_flds_id = 0
	else:
		curr_flds_id = fld_def_arr[-1] + 1

	input_flds = [	SDataField(df_type=df_type.varobj, var_num=0),
					SDataField(df_type=df_type.obj),
					SDataField(df_type=df_type.varobj, var_num=1)]
	output_flds = [	SDataField(df_type=df_type.mod),
					SDataField(df_type=df_type.var, var_num=0),
					SDataField(df_type=df_type.obj),
					SDataField(df_type=df_type.var, var_num=1)
					]
	vars_dict = dict()
	field_id = 0
	for el, fld in enumerate(input_flds):
		if input_flds[el].df_type == df_type.varobj:
			vars_dict[input_flds[el].var_num] = field_id
			field_id += 1
		elif input_flds[el].df_type == df_type.obj or input_flds[el].df_type == df_type.bool:
			field_id += 1
	# note, no increment of field_id


	input1 = [[aname, action_ids[0], anobject] for aname in name_ids for anobject in object_ids]
	output1 = [[dm_type.Insert, action_ids[2]] for aname in name_ids for anobject in object_ids]
	input2 = [[aname, action_ids[1], anobject] for aname in name_ids for anobject in object_ids]
	output2 = [[dm_type.Remove, action_ids[2]] for aname in name_ids for anobject in object_ids]
	input = input1 + input2
	output = output1 + output2

	ivec = make_vec(input, input_flds, els_arr)
	ovec = make_vec(output, output_flds, els_arr)

	input_flds_arr.append(input_flds)
	output_flds_arr.append(output_flds)
	vars_dict_arr.append(vars_dict)

	fld_def_arr.extend([curr_flds_id] * len(input))

	return input, output, ivec, ovec
"""
input = []
output = []

def make_fld(els_set, df_type, sel_el=None, var_id=None):
	return [els_set, df_type, sel_el, var_id]

gen_rules = []
gen_rules += [[	[	make_fld(els_set=name_set, df_type=df_type.varobj, var_id=0),
					make_fld(els_set=action_set, df_type=df_type.obj, sel_el='picked up'),
					make_fld(els_set=object_set, df_type=df_type.varobj, var_id=1)],
				[	make_fld(els_set=[], df_type=df_type.mod, sel_el=dm_type.Insert),
					make_fld(els_set=[], df_type=df_type.var, var_id=0),
					make_fld(els_set=action_set, df_type=df_type.obj, sel_el='has'),
					make_fld(els_set=[], df_type=df_type.var, var_id=1),
					 ]]]
gen_rules += [[	[	make_fld(els_set=name_set, df_type=df_type.varobj, var_id=0),
					make_fld(els_set=action_set, df_type=df_type.obj, sel_el='put down'),
					make_fld(els_set=object_set, df_type=df_type.varobj, var_id=1)],
				[	make_fld(els_set=[], df_type=df_type.mod, sel_el=dm_type.Remove),
					make_fld(els_set=[], df_type=df_type.var, var_id=0),
					make_fld(els_set=action_set, df_type=df_type.obj, sel_el='has'),
					make_fld(els_set=[], df_type=df_type.var, var_id=1),
					 ]]]
gen_rules += [[	[	make_fld(els_set=name_set, df_type=df_type.varobj, var_id=0),
					make_fld(els_set=action_set, df_type=df_type.obj, sel_el='went to'),
					make_fld(els_set=place_set, df_type=df_type.varobj, var_id=1)],
				[	make_fld(els_set=[], df_type=df_type.mod, sel_el=dm_type.Modify),
					make_fld(els_set=[], df_type=df_type.var, var_id=0),
					make_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located at'),
					make_fld(els_set=[], df_type=df_type.varmod, var_id=1),
					 ]]]

input_flds_arr, output_flds_arr, vars_dict_arr, fld_def_arr, input, output, ivec, ovec = \
	gen_phrases(gen_rules, els_dict=els_dict, els_arr=els_arr)
"""
# pickup_input, pickup_output, pickup_ivec, pickup_ovec = make_pickups(input_flds, output_flds, vars_dict, fld_def_arr)
input.extend(pickup_input)
output.extend(pickup_output)
ivec = pickup_ivec
ovec = pickup_ovec
# else np.concatenate
"""
numrecs = len(input)
input_dim = ivec.shape[1]
# extra = np.asarray([[1.0, 0.0] if output[i][3] else [0.0, 1.0] for i in range(numrecs)])
# ovec = np.concatenate((ovec, extra), axis=1)

norm = lambda vec: vec / np.linalg.norm(vec, axis=1, keepdims=True)

ivec_norm = norm(ivec)
ovec_norm = norm(ovec)
v_x = tf.Variable(tf.constant(ivec_norm.astype(np.float32)), dtype=tf.float32, trainable=False, name='v_x')
v_o = tf.Variable(tf.constant(ovec_norm.astype(np.float32)), dtype=tf.float32, trainable=False, name='v_o')
v_r1 = tf.Variable(tf.random_uniform([c_rsize], minval=0, maxval=numrecs-1, dtype=tf.int32),
				   trainable=False, name='v_r1')
v_r2 = tf.Variable(tf.random_uniform([c_rsize], minval=0, maxval=numrecs-1, dtype=tf.int32),
				   trainable=False, name='v_r2')

t_x1 = tf.gather(v_x, v_r1, name='t_x1')
t_x2 = tf.gather(v_x, v_r2, name='t_x2')
t_o1 = tf.gather(v_o, v_r1, name='t_o1')
t_o2 = tf.gather(v_o, v_r2, name='t_o2')
# t_y = tf.matmul(t_x, tf.clip_by_value(v_W, 0.0, 10.0), name='t_y') # + b
v_W, t_y = build_nn('main', v_x, input_dim, b_reuse=False)

t_y1 = tf.gather(t_y, v_r1, name='t_y1')
t_y2 = tf.gather(t_y, v_r2, name='t_y2')

t_cdo = tf.reduce_sum(tf.multiply(t_o1, t_o2), axis=1, name='t_cdo')
t_cdy = tf.reduce_sum(tf.multiply(t_y1, t_y2), axis=1, name='t_cdy')
t_err = tf.reduce_mean((t_cdo - t_cdy) ** 2, name='t_err')
op_train_step = tf.train.AdamOptimizer(FLAGS.nn_lrn_rate).minimize(t_err, name='op_train_step')

v_y = tf.Variable(tf.zeros([numrecs, c_key_dim], dtype=tf.float32), name='v_y')
op_y = tf.assign(v_y, t_y, name='op_y')
db_size = int(float(numrecs/ c_eval_db_factor) * (1.0 - c_test_pct))
test_size = numrecs - db_size # int(float(numrecs / c_eval_db_factor) * c_test_pct)
t_key_db = tf.slice(input_=v_y, begin=[0, 0],
				  size=[db_size, c_key_dim],
				  name='t_key_db')
t_key_test = tf.slice(input_=v_y,
					begin=[db_size, 0],
					size=[test_size, c_key_dim], name='t_key_test')
t_eval_key_cds =  tf.matmul(t_key_test, t_key_db, transpose_b=True, name='t_eval_key_cds')
t_key_cds, t_key_idxs = tf.nn.top_k(t_eval_key_cds, c_key_num_ks, sorted=True, name='t_keys')

logger = init_logging()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver({"W":v_W}, max_to_keep=3)
ckpt = None
if FLAGS.save_dir:
	ckpt = tf.train.get_checkpoint_state(FLAGS.save_dir)
if ckpt and ckpt.model_checkpoint_path:
	logger.info('restoring from %s', ckpt.model_checkpoint_path)
	saver.restore(sess, ckpt.model_checkpoint_path)

if FLAGS.debug:
	sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type="curses")
	sess.add_tensor_filter("stop_reached", stop_reached)

losses = []

for step in range(c_num_steps):
	sess.run(tf.variables_initializer([v_r1, v_r2]))
	if step == 0:
		errval = math.sqrt(sess.run(t_err))
		logger.info('Starting error: %f', errval)
	elif step % (c_num_steps / 1000) == 0:
		errval = np.mean(losses)
		losses = []
		logger.info('step: %d: error: %f', step, errval)
	if step % (c_num_steps / 100) == 0:
		if saver and FLAGS.save_dir:
			logger.info('saving v_W to %s', FLAGS.save_dir)
			saved_file = saver.save(sess,
									os.path.join(FLAGS.save_dir, 'model.ckpt'),
									step)
	outputs = sess.run([t_err, op_train_step])
	losses.append(math.sqrt(outputs[0]))

sess.run(op_y)
r_keys = sess.run([t_key_cds, t_key_idxs])
logger.info(r_keys)

def print_phrase(output_flds, phrase, vars_dict, out_phrase, out_str):
	field_id = 0  # what field are we up to
	for el, fld in enumerate(output_flds):
		if fld.df_type == df_type.varobj:
			field_id += 1
		elif fld.df_type == df_type.obj:
			out_str += els_arr[out_phrase[field_id]] + ' '
			field_id += 1
		elif fld.df_type == df_type.mod:
			if out_phrase[field_id] == dm_type.Insert.value - 1:
				out_str = 'Insert: '
			elif out_phrase[field_id] == dm_type.Remove.value - 1:
				out_str = 'Remove: '
			elif out_phrase[field_id] == dm_type.Modify.value - 1:
				out_str = 'Modify: '
			field_id += 1
		elif fld.df_type == df_type.bool:
			if out_phrase[field_id]:
				out_str = 'true that '
			else:
				out_str = 'false that '
			field_id += 1
		elif fld.df_type == df_type.var:
			input_id = vars_dict[output_flds[el].var_num]
			if def_article[phrase[input_id]]:
				out_str += 'the '
			out_str += els_arr[phrase[input_id]] + ' '
		# note, no increment of field_id
		elif fld.df_type == df_type.varmod:
			input_id = vars_dict[output_flds[el].var_num]
			out_str += '*'
			if def_article[phrase[input_id]]:
				out_str += 'the '
			out_str += els_arr[phrase[input_id]] + '* '
		# note, no increment of field_id
		else:
			logger.error('Invalid field ID. Exiting')
			exit()

	return out_str


for iiphrase, phrase in enumerate( input[db_size:]):
	fld_def = fld_def_arr[db_size + iiphrase]
	input_flds, output_flds, vars_dict = input_flds_arr[fld_def], output_flds_arr[fld_def], vars_dict_arr[fld_def],
	out_phrase = output[r_keys[1][iiphrase][0]]
	out_str = "input: "
	for id in phrase:
		if def_article[id]:
			out_str += "the "
		out_str += els_arr[id] + ' '
	logger.info(out_str)
	for iout in range(c_key_num_ks-1):
		out_str = print_phrase(output_flds, phrase, vars_dict, out_phrase, out_str)
		logger.info(out_str)


		field_id = 0  # what field are we up to
		for el, fld in enumerate(output_flds):
			if fld.df_type == df_type.varobj:
				field_id += 1
			elif fld.df_type == df_type.obj:
				out_str += els_arr[out_phrase[field_id]] + ' '
				field_id += 1
			elif fld.df_type == df_type.mod:
				if out_phrase[field_id] == dm_type.Insert.value - 1:
					out_str = 'Insert: '
				elif out_phrase[field_id] == dm_type.Remove.value - 1:
					out_str = 'Remove: '
				elif out_phrase[field_id] == dm_type.Modify.value - 1:
					out_str = 'Modify: '
				field_id += 1
			elif fld.df_type == df_type.bool:
				if out_phrase[field_id]:
					out_str = 'true that '
				else:
					out_str = 'false that '
				field_id += 1
			elif fld.df_type == df_type.var:
				input_id = vars_dict[output_flds[el].var_num]
				if def_article[phrase[input_id]]:
					out_str += 'the '
				out_str += els_arr[phrase[input_id]] + ' '
				# note, no increment of field_id
			elif fld.df_type == df_type.varmod:
				input_id = vars_dict[output_flds[el].var_num]
				out_str += '*'
				if def_article[phrase[input_id]]:
					out_str += 'the '
				out_str += els_arr[phrase[input_id]] + '* '
			# note, no increment of field_id
			else:
				logger.error('Invalid field ID. Exiting')
				exit()
		logger.info(out_str)

sess.close()

logger.info('done.')
