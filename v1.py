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
c_num_inputs = 3
c_num_outputs = 3
c_rsize = ((max_names*max_objects)**2) / 8
c_num_steps = 0 # 100000
c_eval_db_factor = 1 # fraction of database to consider
c_test_pct = 0.1
c_key_num_ks = 5
c_max_vars = 10
c_key_dim = 15

df_type = Enum('df_type', 'bool var obj varobj')

# df_type_bool = 0
# df_type_var = 1
# df_type_obj = 2
# df_type_varobj = 3
num_df_types = len(df_type)

SDataField = collections.namedtuple('SDataField', 'df_type, var_num')
SDataField.__new__.__defaults__ = (None, None, None)

def stop_reached(datum, tensor):
	if datum.node_name == 't_for_stop': # and tensor > 100:
		return True
	return False

t_for_stop = tf.constant(5.0, name='t_for_stop')

def make_vec(arr, data_flds):
	field_id = 0  # what field are we up to
	for el, fld in enumerate(data_flds):
		field_id = min(len(arr[0])-1, field_id)
		# dfa = [arr[i][el] for i in range(numrecs)]
		subvec0 = np.zeros((numrecs, num_df_types))
		subvec0[:, data_flds[el].df_type.value - 1] = 1
		a = np.asarray([arr[i][field_id] for i in range(numrecs)])
		if data_flds[el].df_type == df_type.varobj:
			subvec1 = np.zeros((numrecs, c_max_vars))
			subvec1[:, data_flds[el].var_num] = 1
			subvec2 = np.zeros((numrecs, num_ids))
			subvec2[np.arange(numrecs), a] = 1
			subvec = np.concatenate((subvec0, subvec1, subvec2), axis=1)
			field_id += 1
		elif data_flds[el].df_type == df_type.obj:
			subvec2 = np.zeros((numrecs, num_ids))
			subvec2[np.arange(numrecs), a] = 1
			subvec = np.concatenate((subvec0, subvec2), axis=1)
			field_id += 1
		elif data_flds[el].df_type == df_type.bool:
			subvec1 = np.zeros((numrecs, 2))
			subvec1[np.arange(numrecs), a.astype(np.int64)] = 1
			subvec = np.concatenate((subvec0, subvec1), axis=1)
			field_id += 1
		elif data_flds[el].df_type == df_type.var:
			subvec1 = np.zeros((numrecs, c_max_vars))
			subvec1[:, data_flds[el].var_num] = 1
			subvec = np.concatenate((subvec0, subvec1), axis=1)
			# note, no increment of field_id
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




logger = logging.getLogger('logic')
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

logger.setLevel(logging.DEBUG)
logger.info('Starting at: %s', strftime("%Y-%m-%d %H:%M:%S", gmtime()))

fh_names = open('names.txt', 'rb')
fr_names = csv.reader(fh_names, delimiter=',')
names = [name.lower().title() for lname in fr_names for name in lname]
random.shuffle(names)
names = names[0:min(max_names,len(names))]
obj_dict = names
def_article = [False for name in names]
num_names = len(names)
name_ids = range(num_names)
num_ids = num_names

fh_objects = open('objects.txt', 'rb')
fr_objects = csv.reader(fh_objects, delimiter=',')
objects = [object.lower() for lobject in fr_objects for object in lobject]
random.shuffle(objects)
objects = objects[0:min(max_objects,len(objects))]
obj_dict += objects
def_article += [True for object in objects]
num_objects = len(objects)
object_ids = range(num_ids, num_ids+num_objects)
num_ids += num_objects

actions = ['picked up', 'put down', 'has']
obj_dict += actions
def_article += [False for action in actions]
action_ids = range(num_ids, num_ids+len(actions))
num_ids += len(actions)

input_flds = [	SDataField(df_type=df_type.varobj, var_num=0),
				SDataField(df_type=df_type.obj),
				SDataField(df_type=df_type.varobj, var_num=1)]
output_flds = [	SDataField(df_type=df_type.bool),
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
output1 = [[True, action_ids[2]] for aname in name_ids for anobject in object_ids]
input2 = [[aname, action_ids[1], anobject] for aname in name_ids for anobject in object_ids]
output2 = [[False, action_ids[2]] for aname in name_ids for anobject in object_ids]
input = input1 + input2
output = output1 + output2

numrecs = len(input)
shuffle_stick = range(numrecs)
random.shuffle(shuffle_stick)
input = [input[i] for i in shuffle_stick]
output = [output[i] for i in shuffle_stick]

ivec = make_vec(input, input_flds)
ovec = make_vec(output, output_flds)
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
test_size = int(float(numrecs / c_eval_db_factor) * c_test_pct)
t_key_db = tf.slice(input_=v_y, begin=[0, 0],
				  size=[db_size, c_key_dim],
				  name='t_key_db')
t_key_test = tf.slice(input_=v_y,
					begin=[db_size, 0],
					size=[test_size, c_key_dim], name='t_key_test')
t_eval_key_cds =  tf.matmul(t_key_test, t_key_db, transpose_b=True, name='t_eval_key_cds')
t_key_cds, t_key_idxs = tf.nn.top_k(t_eval_key_cds, c_key_num_ks, sorted=True, name='t_keys')


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

for iiphrase, phrase in enumerate( input[db_size:]):
	out_phrase = output[r_keys[1][iiphrase][0]]
	out_str = "input: "
	for id in phrase:
		if def_article[id]:
			out_str += "the "
		out_str += obj_dict[id] + ' '
	logger.info(out_str)
	for iout in range(c_key_num_ks-1):
		field_id = 0  # what field are we up to
		for el, fld in enumerate(output_flds):
			if output_flds[el].df_type == df_type.varobj:
				field_id += 1
			elif output_flds[el].df_type == df_type.obj:
				out_str += obj_dict[out_phrase[field_id]] + ' '
				field_id += 1
			elif output_flds[el].df_type == df_type.bool:
				if out_phrase[field_id]:
					out_str = 'Insert: '
				else:
					out_str = 'Remove: '
				field_id += 1
			elif output_flds[el].df_type == df_type.var:
				input_id = vars_dict[output_flds[el].var_num]
				if def_article[phrase[input_id]]:
					out_str += 'the '
				out_str += obj_dict[phrase[input_id]] + ' '
				# note, no increment of field_id
			else:
				logger.error('Invalid field ID. Exiting')
				exit()
		logger.info(out_str)

sess.close()

logger.info('done.')