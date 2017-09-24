import tensorflow as tf

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
tf.flags.DEFINE_bool('learn', False,
					 'learn rather than test ')

batch_size = 256
max_names = 5
max_objects = 5
max_countries = 5
c_num_inputs = 3
c_num_outputs = 3
c_rsize = ((max_names*max_objects)**2) / 8
if FLAGS.learn:
	c_num_steps = 100000
else:
	c_num_steps = 0
c_eval_db_factor = 1 # fraction of database to consider
c_test_pct = 0.1
c_key_num_ks = 5
c_max_vars = 15
c_key_dim = 15
c_max_phrases_per_rule = 25
c_story_len = 5
c_ovec_len = 200

# logger = None

actions = ['picked up', 'put down', 'has', 'went to', 'is located in', 'is free in']

