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
tf.flags.DEFINE_bool('learn', True,
					 'learn rather than test ')

batch_size = 256
max_names = 5
max_objects = 5
max_countries = 5
c_num_inputs = 3
c_num_outputs = 3
c_rsize = ((max_names*max_objects)**2) / 8
if FLAGS.learn:
	c_num_steps = 10000
else:
	c_num_steps = 0
c_eval_db_factor = 1 # fraction of database to consider
c_test_pct = 0.1
c_key_num_ks = 5
c_max_vars = 35
c_key_dim = 15
c_max_phrases_per_rule = 25
c_story_len = 350
c_ovec_len = 500
c_story_only = False
c_curriculum_story_len = 1
c_curriculum_num_stories = 3
c_query_story_len = 1
c_query_num_stories = 1
c_eval_story_len = 1
c_eval_num_stories = 1
c_match_batch_size = 3
c_mismatch_batch_size = 5
c_num_k_eval = 5
c_num_clusters = 5
c_kmeans_num_batches = 1
c_kmeans_num_db_segs = 2
c_kmeans_iters = 6



# logger = None

query_action = ['who', 'what', 'where', 'what has', 'what does', 'have', 'where is', 'is in', 'know', 'asked', 'told', 'that', 'asked', 'for']
person_place_action = ['is located in', 'went to']
object_place_action = ['is located in', 'is free in' ]
person_object_dynamic_action = ['picked up', 'put down']
person_object_static_action = ['has', 'wants']
person_person_static_action = ['likes']
person_object_dynamic_3_action = ['gave to']
knowledge_action = ['knows that']

uniquify = lambda llist: list(set(llist))

person_object_action = uniquify(person_object_dynamic_action + person_object_static_action)
actions = uniquify(query_action + person_place_action + object_place_action
				   + person_object_action + person_person_static_action + person_object_dynamic_3_action
				   + knowledge_action)
# actions = ['picked up', 'put down', 'has', 'went to', 'is located in', 'is free in', 'who']

