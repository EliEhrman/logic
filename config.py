import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_bool('debug', False,
					 'call tfdbg ')
tf.flags.DEFINE_bool('heavy', True,
					 'run on a serious GPGPU ')
tf.flags.DEFINE_bool('use_log_file', False,
					 'send output to pylog.txt ')
tf.flags.DEFINE_float('nn_lrn_rate', 0.03,
					 'base learning rate for nn ')
# tf.flags.DEFINE_string('save_dir', '/tmp/logicmodels',
tf.flags.DEFINE_string('save_dir', '',
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
c_query_story_len = 15
c_query_num_stories = 3
c_eval_story_len = 15
c_eval_num_stories = 3
c_match_batch_size = 3
c_mismatch_batch_size = 5
c_num_k_eval = 5
c_num_clusters = 15
c_kmeans_num_batches = 1
c_kmeans_num_db_segs = 2
c_kmeans_iters = 6
c_set_compress_cd_factor = 0.8
c_cascade_level = 3 # should be at least 3
c_cascade_max_phrases = 3 # should not be shorter than the longest rule of the oracle
c_rule_cluster_thresh_levels = [1.0, 0.7, 0.2, -0.5] # [1.0, -0.5] #
c_gg_graduate_len = 10
c_gg_validate_thresh = 10 # number of points to accumulate before finding missing surprising
c_gg_confirm_thresh = 25
c_templ_learn_every_num_perms = 12
c_gg_learn_every_num_perms = 50 # 10
c_gg_num_learn_steps = 200000
c_gg_learn_test_every = 1000
c_gg_learn_good_thresh = 0.01
c_gg_learn_give_up_at = 100000
c_gg_learn_give_up_thresh = 0.1
c_b_nbns = True # Necessary but not sufficient
c_fudge_thresh_cd = 0.97 # lowers the thresh to allow just outside
c_gg_scoring_eid_thresh = 5 # length of eid set after learn for score to be valid
c_gg_max_perms = 30
c_negative_penalty = 5 # applied when an igg wins the score/len contest
c_points_penalty_value = 2
c_cd_epsilon = 0.01
c_expands_min_tries = 30
c_expands_score_thresh = 0.8
c_expands_score_min_thresh = 0.3
c_select_cont_review_null_prob = 0.2
c_select_cont_random_prob = 0.001
c_select_cont_score_bonus = 0.2
c_select_cont_untried_bonus = 1.0
c_gg_starting_penalty = -10
c_score_loser_penalty = 1
c_score_winner_bonus = 5
c_cont_not_parent_max = 10.0


# logger = None

# sample_el = 'to' # used as an example that must be in el db or glv dict
query_action = ['who', 'what', 'where', 'what has', 'what does', 'have', 'where is', 'is in', 'know', 'asked', 'told', 'that', 'asked', 'for']
person_place_action = ['is located in', 'went to', 'go to']
object_place_action = ['is located in', 'is free in' ]
person_object_dynamic_action = ['picked up', 'put down', 'pick up']
person_object_static_action = ['has', 'wants']
person_person_static_action = ['likes']
person_object_dynamic_3_action = ['gave to']
knowledge_action = ['knows that']
decision_action = ['decided to']

uniquify = lambda llist: list(set(llist))

person_object_action = uniquify(person_object_dynamic_action + person_object_static_action)
actions = uniquify(query_action + person_place_action + object_place_action
				   + person_object_action + person_person_static_action + person_object_dynamic_3_action
				   + knowledge_action + decision_action)
# actions = ['picked up', 'put down', 'has', 'went to', 'is located in', 'is free in', 'who']

