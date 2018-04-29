from enum import Enum

c_sample_el = 'army'
c_cont_score_thresh = 0.95 # better than this, we don't bother trying to find more clauses
c_cont_score_min = 0.1
c_cont_min_tests = 10
c_num_turns_per_play = 500
c_num_plays = 5300
c_expands_min_tries = 30
c_expands_score_thresh = 4.0
c_expands_score_min_thresh = 1.25
c_score_loser_penalty = 1
c_score_winner_bonus = 5
c_cont_stats_min_predictions = 100
c_b_add_terr_type_to_phrases = False
c_b_add_native_to_phrases = False
c_b_add_can_pass_to_phrases = False
c_b_add_supply_to_phrases = False
c_b_add_owns_to_phrases = True

orders_success_fnt = '~/tmp/orders_success.txt'
orders_failed_fnt = '~/tmp/orders_failed.txt'
db_fnt = '~/tmp/wdlengrps.txt'
perm_fnt = '~/tmp/wdperms.txt'
W_fnt = '~/tmp/wdWs.txt'

# c_b_compare_conts = False replaced by next few individual flags
c_b_load_cont_stats = True # Loads cont stats from conts file and builds a cont for each one
c_b_analyze_conts = True # Analyses accuracy and usefulness of each cont
c_b_modify_conts = False # if c_b_analyze_conts is True, Decides which conts to keep and when to try and create new cont mutations
c_b_learn_conts = True # At the end of game or play pahse (30 to 300 turns), Learns the W for success prediction
c_b_cont_stats_save = True # saves all cont modification and W to file, including new stats
c_b_collect_cont_stats = True # Adds to matches and predict statistic for all conts
c_b_save_orders = True
c_b_load_cont_mgr = False # loads cont manager and the conts from wdlengrps
c_b_add_to_db_len_grps = False # Assumes c_b_load_cont_mgr, makes one cont active and loads its len grps. When it comes time to play, learns from
c_b_init_cont_stats_from_cont_mgr = False # This is how you build cont stats the first time from cont groups
c_b_play_from_saved = True # Means we use the saved orders_success file and use some AI to create move. Alternative is to use the oracle move creator
c_use_rule_thresh = 0.99
c_num_montes = 1
# c_include_pass_statements = False
c_preferred_nation = None
c_b_predict_success = False
c_rnd_bad_move = 0.0
c_rnd_fleet_army_wrong = 0.0


# c_target_gens = ['c:s,l:army:0.2,l:in:1.0,l:munich:-0.5,l:move:1.0,l:to:1.0,l:piedmont:-0.5,l:succeeded:1.0,c:e']
c_target_gens = ['l:army:0.2,l:in:1.0,l:munich:-0.5,l:move:1.0,l:to:1.0,l:piedmont:-0.5',
				 'l:army:1.0,l:in:1.0,l:munich:-0.5,l:convoy:1.0,l:move:1.0,l:to:1.0,l:piedmont:-0.5']

c_admin_action = None # 'DeleteGames'
c_b_play_human = False
c_starting_user_id = 6
c_human_uids = [7]
c_gname_human_prefix = 'tplay'
c_cont_stats_init_thresh = 0.1
c_cont_stats_init_exclude_irrelevant = True
c_cont_stats_fnt = '~/tmp/cont_stats.txt'
c_cont_forbidden_fn = 'wd_forbidden.txt'
c_cont_forbidden_version = 1

e_move_type = Enum('e_move_type', 'none move support convoy hold support_hold convoy_move')

c_oracle_support_prob = 0.5
c_oracle_hold_prob = 0.3
c_oracle_fleet_first_prob = 0.1
c_oracle_convoy_err_prob = 0.3

c_classic_AI_defensive_bias = 1.2
c_classic_AI_max_successes = 10
c_classic_AI_num_option_runs = 5
c_classic_AI_stage_score_factor = 0.3
c_classic_AI_rejoiner_min = 0.1
c_classic_AI_abandon_prob = 0.3
c_classic_AI_contested_repl = 3

c_freq_stats_newbie_thresh = 5
c_freq_stats_mature_thresh = 10
c_freq_stats_drop_thresh = 0.1
c_freq_stats_version = 1

c_num_game_store_options = 3

c_alliance_terminate_thresh = 0.4
c_alliance_accept_thresh = 0.5
c_alliance_propose_thresh = 0.6
c_alliance_wait_to_propose = 2
c_alliance_wait_after_terminate = 4
c_alliance_move_per_turn = 0.3
c_alliance_max_move_per_turn = 0.1
c_alliance_notice_time = 2
c_alliance_oversize_limit = 11

c_max_units_for_status = 15 # as far as status stats are concerned we stop at this number
c_alliance_stats_turn_delay = 5
c_alliance_stats_fnt = '~/tmp/strength_stats.txt'
c_terr_stats_fnt = '~/tmp/terr_stats.txt'
c_unit_stats_fnt = '~/tmp/unit_stats.txt'
c_alliance_sel_mat_fnt = '~/tmp/alliance_sel_mat.txt'

c_alliance_prediction_k = 100

class cl_wd_state(object):

	def __init__(self):
		self.__gameID = None
		self.__all_dicts = None
		self.__el_set_arr = None
		self.__learn_vars = None
		self.__db_len_grps = None
		self.__db_cont_mgr = None
		self.__i_active_cont = None
		self.__sess = None
		self.__db  = None
		self.__cursor = None
		self.__gname = None
		self.__l_humaans = None
		self.__country_names_tbl = None
		self.__terr_id_tbl = None
		self.__supply_tbl = None
		self.__terr_type_tbl = None
		self.__army_can_pass_tbl = None
		self.__fleet_can_pass_tbl = None
		self.__init_db = None
		self.__b_waiting_for_AI = None
		self.__game_store = None
		self.__alliance_data = None
		self.__unit_owns_tbl = None
		self.__terr_owns_tbl = None
		self.__sqlOrderComplete = None
		self.__sql_get_unit_id = None
		self.__l_sql_action_orders = None
		self.__orders_list = None
		self.__orders_status_list = None
		self.__status_db = None
		self.__distance_calc = None
		self.__alliance_stats = None

	def set_alliance_stats(self, alliance_stats):
		self.__alliance_stats = alliance_stats

	def get_alliance_stats(self):
		return self.__alliance_stats

	def set_at_main(self, all_dicts, el_set_arr, learn_vars, country_names_tbl):
		self.__all_dicts = all_dicts
		self.__el_set_arr = el_set_arr
		self.__learn_vars = learn_vars
		self.__country_names_tbl = country_names_tbl

	def set_gameID(self, gameID):
		self.__gameID = gameID

	def get_at_do_wd(self):
		return self.__gameID, self.__all_dicts, self.__el_set_arr, self.__learn_vars

	def set_at_do_wd(self, db_len_grps, db_cont_mgr, i_active_cont, sess):
		self.__db_len_grps = db_len_grps
		self.__db_cont_mgr = db_cont_mgr
		self.__i_active_cont = i_active_cont
		self.__sess = sess

	def get_at_play(self):
		return self.__gameID, self.__all_dicts, self.__db_len_grps, self.__db_cont_mgr, \
			   self.__i_active_cont, self.__el_set_arr, self.__sess, self.__learn_vars, self.__country_names_tbl

	def set_distance_params(self, distance_calc):
		self.__distance_calc = distance_calc

	def set_at_play(self, db, cursor, gname, l_humaans, country_names_tbl,
					terr_id_tbl, supply_tbl, terr_type_tbl, army_can_pass_tbl,
					fleet_can_pass_tbl, init_db, b_waiting_for_AI,
					game_store, alliance_data):
		self.__db  = db
		self.__cursor = cursor
		self.__gname = gname
		self.__l_humaans = l_humaans
		self.__terr_id_tbl = terr_id_tbl
		self.__supply_tbl = supply_tbl
		self.__terr_type_tbl = terr_type_tbl
		self.__army_can_pass_tbl = army_can_pass_tbl
		self.__fleet_can_pass_tbl = fleet_can_pass_tbl
		self.__init_db = init_db
		self.__b_waiting_for_AI = b_waiting_for_AI
		self.__game_store = game_store
		self.__alliance_data = alliance_data

	def set_at_play_turn_tbls(self, unit_owns_tbl, terr_owns_tbl):
		self.__unit_owns_tbl = unit_owns_tbl
		self.__terr_owns_tbl = terr_owns_tbl

	def get_at_play_turn(self):
		return 	self.__all_dicts, self.__db_len_grps, self.__db_cont_mgr, self.__i_active_cont, self.__el_set_arr, \
				self.__sess, self.__learn_vars, \
				self.__db, self.__cursor, self.__gname, self.__l_humaans, self.__country_names_tbl, \
				self.__terr_id_tbl, self.__supply_tbl, self.__terr_type_tbl, \
				self.__army_can_pass_tbl, self.__fleet_can_pass_tbl, \
				self.__init_db, self.__b_waiting_for_AI, self.__game_store, self.__alliance_data

	def set_at_play_turn(self, sql_complete_order, sql_get_unit_id, l_sql_action_orders, orders_list, orders_status_list, status_db):
		self.__sql_complete_order = sql_complete_order
		self.__sql_get_unit_id = sql_get_unit_id
		self.__l_sql_action_orders = l_sql_action_orders
		self.__orders_list = orders_list
		self.__orders_status_list = orders_status_list
		self.__status_db = status_db

	def get_at_create_move_orders2(self):
		return 	self.__db, self.__cursor, self.__gameID, self.__sql_complete_order, self.__sql_get_unit_id, self.__l_sql_action_orders, \
				  self.__terr_id_tbl, self.__orders_list, self.__orders_status_list, self.__b_waiting_for_AI

	def get_at_classic_AI(	self):
		return 	self.__init_db, self.__status_db, self.__db_cont_mgr, self.__country_names_tbl, self.__l_humaans, \
				self.__unit_owns_tbl, self.__all_dicts, self.__terr_owns_tbl, self.__supply_tbl, \
				self.__b_waiting_for_AI, self.__game_store

	def set_gameID(self, gameID):
		self.__gameID = gameID

	def get_distance_params(self):
		return self.__distance_calc
