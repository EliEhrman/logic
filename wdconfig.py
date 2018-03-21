from enum import Enum

c_sample_el = 'army'
c_cont_score_thresh = 0.95 # better than this, we don't bother trying to find more clauses
c_cont_score_min = 0.1
c_cont_min_tests = 10
c_num_turns_per_play = 300
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

orders_success_fnt = '~/tmp/orders_success.txt'
orders_failed_fnt = '~/tmp/orders_failed.txt'
db_fnt = '~/tmp/wdlengrps.txt'

# c_b_compare_conts = False replaced by next few individual flags
c_b_load_cont_stats = False # Loads cont stats from conts file and builds a cont for each one
c_b_analyze_and_modify_conts = False # Analyses accuracy and usefulness of each cont and Decides which conts to keep and when to try and create new cont mutations
c_b_learn_conts = False # At the end of game or play pahse (30 to 300 turns), Learns the W for success prediction
c_b_cont_stats_save = False # saves all cont modification and W to file, including new stats
c_b_collect_cont_stats = False # Adds to matches and predict statistic for all conts
c_b_save_orders = True
c_b_load_cont_mgr = False # loads cont manager and the conts from wdlengrps
c_b_add_to_db_len_grps = False # Assumes c_b_load_cont_mgr, makes one cont active and loads its len grps. When it comes time to play, learns from
c_b_init_cont_stats_from_cont_mgr = False # This is how you build cont stats the first time from cont groups
c_b_play_from_saved = False # Means we use the saved orders_success file and use some AI to create move. Alternative is to use the oracle move creator
c_use_rule_thresh = 0.99
c_num_montes = 1
# c_include_pass_statements = False
c_preferred_nation = None
c_b_predict_success = False
c_rnd_bad_move = 0.0
c_rnd_fleet_army_wrong = 0.0


# c_target_gens = ['c:s,l:army:0.2,l:in:1.0,l:munich:-0.5,l:move:1.0,l:to:1.0,l:piedmont:-0.5,l:succeeded:1.0,c:e']
c_target_gens = ['l:army:0.2,l:in:1.0,l:munich:-0.5,l:move:1.0,l:to:1.0,l:piedmont:-0.5']

c_admin_action = None # 'DeleteGames'
c_cont_stats_init_thresh = 0.1
c_cont_stats_init_exclude_irrelevant = True
c_cont_stats_fnt = '~/tmp/cont_stats.txt'

e_move_type = Enum('e_move_type', 'none move support convoy hold support_hold')

c_oracle_support_prob = 0.5
c_oracle_hold_prob = 0.1
c_oracle_fleet_first_prob = 0.1
