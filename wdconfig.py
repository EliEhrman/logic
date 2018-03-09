c_sample_el = 'army'
c_cont_score_thresh = 0.95 # better than this, we don't bother trying to find more clauses
c_cont_score_min = 0.1
c_cont_min_tests = 10
c_num_turns_per_play = 15
c_num_plays = 530
c_expands_min_tries = 30
c_expands_score_thresh = 0.95
c_expands_score_min_thresh = 0.1
c_score_loser_penalty = 1
c_score_winner_bonus = 5

orders_success_fnt = '~/tmp/orders_success.txt'
orders_failed_fnt = '~/tmp/orders_failed.txt'
db_fnt = '~/tmp/wdlengrps.txt'

c_b_save_orders = False
c_b_add_to_db_len_grps = True # normally keep this and above complements
c_use_rule_thresh = 0.99
c_num_montes = 1
c_include_pass_statements = False
c_preferred_nation = None
c_b_predict_success = False
c_target_gens = ['c:s,l:army:0.2,l:in:1.0,l:munich:-0.5,l:move:1.0,l:to:1.0,l:piedmont:-0.5,l:succeeded:1.0,c:e']
