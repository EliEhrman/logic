c_max_people = 5
c_max_countries = 5
c_max_objects = 5
c_save_every = 2
c_story_len = 25
c_num_stories = 50
c_num_plays = 100
c_cont_score_thresh = 0.95 # better than this, we don't bother trying to find more clauses
c_cont_score_min = 0.1
c_cont_min_tests = 10
c_expands_min_tries = 100
c_expands_score_thresh = 0.8
c_expands_score_min_thresh = 0.3
c_score_loser_penalty = 1
c_score_winner_bonus = 5
c_freq_stats_newbie_thresh = 5
c_freq_stats_mature_thresh = 10
c_freq_stats_drop_thresh = 0.1
c_freq_stats_version = 1
c_b_save_freq_stats = True
c_b_learn_full_rules = False



glv_file_list = ['adv/name', 'adv/object', 'adv/countrie', 'adv/action']
cap_first_arr = [True, False, True, False]
def_article_arr = [False, True, False, False]
cascade_els_arr = [True, True, True, False]
set_sel_arr = [c_max_people, c_max_objects, c_max_countries, -1]
sample_el = 'has' # used as an example that must be in el db or glv dict
rules2_fn = 'adv/rules.txt'

db_fnt = '~/tmp/advlengrps.txt'
perm_fnt = '~/tmp/advperms.txt'
W_fnt = '~/tmp/advWs.txt'
phrase_freq_fnt = '~/tmp/adv_phrase_freq.txt'
