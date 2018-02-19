c_max_people = 5
c_max_countries = 5
c_max_objects = 5
c_save_every = 2
c_story_len = 25
c_num_stories = 10
c_num_plays = 10
c_cont_score_thresh = 0.95 # better than this, we don't bother trying to find more clauses
c_cont_score_min = 0.1
c_cont_min_tests = 10

glv_file_list = ['adv/name', 'adv/object', 'adv/countrie', 'adv/action']
cap_first_arr = [True, False, True, False]
def_article_arr = [False, True, False, False]
cascade_els_arr = [True, True, True, False]
set_sel_arr = [c_max_people, c_max_objects, c_max_countries, -1]
sample_el = 'has' # used as an example that must be in el db or glv dict
