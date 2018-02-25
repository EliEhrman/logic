import csv
from os.path import expanduser

import utils
import learn
import clrecgrp
import addlearn
import adv_config

db_fnt = '~/tmp/advlengrps.txt'

c_len_grps_version = 2

def save_len_grps(db_len_grps, blocked_len_grps, gg_cont_list, i_gg_cont):
	db_fn = expanduser(db_fnt)
	db_fh = open(db_fn, 'wb')
	db_csvr = csv.writer(db_fh, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
	cont_list_size = 0 if gg_cont_list == None else len(gg_cont_list)
	db_csvr.writerow(['Num gg cont rules', cont_list_size, 'version', c_len_grps_version])
	for gg_cont in gg_cont_list:
		gg_cont.save(db_csvr)
	def save_a_len_grps(which_len_grps, i_gg_cont, b_normal):
		title = 'Num len grps' if b_normal else 'Num blocking len grps'
		db_csvr.writerow([title, len(which_len_grps), 'gg cont id', i_gg_cont, 'version', c_len_grps_version])
		for len_grp in which_len_grps:
			len_grp.save(db_csvr)
	save_a_len_grps(db_len_grps, i_gg_cont, b_normal=True)
	save_a_len_grps(blocked_len_grps, i_gg_cont, b_normal=False)

	db_fh.close()

def load_len_grps():
	db_fn = expanduser(db_fnt)
	i_gg_cont = -1
	db_len_grps, blocked_len_grps, gg_cont_list = [], [], []
	try:
		db_fh = open(db_fn, 'rb')
		db_csvr = csv.reader(db_fh, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
		_, gg_cont_list_size, _, version_str = next(db_csvr)
		for i_cont in range(int(gg_cont_list_size)):
			gg_cont = addlearn.cl_add_gg(b_from_load=True)
			gg_cont.load(db_csvr)
			gg_cont_list.append(gg_cont)

		def load_a_len_grps(which_len_grps, b_normal):
			_, num_len_grps, _, i_gg_cont, _, version_str = next(db_csvr)
			for i_len_grp in range(int(num_len_grps)):
				len_grp = clrecgrp.cl_len_grp(b_from_load = True)
				len_grp.load(db_csvr, b_blocking= not b_normal)
				which_len_grps.append(len_grp)
			return int(version_str), i_gg_cont
		version, _ = load_a_len_grps(db_len_grps, b_normal=True)
		if version > 1:
			_, i_gg_cont = load_a_len_grps(blocked_len_grps, b_normal=False)

	except IOError:
	# except:
		print('Could not open db_len_grps file! Starting from scratch.')

	return db_len_grps, blocked_len_grps, gg_cont_list, int(i_gg_cont)

def learn_more(gg_cont_list, i_gg_cont, db_len_grps):
	gg_cont_list, ibest = learn.learn_more(	gg_cont_list, i_gg_cont, db_len_grps, adv_config.c_cont_score_thresh,
								adv_config.c_cont_score_min, adv_config.c_cont_min_tests)
	return gg_cont_list, ibest


def do_learn_rule_from_step(event_as_decided, event_step_id, story_db, one_decide, seed,
							def_article_dict, db_len_grps, sess, el_set_arr, glv_dict,
							els_sets, cascade_dict, gg_cont):
	expected_but_not_found_list = []
	b_blocking = False
	# story_els_set = utils.combine_sets([els_sets.objects, els_sets.places, els_sets.names])
	# cascade_els = story_els_set[2]
	cascade_els = [el for el in cascade_dict.keys() if cascade_dict[el]]

	learn.learn_one_story_step2(story_db, [one_decide], cascade_els, event_as_decided, def_article_dict,
							 db_len_grps, el_set_arr, glv_dict, sess, event_step_id,
							 expected_but_not_found_list,
							 0, gg_cont, b_blocking)
