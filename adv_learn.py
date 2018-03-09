import sys
import csv
from os.path import expanduser
from StringIO import StringIO
import copy

import utils
import learn
import clrecgrp
import addlearn
import dmlearn
import adv_config

db_fnt = '~/tmp/advlengrps.txt'

c_len_grps_version = 4

def save_db_status(db_len_grps, db_cont_mgr):
	db_fn = expanduser(db_fnt)
	db_fh = open(db_fn, 'wb')
	db_csvr = csv.writer(db_fh, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
	def save_a_len_grps(which_len_grps):
		# title = 'Num len grps' if b_normal else 'Num blocking len grps'
		db_csvr.writerow(['Num len grps', len(which_len_grps)])
		for len_grp in which_len_grps:
			len_grp.save(db_csvr)
	# cont_list_size = 0 if gg_cont_list == None else len(gg_cont_list)
	db_csvr.writerow(['version', c_len_grps_version])
	db_cont_mgr.save(db_csvr)
	for cont in db_cont_mgr.get_cont_list():
		if cont.is_active():
			num_data_rows = 1
			for len_grp in db_len_grps:
				num_data_rows += len_grp.get_num_data_rows()
			cont.set_num_rows_grp_data(num_data_rows)
		cont.save(db_csvr)
		if cont.is_active():
			save_a_len_grps(db_len_grps)
	# save_a_len_grps(blocked_len_grps, i_gg_cont, b_normal=False)

	db_fh.close()

def load_cont_mgr():
	db_fn = expanduser(db_fnt)
	i_gg_cont = -1
	# db_len_grps, blocked_len_grps, gg_cont_list = [], [], []
	db_cont_mgr = addlearn.cl_cont_mgr()
	try:
	# if True:
		with open(db_fn, 'rb') as db_fh:
			db_csvr = csv.reader(db_fh, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
			_, version_str = next(db_csvr)
			if int(version_str) != c_len_grps_version:
				raise ValueError('db len grps file version cannot be used. Starting from scratch')
				# return db_len_grps, blocked_len_grps, gg_cont_list, int(i_gg_cont)

			# _, gg_cont_list_size = next(db_csvr)
			gg_cont_list_size = db_cont_mgr.load(db_csvr)
			for i_cont in range(gg_cont_list_size):
				gg_cont = addlearn.cl_add_gg(b_from_load=True)
				gg_cont.load(db_csvr, b_null=(i_cont==0))
				db_cont_mgr.add_cont(gg_cont)

			# def load_a_len_grps(which_len_grps, b_normal):

			# version, _ = load_a_len_grps(db_len_grps, b_normal=True)
		# if version > 1:
		# 	_, i_gg_cont = load_a_len_grps(blocked_len_grps, b_normal=False)

	except ValueError as verr:
		print(verr.args)
	except IOError:
	# except:
		print('Could not open db_len_grps file! Starting from scratch.')
	except:
		print "Unexpected error:", sys.exc_info()[0]
		raise

	return db_cont_mgr

def load_len_grps(grp_data_list, db_len_grps):
	assert  db_len_grps == []
	data_list = copy.deepcopy(grp_data_list)
	# f = StringIO(s_grp_data)
	# db_csvr = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
	# db_len_grps = []
	_, num_len_grps = data_list.pop(0)
	for i_len_grp in range(int(num_len_grps)):
		len_grp = clrecgrp.cl_len_grp(b_from_load=True)
		len_grp.load(data_list)
		db_len_grps.append(len_grp)
	return db_len_grps


# def learn_more(gg_cont_list, i_gg_cont, db_len_grps):
def sel_cont_and_len_grps(db_cont_mgr):
	db_len_grps = []
	sel_cont, ibest = db_cont_mgr.select_cont()
	grp_data = sel_cont.get_grp_data()
	# if there is no grp data return and db_len_grps will be empty
	# if there is data but no cont can be created, just keep learning with what we have
	# if new conts are created, select from all of them again
	if grp_data != []:
		load_len_grps(grp_data, db_len_grps)
	# if db_len_grps != []:
	# 	dmlearn.learn_reset()
	# 	del db_len_grps[:]


	# gg_cont_list, ibest = learn.learn_more(	gg_cont_list, i_gg_cont, db_len_grps, adv_config.c_cont_score_thresh,
	# 						adv_config.c_cont_score_min, adv_config.c_cont_min_tests)
	# return gg_cont_list, ibest
	return db_len_grps, ibest

def create_new_conts(glv_dict, db_cont_mgr, db_len_grps, i_active_cont):
	b_keep_working = db_cont_mgr.create_new_conts(	glv_dict, db_len_grps, i_active_cont, adv_config.c_cont_score_thresh,
													adv_config.c_cont_score_min, adv_config.c_cont_min_tests)
	return b_keep_working


def do_learn_rule_from_step(event_as_decided, event_step_id, story_db, one_decide, seed,
							def_article_dict, db_len_grps, sess, el_set_arr, glv_dict,
							els_sets, cascade_dict, gg_cont, db_cont_mgr):
	expected_but_not_found_list = []
	b_blocking = False
	# story_els_set = utils.combine_sets([els_sets.objects, els_sets.places, els_sets.names])
	# cascade_els = story_els_set[2]
	cascade_els = [el for el in cascade_dict.keys() if cascade_dict[el]]

	cont_status = gg_cont.get_status()
	b_std_learn = True
	cont_use = gg_cont
	if cont_status == addlearn.cl_cont_mgr.status.untried:
		cont_parent = db_cont_mgr.get_cont(gg_cont.get_parent_id())
		if not cont_parent.is_null():
			b_std_learn = False
			b_first_run = True

	while True:

		stats = learn.learn_one_story_step2(story_db, [one_decide], cascade_els, event_as_decided, def_article_dict,
											db_len_grps, el_set_arr, glv_dict, sess, event_step_id,
											expected_but_not_found_list,
											0, cont_use, b_blocking, b_test_rule=not b_std_learn)
		if not b_std_learn:
			if b_first_run:
				cont_use = cont_parent
				b_first_run = False
				child_stats = stats
			else:
				params = gg_cont.get_status_params()
				if stats[0]:
					if not stats[1]:
						if child_stats[0] and gg_cont.is_blocking() and child_stats[1]:
							gg_cont.set_status_params([params[0] + 1.0, params[1] + 1.0])
						elif not gg_cont.is_blocking() and not child_stats[0]:
							gg_cont.set_status_params([params[0] + 1.0, params[1] + 1.0])
						else:
							gg_cont.set_status_params([params[0], params[1] + 1.0])
				break
		else:
			break

