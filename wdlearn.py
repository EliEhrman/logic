from __future__ import print_function
import sys
from os.path import expanduser
import copy
import csv
import learn
import addlearn
import els
import clrecgrp
import wdconfig

db_fnt = '~/tmp/wdlengrps.txt'
# db_fn = '/home/eli/tmp/lengrps.txt'

c_len_grps_version = 2

def init_learn():
	addlearn.cl_cont_mgr.c_expands_min_tries = wdconfig.c_expands_min_tries
	addlearn.cl_cont_mgr.c_expands_score_thresh = wdconfig.c_expands_score_thresh
	addlearn.cl_cont_mgr.c_expands_score_min_thresh = wdconfig.c_expands_score_min_thresh


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


# def save_len_grps(db_len_grps, blocked_len_grps):
# 	db_fh = open(db_fn, 'wb')
# 	db_csvr = csv.writer(db_fh, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
# 	def save_a_len_grps(which_len_grps, b_normal):
# 		title = 'Num len grps' if b_normal else 'Num blocking len grps'
# 		db_csvr.writerow([title, len(which_len_grps), c_len_grps_version])
# 		for len_grp in which_len_grps:
# 			len_grp.save(db_csvr)
# 	save_a_len_grps(db_len_grps, b_normal=True)
# 	save_a_len_grps(blocked_len_grps, b_normal=False)
#
# 	db_fh.close()

def load_cont_mgr():
	db_fn = expanduser(db_fnt)
	i_gg_cont = -1
	db_cont_mgr = addlearn.cl_cont_mgr()
	try:
		with open(db_fn, 'rb') as db_fh:
			db_csvr = csv.reader(db_fh, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
			_, version_str = next(db_csvr)
			if int(version_str) != c_len_grps_version:
				raise ValueError('db len grps file version cannot be used. Starting from scratch')

			gg_cont_list_size = db_cont_mgr.load(db_csvr)
			for i_cont in range(gg_cont_list_size):
				gg_cont = addlearn.cl_add_gg(b_from_load=True)
				gg_cont.load(db_csvr, b_null=(i_cont==0))
				db_cont_mgr.add_cont(gg_cont)


	except ValueError as verr:
		print(verr.args)
	except IOError:
		print('Could not open db_len_grps file! Starting from scratch.')
	except:
		print('Unexpected error:', sys.exc_info()[0])
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


def sel_cont_and_len_grps(db_cont_mgr):
	db_len_grps = []
	sel_cont, ibest = db_cont_mgr.select_cont()
	grp_data = sel_cont.get_grp_data()
	# if there is no grp data return and db_len_grps will be empty
	# if there is data but no cont can be created, just keep learning with what we have
	# if new conts are created, select from all of them again
	if grp_data != []:
		load_len_grps(grp_data, db_len_grps)

	return db_len_grps, ibest



# def load_len_grps():
# 	db_len_grps, blocked_len_grps = [], []
# 	try:
# 		db_fh = open(db_fn, 'rb')
# 		db_csvr = csv.reader(db_fh, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
# 		def load_a_len_grps(which_len_grps, b_normal):
# 			_, num_len_grps,version_str = next(db_csvr)
# 			for i_len_grp in range(int(num_len_grps)):
# 				len_grp = clrecgrp.cl_len_grp(b_from_load = True)
# 				len_grp.load(db_csvr, b_blocking= not b_normal)
# 				which_len_grps.append(len_grp)
# 			return int(version_str)
# 		version = load_a_len_grps(db_len_grps, b_normal=True)
# 		if version > 1:
# 			_ = load_a_len_grps(blocked_len_grps, b_normal=False)
#
# 	except IOError:
# 		print('Could not open db_len_grps file! Starting from scratch.')
#
# 	return db_len_grps, blocked_len_grps

def create_new_conts(db_cont_mgr, db_len_grps, i_active_cont):
	b_keep_working = db_cont_mgr.create_new_conts(	db_len_grps, i_active_cont, wdconfig.c_cont_score_thresh,
													wdconfig.c_cont_score_min, wdconfig.c_cont_min_tests)
	return b_keep_working



def learn_orders_success(init_pl, status_pl, orders_pl, results_pl, all_the_dicts, db_len_grps, db_cont_mgr, i_active_cont,
						 el_set_arr, sess, learn_vars):
	glv_dict, def_article_dict, cascade_dict = all_the_dicts
	init_db , status_db, orders_db, results_db = \
		els.make_rec_list(init_pl), els.make_rec_list(status_pl), els.make_rec_list(orders_pl), els.make_rec_list(results_pl),
	cascade_els = [el for el in cascade_dict.keys() if cascade_dict[el]]
	full_db = init_db + status_db
	num_orders = len(orders_db)
	event_step_id = learn_vars[0]
	b_keep_working = True

	curr_cont = db_cont_mgr.get_cont(i_active_cont)
	for iorder in range(num_orders):
		event_step_id += 1
		order = orders_db.pop(0)
		out_str = ''
		out_str = els.print_phrase(order.phrase(), order.phrase(), out_str, def_article_dict)
		print('New order: ', out_str)
		out_str = ''
		out_str = els.print_phrase(results_pl[iorder], results_pl[iorder], out_str, def_article_dict)
		print('Result of order: ', out_str)
		expected_but_not_found_list, null_expected_list = [], []

		cont_status = curr_cont.get_status()
		b_std_learn = True
		cont_use = curr_cont
		if cont_status == addlearn.cl_cont_mgr.status.untried:
			cont_parent = db_cont_mgr.get_cont(curr_cont.get_parent_id())
			if not cont_parent.is_null():
				b_std_learn = False
				b_first_run = True

		while True:
			stats = learn.learn_one_story_step2(full_db+orders_db, [order.phrase()], cascade_els, [results_pl[iorder]],
									def_article_dict, db_len_grps, el_set_arr, glv_dict, sess, event_step_id,
									expected_but_not_found_list, level_depr=0, gg_cont=cont_use,
									b_blocking_depr=False, b_test_rule=not b_std_learn)
			if not b_std_learn:
				if b_first_run:
					cont_use = cont_parent
					b_first_run = False
					child_stats = stats
				else:
					params = curr_cont.get_status_params()
					if stats[0]:
						if not stats[1]:
							if child_stats[0] and curr_cont.is_blocking() and child_stats[1]:
								curr_cont.set_status_params([params[0] + 1.0, params[1] + 1.0])
							elif not curr_cont.is_blocking() and not child_stats[0]:
								curr_cont.set_status_params([params[0] + 1.0, params[1] + 1.0])
							else:
								curr_cont.set_status_params([params[0], params[1] + 1.0])
					break
			else:
				break
		# for not_found in expected_but_not_found_list:
		# 	learn.learn_one_story_step(full_db + orders_db, not_found[0], cascade_els, [not_found[1]],
		# 							   def_article_dict, db_cont_mgr, i_active_cont, el_set_arr, glv_dict, sess,
		# 							   event_step_id,
		# 							   null_expected_list, b_blocking=True)
		orders_db.append(order)

		if iorder % 10 == 0:
			b_keep_working = create_new_conts(db_cont_mgr, db_len_grps, i_active_cont)
			save_db_status(db_len_grps, db_cont_mgr)
			if not b_keep_working:
				break
			# save_len_grps(db_len_grps, blocked_len_grps)

	learn_vars[0] = event_step_id

	return b_keep_working