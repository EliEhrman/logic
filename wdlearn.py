from __future__ import print_function
import sys
import csv
import learn
import els
import clrecgrp

db_fn = '/home/eli/tmp/lengrps.txt'

c_len_grps_version = 2

def save_len_grps(db_len_grps, blocked_len_grps):
	db_fh = open(db_fn, 'wb')
	db_csvr = csv.writer(db_fh, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
	def save_a_len_grps(which_len_grps, b_normal):
		title = 'Num len grps' if b_normal else 'Num blocking len grps'
		db_csvr.writerow([title, len(which_len_grps), c_len_grps_version])
		for len_grp in which_len_grps:
			len_grp.save(db_csvr)
	save_a_len_grps(db_len_grps, b_normal=True)
	save_a_len_grps(blocked_len_grps, b_normal=False)

	db_fh.close()

def load_len_grps():
	db_len_grps, blocked_len_grps = [], []
	try:
		db_fh = open(db_fn, 'rb')
		db_csvr = csv.reader(db_fh, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
		def load_a_len_grps(which_len_grps, b_normal):
			_, num_len_grps,version_str = next(db_csvr)
			for i_len_grp in range(int(num_len_grps)):
				len_grp = clrecgrp.cl_len_grp(b_from_load = True)
				len_grp.load(db_csvr, b_blocking= not b_normal)
				which_len_grps.append(len_grp)
			return int(version_str)
		version = load_a_len_grps(db_len_grps, b_normal=True)
		if version > 1:
			_ = load_a_len_grps(blocked_len_grps, b_normal=False)

	except IOError:
		print('Could not open db_len_grps file! Starting from scratch.')

	return db_len_grps, blocked_len_grps


def learn_orders_success(init_pl, status_pl, orders_pl, results_pl, all_the_dicts, db_len_grps, blocked_len_grps,
						 el_set_arr, sess, learn_vars):
	glv_dict, def_article_dict, cascade_dict = all_the_dicts
	init_db , status_db, orders_db, results_db = \
		els.make_rec_list(init_pl), els.make_rec_list(status_pl), els.make_rec_list(orders_pl), els.make_rec_list(results_pl),
	cascade_els = [el for el in cascade_dict.keys() if cascade_dict[el]]
	full_db = init_db + status_db
	num_orders = len(orders_db)
	event_step_id = learn_vars[0]
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
		learn.learn_one_story_step(full_db+orders_db, [order.phrase()], cascade_els, [results_pl[iorder]],
								   def_article_dict, db_len_grps, el_set_arr, glv_dict, sess, event_step_id,
								   expected_but_not_found_list, b_blocking=False)
		for not_found in expected_but_not_found_list:
			learn.learn_one_story_step(full_db + orders_db, not_found[0], cascade_els, [not_found[1]],
									   def_article_dict, blocked_len_grps, el_set_arr, glv_dict, sess,
									   event_step_id,
									   null_expected_list, b_blocking=True)
		orders_db.append(order)

		if iorder % 10 == 0:
			save_len_grps(db_len_grps, blocked_len_grps)

	learn_vars[0] = event_step_id

	return