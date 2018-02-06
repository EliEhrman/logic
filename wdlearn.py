from __future__ import print_function
import sys
import csv
import learn
import els
import clrecgrp

db_fn = '/tmp/lengrps.txt'



def save_len_grps(db_len_grps):
	db_fh = open(db_fn, 'wb')
	db_csvr = csv.writer(db_fh, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
	db_csvr.writerow(['Num len grps', len(db_len_grps)])
	for len_grp in db_len_grps:
		len_grp.save(db_csvr)

	db_fh.close()

def load_len_grps():
	db_len_grps = []
	try:
		db_fh = open(db_fn, 'rb')
		db_csvr = csv.reader(db_fh, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
		_, num_len_grps = next(db_csvr)
		for i_len_grp in range(int(num_len_grps)):
			len_grp = clrecgrp.cl_len_grp(b_from_load = True)
			len_grp.load(db_csvr)
			db_len_grps.append(len_grp)

	except IOError:
		print('Could not open db_len_grps file! Starting from scratch.')

	return db_len_grps


def learn_orders_success(init_pl, status_pl, orders_pl, results_pl, all_the_dicts, db_len_grps, el_set_arr, sess, learn_vars):
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
		learn.learn_one_story_step(full_db+orders_db, order.phrase(), cascade_els, [results_pl[iorder]],
								   def_article_dict, db_len_grps, el_set_arr, glv_dict, sess, event_step_id)
		orders_db.append(order)
		if iorder % 10 == 0:
			save_len_grps(db_len_grps)

	learn_vars[0] = event_step_id

	return