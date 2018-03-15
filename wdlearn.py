from __future__ import print_function
import sys
from os.path import expanduser
import copy
import csv
import learn
import addlearn
import els
import makerecs as mr
import compare_conts as cc
import clrecgrp
import rules
# import utils
import wdconfig

# wdconfig.db_fnt = '~/tmp/wdlengrps.txt'
# db_fn = '/home/eli/tmp/lengrps.txt'

c_len_grps_version = 2

def init_learn():
	addlearn.cl_cont_mgr.c_expands_min_tries = wdconfig.c_expands_min_tries
	addlearn.cl_cont_mgr.c_expands_score_thresh = wdconfig.c_expands_score_thresh
	addlearn.cl_cont_mgr.c_expands_score_min_thresh = wdconfig.c_expands_score_min_thresh
	clrecgrp.cl_templ_grp.c_score_loser_penalty = wdconfig.c_score_loser_penalty
	clrecgrp.cl_templ_grp.c_score_winner_bonus = wdconfig.c_score_winner_bonus


def save_db_status(db_len_grps, db_cont_mgr):
	db_fn = expanduser(wdconfig.db_fnt)
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
	db_fn = expanduser(wdconfig.db_fnt)
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

def load_len_grps(grp_data_list, db_len_grps, b_cont_blocking):
	assert  db_len_grps == []
	data_list = copy.deepcopy(grp_data_list)
	# f = StringIO(s_grp_data)
	# db_csvr = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
	# db_len_grps = []
	_, num_len_grps = data_list.pop(0)
	for i_len_grp in range(int(num_len_grps)):
		len_grp = clrecgrp.cl_len_grp(b_from_load=True)
		len_grp.load(data_list, b_cont_blocking)
		db_len_grps.append(len_grp)
	return db_len_grps

def add_success_to_targets(target_str_list):
	target_rule_list = []
	for target_rule_str in target_str_list:
		target_rule_grp = mr.extract_rec_from_str(target_rule_str)
		target_rule_grp += [[rules.rec_def_type.like, 'succeeded', 1.0]]
		target_rule_grp = mr.make_rec_from_phrase_arr([target_rule_grp])
		target_rule_list.append(target_rule_grp)

	return  target_rule_list



def set_target_gens():
	target_rule_list = add_success_to_targets(wdconfig.c_target_gens)
	clrecgrp.cl_templ_grp.c_target_gens = target_rule_list


def sel_cont_and_len_grps(db_cont_mgr):
	db_len_grps = []
	sel_cont, ibest = db_cont_mgr.select_cont()
	if ibest >= 0:
		b_cont_blocking = sel_cont.is_blocking()
		grp_data = sel_cont.get_grp_data()
		# if there is no grp data return and db_len_grps will be empty
		# if there is data but no cont can be created, just keep learning with what we have
		# if new conts are created, select from all of them again
		if grp_data != []:
			load_len_grps(grp_data, db_len_grps, b_cont_blocking)

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

def create_new_conts(glv_dict, db_cont_mgr, db_len_grps, i_active_cont):
	b_keep_working = db_cont_mgr.create_new_conts(	glv_dict, db_len_grps, i_active_cont, wdconfig.c_cont_score_thresh,
													wdconfig.c_cont_score_min, wdconfig.c_cont_min_tests)
	return b_keep_working

def load_order_freq_tbl(freq_tbl, fnt):
	try:
		o_fn = expanduser(fnt)
		with open(o_fn, 'rb') as o_fhr:
			o_csvr = csv.reader(o_fhr, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
			for row in o_csvr:
				freq_tbl[tuple(row[1:])] = int(row[0])
	except IOError:
		return


def create_order_freq_tbl(orders_list, order_status_list):
	success_orders_freq, failed_orders_freq = dict(), dict()

	load_order_freq_tbl(success_orders_freq, wdconfig.orders_success_fnt)
	load_order_freq_tbl(failed_orders_freq, wdconfig.orders_failed_fnt)


	def add_to_tbl(freq_tbl, order):
		freq = freq_tbl.get(order, -1)
		if freq == -1:
			freq_tbl[order] = 1
		else:
			freq_tbl[order] = freq + 1

	for iorder, order in enumerate(orders_list):
		if order_status_list[iorder].status:
			add_to_tbl(success_orders_freq, tuple(order))
		else:
			add_to_tbl(failed_orders_freq, tuple(order))

	def save_tbl(freq_tbl, fnt):
		o_fn = expanduser(fnt)
		o_fhw = open(o_fn, 'wb')
		o_csvw = csv.writer(o_fhw, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
		for korder, vfreq in freq_tbl.iteritems():
			row = [vfreq] + list(korder)
			o_csvw.writerow(row)

		o_fhw.close()

	save_tbl(success_orders_freq, wdconfig.orders_success_fnt)
	save_tbl(failed_orders_freq, wdconfig.orders_failed_fnt)

	return True

def collect_cont_stats(init_pl, status_pl, orders_pl, results_pl, all_the_dicts, db_cont_mgr):
	glv_dict, def_article_dict, cascade_dict = all_the_dicts
	init_db , status_db, orders_db, results_db = \
		els.make_rec_list(init_pl), els.make_rec_list(status_pl), els.make_rec_list(orders_pl), els.make_rec_list(results_pl),
	cascade_els = [el for el in cascade_dict.keys() if cascade_dict[el]]
	full_db = init_db + status_db
	num_orders = len(orders_db)
	b_keep_working = True
	cont_stats_mgr = db_cont_mgr.get_cont_stats_mgr()
	cont_stats_list = cont_stats_mgr.get_cont_stats_list()

	# curr_cont = db_cont_mgr.get_cont(i_active_cont)
	for iorder in range(num_orders):
		order_rec = orders_db.pop(0)
		order = order_rec.phrase()
		out_str = ''
		out_str = els.print_phrase(order, order, out_str, def_article_dict)
		print('Order for cont stats: ', out_str)
		out_str = ''
		out_str = els.print_phrase(results_pl[iorder], results_pl[iorder], out_str, def_article_dict)
		print('Result of order: ', out_str)

		b_target_success = False
		b_match_cand = False
		cont_match_list = []
		for srule in wdconfig.c_target_gens:
			target_rule = mr.extract_rec_from_str(srule)
			# target_rule_rec = mr.make_rec_from_phrase_arr(target_rule)
			# if not mr.does_match_rule(glv_dict, target_rule_rec, mr.make_rec_from_phrase_arr(order)):
			if not mr.does_match_rule(glv_dict, target_rule, order):
				continue

			b_match_cand = True

			target_rule += [[rules.rec_def_type.like, 'succeeded', 1.0]]
			# target_rule_rec = mr.make_rec_from_phrase_arr(target_rule_grp)
			# success_result = order + [[rules.rec_def_type.obj, 'succeeded']]
			# target_rule = mr.extract_rec_from_str(srule)
			# if mr.does_match_rule(glv_dict, target_rule, mr.make_rec_from_phrase_arr([results_pl[iorder]])):
			if mr.does_match_rule(glv_dict, target_rule, results_pl[iorder]):
				b_target_success = True
				break

		if b_match_cand:

			# order_template = ['?', '?', 'in', '?', 'move', 'to', '?', '?']
			# if not els.match_rec_to_templ(order, order_template):
			# 	continue
			# success_result = order + [[rules.rec_def_type.obj, 'succeeded']]
			# b_success =  mr.match_rec_exact(success_result, results_pl[iorder])
			cont_stats_mgr.add_match(b_target_success)

			for istats, cont_stats in enumerate(cont_stats_list):
				b_match,_ = learn.learn_one_story_step2(full_db + orders_db, [order], cascade_els, [results_pl[iorder]],
													def_article_dict, [], [], glv_dict, sess=None, event_step_id=-1,
													expected_but_not_found_list=[], level_depr=0, gg_cont=cont_stats.get_cont(),
													b_blocking_depr=False, b_test_rule=True)
				if istats == 0 and not b_match:
					print('First rule failed. Why?')
					b_match,_ = learn.learn_one_story_step2(full_db + orders_db, [order], cascade_els, [results_pl[iorder]],
														def_article_dict, [], [], glv_dict, sess=None, event_step_id=-1,
														expected_but_not_found_list=[], level_depr=0, gg_cont=cont_stats.get_cont(),
														b_blocking_depr=False, b_test_rule=True)
				cont_stats.add_match(b_match)
				cont_match_list.append(b_match)

			cont_stats_mgr.predict_success_rate(cont_match_list)

		orders_db.append(order_rec)

	return b_keep_working

def compare_conts_learn(db_cont_mgr):
	cont_stats_mgr = db_cont_mgr.get_cont_stats_mgr()
	if cont_stats_mgr:
		cont_stats_mgr.do_learn()
		cont_stats_mgr.save(wdconfig.c_cont_stats_fnt)
	return

def init_cont_stats_from_file():
	db_cont_mgr = addlearn.cl_cont_mgr()
	b_load_done = db_cont_mgr.init_cont_stats_mgr_from_file(wdconfig.c_cont_stats_fnt)
	if not b_load_done:
		db_cont_mgr = None
	return db_cont_mgr


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

		# The first returned value is whether it matched. The second is whether the result occurred
		# This will be the same regardless of whether the cont is a blocking cont or not
		# The first param of the status params is how many matches were true for the parent
		# The second param is how many results occurred
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
					if not stats[0]:
						if child_stats[1]:
							print('Interesting option here. Child matched preconds but parent did not. Investigate.')
					else:
						params = curr_cont.get_status_params()
						if len(params) < 4:
							num_child_hits, num_child_results, num_parent_hits, num_parent_results = 0.0, 0.0, 0.0, 0.0
						else:
							num_child_hits, num_child_results, num_parent_hits, num_parent_results = params
						if child_stats[0] and stats[1] != child_stats[1]:
							print('Interesting option here. Result occurred in only one of parent and child. Investigate.')
						num_parent_hits += 1.0
						if stats[1]:
							num_parent_results += 1.0
						if child_stats[0]:
							num_child_hits += 1.0
							if child_stats[1]:
								num_child_results += 1.0
						curr_cont.set_status_params([num_child_hits, num_child_results, num_parent_hits, num_parent_results ])
					# if stats[0]:
					# 	if not stats[1]:
					# 		if child_stats[0] and curr_cont.is_blocking() and child_stats[1]:
					# 			curr_cont.set_status_params([params[0] + 1.0, params[1] + 1.0])
					# 		elif not curr_cont.is_blocking() and not child_stats[0]:
					# 			curr_cont.set_status_params([params[0] + 1.0, params[1] + 1.0])
					# 		else:
					# 			curr_cont.set_status_params([params[0], params[1] + 1.0])
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
			b_keep_working = create_new_conts(glv_dict, db_cont_mgr, db_len_grps, i_active_cont)
			save_db_status(db_len_grps, db_cont_mgr)
			if not b_keep_working:
				break
			# save_len_grps(db_len_grps, blocked_len_grps)

	learn_vars[0] = event_step_id

	return b_keep_working