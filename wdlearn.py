from __future__ import print_function
import sys
import os.path
from os.path import expanduser
from shutil import copyfile
import copy
import csv
import random

import learn
import addlearn
import els
import makerecs as mr
import compare_conts as cc
import clrecgrp
import rules
import forbidden
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
	if os.path.isfile(db_fn):
		copyfile(db_fn, db_fn+'.bak')
	db_fh = open(db_fn, 'wb')
	db_csvr = csv.writer(db_fh, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
	def save_a_len_grps(which_len_grps):
		# title = 'Num len grps' if b_normal else 'Num blocking len grps'
		db_csvr.writerow(['Num len grps', len(which_len_grps)])
		for len_grp in which_len_grps:
			len_grp.save(db_csvr)
	# cont_list_size = 0 if gg_cont_list == None else len(gg_cont_list)
	db_cont_mgr.create_perm_dict(db_len_grps)
	db_cont_mgr.save_perm_dict(wdconfig.perm_fnt)
	db_cont_mgr.create_W_dict(db_len_grps)
	db_cont_mgr.save_W_dict(wdconfig.W_fnt)
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
		# raise

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

def load_order_freq_tbl(freq_tbl, id_dict, max_id, fnt):
	try:
		o_fn = expanduser(fnt)
		with open(o_fn, 'rb') as o_fhr:
			o_csvr = csv.reader(o_fhr, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
			_, _, version_str, _, snum_orders = next(o_csvr)
			version = int(version_str)
			if version != wdconfig.c_freq_stats_version:
				raise IOError
			for iorder in range(int(snum_orders)) :
				row = next(o_csvr)
				oid, co_dict = int(row[1]), dict()
				if oid > max_id[0]:
					max_id[0] = oid
				freq_tbl[tuple(row[2:])] = (int(row[0]), oid, co_dict)
				id_dict[oid] = tuple(row[2:])
				l_co_oids = next(o_csvr)
				for co_soid in l_co_oids:
					co_oid, co_freq = co_soid.split(':')
					co_dict[int(co_oid)] = int(co_freq)

	except IOError:
		print('Error. Cannot load order frequency stats.')
		return

# class cl_order_data(object):

class cl_order_freq_data(object):
	def __init__(self):
		self.__freq_tbl = dict() # dict mapping move/order to *index* into l_oids, l_freqs, l_b_tested and l_b_allowed
		self.__d_oids_to_move = dict() # dict mapping oids to move/order
		self.__d_oids = dict() # dict mapping oids directly to thr *index* into l_oids, l_freqs, l_b_tested and l_b_allowed
		self.__d_units = dict()
		self.__l_oids = []
		self.__l_freqs = []
		self.__l_co_dicts = []
		self.__l_b_tested = []
		self.__l_b_allowed = []
		self.__forbidden_state = []
		self.__cascade_els = []
		self.__glv_dict = dict()
		self.__status_db = []

	def init_for_move(self, full_db):
		self.__status_db = full_db

	def is_init_for_move(self):
		return self.__status_db != []

	# I expect not to use this. Full testing of all units for move
	def test_for_allowed(self, l_all_units, full_db):
		self.__status_db = full_db
		for kmove, vidx in self.__freq_tbl:
			move = list(kmove)
			unit_data = [move[0], move[2]]
			b_can_do, b_tested = True, False
			if unit_data in l_all_units:
				b_tested = False
			if b_tested:
				b_can_do = forbidden.test_move_forbidden(move, self.__forbidden_state, full_db,
														 self.__cascade_els, self.__glv_dict)
			self.__l_b_tested[vidx] = b_tested
			self.__l_b_allowed[vidx] = b_can_do
			# if b_can_do:
			# 	l_unit_moves = unit_dict.get(tuple(unit_data), [])
			# 	l_unit_moves.append(oid)
			# 	unit_dict[tuple(unit_data)] = l_unit_moves

	def reset_for_move(self):
		self.__status_db = []
		self.__l_b_tested = [False for _ in self.__l_freqs]
		self.__l_b_allowed = [False for _ in self.__l_freqs]

	def is_init_for_game(self):
		return self.__forbidden_state != []

	def init_for_game(self, forbidden_state, fnt, cascade_els, glv_dict):
		self.__forbidden_state = forbidden_state
		self.__cascade_els = cascade_els
		self.__glv_dict = glv_dict
		# freq_tbl, id_dict, unit_dict, max_id = dict(), dict(), dict(), [-1]
		max_id = [-1]
		try:
			o_fn = expanduser(fnt)
			with open(o_fn, 'rb') as o_fhr:
				o_csvr = csv.reader(o_fhr, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
				_, _, version_str, _, snum_orders = next(o_csvr)
				version = int(version_str)
				if version != wdconfig.c_freq_stats_version:
					raise IOError
				for iorder in range(int(snum_orders)) :
					row = next(o_csvr)
					l_co_oids = next(o_csvr)
					oid, co_dict = int(row[1]), dict()
					if oid > max_id[0]:
						max_id[0] = oid
					move = row[2:]
					unit_data = [move[0], move[2]]
					# b_can_do = True
					# if unit_data not in l_all_units:
					# 	b_can_do = False
					# if b_can_do and forbidden.test_move_forbidden(move, l_f_rules, full_db, cascade_els, glv_dict):
					# 	b_can_do = False
					# if b_can_do:
					l_unit_moves = self.__d_units.get(tuple(unit_data), [])
					l_unit_moves.append(oid)
					self.__d_units[tuple(unit_data)] = l_unit_moves
					idx = len(self.__l_freqs)
					self.__l_freqs.append(int(row[0]))
					self.__l_oids.append(oid)
					self.__l_co_dicts.append(co_dict)
					self.__l_b_tested.append(False)
					self.__l_b_allowed.append(False)
					self.__freq_tbl[tuple(move)] = idx
					self.__d_oids_to_move[oid] = tuple(move)
					self.__d_oids[oid] = idx

					for co_soid in l_co_oids:
						co_oid, co_freq = co_soid.split(':')
						co_dict[int(co_oid)] = int(co_freq)

		except IOError:
			print('Error. Cannot load order frequency stats.')
			return [-1]

		return max_id

	def get_moves(self, unit_data, l_order_templ, target_neighbors=[], max_len=1000, max_num_moves=1000):
		# success_orders_freq, oid_dict, success_unit_dict = success_orders_data

		# b_max_reached = False
		order_list = []
		l_all_poss_oids = self.__d_units.get(tuple(unit_data), [])
		if l_all_poss_oids == []:
			return order_list
		for poss_oid in l_all_poss_oids:
			if len(order_list) >= max_num_moves:
				return order_list
			korder = self.__d_oids_to_move[poss_oid]
			b_success = False
			for one_templ in l_order_templ:
				if b_success:
					break
				b_success = True
				order = list(one_templ)
				for iel, el in enumerate(korder):
					if iel > max_len:
						break
					if order[iel] == '?':
						order[iel] = el
					else:
						if order[iel] != el:
							b_success = False
							break

			if b_success and target_neighbors != []:
				u = set.intersection(set(korder) - set([unit_data[1]]), set(target_neighbors))
				b_success = (len(u) > 0)

			if b_success:
				# order_list.append(list(korder))
				move = list(korder) # not really very deep copy, may be too shallow
				idx = self.__d_oids[poss_oid]
				if not self.__l_b_tested[idx]:
					b_can_do = not forbidden.test_move_forbidden(move, self.__forbidden_state, self.__status_db,
															 self.__cascade_els, self.__glv_dict)
					self.__l_b_tested[idx] = True
					self.__l_b_allowed[idx] = b_can_do
				else:
					b_can_do = self.__l_b_allowed[idx]
				if b_can_do:
					order_list.append(move)

		random.shuffle(order_list)
		return order_list

	def get_colist_moves(self, order, colist_req_thresh, colist_strong_thresh):
		# freq_tbl, oid_dict, unit_dict = freq_data
		l_colist_orders = []
		idx = self.__freq_tbl[tuple(order)]
		# freq, id, colist_dict = freq_tbl[tuple(order)]
		colist_dict, id, freq = self.__l_co_dicts[idx], self.__l_oids[idx], self.__l_freqs[idx]
		l_scores, l_oids, l_freqs = [], [], []
		for koid, vco_freq in colist_dict.iteritems():
			# l_scores.append((vco_freq) / float(freq))
			l_freqs.append(vco_freq)
			l_oids.append(koid)
		if len(l_freqs) == 0:
			return [], [], []
		i_max_score = max(xrange(len(l_freqs)), key=l_freqs.__getitem__)
		max_score =  float(l_freqs[i_max_score]) / float(freq)
		l_b_req, l_b_rev_req = [False for score in l_freqs], [False for score in l_freqs]
		if max_score > colist_strong_thresh:
			l_b_req[i_max_score] = True
			rev_oid = l_oids[i_max_score]
			# rev_order = oid_dict.get(oid, [])
			rev_order = self.__d_oids_to_move.get(rev_oid, [])
			if rev_order != []:
				# rev_idx = self.__freq_tbl[tuple(order)]
				rev_idx = self.__d_oids[rev_oid]
				rev_colist_dict, rev_id, rev_freq = self.__l_co_dicts[rev_idx], self.__l_oids[rev_idx], self.__l_freqs[rev_idx]
				# rev_freq, rev_id, rev_colist_dict = freq_tbl[tuple(rev_order)]
				rev_rev_freq = rev_colist_dict.get(id, -1)
				if rev_rev_freq != -1:
					rev_score = float(rev_rev_freq) / float(rev_freq)
					if rev_score > colist_req_thresh:
						l_b_rev_req[i_max_score] = True

		l_colist_orders = [self.__d_oids_to_move[oid] for oid in l_oids]

		return l_colist_orders, l_b_req, l_b_rev_req



def load_order_freq(l_all_units,
					forbidden_state, full_db, cascade_els, glv_dict, fnt):
	freq_tbl, id_dict, unit_dict, max_id = dict(), dict(), dict(), [-1]
	try:
		o_fn = expanduser(fnt)
		with open(o_fn, 'rb') as o_fhr:
			o_csvr = csv.reader(o_fhr, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
			_, _, version_str, _, snum_orders = next(o_csvr)
			version = int(version_str)
			if version != wdconfig.c_freq_stats_version:
				raise IOError
			for iorder in range(int(snum_orders)) :
				row = next(o_csvr)
				l_co_oids = next(o_csvr)
				oid, co_dict = int(row[1]), dict()
				if oid > max_id[0]:
					max_id[0] = oid
				move = row[2:]
				unit_data = [move[0], move[2]]
				b_can_do = True
				if unit_data not in l_all_units:
					b_can_do = False
				if b_can_do and forbidden.test_move_forbidden(move, forbidden_state, full_db, cascade_els, glv_dict):
					b_can_do = False
				if b_can_do:
					l_unit_moves = unit_dict.get(tuple(unit_data), [])
					l_unit_moves.append(oid)
					unit_dict[tuple(unit_data)] = l_unit_moves
				freq_tbl[tuple(move)] = (int(row[0]), oid, co_dict)
				id_dict[oid] = tuple(row[2:])
				for co_soid in l_co_oids:
					co_oid, co_freq = co_soid.split(':')
					co_dict[int(co_oid)] = int(co_freq)

	except IOError:
		print('Error. Cannot load order frequency stats.')
		return dict(), dict(), dict(), [-1]

	return freq_tbl, id_dict, unit_dict, max_id



def create_order_freq_tbl(orders_list, order_status_list):
	success_orders_freq, failed_orders_freq = dict(), dict()
	max_id = [-1]
	success_id_dict, failed_id_dict = dict(), dict()

	load_order_freq_tbl(success_orders_freq, success_id_dict, max_id, wdconfig.orders_success_fnt)
	# load_order_freq_tbl(failed_orders_freq, wdconfig.orders_failed_fnt)


	def add_to_tbl(freq_tbl, id_dict, order, max_id):
		freq_data = freq_tbl.get(order, (-1, -1))
		if freq_data[0] == -1:
			max_id[0] += 1
			freq_tbl[order] = (1, max_id[0], dict())
			id_dict[max_id[0]] = order
		else:
			freq_tbl[order] = (freq_data[0] + 1, freq_data[1], freq_data[2])

	def update_colist(freq_tbl, order, l_orders):
		freq, id, colist_dict = freq_tbl[order]
		# print('colist_dict:', colist_dict)
		# co_ids = []
		for i_co_order, co_order in enumerate(l_orders):
			co_freq, co_id, co_colist_dict = freq_tbl[co_order]
			if id == co_id:
				continue
			# co_ids.append(str(co_id))
			old_num_co = colist_dict.get(co_id, -1)
			if old_num_co == -1:
				if freq < wdconfig.c_freq_stats_newbie_thresh:
					colist_dict[co_id] = 1
			else:
				colist_dict[co_id] = old_num_co + 1
		# print('co_ids:', ' '.join(co_ids))

	def clean_colist(freq_tbl, order):
		freq, id, colist_dict = freq_tbl[order]
		if freq < wdconfig.c_freq_stats_mature_thresh:
			return
		new_colist_dict = dict()
		for kid, vfreq in colist_dict.iteritems():
			fract = float(vfreq) / float(freq)
			if fract > wdconfig.c_freq_stats_drop_thresh:
				new_colist_dict[kid] = vfreq
		freq_tbl[order] = (freq, id, new_colist_dict)





	l_success_orders, l_failed_orders = [], []
	for iorder, order in enumerate(orders_list):
		if order_status_list[iorder].status:
			add_to_tbl(success_orders_freq, success_id_dict, tuple(order), max_id)
			l_success_orders.append(tuple(order))
		else:
			add_to_tbl(failed_orders_freq, failed_id_dict, tuple(order), max_id)
			l_failed_orders.append(tuple(order))

	if len(orders_list) > 0:
		del iorder, order

	for order in l_success_orders:
		update_colist(success_orders_freq, order, l_success_orders)
		clean_colist(success_orders_freq, order)

	def save_tbl(freq_tbl, fnt):
		o_fn = expanduser(fnt)
		o_fhw = open(o_fn, 'wb')
		o_csvw = csv.writer(o_fhw, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
		o_csvw.writerow(['frequency stats', 'version', wdconfig.c_freq_stats_version, 'num items', len(freq_tbl)])
		for korder, vdata in freq_tbl.iteritems():
			row = [vdata[0], vdata[1]] + list(korder)
			o_csvw.writerow(row)
			row = [str(kco) + ':' + str(vco) for kco, vco in vdata[2].iteritems()]
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

			for istats, cont_stats in enumerate(cont_stats_list):
				b_match,_ = learn.learn_one_story_step2(full_db + orders_db, [order], cascade_els, [results_pl[iorder]],
													def_article_dict, [], [], glv_dict, sess=None, event_step_id=-1,
													expected_but_not_found_list=[], level_depr=0, gg_cont=cont_stats.get_cont(),
													b_blocking_depr=False, b_test_rule=True)
				# if istats == 0 and not b_match:
				# 	print('First rule failed. Why?')
				# 	b_match,_ = learn.learn_one_story_step2(full_db + orders_db, [order], cascade_els, [results_pl[iorder]],
				# 										def_article_dict, [], [], glv_dict, sess=None, event_step_id=-1,
				# 										expected_but_not_found_list=[], level_depr=0, gg_cont=cont_stats.get_cont(),
				# 										b_blocking_depr=False, b_test_rule=True)
				cont_stats.add_match(b_match)
				cont_match_list.append(b_match)

			if cont_stats_mgr.get_W() != None:
				cont_stats_mgr.add_prediction(cont_stats_mgr.predict_success_rate(cont_match_list))
			cont_stats_mgr.add_match(b_target_success, cont_match_list)

		orders_db.append(order_rec)

	return b_keep_working

# These two are not specific to wd and should move to compare_conts or add learn
def compare_conts_learn(db_cont_mgr):
	cont_stats_mgr = db_cont_mgr.get_cont_stats_mgr()
	if cont_stats_mgr:
		cont_stats_mgr.do_learn()
	return

def cont_stats_save(db_cont_mgr, fnt):
	cont_stats_mgr = db_cont_mgr.get_cont_stats_mgr()
	if cont_stats_mgr:
		cont_stats_mgr.save(fnt)
	return

def init_cont_stats_from_file():
	db_cont_mgr = addlearn.cl_cont_mgr()
	b_load_done = db_cont_mgr.init_cont_stats_mgr_from_file(wdconfig.c_cont_stats_fnt,
															wdconfig.c_cont_forbidden_fn, wdconfig.c_cont_forbidden_version,
															wdconfig.c_b_analyze_conts, wdconfig.c_b_modify_conts)
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

		# if iorder % 10 == 0:
		# 	b_keep_working = create_new_conts(glv_dict, db_cont_mgr, db_len_grps, i_active_cont)
		# 	save_db_status(db_len_grps, db_cont_mgr)
		# 	if not b_keep_working:
		# 		break
			# save_len_grps(db_len_grps, blocked_len_grps)

	learn_vars[0] = event_step_id

	return b_keep_working

def register_country_moves(wd_game_state, scountry, l_country_orders):
	wd_game_state.set_country_moves(wd_game_state.get_d_countries()[scountry], l_country_orders)
