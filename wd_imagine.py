from __future__ import print_function
import random
import numpy as np

import wdconfig
import wdlearn
import els
import learn
import rules
import makerecs as mr

# def imagine_init

class cl_distance_calc(object):
	def __init__(self):
		# The goal is to create a matrix of distances between all the terrs
		# The method is to fill in the places where the dstance is 1 by virtue of a direct move
		# between the two terrs
		# The keep iterating over all the matrix where if a is x from b and b is y from c and x + y
		# is less than the figure in the matrix, replace it
		success_orders_freq, success_id_dict, max_id = dict(), dict(), [-1]
		wdlearn.load_order_freq_tbl(success_orders_freq, success_id_dict, max_id, wdconfig.orders_success_fnt)
		def add_to_dict(d, l, sterr):
			iterr = d.get(sterr, -1)
			if iterr == -1:
				d[sterr] = len(l)
				l.append(sterr)

		d_terrs, l_terr_ids = dict(), []
		for koid, vmove in success_id_dict.iteritems():
			if vmove[3] == 'move':
				add_to_dict(d_terrs, l_terr_ids, vmove[2])
				add_to_dict(d_terrs, l_terr_ids, vmove[5])

		num_terrs = len(d_terrs)
		matrix = [[1000 for _ in range(num_terrs)] for _ in range(num_terrs)]

		for koid, vmove in success_id_dict.iteritems():
			if vmove[3] == 'move':
				matrix[d_terrs[vmove[2]]][d_terrs[vmove[5]]] = 1

		num_stalled = 0
		while num_stalled < 3:
			b_better = False
			for iterr, terr_row in enumerate(matrix):
				for iterr2, dist in enumerate(terr_row):
					if iterr == iterr2:
						continue
					if dist > 1:
						l_dist_data = [(dist1_3 + matrix[iterr3][iterr2], iterr3) for iterr3, dist1_3 in enumerate(terr_row)
										if dist1_3 < 1000 and matrix[iterr3][iterr2] < 1000 and iterr3 != iterr and iterr3 != iterr2]
						if l_dist_data != []:
							best = min(l_dist_data, key=lambda x:x[0])
							if best[0] < dist:
								matrix[iterr][iterr2] = best[0]
								b_better = True

			num_stalled = 0 if b_better else (num_stalled + 1)

		# Finally, zero distances to self
		for iterr, terr_row in enumerate(matrix):
			for iterr2, dist in enumerate(terr_row):
				if iterr != iterr2:
					continue
				matrix[iterr][iterr2] = 0

		self.__1_neighbors = []
		for iterr in range(len(l_terr_ids)):
			sin = set()
			for iterr2, dist in enumerate(matrix[iterr]):
				if dist <= 1:
					sin.add(iterr2)
			self.__1_neighbors.append([l_terr_ids[ine] for ine in sin])

		self.__d_terrs = d_terrs
		self.__matrix = matrix
		self.__l_terrs = l_terr_ids

	def get_distance(self, sterr1, sterr2):
		return self.__matrix[self.__d_terrs[sterr1]][self.__d_terrs[sterr2]]

	def get_neighbors(self, sterr, max_dist):
		if max_dist != 1:
			print('Only 1-neighbors pre-calculated at present. Exiting')
			exit(1)
		iterr = self.__d_terrs[sterr]
		return self.__1_neighbors[iterr]

def get_colist_moves(order, freq_data, colist_req_thresh, colist_strong_thresh):
	freq_tbl, oid_dict, unit_dict = freq_data
	l_colist_orders = []
	freq, id, colist_dict = freq_tbl[tuple(order)]
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
		oid = l_oids[i_max_score]
		rev_order = oid_dict.get(oid, [])
		if rev_order != []:
			rev_freq, rev_id, rev_colist_dict = freq_tbl[tuple(rev_order)]
			rev_rev_freq = rev_colist_dict.get(id, -1)
			if rev_rev_freq != -1:
				rev_score = float(rev_rev_freq) / float(rev_freq)
				if rev_score > colist_req_thresh:
					l_b_rev_req[i_max_score] = True

	l_colist_orders = [oid_dict[oid] for oid in l_oids]

	return l_colist_orders, l_b_req, l_b_rev_req

	# l_ifiltered = [iscore for iscore, score in enumerate(l_scores) if score > colist_strong_thresh]
	# l_thresh_scores = [l_scores[iscore] for iscore in l_ifiltered]
	# l_thresh_oids = [l_oids[iscore] for iscore in l_ifiltered]
	# l_b_req = [score > colist_req_thresh for score in l_thresh_scores]
	# # l_sorted = np.argpartition(l_scores, -2)[-2:]
	# l_b_rev_req = [False for _ in l_thresh_oids]
	# for ithresh, oid in enumerate(l_thresh_oids):
	# 	if not l_b_req[ithresh]:
	# 		continue
	# 	rev_order = oid_dict.get(oid, [])
	# 	if rev_order == []:
	# 		continue
	# 	rev_freq, rev_id, rev_colist_dict = freq_tbl[tuple(rev_order)]
	# 	rev_rev_freq = rev_colist_dict.get(id, -1)
	# 	if rev_rev_freq == -1:
	# 		continue
	# 	rev_score = float(rev_rev_freq) / float(rev_freq)
	# 	if rev_score > colist_req_thresh:
	# 		l_b_rev_req[ithresh] = True


def get_moves(unit_data, l_order_templ, success_orders_data, max_len=1000, max_num_moves=1000):
	success_orders_freq, oid_dict, success_unit_dict = success_orders_data

	b_max_reached = False
	order_list = []
	l_all_poss_oids = success_unit_dict.get(tuple(unit_data), [])
	if l_all_poss_oids == []:
		return order_list

	# for korder, vfreq in success_orders_freq.iteritems():
	for poss_oid in l_all_poss_oids:
		korder = oid_dict[poss_oid]
		for one_templ in l_order_templ:
			order = list(one_templ)
			b_success = True
			for iel, el in enumerate(korder):
				if iel > max_len:
					break
				if order[iel] == '?':
					order[iel] = el
				else:
					if order[iel] != el:
						b_success = False
						break
			if b_success:
				if len(order_list) >= max_num_moves:
					b_max_reached = True
					break
				order_list.append(list(korder))
		if b_max_reached:
			break

	random.shuffle(order_list)
	return order_list

def select_move(unit_data, order_templ, success_orders_freq):
	order_list = get_moves(unit_data, [order_templ], [success_orders_freq, dict(), dict()])

	return random.choice(order_list)

def create_move_orders_by_monte(	init_db, army_can_pass_tbl, fleet_can_pass_tbl, status_db, db_cont_mgr,
									country_names_tbl, unit_owns_tbl,
									all_the_dicts, terr_owns_tbl, supply_tbl, num_montes, preferred_nation,
									b_predict_success):
	glv_dict, def_article_dict, cascade_dict = all_the_dicts
	cascade_els = [el for el in cascade_dict.keys() if cascade_dict[el]]
	success_orders_freq = dict()
	wdlearn.load_order_freq_tbl(success_orders_freq, wdconfig.orders_success_fnt)
	if b_predict_success:
		icc_list = db_cont_mgr.get_conts_above(wdconfig.c_use_rule_thresh)
	full_db = init_db + status_db

	supply_set = set()
	for kcountry, vterr_list in supply_tbl.iteritems():
		for terr_stat in vterr_list:
			supply_set.add(terr_stat[1])

	country_dict = dict()
	for icountry, country_name in enumerate(country_names_tbl):
		country_dict[country_name] = icountry

	terr_owner_dict = dict()
	for kcountry, vterr_list in 	terr_owns_tbl.iteritems():
		for terr_data in vterr_list:
			# icountry = country_dict[kcountry]
			terr_owner_dict[terr_data[0]] = kcountry
			# if terr_data[0] in supply_set:
			# 	num_supplies_list[icountry] += 1

	monte_success_lists, monte_num_supplies_list, monte_orders_list = [], [], []

	for imonte in range(num_montes):
		num_supplies_list = [0 for country in country_names_tbl]
		orders_list, icountry_list, order_country_list = [], [], []
		for icountry in range(1, len(country_names_tbl)):
			scountry = country_names_tbl[icountry]
			print('Orders for ', scountry)
			unit_list = unit_owns_tbl.get(scountry, None)
			if unit_list == None:
				continue
			icountry_list.append(icountry)
			for unit_data in unit_list:
				order_template = [unit_data[0], 'in', unit_data[1], 'move', 'to', '?']
				order = select_move(unit_data, order_template, success_orders_freq)
				orders_list.append(order)
				order_country_list.append(country_names_tbl[icountry])

		orders_db = els.convert_list_to_phrases(orders_list)
		num_orders = len(orders_db)
		success_list = []
		for iorder in range(num_orders):
			if b_predict_success:
				order = orders_db.pop(0)
				order_rec, _ = mr.make_rec_from_phrase_list([order])
				order_rec_AND, _ = mr.make_rec_from_phrase_list([order], b_force_AND=True)
				success_result = order + [[rules.rec_def_type.obj, 'succeeded']]
				out_str = ''
				out_str = els.print_phrase(order, order, out_str, def_article_dict)
				print('Testing order: ', out_str)

				b_succeed = True
				for icc in icc_list:
					curr_cont = db_cont_mgr.get_cont(icc)
					if curr_cont.get_level() > 1:
						cont_result = mr.replace_vars_in_phrase(order_rec_AND, curr_cont.get_gens_rec())
					else:
						cont_result = mr.replace_vars_in_phrase(order_rec, curr_cont.get_gens_rec())

					if not mr.match_rec_exact(success_result, cont_result[1:-1]):
						continue

					stats = learn.learn_one_story_step2(els.make_rec_list(full_db + orders_db), [order], cascade_els, [success_result],
														def_article_dict=def_article_dict, db_len_grps = None,
														el_set_arr=None, glv_dict=glv_dict, sess = None, event_step_id= -1,
														expected_but_not_found_list = [], level_depr=0, gg_cont=curr_cont,
														b_blocking_depr=False, b_test_rule=True)
					if stats[0] and stats[1] and curr_cont.is_blocking():
						print('Order blocked:', ' '.join(orders_list[iorder]), 'by rule:', curr_cont.get_rule_str())
						b_succeed = False
						break
				orders_db.append(order)
			else: # if not b_predict_success
				b_succeed = True

			# end if b_predict success
			if b_succeed:
				terr_at = orders_list[iorder][-1]
			else:
				terr_at = orders_list[iorder][2]

			prev_owner = terr_owner_dict.get(terr_at, 'neutral')
			if terr_at in supply_set and order_country_list[iorder] != prev_owner:
				num_supplies_list[country_dict[prev_owner]] -= 1
				num_supplies_list[country_dict[order_country_list[iorder]]] += 1

			success_list.append(b_succeed)
		monte_success_lists.append(success_list)
		monte_num_supplies_list.append(num_supplies_list)
		monte_orders_list.append(orders_list)

	# status is highest value, index of monte. Initialized to 0 insead of -1 just in casee..
	best_monte_by_country = [[-1000, 0] for country in country_names_tbl]
	for imonte in range(num_montes):
		for icountry, status in enumerate(best_monte_by_country):
			if monte_num_supplies_list[imonte][icountry] > status[0]:
				status[0] = monte_num_supplies_list[imonte][icountry]
				status[1] = imonte

	orders_list, success_list = [], []
	for iorder, country in enumerate(order_country_list):
		icountry = country_dict[country]
		if preferred_nation == None or country != preferred_nation:
			imonte = random.randint(0, num_montes-1)
		else:
			imonte = best_monte_by_country[icountry][1]

		orders_list.append(monte_orders_list[imonte][iorder])
		success_list.append(monte_success_lists[imonte][iorder])

	return orders_list, orders_db, success_list, icountry_list


