from __future__ import print_function
import random
import collections

import wdconfig
import wdlearn
import els
import learn
import rules
import makerecs as mr

# def imagine_init

def select_move(order_templ, success_orders_freq):
	order_list = []
	for korder, vfreq in success_orders_freq.iteritems():
		order = list(order_templ)
		b_success = True
		for iel, el in enumerate(korder):
			if order[iel] == '?':
				order[iel] = el
			else:
				if order[iel] != el:
					b_success = False
					break
		if b_success:
			order_list.append(order)

	return random.choice(order_list)

def create_move_orders(	init_db, status_db, db_cont_mgr, country_names_tbl, unit_owns_tbl,
						all_the_dicts, terr_owns_tbl, supply_tbl):
	glv_dict, def_article_dict, cascade_dict = all_the_dicts
	cascade_els = [el for el in cascade_dict.keys() if cascade_dict[el]]
	success_orders_freq = dict()
	wdlearn.load_order_freq_tbl(success_orders_freq, wdconfig.orders_success_fnt)
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

	for imonte in range(wdconfig.c_num_montes):
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
				order = select_move(order_template, success_orders_freq)
				orders_list.append(order)
				order_country_list.append(country_names_tbl[icountry])

		orders_db = els.convert_list_to_phrases(orders_list)
		num_orders = len(orders_db)
		success_list = []
		for iorder in range(num_orders):
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
					cont_result = mr.get_result_for_cvo_and_rec(order_rec_AND, curr_cont.get_gens_rec())
				else:
					cont_result = mr.get_result_for_cvo_and_rec(order_rec, curr_cont.get_gens_rec())

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
			if b_succeed:
				terr_at = orders_list[iorder][-1]
			else:
				terr_at = orders_list[iorder][2]

			prev_owner = terr_owner_dict.get(terr_at, 'neutral')
			if terr_at in supply_set and order_country_list[iorder] != prev_owner:
				num_supplies_list[country_dict[prev_owner]] -= 1
				num_supplies_list[country_dict[order_country_list[iorder]]] += 1

			orders_db.append(order)
			success_list.append(b_succeed)
		monte_success_lists.append(success_list)
		monte_num_supplies_list.append(num_supplies_list)
		monte_orders_list.append(orders_list)

	# status is highest value, index of monte. Initialized to 0 insead of -1 just in casee..
	best_monte_by_country = [[-1000, 0] for country in country_names_tbl]
	for imonte in range(wdconfig.c_num_montes):
		for icountry, status in enumerate(best_monte_by_country):
			if monte_num_supplies_list[imonte][icountry] > status[0]:
				status[0] = monte_num_supplies_list[imonte][icountry]
				status[1] = imonte

	orders_list, success_list = [], []
	for iorder, country in enumerate(order_country_list):
		icountry = country_dict[country]
		if country == 'england':
			imonte = best_monte_by_country[icountry][1]
		else:
			imonte = random.randint(0, wdconfig.c_num_montes-1)
		orders_list.append(monte_orders_list[imonte][iorder])
		success_list.append(monte_success_lists[imonte][iorder])

	return orders_list, orders_db, success_list, icountry_list

