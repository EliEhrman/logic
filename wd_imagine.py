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

def get_moves(order_templ, success_orders_freq, max_len=1000):
	order_list = []
	for korder, vfreq in success_orders_freq.iteritems():
		order = list(order_templ)
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
			order_list.append(list(korder))

	return order_list

def select_move(order_templ, success_orders_freq):
	order_list = get_moves(order_templ, success_orders_freq)

	return random.choice(order_list)

def create_move_orders(	init_db, army_can_pass_tbl, fleet_can_pass_tbl, status_db, db_cont_mgr,
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
				order = select_move(order_template, success_orders_freq)
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

def create_move_orders2(init_db, army_can_pass_tbl, fleet_can_pass_tbl, status_db, db_cont_mgr,
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

	cont_stats_mgr = db_cont_mgr.get_cont_stats_mgr()
	num_rules = len(cont_stats_mgr.get_cont_stats_list())
	l_rules = [cont_stat.get_cont().get_rule() for cont_stat in cont_stats_mgr.get_cont_stats_list()]
	l_scvos = [mr.gen_cvo_str(rule) for rule in l_rules]
	l_lens = [mr.get_olen(scvo) for scvo in l_scvos]
	l_gens_recs = [cont_stat.get_cont().get_gens_rec() for cont_stat in cont_stats_mgr.get_cont_stats_list()]

	orders_list = []
	icountry_list = []

	for icountry in range(1, len(country_names_tbl)):
		scountry = country_names_tbl[icountry]
		print('Thinking for ', scountry)
		l_poss_supplies = set()
		unit_list = unit_owns_tbl.get(scountry, None)
		if unit_list == None:
			continue
		for unit_data in unit_list:
			order_template = [unit_data[0], 'in', unit_data[1], 'move', 'to', '?']
			l_pos_orders = get_moves(order_template, success_orders_freq)
			for poss_order in l_pos_orders:
				dest_name = poss_order[5]
				prev_owner = terr_owner_dict.get(dest_name, 'neutral')
				if dest_name in supply_set and scountry != prev_owner:
					l_poss_supplies.add(dest_name)
		# if len(l_poss_supplies) == 0:
		# 	continue
		random.shuffle(unit_list)
		del l_pos_orders, order_template, poss_order, dest_name, prev_owner
		l_unit_avail = [True for udata in unit_list]
		country_orders_list = []
		for target_name in l_poss_supplies:
			# target_name = random.sample(l_poss_supplies, 1)[0]
			target_template = ['?', '?', 'in', '?', 'move', 'to', target_name, 'succeeded', '?']
			l_all_poss_orders = []
			l_successes = []
			for iunit, unit_data in enumerate(unit_list):
				if not l_unit_avail[iunit]:
					continue
				order_template = [unit_data[0], 'in', unit_data[1]]
				l_poss_orders = get_moves(order_template, success_orders_freq, max_len=len(order_template)-1)
				l_poss_order_phrases = els.convert_list_to_phrases(l_poss_orders)
				for iorder, poss_order in enumerate(l_poss_order_phrases):
					order_rec, _ = mr.make_rec_from_phrase_list([poss_order])
					order_scvo = mr.gen_cvo_str(order_rec)
					order_len = mr.get_olen(order_scvo)
					match_pattern = [False for i in range(num_rules)]
					b_one_success = False
					for irule, rule in enumerate(l_rules):
						if order_len != l_lens[irule] or order_scvo != l_scvos[irule]:
							continue
						if not mr.does_match_rule(glv_dict, l_rules[irule], order_rec):
							continue
						result_rec = mr.replace_vars_in_phrase(order_rec, l_gens_recs[irule])
						if not els.match_rec_to_templ(result_rec, target_template):
							continue
						b_one_success = True
						match_pattern[irule] = True
					if b_one_success:
						success = cont_stats_mgr.predict_success_rate(match_pattern)
						l_successes.append([success, poss_order, iunit, match_pattern, l_poss_orders[iorder]])
			# end of loop over units who can reach the country
			if l_successes == []:
				continue
			random.shuffle(l_successes)
			first_stage_success, first_stage_order, first_stage_iunit, \
			first_stage_match_pattern, first_stage_order_words = \
				sorted(l_successes, key=lambda x: x[0], reverse=True)[0]
			l_unit_avail[first_stage_iunit] = False
			country_orders_list.append(first_stage_order_words)
			# first_stage_order = l_poss_order_phrases[i_stage_order]

			l_blocks = []
			for i_opp_country in range(1, len(country_names_tbl)):
				if i_opp_country == icountry:
					continue
				sopp = country_names_tbl[i_opp_country]
				opp_unit_list = unit_owns_tbl.get(sopp, None)
				if opp_unit_list == None:
					continue

				for opp_unit_data in opp_unit_list:
					order_template = [opp_unit_data[0], 'in', opp_unit_data[1]]
					l_poss_orders = get_moves(order_template, success_orders_freq, max_len=len(order_template)-1)
					l_poss_order_phrases = els.convert_list_to_phrases(l_poss_orders)
					l_successes = []
					for iorder, poss_order in enumerate(l_poss_order_phrases):
						order_rec, _ = mr.make_rec_from_phrase_list([first_stage_order, poss_order])
						order_scvo = mr.gen_cvo_str(order_rec)
						order_len = mr.get_olen(order_scvo)
						# match_pattern = [False for i in range(num_rules)]
						match_pattern = list(first_stage_match_pattern)
						b_one_success = False
						for irule, rule in enumerate(l_rules):
							if order_len != l_lens[irule] or order_scvo != l_scvos[irule]:
								continue
							if not mr.does_match_rule(glv_dict, l_rules[irule], order_rec):
								continue
							result_rec = mr.replace_vars_in_phrase(order_rec, l_gens_recs[irule])
							if not els.match_rec_to_templ(result_rec, target_template):
								continue
							b_one_success = True
							match_pattern[irule] = True
						if b_one_success:
							success = cont_stats_mgr.predict_success_rate(match_pattern)
							l_blocks.append([success, match_pattern])
			# end loop over opposing countries
			if l_blocks == []:
				continue

			random.shuffle(l_blocks)
			block_stage_success, block_stage_match_pattern = \
				sorted(l_blocks, key=lambda x: x[0], reverse=False)[0]

			l_rejoiners = []
			for iunit, unit_data in enumerate(unit_list):
				if not l_unit_avail[iunit]:
					continue
				order_template = [unit_data[0], 'in', unit_data[1]]
				l_poss_orders = get_moves(order_template, success_orders_freq, max_len=len(order_template)-1)
				l_poss_order_phrases = els.convert_list_to_phrases(l_poss_orders)
				for iorder, poss_order in enumerate(l_poss_order_phrases):
					order_rec, _ = mr.make_rec_from_phrase_list([first_stage_order, poss_order])
					order_scvo = mr.gen_cvo_str(order_rec)
					order_len = mr.get_olen(order_scvo)
					# match_pattern = [False for i in range(num_rules)]
					match_pattern = list(block_stage_match_pattern)
					b_one_success = False
					for irule, rule in enumerate(l_rules):
						if order_len != l_lens[irule] or order_scvo != l_scvos[irule]:
							continue
						if not mr.does_match_rule(glv_dict, l_rules[irule], order_rec):
							continue
						result_rec = mr.replace_vars_in_phrase(order_rec, l_gens_recs[irule])
						if not els.match_rec_to_templ(result_rec, target_template):
							continue
						b_one_success = True
						match_pattern[irule] = True
					if b_one_success:
						success = cont_stats_mgr.predict_success_rate(match_pattern)
						l_rejoiners.append([success, l_poss_orders[iorder], iunit])

			if l_rejoiners == []:
				continue

			random.shuffle(l_rejoiners)
			rejoiner_success, rejoiner_stage_order, rejoiner_stage_iunit = \
				sorted(l_rejoiners, key=lambda x: x[0], reverse=True)[0]
			l_unit_avail[rejoiner_stage_iunit] = False
			country_orders_list.append(rejoiner_stage_order)
		# end loop over target names
		for iunit, unit_data in enumerate(unit_list):
			if not l_unit_avail[iunit]:
				continue

			order_template = [unit_data[0], 'in', unit_data[1], 'move', 'to', '?']
			l_pos_orders = get_moves(order_template, success_orders_freq)

			country_orders_list.append(random.choice(l_pos_orders))

		if country_orders_list != []:
			orders_list += country_orders_list
			icountry_list.append(icountry)


	orders_db = els.convert_list_to_phrases(orders_list)
	success_list = [True for o in orders_list]
	return orders_list, orders_db, success_list, icountry_list

