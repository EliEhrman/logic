from __future__ import print_function
import random
import copy

import wdconfig
import wdlearn
import els
import learn
import rules
import makerecs as mr
import wd_imagine

def sel_orders(	glv_dict, cont_stats_mgr, l_target_templates, unit_list, success_orders_freq,
				num_rules, l_rules, l_lens, l_scvos, l_gens_recs, prev_stage_score, l_unit_avail, country_orders_list,
				first_stage_order, prev_stage_match_pattern, b_my_move, b_offensive):
	l_successes = []
	for iunit, unit_data in enumerate(unit_list):
		if len(l_successes) > wdconfig.c_classic_AI_max_successes:
			break
		if b_my_move and not l_unit_avail[iunit]:
			continue
		order_template = [unit_data[0], 'in', unit_data[1]]
		l_poss_orders = wd_imagine.get_moves([order_template], success_orders_freq, max_len=len(order_template) - 1)
		l_poss_order_phrases = els.convert_list_to_phrases(l_poss_orders)
		for iorder, poss_order in enumerate(l_poss_order_phrases):
			if first_stage_order == None:
				order_rec, _ = mr.make_rec_from_phrase_list([poss_order])
			else:
				order_rec, _ = mr.make_rec_from_phrase_list([first_stage_order, poss_order])
			order_scvo = mr.gen_cvo_str(order_rec)
			order_len = mr.get_olen(order_scvo)
			if prev_stage_match_pattern == None:
				match_pattern = [False for i in range(num_rules)]
			else:
				match_pattern = copy.deepcopy(prev_stage_match_pattern)
			b_one_success = False
			for irule, rule in enumerate(l_rules):
				if order_len != l_lens[irule] or order_scvo != l_scvos[irule]:
					continue
				if not mr.does_match_rule(glv_dict, l_rules[irule], order_rec):
					continue
				result_rec = mr.replace_vars_in_phrase(order_rec, l_gens_recs[irule])
				b_found = False
				for target_template in l_target_templates:
					if els.match_rec_to_templ(result_rec, target_template):
						b_found = True
				if not b_found:
					continue
				b_one_success = True
				match_pattern[irule] = True
			if b_one_success:
				success = cont_stats_mgr.predict_success_rate(match_pattern)
				l_successes.append([success, poss_order, iunit, match_pattern, l_poss_orders[iorder]])
	# end of loop over units who can reach the country
	if l_successes == []:
		return False, None, None, None

	#		Off	Def
	#
	#	My	S	F
	#
	#	His	F	S
	#
	#
	#	O	M	S
	#	O	H	F
	#	D	M	F
	#	D	H	S
	#
	#	O = 1 D = 0
	#	M = 1 H = 0
	#	S = 1 F = 0
	#
	#	1	1	1
	#	1	0	0
	#	0	1	0
	#	0	0	1
	#
	#	To make the move succeed, highest score first, reverse = T
	#
	#	=> reverse=NXOR(O, M)

	random.shuffle(l_successes)
	this_stage_success, this_stage_order, this_stage_iunit, \
	this_stage_match_pattern, this_stage_order_words = \
		sorted(l_successes, key=lambda x: x[0], reverse=(b_my_move == b_offensive))[0]

	if b_my_move:
		b_take_the_move = False
		if prev_stage_score == None:
			b_take_the_move = True
			diff = 1.0
		elif b_offensive and this_stage_success > prev_stage_score:
			diff = this_stage_success - prev_stage_score
			b_take_the_move = True
		elif not b_offensive and this_stage_success < prev_stage_score:
			diff = prev_stage_score - this_stage_success
			b_take_the_move = True
		if b_take_the_move and random.random() < diff:
			l_unit_avail[this_stage_iunit] = False
			country_orders_list.append(this_stage_order_words)
			return True, this_stage_success, this_stage_order, this_stage_match_pattern
		else:
			return False, None, None, None

	return True, this_stage_success, this_stage_order, this_stage_match_pattern


def create_defensive(	glv_dict, cont_stats_mgr, unit_list, block_unit_list,
						target_name, country_orders_list, success_orders_freq,
						num_rules, l_rules, l_lens, l_scvos, l_gens_recs, l_unit_avail):
	# for unit_data in block_unit_list:
	# 	threaten_template = [unit_data[0], 'in', unit_data[1], 'move', 'to', target_name]
	# 	l_pos_orders = wd_imagine.get_moves(threaten_template, success_orders_freq)
	# 	threaten_unit_data = []
	# 	for poss_order in l_pos_orders:
	# 		dest_name = poss_order[5]
	# 		prev_owner = terr_owner_dict.get(dest_name, 'neutral')
	# 		if dest_name in supply_set and prev_owner == scountry:
	# 			s_poss_targets.add(dest_name)
	move_target_template = ['?', '?', 'in', '?', 'move', 'to', target_name, 'succeeded', '?']
	convoy_target_template = ['?', '?', 'in', '?', 'convoy', 'move', 'to', target_name, 'succeeded', '?']
	l_target_templates = [move_target_template, convoy_target_template]

	b_success, first_stage_success, first_stage_order, first_stage_match_pattern = \
		sel_orders(glv_dict, cont_stats_mgr, l_target_templates, block_unit_list, success_orders_freq,
				   num_rules, l_rules, l_lens, l_scvos, l_gens_recs,
				   prev_stage_score=None, l_unit_avail=[],
				   country_orders_list=[], first_stage_order=None,
				   prev_stage_match_pattern=None, b_my_move=False, b_offensive=False)

	if not b_success:
		return

	b_success, block_stage_success, _, block_stage_match_pattern = \
		sel_orders(glv_dict, cont_stats_mgr, l_target_templates, unit_list, success_orders_freq,
				   num_rules, l_rules, l_lens, l_scvos, l_gens_recs,
				   prev_stage_score=first_stage_success, l_unit_avail=l_unit_avail,
				   country_orders_list=country_orders_list, first_stage_order=first_stage_order,
				   prev_stage_match_pattern=first_stage_match_pattern, b_my_move=True, b_offensive=False)

	if not b_success or block_stage_success >= first_stage_success:
		return

	b_success, r1_stage_success, _, r1_stage_match_pattern = \
		sel_orders(glv_dict, cont_stats_mgr, l_target_templates, block_unit_list, success_orders_freq,
				   num_rules, l_rules, l_lens, l_scvos, l_gens_recs,
				   prev_stage_score=block_stage_success, l_unit_avail=[],
				   country_orders_list=[], first_stage_order=first_stage_order,
				   prev_stage_match_pattern=block_stage_match_pattern, b_my_move=False, b_offensive=False)

	if not b_success or r1_stage_success <= block_stage_success:
		return

	b_success, r2_stage_success, _, r2_stage_match_pattern = \
		sel_orders(glv_dict, cont_stats_mgr, l_target_templates, unit_list, success_orders_freq,
				   num_rules, l_rules, l_lens, l_scvos, l_gens_recs, l_unit_avail=l_unit_avail,
				   country_orders_list=country_orders_list, first_stage_order=first_stage_order,
				   prev_stage_match_pattern=r1_stage_match_pattern, b_my_move=True, b_offensive=False)

	if not b_success:
		return



def create_offensive(	glv_dict, cont_stats_mgr, unit_list, block_unit_list,
						target_name, country_orders_list, success_orders_freq,
						num_rules, l_rules, l_lens, l_scvos, l_gens_recs, l_unit_avail):
	# target_template = ['?', '?', 'in', '?', 'move', 'to', target_name, 'succeeded', '?']
	move_target_template = ['?', '?', 'in', '?', 'move', 'to', target_name, 'succeeded', '?']
	convoy_target_template = ['?', '?', 'in', '?', 'convoy', 'move', 'to', target_name, 'succeeded', '?']
	l_target_templates = [move_target_template, convoy_target_template]

	b_success, first_stage_success, first_stage_order, first_stage_match_pattern = \
		sel_orders(glv_dict, cont_stats_mgr, l_target_templates, unit_list, success_orders_freq,
				   num_rules, l_rules, l_lens, l_scvos, l_gens_recs,
				   prev_stage_score=None, l_unit_avail = l_unit_avail,
				   country_orders_list=country_orders_list, first_stage_order=None,
				   prev_stage_match_pattern=None, b_my_move=True, b_offensive=True)

	if not b_success:
		return

	b_success, block_stage_success, _, block_stage_match_pattern = \
		sel_orders(glv_dict, cont_stats_mgr, l_target_templates, block_unit_list, success_orders_freq,
				   num_rules, l_rules, l_lens, l_scvos, l_gens_recs,
				   prev_stage_score=first_stage_success, l_unit_avail=[],
				   country_orders_list=[], first_stage_order=first_stage_order,
				   prev_stage_match_pattern=first_stage_match_pattern, b_my_move=False, b_offensive=True)

	if not b_success or block_stage_success >= first_stage_success:
		return

	b_success, rejoiner_stage_success, _, _ = \
		sel_orders(glv_dict, cont_stats_mgr, l_target_templates, unit_list, success_orders_freq,
				   num_rules, l_rules, l_lens, l_scvos, l_gens_recs,
				   prev_stage_score=block_stage_success, l_unit_avail=l_unit_avail,
				   country_orders_list=country_orders_list, first_stage_order=first_stage_order,
				   prev_stage_match_pattern=block_stage_match_pattern, b_my_move=True, b_offensive=True)


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
		s_poss_targets = set()
		unit_list = unit_owns_tbl.get(scountry, None)
		if unit_list == None:
			continue

		b_offensive = True
		for unit_data in unit_list:
			move_order_template = [unit_data[0], 'in', unit_data[1], 'move', 'to', '?']
			convoy_order_template = [unit_data[0], 'in', unit_data[1], 'convoy', 'move', 'to', '?']
			l_pos_orders = wd_imagine.get_moves([move_order_template, convoy_order_template], success_orders_freq)
			for poss_order in l_pos_orders:
				dest_name = poss_order[5]
				prev_owner = terr_owner_dict.get(dest_name, 'neutral')
				if dest_name in supply_set and scountry != prev_owner:
					s_poss_targets.add(dest_name)

		l_target_data = []
		for a_target in s_poss_targets:
			l_target_data.append([a_target, b_offensive])


		b_offensive = False
		s_poss_targets = set()
		block_unit_list = []
		for i_opp_country in range(1, len(country_names_tbl)):
			if i_opp_country == icountry:
				continue
			sopp = country_names_tbl[i_opp_country]
			opp_unit_list = unit_owns_tbl.get(sopp, None)
			if opp_unit_list == None:
				continue
			block_unit_list += opp_unit_list

		random.shuffle(block_unit_list)

		for unit_data in block_unit_list:
			order_template = [unit_data[0], 'in', unit_data[1], 'move', 'to', '?']
			l_pos_orders = wd_imagine.get_moves([order_template], success_orders_freq)
			for poss_order in l_pos_orders:
				dest_name = poss_order[5]
				prev_owner = terr_owner_dict.get(dest_name, 'neutral')
				if dest_name in supply_set and prev_owner == scountry:
					s_poss_targets.add(dest_name)


		for a_target in s_poss_targets:
			l_target_data.append([a_target, b_offensive])
		# s_poss_targets.add([dest_name, b_offensive, random.random() * wdconfig.c_classic_AI_defensive_bias])

		l_target_data = [target_data + [(1.0 if target_data[1]
										 else wdconfig.c_classic_AI_defensive_bias) * random.random()]
						 for target_data in l_target_data]
		l_target_data = sorted(l_target_data, key=lambda x: x[2], reverse=True)

		random.shuffle(unit_list)
		del l_pos_orders, order_template, poss_order, dest_name, prev_owner
		l_unit_avail = [True for udata in unit_list]
		country_orders_list = []
		for target_data in l_target_data:
			target_name, b_offensive, _ = target_data
			if b_offensive:
				create_offensive(glv_dict, cont_stats_mgr, unit_list, block_unit_list,
								 target_name, country_orders_list, success_orders_freq,
								 num_rules, l_rules, l_lens, l_scvos, l_gens_recs, l_unit_avail)
			else:
				create_defensive(glv_dict, cont_stats_mgr, unit_list, block_unit_list,
								 target_name, country_orders_list, success_orders_freq,
								 num_rules, l_rules, l_lens, l_scvos, l_gens_recs, l_unit_avail)
		# end loop over target names
		for iunit, unit_data in enumerate(unit_list):
			if not l_unit_avail[iunit]:
				continue

			order_template = [unit_data[0], 'in', unit_data[1], 'move', 'to', '?']
			l_pos_orders = wd_imagine.get_moves([order_template], success_orders_freq)

			country_orders_list.append(random.choice(l_pos_orders))

		if country_orders_list != []:
			orders_list += country_orders_list
			icountry_list.append(icountry)


	orders_db = els.convert_list_to_phrases(orders_list)
	success_list = [True for o in orders_list]
	return orders_list, orders_db, success_list, icountry_list


