from __future__ import print_function
import random
import copy

import wdconfig
import wdlearn
import els
import learn
import rules
import makerecs as mr
import forbidden
import wd_imagine

def extra_orders(	glv_dict, cont_stats_mgr, num_rules, l_rules, l_lens, l_scvos, first_stage_order,
					i_colist_unit, colist_order, l_unit_avail, country_orders_list, prev_stage_match_pattern):
	l_unit_avail[i_colist_unit] = False
	country_orders_list.append(colist_order)
	match_pattern = copy.deepcopy(prev_stage_match_pattern)

	colist_order_phrase = els.convert_list_to_phrases([colist_order])
	order_rec, _ = mr.make_rec_from_phrase_list([first_stage_order, colist_order_phrase[0]])

	order_scvo = mr.gen_cvo_str(order_rec)
	order_len = mr.get_olen(order_scvo)
	# b_one_success = False
	for irule, rule in enumerate(l_rules):
		if order_len != l_lens[irule] or order_scvo != l_scvos[irule]:
			continue
		if not mr.does_match_rule(glv_dict, l_rules[irule], order_rec):
			continue
		match_pattern[irule] = True
		# b_one_success = True

	# if b_one_success:
	success = cont_stats_mgr.predict_success_rate(match_pattern)

	return match_pattern, success


def sel_orders(	glv_dict, cont_stats_mgr, l_target_templates, l_template_bonuses, unit_list, success_orders_data,
				num_rules, l_rules, l_lens, l_scvos, l_gens_recs, prev_stage_score, l_unit_avail, country_orders_list,
				first_stage_order, prev_stage_match_pattern, b_my_move, b_offensive):
	# success_orders_freq, oid_dict, success_unit_dict = success_orders_data
	l_successes = []
	for iunit, unit_data in enumerate(unit_list):
		if len(l_successes) > wdconfig.c_classic_AI_max_successes:
			break
		if b_my_move and not l_unit_avail[iunit]:
			continue
		order_template = [unit_data[0], 'in', unit_data[1]]
		l_poss_orders = wd_imagine.get_moves(unit_data, [order_template], success_orders_data, max_len=len(order_template) - 1)
		l_poss_order_phrases = els.convert_list_to_phrases(l_poss_orders)
		for iorder, poss_order in enumerate(l_poss_order_phrases):
			b_colist_extra = False
			b_req_failed = False
			l_req_colist, l_b_req, l_b_rev_req = \
				wd_imagine.get_colist_moves(l_poss_orders[iorder], success_orders_data,
											colist_req_thresh=0.3, colist_strong_thresh=0.2)
			for i_b_req, b_req in enumerate(l_b_req):
				if not b_req:
					continue
				req_order = list(l_req_colist[i_b_req])
				if b_my_move:
					if req_order in country_orders_list:
						# The following BREAKS the requirement that ALL the reqs must be present
						# This is done because we don't have a good implementation for support which should say
						# whether you are supporting an army or a fleet
						# Also, for support hold, only one of the options is enough
						# Once support syntax is fixed, we can have an AND thresh (all are required) and a lower
						# OR thresh
						b_req_failed = False
						break # should be continue
					b_req_failed = True
					if not l_b_rev_req[i_b_req]:
						break
				else:
					b_req_failed = True
				for i_req_unit, req_unit in enumerate(unit_list):
					if l_unit_avail != [] and not l_unit_avail[i_req_unit]:
						continue
					templ = [req_unit[0], 'in', req_unit[1]]
					if templ == req_order[0:3]:
						l_poss_req_orders = wd_imagine.get_moves(req_unit, [req_order], success_orders_data)
						if l_poss_req_orders != []:
							b_req_failed = False
							if b_my_move:
								b_colist_extra, i_extra_colist_unit, extra_colist_order = True, i_req_unit, req_order

						break # out of this inner loop only
				# The following should not really be commented out. See comment a few lines up
				if b_req_failed:
					break

			if b_req_failed:
				continue

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
				for itarget, target_template in enumerate(l_target_templates):
					if els.match_rec_to_templ(result_rec, target_template):
						b_found = True
						bonus = l_template_bonuses[itarget] # might change in future iterations of the loop, but difficult to believe
						break
				if not b_found:
					continue
				b_one_success = True
				match_pattern[irule] = True
			if b_one_success:
				success = cont_stats_mgr.predict_success_rate(match_pattern) + bonus

				if b_colist_extra:
					l_successes.append([success, poss_order, iunit, match_pattern, l_poss_orders[iorder],
										b_colist_extra, i_extra_colist_unit, extra_colist_order])
				else:
					l_successes.append([success, poss_order, iunit, match_pattern, l_poss_orders[iorder],
										b_colist_extra, None, None])
	# end of loop over units who can reach the country
	if l_successes == []:
		return False, None, None, None, None, False, None, None

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
	this_stage_match_pattern, this_stage_order_words, \
	b_colist_extra, i_extra_colist_unit, extra_colist_order = \
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
		if b_take_the_move and random.random() < (diff * 3):
			l_unit_avail[this_stage_iunit] = False
			country_orders_list.append(this_stage_order_words)
			# return True, this_stage_success, this_stage_order, this_stage_match_pattern, this_stage_order_words
		else:
			return False, None, None, None, None, False, None, None

	return True, this_stage_success, this_stage_order, this_stage_match_pattern, this_stage_order_words, \
		   b_colist_extra, i_extra_colist_unit, extra_colist_order


def create_defensive(	glv_dict, cont_stats_mgr, unit_list, block_unit_list,
						target_name, country_orders_list, success_orders_data,
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
	l_template_bonuses = [0.0, 1.0]

	b_success, first_stage_success, first_stage_order, first_stage_match_pattern, first_stage_words, _, _, _ = \
		sel_orders(glv_dict, cont_stats_mgr, l_target_templates, l_template_bonuses, block_unit_list, success_orders_data,
				   num_rules, l_rules, l_lens, l_scvos, l_gens_recs,
				   prev_stage_score=None, l_unit_avail=[],
				   country_orders_list=[], first_stage_order=None,
				   prev_stage_match_pattern=None, b_my_move=False, b_offensive=False)

	if not b_success:
		return 0.0

	print('Worrying about', ' '.join(first_stage_words), 'est. success:', first_stage_success)

	b_success, block_stage_success, _, block_stage_match_pattern, block_stage_words, \
	b_colist_extra, i_extra_colist_unit, extra_colist_order = \
		sel_orders(glv_dict, cont_stats_mgr, l_target_templates, l_template_bonuses, unit_list, success_orders_data,
				   num_rules, l_rules, l_lens, l_scvos, l_gens_recs,
				   prev_stage_score=first_stage_success, l_unit_avail=l_unit_avail,
				   country_orders_list=country_orders_list, first_stage_order=first_stage_order,
				   prev_stage_match_pattern=first_stage_match_pattern, b_my_move=True, b_offensive=False)

	if not b_success or block_stage_success >= first_stage_success:
		return -first_stage_success

	print('Trying to block with', ' '.join(block_stage_words), 'est. success:', block_stage_success)

	if b_colist_extra:
		block_stage_match_pattern, block_stage_success = \
			extra_orders(	glv_dict, cont_stats_mgr, num_rules, l_rules, l_lens, l_scvos, first_stage_order,
							i_extra_colist_unit, extra_colist_order, l_unit_avail,
							country_orders_list, block_stage_match_pattern)
		print('Adding a required colist order:', ' '.join(extra_colist_order), 'est. success:', block_stage_success)

	b_success, r1_stage_success, _, r1_stage_match_pattern, r1_stage_words, _, _, _ = \
		sel_orders(glv_dict, cont_stats_mgr, l_target_templates, l_template_bonuses, block_unit_list, success_orders_data,
				   num_rules, l_rules, l_lens, l_scvos, l_gens_recs,
				   prev_stage_score=block_stage_success, l_unit_avail=[],
				   country_orders_list=[], first_stage_order=first_stage_order,
				   prev_stage_match_pattern=block_stage_match_pattern, b_my_move=False, b_offensive=False)

	if not b_success or r1_stage_success <= block_stage_success:
		return -block_stage_success

	print('Worrying will respond with', ' '.join(r1_stage_words), 'est. success:', r1_stage_success)

	b_success, r2_stage_success, _, r2_stage_match_pattern, r2_stage_words, \
	b_colist_extra, i_extra_colist_unit, extra_colist_order = \
		sel_orders(glv_dict, cont_stats_mgr, l_target_templates, l_template_bonuses, unit_list, success_orders_data,
				   num_rules, l_rules, l_lens, l_scvos, l_gens_recs,
				   prev_stage_score=r1_stage_success, l_unit_avail=l_unit_avail,
				   country_orders_list=country_orders_list, first_stage_order=first_stage_order,
				   prev_stage_match_pattern=r1_stage_match_pattern, b_my_move=True, b_offensive=False)

	if not b_success:
		return -r1_stage_success

	print('Second rejoiner', ' '.join(r2_stage_words), 'est. success:', r2_stage_success)

	if b_colist_extra:
		_, r2_stage_success = \
			extra_orders(	glv_dict, cont_stats_mgr, num_rules, l_rules, l_lens, l_scvos, first_stage_order,
							i_extra_colist_unit, extra_colist_order, l_unit_avail,
							country_orders_list, r2_stage_match_pattern)
		print('Adding a required colist order:', ' '.join(extra_colist_order), 'est. success:', r2_stage_success)

	return r2_stage_success



def create_offensive(	glv_dict, cont_stats_mgr, unit_list, block_unit_list,
						target_name, country_orders_list, success_orders_data,
						 num_rules, l_rules, l_lens, l_scvos, l_gens_recs, l_unit_avail):
	# target_template = ['?', '?', 'in', '?', 'move', 'to', target_name, 'succeeded', '?']
	move_target_template = ['?', '?', 'in', '?', 'move', 'to', target_name, 'succeeded', '?']
	convoy_target_template = ['?', '?', 'in', '?', 'convoy', 'move', 'to', target_name, 'succeeded', '?']
	l_target_templates = [move_target_template, convoy_target_template]
	l_template_bonuses = [0.0, 1.0]

	b_success, first_stage_success, first_stage_order, first_stage_match_pattern, first_stage_words, \
	b_colist_extra, i_extra_colist_unit, extra_colist_order = \
		sel_orders(glv_dict, cont_stats_mgr, l_target_templates, l_template_bonuses, unit_list, success_orders_data,
				   num_rules, l_rules, l_lens, l_scvos, l_gens_recs,
				   prev_stage_score=None, l_unit_avail = l_unit_avail,
				   country_orders_list=country_orders_list, first_stage_order=None,
				   prev_stage_match_pattern=None, b_my_move=True, b_offensive=True)

	if not b_success:
		return 0.0

	print('Proposing offensive', ' '.join(first_stage_words), 'est. success:', first_stage_success)

	if b_colist_extra:
		first_stage_match_pattern, first_stage_success = \
			extra_orders(	glv_dict, cont_stats_mgr, num_rules, l_rules, l_lens, l_scvos, first_stage_order,
							i_extra_colist_unit, extra_colist_order, l_unit_avail,
							country_orders_list, first_stage_match_pattern)
		print('Adding a required colist order:', ' '.join(extra_colist_order), 'est. success:', first_stage_success)

	b_success, block_stage_success, _, block_stage_match_pattern, block_stage_words, _, _, _ = \
		sel_orders(glv_dict, cont_stats_mgr, l_target_templates, l_template_bonuses, block_unit_list, success_orders_data,
				   num_rules, l_rules, l_lens, l_scvos, l_gens_recs,
				   prev_stage_score=first_stage_success, l_unit_avail=[],
				   country_orders_list=[], first_stage_order=first_stage_order,
				   prev_stage_match_pattern=first_stage_match_pattern, b_my_move=False, b_offensive=True)

	if not b_success or block_stage_success >= first_stage_success:
		return first_stage_success

	print('Might get blocked by', ' '.join(block_stage_words), 'est. success:', block_stage_success)

	b_success, rejoiner_stage_success, _, rejoiner_stage_match_pattern, rejoiner_stage_words, \
	b_colist_extra, i_extra_colist_unit, extra_colist_order = \
		sel_orders(glv_dict, cont_stats_mgr, l_target_templates, l_template_bonuses, unit_list, success_orders_data,
				   num_rules, l_rules, l_lens, l_scvos, l_gens_recs,
				   prev_stage_score=block_stage_success, l_unit_avail=l_unit_avail,
				   country_orders_list=country_orders_list, first_stage_order=first_stage_order,
				   prev_stage_match_pattern=block_stage_match_pattern, b_my_move=True, b_offensive=True)

	if not b_success:
		return block_stage_success

	print('My rejoiner', ' '.join(rejoiner_stage_words), 'est. success:', rejoiner_stage_success)

	if b_colist_extra:
		_, rejoiner_stage_success = \
			extra_orders(	glv_dict, cont_stats_mgr, num_rules, l_rules, l_lens, l_scvos, first_stage_order,
							i_extra_colist_unit, extra_colist_order, l_unit_avail,
							country_orders_list, rejoiner_stage_match_pattern)
		print('Adding a required colist order:', ' '.join(extra_colist_order), 'est. success:', rejoiner_stage_success)

	return rejoiner_stage_success


def create_move_orders2(init_db, army_can_pass_tbl, fleet_can_pass_tbl, status_db, db_cont_mgr,
						country_names_tbl, l_humaan_countries, unit_owns_tbl,
						all_the_dicts, terr_owns_tbl, supply_tbl,
						b_waiting_for_AI, game_store,
						num_montes, preferred_nation, b_predict_success):
	glv_dict, def_article_dict, cascade_dict = all_the_dicts
	cascade_els = [el for el in cascade_dict.keys() if cascade_dict[el]]

	if b_predict_success:
		icc_list = db_cont_mgr.get_conts_above(wdconfig.c_use_rule_thresh)
	full_db = els.make_rec_list(init_db + status_db)

	l_f_rules = db_cont_mgr.get_forbidden_rules()

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

	l_country_options = [[] for _ in country_names_tbl]
	# for ioption_run in range(wdconfig.c_classic_AI_num_option_runs):
	num_option_iters = wdconfig.c_classic_AI_num_option_runs
	if len(l_humaan_countries) > 0:
		if not b_waiting_for_AI or game_store == []:
			num_option_iters = 1
		else:
			num_option_iters = 0

	if num_option_iters > 0:
		l_all_units = []
		for icountry in range(1, len(country_names_tbl)):
			scountry = country_names_tbl[icountry]
			unit_list = unit_owns_tbl.get(scountry, None)
			if unit_list == None:
				continue
			for unit_data in unit_list:
				l_all_units += [unit_data]

		success_orders_freq, success_order_id_dict, success_unit_dict, max_id  = \
			wdlearn.load_order_freq(l_all_units,
									l_f_rules, full_db, cascade_els, glv_dict, wdconfig.orders_success_fnt)
		success_orders_data = [success_orders_freq, success_order_id_dict, success_unit_dict]

	for ioption_run in range(num_option_iters):
		if game_store == []:
			game_store[:] = [[] for _ in country_names_tbl]

		for icountry in range(1, len(country_names_tbl)):
			if icountry in l_humaan_countries:
				continue
			b_has_success_score = False
			success_score = 0.0
			scountry = country_names_tbl[icountry]
			print('Thinking for ', scountry)
			s_poss_targets = set()
			s_poss_staging = set()
			unit_list = unit_owns_tbl.get(scountry, None)
			if unit_list == None:
				continue

			b_offensive = True

			for unit_data in unit_list:
				move_order_template = [unit_data[0], 'in', unit_data[1], 'move', 'to', '?']
				convoy_order_template = [unit_data[0], 'in', unit_data[1], 'convoy', 'move', 'to', '?']
				l_pos_orders = wd_imagine.get_moves(unit_data, [move_order_template, convoy_order_template], success_orders_data)
				for poss_order in l_pos_orders:
					# if forbidden.test_move_forbidden(poss_order, l_f_rules, full_db, cascade_els, glv_dict):
					# 	continue
					dest_name = poss_order[5]
					prev_owner = terr_owner_dict.get(dest_name, 'neutral')
					if dest_name in supply_set and scountry != prev_owner:
						s_poss_targets.add(dest_name)
					else:
						s_poss_staging.add(dest_name)

			# s_poss_staging = [stage for stage in s_poss_staging if stage not in s_poss_targets]

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
				l_pos_orders = wd_imagine.get_moves(unit_data, [order_template], success_orders_data)
				for poss_order in l_pos_orders:
					# if forbidden.test_move_forbidden(poss_order, l_f_rules, full_db, cascade_els, glv_dict):
					# 	continue
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
				b_has_success_score = True
				if b_offensive:
					success_score += create_offensive(	glv_dict, cont_stats_mgr, unit_list, block_unit_list,
														target_name, country_orders_list, success_orders_data,
														num_rules, l_rules, l_lens, l_scvos, l_gens_recs, l_unit_avail)
				else:
					success_score += create_defensive(	glv_dict, cont_stats_mgr, unit_list, block_unit_list,
														target_name, country_orders_list, success_orders_data,
														num_rules, l_rules, l_lens, l_scvos, l_gens_recs, l_unit_avail)
			# end loop over target names and data in l_target_data
			if any(l_unit_avail):
				print('Staging moves for:', scountry)
			for stage_dest in s_poss_staging:
				if not any(l_unit_avail):
					break
				b_has_success_score = True
				success_score += wdconfig.c_classic_AI_stage_score_factor *\
								 create_offensive(	glv_dict, cont_stats_mgr, unit_list, block_unit_list,
													stage_dest, country_orders_list, success_orders_data,
													num_rules, l_rules, l_lens, l_scvos, l_gens_recs, l_unit_avail)

			# remaining units have been assigned no purpose, so they make a random move
			# for iunit, unit_data in enumerate(unit_list):
			# 	if not l_unit_avail[iunit]:
			# 		continue
			#
			# 	order_template = [unit_data[0], 'in', unit_data[1], 'move', 'to', '?']
			# 	l_pos_orders = wd_imagine.get_moves([order_template], success_orders_data)
			#
			# 	rnd_order = random.choice(l_pos_orders)
			# 	print('random move:', ' '.join(rnd_order))
			# 	country_orders_list.append(rnd_order)

			if country_orders_list != []:
				num_options_stored = len(game_store[icountry])
				option_for_store = [country_orders_list, success_score if b_has_success_score else -1000.0]
				if num_options_stored == wdconfig.c_num_game_store_options:
					if option_for_store[1] > game_store[icountry][wdconfig.c_num_game_store_options-1][1]:
						game_store[icountry][wdconfig.c_num_game_store_options - 1] = option_for_store
						game_store[icountry] = sorted(game_store[icountry], key=lambda x: x[1], reverse=True)
				else:
					game_store[icountry].append(option_for_store)
					game_store[icountry] = sorted(game_store[icountry], key=lambda x: x[1], reverse=True)

				# l_country_options[icountry].append()
				# orders_list += country_orders_list
				# icountry_list.append(icountry)

			print('Total success score for ', scountry, 'for this option run is', success_score)
		# end loop over each country for that option run
	# end if not b_waiting_for_AI or game_store = []

	if b_waiting_for_AI:
		for icountry in range(1, len(country_names_tbl)):
			if len(game_store[icountry]) == 0 or game_store[icountry][0] == []:
				continue
			country_orders_list, success_score = game_store[icountry][0]
			orders_list += country_orders_list
			icountry_list.append(icountry)
		game_store[:] = [] # delete the contents t=not the ref

	orders_db = els.convert_list_to_phrases(orders_list)
	success_list = [True for o in orders_list]
	return orders_list, orders_db, success_list, icountry_list


