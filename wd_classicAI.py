from __future__ import print_function
import random
import copy
from operator import itemgetter
import sys

import wdconfig
import wdlearn
import els
import learn
import rules
import makerecs as mr
import forbidden
import wd_imagine

def extra_orders(	glv_dict, cont_stats_mgr, wd_game_state, first_stage_order,
					i_colist_unit, colist_order, l_unit_avail, country_orders_list, prev_stage_match_pattern):
	num_rules, l_rules, l_scvos, l_lens, _ = wd_game_state.get_game_rules()
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

# First shot at ally support. b_my_move will be False, but we score according to b_offensive
# The move must have a colist requirement and the requirement must be be in the country orders list
def sel_orders(	glv_dict, cont_stats_mgr, l_target_templates, l_template_bonuses, unit_list, wd_game_state,
				target_name, prev_stage_score, l_unit_avail, country_orders_list,
				first_stage_order, prev_stage_match_pattern, b_my_move, b_offensive, b_ally_support=False):
	success_orders_data = wd_game_state.get_success_orders_data()
	num_rules, l_rules, l_scvos, l_lens, l_gens_recs = wd_game_state.get_game_rules()
	distance_calc = wd_game_state.get_distance_params()
	target_neighbors = distance_calc.get_neighbors(target_name, wdconfig.c_classic_AI_action_distance_to_target)
	# success_orders_freq, oid_dict, success_unit_dict = success_orders_data
	l_successes = []
	for iunit, unit_data in enumerate(unit_list):
		if len(l_successes) > wdconfig.c_classic_AI_max_successes:
			break
		if b_my_move and not l_unit_avail[iunit]:
			continue
		if distance_calc.get_distance(unit_data[1], target_name) > wdconfig.c_classic_AI_unit_distance_to_target:
			continue
		order_template = [unit_data[0], 'in', unit_data[1]]
		l_poss_orders = success_orders_data.get_moves(	unit_data, [order_template], target_neighbors,
														max_len=len(order_template) - 1)
		l_poss_order_phrases = els.convert_list_to_phrases(l_poss_orders)
		for iorder, poss_order in enumerate(l_poss_order_phrases):
			b_colist_extra = False
			b_req_failed = False
			l_req_colist, l_b_req, l_b_rev_req = \
				success_orders_data.get_colist_moves(	l_poss_orders[iorder],
														colist_req_thresh=0.3, colist_strong_thresh=0.2)
			for i_b_req, b_req in enumerate(l_b_req):
				if not b_req:
					continue
				req_order = list(l_req_colist[i_b_req])
				if b_my_move or b_ally_support:
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
				if b_ally_support:
					break
				for i_req_unit, req_unit in enumerate(unit_list):
					if l_unit_avail != [] and not l_unit_avail[i_req_unit]:
						continue
					templ = [req_unit[0], 'in', req_unit[1]]
					if templ == req_order[0:3]:
						l_poss_req_orders = success_orders_data.get_moves(req_unit, [req_order])
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
		sorted(l_successes, key=lambda x: x[0], reverse=((b_my_move or b_ally_support) == b_offensive))[0]

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
						target_name, country_orders_list, wd_game_state, l_unit_avail):
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
		sel_orders(glv_dict, cont_stats_mgr, l_target_templates, l_template_bonuses, block_unit_list,
				   wd_game_state, target_name,
				   prev_stage_score=None, l_unit_avail=[],
				   country_orders_list=[], first_stage_order=None,
				   prev_stage_match_pattern=None, b_my_move=False, b_offensive=False)

	if not b_success:
		return 0.0

	print('Worrying about', ' '.join(first_stage_words), 'est. success:', first_stage_success)

	b_success, block_stage_success, _, block_stage_match_pattern, block_stage_words, \
	b_colist_extra, i_extra_colist_unit, extra_colist_order = \
		sel_orders(glv_dict, cont_stats_mgr, l_target_templates, l_template_bonuses, unit_list,
				   wd_game_state, target_name,
				   prev_stage_score=first_stage_success, l_unit_avail=l_unit_avail,
				   country_orders_list=country_orders_list, first_stage_order=first_stage_order,
				   prev_stage_match_pattern=first_stage_match_pattern, b_my_move=True, b_offensive=False)

	if not b_success or block_stage_success >= first_stage_success:
		return -first_stage_success

	print('Trying to block with', ' '.join(block_stage_words), 'est. success:', block_stage_success)

	if b_colist_extra:
		block_stage_match_pattern, block_stage_success = \
			extra_orders(	glv_dict, cont_stats_mgr, wd_game_state, first_stage_order,
							i_extra_colist_unit, extra_colist_order, l_unit_avail,
							country_orders_list, block_stage_match_pattern)
		print('Adding a required colist order:', ' '.join(extra_colist_order), 'est. success:', block_stage_success)

	b_success, r1_stage_success, _, r1_stage_match_pattern, r1_stage_words, _, _, _ = \
		sel_orders(glv_dict, cont_stats_mgr, l_target_templates, l_template_bonuses, block_unit_list,
				   wd_game_state, target_name,
				   prev_stage_score=block_stage_success, l_unit_avail=[],
				   country_orders_list=[], first_stage_order=first_stage_order,
				   prev_stage_match_pattern=block_stage_match_pattern, b_my_move=False, b_offensive=False)

	if not b_success or r1_stage_success <= block_stage_success:
		return -block_stage_success

	print('Worrying will respond with', ' '.join(r1_stage_words), 'est. success:', r1_stage_success)

	b_success, r2_stage_success, _, r2_stage_match_pattern, r2_stage_words, \
	b_colist_extra, i_extra_colist_unit, extra_colist_order = \
		sel_orders(glv_dict, cont_stats_mgr, l_target_templates, l_template_bonuses, unit_list,
				   wd_game_state, target_name,
				   prev_stage_score=r1_stage_success, l_unit_avail=l_unit_avail,
				   country_orders_list=country_orders_list, first_stage_order=first_stage_order,
				   prev_stage_match_pattern=r1_stage_match_pattern, b_my_move=True, b_offensive=False)

	if not b_success:
		return -r1_stage_success

	print('Second rejoiner', ' '.join(r2_stage_words), 'est. success:', r2_stage_success)

	if b_colist_extra:
		_, r2_stage_success = \
			extra_orders(	glv_dict, cont_stats_mgr, wd_game_state, first_stage_order,
							i_extra_colist_unit, extra_colist_order, l_unit_avail,
							country_orders_list, r2_stage_match_pattern)
		print('Adding a required colist order:', ' '.join(extra_colist_order), 'est. success:', r2_stage_success)

	return r2_stage_success



def create_offensive(glv_dict, cont_stats_mgr, unit_list, l_units_of_others,
					 target_name, country_orders_list, wd_game_state, l_unit_avail,
					 ally_order):
	# target_template = ['?', '?', 'in', '?', 'move', 'to', target_name, 'succeeded', '?']
	move_target_template = ['?', '?', 'in', '?', 'move', 'to', target_name, 'succeeded', '?']
	convoy_target_template = ['?', '?', 'in', '?', 'convoy', 'move', 'to', target_name, 'succeeded', '?']
	l_target_templates = [move_target_template, convoy_target_template]
	l_template_bonuses = [0.0, 1.0]

	rollback_country_orders_list = copy.deepcopy(country_orders_list)
	rollback_l_units_avail = copy.deepcopy(l_unit_avail)


	b_success, first_stage_success, first_stage_order, first_stage_match_pattern, first_stage_words, \
	b_colist_extra, i_extra_colist_unit, extra_colist_order = \
		sel_orders(glv_dict, cont_stats_mgr, l_target_templates, l_template_bonuses, unit_list,
				   wd_game_state, target_name,
				   prev_stage_score=None, l_unit_avail = l_unit_avail,
				   country_orders_list=country_orders_list, first_stage_order=None,
				   prev_stage_match_pattern=None, b_my_move=True, b_offensive=True)

	if not b_success:
		return 0.0

	print('Proposing offensive', ' '.join(first_stage_words), 'est. success:', first_stage_success)

	if b_colist_extra:
		first_stage_match_pattern, first_stage_success = \
			extra_orders(	glv_dict, cont_stats_mgr, wd_game_state, first_stage_order,
							i_extra_colist_unit, extra_colist_order, l_unit_avail,
							country_orders_list, first_stage_match_pattern)
		print('Adding a required colist order:', ' '.join(extra_colist_order), 'est. success:', first_stage_success)

	b_success, block_stage_success, _, block_stage_match_pattern, block_stage_words, _, _, _ = \
		sel_orders(glv_dict, cont_stats_mgr, l_target_templates, l_template_bonuses, l_units_of_others,
				   wd_game_state, target_name,
				   prev_stage_score=first_stage_success, l_unit_avail=[],
				   country_orders_list=[], first_stage_order=first_stage_order,
				   prev_stage_match_pattern=first_stage_match_pattern, b_my_move=False, b_offensive=True)

	if not b_success or block_stage_success >= first_stage_success:
		return first_stage_success

	print('Might get blocked by', ' '.join(block_stage_words), 'est. success:', block_stage_success)

	b_success, rejoiner_stage_success, _, rejoiner_stage_match_pattern, rejoiner_stage_words, \
	b_colist_extra, i_extra_colist_unit, extra_colist_order = \
		sel_orders(glv_dict, cont_stats_mgr, l_target_templates, l_template_bonuses, unit_list,
				   wd_game_state, target_name,
				   prev_stage_score=block_stage_success, l_unit_avail=l_unit_avail,
				   country_orders_list=country_orders_list, first_stage_order=first_stage_order,
				   prev_stage_match_pattern=block_stage_match_pattern, b_my_move=True, b_offensive=True)

	if not b_success or rejoiner_stage_success < wdconfig.c_classic_AI_rejoiner_min:
		if ally_order == []:
			b_success, ally_stage_success, _, ally_stage_match_pattern, ally_stage_words, _, _, _ = \
				sel_orders(	glv_dict, cont_stats_mgr, l_target_templates, l_template_bonuses, l_units_of_others,
							wd_game_state, target_name,
							prev_stage_score=block_stage_success, l_unit_avail=[],
							country_orders_list=country_orders_list, first_stage_order=first_stage_order,
							prev_stage_match_pattern=block_stage_match_pattern,
							b_my_move=False, b_offensive=True, b_ally_support=True)
			if b_success and ally_stage_success > wdconfig.c_classic_AI_ally_success_min:
				ally_order[:] = ally_stage_words
				print(	'Possible ally move can help a hopeless move:', ' '.join(ally_stage_words),
						'est. success', ally_stage_success)
				return ally_stage_success

		if random.random() < wdconfig.c_classic_AI_abandon_prob:
			print('Offensive move blocked without rejoiner. Abandoning move')
			country_orders_list[:] = rollback_country_orders_list
			l_unit_avail[:] = rollback_l_units_avail

		return block_stage_success

	print('My rejoiner', ' '.join(rejoiner_stage_words), 'est. success:', rejoiner_stage_success)

	if b_colist_extra:
		_, rejoiner_stage_success = \
			extra_orders(	glv_dict, cont_stats_mgr, wd_game_state, first_stage_order,
							i_extra_colist_unit, extra_colist_order, l_unit_avail,
							country_orders_list, rejoiner_stage_match_pattern)
		print('Adding a required colist order:', ' '.join(extra_colist_order), 'est. success:', rejoiner_stage_success)

	return rejoiner_stage_success

def bring_closer_to_contested(	glv_dict, cont_stats_mgr, iunit, unit_data, contested_goal, distance_calc,
								l_rules, l_scvos, l_lens, l_unit_avail, num_rules,
								country_orders_list, success_orders_data):
	start_distance = distance_calc.get_distance(unit_data[1], contested_goal)
	if start_distance == 0:
		print('ClassicAI concern: An avail unit in the contested area wants to move out')
		return False
 	elif start_distance == 1:
		print('ClassicAI concern: An avail unit adjacent to contested area not supporting')
		# horder = [unit_data[0], 'in', unit_data[1], 'hold']
		# print(' '.join(horder))
		# country_orders_list.append(horder)
		# l_unit_avail[iunit] = False
		# return True
		return False

	def get_comb_success(match_pattern, order_phrase):
		order_rec, _ = mr.make_rec_from_phrase_list(order_phrase)
		order_scvo = mr.gen_cvo_str(order_rec)
		order_len = mr.get_olen(order_scvo)
		b_one_success = False
		for irule, rule in enumerate(l_rules):
			if order_len != l_lens[irule] or order_scvo != l_scvos[irule]:
				continue
			if not mr.does_match_rule(glv_dict, l_rules[irule], order_rec):
				continue
			b_one_success = True
			match_pattern[irule] = True
		if b_one_success:
			return True, cont_stats_mgr.predict_success_rate(match_pattern)

		return False, 0.0

	move_order_template = [unit_data[0], 'in', unit_data[1], 'move', 'to', '?']
	l_pos_orders = success_orders_data.get_moves(unit_data, [move_order_template])
	l_choices = [[distance_calc.get_distance(poss_order[5], contested_goal), poss_order] for poss_order in l_pos_orders]
	l_choices.sort(key=lambda x: x[0])
	for choice in l_choices:
		order_phrase = els.convert_list_to_phrases([choice[1]])
		match_pattern = [False for i in range(num_rules)]
		b_initial_success, initial_success = get_comb_success(match_pattern, order_phrase)
		if initial_success > 0.0:
			b_go_for_it = True
			for other_order in country_orders_list:
				comb_order_phrase = els.convert_list_to_phrases([choice[1], other_order])
				b_comb_success, comb_success = get_comb_success(match_pattern, comb_order_phrase)
				if b_comb_success and comb_success < initial_success:
					b_go_for_it = False
					break
			if b_go_for_it:
				print(' '.join(choice[1]))
				country_orders_list.append(choice[1])
				l_unit_avail[iunit] = False
				return True

	return False


class cl_country_data(object):
	def __init__(self, scountry, unit_list, s_contested, block_unit_list, l_target_data, s_poss_staging):
		self.__scountry = scountry
		self.__unit_list = unit_list
		self.__s_contested = s_contested
		self.__block_unit_list = block_unit_list
		self.__l_target_data = l_target_data
		self.__s_poss_staging = s_poss_staging

	def get_vars(self):
		return self.__unit_list, self.__s_contested, self.__block_unit_list, self.__l_target_data, self.__s_poss_staging

	def get_targets(self):
		return [target_data[0] for target_data in self.__l_target_data]

	def get_unit_list(self):
		return self.__unit_list

class cl_game_option(object):
	def __init__(self, scountry, l_orders, score, ally_order, donated, status_db, country_names_tbl):
		self.__l_orders = l_orders
		self.__score = score
		self.__ally_order = ally_order
		self.__scountry = scountry
		if ally_order != None and ally_order != [] and ally_order != ['None']:
			# ally_unit_type, ally_order_src = ally_order[0,2]
			ally_unit_type, ally_order_src = itemgetter(*[0,2])(ally_order)
			sally, b_valid = '', False
			for scountry2 in country_names_tbl:
				extract_phrase = els.convert_list_to_phrases([[scountry2, 'owns', ally_unit_type, 'in', ally_order_src]])
				if extract_phrase[0] in status_db:
					sally = scountry2
					break
			if sally != '':
				ally_phrase = els.convert_list_to_phrases([[scountry, 'allied', 'to', sally]])
				if ally_phrase[0] in status_db:
					b_valid = True
			if not b_valid:
				sally = ''
		else:
			sally, b_valid = '', False

		self.__b_needs_ally = b_valid
		self.__sally = sally
		self.__donated = donated
		self.__num_beaten = 0

	def get_key_data(self):
		return self.__donated, self.__scountry, self.__b_needs_ally, self.__score

	def get_score(self):
		return self.__score

	def set_num_beaten(self, num_beaten):
		self.__num_beaten = num_beaten

	def get_num_beaten(self):
		return self.__num_beaten

	def get_need_ally(self):
		return self.__b_needs_ally

	def get_ally_data(self):
		return self.__sally, self.__ally_order

	def get_orders(self):
		return self.__l_orders

class cl_game_option_state(object):
	def __init__(self):
		self.__d_stores = dict()
		self.__l_donated = None
		self.__d_country_data = dict()
		# self.__success_orders_data = []
		self.__l_options = []
		self.__l_num_proposed = []

	def init_donated(self, scountry, l_donated):
		for one_donated in [('None', 'None')] + l_donated:
			self.__d_stores[(one_donated, scountry, True)] = len(self.__l_options)
			self.__l_options.append([])
			self.__l_num_proposed.append(0)
			self.__d_stores[(one_donated, scountry, False)] = len(self.__l_options)
			self.__l_options.append([])
			self.__l_num_proposed.append(-1 if one_donated == ('None', 'None') else 0)

	def clear_donated(self):
		self.__d_stores.clear()
		self.__l_options = []
		self.__l_num_proposed = []
		self.__d_country_data = dict()

	def is_donated_initialized(self):
		return len(self.__d_stores) > 0

	def add_country_data(self, scountry, unit_list, s_contested, block_unit_list, l_target_data, s_poss_staging):
		self.__d_country_data[scountry] = cl_country_data(	scountry, unit_list, s_contested, block_unit_list,
															l_target_data, s_poss_staging)

	def has_country_data(self, scountry):
		return self.__d_country_data.get(scountry, None) != None

	def get_country_data(self, scountry):
		return self.__d_country_data[scountry].get_vars()

	def get_country_targets(self, scountry):
		country_data = self.__d_country_data.get(scountry, None)
		return country_data.get_targets() if country_data != None else []

	def propose_donated(self, scountry):
		min_num_proposed, min_option = sys.maxint, []
		for koption, vioption in self.__d_stores.iteritems():
			donated, scountry2, b_needs_ally = koption
			num_proposed = self.__l_num_proposed[vioption]
			if scountry2 != scountry:
				continue
			if num_proposed == -1:
				self.__l_num_proposed[vioption] += 1
				return donated, b_needs_ally
			# num_beaten = voption.get_num_beaten()
			if num_proposed < min_num_proposed:
				min_option, min_num_proposed = koption, num_proposed

		ioption = self.__d_stores[min_option]
		self.__l_num_proposed[ioption] += 1
		donated, _, b_needs_ally = min_option
		return donated, b_needs_ally

	def get_option(self, donated, scountry, b_needs_ally):
		ioption = self.__d_stores.get((donated, scountry, b_needs_ally), -1)
		if ioption == -1:
			print('Error! unrecognised game option!', donated, scountry, b_needs_ally )
			return [], -1, -1
		return self.__l_options[ioption], self.__l_num_proposed[ioption], ioption

	def add_option(self, option):
		donated, scountry, b_needs_ally, score = option.get_key_data()
		option2, _, ioption = self.get_option(donated, scountry, b_needs_ally)
		if ioption < 0:
			return
		if option2 == [] or option2.get_score() < score:
			if option2 == []:
				num_beaten = 0
			else:
				num_beaten = option2.get_num_beaten() + 1

			option.set_num_beaten(num_beaten)
			self.__l_options[ioption] = option
			# self.__d_stores[(donated, scountry, b_needs_ally)] = option

	def get_best_option(self, scountry, donated, b_no_ally):
		option_wo_ally, _, _ = self.get_option(donated, scountry, False)
		if b_no_ally:
			return option_wo_ally
		b_needs_ally = True
		option_with_ally, _, _ = self.get_option(donated, scountry, b_needs_ally)
		# option_wo_ally = self.__d_stores.get((donated, scountry, False), [])
		if option_wo_ally == []:
			return option_with_ally
		if option_with_ally == []:
			return option_wo_ally
		if option_with_ally.get_score() > option_wo_ally.get_score():
			return option_with_ally
		return option_wo_ally
		# return self.__d_stores.get((donated, scountry, b_needs_ally), [])

	def get_allies(self, scountry, status_db, country_names_tbl):
		l_allies= []
		for scountry2 in country_names_tbl:
			ally_rec = els.convert_list_to_phrases([[scountry, 'allied', 'to', scountry2]])
			if ally_rec[0] in status_db:
				l_allies.append(scountry2)
		return l_allies

	def get_unit_list(self, scountry):
		country_data = self.__d_country_data.get(scountry, None)
		return country_data.get_unit_list() if country_data != None else []

	def set_success_orders_data(self, success_orders_data):
		print('Error. set_success_orders_data removed')
		raise ValueError('set_success_orders_data removed')
		# self.__success_orders_data = success_orders_data

	def get_success_orders_data(self):
		print('Error. get_success_orders_data removed')
		raise ValueError('get_success_orders_data removed')
		# return self.__success_orders_data

def classic_AI(wd_game_state, b_predict_success):
	init_db, status_db, db_cont_mgr, country_names_tbl, _, unit_owns_tbl, \
	all_the_dicts, terr_owns_tbl, supply_tbl, b_waiting_for_AI, game_option_state  = \
		wd_game_state.get_at_classic_AI()
	glv_dict, def_article_dict, cascade_dict = all_the_dicts
	cascade_els = [el for el in cascade_dict.keys() if cascade_dict[el]]
	distance_calc = wd_game_state.get_distance_params()
	if game_option_state == None:
		game_option_state = cl_game_option_state()
		wd_game_state.set_game_state(game_option_state)
	success_orders_data = wd_game_state.get_success_orders_data()
	if success_orders_data == None:
		success_orders_data = wdlearn.cl_order_freq_data()
		wd_game_state.set_success_orders_data(success_orders_data)
	alliance_state = wd_game_state.get_alliance_state()
	human_by_icountry = alliance_state.get_human_by_icountry()

	if b_predict_success:
		icc_list = db_cont_mgr.get_conts_above(wdconfig.c_use_rule_thresh)
	full_db = els.make_rec_list(init_db + status_db)

	forbidden_state = db_cont_mgr.get_forbidden_rules()

	supply_set = set()
	for kcountry, vterr_list in supply_tbl.iteritems():
		for terr_stat in vterr_list:
			supply_set.add(terr_stat[1])

	country_dict = dict()
	for icountry, country_name in enumerate(country_names_tbl):
		country_dict[country_name] = icountry

	terr_owner_dict = dict()
	for kcountry, vterr_list in terr_owns_tbl.iteritems():
		for terr_data in vterr_list:
			# icountry = country_dict[kcountry]
			terr_owner_dict[terr_data[0]] = kcountry
			# if terr_data[0] in supply_set:
			# 	num_supplies_list[icountry] += 1

	cont_stats_mgr = db_cont_mgr.get_cont_stats_mgr()
	if wd_game_state.is_game_rules_initialized():
		num_rules, l_rules, l_scvos, l_lens, l_gens_recs = wd_game_state.get_game_rules()
	else:
		num_rules = len(cont_stats_mgr.get_cont_stats_list())
		l_rules = [cont_stat.get_cont().get_rule() for cont_stat in cont_stats_mgr.get_cont_stats_list()]
		l_scvos = [mr.gen_cvo_str(rule) for rule in l_rules]
		l_lens = [mr.get_olen(scvo) for scvo in l_scvos]
		l_gens_recs = [cont_stat.get_cont().get_gens_rec() for cont_stat in cont_stats_mgr.get_cont_stats_list()]
		wd_game_state.init_game_rules(num_rules, l_rules, l_scvos, l_lens, l_gens_recs)

	orders_list = []
	icountry_list = []

	l_country_options = [[] for _ in country_names_tbl]
	# for ioption_run in range(wdconfig.c_classic_AI_num_option_runs):
	num_option_iters = wdconfig.c_classic_AI_num_option_runs
	if any(human_by_icountry):
		if not b_waiting_for_AI or not game_option_state.is_donated_initialized():
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

		# success_orders_data = game_option_state.get_success_orders_data()
		if not success_orders_data.is_init_for_game():
			success_orders_data.init_for_game(forbidden_state,  wdconfig.orders_success_fnt, cascade_els, glv_dict)
		if not success_orders_data.is_init_for_move():
			success_orders_data.init_for_move(full_db)

		# if success_orders_data == []:
		# 	success_orders_freq, success_order_id_dict, success_unit_dict, max_id  = \
		# 		wdlearn.load_order_freq(l_all_units,
		# 								forbidden_state, full_db, cascade_els, glv_dict, wdconfig.orders_success_fnt)
		# 	success_orders_data = [success_orders_freq, success_order_id_dict, success_unit_dict]
		# 	game_option_state.set_success_orders_data(success_orders_data)

		if not game_option_state.is_donated_initialized():
			for icountry in range(1, len(country_names_tbl)):
				# if human_by_icountry[icountry]:
				# 	continue
				scountry = country_names_tbl[icountry]
				s_poss_targets = set()
				s_poss_staging = set()
				unit_list = unit_owns_tbl.get(scountry, None)
				if unit_list == None:
					continue

				b_offensive = True

				for unit_data in unit_list:
					move_order_template = [unit_data[0], 'in', unit_data[1], 'move', 'to', '?']
					convoy_order_template = [unit_data[0], 'in', unit_data[1], 'convoy', 'move', 'to', '?']
					l_pos_orders = success_orders_data.get_moves(unit_data, [move_order_template])
					for poss_order in l_pos_orders:
						# if forbidden.test_move_forbidden(poss_order, forbidden_state, full_db, cascade_els, glv_dict):
						# 	continue
						dest_name = poss_order[5]
						if dest_name == 'to':
							dest_name = poss_order[6] # convoy order
						prev_owner = terr_owner_dict.get(dest_name, 'neutral')
						if dest_name in supply_set and scountry != prev_owner:
							s_poss_targets.add(dest_name)
						else:
							s_poss_staging.add(dest_name)

						del poss_order, dest_name, prev_owner

					del unit_data, move_order_template, convoy_order_template, l_pos_orders

				# s_poss_staging = [stage for stage in s_poss_staging if stage not in s_poss_targets]

				s_contested = set(s_poss_targets)

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
					del sopp
					if opp_unit_list == None:
						continue
					block_unit_list += opp_unit_list
					del opp_unit_list, i_opp_country

				random.shuffle(block_unit_list)

				for unit_data in block_unit_list:
					order_template = [unit_data[0], 'in', unit_data[1], 'move', 'to', '?']
					l_pos_orders = success_orders_data.get_moves(unit_data, [order_template])
					for poss_order in l_pos_orders:
						# if forbidden.test_move_forbidden(poss_order, forbidden_state, full_db, cascade_els, glv_dict):
						# 	continue
						dest_name = poss_order[5]
						prev_owner = terr_owner_dict.get(dest_name, 'neutral')
						if dest_name in supply_set and prev_owner == scountry:
							s_poss_targets.add(dest_name)
						del poss_order, dest_name, prev_owner
					del unit_data, order_template, l_pos_orders

				s_contested = s_contested.union(s_poss_targets)

				for a_target in s_poss_targets:
					l_target_data.append([a_target, b_offensive])
				# s_poss_targets.add([dest_name, b_offensive, random.random() * wdconfig.c_classic_AI_defensive_bias])

				# l_target_data = [target_data[:1]
				# 				 + distance_calc.get_neighbors(target_data[0],
				# 											   wdconfig.c_classic_AI_action_distance_to_target)
				# 				 for target_data in l_target_data]

				# del l_pos_orders, , poss_order, dest_name, prev_owner

				game_option_state.add_country_data(	scountry, unit_list, s_contested, block_unit_list,
													l_target_data, s_poss_staging)
				del icountry, scountry, s_poss_targets, s_poss_staging, \
					unit_list, b_offensive, s_contested, l_target_data, block_unit_list

			for icountry in range(1, len(country_names_tbl)):
				# if human_by_icountry[icountry]:
				# 	continue
				scountry = country_names_tbl[icountry]
				l_allies = game_option_state.get_allies(scountry, status_db, country_names_tbl)
				unit_list = game_option_state.get_unit_list(scountry)
				s_donated = set()
				for sally in l_allies:
					their_targets = game_option_state.get_country_targets(sally)
					for starget in their_targets:
						for unit_data in unit_list:
							if distance_calc.get_distance(unit_data[1], starget) == 1:
								s_donated.add(tuple(unit_data))
							del unit_data
						del starget

					del sally, their_targets

				game_option_state.init_donated(scountry, list(s_donated))
				del icountry, scountry, l_allies, unit_list, s_donated


	for ioption_run in range(num_option_iters):
		# if game_option_state == []:
		# 	game_option_state[:] = [[] for _ in country_names_tbl]


		for icountry in range(1, len(country_names_tbl)):
			if human_by_icountry[icountry]:
				continue
			b_has_success_score = False
			success_score = 0.0
			scountry = country_names_tbl[icountry]
			if not game_option_state.has_country_data(scountry):
				continue
			unit_list, s_contested, block_unit_list, l_target_data, s_poss_staging = \
				game_option_state.get_country_data(scountry)

			l_target_data = [target_data[:2] for target_data in l_target_data]

			l_target_data = [target_data + [(1.0 if target_data[1]
											 else wdconfig.c_classic_AI_defensive_bias) * random.random()]
							 for target_data in l_target_data]
			l_target_data = sorted(l_target_data, key=lambda x: x[2], reverse=True)

			random.shuffle(unit_list)
			print('Thinking for ', scountry)
			donated, b_needs_ally = game_option_state.propose_donated(scountry)
			print('Calculating an option for', 'Need an' if b_needs_ally else 'Dont need an', 'ally. Donating', ' '.join(donated))
			l_unit_avail = [udata != list(donated) for udata in unit_list]
			country_orders_list = []
			ally_orders = [] if b_needs_ally else ['None']
			for target_data in l_target_data:
				target_name, b_offensive, _ = target_data
				b_has_success_score = True
				if b_offensive:
					success_score += create_offensive(	glv_dict, cont_stats_mgr, unit_list, block_unit_list,
														target_name, country_orders_list, wd_game_state,
														l_unit_avail, ally_orders)
				else:
					success_score += create_defensive(	glv_dict, cont_stats_mgr, unit_list, block_unit_list,
														target_name, country_orders_list, wd_game_state,
														l_unit_avail)
			# end loop over target names and data in l_target_data

			l_contested = list(s_contested)
			random.shuffle(l_contested)
			if any(l_unit_avail):
				print('Staging moves for:', scountry)
				# distance_calc = wd_game_state.get_distance_params()

			l_contested = l_contested * wdconfig.c_classic_AI_contested_repl
			random.shuffle(l_contested)

			for contested_goal in l_contested:
				if not any(l_unit_avail):
					break

				# avail_data = next(unit_data for iunit, unit_data in enumerate(unit_list) if l_unit_avail[iunit])
				for iunit, avail_data in enumerate(unit_list):
					if not l_unit_avail[iunit]:
						continue

					if bring_closer_to_contested(	glv_dict, cont_stats_mgr, iunit, avail_data,
													contested_goal, distance_calc,
													l_rules, l_scvos, l_lens, l_unit_avail, num_rules,
													country_orders_list, success_orders_data):
						break

			for stage_dest in s_poss_staging:
				if not any(l_unit_avail):
					break
				b_has_success_score = True
				success_score += wdconfig.c_classic_AI_stage_score_factor *\
								 create_offensive(	glv_dict, cont_stats_mgr, unit_list, block_unit_list,
													stage_dest, country_orders_list, wd_game_state,
													l_unit_avail, ally_orders)

			if any(l_unit_avail):
				print('Tried everything.', scountry, 'still has units available')
				for iunit, avail_data in enumerate(unit_list):
					if not l_unit_avail[iunit]:
						continue
					out_of_options_order = [avail_data[0], 'in', avail_data[1], 'hold']
					country_orders_list.append(out_of_options_order)
					print(' '.join(out_of_options_order))
					l_unit_avail[iunit] = False

			if country_orders_list != []:
				game_option_state.add_option(cl_game_option(scountry, country_orders_list,
															success_score if b_has_success_score else -1000.0,
															ally_orders, donated, status_db, country_names_tbl))
				# num_options_stored = len(game_option_state[icountry])
				# option_for_store = [country_orders_list, success_score if b_has_success_score else -1000.0, ally_orders]
				# if num_options_stored == wdconfig.c_num_game_store_options:
				# 	if option_for_store[1] > game_option_state[icountry][wdconfig.c_num_game_store_options-1][1]:
				# 		game_option_state[icountry][wdconfig.c_num_game_store_options - 1] = option_for_store
				# 		game_option_state[icountry] = sorted(game_option_state[icountry], key=lambda x: x[1], reverse=True)
				# else:
				# 	game_option_state[icountry].append(option_for_store)
				# 	game_option_state[icountry] = sorted(game_option_state[icountry], key=lambda x: x[1], reverse=True)

				# l_country_options[icountry].append()
				# orders_list += country_orders_list
				# icountry_list.append(icountry)

			print('Total success score for ', scountry, 'for this option run is', success_score)
		# end loop over each country for that option run
	# end if not b_waiting_for_AI or game_option_state = []

	if b_waiting_for_AI:
		l_donated, l_asked_ally, l_already_asked, l_done, l_ally_orders, l_best_options = [], [], [], [], [], []
		for icountry, _ in enumerate(country_names_tbl):
			l_donated.append(('None', 'None'))
			l_asked_ally.append(False)
			l_already_asked.append(icountry==0 or human_by_icountry[icountry])
			l_done.append(icountry==0 or human_by_icountry[icountry])
			l_ally_orders.append([])
			l_best_options.append([])

		d_countries =  {scountry: icountry for icountry, scountry in enumerate(country_names_tbl)}
		shuffled_country_names_ids = range(1, len(country_names_tbl))
		random.shuffle(shuffled_country_names_ids)

		for i_request_try in range(wdconfig.c_classic_AI_request_help_sanity_limit):
			if all(l_done):
				break
			for icountry in shuffled_country_names_ids:
				if l_done[icountry]:
					continue
				scountry = country_names_tbl[icountry]
				donated, b_asked_ally = l_donated[icountry], l_asked_ally[icountry]
				best_option = game_option_state.get_best_option(scountry, donated, b_no_ally=b_asked_ally)
				if best_option == []:
					l_done[icountry] = True
					continue
				if best_option.get_need_ally():
					sally, ally_orders = best_option.get_ally_data()
					ially = d_countries[sally]
					l_asked_ally[icountry] = True
					if not l_already_asked[ially]:
						l_already_asked[ially] = True
						best_ally_option = game_option_state.get_best_option(	sally, (ally_orders[0], ally_orders[2]),
																				b_no_ally=l_asked_ally[ially])
						#if giving a unit means that I have no option, I'm not going to do it.
						# In the future consider
						if best_ally_option != []:
							l_donated[ially] = (ally_orders[0], ally_orders[2])
							l_ally_orders[icountry] = ally_orders
							l_best_options[icountry] = best_option
							l_done[ially] = False
							l_done[icountry] = True
					# no else, keep going round
				else:
					l_best_options[icountry] = best_option
					l_done[icountry] = True

		for icountry, scountry in enumerate(country_names_tbl):
			if icountry == 0 or human_by_icountry[icountry]:
				continue
			print('Conclusion for ', scountry)
			ally_orders = l_ally_orders[icountry]
			if ally_orders != []:
				orders_list += [ally_orders]
				print('Ally order:', ' '.join(ally_orders))
			best_option = l_best_options[icountry]
			if best_option == []:
				print('No best option for ', scountry)
			else:
				orders_list += best_option.get_orders()
				print('Score:', best_option.get_score(), 'Orders:')
				for order in best_option.get_orders():
					print(' '.join(order))
			# if len(game_option_state[icountry]) == 0 or game_option_state[icountry][0] == []:
			# 	continue
			# country_orders_list, success_score = game_option_state[icountry][0]
			# orders_list += country_orders_list
			icountry_list.append(icountry)
		# game_option_state[:] = [] # delete the contents t=not the ref
		game_option_state.clear_donated()
		# game_option_state.set_success_orders_data([])
		success_orders_data.reset_for_move()

	orders_db = els.convert_list_to_phrases(orders_list)
	success_list = [True for o in orders_list]
	return orders_list, orders_db, success_list, icountry_list


def alliance_AI(alliance_state):
	# What is a little confusing here is that the function is called where it is assumed that it is
	# same country until the function returns. The caller knows which county

	l_notice_option = alliance_state.select_option_type([	'leave alliance notice', 'now allied notice',
															'no alliance notice', 'alliance accepted',
															'alliance rejected'])
	for notice_option in l_notice_option:
		alliance_state.remove_option(notice_option[0], baccept=True)

	l_app_ally_options = alliance_state.select_option_type(['app_ally', 'app_join'])
	for app_ally_option in l_app_ally_options:
		score = alliance_state.score_alliance_option(app_ally_option[1])
		alliance_state.remove_option(app_ally_option[0], baccept=(score >= wdconfig.c_alliance_accept_thresh))


	l_leave_option = alliance_state.select_option_type(['leave_alliance'])
	if l_leave_option == []:
		l_join_options = alliance_state.select_option_type(['ally_req', 'join_req'])
		opt_opts = []
		if l_join_options != []:
			for option_data in l_join_options:
				iopt, option = option_data
				score = alliance_state.score_alliance_option(option)
				opt_opts.append([score, iopt])
			sorted_opt_opts = sorted(opt_opts, key=lambda x: x[0], reverse=True)
			b_accept_first = False
			if sorted_opt_opts[0][0] >= wdconfig.c_alliance_propose_thresh:
				b_accept_first = True
			for isorted, sorted_opt in enumerate(sorted_opt_opts):
				alliance_state.remove_option(sorted_opt[1], isorted==0 and b_accept_first)
				# wd_game_state.get_alliance_state().exec_option(option, baccept=True)
	else:
		iopt, option = l_leave_option[0]
		score = alliance_state.score_alliance_option(option)
		# if score < wdconfig.c_alliance_terminate_thresh:
		alliance_state.remove_option(iopt, score < wdconfig.c_alliance_terminate_thresh)

	return alliance_state.move_on_resp('done')
	# if user_option_status != 'good':
	# 	break

