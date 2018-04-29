import random
import copy
# from builtins import enumerate

import wdconfig

def make_alliances(wd_game_state, game_turn, country_names_tbl, unit_owns_tbl, alliance_data, statement_list):
	if alliance_data == []:
		alliance_rel = [[random.random() for _ in country_names_tbl] for _ in country_names_tbl]
		# alliance_timer = -1
		# alliance is represented as a value of wdconfig.c_alliance_notice_time + 1, then notice then remove
		alliance_matrix = [[-1 for _ in country_names_tbl] for _ in country_names_tbl]
		propose_times = [[-1 for _ in country_names_tbl] for _ in country_names_tbl]
		terminate_times = [[-1 for _ in country_names_tbl] for _ in country_names_tbl]
		alliance_data[:] = [game_turn-1, alliance_rel, alliance_matrix, propose_times, terminate_times]

	def create_output(state_matrix, p_statement_list):
		msgs = []
		for icountry, scountry in enumerate(country_names_tbl):
			if icountry == 0:
				continue
			l_allies = []
			for icountry2, scountry2 in enumerate(country_names_tbl):
				if icountry2 == 0 or icountry2 == icountry:
					continue

				state = state_matrix[icountry][icountry2]
				if state >= 0:
					stmt = [scountry, 'allied', 'to', scountry2]
					p_statement_list.append(stmt)
					print(' '.join(stmt))
					l_allies.append(scountry2)
					if state <= wdconfig.c_alliance_notice_time:
						stmt2 = [scountry, 'alliance', 'to', scountry2, 'on', 'notice']
						p_statement_list.append(stmt2)
						print(' '.join(stmt2))
			if len(l_allies) > 0:
				msgs.append(scountry + ' allied to ' + ','.join(l_allies))

		return msgs

	alliance_timer, alliance_rel, alliance_matrix, propose_times, terminate_times = alliance_data
	if alliance_timer >= game_turn:
		return create_output(alliance_matrix, statement_list)
	alliance_data[0] = game_turn

	prev_pos_statement_list = copy.deepcopy(statement_list)
	create_output(alliance_matrix, prev_pos_statement_list)

	base_strength_diffs = wd_game_state.get_alliance_stats().predict_force_diffs(prev_pos_statement_list, unit_owns_tbl)
	print(base_strength_diffs)

	strength_effect_tbl = [[0.0 for _ in country_names_tbl] for _ in country_names_tbl]
	for icountry, scountry in enumerate(country_names_tbl):
		if icountry == 0:
			continue
		for icountry2, scountry2 in enumerate(country_names_tbl):
			if icountry2 == 0:
				continue

			if icountry == icountry2:
				continue

			ally_phrase = [scountry, 'allied', 'to', scountry2]
			if ally_phrase in prev_pos_statement_list:
				b_add = -1.0
				print('Consider deleting', ' '.join(ally_phrase))
				modify_list = [[0, ally_phrase, []]]
			else:
				b_add = 1.0
				print('Consider adding', ' '.join(ally_phrase))
				modify_list = [[0, [], ally_phrase]]
			new_strength_diffs = wd_game_state.get_alliance_stats().predict_force_diffs(prev_pos_statement_list, unit_owns_tbl, modify_list)
			diff_strength_diffs = new_strength_diffs - base_strength_diffs
			# alliance_dir = 1 if diff_strength_diffs[icountry] > 0.0 else (-1 if diff_strength_diffs[icountry] > 0.0 else 0)
			# alliance_diff = 0.0
			# if base_strength_diffs[icountry] != 0.0:
			limit = wdconfig.c_alliance_max_move_per_turn
			alliance_diff = b_add * diff_strength_diffs[icountry] # / abs(base_strength_diffs[icountry])
			strength_effect_tbl[icountry][icountry2] = max(-limit, min(wdconfig.c_alliance_move_per_turn * alliance_diff, limit))
			print(new_strength_diffs, diff_strength_diffs, alliance_diff)

	del icountry, scountry, icountry2, scountry2, ally_phrase, new_strength_diffs, diff_strength_diffs, alliance_diff

	# alliance_rel[:] = [[max(0, min((wdconfig.c_alliance_move_per_turn * (1.0 if random.random() < 0.5 else -1.0)) + rel, 1.0))
	# 					for rel in country_alliance_rel] for country_alliance_rel in alliance_rel]
	# alliance_rel[:] = [[max(0, min((wdconfig.c_alliance_move_per_turn
	# 								* (1.0 if strength_effect_tbl[icountry][icountry2] == 1
	# 								   else (-1.0 if strength_effect_tbl[icountry][icountry2] == -1 else 0.0))) + rel, 1.0))
	# 					]
	# 					for icountry, country_alliance_rel in enumerate(alliance_rel)]
	# for icountry, country_alliance_rel in enumerate(alliance_rel):
	# 	for icountry2, rel in enumerate(country_alliance_rel):
	# 		new_rel = (wdconfig.c_alliance_move_per_turn \
	# 				  * (1.0 if strength_effect_tbl[icountry][icountry2] == 1
	# 					 else (-1.0 if strength_effect_tbl[icountry][icountry2] == -1 else 0.0))) + rel
	# 		alliance_rel[icountry][icountry2] = max(0, min(new_rel, 1.0))
	# r_countries = range(len(country_names_tbl))
	# alliance_rel[:] = [[alliance_rel[icountry][icountry2] + strength_effect_tbl[icountry][icountry2]
	# 					for icountry2 in r_countries] for icountry in r_countries] # alliance_rel + strength_effect_tbl
	for icountry, country_alliance_rel in enumerate(alliance_rel):
		for icountry2, rel in enumerate(country_alliance_rel):
			alliance_rel[icountry][icountry2] = max(0, min(rel + strength_effect_tbl[icountry][icountry2], 1))

	for icountry, scountry in enumerate(country_names_tbl):
		l_units = unit_owns_tbl.get(scountry, [])
		if l_units == [] or len(l_units) > wdconfig.c_alliance_oversize_limit:
			alliance_rel[icountry] = [0.0 for _ in country_names_tbl]
			if l_units == []:
				alliance_matrix[icountry] = [-1 for _ in country_names_tbl]
				propose_times[icountry] = [-1 for _ in country_names_tbl]
				terminate_times[icountry] = [-1 for _ in country_names_tbl]
			for icountry2, _ in enumerate(country_names_tbl):
				alliance_rel[icountry2][icountry] = 0.0
				if l_units == []:
					alliance_matrix[icountry2][icountry] = -1
					propose_times[icountry2][icountry] = -1
					terminate_times[icountry2][icountry] = -1


	for icountry, scountry in enumerate(country_names_tbl):
		if icountry == 0:
			continue
		for icountry2, scountry2 in enumerate(country_names_tbl):
			if icountry2 == 0:
				continue

			if icountry == icountry2:
				continue

			rel, state = alliance_rel[icountry][icountry2], alliance_matrix[icountry][icountry2]
			since_propose, since_terminate = propose_times[icountry][icountry2], terminate_times[icountry][icountry2]
			new_rel, new_state, new_since_propose, new_since_terminate = rel, state, since_propose, since_terminate
			if state > wdconfig.c_alliance_notice_time:
				if rel < wdconfig.c_alliance_terminate_thresh:
					new_state = wdconfig.c_alliance_notice_time
					if icountry > icountry2:
						alliance_matrix[icountry2][icountry] =  wdconfig.c_alliance_notice_time - 1
					else:
						alliance_matrix[icountry2][icountry] = wdconfig.c_alliance_notice_time
			elif state <= wdconfig.c_alliance_notice_time and state > 0:
				new_state = state - 1
			elif state == 0:
				new_state = -1
				new_since_terminate = wdconfig.c_alliance_wait_after_terminate
			elif state < 0:
				if rel > wdconfig.c_alliance_propose_thresh:
					if since_terminate <= 0 and since_propose <= 0:
						# propose
						rel2 = alliance_rel[icountry2][icountry]
						if rel2 > wdconfig.c_alliance_accept_thresh:
							new_state = wdconfig.c_alliance_notice_time + 1
							alliance_matrix[icountry2][icountry] = wdconfig.c_alliance_notice_time + 1
						else:
							new_since_propose = wdconfig.c_alliance_wait_to_propose

			if since_propose >= 0:
				new_since_propose -= 1
			if since_terminate >= 0:
				new_since_terminate -= 1

			alliance_matrix[icountry][icountry2] = new_state
			propose_times[icountry][icountry2], terminate_times[icountry][icountry2] = new_since_propose, new_since_terminate


	return create_output(alliance_matrix, statement_list)
