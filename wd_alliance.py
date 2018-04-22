import random
# from builtins import enumerate

import wdconfig

def make_alliances(game_turn, country_names_tbl, alliance_data, statement_list):
	if alliance_data == []:
		alliance_rel = [[random.random() for _ in country_names_tbl] for _ in country_names_tbl]
		# alliance_timer = -1
		# alliance is represented as a value of wdconfig.c_alliance_notice_time + 1, then notice then remove
		alliance_matrix = [[-1 for _ in country_names_tbl] for _ in country_names_tbl]
		propose_times = [[-1 for _ in country_names_tbl] for _ in country_names_tbl]
		terminate_times = [[-1 for _ in country_names_tbl] for _ in country_names_tbl]
		alliance_data[:] = [game_turn-1, alliance_rel, alliance_matrix, propose_times, terminate_times]

	def create_output(state_matrix):
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
					statement_list.append(stmt)
					print(' '.join(stmt))
					l_allies.append(scountry2)
					if state <= wdconfig.c_alliance_notice_time:
						stmt2 = [scountry, 'alliance', 'to', scountry2, 'on', 'notice']
						statement_list.append(stmt2)
						print(' '.join(stmt2))
			if len(l_allies) > 0:
				msgs.append(scountry + ' allied to ' + ','.join(l_allies))

		return msgs

	alliance_timer, alliance_rel, alliance_matrix, propose_times, terminate_times = alliance_data
	if alliance_timer >= game_turn:
		return create_output(alliance_matrix)
	alliance_data[0] = game_turn
	alliance_rel[:] = [[max(0, min((wdconfig.c_alliance_move_per_turn * (1.0 if random.random() < 0.5 else -1.0)) + rel, 1.0))
						for rel in country_alliance_rel] for country_alliance_rel in alliance_rel]
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


	return create_output(alliance_matrix)
