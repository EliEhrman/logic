import random
import copy
# from builtins import enumerate
import string

import wdconfig

class cl_alliance_state(object):
	def __init__(self):
		self.__alliance_rel = []
		self.__alliance_matrix = []
		self.__propose_times = []
		self.__terminate_times = []
		self.__alliance_timer = -1
		self.__alliance_grps = [] # listing of memebrs in group
		self.__alliance_membership = [] # rec for each country containing index of grp it belongs to

	def make_alliances(self, wd_game_state, game_turn, country_names_tbl, unit_owns_tbl, statement_list):
		if self.__alliance_rel == []:
			self.__alliance_rel = [[random.random() for _ in country_names_tbl] for _ in country_names_tbl]
			# alliance_timer = -1
			# alliance is represented as a value of wdconfig.c_alliance_notice_time + 1, then notice then remove
			self.__alliance_matrix = [[-1 for _ in country_names_tbl] for _ in country_names_tbl]
			self.__propose_times = [[-1 for _ in country_names_tbl] for _ in country_names_tbl]
			self.__terminate_times = [[-1 for _ in country_names_tbl] for _ in country_names_tbl]
			self.__alliance_grps = [[icountry] for icountry in range(len(country_names_tbl)) if icountry != 0]
			self.__alliance_membership = [-1 if icountry == 0 else (icountry-1) for icountry in range(len(country_names_tbl))]
		# alliance_data[:] = [game_turn-1, alliance_rel, alliance_matrix, propose_times, terminate_times]

		def create_output(state_matrix, p_statement_list):
			alli_names = list(string.ascii_uppercase)
			msgs = []
			for icountry, scountry in enumerate(country_names_tbl):
				if icountry == 0:
					continue
				# l_allies = []
				for icountry2, scountry2 in enumerate(country_names_tbl):
					if icountry2 == 0 or icountry2 == icountry:
						continue

					state = state_matrix[icountry][icountry2]
					if state >= 0:
						stmt = [scountry, 'allied', 'to', scountry2]
						p_statement_list.append(stmt)
						print(' '.join(stmt))
						# l_allies.append(scountry2)
						if state <= wdconfig.c_alliance_notice_time:
							stmt2 = [scountry, 'alliance', 'to', scountry2, 'on', 'notice']
							p_statement_list.append(stmt2)
							print(' '.join(stmt2))
				# if len(l_allies) > 0:
				# 	msgs.append(scountry + ' allied to ' + ','.join(l_allies))
			alli_num = 0
			for alli_grp in self.__alliance_grps:
				if len(alli_grp) <= 1:
					continue
				msgs.append(alli_names[alli_num] + '-Alliance consists of ' + ','.join([country_names_tbl[icountry] for icountry in alli_grp]))
				alli_num += 1


			return msgs

		# alliance_timer, alliance_rel, alliance_matrix, propose_times, terminate_times = alliance_data
		if self.__alliance_timer >= game_turn:
			return create_output(self.__alliance_matrix, statement_list)
		self.__alliance_timer = game_turn

		prev_pos_statement_list = copy.deepcopy(statement_list)
		create_output(self.__alliance_matrix, prev_pos_statement_list)

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
		for icountry, country_alliance_rel in enumerate(self.__alliance_rel):
			for icountry2, rel in enumerate(country_alliance_rel):
				self.__alliance_rel[icountry][icountry2] = max(0, min(rel + strength_effect_tbl[icountry][icountry2], 1))

		for icountry, scountry in enumerate(country_names_tbl):
			l_units = unit_owns_tbl.get(scountry, [])
			if l_units == [] or len(l_units) > wdconfig.c_alliance_oversize_limit:
				self.__alliance_rel[icountry] = [0.0 for _ in country_names_tbl]
				if l_units == []:
					self.__alliance_matrix[icountry] = [-1 for _ in country_names_tbl]
					self.__propose_times[icountry] = [-1 for _ in country_names_tbl]
					self.__terminate_times[icountry] = [-1 for _ in country_names_tbl]
				for icountry2, _ in enumerate(country_names_tbl):
					self.__alliance_rel[icountry2][icountry] = 0.0
					if l_units == []:
						self.__alliance_matrix[icountry2][icountry] = -1
						self.__propose_times[icountry2][icountry] = -1
						self.__terminate_times[icountry2][icountry] = -1

		del scountry
		shuffled_country_names_tbl = range(len(country_names_tbl))
		random.shuffle(shuffled_country_names_tbl)


		for icountry in shuffled_country_names_tbl:
			if icountry == 0:
				continue
			i_my_grp = self.__alliance_membership[icountry]
			if icountry not in self.__alliance_grps[i_my_grp]:
				print('Coding error!. Exiting')
				exit(1)

			if len(self.__alliance_grps[i_my_grp]) == 1:
				grp_rels = []
				for igrp, grp in enumerate(self.__alliance_grps):
					if igrp == i_my_grp or grp == []:
						continue
					arel = 0.0
					for icountry2 in grp:
						arel += self.__alliance_rel[icountry][icountry2]
					grp_rels.append([arel / float(len(grp)), igrp])
				sorted_grp_rels = sorted(grp_rels, key=lambda x: x[0], reverse=True)
				b_grp_found = False
				for p_grp_rels in sorted_grp_rels:
					if p_grp_rels[0] < wdconfig.c_alliance_propose_thresh:
						break
					b_can_do = True
					bgrp = self.__alliance_grps[p_grp_rels[1]]
					if len(bgrp) >= wdconfig.c_alliance_max_grp_size:
						continue
					for icountry3 in bgrp:
						if self.__propose_times[icountry][icountry3] >= 0 or self.__terminate_times[icountry][icountry3] >= 0:
							b_can_do = False
							break
					if not b_can_do:
						continue
					for icountry3 in bgrp:
						if self.__alliance_rel[icountry3][icountry] < wdconfig.c_alliance_accept_thresh:
							b_can_do = False
							break
					if b_can_do:
						for icountry7 in bgrp:
							self.__alliance_matrix[icountry][icountry7] = wdconfig.c_alliance_notice_time + 1
							self.__alliance_matrix[icountry7][icountry] = wdconfig.c_alliance_notice_time + 1
						self.__alliance_grps[i_my_grp] = []
						self.__alliance_membership[icountry] = p_grp_rels[1]
						bgrp.append(icountry)
						b_grp_found = True
					else:
						for icountry4 in bgrp:
							self.__propose_times[icountry][icountry4] = wdconfig.c_alliance_wait_to_propose
					if b_grp_found:
						break
			else: # already a member of a larger group
				arel = 0.0
				mgrp = self.__alliance_grps[i_my_grp]
				for icountry2 in mgrp:
					if icountry2 == icountry:
						continue
					arel += self.__alliance_rel[icountry][icountry2]
				if arel / float(len(mgrp)-1) <  wdconfig.c_alliance_terminate_thresh:
					for icountry5 in mgrp:
						if icountry5 == icountry:
							continue
						self.__alliance_matrix[icountry][icountry5] = wdconfig.c_alliance_notice_time
						self.__alliance_matrix[icountry5][icountry] = wdconfig.c_alliance_notice_time
					# remove from grp
					mgrp.remove(icountry)
					#find new grp
					for ivgrp, vgrp in enumerate(self.__alliance_grps):
						if vgrp == []:
							vgrp.append(icountry)
							self.__alliance_membership[icountry] = ivgrp
							break




		for icountry, scountry in enumerate(country_names_tbl):
			if icountry == 0:
				continue
			for icountry2, scountry2 in enumerate(country_names_tbl):
				if icountry2 == 0:
					continue

				if icountry == icountry2:
					continue

				rel, state = self.__alliance_rel[icountry][icountry2], self.__alliance_matrix[icountry][icountry2]
				since_propose, since_terminate = self.__propose_times[icountry][icountry2], self.__terminate_times[icountry][icountry2]
				new_rel, new_state, new_since_propose, new_since_terminate = rel, state, since_propose, since_terminate
				if state > wdconfig.c_alliance_notice_time:
					print('would have cancelled in old system')
					# if rel < wdconfig.c_alliance_terminate_thresh:
					# 	new_state = wdconfig.c_alliance_notice_time
					# 	if icountry > icountry2:
					# 		self.__alliance_matrix[icountry2][icountry] =  wdconfig.c_alliance_notice_time - 1
					# 	else:
					# 		self.__alliance_matrix[icountry2][icountry] = wdconfig.c_alliance_notice_time
				elif state <= wdconfig.c_alliance_notice_time and state > 0:
					new_state = state - 1
				elif state == 0:
					new_state = -1
					new_since_terminate = wdconfig.c_alliance_wait_after_terminate
				elif state < 0:
					if rel > wdconfig.c_alliance_propose_thresh:
						print('Would have proposed in the old system')
						# if since_terminate <= 0 and since_propose <= 0:
						# 	# propose
						# 	rel2 = self.__alliance_rel[icountry2][icountry]
						# 	if rel2 > wdconfig.c_alliance_accept_thresh:
						# 		new_state = wdconfig.c_alliance_notice_time + 1
						# 		self.__alliance_matrix[icountry2][icountry] = wdconfig.c_alliance_notice_time + 1
						# 	else:
						# 		new_since_propose = wdconfig.c_alliance_wait_to_propose

				if since_propose >= 0:
					new_since_propose -= 1
				if since_terminate >= 0:
					new_since_terminate -= 1

				self.__alliance_matrix[icountry][icountry2] = new_state
				self.__propose_times[icountry][icountry2], self.__terminate_times[icountry][icountry2] \
					= new_since_propose, new_since_terminate


		return create_output(self.__alliance_matrix, statement_list)
