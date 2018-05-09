import random
import copy
# from builtins import enumerate
import string
import time

import wdconfig
import wd_classicAI

class cl_alliance_state(object):
	class __cl_user_option(object):
		def __init__(self, id=-1, turn_id=-1, stype='', scountry='', l_params=[], text=''):
			self.__id = id
			self.__turn_id = turn_id
			self.__b_processed = False
			self.__b_accepted = False
			self.__l_params = l_params
			self.__stype = stype
			self.__scountry = scountry
			self.__text = text

		def get_scountry(self):
			return self.__scountry

		def get_stype(self):
			return self.__stype

		def get_params(self):
			return self.__l_params

		def get_id(self):
			return self.__id

	def __init__(self):
		self.__alliance_rel = []
		self.__alliance_matrix = []
		self.__propose_times = []
		self.__terminate_times = []
		self.__alliance_timer = -1
		self.__alliance_grps = [] # listing of members in group
		self.__alliance_membership = [] # rec for each country containing index of grp it belongs to
		self.__country_names_tbl = []
		self.__d_countries = dict()
		self.__l_user_l_options = []
		self.__shuffled_country_ids = []
		self.__max_option_id = -1
		self.__shuffled_curr_idx = 0 # index into suffled_country_idx whose turn it is now
		self.__alliance_names = [	'Empire of the Sun', 'Brothers of Liberty', 'Rebel Alliance', 'World Empire',
									'Freedom Federation', 'United Peoples', 'Free Forces']
		self.__l_options_for_ack = []
		# each of the following [[list of iopts],[same list of receved],[same list of accept not reject]
		self.__l_option_actions_pending = []
		self.__saved_gen_statements = [] # status db not including alliance statmnts
		self.__l_alliance_statements = []
		self.__l_alliance_msgs = []
		self.__b_option_created = False

	def get_statements(self):
		return self.__l_alliance_statements

	def get_msgs(self):
		return self.__l_alliance_msgs

	def find_option_by_id(self, l_options, iopt):
		def find(f, seq):
			"""Return first item in sequence where f(item) == True."""
			for item in seq:
				if f(item):
					return item

		return find(lambda option: option.get_id() == iopt, l_options)

	def select_option(self):
		l_curr_options = self.__l_user_l_options[self.__shuffled_country_ids[self.__shuffled_curr_idx]]
		if (len(l_curr_options)) < 1:
			raise ValueError('l_curr_options should not be empty')
		# irnd = random.randint(0, len(l_curr_options)-1)
		option = random.choice(l_curr_options)
		return option.get_id(), option

	def select_option_type(self, l_stype):
		l_ret = []
		l_curr_options = self.__l_user_l_options[self.__shuffled_country_ids[self.__shuffled_curr_idx]]
		for option in l_curr_options:
			if option.get_stype() in l_stype:
				l_ret.append([option.get_id(), option])
		return l_ret

	def remove_option(self, iopt, baccept):
		l_curr_options = self.__l_user_l_options[self.__shuffled_country_ids[self.__shuffled_curr_idx]]
		option = self.find_option_by_id(l_curr_options, iopt)
		self.exec_option(option, baccept)
		l_curr_options.remove(option)

	def exec_option(self, option, baccept):
		stype = option.get_stype()
		alliance_stypes = [	'join_req', 'ally_req', 'leave_alliance', 'leave alliance notice', 'now allied notice',
							'no alliance notice', 'alliance accepted', 'alliance rejected', 'app_ally', 'app_join']
		if stype in alliance_stypes:
			self.exec_alliance_option(option, baccept)

	# 'done', 'pause', 'another', 'skip', 'clearall'
	# returns 'good', 'paused', 'finished'
	def find_another(self):
		b_found = False
		for itry in range(len(self.__shuffled_country_ids)):
			self.__shuffled_curr_idx += 1
			if self.__shuffled_curr_idx >= len(self.__shuffled_country_ids):
				self.__shuffled_curr_idx = 0
			l_curr_options = self.__l_user_l_options[self.__shuffled_country_ids[self.__shuffled_curr_idx]]
			if len(l_curr_options) > 0:
				b_found = True
				break
		return 'good' if b_found else 'finished'

	def move_on_resp(self, resp):
		if resp == 'done':
			return self.find_another()
		elif resp == 'pause':
			return 'paused'
		elif resp == 'finished':
			return 'finished'
		else:
			raise ValueError('Unknown move on user option')


	def add_option(self, stype='', scountry = '', l_params=[], text = ''):
		self.__b_option_created = True
		self.__max_option_id += 1
		option = self.__cl_user_option(self.__max_option_id, self.__alliance_timer, stype, scountry, l_params, text)
		self.__l_user_l_options[self.__d_countries[scountry]].append(option)
		return option

	def create_alliance_options(self, scountry):
		icountry = self.__d_countries[scountry]
		i_my_grp = self.__alliance_membership[icountry]
		if icountry not in self.__alliance_grps[i_my_grp]:
			print('Coding error!. Exiting')
			exit(1)

		if len(self.__alliance_grps[i_my_grp]) == 1:
			grp_rels = []
			for igrp, grp in enumerate(self.__alliance_grps):
				b_can_do = False
				if igrp == i_my_grp or grp == []:
					continue
				if len(grp) > 1:
					b_can_do = True
					for icountry8 in grp:
						if self.__alliance_matrix[icountry][icountry8] > 0 and self.__propose_times[icountry][icountry8] >= 0:
							b_can_do = False
					if b_can_do:
						req_msg, option_type, params = 'Apply to join ' + self.__alliance_names[igrp], 'join_req', [igrp]
				else:
					if self.__alliance_matrix[icountry][grp[0]] <= 0 and self.__propose_times[icountry][grp[0]] < 0:
						b_can_do = True
						req_msg = 'Make request to ally with ' + self.__country_names_tbl[grp[0]]
						option_type, params = 'ally_req', [grp[0]]
				if b_can_do:
					self.add_option(option_type, scountry, params, req_msg)
		else:
			self.add_option('leave_alliance', scountry, [i_my_grp], 'Leave the ' + self.__alliance_names[i_my_grp])

	def score_alliance_option(self, option):
		def abort_if_not_single(icountry, abort_val, val):
			igrp = self.__alliance_membership[icountry]
			grp = self.__alliance_grps[igrp]
			return abort_val if len(grp) > 1 else val
		def abort_if_single(icountry, abort_val, val):
			igrp = self.__alliance_membership[icountry]
			grp = self.__alliance_grps[igrp]
			return abort_val if len(grp) == 1 else val
		def calc_grp_score(igrp):
			grp = self.__alliance_grps[params[0]]
			arel, fnum = 0.0, 0.0
			for icountry2 in grp:
				if icountry == icountry2:
					continue
				arel += self.__alliance_rel[icountry][icountry2]
				fnum += 1.0
			arel /= float(fnum)
			return arel
		stype, params, icountry = option.get_stype(), option.get_params(), self.__d_countries[option.get_scountry()]
		if stype =='leave_alliance':
			arel = calc_grp_score(params[0])
			arel = abort_if_single(icountry, 1.0, arel)
		elif stype == 'join_req':
			arel = calc_grp_score(params[0])
			arel = abort_if_not_single(icountry, 0.0, arel)
			grp = self.__alliance_grps[params[0]]
			arel = 0.0 if len(grp) <= 1 else arel
		elif stype == 'ally_req':
			arel = self.__alliance_rel[icountry][params[0]]
			arel = abort_if_not_single(icountry, 0.0, arel)
			arel = abort_if_not_single(params[0], 0.0, arel)
		elif stype == 'app_ally':
			arel = self.__alliance_rel[icountry][params[0]]
			arel = abort_if_not_single(icountry, 0.0, arel)
			arel = abort_if_not_single(params[0], 0.0, arel)
		elif stype == 'app_join':
			arel = self.__alliance_rel[icountry][params[0]]
			arel = abort_if_not_single(params[0], 0.0, arel)
			i_my_grp = self.__alliance_membership[icountry]
			mgrp = self.__alliance_grps[i_my_grp]
			if len(mgrp) < 2 or len(mgrp) >= wdconfig.c_alliance_max_grp_size:
				arel = 0.0
		else:
			arel = 0.0
		return arel
			# grp_rels.append([arel / float(len(grp)), igrp])

			# sorted_grp_rels = sorted(grp_rels, key=lambda x: x[0], reverse=True)

	def exec_alliance_option(self, option, baccept):
		stype = option.get_stype()
		scountry = option.get_scountry()
		icountry = self.__d_countries[scountry]
		i_my_grp = self.__alliance_membership[icountry]

		if stype == 'leave_alliance' and baccept: #nothin happens if you reject leaving alliance
			mgrp = self.__alliance_grps[i_my_grp]
			for icountry5 in mgrp:
				if icountry5 == icountry:
					continue
				self.__alliance_matrix[icountry][icountry5] = wdconfig.c_alliance_notice_time
				self.__alliance_matrix[icountry5][icountry] = wdconfig.c_alliance_notice_time
				self.add_option('leave alliance notice', self.__country_names_tbl[icountry5], [icountry],
								scountry + ' has left the alliance. Please acknowledge')
			# remove from grp
			mgrp.remove(icountry)
			# find new grp
			for ivgrp, vgrp in enumerate(self.__alliance_grps):
				if vgrp == []:
					vgrp.append(icountry)
					self.__alliance_membership[icountry] = ivgrp
					break
		elif stype == 'join_req' and baccept:
			igrp = option.get_params()[0]
			l_new_iopts = []
			for icountry4 in self.__alliance_grps[igrp]:
				req_msg = self.__country_names_tbl[icountry] + ' requested to join your alliance '
				option_type, params = 'app_join', [icountry]
				new_option = self.add_option(option_type, self.__country_names_tbl[icountry4], params, req_msg)
				self.__l_options_for_ack.append(new_option)
				# each of the following [[list of iopts],[same list of receved],[same list of accept not reject]
				l_new_iopts.append(new_option.get_id())
			self.__l_option_actions_pending.append([l_new_iopts, [False for _ in l_new_iopts], [False for _ in l_new_iopts]])
		elif stype == 'leave alliance notice':
			pass
		elif stype == 'now allied notice':
			pass
		elif stype == 'no alliance notice':
			pass
		elif stype == 'alliance accepted':
			pass
		elif stype == 'alliance rejected':
			pass
		elif stype == 'ally_req' and baccept:
			icountry2 = option.get_params()[0]
			req_msg = self.__country_names_tbl[icountry] + ' requested an alliance with you '
			option_type, params = 'app_ally', [icountry]
			self.add_option(option_type, self.__country_names_tbl[icountry2], params, req_msg)
		elif stype == 'app_ally':
			icountry7 = option.get_params()[0] # this is the country that applied. icountry is the country that accepted
			if baccept:
				# for icountry7 in bgrp:
				self.__alliance_matrix[icountry][icountry7] = wdconfig.c_alliance_notice_time + 1
				self.__alliance_matrix[icountry7][icountry] = wdconfig.c_alliance_notice_time + 1
				i_his_grp = self.__alliance_membership[icountry7]
				his_grp = self.__alliance_grps[i_his_grp]
				self.__alliance_grps[i_my_grp] = []
				self.__alliance_membership[icountry] = i_his_grp
				his_grp.append(icountry)
				self.add_option('now allied notice', self.__country_names_tbl[icountry7], [icountry],
								'You are now allied to ' + scountry + '. Please acknowledge')
			else:
				# for icountry4 in bgrp:
				self.__propose_times[icountry7][icountry] = wdconfig.c_alliance_wait_to_propose
				self.add_option('no alliance notice', self.__country_names_tbl[icountry7], [icountry],
								scountry + ' does not want an alliance with you. Please acknowledge')
		elif stype == 'app_join':
			icountry3, iopt = option.get_params()[0], option.get_id() #icountry3 is the country that applied
			mgrp = self.__alliance_grps[i_my_grp] # the group accepting or rejecting
			bfound, bcomplete = False, False
			for opt_action in self.__l_option_actions_pending:
				l_iopts, l_acked, l_accepted = opt_action
				if iopt not in l_iopts:
					continue
				iiopt = l_iopts.index(iopt)
				l_acked[iiopt], l_accepted[iiopt] = True, baccept
				if all(l_acked):
					bcomplete = True
					if all(l_accepted):
						for icountry6 in mgrp:
							self.__alliance_matrix[icountry3][icountry6] = wdconfig.c_alliance_notice_time + 1
							self.__alliance_matrix[icountry6][icountry3] = wdconfig.c_alliance_notice_time + 1
							self.add_option('alliance accepted', self.__country_names_tbl[icountry6], [icountry3],
											'Alliance has accepted ' + self.__country_names_tbl[icountry3]
											+ '. Please acknowledge')
						self.add_option('alliance accepted', self.__country_names_tbl[icountry3], [icountry3],
										'Alliance has accepted ' + self.__country_names_tbl[icountry3]
										+ '. Please acknowledge')
						i_old_grp = self.__alliance_membership[icountry3]
						self.__alliance_grps[i_old_grp] = []
						self.__alliance_membership[icountry3] = i_my_grp
						mgrp.append(icountry3)
					else:  # rejected
						for icountry6 in mgrp:
							self.__propose_times[icountry3][icountry6] = wdconfig.c_alliance_wait_to_propose
							self.add_option('alliance rejected', self.__country_names_tbl[icountry6], [icountry3],
											self.__country_names_tbl[icountry3]
											+ ' has not been accepted to the alliance. Please acknowledge')
						self.add_option('alliance rejected', self.__country_names_tbl[icountry3], [icountry3],
										self.__country_names_tbl[icountry3]
										+ ' has not been accepted to the alliance. Please acknowledge')
					# else - do no more, still waiting for the next response
					l_del_opts = []
					for iopt2 in l_iopts:
						for ack_opt in self.__l_options_for_ack:
							if ack_opt.get_id() == iopt2:
								l_del_opts.append(ack_opt)
					for del_opt in l_del_opts:
						self.__l_options_for_ack.remove(del_opt)
				# end of ial all acked
				bfound = True
				break
			# end loop over opt actions
			if bfound and bcomplete:
				self.__l_option_actions_pending.remove(opt_action)

	def create_output(self):
		self.__l_alliance_statements = []
		self.__l_alliance_msgs = []
		# alli_names = list(string.ascii_uppercase)
		# msgs = []
		for icountry, scountry in enumerate(self.__country_names_tbl):
			if icountry == 0:
				continue
			# l_allies = []
			for icountry2, scountry2 in enumerate(self.__country_names_tbl):
				if icountry2 == 0 or icountry2 == icountry:
					continue

				state = self.__alliance_matrix[icountry][icountry2]
				if state >= 0:
					stmt = [scountry, 'allied', 'to', scountry2]
					self.__l_alliance_statements.append(stmt)
					print(' '.join(stmt))
					# l_allies.append(scountry2)
					if state <= wdconfig.c_alliance_notice_time:
						stmt2 = [scountry, 'alliance', 'to', scountry2, 'on', 'notice']
						self.__l_alliance_statements.append(stmt2)
						print(' '.join(stmt2))
			# if len(l_allies) > 0:
			# 	msgs.append(scountry + ' allied to ' + ','.join(l_allies))
		# alli_num = 0
		for igrp, alli_grp in enumerate(self.__alliance_grps):
			if len(alli_grp) <= 1:
				continue
			self.__l_alliance_msgs.append(	'<p>' + self.__alliance_names[igrp] + ' consists of '
											+ ','.join([self.__country_names_tbl[icountry] for icountry in alli_grp])
											+ '</p>')
			# alli_num += 1

		# msgs.append('<p><form action="/board.php" method="get">'
		# 			+ '<input type="hidden" name="gameID" value=' + str(wd_game_state.get_gameID()) + '>'
		# 			+ '<input type="hidden" name="reqID" value=' + str(int(time.time()))[-6:] + '>'
		# 			+ '<input type="hidden" name="reqType" value=' + 'from_other' + '>'
		# 			+ '<input type="hidden" name="updateDB" value="Yes">'
		# 			+ '<select name="requests">'
		# 			+ '<option value="Yes">Yes</option>'
		# 			+ '<option value="No">No</option>'
		# 			+ '</select><br><br>'
		# 			+ '<button type="submit">Make Request</button><br>'
		# 			+ '</form><br>')
		# return msgs


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
			self.__d_countries = {scountry:icountry for icountry, scountry in enumerate(country_names_tbl)}
			self.__country_names_tbl = country_names_tbl
			self.__l_user_l_options = [[] for _ in country_names_tbl]
			self.__shuffled_country_ids = range(1, len(country_names_tbl))
			random.shuffle(self.__shuffled_country_ids)
		# alliance_data[:] = [game_turn-1, alliance_rel, alliance_matrix, propose_times, terminate_times]


		prev_pos_statement_list = self.__saved_gen_statements + self.__l_alliance_statements
		# create_output(self.__alliance_matrix, prev_pos_statement_list)

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

		"""
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
		"""



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


		return # create_output(self.__alliance_matrix, statement_list)

	def process_alliance_user_options(self):
		num_quiet = 0
		while num_quiet <= len(self.__country_names_tbl):
			if wd_classicAI.alliance_AI(self) != 'good':
				break
			if self.__b_option_created:
				num_quiet = -1
			self.__b_option_created = False
			num_quiet += 1

	def process_alliance_data(self, wd_game_state, game_turn, country_names_tbl, unit_owns_tbl, statement_list):
		if self.__alliance_timer < game_turn:
			self.__saved_gen_statements = copy.deepcopy(statement_list)
			self.make_alliances(wd_game_state, game_turn, country_names_tbl, unit_owns_tbl, statement_list)
			# out of date comment:
			# Don't get confused. This is for next time. The effects will not be felt on the
			# next line until the next turn
			for scountry in self.__country_names_tbl[1:]:
				self.create_alliance_options(scountry)

		self.process_alliance_user_options()

		self.create_output()

		self.__alliance_timer = game_turn
