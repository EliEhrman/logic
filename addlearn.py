from __future__ import print_function
import sys

import random
import config
import makerecs as mr
import clrecgrp
import compare_conts as cc

class cl_add_gg(object):
	def __init__(	self, b_from_load, templ_len=None, scvo=None, gens_rec=None,
					score=None, rule_str=None, level=0, b_blocking=None, cont_id=0, parent_id=None):
		self.__b_null_cont = True
		if templ_len is not None:
			self.__b_null_cont = False
		self.__templ_len = templ_len
		self.__scvo = scvo
		self.__gens_rec = gens_rec
		self.__initial_score = score
		self.__rule_str = rule_str
		if b_from_load or self.__b_null_cont:
			self.__rule = None
		else:
			self.__rule = mr.extract_rec_from_str(rule_str)
		self.__level = level
		self.__b_blocking = b_blocking
		self.__b_active = False
		self.__num_rows_grp_data = 0
		# self.__num_grps_rows = 0
		self.__grp_data = []
		self.__id = cont_id
		self.__parent_id = parent_id
		self.__status = ''
		self.__status_params = [0.0, 0.0]


	def update_stats(self, score, rule_str ):
		self.__initial_score = score
		self.__rule = mr.extract_rec_from_str(rule_str)

	def get_level(self):
		return self.__level

	def get_initial_score(self):
		return self.__initial_score

	def is_active(self):
		return self.__b_active

	def is_blocking(self):
		return self.__b_blocking

	def set_active(self, b_active):
		self.__b_active = b_active

	def is_null(self):
		return self.__b_null_cont

	def get_num_rows_grp_data(self):
		return self.__num_rows_grp_data

	def get_rule(self):
		return self.__rule

	def get_rule_str(self):
		return self.__rule_str

	def get_gens_rec(self):
		return self.__gens_rec

	def set_num_rows_grp_data(self, num_rows):
		self.__num_rows_grp_data = num_rows

	def get_grp_data(self):
		return self.__grp_data

	def get_id(self):
		return self.__id

	def get_parent_id(self):
		return self.__parent_id

	def set_score(self, score):
		self.__initial_score = score

	def get_status(self):
		return self.__status

	def set_status(self, new_status):
		self.__status = new_status

	def get_status_params(self):
		return self.__status_params

	def set_status_params(self, new_params):
		self.__status_params = new_params

	def save(self, db_csvr, b_write_grp_data=True):
		if self.__b_null_cont:
			db_csvr.writerow(['gg null cont rule', 'active', self.__b_active, 'num grp data rows', self.__num_rows_grp_data])
		else:
			num_rows_grp_data = self.__num_rows_grp_data if b_write_grp_data else 0
			db_csvr.writerow([	'gg cont rule', 'active:', self.__b_active, 'num grp data rows:', num_rows_grp_data,
								'templ len:', self.__templ_len, 'scvo:', self.__scvo,
								'gens rec:', mr.gen_rec_str(self.__gens_rec	),
								'score:', self.__initial_score, 'rule str:', self.__rule_str,
								'level:', self.__level, 'is blocking:', self.__b_blocking,
								'id:', self.__id, 'parent id:', self.__parent_id,
								'status:', self.__status,
								'status params:', '|'.join([str(v) for v in self.__status_params])])
		if b_write_grp_data and not self.__b_active and self.__num_rows_grp_data:
			for row in self.__grp_data:
				db_csvr.writerow(row)

		return

	def load(self, db_csvr, b_null):
		if b_null:
			_, _, sb_active, _, s_num_rows_grp_data = next(db_csvr)
		else:
			_,  _, sb_active, _, s_num_rows_grp_data, _, s_templ_len, _, self.__scvo, _, s_gens_rec, _, sscore, \
			_, s_rule_str, _, slevel, _, sb_blocking, _, sid, _, s_parent_id, \
			_, self.__status, _, s_status_params = next(db_csvr)
			self.__templ_len, self.__initial_score, self.__level = int(s_templ_len), float(sscore), int(slevel)
			self.__gens_rec = mr.extract_rec_from_str(s_gens_rec)
			self.__rule_str = s_rule_str
			self.__rule = mr.extract_rec_from_str(s_rule_str)
			self.__b_blocking = sb_blocking == 'True'
			self.__id, self.__parent_id = int(sid), int(s_parent_id)
			self.__b_null_cont = False
			l_status_params = s_status_params.split('|')
			self.__status_params = [float(sstatus) for sstatus in l_status_params]

		self.__b_active, self.__num_rows_grp_data = sb_active == 'True', int(s_num_rows_grp_data)
		self.__grp_data = []
		for irow in range(self.__num_rows_grp_data):
			self.__grp_data.append(next(db_csvr))

		return

	def filter(self, glv_dict, perm_gens_list, perm_preconds_list, perm_phrases_list,
							 step_results, perm_scvo_list, loop_level):
		match_list = []
		gens_list = []
		result_list = []
		normal_not_block_list = []
		for iperm, perm_scvo in enumerate(perm_scvo_list):
			# if perm_scvo == self.__scvo and mr.does_match_rule(glv_dict, self.__rule, perm_preconds_list[iperm]):
			if mr.match_partial_scvo(perm_scvo, self.__scvo, loop_level) \
					and mr.match_partial_rule(glv_dict, self.__rule, perm_preconds_list[iperm], loop_level):
				match_list.append(iperm)
				if loop_level == self.__level-1:
					b_result_found = False
					generated_result = mr.replace_vars_in_phrase(perm_preconds_list[iperm], self.__gens_rec)
					expected_result = generated_result[1:-1]
					for igens, one_perm_gens in enumerate(perm_gens_list[iperm]):
						event_result = step_results[igens]
						# if one_perm_gens == self.__gens_rec:
						if mr.match_rec_exact(expected_result, event_result):
							b_result_found = True
							gens_list.append(igens)
							normal_not_block_list.append(True)
							result_list.append(event_result)
							break
					if not b_result_found:
						gens_list.append(-1)
						normal_not_block_list.append(False)
						result_list.append(expected_result)

		# return match_list, gens_list
		return match_list, result_list, normal_not_block_list


# def learn_add_step(the_rest_db, orders, cascade_els, step_results, def_article_dict,
# 						 db_len_grps, el_set_arr, glv_dict, sess, event_step_id, expected_but_not_found_list,
# 						 b_blocking):
# 	return

class cl_cont_mgr(object):
	class Enum(set):
		def __getattr__(self, name):
			if name in self:
				return name
			raise AttributeError

	# The difference between expands and perfect or blocks and perfect_blocks is that
	# one is through tries and the other is just a perfect score on the gg score
	status = Enum(['untried', 'initial', 'perfect', 'expands', 'perfect_block', 'blocks',
				   'partial_expand', 'partial_block', 'irrelevant', 'mutant'])
	c_expands_min_tries = -1
	c_expands_score_thresh = 0.0
	c_expands_score_min_thresh = 0.0

	def __init__(self):
		null_cont = cl_add_gg(b_from_load=False)
		self.__cont_list = [null_cont]
		self.__max_cont_id = 0
		# self.__b_cont_stats_initialized = False
		self.__cont_stats_mgr = None
		print(self.status.untried)
		return

	def add_cont(self, gg_cont):
		self.__cont_list.append(gg_cont)

	def get_cont_stats_mgr(self):
		return self.__cont_stats_mgr

	def init_cont_stats_mgr(self, thresh, target_gens, glv_dict, b_exclude_irrelevant):
		exclude_list = []
		if b_exclude_irrelevant:
			exclude_list += [self.status.irrelevant]
		self.__cont_stats_mgr = cc.cl_cont_stats_mgr()
		target_conts_list = []
		for cont in self.__cont_list:
			if cont.is_null():
				continue
			if self.is_gens_in_target(glv_dict, target_gens,
									  cont.get_rule(), cont.get_gens_rec()):
				target_conts_list.append(cont)

		self.__cont_stats_mgr.init_from_list(target_conts_list, thresh, exclude_list)

	def delete_cont_by_rule(self, rule):
		for icc, ccont in enumerate(self.__cont_list):
			if ccont.is_null():
				continue
			cont_rule = ccont.get_rule()
			# In the following match we are ignoring the cd component of the like.
			# For now, I think that doesn't matter too much
			if mr.match_rec_exact(rule, cont_rule):
				print('db_cont_mgr: removing rule', icc)
				del self.__cont_list[icc]
				break

	def new_cont_by_rule(self, rule_params, status):
		rule, level, gens_rec, parent_id = rule_params
		rule_str = mr.gen_rec_str(rule)
		scvo = mr.gen_cvo_str(rule)
		self.__max_cont_id += 1
		new_cont = cl_add_gg(	b_from_load=False, templ_len=len(rule), scvo=scvo, gens_rec=gens_rec,
								score=-1.0, rule_str=rule_str, level=level+1, b_blocking=False,
								cont_id=self.__max_cont_id, parent_id=parent_id)
		new_cont.set_status(status)
		# self.add_cont(new_cont)
		return new_cont


	def init_cont_stats_mgr_from_file(self, fnt, b_analyze_and_modify_conts):
		self.__cont_stats_mgr = cc.cl_cont_stats_mgr()
		b_load_done = self.__cont_stats_mgr.load(fnt)
		if b_load_done:
			self.set_max_cont_id(self.__cont_stats_mgr.get_max_cont_id())
			if b_analyze_and_modify_conts:
				self.__cont_stats_mgr.analyze(self)
				self.__cont_stats_mgr.create_new_conts(self)
			stats_list = self.__cont_stats_mgr.get_cont_stats_list()
			for cont_stat in stats_list:
				self.add_cont(cont_stat.get_cont())

		return b_load_done

	# def get_cont_stats_list(self):
	# 	return self.__cont_stats_list
	#
	# def set_cont_stats_list(self, stats_list):
	# 	self.__cont_stats_list = stats_list
	#
	def get_max_cont_id(self):
		return self.__max_cont_id

	def set_max_cont_id(self, id):
		self.__max_cont_id = id

	def get_cont(self, icont):
		return self.__cont_list[icont]

	def get_cont_list(self):
		return self.__cont_list

	def new_cont(self, b_from_load, templ_len, scvo, gens_rec,
					score, rule_str, level, b_blocking, parent_id, gg_src):
		self.__max_cont_id += 1
		gg_src.set_cont_id(self.__max_cont_id)
		self.add_cont(cl_add_gg(b_from_load, templ_len, scvo, gens_rec,
					score, rule_str, level, b_blocking, self.__max_cont_id, parent_id))
		return

	def get_conts_above(self, score):
		icc_list = []
		for icc, ccont in enumerate(self.__cont_list):
			if icc == 0:
				continue
			status = ccont.get_status()
			params = ccont.get_status_params()
			if status == self.status.initial \
					or status == self.status.irrelevant\
					or status == self.status.untried:
				continue
			if ccont.get_initial_score() > score:
				icc_list.append(icc)
		return icc_list

	def select_cont(self):
		print('Select cont called.')
		best_cc = self.__cont_list[0]
		best_score = 0.0
		ibest = 0
		b_has_best_score_bonus = False
		b_has_parent_bonus = False
		b_parent_score = False
		if len(self.__cont_list) > 1:
			b_can_null, b_can_initial, b_can_untried, b_can_partial  = False, False, False, False
			untried_min_level, partial_min_level = sys.maxint, sys.maxint
			sel_type = 'null'
			for icc, ccont in enumerate(self.__cont_list):
				if icc == 0:
					b_can_null = True # Do more about this later
					continue
				status = ccont.get_status()
				params = ccont.get_status_params()
				if status == self.status.partial_expand or status == self.status.partial_block:
					no_new_cont_count = params[1]
					if no_new_cont_count >= config.c_cont_not_parent_max:
						print('icc partial:', icc, 'blocked by no new cont')
						continue
					b_can_partial = True
					level = ccont.get_level()
					if level > partial_min_level:
						print('icc partial:', icc, 'blocked by level', level)
						continue
					partial_min_level = level
					if b_can_untried or b_can_initial:
						print('icc partial:', icc, 'blocked by pre-set sel type')
						continue
					sel_type = 'partial'
					print('Select cont. icc:', icc, 'status:', status, 'level', level)
				elif status == self.status.untried:
					b_can_untried = True
					level = ccont.get_level()
					if level > untried_min_level:
						print('icc untried:', icc, 'blocked by level', level)
						continue
					untried_min_level = level
					if b_can_initial:
						print('icc untried:', icc, 'blocked by active intial')
						continue
					sel_type = 'untried'
				elif status == self.status.initial:
					no_new_cont_count = params[1]
					if no_new_cont_count >= config.c_cont_not_parent_max:
						print('icc initial:', icc, 'blocked by no new cont')
						continue
					b_can_initial = True
					sel_type = 'initial'



			for icc, ccont in enumerate(self.__cont_list):
				if icc == 0:
					if random.random() < config.c_select_cont_review_null_prob:
						print('cont random null override:')
						break
					else:
						continue
				status = ccont.get_status()
				params = ccont.get_status_params()
				if status == self.status.perfect or status == self.status.perfect_block \
						or status == self.status.irrelevant or status == self.status.blocks \
						or status == self.status.expands:
					print('Cont ignore icc', icc, 'status', status)
					continue
				score_bonus = 0.0
				if random.random() < config.c_select_cont_random_prob:
					best_cc = ccont
					ibest = icc
					b_has_best_score_bonus = False
					print('cont random override for icc', icc, 'status:', status)
					break
				if status == self.status.partial_expand or status == self.status.partial_block:
					if sel_type != 'partial':
						continue
					no_new_cont_count = params[1]
					if no_new_cont_count >= config.c_cont_not_parent_max:
						continue
					level  = ccont.get_level()
					if level > partial_min_level:
						continue
					score = ccont.get_initial_score()
					score_bonus += params[0]
					print('icc partial:', icc, 'score:', score, 'score bonus:', score_bonus)
					ccont.set_status_params([params[0] + config.c_select_cont_score_bonus, params[1]])
				elif status == self.status.untried:
					if sel_type != 'untried':
						continue
					level = ccont.get_level()
					if level > untried_min_level:
						continue
					score = ccont.get_initial_score()
					print('icc untried:', icc, 'score:', score)
					# exit()
				elif status == self.status.initial:
					if sel_type != 'initial':
						continue
					no_new_cont_count = params[1]
					if no_new_cont_count >= config.c_cont_not_parent_max:
						continue
					score = ccont.get_initial_score()
					score_bonus += params[0]
					ccont.set_status_params([params[0] + config.c_select_cont_score_bonus, params[1]])
					print('icc inital:', icc, 'score:', score, 'score bonus:', score_bonus)
				total_score = score + score_bonus
				if total_score > best_score:
					best_cc = ccont
					best_score = total_score
					ibest = icc
					b_has_best_score_bonus = score_bonus > 0.0
					print('cont best:', icc, 'best score', best_score)

		# ibest = 24

		if ibest >= 0:
			for icc, ccont in enumerate(self.__cont_list):
				if icc == ibest:
					print('Final cont selection. icc', icc)
					ccont.set_active(True)
					if b_has_best_score_bonus:
						params = ccont.get_status_params()
						ccont.set_status_params([0.0, params[1]])
					status = ccont.get_status()
					if status == self.status.initial or status == self.status.partial_expand or status == self.status.partial_block:
						params = ccont.get_status_params()
						ccont.set_status_params([params[0], params[1] + 1.0])

				else:
					ccont.set_active(False)

			best_cc = self.__cont_list[ibest] # just to make sure we access it again explicitly
		return best_cc, ibest

	def is_gens_in_target(self, glv_dict, target_list, rule_grp, gens_rec):
		if target_list == None:
			return True

		b_success = False
		for target_rule_grp in target_list:
			gens_rule_grp = mr.replace_vars_in_phrase(rule_grp, gens_rec)
			# target_rule_grp = mr.extract_rec_from_str(target_rule_str)
			if mr.rule_grp_is_one_in_two(glv_dict, gens_rule_grp, target_rule_grp):
				b_success = True
				break

		return b_success


	def create_new_conts(self, glv_dict, db_len_grps, i_active_cont, score_thresh, score_min, min_tests):
		parent_cont = self.__cont_list[i_active_cont]
		assert parent_cont.is_active()
		level = parent_cont.get_level() + 1

		valid_ggs = []
		for len_grp in db_len_grps:
			valid_ggs += len_grp.get_valid_ggs()
		# b_pick_new = False
		for gg_stats in valid_ggs:
			templ_len, templ_scvo, b_blocking, igg, num_successes, num_tests, rule_str, cont_id, gg = gg_stats
			if num_tests < min_tests: # for now, don't update rule
				continue
			score = num_successes / num_tests
			if score < score_min: #  or score > score_thresh:
				continue
			# b_pick_new = True
			if cont_id >= 0:
				self.__cont_list[cont_id].update_stats(score, rule_str)
			else:
				self.new_cont(	b_from_load=False, templ_len=templ_len, scvo=templ_scvo,
								gens_rec=gg.get_gens_rec(), score=score, rule_str=rule_str,
								level=level, b_blocking=b_blocking, parent_id=parent_cont.get_id(),
								gg_src=gg)
				new_cont = self.get_cont(self.__max_cont_id)
				if score > 1.0 - config.c_cd_epsilon:
					new_cont.set_status(self.status.perfect_block if b_blocking else self.status.perfect)
				elif parent_cont.is_null():
					if self.is_gens_in_target(	glv_dict, clrecgrp.cl_templ_grp.c_target_gens,
												gg.get_rule_grp(), gg.get_gens_rec()):
						new_cont.set_status(self.status.initial)
					else:
						new_cont.set_status(self.status.irrelevant)
				else:
					new_cont.set_status(self.status.untried)
				print('new cont created. parent:', parent_cont.get_id(), 'new id', self.__max_cont_id,
					  'status:', new_cont.get_status())
				new_cont.set_status_params([0.0, 0.0])
				parent_params = parent_cont.get_status_params()
				if parent_params != []:
					parent_cont.set_status_params([parent_params[0], 0.0])

		status = parent_cont.get_status()
		params = parent_cont.get_status_params()
		if status == self.status.untried:
			# num_occurred = 0.0
			if len(params) < 4:
				return True
			else:
				num_child_hits, num_child_results, num_parent_hits, num_parent_results = params

			if num_child_hits > self.c_expands_min_tries:
				child_prob = num_child_results / num_child_hits
				parent_prob = num_parent_results / num_parent_hits
				if parent_cont.is_blocking():
					expands_score = self.c_expands_score_thresh if child_prob <= 0.0 else parent_prob / child_prob
					# expands_score = num_hits / num_tries
				else:
					expands_score = self.c_expands_score_thresh if child_prob >= 0.9999  else (1.0 - parent_prob) / (1.0 - child_prob)

				print('untried results. num_child_hits', num_child_hits, 'num_child_results', num_child_results,
					  'num_parent_hits', num_parent_hits, 'num_parent_results', num_parent_results)
				if expands_score >= self.c_expands_score_thresh:
					parent_cont.set_status(self.status.blocks if parent_cont.is_blocking() else self.status.expands)
					parent_cont.set_score(expands_score/self.c_expands_score_thresh)
					parent_cont.set_status_params([0.0, 0.0])
				elif  expands_score < self.c_expands_score_min_thresh:
					parent_cont.set_status(self.status.irrelevant)
				else:
					parent_cont.set_status(self.status.partial_block if parent_cont.is_blocking() else self.status.partial_expand)
					parent_cont.set_score(expands_score/self.c_expands_score_thresh)
					parent_cont.set_status_params([0.0, 0.0])

				print('untried matured to status:', parent_cont.get_status())
				return False

		# return True means I see no reason to stop learning from this cont
		return True

	def save(self, db_csvr):
		db_csvr.writerow(['db cont mgr', 'num conts', len(self.__cont_list), 'max cont id', self.__max_cont_id])

	def load(self, db_csvr):
		_, _ , s_num_conts, _, s_max_cont_id = next(db_csvr)
		self.__max_cont_id = int(s_max_cont_id)
		self.__cont_list = [] # all conts including the null cont will be added explicitly by caller

		return int(s_num_conts)


