import random
import makerecs as mr

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

	def update_stats(self, score, rule_str ):
		self.__initial_score = score
		self.__rule = mr.extract_rec_from_str(rule_str)

	def get_level(self):
		return self.__level

	def get_initial_score(self):
		return self.__initial_score

	def is_active(self):
		return self.__b_active

	def set_active(self, b_active):
		self.__b_active = b_active

	def is_null(self):
		return self.__b_null_cont

	def get_num_rows_grp_data(self):
		return self.__num_rows_grp_data

	def set_num_rows_grp_data(self, num_rows):
		self.__num_rows_grp_data = num_rows

	def get_grp_data(self):
		return self.__grp_data

	def get_id(self):
		return self.__id

	def get_parent_id(self):
		return self.__parent_id

	def save(self, db_csvr):
		if self.__b_null_cont:
			db_csvr.writerow(['gg null cont rule', 'active', self.__b_active, 'num grp data rows', self.__num_rows_grp_data])
		else:
			db_csvr.writerow([	'gg cont rule', 'active', self.__b_active, 'num grp data rows', self.__num_rows_grp_data,
								'templ len', self.__templ_len, 'scvo', self.__scvo,
								'gens rec', mr.gen_rec_str(self.__gens_rec	),
								'score', self.__initial_score, 'rule str', self.__rule_str,
								'level', self.__level, 'is blocking', self.__b_blocking,
								'id', self.__id, 'parent id', self.__parent_id])
		if not self.__b_active and self.__num_rows_grp_data:
			for row in self.__grp_data:
				db_csvr.writerow(row)

		return

	def load(self, db_csvr, b_null):
		if b_null:
			_, _, sb_active, _, s_num_rows_grp_data = next(db_csvr)
		else:
			_,  _, sb_active, _, s_num_rows_grp_data, _, s_templ_len, _, self.__scvo, _, s_gens_rec, _, sscore, \
			_, s_rule_str, _, slevel, _, sb_blocking, _, sid, _, s_parent_id = next(db_csvr)
			self.__templ_len, self.__initial_score, self.__level = int(s_templ_len), float(sscore), int(slevel)
			self.__gens_rec = mr.extract_rec_from_str(s_gens_rec)
			self.__rule_str = s_rule_str
			self.__rule = mr.extract_rec_from_str(s_rule_str)
			self.__b_blocking = sb_blocking == 'True'
			self.__id, self.__parent_id = int(sid), int(s_parent_id)
			self.__b_null_cont = False

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
					generated_result = mr.get_result_for_cvo_and_rec(perm_preconds_list[iperm], self.__gens_rec)
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
	def __init__(self):
		null_cont = cl_add_gg(b_from_load=False)
		self.__cont_list = [null_cont]
		self.__max_cont_id = 0
		return

	def add_cont(self, gg_cont):
		self.__cont_list.append(gg_cont)

	def get_max_cont_id(self):
		return self.__max_cont_id

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

	def select_cont(self):
		best_gg = self.__cont_list[0]
		best_score = 0.0
		ibest = 0
		if len(self.__cont_list) > 1:
			for igg, gg in enumerate(self.__cont_list):
				if igg == 0:
					if random.random() < 0.2/(float(len(self.__cont_list))):
						break
					else:
						continue
				if gg.get_initial_score() > best_score:
					best_gg = gg
					best_score = gg.get_initial_score()
					ibest = igg

		for igg, gg in enumerate(self.__cont_list):
			if igg == ibest:
				gg.set_active(True)
			else:
				gg.set_active(False)

		return best_gg, ibest

	def create_new_conts(self, db_len_grps, i_active_cont, score_thresh, score_min, min_tests):
		parent_cont = self.__cont_list[i_active_cont]
		assert parent_cont.is_active()
		level = parent_cont.get_level() + 1
		parent_with_new_child_list = []

		valid_ggs = []
		for len_grp in db_len_grps:
			valid_ggs += len_grp.get_valid_ggs()
		# b_pick_new = False
		for gg_stats in valid_ggs:
			templ_len, templ_scvo, b_blocking, igg, num_successes, num_tests, rule_str, cont_id, gg = gg_stats
			if num_tests < min_tests: # for now, don't update rule
				continue
			score = num_successes / num_tests
			if score < score_min or score > score_thresh:
				continue
			# b_pick_new = True
			if cont_id >= 0:
				self.__cont_list[cont_id].update_stats(score, rule_str)
			else:
				self.new_cont(	b_from_load=False, templ_len=templ_len, scvo=templ_scvo,
								gens_rec=gg.get_gens_rec(), score=score, rule_str=rule_str,
								level=level, b_blocking=b_blocking, parent_id=parent_cont.get_id(),
								gg_src=gg)

		return b_pick_new

	def save(self, db_csvr):
		db_csvr.writerow(['db cont mgr', 'num conts', len(self.__cont_list), 'max cont id', self.__max_cont_id])

	def load(self, db_csvr):
		_, _ , s_num_conts, _, s_max_cont_id = next(db_csvr)
		self.__max_cont_id = int(s_max_cont_id)
		self.__cont_list = [] # all conts including the null cont will be added explicitly by caller

		return int(s_num_conts)


