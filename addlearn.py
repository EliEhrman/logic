import makerecs as mr

class cl_add_gg(object):
	def __init__(	self, b_from_load, templ_len=None, scvo=None, gens_rec=None,
					score=None, rule_str=None, level=None, b_blocking=None):
		self.__templ_len = templ_len
		self.__scvo = scvo
		self.__gens_rec = gens_rec
		self.__initial_score = score
		self.__rule_str = rule_str
		self.__rule = mr.extract_rec_from_str(rule_str)
		self.__level = level
		self.__b_blocking = b_blocking

	def get_level(self):
		return self.__level

	def get_initial_score(self):
		return self.__initial_score

	def save(self, db_csvr):
		db_csvr.writerow([	'gg cont rule', 'templ len', self.__templ_len, 'scvo', self.__scvo,
							'gens rec', mr.gen_rec_str(self.__gens_rec	),
							'score', self.__initial_score, 'rule str', self.__rule_str,
							'level', self.__level, 'is blocking', self.__b_blocking])
		return

	def load(self, db_csvr):
		_, _, s_templ_len, _, self.__scvo, _, s_gens_rec, _, sscore, \
		_, s_rule_str, _, slevel, _, sb_blocking = next(db_csvr)
		self.__templ_len, self.__initial_score, self.__level = int(s_templ_len), float(sscore), int(slevel)
		self.__gens_rec = mr.extract_rec_from_str(s_gens_rec)
		self.__rule_str = s_rule_str
		self.__rule = mr.extract_rec_from_str(s_rule_str)
		self.__b_blocking = sb_blocking == 'True'
		return

	def filter(self, glv_dict, perm_gens_list, perm_preconds_list, perm_phrases_list,
							 step_results, perm_scvo_list, loop_level):
		match_list = []
		gens_list = []
		result_list = []
		normal_not_block_list = []
		for iperm, perm_scvo in enumerate(perm_scvo_list):
			if perm_scvo == self.__scvo and mr.does_match_rule(glv_dict, self.__rule, perm_preconds_list[iperm]):
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


def learn_add_step(the_rest_db, orders, cascade_els, step_results, def_article_dict,
						 db_len_grps, el_set_arr, glv_dict, sess, event_step_id, expected_but_not_found_list,
						 b_blocking):
	return
