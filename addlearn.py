import makerecs as mr

class cl_add_gg(object):
	def __init__(	self, b_from_load, templ_len=None, scvo=None, gens_rec=None,
					score=None, rule_str=None, level=None):
		self.__templ_len = templ_len
		self.__scvo = scvo
		self.__gens_rec = gens_rec
		self.__initial_score = score
		self.__rule_str = rule_str
		self.__level = level

	def get_level(self):
		return self.__level

	def get_initial_score(self):
		return self.__initial_score

	def save(self, db_csvr):
		db_csvr.writerow(['gg cont rule', self.__templ_len, self.__scvo, mr.gen_rec_str(self.__gens_rec	),
							self.__initial_score, self.__rule_str, self.__level])
		return

	def load(self, db_csvr):
		_, s_templ_len, self.__scvo, s_gens_rec, sscore, s_rule_str, slevel = next(db_csvr)
		self.__templ_len, self.__initial_score, self.__level = int(s_templ_len), float(sscore), int(slevel)
		self.__gens_rec = mr.extract_rec_from_str(s_gens_rec)
		self.__rule_str = s_rule_str
		return

	def filter(self, perm_gens_list, perm_preconds_list, perm_phrases_list,
							 step_results, perm_scvo_list, loop_level):
		match_list = []
		gens_list = []
		for iperm, perm_scvo in enumerate(perm_scvo_list):
			if perm_scvo == self.__scvo:
				match_list.append(iperm)
				if loop_level == self.__level-1:
					b_result_found = False
					for igens, one_perm_gens in enumerate(perm_gens_list[iperm]):
						if one_perm_gens == self.__gens_rec:
							b_result_found = True
							gens_list.append(igens)
							break
					if not b_result_found:
						gens_list.append(-1)
		return match_list, gens_list


def learn_add_step(the_rest_db, orders, cascade_els, step_results, def_article_dict,
						 db_len_grps, el_set_arr, glv_dict, sess, event_step_id, expected_but_not_found_list,
						 b_blocking):
	return
