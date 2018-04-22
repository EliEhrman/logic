from os.path import expanduser
import csv
import random
import copy

# import rules
import cascade
import els
import makerecs as mr



def load_forbidden(fn, version, l_rules):
	try:
		with open(fn, 'rb') as fh:
			csvr = csv.reader(fh, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
			_, version_str = next(csvr)
			version_num = int(version_str)
			if version_num == version:
				_, snum_rules = next(csvr)
				num_rules = int(snum_rules)
			else:
				raise IOError

			l_rules[:] = []
			for irule in range(num_rules):
				l_rules.append(mr.extract_rec_from_str(next(csvr)[0]))


	except IOError:
		print('Could not open forbidden rule file!')
		return False

	return True

def test_move_forbidden(l_move, l_rules, db, cascade_els, glv_dict):
	for rule in l_rules:
		if does_rule_fire(l_move, db, rule, cascade_els, glv_dict):
			print(' '.join(l_move+['forbidden!!!!']))
			return True

	return False

# This is the WRONG place. It should be in rules.py but rules is a dep of makerecs and els so it doesn't seem to want to import them
def does_rule_fire(move, story_db, rule, cascade_els, glv_dict):
	move_phrase = els.convert_list_to_phrases([move])
	# move_rec, _ = mr.make_rec_from_phrase_list(move_phrase)
	rule_arr = mr.make_rule_arr(rule)
	num_levels = len(rule_arr)

	step_phrases_list = [move_phrase]
	step_story_idx_list = []

	# src_perm_preconds_list = [move_rec]
	# src_perm_phrase_list = [move_rec]
	# src_pcvo_list = [mr.gen_cvo_str(move_rec)]

	for level in range(num_levels):

		pcvo_alist = []
		perm_preconds_alist = []
		perm_phrase_alist = []
		perm_story_idx_alist = []

		for i_prev_perm, step_phrases in enumerate(step_phrases_list):

			if len(step_story_idx_list) > i_prev_perm:
				perm_story_idxs = step_story_idx_list[i_prev_perm]
			else:
				perm_story_idxs = []

			pcvo_list = []
			perm_preconds_list = []
			perm_phrase_list = []
			perm_story_idx_list = []

			all_combs = [[]]
			if level > 0:
				all_combs = cascade.get_ext_phrase_cascade2(cascade_els, story_db, step_phrases, '',
														   num_recurse_levels=2, max_num_phrases=1)
				all_combs = [acomb for acomb in all_combs if acomb not in perm_story_idxs]
				all_combs = sorted(all_combs, key=len)
				# shuffle combs because each gg gets only one so we don't want to introduce an order-based bias
				random.shuffle(all_combs)
			# else:
			# 	perm_preconds_list = src_perm_preconds_list
			# 	perm_phrase_list = src_perm_phrase_list
			# 	pcvo_list = src_pcvo_list
			# 	perm_story_idx_list = []

			for one_perm in all_combs:
				preconds_recs, vars_dict, phrase_recs = mr.make_perm_preconds_rec(step_phrases, one_perm, story_db)
				perm_preconds_list.append(preconds_recs)
				pcvo_list.append(mr.gen_cvo_str(preconds_recs))
				perm_phrase_list.append(phrase_recs)
				# perm_gens_list.append(gens_recs[0].phrase())
				perm_story_idx_list += [list(perm_story_idxs) + [one_perm]]

			match_list = filter(glv_dict, rule, perm_preconds_list, None, pcvo_list, level)
			for imatch, one_match in enumerate(match_list):
				# if not b_test_rule and b_cont_blocking and len(normal_not_blocking_list) > imatch \
				# 		and normal_not_blocking_list[imatch] == True:
				# 	# Very important. A rule that is already blocking cannot block the block. It can only extend it
				# 	# This means that when using the rules we can iterate down the succeed rules and check whether
				# 	# they have a block
				# 	continue
				pcvo_alist.append(pcvo_list[one_match])
				perm_preconds_alist.append(perm_preconds_list[one_match])
				perm_phrase_alist.append(perm_phrase_list[one_match])
				perm_story_idx_alist.append(perm_story_idx_list[one_match])

			step_phrases_list = copy.deepcopy(perm_phrase_alist)
			step_story_idx_list = copy.deepcopy(perm_story_idx_alist)
	# end of level loop
	return len(step_phrases_list) > 0

def filter(	glv_dict, rule, perm_preconds_list, perm_phrases_list,
			perm_scvo_list, loop_level):
	rule_scvo = mr.gen_cvo_str(rule)
	match_list = []
	for iperm, perm_scvo in enumerate(perm_scvo_list):
		# if perm_scvo == self.__scvo and mr.does_match_rule(glv_dict, self.__rule, perm_preconds_list[iperm]):
		if mr.match_partial_scvo(perm_scvo, rule_scvo, loop_level) \
				and mr.match_partial_rule(glv_dict, rule, perm_preconds_list[iperm], loop_level):
			match_list.append(iperm)

	# return match_list, gens_list
	return match_list
