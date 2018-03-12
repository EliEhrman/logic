from __future__ import print_function
import itertools
import copy

import cascade
import rules
import els
import random
import addlearn

import makerecs as mr
from clrecgrp import cl_templ_grp
from clrecgrp import cl_len_grp

def select_cont(db_cont_mgr):
	gg_cont_list = db_cont_mgr.get_cont_list()
	best_gg = gg_cont_list[0]
	best_score = 0.0
	ibest = 0
	if len(gg_cont_list)  > 1:
		for igg, gg in enumerate(gg_cont_list[1:]):
			if gg.get_initial_score() > best_score:
				best_gg = gg
				best_score = gg.get_initial_score()
				ibest = igg

	return best_gg, ibest

def create_new_conts(db_cont_mgr, db_len_grps, parent_cont, score_thresh, score_min, min_tests):
	if parent_cont == None:
		curr_gg_cont = None
		level = 1
	else:
		level = parent_cont.get_level()+1

	valid_ggs = []
	for len_grp in db_len_grps:
		valid_ggs += len_grp.get_valid_ggs()
	b_pick_new = False
	for gg_stats in valid_ggs:
		templ_len, templ_scvo, b_blocking, igg, num_successes, num_tests, rule_str, gg = gg_stats
		if num_tests < min_tests:
			continue
		score = num_successes / num_tests
		if score < score_min or score > score_thresh:
			continue
		b_pick_new = True
		db_cont_mgr.add_cont(addlearn.cl_add_gg(	b_from_load=False, templ_len=templ_len, scvo=templ_scvo,
												gens_rec=gg.get_gens_rec(), score=score, rule_str=rule_str,
												level=level, b_blocking=b_blocking))

	return b_pick_new


def learn_more(gg_cont_list, i_gg_cont, db_len_grps, score_thresh, score_min, min_tests):
	if gg_cont_list == None or gg_cont_list == []:
		curr_gg_cont = None
		level = 1
	else:
		curr_gg_cont = gg_cont_list[i_gg_cont]
		level = curr_gg_cont.get_level()+1

	valid_ggs = []
	for len_grp in db_len_grps:
		valid_ggs += len_grp.get_valid_ggs()

	# new_gg_cont_list = []
	b_pick_new = False
	for gg_stats in valid_ggs:
		templ_len, templ_scvo, b_blocking, igg, num_successes, num_tests, rule_str, gg = gg_stats
		if num_tests < min_tests:
			continue
		score = num_successes / num_tests
		if score < score_min or score > score_thresh:
			continue
		b_pick_new = True
		gg_cont_list.append(addlearn.cl_add_gg(	b_from_load=False, templ_len=templ_len, scvo=templ_scvo,
												gens_rec=gg.get_gens_rec(), score=score, rule_str=rule_str,
												level=level, b_blocking=b_blocking))

	best_gg = None
	best_score = 0.0
	ibest = -1
	if b_pick_new:
		for igg, gg in enumerate(gg_cont_list):
			if gg.get_initial_score() > best_score:
				best_gg = gg
				best_score = gg.get_initial_score()
				ibest = igg

	if best_gg == None:
		return [], -1
	else:
		return gg_cont_list, ibest

	return gg_cont_list, ibest

# modify gg, all its creation paths, save, load and gg cont creation

def process_one_perm(	perm_gens_list, iperm, event_step_id, perm_preconds_list, perm_phrases_list,
						step_results, pcvo_list, perm_results_blocked_list,
						expected_but_not_found_list, b_blocking_depr, gg_confirmed_list,
						b_null_results, result_confirmed_list, event_result_score_list,
						db_len_grps, sess, el_set_arr, glv_dict, def_article_dict, curr_cont):
	if not b_null_results and all(result_confirmed_list):
		return
	b_blocking = perm_results_blocked_list[iperm]
	b_cont_blocking = not curr_cont.is_null and curr_cont.is_blocking()
	comb_len = len(perm_preconds_list[iperm])
	print('Evaluating the following perm', ('for block' if b_blocking else ''), ':')
	out_str = ''
	out_str = els.print_phrase(perm_preconds_list[iperm], perm_preconds_list[iperm], out_str, def_article_dict)
	print(out_str)
	for one_perm_gens in perm_gens_list[iperm]:
		out_str = ''
		out_str = els.print_phrase(one_perm_gens, one_perm_gens, out_str, def_article_dict)
		print('perm result: ', out_str)

	i_len_grp = -1
	for igrp, len_grp in enumerate(db_len_grps):
		if len_grp.len() == comb_len:
			i_len_grp = igrp
			perm_templ = len_grp.find_templ(pcvo_list[iperm])
			if not perm_templ:
				if b_null_results:
					continue
				# gg = cl_gens_grp(gens_rec=perm_gens_list[iperm], preconds_rec=perm_preconds_list[iperm])
				len_grp.add_templ(cl_templ_grp(b_from_load=False, templ_len=comb_len, scvo=pcvo_list[iperm],
											   preconds_rec=perm_preconds_list[iperm],
											   gens_rec_list=perm_gens_list[iperm],
											   event_result_list=step_results,
											   eid=event_step_id, b_blocking=b_blocking,
											   b_cont_blocking=b_cont_blocking))
			else:
				# num_pggs = perm_templ.get_num_pggs()
				# if num_pggs >= 2:
				# print('Getting score for multi-gg templ.')
				event_result_score_list = \
					perm_templ.get_match_score(	def_article_dict, perm_preconds_list[iperm],
												perm_phrases_list[iperm],
												step_results, event_step_id, b_blocking, event_result_score_list,
												result_confirmed_list, expected_but_not_found_list,
												gg_confirmed_list)

				if not b_null_results and all(result_confirmed_list):
					print('All results match confirmed ggs')
					mr.report_confirmers(db_len_grps, gg_confirmed_list, el_set_arr,
										 def_article_dict, glv_dict)
					break
				# if score >= 1.0:
				# 	print('match perfect!')
				# comb_len_passed = comb_len

				perm_templ.add_perm(preconds_rec=perm_preconds_list[iperm],
									gens_rec_list=perm_gens_list[iperm],
									perm_result_list=step_results,
									perm_result_blocked=perm_results_blocked_list[iperm],
									eid=event_step_id,
									db_len_grps=db_len_grps)

				perm_templ.do_learn(def_article_dict, sess, el_set_arr, curr_cont, event_step_id)
			# score = perm_templ.get_match_score(perm_preconds_list[iperm], events_to_queue, event_result_score_list, b_real_score=False)

			break
		# end if len matches
	# end loop over len groups
	if i_len_grp == -1:
		if not b_null_results:
			db_len_grps.append(cl_len_grp(b_from_load=False, init_len=comb_len, first_scvo=pcvo_list[iperm],
										  preconds_rec=perm_preconds_list[iperm],
										  gens_rec_list=perm_gens_list[iperm], event_result_list=step_results,
										  eid=event_step_id, b_blocking=b_blocking, b_cont_blocking=b_cont_blocking))
		# end of one perm in all_perms

# def learn_one_story_step(story_db, step_phrases, cascade_els, step_results, def_article_dict,
# 						 db_len_grps, el_set_arr, glv_dict, sess, event_step_id, expected_but_not_found_list,
# 						 level, gg_cont, b_blocking):
# 	event_result_score_list = [[] for _ in step_results]
# 	result_confirmed_list = [False for _ in step_results]
# 	gg_confirmed_list = [[] for _ in step_results]
#
# 	b_null_results = (result_confirmed_list == [])
#
# 	all_combs = [[]]
# 	if level > 0:
# 		all_combs = cascade.get_ext_phrase_cascade(cascade_els, story_db, step_phrases, '',
# 												   num_recurse_levels=2, max_num_phrases=1)
# 		all_combs = sorted(all_combs, key=len)
#
# 	for one_comb in all_combs:
# 		if not b_null_results and all(result_confirmed_list):
# 			break
#
# 		all_perms = list(itertools.permutations(one_comb, len(one_comb)))
#
# 		rule_base = step_phrases
# 		pcvo_list = []
# 		# gcvo_list = []
# 		perm_preconds_list = []
# 		perm_phrases_list = []
# 		perm_gens_list = []
# 		for one_perm in all_perms:
# 			new_conds, vars_dict, rule_phrases = mr.make_preconds_rule_from_phrases(rule_base, one_perm, story_db)
# 			gens_recs_arr = []
# 			if b_null_results:
# 				new_rule = rules.nt_rule(preconds=new_conds, gens=[])
# 				preconds_recs, gens_recs = \
# 					rules.gen_for_rule(b_gen_for_learn=True, rule=new_rule)
# 				perm_gens_list.append([])
# 			else:
# 				for event_result in step_results:
# 					new_gens = mr.make_gens_rule_from_phrases(vars_dict, event_result)
# 					new_rule = rules.nt_rule(preconds=new_conds, gens=new_gens)
# 					# overwrite the preconds_rec because it should always be identical in this inner loop
# 					preconds_recs, gens_recs = \
# 						rules.gen_for_rule(b_gen_for_learn=True, rule=new_rule)
# 					gens_recs_arr.append(gens_recs[0].phrase())
# 				perm_gens_list.append(gens_recs_arr)
# 			# take first el because the result is put into a list
# 			perm_preconds_list.append(preconds_recs[0].phrase())
# 			perm_phrases_list.append(rule_phrases)
# 			# perm_gens_list.append(gens_recs[0].phrase())
# 			pcvo_list.append(mr.gen_cvo_str(preconds_recs[0].phrase()))
#
# 		# end of one_perm in all perms
# 		# The following cuts down the number of perms by trying to use only the lexically first
# 		# member. After all there is no real difference to the order of the perm it's just that for a
# 		# particular comb we want everybody to agree on the order to cut down equivalent rules.
# 		# However, there is no guarantee that there will always be only one first
# 		pstr_min = min(pcvo_list)
# 		plist_min = [iperm for iperm, scvo in enumerate(pcvo_list) if scvo <= pstr_min]
# 		for iperm in plist_min:
# 			process_one_perm(perm_gens_list, iperm, event_step_id, perm_preconds_list, perm_phrases_list,
# 							 step_results, pcvo_list, expected_but_not_found_list, b_blocking, gg_confirmed_list,
# 							 b_null_results, result_confirmed_list, event_result_score_list,
# 							 db_len_grps, sess, el_set_arr, glv_dict, def_article_dict)
#
# 	if b_null_results:
# 		# Shouldn't be able to enter the following loop, but we'll get out anyway
# 		return
# 	for i_event_result, event_result_score in enumerate(event_result_score_list):
# 		top_score, i_top_score, top_score_len, top_score_scvo, top_score_igg = -1.0, -1, 1000, '', 0
# 		participants = set()
# 		random.shuffle(event_result_score)
# 		for iresult, one_score in enumerate(event_result_score):
# 			print(('Blocking' if b_blocking else 'Normal'), 'event result:', one_score)
# 			templ_score, templ_len, templ_scvo, templ_igg = one_score
# 			participants.add((templ_len, templ_scvo, templ_igg))
# 			# if templ_len < top_score_len:
# 			if templ_score > top_score or (templ_score == top_score and templ_len < top_score_len):
# 				top_score, i_top_score, top_score_len, top_score_scvo, top_score_igg = \
# 					templ_score, iresult, templ_len, templ_scvo, templ_igg
#
# 		if i_top_score > -1:
# 			for igrp, len_grp in enumerate(db_len_grps):
# 				if len_grp.len() == top_score_len:
# 					print('Successful match using templ len, scvo, igg, score:', top_score_len, top_score_scvo,
# 						  top_score_igg, top_score, 'For event result', step_results[i_event_result])
# 					top_score_templ = len_grp.find_templ(top_score_scvo)
# 					top_score_templ.add_point(top_score_igg)
# 					break
#
# 			winner = (top_score_len, top_score_scvo, top_score_igg)
# 			for one_particp in participants:
# 				for igrp, len_grp in enumerate(db_len_grps):
# 					if len_grp.len() == one_particp[0]:
# 						particp_templ = len_grp.find_templ(one_particp[1])
# 						particp_templ.apply_penalty(one_particp[2], -5 if one_particp == winner else 1)
# 						break
#
# 	return

def learn_one_story_step2(story_db, step_phrases_src, cascade_els, step_results_src, def_article_dict,
						  db_len_grps, el_set_arr, glv_dict, sess, event_step_id, expected_but_not_found_list,
						  level_depr, gg_cont, b_blocking_depr, b_test_rule=False):
	pcvo_blist = []
	perm_preconds_blist = []
	perm_phrases_blist = []
	perm_gens_blist = []
	perm_results_blocked_blist = []

	step_results_list = [step_results_src]
	step_results_blocked_list = [False]
	step_phrases_list = [step_phrases_src]
	step_story_idx_list = [] # change this is the initial event is a phrase in the story

	if gg_cont.is_null():
		level_loop_range = 1
	else:
		level_loop_range = gg_cont.get_level() + 1

	b_cont_blocking = gg_cont.is_blocking()


	for level in range(level_loop_range):
		if b_test_rule and level == (level_loop_range - 1):
			if step_phrases_list == []:
				return [False, False]
			if b_cont_blocking:
				step_results_matched = step_results_blocked_list
			else:
				step_results_matched = [not b for b in step_results_blocked_list]
			if any(step_results_matched):
				return [True, not b_cont_blocking]
			return [True, b_cont_blocking]
		pcvo_alist = []
		perm_preconds_alist = []
		perm_phrases_alist = []
		perm_gens_alist = []
		perm_results_alist = []
		perm_results_blocked_alist = []
		perm_story_idx_alist = []

		for i_prev_perm, step_phrases in enumerate(step_phrases_list):
			step_results = step_results_list[i_prev_perm]
			step_results_blocked = step_results_blocked_list[i_prev_perm]
			event_result_score_list = [[] for _ in step_results]
			result_confirmed_list = [False for _ in step_results]
			gg_confirmed_list = [[] for _ in step_results]
			if len(step_story_idx_list) > i_prev_perm:
				perm_story_idxs = step_story_idx_list[i_prev_perm]
			else:
				perm_story_idxs = []

			b_null_results = (result_confirmed_list == [])
			all_combs = [[]]
			if level > 0:
				all_combs = cascade.get_ext_phrase_cascade2(cascade_els, story_db, step_phrases, '',
														   num_recurse_levels=2, max_num_phrases=1)
				all_combs = [acomb for acomb in all_combs if acomb not in perm_story_idxs]
				all_combs = sorted(all_combs, key=len)
				# shuffle combs because each gg gets only one so we don't want to introduce an order-based bias
				random.shuffle(all_combs)

			for one_comb in all_combs:
				pcvo_list = []
				perm_preconds_list = []
				perm_phrases_list = []
				perm_gens_list = []
				perm_story_idx_list = []

				if not b_null_results and all(result_confirmed_list):
					break

				all_perms = list(itertools.permutations(one_comb, len(one_comb)))

				rule_base = step_phrases
				for one_perm in all_perms:
					preconds_recs, vars_dict, rule_phrases = mr.make_perm_preconds_rec(rule_base, one_perm, story_db)
					gens_recs_arr = []
					if b_null_results:
						perm_gens_list.append([])
					else:
						for event_result in step_results:
							gens_recs_arr.append(mr.make_rec_from_phrase_arr([mr.place_vars_in_phrase(vars_dict, event_result)]))
						perm_gens_list.append(gens_recs_arr)

					# new_conds, vars_dict, rule_phrases = mr.make_preconds_rule_from_phrases(rule_base, one_perm, story_db)
					# gens_recs_arr = []
					# if b_null_results:
					# 	new_rule = rules.nt_rule(preconds=new_conds, gens=[])
					# 	preconds_recs, gens_recs = \
					# 		rules.gen_for_rule(b_gen_for_learn=True, rule=new_rule)
					# 	perm_gens_list.append([])
					# else:
					# 	for event_result in step_results:
					# 		new_gens = mr.make_gens_rule_from_phrases(vars_dict, event_result)
					# 		new_rule = rules.nt_rule(preconds=new_conds, gens=new_gens)
					# 		# overwrite the preconds_rec because it should always be identical in this inner loop
					# 		preconds_recs, gens_recs = \
					# 			rules.gen_for_rule(b_gen_for_learn=True, rule=new_rule)
					# 		gens_recs_arr.append(gens_recs[0].phrase())
					# 	perm_gens_list.append(gens_recs_arr)
					# take first el because the result is put into a list
					# perm_preconds_list.append(preconds_recs[0].phrase())
					# pcvo_list.append(mr.gen_cvo_str(preconds_recs[0].phrase()))
					perm_preconds_list.append(preconds_recs)
					pcvo_list.append(mr.gen_cvo_str(preconds_recs))
					perm_phrases_list.append(rule_phrases)
					# perm_gens_list.append(gens_recs[0].phrase())
					perm_story_idx_list += [list(perm_story_idxs) + [one_comb]]

				# end of one_perm in all perms
				# The following cuts down the number of perms by trying to use only the lexically first
				# member. After all there is no real difference to the order of the perm it's just that for a
				# particular comb we want everybody to agree on the order to cut down equivalent rules.
				# However, there is no guarantee that there will always be only one first
				pstr_min = min(pcvo_list)
				plist_min = [iperm for iperm, scvo in enumerate(pcvo_list) if scvo <= pstr_min]
				random.shuffle(plist_min) # same reason as shuffle for one_comb
				pcvo_list = [pcvo_list[imin] for imin in plist_min]
				perm_preconds_list = [perm_preconds_list[imin] for imin in plist_min]
				perm_phrases_list = [perm_phrases_list[imin] for imin in plist_min]
				perm_gens_list = [perm_gens_list[imin] for imin in plist_min]

				if gg_cont.is_null() or level == level_loop_range - 1:
					pcvo_alist += pcvo_list
					perm_preconds_alist += perm_preconds_list
					perm_phrases_alist += perm_phrases_list
					perm_gens_alist += perm_gens_list
					perm_story_idx_alist += perm_story_idx_list
				else:
					# match_list, match_gens_list = \
					match_list, result_list, normal_not_blocking_list = \
						gg_cont.filter(glv_dict, perm_gens_list, perm_preconds_list, perm_phrases_list,
										step_results, pcvo_list, level)
					# Each ipem will produce one item in the match list and ONE item in the normal_not_blocking list
					# This is true even if there are multiple results. If one matches out of these multiple results
					# there will be an entry of True and if NONE, then there will be an entry of false
					# There is a potential bug here - not for now it's not real. If we are filtering we assume we are
					# level 1 or above and therefore there is only one result
					for imatch, one_match in enumerate(match_list):
						# if not b_test_rule and b_cont_blocking and len(normal_not_blocking_list) > imatch \
						# 		and normal_not_blocking_list[imatch] == True:
						# 	# Very important. A rule that is already blocking cannot block the block. It can only extend it
						# 	# This means that when using the rules we can iterate down the succeed rules and check whether
						# 	# they have a block
						# 	continue
						pcvo_alist.append(pcvo_list[one_match])
						perm_preconds_alist.append(perm_preconds_list[one_match])
						perm_phrases_alist.append(perm_phrases_list[one_match])
						perm_gens_alist.append(perm_gens_list[one_match])
						perm_story_idx_alist.append(perm_story_idx_list[one_match])
						if level_loop_range > 1 and level == (level_loop_range - 2):
							perm_results_alist.append([result_list[imatch]])
							perm_results_blocked_alist.append(not normal_not_blocking_list[imatch])
			# end loop of one_comb over all_combs
			# if the results haven't changed, we need to copy over the results of the last level loop
			if not (level_loop_range > 1 and level == (level_loop_range - 2)) and len(pcvo_alist) > 0:
				# perm_results_alist.append([step_results]*len(pcvo_alist))
				perm_results_alist += [step_results]*len(pcvo_alist)
				perm_results_blocked_alist += [step_results_blocked]*len(pcvo_alist)
		# end loop over all step phrases alternatives

		if level == level_loop_range - 1:
			pcvo_blist = pcvo_alist
			perm_preconds_blist = perm_preconds_alist
			perm_phrases_blist = perm_phrases_alist
			perm_gens_blist = perm_gens_alist
			perm_results_blocked_blist = perm_results_blocked_alist

		else:
			step_phrases_list = copy.deepcopy(perm_phrases_alist)
			# if level_loop_range > 1 and level == (level_loop_range - 2):
			step_results_list =  copy.deepcopy(perm_results_alist)
			step_results_blocked_list = copy.deepcopy(perm_results_blocked_alist)
			step_story_idx_list = copy.deepcopy(perm_story_idx_alist)
	# end of level loop
	# if b_cont_blocking and pcvo_blist != []:
	# 	perm_results_blocked_blist = [not b for b in perm_results_blocked_blist]
	for iperm, _ in enumerate(pcvo_blist):
		process_one_perm(perm_gens_blist, iperm, event_step_id, perm_preconds_blist, perm_phrases_blist,
						 step_results, pcvo_blist, perm_results_blocked_blist,
						 expected_but_not_found_list, b_blocking_depr, gg_confirmed_list,
						 b_null_results, result_confirmed_list, event_result_score_list,
						 db_len_grps, sess, el_set_arr, glv_dict, def_article_dict, gg_cont)
	if b_null_results:
		# Shouldn't be able to enter the following loop, but we'll get out anyway
		return
	for i_event_result, event_result_score in enumerate(event_result_score_list):
		top_score, i_top_score, top_score_len, top_score_scvo, top_score_igg, b_top_blocking = -1.0, -1, 1000, '', 0, None
		participants = set()
		random.shuffle(event_result_score)
		for iresult, one_score in enumerate(event_result_score):
			templ_score, templ_len, templ_scvo, templ_igg, b_result_blocking = one_score
			print(('Blocking' if b_result_blocking else 'Normal'), 'event result:', one_score)
			participants.add((templ_len, templ_scvo, templ_igg, b_result_blocking))
			# if templ_len < top_score_len:
			# if templ_score > top_score or (templ_score == top_score and templ_len < top_score_len):
			if templ_score > top_score:
					top_score, i_top_score, top_score_len, top_score_scvo, top_score_igg, b_top_blocking = \
					templ_score, iresult, templ_len, templ_scvo, templ_igg, b_result_blocking

	if i_top_score > -1:
		for igrp, len_grp in enumerate(db_len_grps):
			if len_grp.len() == top_score_len:
				print('Successful match using templ len, scvo, igg, score:', top_score_len, top_score_scvo,
					  top_score_igg, top_score, b_top_blocking, 'For event result', step_results[i_event_result])
				top_score_templ = len_grp.find_templ(top_score_scvo)
				top_score_templ.add_point(top_score_igg)
				break

		winner = (top_score_len, top_score_scvo, top_score_igg, b_top_blocking)
		for one_particp in participants:
			for igrp, len_grp in enumerate(db_len_grps):
				if len_grp.len() == one_particp[0]:
					particp_templ = len_grp.find_templ(one_particp[1])
					particp_templ.apply_penalty(one_particp[2], one_particp == winner)
					break

	return []

