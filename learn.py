from __future__ import print_function
import itertools
import cascade
import rules
import els
import random
import addlearn
import copy

import makerecs as mr
from clrecgrp import cl_templ_grp
from clrecgrp import cl_len_grp

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
		templ_len, templ_scvo, igg, num_successes, num_tests, rule_str, gg = gg_stats
		if num_tests < min_tests:
			continue
		score = num_successes / num_tests
		if score < score_min or score > score_thresh:
			continue
		b_pick_new = True
		gg_cont_list.append(addlearn.cl_add_gg(	b_from_load=False, templ_len=templ_len, scvo=templ_scvo,
												gens_rec=gg.get_gens_rec(), score=score, rule_str=rule_str,
												level=level))

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

def process_one_perm(	perm_gens_list, iperm, event_step_id, perm_preconds_list, perm_phrases_list,
						step_results, pcvo_list, expected_but_not_found_list, b_blocking, gg_confirmed_list,
						b_null_results, result_confirmed_list, event_result_score_list,
						db_len_grps, sess, el_set_arr, glv_dict, def_article_dict):
	if not b_null_results and all(result_confirmed_list):
		return
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
											   eid=event_step_id, b_blocking=b_blocking))
			else:
				# num_pggs = perm_templ.get_num_pggs()
				# if num_pggs >= 2:
				# print('Getting score for multi-gg templ.')
				event_result_score_list = \
					perm_templ.get_match_score(	def_article_dict, perm_preconds_list[iperm],
												perm_phrases_list[iperm],
												step_results, event_step_id, event_result_score_list,
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
									eid=event_step_id,
									db_len_grps=db_len_grps)

				perm_templ.do_learn(def_article_dict, sess, el_set_arr)
			# score = perm_templ.get_match_score(perm_preconds_list[iperm], events_to_queue, event_result_score_list, b_real_score=False)

			break
		# end if len matches
	# end loop over len groups
	if i_len_grp == -1:
		if not b_null_results:
			db_len_grps.append(cl_len_grp(b_from_load=False, init_len=comb_len, first_scvo=pcvo_list[iperm],
										  preconds_rec=perm_preconds_list[iperm],
										  gens_rec_list=perm_gens_list[iperm], event_result_list=step_results,
										  eid=event_step_id))
		# end of one perm in all_perms

def learn_one_story_step(story_db, step_phrases, cascade_els, step_results, def_article_dict,
						 db_len_grps, el_set_arr, glv_dict, sess, event_step_id, expected_but_not_found_list,
						 level, gg_cont, b_blocking):
	event_result_score_list = [[] for _ in step_results]
	result_confirmed_list = [False for _ in step_results]
	gg_confirmed_list = [[] for _ in step_results]

	b_null_results = (result_confirmed_list == [])

	all_combs = [[]]
	if level > 0:
		all_combs = cascade.get_ext_phrase_cascade(cascade_els, story_db, step_phrases, '',
												   num_recurse_levels=2, max_num_phrases=1)
		all_combs = sorted(all_combs, key=len)

	for one_comb in all_combs:
		if not b_null_results and all(result_confirmed_list):
			break

		all_perms = list(itertools.permutations(one_comb, len(one_comb)))

		rule_base = step_phrases
		pcvo_list = []
		# gcvo_list = []
		perm_preconds_list = []
		perm_phrases_list = []
		perm_gens_list = []
		for one_perm in all_perms:
			new_conds, vars_dict, rule_phrases = mr.make_preconds_rule_from_phrases(rule_base, one_perm, story_db)
			gens_recs_arr = []
			if b_null_results:
				new_rule = rules.nt_rule(preconds=new_conds, gens=[])
				preconds_recs, gens_recs = \
					rules.gen_for_rule(b_gen_for_learn=True, rule=new_rule)
				perm_gens_list.append([])
			else:
				for event_result in step_results:
					new_gens = mr.make_gens_rule_from_phrases(vars_dict, event_result)
					new_rule = rules.nt_rule(preconds=new_conds, gens=new_gens)
					# overwrite the preconds_rec because it should always be identical in this inner loop
					preconds_recs, gens_recs = \
						rules.gen_for_rule(b_gen_for_learn=True, rule=new_rule)
					gens_recs_arr.append(gens_recs[0].phrase())
				perm_gens_list.append(gens_recs_arr)
			# take first el because the result is put into a list
			perm_preconds_list.append(preconds_recs[0].phrase())
			perm_phrases_list.append(rule_phrases)
			# perm_gens_list.append(gens_recs[0].phrase())
			pcvo_list.append(mr.gen_cvo_str(preconds_recs[0].phrase()))

		# end of one_perm in all perms
		# The following cuts down the number of perms by trying to use only the lexically first
		# member. After all there is no real difference to the order of the perm it's just that for a
		# particular comb we want everybody to agree on the order to cut down equivalent rules.
		# However, there is no guarantee that there will always be only one first
		pstr_min = min(pcvo_list)
		plist_min = [iperm for iperm, scvo in enumerate(pcvo_list) if scvo <= pstr_min]
		for iperm in plist_min:
			process_one_perm(perm_gens_list, iperm, event_step_id, perm_preconds_list, perm_phrases_list,
							 step_results, pcvo_list, expected_but_not_found_list, b_blocking, gg_confirmed_list,
							 b_null_results, result_confirmed_list, event_result_score_list,
							 db_len_grps, sess, el_set_arr, glv_dict, def_article_dict)

	if b_null_results:
		# Shouldn't be able to enter the following loop, but we'll get out anyway
		return
	for i_event_result, event_result_score in enumerate(event_result_score_list):
		top_score, i_top_score, top_score_len, top_score_scvo, top_score_igg = -1.0, -1, 1000, '', 0
		participants = set()
		random.shuffle(event_result_score)
		for iresult, one_score in enumerate(event_result_score):
			print(('Blocking' if b_blocking else 'Normal'), 'event result:', one_score)
			templ_score, templ_len, templ_scvo, templ_igg = one_score
			participants.add((templ_len, templ_scvo, templ_igg))
			# if templ_len < top_score_len:
			if templ_score > top_score or (templ_score == top_score and templ_len < top_score_len):
				top_score, i_top_score, top_score_len, top_score_scvo, top_score_igg = \
					templ_score, iresult, templ_len, templ_scvo, templ_igg

		if i_top_score > -1:
			for igrp, len_grp in enumerate(db_len_grps):
				if len_grp.len() == top_score_len:
					print('Successful match using templ len, scvo, igg, score:', top_score_len, top_score_scvo,
						  top_score_igg, top_score, 'For event result', step_results[i_event_result])
					top_score_templ = len_grp.find_templ(top_score_scvo)
					top_score_templ.add_point(top_score_igg)
					break

			winner = (top_score_len, top_score_scvo, top_score_igg)
			for one_particp in participants:
				for igrp, len_grp in enumerate(db_len_grps):
					if len_grp.len() == one_particp[0]:
						particp_templ = len_grp.find_templ(one_particp[1])
						particp_templ.apply_penalty(one_particp[2], -5 if one_particp == winner else 1)
						break

	return

def learn_one_story_step2(story_db, step_phrases_src, cascade_els, step_results_src, def_article_dict,
						 db_len_grps, el_set_arr, glv_dict, sess, event_step_id, expected_but_not_found_list,
						 level_depr, gg_cont, b_blocking):
	pcvo_blist = []
	perm_preconds_blist = []
	perm_phrases_blist = []
	perm_gens_blist = []

	step_results_list = [step_results_src]
	step_phrases_list = [step_phrases_src]

	if gg_cont == None:
		level_loop_range = 1
	else:
		level_loop_range = gg_cont.get_level() + 1

	for level in range(level_loop_range):
		pcvo_alist = []
		perm_preconds_alist = []
		perm_phrases_alist = []
		perm_gens_alist = []
		perm_results_alist = []

		for i_prev_perm, step_phrases in enumerate(step_phrases_list):
			step_results = step_results_list[i_prev_perm]
			event_result_score_list = [[] for _ in step_results]
			result_confirmed_list = [False for _ in step_results]
			gg_confirmed_list = [[] for _ in step_results]

			b_null_results = (result_confirmed_list == [])
			all_combs = [[]]
			if level > 0:
				all_combs = cascade.get_ext_phrase_cascade2(cascade_els, story_db, step_phrases, '',
														   num_recurse_levels=2, max_num_phrases=1)
				all_combs = sorted(all_combs, key=len)

			for one_comb in all_combs:
				pcvo_list = []
				perm_preconds_list = []
				perm_phrases_list = []
				perm_gens_list = []

				if not b_null_results and all(result_confirmed_list):
					break

				all_perms = list(itertools.permutations(one_comb, len(one_comb)))

				rule_base = step_phrases
				for one_perm in all_perms:
					new_conds, vars_dict, rule_phrases = mr.make_preconds_rule_from_phrases(rule_base, one_perm, story_db)
					gens_recs_arr = []
					if b_null_results:
						new_rule = rules.nt_rule(preconds=new_conds, gens=[])
						preconds_recs, gens_recs = \
							rules.gen_for_rule(b_gen_for_learn=True, rule=new_rule)
						perm_gens_list.append([])
					else:
						for event_result in step_results:
							new_gens = mr.make_gens_rule_from_phrases(vars_dict, event_result)
							new_rule = rules.nt_rule(preconds=new_conds, gens=new_gens)
							# overwrite the preconds_rec because it should always be identical in this inner loop
							preconds_recs, gens_recs = \
								rules.gen_for_rule(b_gen_for_learn=True, rule=new_rule)
							gens_recs_arr.append(gens_recs[0].phrase())
						perm_gens_list.append(gens_recs_arr)
					# take first el because the result is put into a list
					perm_preconds_list.append(preconds_recs[0].phrase())
					perm_phrases_list.append(rule_phrases)
					# perm_gens_list.append(gens_recs[0].phrase())
					pcvo_list.append(mr.gen_cvo_str(preconds_recs[0].phrase()))

				# end of one_perm in all perms
				# The following cuts down the number of perms by trying to use only the lexically first
				# member. After all there is no real difference to the order of the perm it's just that for a
				# particular comb we want everybody to agree on the order to cut down equivalent rules.
				# However, there is no guarantee that there will always be only one first
				pstr_min = min(pcvo_list)
				plist_min = [iperm for iperm, scvo in enumerate(pcvo_list) if scvo <= pstr_min]
				pcvo_list = [pcvo_list[imin] for imin in plist_min]
				perm_preconds_list = [perm_preconds_list[imin] for imin in plist_min]
				perm_phrases_list = [perm_phrases_list[imin] for imin in plist_min]
				perm_gens_list = [perm_gens_list[imin] for imin in plist_min]

				if gg_cont == None or level == level_loop_range - 1:
					pcvo_alist += pcvo_list
					perm_preconds_alist += perm_preconds_list
					perm_phrases_alist += perm_phrases_list
					perm_gens_alist += perm_gens_list
				else:
					match_list, match_gens_list = \
						gg_cont.filter(perm_gens_list, perm_preconds_list, perm_phrases_list,
										step_results, pcvo_list, level)
					for imatch, one_match in enumerate(match_list):
						pcvo_alist.append(pcvo_list[one_match])
						perm_preconds_alist.append(perm_preconds_list[one_match])
						perm_phrases_alist.append(perm_phrases_list[one_match])
						perm_gens_alist.append(perm_gens_list[one_match])
						if level_loop_range > 1 and level == level_loop_range - 2:
							perm_results_alist.append([] if match_gens_list[imatch] == -1
													  else [step_results[match_gens_list[imatch]]])
			# end loop of one_comb over all_combs
		# end loop over all step phrases alternatives

		if level == level_loop_range - 1:
			pcvo_blist = pcvo_alist
			perm_preconds_blist = perm_preconds_alist
			perm_phrases_blist = perm_phrases_alist
			perm_gens_blist = perm_gens_alist
		else:
			step_phrases_list = copy.deepcopy(perm_phrases_alist)
			if level_loop_range > 1 and level == level_loop_range - 2:
				step_results_list =  copy.deepcopy(perm_results_alist)
	# end of level loop
	for iperm, _ in enumerate(pcvo_blist):
		process_one_perm(perm_gens_blist, iperm, event_step_id, perm_preconds_blist, perm_phrases_blist,
						 step_results, pcvo_blist, expected_but_not_found_list, b_blocking, gg_confirmed_list,
						 b_null_results, result_confirmed_list, event_result_score_list,
						 db_len_grps, sess, el_set_arr, glv_dict, def_article_dict)
	if b_null_results:
		# Shouldn't be able to enter the following loop, but we'll get out anyway
		return
	for i_event_result, event_result_score in enumerate(event_result_score_list):
		top_score, i_top_score, top_score_len, top_score_scvo, top_score_igg = -1.0, -1, 1000, '', 0
		participants = set()
		random.shuffle(event_result_score)
		for iresult, one_score in enumerate(event_result_score):
			print(('Blocking' if b_blocking else 'Normal'), 'event result:', one_score)
			templ_score, templ_len, templ_scvo, templ_igg = one_score
			participants.add((templ_len, templ_scvo, templ_igg))
			# if templ_len < top_score_len:
			if templ_score > top_score or (templ_score == top_score and templ_len < top_score_len):
				top_score, i_top_score, top_score_len, top_score_scvo, top_score_igg = \
					templ_score, iresult, templ_len, templ_scvo, templ_igg

	if i_top_score > -1:
		for igrp, len_grp in enumerate(db_len_grps):
			if len_grp.len() == top_score_len:
				print('Successful match using templ len, scvo, igg, score:', top_score_len, top_score_scvo,
					  top_score_igg, top_score, 'For event result', step_results[i_event_result])
				top_score_templ = len_grp.find_templ(top_score_scvo)
				top_score_templ.add_point(top_score_igg)
				break

		winner = (top_score_len, top_score_scvo, top_score_igg)
		for one_particp in participants:
			for igrp, len_grp in enumerate(db_len_grps):
				if len_grp.len() == one_particp[0]:
					particp_templ = len_grp.find_templ(one_particp[1])
					particp_templ.apply_penalty(one_particp[2], -5 if one_particp == winner else 1)
					break

	return

