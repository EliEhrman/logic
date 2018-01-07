from __future__ import print_function
import sys
import math
import time
import numpy as np
import random
import itertools

import config
import rules
import story
import cascade
from rules import conn_type
import els
import dmlearn
import ykmeans
import utils
import makerecs as mr
import clrecgrp
from clrecgrp import cl_gens_grp
from clrecgrp import cl_templ_grp
from clrecgrp import cl_len_grp


def create_train_vecs(glv_dict, def_article_dict,
					  num_stories, num_story_steps, b_for_query, ivec_dim_dict_fixed = None):
	start_rule_names = ['objects_start', 'people_start']
	event_rule_names = ['pickup_rule', 'went_rule']
	state_from_event_names = ['gen_rule_picked_up', 'gen_rule_picked_up_free', 'gen_rule_went', 'gen_rule_has_and_went',
							  'gen_rule_knows_dynamic_action']

	cl_templ_grp.glv_dict = glv_dict
	cl_templ_grp.glv_len = len(glv_dict[config.sample_el])
	sess = dmlearn.init_templ_learn()

	train_rules = []
	event_results = []
	event_step_id = -1
	event_result_id_arr = []
	db_len_grps = []
	el_set_arr = []

	for i_one_story in range(num_stories):
		els_arr, els_dict, def_article, num_els, els_sets = els.init_objects()
		# els.quality_of_els_sets(glv_dict, els_sets)
		all_rules = rules.init_all_rules(els_sets, els_dict)
		rule_dict = {rule.name: rule for rule in all_rules}

		rule_select = lambda type, ruleset: filter(lambda one_rule: one_rule.type == type, ruleset)
		block_event_rules = rule_select(rules.rule_type.block_event, all_rules)

		start_story_rules = [rule_dict[rule_name] for rule_name in start_rule_names]
		event_rules = [rule_dict[rule_name] for rule_name in event_rule_names]
		state_from_event_rules = [rule_dict[rule_name] for rule_name in state_from_event_names]
		story_db = []
		for rule in start_story_rules:
			src_recs, recs = rules.gen_for_rule(b_gen_for_learn=True, rule=rule)

			for rec in recs:
				phrase = rec.phrase()
				if phrase[0][1] == conn_type.start:
					if phrase[1][1] == conn_type.Insert:
						story_db.append(rules.C_phrase_rec(phrase[2:-1]))

		print('Current state of story DB')
		for phrase_rec in story_db:
			out_str = ''
			out_str = els.output_phrase(def_article_dict, out_str, phrase_rec.phrase())
			print(out_str)

		total_event_rule_prob = sum(one_rule.prob for one_rule in event_rules)

		localtime = time.asctime(time.localtime(time.time()))
		print("Local current time :", localtime)
		event_queue = []
		for i_story_step in range(num_story_steps):
			event_phrase, event_queue, b_person_to_person_ask = \
				story.create_story_event(els_dict, els_arr, def_article_dict, story_db,
								   total_event_rule_prob, event_queue,
								   i_story_step, event_rules=event_rules,
								   block_event_rules=block_event_rules)

			if not event_phrase:
				continue

			_, events_to_queue =  story.infer_from_story(	els_dict, els_arr, def_article_dict, story_db, b_apply_results=False,
															story_step=event_phrase, step_effect_rules=state_from_event_rules,
															b_remove_mod_hdr=False)

			event_result_score_list = []
			result_confirmed_list = [False for _ in events_to_queue]
			gg_confirmed_list = [[] for _ in events_to_queue]

			for event_result in events_to_queue:
				print('Evaluating the following result:')
				out_str = ''
				out_str = els.print_phrase(event_result, event_result, out_str,
										   def_article_dict)
				print(out_str)
				event_result_score_list.append([])

			event_step_id += 1

			# all_combs = cascade.get_cascade_combs(els_sets, story_db, event_phrase)
			all_combs = cascade.get_phrase_cascade(els_sets, story_db, event_phrase)

			all_combs = sorted(all_combs, key=len)

			for one_comb in all_combs:
				if all(result_confirmed_list):
					break

				all_perms = list(itertools.permutations(one_comb, len(one_comb)))

				rule_base = [event_phrase]
				pcvo_list = []
				# gcvo_list = []
				perm_preconds_list = []
				perm_gens_list = []
				for one_perm in all_perms:
					new_conds, vars_dict = mr.make_preconds_rule_from_phrases(rule_base, one_perm, story_db)
					gens_recs_arr = []
					for event_result in events_to_queue:
						new_gens = mr.make_gens_rule_from_phrases(vars_dict, event_result)
						new_rule = rules.nt_rule(preconds=new_conds, gens=new_gens)
						# overwrite the preconds_rec because it should always be identical in this inner loop
						preconds_recs, gens_recs = \
							rules.gen_for_rule(b_gen_for_learn=True, rule=new_rule)
						gens_recs_arr.append(gens_recs[0].phrase())
					# take first el because the result is put into a list
					perm_preconds_list.append(preconds_recs[0].phrase())
					# perm_gens_list.append(gens_recs[0].phrase())
					perm_gens_list.append(gens_recs_arr)
					pcvo_list.append(mr.gen_cvo_str(preconds_recs[0].phrase()))

				# end of one_perm in all perms
				# The following cuts down the number of perms by trying to use only the lexically first
				# member. After all there is no real difference to the order of the perm it's just that for a
				# particular comb we want everybody to agree on the order to cut down equivalent rules.
				# However, there is no guarantee that there will always be only one first
				pstr_min = min(pcvo_list)
				plist_min = [iperm for iperm, scvo in enumerate(pcvo_list) if scvo <= pstr_min ]
				for iperm in plist_min:
					if all(result_confirmed_list):
						break
					comb_len = len(perm_preconds_list[iperm])
					print('Evaluating the following perm:')
					out_str = ''
					out_str = els.print_phrase(perm_preconds_list[iperm], perm_preconds_list[iperm], out_str, def_article_dict)
					print(out_str)
					for one_perm_gens in perm_gens_list[iperm]:
						out_str = ''
						out_str = els.print_phrase(one_perm_gens, one_perm_gens, out_str, def_article_dict)
						print(out_str)

					i_len_grp = -1
					for igrp, len_grp in enumerate(db_len_grps):
						if len_grp.len() == comb_len:
							i_len_grp = igrp
							perm_templ = len_grp.find_templ(pcvo_list[iperm])
							if not perm_templ:
								# gg = cl_gens_grp(gens_rec=perm_gens_list[iperm], preconds_rec=perm_preconds_list[iperm])
								len_grp.add_templ(cl_templ_grp(comb_len, pcvo_list[iperm],
															   preconds_rec=perm_preconds_list[iperm],
															   gens_rec_list=perm_gens_list[iperm],
															   event_result_list=events_to_queue,
															   eid=event_step_id))
							else:
								num_pggs = perm_templ.get_num_pggs()
								if num_pggs >= 2:
									print('Getting score for multi-gg templ.')
									perm_templ.printout(def_article_dict)
									event_result_score_list = perm_templ.get_match_score(perm_preconds_list[iperm],
																						 events_to_queue,
																						 event_result_score_list,
																						 result_confirmed_list,
																						 gg_confirmed_list)

									if all(result_confirmed_list):
										print('All results match confirmed ggs')
										mr.report_confirmers(db_len_grps, gg_confirmed_list, el_set_arr,
															 def_article_dict, glv_dict)
										break
									# if score >= 1.0:
									# 	print('match perfect!')
										# comb_len_passed = comb_len

								perm_templ.add_perm(preconds_rec=perm_preconds_list[iperm],
													gens_rec_list=perm_gens_list[iperm],
													perm_result_list=events_to_queue,
													eid=event_step_id)

								print('Doing learning.')
								perm_templ.printout(def_article_dict)
								perm_templ.do_learn(sess, el_set_arr)
								# score = perm_templ.get_match_score(perm_preconds_list[iperm], events_to_queue, event_result_score_list, b_real_score=False)

							break
						#end if len matches
					# end loop over len groups
					if i_len_grp == -1:
						db_len_grps.append(cl_len_grp(comb_len, pcvo_list[iperm], perm_preconds_list[iperm],
													  perm_gens_list[iperm], events_to_queue, event_step_id))
				# end of one perm in all_perms
			#end of one_comb in all_combs
			for i_event_result, event_result_score in enumerate(event_result_score_list):
				top_score, i_top_score, top_score_len, top_score_scvo, top_score_igg  = -1.0, -1, 1000, '', 0
				for iresult, one_score in enumerate(event_result_score):
					templ_score, templ_len, templ_scvo, templ_igg = one_score
					# if templ_len < top_score_len:
					if templ_score > top_score or (templ_score == top_score and templ_len < top_score_len):
						top_score, i_top_score, top_score_len, top_score_scvo, top_score_igg = \
							templ_score, iresult, templ_len, templ_scvo, templ_igg

				if i_top_score > -1:
					for igrp, len_grp in enumerate(db_len_grps):
						if len_grp.len() == top_score_len:
							print('Successful match using templ len, scvo, igg:', top_score_len, top_score_scvo,
								  top_score_igg, 'For event result', events_to_queue[i_event_result])
							top_score_templ = len_grp.find_templ(top_score_scvo)
							top_score_templ.add_point(top_score_igg)
							break


			for event_result in events_to_queue:
				story_db = rules.apply_mods(story_db, [rules.C_phrase_rec(event_result)], i_story_step)
			# end of one event_result
		# end of i_one_step loop
	# end of num stories loop

	# for rule in train_rules:
	# 	out_str = 'rule print: \n'
	# 	out_str = rules.print_rule(rule, out_str)
	# 	print(out_str)


def do_learn(glv_dict, def_article_dict):
	# input_flds_arr, output_flds_arr, fld_def_arr, \
	# input_db, output_db, ivec_pos_list, ovec, ivec_arr_db, ivec_dim_dict_db, ivec_dim_by_rule, \
	# dict_id_list_of_lists, _, _ = \
	create_train_vecs(glv_dict, def_article_dict,
					  config.c_curriculum_num_stories, config.c_curriculum_story_len, b_for_query=False)
	print('Done!')
	exit(1)

