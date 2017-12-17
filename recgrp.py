from __future__ import print_function
import sys
import math
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
	state_from_event_names = ['gen_rule_picked_up', 'gen_rule_picked_up_free', 'gen_rule_went', 'gen_rule_has_and_went']

	train_rules = []
	event_results = []
	event_result_id = -1
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

			for event_result in events_to_queue:
				event_result_id += 1
				# all_perms = cascade.get_obj_cascade(els_sets, event_result[1:], story_db, event_phrase, b_for_query=b_for_query)
				all_combs = cascade.get_cascade_combs(els_sets, story_db, event_phrase)

				for one_comb in all_combs:
					all_perms = list(itertools.permutations(one_comb, len(one_comb)))
					# all_perms = [k for j in [list(itertools.permutations(i, len(i))) for i in all_perms] for k in j]

					event_results += [event_result] * len(all_perms)

					rule_base = [event_phrase]
					pcvo_list = []
					gcvo_list = []
					perm_preconds_list = []
					perm_gens_list = []
					for one_perm in all_perms:
						new_rule = mr.make_rule_from_phrases(rule_base, one_perm, story_db, event_result)

						preconds_recs, gens_recs = \
							rules.gen_for_rule(b_gen_for_learn=True, rule=new_rule)
						# take first el because the result is put into a list
						perm_preconds_list.append(preconds_recs[0].phrase())
						perm_gens_list.append(gens_recs[0].phrase())
						pcvo_list.append(mr.gen_cvo_str(preconds_recs[0].phrase()))
						gcvo_list.append(mr.gen_cvo_str(gens_recs[0].phrase()))

					# end of one_perm in all perms
					pstr_min = min(pcvo_list)
					plist_min = [iperm for iperm, scvo in enumerate(pcvo_list) if scvo <= pstr_min ]
					for iperm in plist_min:
						comb_len = len(perm_preconds_list[iperm])
						i_len_grp = -1
						for igrp, len_grp in enumerate(db_len_grps):
							if len_grp.len() == comb_len:
								i_len_grp = igrp
								perm_templ = len_grp.find_templ(pcvo_list[iperm])
								if not perm_templ:
									gg = cl_gens_grp(gens_rec=perm_gens_list[iperm], preconds_rec=perm_preconds_list[iperm])
									len_grp.add_templ(cl_templ_grp(pcvo_list[iperm], gens_grp=gg, preconds_rec=perm_preconds_list[iperm]))
								else:
									perm_gg, perm_igg = perm_templ.find_gg(gens_rec=perm_gens_list[iperm])
									if not perm_gg:
										gg = cl_gens_grp(gens_rec=perm_gens_list[iperm],
														 preconds_rec=perm_preconds_list[iperm])
										perm_igg = perm_templ.add_gg(gg, preconds_rec=perm_preconds_list[iperm])
									else:
										perm_gg.add_perm(preconds_rec=perm_preconds_list[iperm])
										perm_templ.add_perm(preconds_rec=perm_preconds_list[iperm])
									gg_list = perm_templ.get_gg_list()
									for igg, test_gg in enumerate(gg_list):
										generated_result = mr.get_result_for_cvo_and_rec(perm_preconds_list[iperm], perm_gens_list[iperm])
										mrg_list = test_gg.get_mrg_list()
										if mr.match_rec_exact(generated_result, event_result):
											b_matched = False
											for mrg in mrg_list:
												b_this_matched = mrg.test(glv_dict, el_set_arr, perm_preconds_list[iperm])
												if b_this_matched:
													b_matched = True
													# add to score for mrg
											if not b_matched:
												for mrg in mrg_list:
													mrg_perm_list = mrg.get_perm_list()
													new_rule = mr.make_rule_grp(glv_dict, mrg_perm_list, el_set_arr)
										else:
											for mrg in mrg_list:
												b_this_matched = mrg.test(glv_dict, el_set_arr, perm_preconds_list[iperm])
												# if b_this_matched:
													# decrease score for mrg
									break
						if i_len_grp == -1:
							db_len_grps.append(cl_len_grp(comb_len, pcvo_list[iperm], perm_preconds_list[iperm], perm_gens_list[iperm]))

				#end of one_comb in all_combs

				story_db = rules.apply_mods(story_db, [rules.C_phrase_rec(event_result)], i_story_step)
			# end of one event_result
		# end of i_one_step loop
	# end of num stories loop

	# for rule in train_rules:
	# 	out_str = 'rule print: \n'
	# 	out_str = rules.print_rule(rule, out_str)
	# 	print(out_str)


def do_learn(glv_dict, def_article_dict):
	input_flds_arr, output_flds_arr, fld_def_arr, \
	input_db, output_db, ivec_pos_list, ovec, ivec_arr_db, ivec_dim_dict_db, ivec_dim_by_rule, \
	dict_id_list_of_lists, _, _ = \
		create_train_vecs(glv_dict, def_article_dict,
						  config.c_curriculum_num_stories, config.c_curriculum_story_len, b_for_query=False)

