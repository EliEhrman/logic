from __future__ import print_function
import numpy as np
import random

import config
import rules
import story
import cascade
from rules import conn_type
import els

def pad_ovec(vecs):
	numrecs = vecs.shape[0]
	pad_len = config.c_ovec_len - vecs.shape[1]
	vecs = np.concatenate((vecs, np.zeros((numrecs, pad_len), dtype=np.int32,)), axis=1)

	return vecs

# A gen rule is how to generate records
# A fld or fld_def (input_flds or output_flds) defines a record of fields for manipulating and learning records
def gen_trainset(gen_rules, els_dict, glv_dict, max_phrases_per_rule):
	global logger

	input_flds_arr = []
	output_flds_arr = []
	fld_def_arr = []
	ivec_arr = []
	ivec_pos_list = []
	ivec_dim_dict = {}
	ivec_dim_by_rule = []

	input = []
	output = []
	input_unsorted = []
	output_unsorted = []
	ivec_list_unsorted = []
	ovec_list_unsorted = []
	igen_list_by_dict_id = []
	for igen, i_and_o_rule in enumerate(gen_rules):
		src_recs, recs = \
			rules.gen_for_rule(b_gen_for_learn=True, rule=i_and_o_rule)
		# if the following assert fails all the subsequent code must be rewritten to account
		# for multiple records per gen_rule
		assert len(recs) == 1, 'expected rule to generate one and only one rec'
		# ivec = make_vec(src_recs, i_and_o_rule.preconds, els_arr)
		ivec = els.make_vec(src_recs, glv_dict)
		fld_def_arr.extend([igen] * len(src_recs))
		# len_so_far = len(ivec_arr)
		ivec_dim = ivec.shape[1]
		dict_id = ivec_dim_dict.get(ivec_dim, None)
		if dict_id == None:
			dict_id = len(ivec_dim_dict)
			ivec_dim_dict[ivec_dim] = dict_id
			igen_list_by_dict_id.append([igen])
		else:
			igen_list_by_dict_id[dict_id].append(igen)
		input_unsorted += src_recs
		ivec_list_unsorted.append(ivec)
		del dict_id, ivec, src_recs

		# ovec = make_vec(recs, i_and_o_rule.gens, els_arr)
		ovec = els.make_vec(recs, glv_dict)
		ovec = pad_ovec(ovec)
		output_unsorted += recs
		ovec_list_unsorted += [ovec]
		del recs

	for dict_id, dict_id_list in enumerate(igen_list_by_dict_id):
		for one_igen in dict_id_list:
			input += [input_unsorted[one_igen]]
			output += [output_unsorted[one_igen]]
			ivec_arr.append(ivec_list_unsorted[one_igen])

			if len(ivec_arr) == 1:
				# all_ivecs = ivec
				all_ovecs = ovec_list_unsorted[one_igen]
			else:
				# all_ivecs = np.concatenate((all_ivecs, ivec), axis=0)
				all_ovecs = np.concatenate((all_ovecs, ovec_list_unsorted[one_igen]), axis=0)

			input_flds_arr.append(gen_rules[one_igen].preconds)
			output_flds_arr.append(gen_rules[one_igen].gens)
			# for now the following commented-put line is replaced by a line identical to ivec_dim_by_rule
			# For multiple recs per rule, return to commented version but check all implications first
			# ivec_pos_list.extend([dict_id for i in recs])
			ivec_pos_list.extend([dict_id])
			ivec_dim_by_rule.append(dict_id)

	# end loop over gen rules

	return 	input_flds_arr, output_flds_arr, fld_def_arr, input, output, \
			ivec_pos_list, all_ovecs, ivec_arr, ivec_dim_dict, ivec_dim_by_rule


def do_learn(els_sets, els_dict, glv_dict, def_article, els_arr, all_rules):
	start_rule_names = ['objects_start', 'people_start']
	event_rule_names = ['pickup_rule']
	state_from_event_names = ['gen_rule_picked_up']
	rule_dict = {rule.name: rule for rule in all_rules}

	rule_select = lambda type, ruleset:  filter(lambda one_rule: one_rule.type == type, ruleset)
	block_event_rules = rule_select(rules.rule_type.block_event, all_rules)

	start_story_rules = [rule_dict[rule_name] for rule_name in start_rule_names]
	event_rules = [rule_dict[rule_name] for rule_name in event_rule_names]
	state_from_event_rules = [rule_dict[rule_name] for rule_name in state_from_event_names]

	train_rules = []
	for i_one_story in range(config.c_curriculum_num_stories):
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
			out_str = els.output_phrase(def_article, els_dict, out_str, phrase_rec.phrase())
			print(out_str)

		total_event_rule_prob = sum(one_rule.prob for one_rule in event_rules)

		event_queue = []
		for i_story_step in range(config.c_curriculum_story_len):
			event_phrase, event_queue, b_person_to_person_ask = \
				story.create_story_event(els_dict, els_arr, def_article, story_db,
								   total_event_rule_prob, event_queue,
								   i_story_step, event_rules=event_rules,
								   block_event_rules=block_event_rules)

			if not event_phrase:
				continue

			_, events_to_queue =  story.infer_from_story(	els_dict, els_arr, def_article, story_db, b_apply_results=False,
															story_step=event_phrase, step_effect_rules=state_from_event_rules,
															b_remove_mod_hdr=False)

			for event_result in events_to_queue:
				all_perms = cascade.get_obj_cascade(els_sets, event_result[1:], story_db, event_phrase)

				rule_base = [event_phrase]
				for one_perm in all_perms:
					rule_cand = list(rule_base)
					for iphrase in one_perm:
						rule_cand += [story_db[iphrase].phrase()]

					branches = []
					vars_dict = {}
					field_id = 0
					for one_phrase in rule_cand:
						rule_fields = []
						for el in one_phrase:
							word = el[1]
							var_field_id = vars_dict.get(word, -1)
							if var_field_id == -1:
								vars_dict[word] = field_id
								rule_fields.append(rules.nt_rule_fld(els_set=[], df_type=rules.df_type.obj, sel_el=word))
							else:
								rule_fields.append(rules.nt_rule_fld(els_set=[], df_type=rules.df_type.var, var_id=var_field_id))
							field_id += 1
						branches.append(rules.nt_tree_junct(single=rule_fields))
					new_conds = rules.nt_tree_junct(branches=branches, logic=conn_type.AND)

					rule_fields = []
					for el in event_result[1:]:
						word = el[1]
						var_field_id = vars_dict.get(word, -1)
						if var_field_id == -1:
							rule_fields.append(rules.nt_rule_fld(els_set=[], df_type=rules.df_type.obj, sel_el=word))
						else:
							rule_fields.append(rules.nt_rule_fld(els_set=[], df_type=rules.df_type.var, var_id=var_field_id))
					new_gens = rules.nt_tree_junct(single = rule_fields)
					new_rule = rules.nt_rule(preconds=new_conds, gens=new_gens)
					train_rules += [new_rule]
				# end of one_perm
			# end of one event_result
			story_db = rules.apply_mods(story_db, [rules.C_phrase_rec(event_result)], i_story_step)
		# end of i_one_step loop
	# end of num stories loop

	for rule in train_rules:
		out_str = 'rule print: \n'
		out_str = rules.print_rule(rule, out_str)
		print(out_str)
	return gen_trainset(train_rules, els_dict=els_dict, glv_dict=glv_dict, max_phrases_per_rule=config.c_max_phrases_per_rule)

	# print ('done')
