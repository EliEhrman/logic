from __future__ import print_function
import random

import config
import rules
import story
import cascade
from rules import conn_type
import els

def do_learn(els_sets, els_dict, def_article, els_arr, all_rules):
	start_rule_names = ['objects_start', 'people_start']
	event_rule_names = ['pickup_rule']
	state_from_event_names = ['gen_rule_picked_up']
	rule_dict = {rule.name: rule for rule in all_rules}

	rule_select = lambda type, ruleset:  filter(lambda one_rule: one_rule.type == type, ruleset)
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
		out_str = els.output_phrase(def_article, els_dict, out_str, phrase_rec.phrase())
		print(out_str)

	total_event_rule_prob = sum(one_rule.prob for one_rule in event_rules)

	event_queue = []
	for i_story_step in range(config.c_story_len):
		event_phrase, event_queue, b_person_to_person_ask = \
			story.create_story_event(els_dict, els_arr, def_article, story_db,
							   total_event_rule_prob, event_queue,
							   i_story_step, event_rules=event_rules,
							   block_event_rules=block_event_rules)

		if not event_phrase:
			continue

		_, events_to_queue =  story.infer_from_story(	els_dict, els_arr, def_article, story_db, b_apply_results=False,
														story_step=event_phrase, step_effect_rules=state_from_event_rules)

		for event_result in events_to_queue:
			all_perms = cascade.get_obj_cascade(els_sets, event_result, story_db, event_phrase)

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
				for el in event_result:
					word = el[1]
					var_field_id = vars_dict.get(word, -1)
					if var_field_id == -1:
						rule_fields.append(rules.nt_rule_fld(els_set=[], df_type=rules.df_type.obj, sel_el=word))
					else:
						rule_fields.append(rules.nt_rule_fld(els_set=[], df_type=rules.df_type.var, var_id=var_field_id))
				new_gens = rules.nt_tree_junct(single = rule_fields)
				new_rule = rules.nt_rule(preconds=new_conds, gens=new_gens)
			# end of one_perm
		# end of one event_result
	# end of i_one_step
	print ('done')
