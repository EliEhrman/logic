from __future__ import print_function
import random

import config
import rules
from rules import conn_type
import els


def create_story(els_dict, def_article, els_arr, story_rules, query_rules, gen_rules):
	story = []
	story_db = []

	b_gen_for_learn = False

	for igen, rule in enumerate(story_rules):
		if not rule.story_based:
			src_recs, recs = rules.gen_for_rule(b_gen_for_learn, rule)

			# The first value in the list is the story mod. It is in int format, so translate to enum value
			# (which is 1-based index)
			# For now, we only support story generation of simple one-branch tree
			for rec in recs:
				if rec[0][1] == conn_type.start:
					if rec[1][1] == conn_type.Insert:
						story_db.append(rec[2:-1])
				else:
					print('Error! for now, story insertions may only be simple single-branch phrases stripped of start/end')
					exit()

	print('Current state of story DB')
	for phrase in story_db:
		out_str = ''
		out_str = els.output_phrase(def_article, els_dict, out_str, phrase)
		print(out_str)

	story_based_rules = []
	for story_rule in story_rules:
		if story_rule.story_based:
			story_based_rules.append(story_rule)

	story_steps = []
	for i_story_step in range(config.c_story_len):
		story_rule = random.choice(story_based_rules)
		src_recs, recs = rules.gen_from_story(els_dict, els_arr, story_rule, story_db)
		if not recs:
			continue
		new_phrase = recs[0][2:-1]
		story_steps.append(new_phrase) # the right way to do this is to see if the insert cmd was present in recs
		out_str = 'Next story step: *** '
		out_str = els.output_phrase(def_article, els_dict, out_str, new_phrase)
		out_str += ' **** '
		print(out_str)
		for igen, gen_rule in enumerate(gen_rules):
			story_with_step = story_db + [new_phrase]
			src_recs, recs = rules.gen_from_story(els_dict, els_arr, gen_rule, story_with_step, gen_by_last=True)
			if not recs:
				continue
			mod_phrase = recs[0][1:-1]
			story_db = rules.apply_mods(story_db, [mod_phrase])

	# src_recs, recs = rules.gen_for_rule(b_gen_for_learn, query_rules[0])
	# sel_query_src = random.choice(src_recs)
	# out_str = 'Query: '
	# out_str = els.print_phrase(sel_query_src, sel_query_src, out_str, def_article, els_dict)
	# print(out_str)
	# sel_query = rules.instatiate_query(query_rules[0], sel_query_src)
	# src_recs, recs = rules.gen_from_story(els_dict, els_arr, sel_query, story_db)

	print('-------- New state of story DB -----')
	for phrase in story_db:
		out_str = ''
		out_str = els.output_phrase(def_article, els_dict, out_str, phrase)
		print(out_str)

	for query_rule in query_rules:
		query_gen = rules.extract_query_gen(query_rule)
		src_recs, recs = rules.gen_for_rule(b_gen_for_learn, query_gen)

		for iiquery in range(15):
			sel_query_src = (random.choice(recs))[1:-1] # remove start-stop
			out_str = 'Query: '
			out_str = els.print_phrase(sel_query_src, sel_query_src, out_str, def_article, els_dict)
			print(out_str + '?')
			ans_recs = []
			gen_src_recs, gen_recs = rules.gen_from_story(els_dict, els_arr, query_rule,
														  story_db+[sel_query_src], gen_by_last=True, multi_ans=True)
			ans_recs += gen_recs
			gen_src_recs, gen_recs = rules.gen_from_story(els_dict, els_arr, query_rule,
														  story_steps+[sel_query_src], gen_by_last=True, multi_ans=True)
			ans_recs += gen_recs
			if ans_recs:
				for ans_rec in ans_recs:
					out_str = 'Answer: '
					out_str = els.output_phrase(def_article, els_dict, out_str, ans_rec[2:-1])
					print(out_str)
					out_str = []
			else:
				print('No answer')

		# end loop to iterate random generated queries
	# end for query_rule over query_rules that gen queries (ie where's the hat, where's the tie ...)

	return story
