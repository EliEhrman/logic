from __future__ import print_function
import random

import config
import rules
from rules import conn_type
import els


def create_story(els_sets, els_dict, def_article, els_arr, story_rules, query_rules, gen_rules,
				 blocking_rules):
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
				phrase = rec.phrase()
				if phrase[0][1] == conn_type.start:
					if phrase[1][1] == conn_type.Insert:
						story_db.append(rules.C_phrase_rec(phrase[2:-1]))
				else:
					print('Error! for now, story insertions may only be simple single-branch phrases stripped of start/end')
					exit()

	print('Current state of story DB')
	for phrase_rec in story_db:
		out_str = ''
		out_str = els.output_phrase(def_article, els_dict, out_str, phrase_rec.phrase())
		print(out_str)

	story_based_rules = []
	for story_rule in story_rules:
		if story_rule.story_based:
			story_based_rules.append(story_rule)

	story_steps = []
	for i_story_step in range(config.c_story_len):
		# two step instead of random.choice for debugging purposes
		i_story_rule = random.randint(0, len(story_based_rules)-1)
		story_rule = story_based_rules[i_story_rule]
		# story_rule = random.choice(story_based_rules)
		src_recs, recs = rules.gen_from_story(els_dict, els_arr, story_rule, story_db)
		if not recs:
			continue
		new_phrase = (recs[0].phrase())[2:-1]
		out_str = 'time:' + str(i_story_step) + '. Next story step: *** '
		out_str = els.output_phrase(def_article, els_dict, out_str, new_phrase)
		out_str += ' **** '
		print(out_str)
		b_blocked = False
		for iblock, block_rule in enumerate(blocking_rules):
			story_with_step = story_db + [rules.C_phrase_rec(new_phrase)]
			src_recs, recs = rules.gen_from_story(els_dict, els_arr, block_rule, story_with_step, gen_by_last=True)
			if recs:
				b_blocked = True
				break
		if b_blocked:
			out_str = 'New step blocked by the following rule: '
			out_str = rules.print_rule(block_rule, out_str)
			# out_str  = els.print_phrase(block_rule, block_rule, out_str, def_article, els_dict)
			print(out_str)
			continue

		def infer_from_story(story_db, b_static_rules):
			b_require_last = not b_static_rules
			for igen, gen_rule in enumerate(gen_rules):
				if b_static_rules:
					story_to_process = story_db
				else:
					story_to_process = story_db + [rules.C_phrase_rec(new_phrase, i_story_step)]
				src_recs, recs = rules.gen_from_story(els_dict, els_arr, gen_rule, story_to_process,
													  gen_by_last=b_require_last, multi_ans=True)
				if not recs:
					continue
				for phrase_rec in recs:
					mod_phrase = (phrase_rec.phrase())[1:-1]
					# out_str = els.output_phrase(def_article, els_dict, out_str, mod_phrase)
					out_str = ''
					out_str  = els.print_phrase(src_recs, mod_phrase, out_str, def_article, els_dict)
					print(out_str)
					story_db = rules.apply_mods(story_db, [rules.C_phrase_rec(mod_phrase)], i_story_step)
			return story_db


		story_steps.append(rules.C_phrase_rec(init_phrase = new_phrase, init_time=i_story_step))

		story_db = infer_from_story(story_db, b_static_rules=False)
		# Process the rules again to update steady state knowledge. Don't use the event itself
		# and don't require the gen_by_last
		story_db = infer_from_story(story_db, b_static_rules=True)

		# for igen, gen_rule in enumerate(gen_rules):
		# 	story_with_step = story_db + [rules.C_phrase_rec(new_phrase)]
		# 	src_recs, recs = rules.gen_from_story(els_dict, els_arr, gen_rule, story_with_step,
		# 										  gen_by_last=True, multi_ans=True)
		# 	if not recs:
		# 		continue
		# 	for phrase_rec in recs:
		# 		mod_phrase = (phrase_rec.phrase())[1:-1]
		# 		out_str  = els.print_phrase(src_recs, mod_phrase, out_str, def_article, els_dict)
		# 		print(out_str)
		# 		story_db = rules.apply_mods(story_db, [rules.C_phrase_rec(mod_phrase)], i_story_step)
		#
		# for igen, gen_rule in enumerate(gen_rules):
		# 	src_recs, recs = rules.gen_from_story(els_dict, els_arr, gen_rule, story_db,
		# 										  gen_by_last=False, multi_ans=True)
		# 	if not recs:
		# 		continue
		# 	for phrase_rec in recs:
		# 		mod_phrase = (phrase_rec.phrase())[1:-1]
		# 		out_str = els.print_phrase(src_recs, mod_phrase, out_str, def_article, els_dict)
		# 		print(out_str)
		# 		story_db = rules.apply_mods(story_db, [rules.C_phrase_rec(mod_phrase)], i_story_step)

	# src_recs, recs = rules.gen_for_rule(b_gen_for_learn, query_rules[0])
	# sel_query_src = random.choice(src_recs)
	# out_str = 'Query: '
	# out_str = els.print_phrase(sel_query_src, sel_query_src, out_str, def_article, els_dict)
	# print(out_str)
	# sel_query = rules.instatiate_query(query_rules[0], sel_query_src)
	# src_recs, recs = rules.gen_from_story(els_dict, els_arr, sel_query, story_db)

	print('-------- New state of story DB -----')
	for phrase_rec in story_db:
		phrase = phrase_rec.phrase()
		out_str = ''
		out_str = els.output_phrase(def_article, els_dict, out_str, phrase)
		print(out_str)

	def do_queries(ask_who, specific_db):
		for query_rule in query_rules:
			query_gen = rules.extract_query_gen(query_rule)
			src_recs, query_recs = rules.gen_for_rule(b_gen_for_learn, query_gen)

			for one_query_rec in query_recs:
				if not ask_who:
					one_query_core = (one_query_rec.phrase())[1:-1] # remove start-stop
					out_str = 'Query: '
					out_str = els.print_phrase(one_query_core, one_query_core, out_str, def_article, els_dict)
					print(out_str + '?')
					phrase_db = specific_db + [rules.C_phrase_rec(one_query_core)]
				else:
					phrase_db = specific_db
				ans_recs = []
				gen_src_recs, gen_recs = rules.gen_from_story(els_dict, els_arr, query_rule,
															  phrase_db,
															  gen_by_last=True, multi_ans=True)
				ans_recs += gen_recs
				if not ask_who:
					gen_src_recs, gen_recs = rules.gen_from_story(els_dict, els_arr, query_rule,
																  story_steps+[rules.C_phrase_rec(one_query_core)],
																  gen_by_last=True, multi_ans=True)
					ans_recs += gen_recs

				if ans_recs:
					for ans_rec in ans_recs:
						ans_phrase = ans_rec.phrase()
						out_str = 'Answer: '
						out_str = els.output_phrase(def_article, els_dict, out_str, ans_phrase[2:-1])
						print(out_str)
						out_str = []
				else:
					print('No answer')

			# end loop to iterate random generated queries
		# end for query_rule over query_rules that gen queries (ie where's the hat, where's the tie ...)

	# do_queries(ask_who=None, specific_db=story_db)

	ask_rules = rules.init_ask_rules(els_sets, els_dict)
	ask_blocking_rules = rules.init_ask_blocking_rules(els_sets, els_dict)

	for ask_rule in ask_rules:
		ask_gen = rules.extract_query_gen(ask_rule)
		_, ask_head_recs = rules.gen_for_rule(b_gen_for_learn, ask_gen)

		for one_query in ask_head_recs:
			sel_query_src = (one_query.phrase())[1:-1]  # remove start-stop
			b_blocked = False
			for i_ask_block, ask_block_rule in enumerate(ask_blocking_rules):
				_, ask_block_recs = rules.gen_from_story(els_dict, els_arr, ask_block_rule, [rules.C_phrase_rec(sel_query_src)], gen_by_last=True)
				if ask_block_recs:
					b_blocked = True
					break
			if b_blocked:
				continue
			ask_src_recs, ask_recs = rules.gen_from_story(els_dict, els_arr, ask_rule,
														  story_db + [rules.C_phrase_rec(sel_query_src)],
														  gen_by_last=True, multi_ans=True)
			if ask_recs:
				out_str = 'Query: '
				out_str = els.print_phrase(sel_query_src, sel_query_src, out_str, def_article, els_dict)
				print(out_str + '?')
				ans_list = []
				asker = sel_query_src[0][1]
				askee = sel_query_src[2][1]
				asked_query = sel_query_src[3:]
				knowledge_query_rules = rules.init_knowledge_query_rules(els_sets, els_dict, askee)

				for query_rule in knowledge_query_rules:
					knowledge_query_gen = rules.extract_query_gen(query_rule)
					_, knowledge_query_recs = rules.gen_for_rule(b_gen_for_learn, knowledge_query_gen)

					for one_knowledge_query in knowledge_query_recs:
						sel_knowledge_query = (one_knowledge_query.phrase())[1:-1]  # remove start-stop
						out_str = 'Knowledge extract query: '
						out_str = els.print_phrase(sel_knowledge_query, sel_knowledge_query, out_str, def_article, els_dict)
						print(out_str + '?')
						_, personal_db_recs = rules.gen_from_story(els_dict, els_arr, query_rule,
																	  story_db + [rules.C_phrase_rec(sel_knowledge_query)],
																	  gen_by_last=True, multi_ans=True)
						# build a db of personal knowledge for one person. To look like story_db, strip start and end
						personal_db = [rules.C_phrase_rec((personal_db_rec.phrase())[1:-1], personal_db_rec.time) for personal_db_rec in personal_db_recs]
						for query_rule in query_rules:
							_, query_personal_recs = rules.gen_from_story(els_dict, els_arr, query_rule,
																		  personal_db + [rules.C_phrase_rec(asked_query)],
																		  gen_by_last=True, multi_ans=True)
							if query_personal_recs:
								for ans_rec in query_personal_recs:
									ans_phrase = ans_rec.phrase()
									out_str = 'Answer: '
									out_str = els.output_phrase(def_article, els_dict, out_str, ans_phrase[2:-1])
									print(out_str)
									ans_list += [ans_rec]
						# end loop of application of queries to personal db
					# end loop over knowlege db extraction queries
				# end loop over query from one person to another
				if ans_list:
					latest_ans_time, latest_idx = 0, -1
					for one_ans_idx, one_ans in enumerate(ans_list):
						if latest_ans_time < one_ans.time:
							latest_idx = one_ans_idx
							latest_ans_time = one_ans.time
					final_ans = ans_list[latest_idx]
					final_ans_phrase_objs = [asker, 'told', askee]
					final_ans_phrase = [[rules.rec_def_type.obj, obj] for obj in final_ans_phrase_objs ]
					final_ans_phrase_rec = rules.C_phrase_rec(final_ans_phrase + final_ans.phrase()[2:-1], latest_ans_time)
				else:
					print('No answer')

	return story
