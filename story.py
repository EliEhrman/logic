from __future__ import print_function
import random

import config
import rules
from rules import conn_type
import els


def infer_from_story(els_dict, els_arr, def_article, story_db, b_static_rules=False, b_apply_results=True,
					 story_step=None, step_time=-1, step_effect_rules=None):
	b_require_last = not b_static_rules
	unapplied_results = []
	for igen, gen_rule in enumerate(step_effect_rules):
		if gen_rule.prob < 1.0:
			rand_param = random.uniform(0.0, 1.0)
			if rand_param > gen_rule.prob:
				continue
		if b_static_rules:
			story_to_process = story_db
		else:
			story_to_process = story_db + [rules.C_phrase_rec(story_step, step_time)]
		src_recs, recs = rules.gen_from_story(els_dict, els_arr, gen_rule, story_to_process,
											  gen_by_last=b_require_last, multi_ans=True)
		if not recs:
			continue
		for phrase_rec in recs:
			mod_phrase = (phrase_rec.phrase())[1:-1]
			# out_str = els.output_phrase(def_article, els_dict, out_str, mod_phrase)
			out_str = ''
			out_str = els.print_phrase(src_recs, mod_phrase, out_str, def_article, els_dict)
			print(out_str)
			if b_apply_results:
				story_db = rules.apply_mods(story_db, [rules.C_phrase_rec(mod_phrase)], step_time)
			else:
				unapplied_results += [mod_phrase[1:]]
	return story_db, unapplied_results


def process_p2p_event(els_sets, els_dict, els_arr, query_rules, story_db, sel_query_src):
	ans_list = []
	events_for_queue = []
	asker = sel_query_src[0][1]
	askee = sel_query_src[2][1]
	asked_query = sel_query_src[3:]
	knowledge_query_rules = rules.init_knowledge_query_rules(els_sets, els_dict, askee)

	for query_rule in knowledge_query_rules:
		knowledge_query_gen = rules.extract_query_gen(query_rule)
		_, knowledge_query_recs = rules.gen_for_rule(b_gen_for_learn=False, rule=knowledge_query_gen)

		for one_knowledge_query in knowledge_query_recs:
			sel_knowledge_query = (one_knowledge_query.phrase())[1:-1]  # remove start-stop
			# out_str = 'Knowledge extract query: '
			# out_str = els.print_phrase(sel_knowledge_query, sel_knowledge_query, out_str, def_article, els_dict)
			# print(out_str + '?')
			_, personal_db_recs = rules.gen_from_story(els_dict, els_arr, query_rule,
													   story_db + [rules.C_phrase_rec(sel_knowledge_query)],
													   gen_by_last=True, multi_ans=True)
			# build a db of personal knowledge for one person. To look like story_db, strip start and end
			personal_db = [rules.C_phrase_rec((personal_db_rec.phrase())[1:-1], personal_db_rec.time) for
						   personal_db_rec in personal_db_recs]
			for query_rule in query_rules:
				_, query_personal_recs = rules.gen_from_story(els_dict, els_arr, query_rule,
															  personal_db + [rules.C_phrase_rec(asked_query)],
															  gen_by_last=True, multi_ans=True)
				if query_personal_recs:
					for ans_rec in query_personal_recs:
						ans_phrase = ans_rec.phrase()
						# out_str = 'Answer: '
						# out_str = els.output_phrase(def_article, els_dict, out_str, ans_phrase[2:-1])
						# print(out_str)
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
		final_ans_phrase_objs = [askee, 'told', asker]
		final_ans_phrase = [[rules.rec_def_type.obj, obj] for obj in final_ans_phrase_objs]
		# final_ans_phrase_rec = rules.C_phrase_rec(final_ans_phrase + final_ans.phrase()[2:-1], latest_ans_time)
		final_ans_phrase_rec = final_ans_phrase + final_ans.phrase()[2:-1]
		events_for_queue += [final_ans_phrase_rec]
		# out_str = 'Response: '
		# out_str = els.output_phrase(def_article, els_dict, out_str, final_ans_phrase_rec)
		# print(out_str)
		# infer_from_story(story_db, b_static_rules=False, story_step=final_ans_phrase_rec, step_time=latest_ans_time,
		# 				 step_effect_rules=large_rules)
	else:
		print('No answer')

	return events_for_queue

def do_queries(els_dict, els_arr, def_article, ask_who, specific_db, story_steps, query_rules):
	for query_rule in query_rules:
		query_gen = rules.extract_query_gen(query_rule)
		src_recs, query_recs = rules.gen_for_rule(b_gen_for_learn=False, rule=query_gen)

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

def create_story_event(els_dict, els_arr, def_article, story_db, total_event_rule_prob,
					   event_queue, i_story_step, event_rules, block_event_rules):

	b_person_to_person_ask = False
	if not event_queue:
		# two step instead of random.choice for debugging purposes
		rule_rand_param = random.uniform(0.0, total_event_rule_prob)
		upto = 0
		for i_story_rule, one_rule in enumerate(event_rules):
			if upto + one_rule.prob >= rule_rand_param:
				break
			upto += one_rule.prob
		else:
			print('Should not get here!')
			exit()

		# i_story_rule = random.randint(0, len(event_from_none_rules)-1)
		story_rule = event_rules[i_story_rule]
		if story_rule.name in rules.person_to_person_ask_rule_names:
			b_person_to_person_ask = True

		_, recs = rules.gen_from_story(els_dict, els_arr, story_rule, story_db)
		if not recs:
			return [], [], b_person_to_person_ask
		event_phrase = (recs[0].phrase())[2:-1]
	else:
		event_phrase = event_queue.pop(0)
	# story_rule = random.choice(event_from_none_rules)
	b_blocked = True
	for iblock, block_rule in enumerate(block_event_rules):
		story_with_step = story_db + [rules.C_phrase_rec(event_phrase)]
		src_recs, recs = rules.gen_from_story(els_dict, els_arr, block_rule, story_with_step, gen_by_last=True)
		if recs:
			break
	else:
		b_blocked = False

	if b_blocked:
		# out_str = 'New step blocked by the following rule: '
		# out_str = rules.print_rule(block_rule, out_str)
		# # out_str  = els.print_phrase(block_rule, block_rule, out_str, def_article, els_dict)
		# print(out_str)
		if block_rule.name == 'want_dont_give_block_rule':
			out_str = 'selfish block on:'
			out_str = els.output_phrase(def_article, els_dict, out_str, event_phrase)
			print(out_str)
		return [], event_queue, b_person_to_person_ask

	out_str = 'time:' + str(i_story_step) + '. Next story step: *** '
	out_str = els.output_phrase(def_article, els_dict, out_str, event_phrase)
	out_str += ' **** '
	print(out_str)

	return event_phrase, event_queue, b_person_to_person_ask

def create_story(els_sets, els_dict, def_article, els_arr, all_rules):
	story = []
	story_db = []

	b_gen_for_learn = False

	rule_select = lambda type, ruleset:  filter(lambda one_rule: one_rule.type == type, ruleset)
	# event_from_none_rules = filter(lambda one_rule: one_rule.type == rules.rule_type.event_from_none, story_rules)
	event_from_none_rules = rule_select(rules.rule_type.event_from_none, all_rules)
	event_from_event_rules = rule_select(rules.rule_type.event_from_event, all_rules)
	state_from_state_rules = rule_select(rules.rule_type.state_from_state, all_rules)
	state_from_event_rules = rule_select(rules.rule_type.state_from_event, all_rules)
	# comprehension based alternative: event_from_none_rules = [one_rule for one_rule in story_rules if one_rule.type == rules.rule_type.event_from_none]
	block_event_rules = rule_select(rules.rule_type.block_event, all_rules)
	story_start_rules = rule_select(rules.rule_type.story_start, all_rules)
	query_rules = rule_select(rules.rule_type.query, all_rules)
	for igen, rule in enumerate(story_start_rules):
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

	total_event_rule_prob = sum(one_rule.prob for one_rule in event_from_none_rules)

	story_steps = []
	event_queue = [] # queue of events in respons to other events
	for i_story_step in range(config.c_story_len):
		if not event_queue and random.uniform(0.0, 1.0) > 0.95:
			do_queries(els_dict, els_arr, def_article, ask_who=None, specific_db=story_db, story_steps=story_steps,
					   query_rules=query_rules)

		event_phrase, event_queue, b_person_to_person_ask =\
			create_story_event(els_dict, els_arr, def_article, story_db,
								total_event_rule_prob, event_queue,
								i_story_step, event_rules=event_from_none_rules,
								block_event_rules=block_event_rules)

		if not event_phrase:
			continue

		story_steps.append(rules.C_phrase_rec(init_phrase = event_phrase, init_time=i_story_step))

		if b_person_to_person_ask:
			event_queue += process_p2p_event(els_sets, els_dict, els_arr, query_rules, story_db, sel_query_src=event_phrase)

		story_db, _ = infer_from_story(	els_dict, els_arr, def_article, story_db, story_step=event_phrase,
										step_time=i_story_step, step_effect_rules=state_from_event_rules)
		# Process the rules again to update steady state knowledge. Don't use the event itself
		# and don't require the gen_by_last
		story_db, _ = infer_from_story(	els_dict, els_arr, def_article, story_db, b_static_rules=True,
										step_time=i_story_step, step_effect_rules=state_from_state_rules)
		# find events that will happen as a result of this rule
		_, events_to_queue =  infer_from_story(	els_dict, els_arr, def_article, story_db, b_apply_results=False,
												story_step=event_phrase, step_effect_rules=event_from_event_rules)
		event_queue += events_to_queue

	# end loop over story event steps times choosing random rules from event_from_none rules

	print('-------- New state of story DB -----')
	for phrase_rec in story_db:
		phrase = phrase_rec.phrase()
		out_str = ''
		out_str = els.output_phrase(def_article, els_dict, out_str, phrase)
		print(out_str)


	# do_queries(ask_who=None, specific_db=story_db)


	return story
