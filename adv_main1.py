"""
This is the an module from which the adv example runs. Like webdip it call the logic
module to learn rules.

However, it has its own dictionary and rules. The first is found in files in subdir adv
and the rules are implemented in adv_rules.py. This creates an oracle for story development
which logic must learn
"""

from __future__ import print_function
import random
import time
from enum import Enum
import importlib

import els
import dmlearn
from clrecgrp import cl_gens_grp
from clrecgrp import cl_templ_grp
from clrecgrp import cl_len_grp
# import wdlearn
# import recgrp
import config
import utils
import rules
from rules import conn_type
import story
import bitvec
import rules2
import mpdb

import adv_config
import adv_learn
import adv_rules

def init_sets(els_lists):
	num_els = 0
	sets = []
	els_arr = []
	els_dict = dict()
	for il, el_list in enumerate(els_lists):
		num_sel = adv_config.set_sel_arr[il]
		if num_sel == -1:
			new_list = el_list
		else:
			new_list = random.sample(el_list, num_sel)
		num_new_els = len(new_list)
		els_arr += new_list
		new_els_range = range(num_els, num_els+num_new_els)
		num_els += num_new_els
		sets += [[new_els_range, num_new_els, new_list]]

	for iel, el_name in enumerate(els_arr):
		els_dict[el_name] = iel

	els_sets = utils.nt_el_sets(names=sets[0], objects=sets[1], places=sets[2], actions=sets[3])

	return els_sets, num_els, els_arr, els_dict


def play(	glv_dict, def_article_dict, cascade_dict, els_lists,
			num_stories, num_story_steps, db_len_grps, db_cont_mgr, i_gg_cont, learn_vars, mpdb_mgr, b_for_query):
	bitvec_mgr = mpdb_mgr.get_bitvec_mgr()
	start_rule_names = ['objects_start', 'people_start', 'people_want_start']  # ['people_start'] #
	event_rule_names = ['pickup_rule', 'went_rule']
	state_from_event_names = ['gen_rule_picked_up', 'gen_rule_picked_up_free', 'gen_rule_went', 'gen_rule_has_and_went',
							  'gen_rule_knows_dynamic_action']
	decide_rule_names = ['pickup_decide_rule']
	event_from_decide_names = ['pickup_rule', 'went_rule']

	gg_cont = db_cont_mgr.get_cont(i_gg_cont)
	# if i_gg_cont < 0:
	# 	gg_cont = None
	# else:
	# 	gg_cont = gg_cont_list[i_gg_cont]

	sess = dmlearn.init_templ_learn()

	train_rules = []
	event_results = []
	event_step_id = [learn_vars[0]]
	event_result_id_arr = []
	# db_len_grps = []
	el_set_arr = []

	for i_one_story in range(num_stories):
		# if i_one_story == 6:
		# 	bitvec_mgr.increase_rule_stages()

		l_player_events = []
		els_sets, num_els, els_arr, els_dict = init_sets(els_lists)
		# els.quality_of_els_sets(glv_dict, els_sets)
		all_rules = adv_rules.init_adv_rules(els_sets, els_dict)
		rule_dict = {rule.name: rule for rule in all_rules}

		rule_select = lambda type, ruleset: filter(lambda one_rule: one_rule.type == type, ruleset)
		block_event_rules = rule_select(rules.rule_type.block_event, all_rules)

		start_story_rules = [rule_dict[rule_name] for rule_name in start_rule_names]
		event_rules = [rule_dict[rule_name] for rule_name in event_rule_names]
		state_from_event_rules = [rule_dict[rule_name] for rule_name in state_from_event_names]
		decide_rules = [rule_dict[rule_name] for rule_name in decide_rule_names]
		event_from_decide_rules = [rule_dict[rule_name] for rule_name in event_from_decide_names]

		e_story_loop_stage = Enum('e_story_loop_stage', 'story_init decision_init decision event state1 complete_state1 state2')
		# e_story_loop_stage = Enum('e_story_loop_stage', 'decision_init decision event state1 complete_state1 state2')

		story_loop_stage = e_story_loop_stage.story_init

		mpdb_mgr.clear_dbs()

		l_story_db_event_refs = []
		story_db = []
		for rule in start_story_rules:
			src_recs, recs = rules.gen_for_rule(b_gen_for_learn=True, rule=rule)

			for rec in recs:
				phrase = rec.phrase()
				if phrase[0][1] == conn_type.start:
					if phrase[1][1] == conn_type.Insert:
						init_phrase = phrase[2:-1]
						story_db.append(rules.C_phrase_rec(init_phrase))
						init_wlist = els.convert_phrase_to_word_list([init_phrase])[0]
						ilen, iphrase = bitvec_mgr.add_phrase(	init_wlist,
																(	i_one_story, e_story_loop_stage.story_init,
																	event_step_id[0]))
						# l_story_db_event_refs.append((ilen, iphrase))
						mpdb_mgr.insert(['main'], (ilen, iphrase))

		print('Current state of story DB')
		for phrase_rec in story_db:
			out_str = ''
			out_str = els.output_phrase(def_article_dict, out_str, phrase_rec.phrase())
			print(out_str)

		db_transfrs =  mpdb_mgr.infer(['main'], (i_one_story, story_loop_stage, event_step_id[0]),
												['state_from_start'])
		for one_transfer in db_transfrs:
			if one_transfer[0][1] == conn_type.Insert:
				db_name = one_transfer[0][2]
				added_wlist = els.convert_phrase_to_word_list([one_transfer[1:]])[0]
				ilen, iphrase = bitvec_mgr.add_phrase(added_wlist,
													  (i_one_story, story_loop_stage, event_step_id[0]))
				mpdb_mgr.insert([db_name], (ilen, iphrase))

		total_event_rule_prob = sum(one_rule.prob for one_rule in event_rules)

		localtime = time.asctime(time.localtime(time.time()))

		mpdb_mgr.show_dbs()

		print("Local current time :", localtime)

		story_names = els_sets[0][2]

		story_loop_stage = e_story_loop_stage.decision_init
		decide_options = []
		player_event = None
		event_result_list = []
		i_story_player = -1
		story_player_name = story_names[i_story_player]

		i_story_step = 0
		c_close_to_inf = 10000
		i_story_loop_stage = -1
		if i_one_story == 0:
			player_decide_rules = adv_rules.init_decide_rules(els_sets, els_dict, story_player_name)
			rule_stats = [[0.0, 0.0] for _ in player_decide_rules]

		while i_story_loop_stage < c_close_to_inf:
			i_story_loop_stage += 1
			if i_story_loop_stage >= c_close_to_inf - 1:
				print('Story loop stage seems stuck in an infinite loop. Next story!')
				i_story_loop_stage = -1
				break
			event_step_id[0] += 1
			if story_loop_stage == e_story_loop_stage.decision_init:
				decide_options = []
				if i_story_player < len(story_names) - 1:
					i_story_player += 1
				else:
					i_story_player = 0
				story_player_name = story_names[i_story_player]
				player_decide_rules = adv_rules.init_decide_rules(els_sets, els_dict, story_player_name)
				ruleid = random.randint(0, len(player_decide_rules)-1)
				# ruleid = random.randint(2, 3)
				# ruleid = 2
				if random.random() < 0.5:
					ruleid = 5
				# print('Change this to return to going to events')
				rule = player_decide_rules[ruleid]
				_, gens_recs = rules.gen_for_rule(b_gen_for_learn=False, rule=rule)
				decide_options += gens_recs
				random.shuffle(decide_options)
				story_loop_stage = e_story_loop_stage.decision
				continue
			elif story_loop_stage == e_story_loop_stage.decision:
				if len(decide_options) == 0:
					story_loop_stage = e_story_loop_stage.decision_init
				else:
					one_decide = (decide_options.pop(0).phrase())[1:-1]
					story_loop_stage = e_story_loop_stage.event
				continue

			elif story_loop_stage == e_story_loop_stage.event:
				# _, event_as_decided = story.infer_from_story(els_dict, els_arr, def_article_dict, story_db,
				# 											 b_apply_results=False,
				# 											 story_step=one_decide,
				# 											 step_effect_rules=event_from_decide_rules,
				# 											 b_remove_mod_hdr=False)
				_, event_as_decided =  mpdb_mgr.run_rule(one_decide, (i_one_story, story_loop_stage, event_step_id[0]),
														'main', ['event_from_decide'])
				if event_as_decided != []:
					out_str = ''
					out_str = els.output_phrase(def_article_dict, out_str, one_decide)
					print(out_str)
					# print(event_as_decided)
					# _, event_blocked = story.infer_from_story(els_dict, els_arr, def_article_dict, story_db,
					# 										  b_apply_results=False,
					# 										  story_step=event_as_decided[0][1:],
					# 										  step_effect_rules=block_event_rules,
					# 										  b_remove_mod_hdr=False)
					# if event_blocked == []:
					player_event = event_as_decided[0]
					out_str = 'time:' + str(i_story_step) + '. Next story step: *** '
					out_str = els.output_phrase(def_article_dict, out_str, player_event[1:])
					out_str += ' **** '
					print(out_str)
					l_player_events.append(els.convert_phrase_to_word_list([player_event[1:]])[0])
					ilen, iphrase = bitvec_mgr.add_phrase(l_player_events[-1], (i_one_story, story_loop_stage, event_step_id[0]))
					# l_story_db_event_refs.append((ilen, iphrase))
					#handle deletes and modifies
					story_loop_stage = e_story_loop_stage.state1
					# else:
					# 	event_as_decided = []
					# 	print('Event blocked!')
					# 	story_loop_stage = e_story_loop_stage.decision
				else:
					story_loop_stage = e_story_loop_stage.decision

				if event_as_decided == []:
					rule_stats[ruleid][0] += 1.0
				else:
					rule_stats[ruleid][1] += 1.0
					print('rule stats:',  rule_stats, 'ruleid:', ruleid, 'rand thresh:', (0.99 * rule_stats[ruleid][0] / (rule_stats[ruleid][0] + rule_stats[ruleid][1])))
				if event_as_decided != [] or (random.random() > (0.99 * rule_stats[ruleid][0] / (rule_stats[ruleid][0] + rule_stats[ruleid][1]))):
					if adv_config.c_b_learn_full_rules:
						adv_learn.do_learn_rule_from_step(	event_as_decided, event_step_id[0], story_db, one_decide, '',
															def_article_dict, db_len_grps, sess,
															el_set_arr, glv_dict, els_sets, cascade_dict,
															gg_cont, db_cont_mgr)
						# unindent the following to go back to rule learning
						mpdb_mgr.learn_rule(one_decide, event_as_decided,
											  (i_one_story, story_loop_stage, event_step_id[0]),
											  'main')


			elif story_loop_stage == e_story_loop_stage.state1:
				# events_to_queue, l_dbs_to_mod = [], []
				# _, events_to_queue = story.infer_from_story(els_dict, els_arr, def_article_dict, story_db,
				# 											b_apply_results=False,
				# 											story_step=player_event[1:],
				# 											step_effect_rules=state_from_event_rules,
				# 											b_remove_mod_hdr=False)
				"""
				Here is the key to writing these rules:
				If you have an event, by default, only the main db knows about it. If there is a state_from_event rule
				then only the main db will be affected by the event. If there is a distr_from_event, then that will
				produce a set of other db's that will now be affected by this knowledge. So any distr will result in the
				state_from_event executing there too.
				If you write a br_state_from_event rule, then the rule is run on the main but state is affected in other
				db's. So if the event has state implications DON"T write a distr_from_event as well, since the same
				state_from_event will now be run on the peripheral db, resulting in the state being created twice.
				Example. gen_has_went from went event. That is distributed ONLY to the guy who went (know I went)
				The consequences of having and going are applied therefore to main and to the guy who went. There is also
				a br_state_from_event rule that will update the db of someone (say, Roy) seeing the guy coming. However,
				if you distribute the went event to Roy, he could be updated twice. So if you must do both, make sure 
				that the addition of state is either unique or else the old data (including the first addtion) must be
				removed.
				Remember, even if a rule is run on main, you can check that a phrase also exists in another db in order to
				work. Just put a var reference on the c:s clause. The pharse must exist in the main as well as in the 
				other db in order for the rule to succeed. 
				Please note. The br_state_from_event path as opposed to distr_from_event wil cause greater difficulty in
				learning. From the player/agent's perspective, there is no event followed by a consequence!    
				"""
				the_main_event = player_event[1:]
				_, events_transfrs =  mpdb_mgr.run_rule(the_main_event, (i_one_story, story_loop_stage, event_step_id[0]),
														'main', ['distr_from_event'])
				l_dbs_to_mod, events_to_queue =  mpdb_mgr.run_rule(	the_main_event,
																	(i_one_story, story_loop_stage, event_step_id[0]),
																	'main', ['state_from_event', 'br_state_from_event'])
				mpdb_mgr.extract_mod_db(l_dbs_to_mod, events_to_queue)
				for trnsfr in events_transfrs:
					if trnsfr[0][1] != conn_type.Broadcast or len(trnsfr[0]) <= 2:
						continue
					for db_name in trnsfr[0][2:]:
						# db_name = trnsfr[0][2]
						l_new_dbs, new_mods = mpdb_mgr.run_rule(trnsfr[1:], (i_one_story, story_loop_stage, event_step_id[0]),
																db_name, ['state_from_event'])
						events_to_queue += new_mods
						l_dbs_to_mod += l_new_dbs
				# do_learn_rule_from_step(events_to_queue, event_step_id, story_db, player_event[1:], '',
				# 						def_article_dict, db_len_grps, sess, el_set_arr, glv_dict, els_sets)
				story_loop_stage = e_story_loop_stage.complete_state1

			else:
				print('Code flow error. Shouldnt get here')
				exit(1)

			if story_loop_stage == e_story_loop_stage.complete_state1:
				for db_name, event_result in zip(l_dbs_to_mod, events_to_queue):
					mpdb_mgr.apply_mods(db_name, event_result, (i_one_story, story_loop_stage, event_step_id[0]))
					# story_db, iremoved, iadded, added_phrase = \
					# 	rules.apply_mods(story_db, [rules.C_phrase_rec(event_result)], i_story_step)
					# if iremoved != -1:
					# 	l_story_db_event_refs.pop(iremoved)
					# if iadded != -1:
					# 	added_wlist = els.convert_phrase_to_word_list([added_phrase])[0]
					# 	ilen, iphrase = bitvec_mgr.add_phrase(added_wlist,
					# 										  (i_one_story, story_loop_stage, event_step_id[0]))
					# 	l_story_db_event_refs.append((ilen, iphrase))
				print('All dbs for step', event_step_id[0], 'in story num', i_one_story, ':')
				mpdb_mgr.show_dbs()
				story_loop_stage = e_story_loop_stage.decision_init
				i_story_step += 1
				if i_story_step >= num_story_steps:
					break
				i_story_loop_stage = -1

			continue


		# end of loop over story steps
		if i_one_story % adv_config.c_save_every == 0:
			b_keep_working = adv_learn.create_new_conts(glv_dict, db_cont_mgr, db_len_grps, i_gg_cont)
			adv_learn.save_db_status(db_len_grps, db_cont_mgr)
			if not b_keep_working:
				break

		if adv_config.c_b_save_freq_stats:
			story_phrases = [crec.phrase() for crec in story_db]
			story_wlists = els.convert_phrase_to_word_list(story_phrases)
			adv_learn.create_phrase_freq_tbl(story_wlists + l_player_events)

	# end of loop over stories
	sess.close()

		# end of i_one_step loop
	# end of num stories loop

	# for rule in train_rules:
	# 	out_str = 'rule print: \n'
	# 	out_str = rules.print_rule(rule, out_str)
	# 	print(out_str)


def do_adv(glv_dict, def_article_dict, cascade_dict, els_lists, learn_vars):
	fixed_rule_mgr = rules2.cl_fixed_rules(adv_config.rules2_fn)
	cl_templ_grp.glv_dict = glv_dict
	cl_gens_grp.glv_len = cl_templ_grp.glv_len = len(glv_dict[adv_config.sample_el])
	adv_learn.init_learn()
	bitvec_mgr = bitvec.cl_bitvec_mgr()
	mpdb_mgr = mpdb.cl_mpdb_mgr(bitvec_mgr, fixed_rule_mgr)

	for iplay in range(adv_config.c_num_plays):
		dmlearn.learn_reset()
		# db_len_grps, blocked_len_grps, gg_cont_list, i_gg_cont = adv_learn.load_cont_mgr()
		db_cont_mgr = adv_learn.load_cont_mgr()

		db_len_grps, i_active_cont  = adv_learn.sel_cont_and_len_grps(db_cont_mgr)

		db_cont_mgr.load_perm_dict(adv_config.perm_fnt)
		max_eid = db_cont_mgr.apply_perm_dict_data(db_len_grps, i_active_cont)
		db_cont_mgr.load_W_dict(adv_config.W_fnt)
		db_cont_mgr.apply_W_dict_data(db_len_grps, i_active_cont)
		learn_vars[0] = max_eid
		# if gg_cont_list != []:
		# 	dmlearn.learn_reset()
		# 	db_len_grps = []

		play(	glv_dict, def_article_dict, cascade_dict, els_lists,
				adv_config.c_num_stories, adv_config.c_story_len, db_len_grps,
				 db_cont_mgr, i_active_cont, learn_vars, mpdb_mgr, b_for_query=False)
	print('Done!')
	exit(1)



def logic_init():
	mod = importlib.import_module('adv2')

	# import adv2
	els_sets, set_names, l_agents, rules_fn, phrase_freq_fnt, bitvec_dict_fnt = mod.mod_init()
	fixed_rule_mgr = rules2.cl_fixed_rules(rules_fn)
	mod.set_rules_mgr(fixed_rule_mgr)
	bitvec_mgr = bitvec.cl_bitvec_mgr(phrase_freq_fnt, bitvec_dict_fnt)


	full_glv_list = [fname+'s.glv' for fname in adv_config.glv_file_list]
	glv_dict, def_article_dict, cascade_dict, els_sets = \
		els.init_glv(full_glv_list, adv_config.cap_first_arr, adv_config.def_article_arr, adv_config.cascade_els_arr)
	event_step_id = -1
	learn_vars = [event_step_id]
	do_adv(glv_dict, def_article_dict, cascade_dict, els_sets, learn_vars)


def main():
	all_dicts = logic_init()

if __name__ == "__main__":
    main()
