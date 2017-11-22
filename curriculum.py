from __future__ import print_function
import numpy as np
import random

import config
import rules
import story
import cascade
from rules import conn_type
import els
import dmlearn
import ykmeans


def pad_ovec(vecs):
	numrecs = vecs.shape[0]
	pad_len = config.c_ovec_len - vecs.shape[1]
	vecs = np.concatenate((vecs, np.zeros((numrecs, pad_len), dtype=np.int32,)), axis=1)

	return vecs

# A gen rule is how to generate records
# A fld or fld_def (input_flds or output_flds) defines a record of fields for manipulating and learning records
def gen_trainset(gen_rules, event_results_unsorted, glv_dict, max_phrases_per_rule, ivec_dim_dict_fixed):
	global logger

	ivec_dim_dict = {}
	igen_list_by_dict_id = []

	b_dim_dict_fixed = False
	if ivec_dim_dict_fixed:
		b_dim_dict_fixed = True
		ivec_dim_dict = ivec_dim_dict_fixed
		igen_list_by_dict_id = [[] for _ in ivec_dim_dict]

	input_flds_arr = []
	output_flds_arr = []
	fld_def_arr = []
	ivec_arr = []
	ivec_pos_list = []
	ivec_dim_by_rule = []

	input = []
	output = []
	input_unsorted = []
	output_unsorted = []
	ivec_list_unsorted = []
	ovec_list_unsorted = []
	igen_used = -1
	igen_to_used_dict = {}
	for igen, i_and_o_rule in enumerate(gen_rules):
		src_recs, recs = \
			rules.gen_for_rule(b_gen_for_learn=True, rule=i_and_o_rule)
		# if the following assert fails all the subsequent code must be rewritten to account
		# for multiple records per gen_rule
		assert len(recs) == 1, 'expected rule to generate one and only one rec'
		# ivec = make_vec(src_recs, i_and_o_rule.preconds, els_arr)
		ivec = els.make_vec(src_recs, glv_dict)
		# len_so_far = len(ivec_arr)
		ivec_dim = ivec.shape[1]
		dict_id = ivec_dim_dict.get(ivec_dim, None)
		if dict_id == None:
			if b_dim_dict_fixed:
				continue
			igen_used += 1
			dict_id = len(ivec_dim_dict)
			ivec_dim_dict[ivec_dim] = dict_id
			igen_list_by_dict_id.append([igen_used])
		else:
			igen_used += 1
			igen_list_by_dict_id[dict_id].append(igen_used)
		igen_to_used_dict[igen_used] = igen
		fld_def_arr.extend([igen_used] * len(src_recs))
		input_unsorted += src_recs
		ivec_list_unsorted.append(ivec)
		del dict_id, ivec, src_recs

		# ovec = make_vec(recs, i_and_o_rule.gens, els_arr)
		ovec = els.make_vec(recs, glv_dict)
		ovec = pad_ovec(ovec)
		output_unsorted += recs
		ovec_list_unsorted += [ovec]
		del recs

	dict_id_list_of_lists = []
	event_results = []
	idx_rule = -1
	for dict_id, dict_id_list in enumerate(igen_list_by_dict_id):
		if not dict_id_list:
			ivec_arr.append([])
			continue
		idx_of_this_dict_id_list = []
		for igen_idx, one_igen in enumerate( dict_id_list):
			idx_rule += 1
			input += [input_unsorted[one_igen]]
			output += [output_unsorted[one_igen]]
			event_results += [event_results_unsorted[one_igen]]

			if len(input) == 1:
				# all_ivecs = ivec
				all_ovecs = ovec_list_unsorted[one_igen]
			else:
				# all_ivecs = np.concatenate((all_ivecs, ivec), axis=0)
				all_ovecs = np.concatenate((all_ovecs, ovec_list_unsorted[one_igen]), axis=0)

			if igen_idx == 0:
				ivec_arr_one_dict_id = ivec_list_unsorted[one_igen]
			else:
				ivec_arr_one_dict_id  = np.concatenate((ivec_arr_one_dict_id, ivec_list_unsorted[one_igen]), axis=0)

			input_flds_arr.append(gen_rules[igen_to_used_dict[one_igen]].preconds)
			output_flds_arr.append(gen_rules[igen_to_used_dict[one_igen]].gens)
			# for now the following commented-put line is replaced by a line identical to ivec_dim_by_rule
			# For multiple recs per rule, return to commented version but check all implications first
			# ivec_pos_list.extend([dict_id for i in recs])
			ivec_pos_list.extend([dict_id])
			ivec_dim_by_rule.append(dict_id)
			idx_of_this_dict_id_list.append(idx_rule)
		# end loop over igen of the same dict_id
		ivec_arr.append(ivec_arr_one_dict_id)
		dict_id_list_of_lists.append(idx_of_this_dict_id_list)

	# end loop over gen rules

	return 	input_flds_arr, output_flds_arr, fld_def_arr, input, output, \
			ivec_pos_list, all_ovecs, ivec_arr, ivec_dim_dict, ivec_dim_by_rule, \
			dict_id_list_of_lists, event_results


def create_train_vecs(els_sets, els_dict, glv_dict, def_article, els_arr, all_rules,
					  num_stories, num_story_steps, b_for_query, ivec_dim_dict_fixed = None):
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
	event_results = []
	for i_one_story in range(num_stories):
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
		for i_story_step in range(num_story_steps):
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
				all_perms = cascade.get_obj_cascade(els_sets, event_result[1:], story_db, event_phrase, b_for_query=b_for_query)
				event_results += [event_result[1:]] * len(all_perms)

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

	# for rule in train_rules:
	# 	out_str = 'rule print: \n'
	# 	out_str = rules.print_rule(rule, out_str)
	# 	print(out_str)
	return gen_trainset(train_rules, event_results, glv_dict=glv_dict,
						max_phrases_per_rule=config.c_max_phrases_per_rule, ivec_dim_dict_fixed=ivec_dim_dict_fixed)

	# print ('done')


def match_rec(rec0, rec1):
	b_matched = True
	var_dict = dict()

	if len(rec0) != len(rec1):
		return False, var_dict

	for iel in range(len(rec0)):
		rt0 = rec0[iel][0]
		rt1 = rec1[iel][0]
		el0 = rec0[iel][1]
		el1 = rec1[iel][1]
		if rt0 != rt1:
			b_matched = False
			break
		if rt0 == rules.rec_def_type.conn:
			if el0 != el1:
				b_matched = False
				break
		elif rt0 == rules.rec_def_type.var:
			if el0 != el1:
				b_matched = False
				break
		else:
			var_dict[iel] = el1

	return b_matched, var_dict

def match_phrases(phrase_db, phrase_q, gens_phrase_db, event_result):
	db_result = gens_phrase_db[1:-1]
	if len(db_result) != len(event_result):
		return False

	b_matched, var_dict = match_rec(phrase_db, phrase_q)

	if not b_matched:
		return False

	# for iel in range(len(phrase_db)):
	# 	rt_db = phrase_db[iel][0]
	# 	rt_q = phrase_q[iel][0]
	# 	el_db = phrase_db[iel][1]
	# 	el_q = phrase_q[iel][1]
	# 	if rt_db != rt_q:
	# 		return False
	# 	if rt_db == rules.rec_def_type.conn:
	# 		if el_db != el_q:
	# 			return False
	# 	elif rt_db == rules.rec_def_type.var:
	# 		if el_db != el_q:
	# 			return False
	# 	else:
	# 		var_dict[iel] = el_q

	for iel in range(len(event_result)):
		rt_db = db_result[iel][0]
		rt_q = event_result[iel][0]
		el_db = db_result[iel][1]
		el_q = event_result[iel][1]
		if rt_db == rules.rec_def_type.var and var_dict.get(el_db, None) != el_q:
				return False
		if rt_db == rules.rec_def_type.obj and el_db != el_q:
			return False


	return True

def eval_eval(nd_top_cds, nd_top_idxs, success_matrix):
	num_evals = nd_top_idxs.shape[0]
	num_scored = 0.0
	total_score = 0.0
	for one_q in range(num_evals):
		score = 0.0
		sum_true = success_matrix[one_q].count(True)
		for ik in range(config.c_num_k_eval):
			idx = nd_top_idxs[one_q][ik]
			if success_matrix[one_q][idx]:
				score += 1.0
		if sum_true > 0:
			final_score = score / float(min(config.c_num_k_eval, sum_true))
			total_score += final_score
			num_scored += 1.0

	if num_scored > 0.0:
		return total_score / num_scored
	else:
		return 0.5

def build_sym_rules(nd_cluster_id_for_each_rec, input_db, output_db):
	clusters = [[] for _ in range(config.c_num_clusters * config.c_kmeans_num_batches)]
	for irec, icluster in enumerate(nd_cluster_id_for_each_rec):
		input_rec = input_db[irec]
		clusters[icluster].append(input_rec)

	new_clusters = []
	for icluster, cluster in enumerate(clusters):
		if not cluster:
			continue
		rule_group_arr = [[cluster[0]]]
		for rule_from_cluster in cluster[1:]:
			b_found = False
			for rule_group in rule_group_arr:
				b_recs_match, _ = match_rec(rule_from_cluster.phrase(), rule_group[0].phrase())
				if b_recs_match:
					rule_group.append(rule_from_cluster)
					b_found = True
					break
			if not b_found:
				rule_group_arr.append([rule_from_cluster])
		if rule_group_arr:
			new_clusters += rule_group_arr

	return new_clusters


def do_learn(els_sets, els_dict, glv_dict, def_article, els_arr, all_rules):
	input_flds_arr, output_flds_arr, fld_def_arr, \
	input_db, output_db, ivec_pos_list, ovec, ivec_arr_db, ivec_dim_dict_db, ivec_dim_by_rule, \
	dict_id_list_of_lists, _ = \
		create_train_vecs(els_sets, els_dict, glv_dict, def_article, els_arr, all_rules,
						  config.c_curriculum_num_stories, config.c_curriculum_story_len, b_for_query=False)

	input_flds_arr, output_flds_arr, fld_def_arr, \
	input_q, output_q, ivec_pos_list, ovec, ivec_arr_q, ivec_dim_dict_q, ivec_dim_by_rule, \
	dict_id_list_of_lists, event_results = \
		create_train_vecs(els_sets, els_dict, glv_dict, def_article, els_arr, all_rules,
						  config.c_query_num_stories, config.c_query_story_len, b_for_query=True)

	success_matrix = [[True for _ in range(len(input_db))] for _ in range(len(input_q))]
	match_pairs = []
	mismatch_pairs = []

	for iq, one_phrase_q in enumerate(input_q):
		for idb, one_phrase_db in enumerate(input_db):
			b_success = match_phrases(one_phrase_db.phrase(), one_phrase_q.phrase(), output_db[idb].phrase(), event_results[iq])
			if b_success:
				match_pairs += [[iq, idb]]
			else:
				mismatch_pairs += [[iq, idb]]
			success_matrix[iq][idb] = b_success

	input_flds_arr, output_flds_arr, fld_def_arr, \
	input_eval, output_eval, ivec_pos_list, ovec, ivec_arr_eval, ivec_dim_dict_eval, ivec_dim_by_rule, \
	dict_id_list_of_lists, event_results_eval = \
		create_train_vecs(els_sets, els_dict, glv_dict, def_article, els_arr, all_rules,
						  config.c_eval_num_stories, config.c_eval_story_len, b_for_query=True,
						  ivec_dim_dict_fixed=ivec_dim_dict_q)

	success_matrix_eval = [[True for _ in range(len(input_db))] for _ in range(len(input_eval))]
	match_pairs_eval = []
	mismatch_pairs_eval = []

	for ieval, one_phrase_eval in enumerate(input_eval):
		for idb, one_phrase_db in enumerate(input_db):
			b_success = match_phrases(one_phrase_db.phrase(), one_phrase_eval.phrase(), output_db[idb].phrase(), event_results_eval[ieval])
			if b_success:
				match_pairs_eval += [[ieval, idb]]
			else:
				mismatch_pairs_eval += [[ieval, idb]]
			try:
				success_matrix_eval[ieval][idb] = b_success
			except:
				print('error!')
				exit()

	t_y_db, l_W_db, l_W_q, l_batch_assigns, t_err, op_train_step = \
		dmlearn.prep_learn(ivec_dim_dict_db, ivec_dim_dict_q, ivec_arr_db, ivec_arr_q, match_pairs, mismatch_pairs)

	t_top_cds_eval, t_top_idxs_eval = dmlearn.prep_eval(ivec_arr_eval, t_y_db, l_W_q)

	sess, saver = dmlearn.init_learn(l_W_db + l_W_q)
	nd_cluster_id_for_each_rec = ykmeans.cluster_db(sess, len(input_db), t_y_db, config.c_num_clusters)
	build_sym_rules(nd_cluster_id_for_each_rec, input_db, output_db)
	nd_top_cds, nd_top_idxs = dmlearn.run_eval(sess, t_top_cds_eval, t_top_idxs_eval)
	print ('pre-learn eval score:', eval_eval(nd_top_cds, nd_top_idxs, success_matrix_eval))
	dmlearn.run_learning(sess, l_batch_assigns, t_err, saver, op_train_step)
	nd_top_cds, nd_top_idxs = dmlearn.run_eval(sess, t_top_cds_eval, t_top_idxs_eval)
	print ('post-learn eval score:', eval_eval(nd_top_cds, nd_top_idxs, success_matrix_eval))
	return