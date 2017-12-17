from __future__ import print_function
import sys
import math
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
import utils

def pad_ovec(vecs):
	numrecs = vecs.shape[0]
	pad_len = config.c_ovec_len - vecs.shape[1]
	vecs = np.concatenate((vecs, np.zeros((numrecs, pad_len), dtype=np.int32,)), axis=1)

	return vecs

# A gen rule is how to generate records
# A fld or fld_def (input_flds or output_flds) defines a record of fields for manipulating and learning records
def gen_trainset(gen_rules, event_results_raw, event_result_ids_raw, glv_dict, max_phrases_per_rule, ivec_dim_dict_fixed):
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
	all_ovecs = []

	input_sorted = []
	output = []
	input_unsorted = []
	output_unsorted = []
	event_results_unsorted = []
	event_result_ids_unsorted = []
	ivec_list_unsorted = []
	ovec_list_unsorted = []
	event_result_id_arr = []
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
		event_results_unsorted.append(event_results_raw[igen])
		event_result_ids_unsorted.append(event_result_ids_raw[igen])
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
			input_sorted += [input_unsorted[one_igen]]
			output += [output_unsorted[one_igen]]
			event_results += [event_results_unsorted[one_igen]]
			event_result_id_arr.append(event_result_ids_unsorted[one_igen])

			# out_str, gens_str = '\t\t' + str(dict_id) + str(igen_idx) + ': preconds: ', ''
			# out_str = els.print_phrase(input_unsorted[one_igen].phrase(), input_unsorted[one_igen].phrase(), out_str, def_article_dict,
			# 						   el_set_arr, glv_dict)
			# gens_str = els.print_phrase(input_unsorted[one_igen].phrase(), event_results_unsorted[one_igen], out_str, def_article_dict,
			# 							el_set_arr, glv_dict)
			# print(out_str + '\n\t\t\tgens:' + gens_str)
			# if input_unsorted[one_igen].phrase()[3][1] != event_results_unsorted[one_igen][3][1] \
			# 		and input_unsorted[one_igen].phrase()[3][1] != event_results_unsorted[one_igen][1][1] \
			# 		and input_unsorted[one_igen].phrase()[4][1] != event_results_unsorted[one_igen][3][1] \
			# 		and input_unsorted[one_igen].phrase()[4][1] != event_results_unsorted[one_igen][1][1]:
			# 	print('there may be a mismatch on eval phrase #', dict_id, igen_idx)

			if len(input_sorted) == 1:
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

	return 	input_flds_arr, output_flds_arr, fld_def_arr, input_sorted, output, \
			ivec_pos_list, all_ovecs, ivec_arr, ivec_dim_dict, ivec_dim_by_rule, \
			dict_id_list_of_lists, event_results, event_result_id_arr


def create_train_vecs(glv_dict, def_article_dict,
					  num_stories, num_story_steps, b_for_query, ivec_dim_dict_fixed = None):
	start_rule_names = ['objects_start', 'people_start']
	event_rule_names = ['pickup_rule', 'went_rule']
	state_from_event_names = ['gen_rule_picked_up', 'gen_rule_picked_up_free', 'gen_rule_went', 'gen_rule_has_and_went']

	train_rules = []
	event_results = []
	event_result_id = -1
	event_result_id_arr = []
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
				all_perms = cascade.get_obj_cascade(els_sets, event_result[1:], story_db, event_phrase, b_for_query=b_for_query)
				event_results += [event_result] * len(all_perms)

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
					if len(rule_cand) == 1:
						new_conds = rules.nt_tree_junct(single=rule_fields)
					else:
						new_conds = rules.nt_tree_junct(branches=branches, logic=conn_type.AND)

					rule_fields = []
					for el in event_result:
						if el[0] == rules.rec_def_type.obj:
							word = el[1]
							var_field_id = vars_dict.get(word, -1)
							if var_field_id == -1:
								rule_fields.append(rules.nt_rule_fld(els_set=[], df_type=rules.df_type.obj, sel_el=word))
							else:
								rule_fields.append(rules.nt_rule_fld(els_set=[], df_type=rules.df_type.var, var_id=var_field_id))
						elif el[0] == rules.rec_def_type.conn:
							rule_fields.append(rules.nt_rule_fld(els_set=[], df_type=rules.df_type.mod, sel_el=el[1]))
						else:
							print('Unexpected type in event_result!')
							exit()
					new_gens = rules.nt_tree_junct(single = rule_fields)
					new_rule = rules.nt_rule(preconds=new_conds, gens=new_gens)
					train_rules += [new_rule]
					event_result_id_arr.append(event_result_id)
				# end of one_perm
				story_db = rules.apply_mods(story_db, [rules.C_phrase_rec(event_result)], i_story_step)
			# end of one event_result
		# end of i_one_step loop
	# end of num stories loop

	# for rule in train_rules:
	# 	out_str = 'rule print: \n'
	# 	out_str = rules.print_rule(rule, out_str)
	# 	print(out_str)
	return gen_trainset(train_rules, event_results, event_result_id_arr, glv_dict=glv_dict,
						max_phrases_per_rule=config.c_max_phrases_per_rule, ivec_dim_dict_fixed=ivec_dim_dict_fixed)

	# print ('done')

def match_rec_with_set(glv_dict, el_set_arr, set_rec, q_rec):
	b_matched = True
	var_dict = dict()

	if len(set_rec) != len(q_rec):
		return False, var_dict

	for iel in range(len(set_rec)):
		set_rt = set_rec[iel][0]
		q_rt = q_rec[iel][0]
		set_el = set_rec[iel][1]
		q_el = q_rec[iel][1]
		if not (set_rt == rules.rec_def_type.set and q_rt == rules.rec_def_type.obj) and set_rt != q_rt:
			b_matched = False
			break
		if set_rt == rules.rec_def_type.conn:
			if set_el != q_el:
				b_matched = False
				break
		elif set_rt == rules.rec_def_type.var:
			if set_el != q_el:
				b_matched = False
				break
		elif set_rt == rules.rec_def_type.set and q_rt == rules.rec_def_type.obj:
			q_vec = glv_dict[q_el]
			set_vec, set_cd, _ = el_set_arr[set_el]
			cd = sum([q_vec[i] * set_val for i, set_val in enumerate(set_vec)])
			if cd < set_cd:
				b_matched = False
				break
			var_dict[iel] = q_el
		else:
			b_matched = False
			break

	return b_matched, var_dict


def match_rule_with_phrase(glv_dict, el_set_arr, phrase_rule, phrase_q, gens_phrase_db, event_result):
	db_result = gens_phrase_db[1:-1]
	if len(db_result) != len(event_result):
		return False, False

	b_matched, var_dict = match_rec_with_set(glv_dict, el_set_arr, phrase_rule, phrase_q)

	if not b_matched:
		return False, False

	for iel in range(len(event_result)):
		rt_db = db_result[iel][0]
		rt_q = event_result[iel][0]
		el_db = db_result[iel][1]
		el_q = event_result[iel][1]
		if rt_db == rules.rec_def_type.var and var_dict.get(el_db, None) != el_q:
			return True, False
		if rt_db == rules.rec_def_type.obj and el_db != el_q:
			return True, False
		if rt_db == rules.rec_def_type.set:
			set_vec, set_cd, _ = el_set_arr[el_db]
			best_cd = -1.0
			best_name = 'not found!'
			for sym, one_vec in glv_dict.iteritems():
				cd = sum([one_vec[i] * set_val for i, set_val in enumerate(set_vec)])
				if cd > best_cd:
					best_name, best_cd = sym, cd
			if best_cd == -1.0 or best_name != el_q:
				return True, False
			# q_vec = glv_dict[el_q]
			# cd = sum([q_vec[i] * set_val for i, set_val in enumerate(set_vec)])
			# if cd < set_cd:
			# 	return True, False

	return True, True

# An instance phrase is one with no sets. It is a specific example from the training set
#
def match_instance_phrase(rec0, rec1, b_preconds):
	b_matched = True

	if len(rec0) != len(rec1):
		return False

	for iel in range(len(rec0)):
		rt0 = rec0[iel][0]
		rt1 = rec1[iel][0]
		el0 = rec0[iel][1]
		el1 = rec1[iel][1]
		if rt0 != rt1:
			b_matched = False
			break
		if rt0 == rules.rec_def_type.conn or rt0 == rules.rec_def_type.var:
			if el0 != el1:
				b_matched = False
				break
		elif rt0 == rules.rec_def_type.obj:
			if not b_preconds and el0 != el1:
				b_matched = False
				break
		else:
			print('incorrect rec_def_type of match_instance_phrase. Exiting!')
			exit()

	return b_matched


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

def compress_el_sets(el_set_arr, veclen):
	new_el_set_arr = []
	new_set_dict = dict()
	valid_sets = [True for _ in el_set_arr]
	idx_new_set = -1
	for iset1, set1 in enumerate(el_set_arr):
		if not valid_sets[iset1]:
			continue
		idx_new_set += 1
		new_set_dict[iset1] = idx_new_set
		new_set = set1
		for iset2, set2 in enumerate(el_set_arr):
			if iset2 <= iset1:
				continue
			if not valid_sets[iset2]:
				continue
			vec_avg1, min_cd1, vec_list1 = new_set
			vec_avg2, min_cd2, vec_list2 = set2
			b_combine = True
			if min_cd1 != min_cd2:
				b_combine = False
			if b_combine:
				cd = sum([vec_avg2[i] * vec_avg1[i] for i in range(veclen)])
				if cd < min_cd1: # they are the same if the code reaches here
					b_combine = False
			if b_combine:
				comb_vec_list = vec_list1 + vec_list2
				vec_avg, min_cd = utils.get_avg_min_cd(comb_vec_list, veclen)
				if min_cd <  min_cd1:
					b_combine = False
			if b_combine:
				new_set = [vec_avg, min_cd1, comb_vec_list]
				valid_sets[iset2] = False
				new_set_dict[iset2] = idx_new_set

		new_el_set_arr.append(new_set)

	return new_el_set_arr, new_set_dict

def compress_rec_rules(rec_rule_arr):
	new_rec_rules = []
	new_rule_dict = dict()
	valid_rules = [True for _ in rec_rule_arr]
	idx_new_rule = -1
	for irule1, rule1 in enumerate(rec_rule_arr):
		if not valid_rules[irule1]:
			continue
		idx_new_rule += 1
		new_rule_dict[irule1] = idx_new_rule
		new_rule = rule1
		for irule2, rule2 in enumerate(rec_rule_arr):
			if irule2 <= irule1:
				continue
			if not valid_rules[irule2]:
				continue
			b_combine = True
			rec0, rec1 = rule1[0], rule2[0]
			ilen0, ilen1 = rule1[1], rule2[1]
			if len(rec0) != len(rec1) or ilen0 != ilen1:
				b_combine = False
			if b_combine:
				for iel in range(len(rec0)):
					rt0 = rec0[iel][0]
					rt1 = rec1[iel][0]
					el0 = rec0[iel][1]
					el1 = rec1[iel][1]
					if rt0 != rt1 or el0 != el1:
						b_combine = False
						break
			if b_combine:
				valid_rules[irule2] = False
				new_rule_dict[irule2] = idx_new_rule

		new_rec_rules.append(new_rule)

	return new_rec_rules

def find_quant_thresh(cd):
	for thresh in config.c_rule_cluster_thresh_levels:
		if cd >= thresh:
			return thresh
	return -1.0

def group_rule_els(glv_dict, rule_cluster, el_set_arr, len_preconds):
	veclen = len(glv_dict[config.sample_el])
	rule_phrase = []
	new_sets = []
	for iel, el in enumerate(rule_cluster[0]):
		if iel >= len_preconds:
			rule_phrase.append([el[0], el[1]])
			continue
		if el[0] != rules.rec_def_type.obj:
			rule_phrase.append([el[0], el[1]])
			continue
		vec_list = []
		for phrase in rule_cluster:
			# phrase = rule.phrase[0]
			vec = glv_dict[phrase[iel][1]]
			vec_list.append(vec)

		vec_avg, min_cd = utils.get_avg_min_cd(vec_list, veclen)
		min_cd = find_quant_thresh(min_cd)
		rule_phrase.append([rules.rec_def_type.set, len(el_set_arr)])
		el_set_arr.append([vec_avg, min_cd, vec_list])

	return rule_phrase
	# return rules.C_phrase_rec(init_phrase=rule_phrase)


		# vec_max = [vec[i] if vec[i] > vec_max[i] else vec_max[i] for i in range(veclen)]
			# vec_min = [vec[i] if vec[i] < vec_min[i] else vec_min[i] for i in range(veclen)]

def build_sym_rules(glv_dict, nd_cluster_id_for_each_rec, input_db, output_db):
	# First we put the rec ids into clusters where each cluster holds those from that cluster
	clusters = [[] for _ in range(config.c_num_clusters * config.c_kmeans_num_batches)]
	for irec, icluster in enumerate(nd_cluster_id_for_each_rec):
		# input_rec = input_db[irec]
		# clusters[icluster].append([input_db[irec], output_db[irec]])
		clusters[icluster].append(irec)

	# The problem now is that we may have totally incompatible records in the same cluster
	# So we break up each cluster into a rule group. We start off by creating one rule group
	# from the first rec in the cluster. If any match it gets added to the rule group,
	# otherwise, we start a new rule group. Henceforth, within the cluster, it will try
	# and match the first from each of the rule groups and only if all fail start its
	# own rule group
	new_clusters = []
	for icluster, cluster in enumerate(clusters):
		if not cluster:
			continue
		rule_group_arr = [[cluster[0]]]
		for rule_from_cluster in cluster[1:]:
			b_found = False
			for rule_group in rule_group_arr:
				# cluster_phrases = input_db[rule_from_cluster].phrase() + output_db[rule_from_cluster].phrase()
				# group_phrases = input_db[rule_group[0]].phrase() + output_db[rule_group[0]].phrase()
				b_preconds_match = match_instance_phrase(input_db[rule_from_cluster].phrase(),
														 input_db[rule_group[0]].phrase(), b_preconds=True)
				b_gens_match = match_instance_phrase(output_db[rule_from_cluster].phrase(),
													 output_db[rule_group[0]].phrase(), b_preconds=False)
				b_recs_match = b_preconds_match and b_gens_match
				# Take care of a rather difficult to believe scenario where everything matches but the output len does not
				if b_recs_match and len(input_db[rule_group[0]].phrase()) != len(input_db[rule_from_cluster].phrase()):
					b_recs_match = False
				if b_recs_match:
					rule_group.append(rule_from_cluster)
					b_found = True
					break
			if not b_found:
				rule_group_arr.append([rule_from_cluster])
		if rule_group_arr:
			new_clusters += rule_group_arr

	el_set_arr = []
	new_rec_rules = []
	for rule_cluster in new_clusters:
		rule_phrase_cluster = \
			[input_db[irule].phrase() + output_db[irule].phrase() for irule in rule_cluster]
		if len(rule_cluster) < 2:
			continue
		new_phrase = group_rule_els(glv_dict, rule_phrase_cluster, el_set_arr, len(input_db[rule_cluster[0]].phrase()))
		# We now need to record the length of the original preconds. So we create a format where each rule
		# contains both input and output phrases and the length of the input phrase
		# At this point we know that all rules in the cluster (that have now been combined into one rule)
		# have the same input and output length.
		new_rec_rules.append([new_phrase, len(input_db[rule_cluster[0]].phrase())])

	el_set_arr, new_set_dict = compress_el_sets(el_set_arr, len(el_set_arr[0][0]))
	for rec_rule in new_rec_rules:
		for el in rec_rule[0]:
			if el[0] == rules.rec_def_type.set:
				el[1] = new_set_dict[el[1]]
	new_rec_rules = compress_rec_rules(new_rec_rules)
	return new_rec_rules, el_set_arr

# For two otherwise identical rules, determine if one is always equal or more specific to the other
# return resolved, if resolved 0 won
def rule_rec_more_specific(rule0, rule1, el_set_arr):
	b_0_advtg, b_1_advtg =  False, False
	for iel, rel0 in enumerate(rule0[0]):
		rt0 = rule0[0][iel][0]
		rt1 = rule1[0][iel][0]
		el0 = rule0[0][iel][1]
		el1 = rule1[0][iel][1]
		if rt0 != rt1:
			return False, False
		if rt0 != rules.rec_def_type.set:
			if el0 != el1:
				return False, False
		else:
			set0, set1 = el_set_arr[el0], el_set_arr[el1]
			if set0[1] > set1[1]:
				b_0_advtg = True
			elif set0[1] < set1[1]:
				b_1_advtg = True

	if (b_0_advtg and b_1_advtg) or (not b_0_advtg and not b_1_advtg):
		return False, False

	return  b_0_advtg





def do_set_eval(glv_dict, def_article_dict, sess, input_db, output_db,  t_y_db, input_eval, event_results_eval, event_result_id_arr):
	nd_cluster_id_for_each_rec = ykmeans.cluster_db(sess, len(input_db), t_y_db, config.c_num_clusters)
	new_rec_rules, el_set_arr = build_sym_rules(glv_dict, nd_cluster_id_for_each_rec, input_db, output_db)

	set_eval_score = 0.0
	set_eval_num = 0.0
	rule_score_arr = []
	events_matched_by_rule = []
	for ieval, one_phrase_eval in enumerate(input_eval):
		out_str, gens_str = '\t\t' + str(ieval) + ': Matched on. preconds: ', ''
		out_str = els.print_phrase(one_phrase_eval.phrase(), one_phrase_eval.phrase(), out_str, def_article_dict,
								   el_set_arr, glv_dict)
		gens_str = els.print_phrase(one_phrase_eval.phrase(), event_results_eval[ieval], out_str, def_article_dict,
									el_set_arr, glv_dict)
		print(out_str + '\n\t\t\tgens:' + gens_str)
		# if 		one_phrase_eval.phrase()[3][1] != event_results_eval[ieval][3][1] \
		# 		and one_phrase_eval.phrase()[3][1] != event_results_eval[ieval][1][1] \
		# 		and one_phrase_eval.phrase()[4][1] != event_results_eval[ieval][3][1] \
		# 		and one_phrase_eval.phrase()[4][1] != event_results_eval[ieval][1][1]:
		# 	print('there may be a mismatch on eval phrase #', ieval)

	# new_rec_rules, el_set_arr = build_sym_rules(glv_dict, nd_cluster_id_for_each_rec, input_db, output_db)

	for idb, one_phrase_db in enumerate(new_rec_rules):
		rule_preconds_phrase = one_phrase_db[0][:one_phrase_db[1]]
		rule_gens_phrase = one_phrase_db[0][one_phrase_db[1]:]
		out_str, gens_str = str(idb) + ': preconds: ', ''
		out_str = els.print_phrase(rule_preconds_phrase, rule_preconds_phrase, out_str, def_article_dict, el_set_arr, glv_dict)
		gens_str = els.print_phrase(rule_preconds_phrase, rule_gens_phrase, out_str, def_article_dict, el_set_arr, glv_dict)
		print(out_str + '\n\tgens:' + gens_str)
		rule_score_arr.append([0.0, 0.0])
		events_matched_by_rule.append([])
		for ieval, one_phrase_eval in enumerate(input_eval):
			b_rule_match, b_result_match = \
				match_rule_with_phrase(	glv_dict, el_set_arr, rule_preconds_phrase,
										one_phrase_eval.phrase(), rule_gens_phrase,
										event_results_eval[ieval])
			if b_rule_match:
				out_str, gens_str = '\t\t' + str(ieval) + ': Matched on. preconds: ', ''
				out_str = els.print_phrase(one_phrase_eval.phrase(), one_phrase_eval.phrase(), out_str, def_article_dict,
										   el_set_arr, glv_dict)
				gens_str = els.print_phrase(one_phrase_eval.phrase(), event_results_eval[ieval], out_str, def_article_dict,
											el_set_arr, glv_dict)
				print(out_str + '\n\t\t\tgens:' + gens_str)
				set_eval_num += 1.0
				rule_score_arr[idb][0] += 1.0
				if b_result_match:
					print('\t\t\tGens matched too')
					set_eval_score += 1.0
					rule_score_arr[idb][1] += 1.0
					events_matched_by_rule[idb].append(event_result_id_arr[ieval])
				else:
					print('\t\tRule matched but gens didnt!')
					continue
		print('\tNum matches:', rule_score_arr[idb][0], '. Num successes: ', rule_score_arr[idb][1])

	# new_rec_rules, el_set_arr = build_sym_rules(glv_dict, nd_cluster_id_for_each_rec, input_db, output_db)

	good_rule_score_arr = []
	good_rule_event_ids = []
	good_rule_len = []
	good_rule_to_orig_map = []
	for idb, one_phrase_db in enumerate(new_rec_rules):
		if rule_score_arr[idb][1] > config.c_eval_num_stories * config.c_eval_story_len / 4 \
				and rule_score_arr[idb][0] == rule_score_arr[idb][1]:
			rule_preconds_phrase = one_phrase_db[0][:one_phrase_db[1]]
			rule_gens_phrase = one_phrase_db[0][one_phrase_db[1]:]
			out_str, gens_str = str(idb) + ': preconds: ', ''
			out_str = els.print_phrase(rule_preconds_phrase, rule_preconds_phrase, out_str, def_article_dict, el_set_arr, glv_dict)
			gens_str = els.print_phrase(rule_preconds_phrase, rule_gens_phrase, out_str, def_article_dict, el_set_arr, glv_dict)
			print(out_str + '\n\tgens:' + gens_str)
			print('All eval preconds that matched, gens did too. Num matches:', rule_score_arr[idb][1])
			good_rule_score_arr.append(rule_score_arr[idb])
			good_rule_event_ids.append(events_matched_by_rule[idb])
			good_rule_len.append(len(one_phrase_db[0]))
			good_rule_to_orig_map.append(idb)

	irule_holding = [True for _ in good_rule_score_arr]
	for gidb1, _ in enumerate(good_rule_score_arr):
		if not irule_holding[gidb1]:
			continue
		events1 = good_rule_event_ids[gidb1]
		for gidb2, _ in enumerate(good_rule_score_arr):
			if not irule_holding[gidb2]:
				continue
			if gidb2 <= gidb1:
				continue
			events2 = good_rule_event_ids[gidb2]
			if set(events1) < set(events2):
				if good_rule_len[gidb1] >=  good_rule_len[gidb2]:
					irule_holding[gidb1] = False
					break
			elif set(events1) > set(events2):
				if good_rule_len[gidb1] <=  good_rule_len[gidb2]:
					irule_holding[gidb2] = False
			elif set(events1) == set(events2):
				if good_rule_len[gidb1] <  good_rule_len[gidb2]:
					irule_holding[gidb2] = False
				elif good_rule_len[gidb1] >  good_rule_len[gidb2]:
					irule_holding[gidb1] = False
					break
				else:
					assert good_rule_len[gidb1] ==  good_rule_len[gidb2]
					b_resolved, b_first_wins = \
						rule_rec_more_specific(new_rec_rules[good_rule_to_orig_map[gidb1]],
											   new_rec_rules[good_rule_to_orig_map[gidb2]],
											   el_set_arr)
					if b_resolved:
						if b_first_wins:
							irule_holding[gidb2] = False
						else:
							irule_holding[gidb1] = False
							break
			else:
				continue

	best_rec_rules = []
	for gidb, idb in enumerate(good_rule_to_orig_map):
		if not irule_holding[gidb]:
			continue
		best_rec_rules.append(new_rec_rules[idb])

	print('best rules:')
	for idb, one_phrase_db in enumerate(best_rec_rules):
		rule_preconds_phrase = one_phrase_db[0][:one_phrase_db[1]]
		rule_gens_phrase = one_phrase_db[0][one_phrase_db[1]:]
		out_str, gens_str = str(idb) + ': preconds: ', ''
		out_str = els.print_phrase(rule_preconds_phrase, rule_preconds_phrase, out_str, def_article_dict, el_set_arr, glv_dict)
		gens_str = els.print_phrase(rule_preconds_phrase, rule_gens_phrase, out_str, def_article_dict, el_set_arr, glv_dict)
		print(out_str + '\n\tgens:' + gens_str)

	if set_eval_num == 0.0:
		print('Not one set eval matched.')
	else:
		print('Set eval. Num matches: ', set_eval_num, ' score:', set_eval_score / set_eval_num)



def do_learn(glv_dict, def_article_dict):
	print('Creating initial db')
	input_flds_arr, output_flds_arr, fld_def_arr, \
	input_db, output_db, ivec_pos_list, ovec, ivec_arr_db, ivec_dim_dict_db, ivec_dim_by_rule, \
	dict_id_list_of_lists, _, _ = \
		create_train_vecs(glv_dict, def_article_dict,
						  config.c_curriculum_num_stories, config.c_curriculum_story_len, b_for_query=False)

	print('Creating query phrases')
	input_flds_arr, output_flds_arr, fld_def_arr, \
	input_q, output_q, ivec_pos_list, ovec, ivec_arr_q, ivec_dim_dict_q, ivec_dim_by_rule, \
	dict_id_list_of_lists, event_results, _ = \
		create_train_vecs(glv_dict, def_article_dict,
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

	print('Creating eval phrases')
	input_flds_arr, output_flds_arr, fld_def_arr, \
	input_eval, output_eval, ivec_pos_list, ovec, ivec_arr_eval, ivec_dim_dict_eval, ivec_dim_by_rule, \
	dict_id_list_of_lists, event_results_eval, event_result_id_arr = \
		create_train_vecs(glv_dict, def_article_dict,
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
	do_set_eval(glv_dict, def_article_dict, sess, input_db, output_db,  t_y_db, input_eval,
				event_results_eval, event_result_id_arr)
	nd_top_cds, nd_top_idxs = dmlearn.run_eval(sess, t_top_cds_eval, t_top_idxs_eval)
	print ('pre-learn eval score:', eval_eval(nd_top_cds, nd_top_idxs, success_matrix_eval))
	dmlearn.run_learning(sess, l_batch_assigns, t_err, saver, op_train_step)
	nd_top_cds, nd_top_idxs = dmlearn.run_eval(sess, t_top_cds_eval, t_top_idxs_eval)
	print ('post-learn eval score:', eval_eval(nd_top_cds, nd_top_idxs, success_matrix_eval))
	do_set_eval(glv_dict, def_article_dict, sess, input_db, output_db,  t_y_db, input_eval,
				event_results_eval, event_result_id_arr)
	return