from __future__ import print_function
# import sys
# import math
import numpy as np
# import random
# import itertools
import csv
from StringIO import StringIO
import copy

import config
import rules
from rules import rec_def_type
# import story
# import cascade
from rules import conn_type
import els
# import dmlearn
# import ykmeans
import utils

def make_preconds_rule_from_phrases(rule_base, one_perm, story_db):
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

	return new_conds, vars_dict, rule_cand

def make_gens_rule_from_phrases(vars_dict, event_result):

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
	new_gens = rules.nt_tree_junct(single=rule_fields)

	return new_gens
	return rules.nt_rule(preconds=new_conds, gens=new_gens)

def gen_cvo_str(rec):
	cvo_str = ''
	for el in rec:
		if el[0] == rules.rec_def_type.conn:
			cvo_str += 'c'
			if el[1] == rules.conn_type.AND:
				cvo_str += 'a'
			elif el[1] == rules.conn_type.OR:
				cvo_str += 'r'
			if el[1] == rules.conn_type.start:
				cvo_str += 's'
			if el[1] == rules.conn_type.end:
				cvo_str += 'e'
		elif el[0] == rules.rec_def_type.var:
			cvo_str += 'v'
			cvo_str += str(el[1]).rjust(2, '0')
		else:
			cvo_str += 'o'

	return cvo_str

def gen_rec_str(rec):
	if rec == None:
		return ''
	lel = []
	for el in rec:
		if el[0] == rules.rec_def_type.conn:
			lcvo = ['c']
			if el[1] == rules.conn_type.AND:
				lcvo += ['a']
			elif el[1] == rules.conn_type.OR:
				lcvo += ['r']
			elif el[1] == rules.conn_type.start:
				lcvo += ['s']
			elif el[1] == rules.conn_type.end:
				lcvo += ['e']
			elif el[1] == rules.conn_type.Insert:
				lcvo += ['i']
			elif el[1] == rules.conn_type.Unique:
				lcvo += ['u']
			elif el[1] == rules.conn_type.Remove:
				lcvo += ['d']
			elif el[1] == rules.conn_type.Modify:
				lcvo += ['m']
			elif el[1] == rules.conn_type.IF:
				lcvo += ['f']
			elif el[1] == rules.conn_type.THEN:
				lcvo += ['t']
			else:
				print('Coding error in gen_rec_str. Unknown rec_def_type. Exiting!')
				exit()
		elif el[0] == rules.rec_def_type.var:
			lcvo = ['v']
			# lcvo += [str(el[1]).rjust(2, '0')]
			lcvo += [el[1]]
		elif el[0] == rules.rec_def_type.obj:
			lcvo = ['o']
			lcvo += [el[1]]
		elif el[0] == rules.rec_def_type.like:
			lcvo = ['l']
			lcvo += [el[1], el[2]]
		else:
			lcvo = ['e', '-1']
		lel += [':'.join(map(str, lcvo))]

	cvo_str =  ','.join(map(str, lel))
	return cvo_str

def extract_rec_from_str(srec):
	if srec == '':
		return None

	f = StringIO(srec)
	# csvw = csv.writer(l)
	rec = []
	lelr = csv.reader(f, delimiter=',')
	row = next(lelr)
	for lcvo in row:
		fcvo = StringIO(lcvo)
		lelf = next(csv.reader(fcvo, delimiter=':'))
		if lelf[0] == 'c':
			el = [rules.rec_def_type.conn]
			if lelf[1] == 'a':
				el += [rules.conn_type.AND]
			elif lelf[1] == 'r':
				el += [rules.conn_type.OR]
			elif lelf[1] == 's':
				el += [rules.conn_type.start]
			elif lelf[1] == 'e':
				el += [rules.conn_type.end]
			elif lelf[1] == 'i':
				el += [rules.conn_type.Insert]
			elif lelf[1] == 'u':
				el += [rules.conn_type.Unique]
			elif lelf[1] == 'm':
				el += [rules.conn_type.Modify]
			elif lelf[1] == 'd':
				el += [rules.conn_type.Remove]
			elif lelf[1] == 'f':
				el += [rules.conn_type.IF]
			elif lelf[1] == 't':
				el += [rules.conn_type.THEN]
			elif lelf[1] == 'b':
				el += [rules.conn_type.Broadcast]
			else:
				print('Unknown rec def. Exiting.')
				exit()
			if lelf[1] in ['s', 'i', 'u', 'm', 'd', 'b']:
				if len(lelf) > 2:
					el += [int(v) for v in lelf[2:]]
		elif lelf[0] == 'v':
			el = [rules.rec_def_type.var]
			el += [int(lelf[1])]
			if len(lelf) > 2 and lelf[2] == 'r':
				el += [rules.conn_type.replace_with_next]
		elif lelf[0] == 'o':
			el = [rules.rec_def_type.obj]
			el += [lelf[1]]
			if len(lelf) > 2 and lelf[2] == 'r':
				el += [rules.conn_type.replace_with_next]
		elif lelf[0] == 'l':
			el = [rules.rec_def_type.like]
			el += [lelf[1], float(lelf[2])]
			if len(lelf) > 3:
				el += [int(lelf[3])]

		else:
			el = [rules.rec_def_type.error]
			el += [lelf[1]]

		rec += [el]

	return rec


def match_gens_phrase(rec0, rec1):
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
			if el0 != el1:
				b_matched = False
				break
		else:
			print('incorrect rec_def_type of match_instance_phrase. Exiting!')
			exit()

	return b_matched

def replace_vars_in_phrase(preconds_rec, gens_rec):
	result = []
	for el in gens_rec:
		if el[0] == rules.rec_def_type.var:
			result.append(preconds_rec[el[1]])
		else:
			result.append(el)

	return result

def place_vars_in_phrase(vars_dict, gens_rec):
	result = []
	for el in gens_rec:
		if el[0] == rules.rec_def_type.obj:
			idx = vars_dict.get(el[1], -1)
			if idx >= 0:
				new_el = []
				for inel, nel in enumerate(el):
					if inel == 0:
						new_el.append(rules.rec_def_type.var)
					elif inel == 1:
						new_el.append(idx)
					else:
						new_el.append(nel)
				result.append(new_el)
			else:
				result.append(el)
		else:
			result.append(el)

	return result

def match_rec_exact(rec0, rec1):
	for iel,_ in enumerate(rec0):
		if rec0[iel][0] != rec1[iel][0] or rec0[iel][1] != rec1[iel][1]:
			return False

	return True

def match_instance_to_rule(glv_dict, el_set_arr, set_rec, q_rec):
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

c_tolerance = 1e-3
def find_quant_thresh(cd):
	for thresh in config.c_rule_cluster_thresh_levels:
		if cd >= (thresh-c_tolerance):
			return thresh
	return -1.0

def make_rule_grp_old(glv_dict, rule_cluster, el_set_arr):
	veclen = len(glv_dict[config.sample_el])
	rule_phrase = []
	for iel, el in enumerate(rule_cluster[0]):
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

def make_rule_grp(glv_dict, rule_cluster, veclen, curr_cont):
	# veclen = len(glv_dict[config.sample_el])
	rule_phrase = []
	prev_rule_grp = []
	b_take_from_prev = False
	if not curr_cont.is_null():
		b_take_from_prev = True
		prev_rule_grp = curr_cont.get_rule()
		if prev_rule_grp[0][0] == rules.rec_def_type.conn and prev_rule_grp[0][1] == rules.conn_type.AND:
			b_prev_has_AND = True
			take_from_prev_until = len(prev_rule_grp) - 1
			start_offset = 0
		else:
			b_prev_has_AND = False
			take_from_prev_until = len(prev_rule_grp)
			start_offset = 1

	for iel, el in enumerate(rule_cluster[0]):
		if b_take_from_prev:
			if iel == 0 and not b_prev_has_AND:
				rule_phrase.append([el[0], el[1]])
				continue
			if iel < take_from_prev_until + start_offset:
				rule_phrase.append(prev_rule_grp[iel-start_offset])
				continue

		if el[0] != rules.rec_def_type.obj:
			rule_phrase.append([el[0], el[1]])
			continue
		vec_list = []
		for phrase in rule_cluster:
			# phrase = rule.phrase[0]
			vec = glv_dict[phrase[iel][1]]
			vec_list.append(vec)

		vec_avg = utils.get_avg_cd(vec_list, veclen)
		best_cd = -1.0
		best_name = 'not found!'
		for sym, one_vec in glv_dict.iteritems():
			cd = sum([one_vec[i] * set_val for i, set_val in enumerate(vec_avg)])
			if cd > best_cd:
				best_name, best_cd, best_vec = sym, cd, one_vec

		min_cd = utils.get_min_cd(vec_list, best_vec, veclen)
		min_cd = find_quant_thresh(min_cd)
		rule_phrase.append([rules.rec_def_type.like, best_name, min_cd])
		# el_set_arr.append([vec_avg, min_cd, vec_list])

	return rule_phrase

def does_match_rule(glv_dict, rule, perm):
	b_match = True
	for iel, el in enumerate(rule):
		if el[0] != rules.rec_def_type.like:
			continue
		vec_rule = glv_dict[el[1]]
		vec_perm = glv_dict[perm[iel][1]]
		cd = sum([vec_perm[i] * rule_val for i, rule_val in enumerate(vec_rule)])
		if cd < (el[2] - config.c_cd_epsilon):
			b_match = False
			break

	return b_match

def rule_grp_is_one_in_two(glv_dict, rule1, rule2):
	b_match = True
	for iel, el in enumerate(rule2):
		if el[0] != rules.rec_def_type.like:
			continue
		vec_rule = glv_dict[el[1]]
		vec_perm = glv_dict[rule1[iel][1]]
		min_cd2 = el[2]
		if rule1[iel][0] == rules.rec_def_type.like:
			min_cd1 = rule1[iel][2]
			if min_cd1 < (min_cd2 - config.c_cd_epsilon):
				b_match = False
				break
		cd = sum([vec_perm[i] * rule_val for i, rule_val in enumerate(vec_rule)])
		if cd < (min_cd2 - config.c_cd_epsilon):
			b_match = False
			break

	return b_match


def get_olen(scvo):
	return scvo.count('o')

def make_vec(glv_dict, perm_rec, olen, glv_len):
	vec = np.zeros(olen*glv_len, dtype=np.float32)
	io = -1
	for el in perm_rec:
		if el[0] == rules.rec_def_type.obj:
			io += 1
			# vec = glv_dict.get(el[1], None)
			# if vec == None:
			# 	print('Error!', el[1], 'does not appear in the glv dict for this logic module. Exiting!')
			# 	exit(1)
			# vec[io*glv_len:(io+1)*glv_len] = vec
			vec[io*glv_len:(io+1)*glv_len] = glv_dict[el[1]]

	if config.c_b_nbns:
		en = np.linalg.norm(vec)
		vec = vec / en

	return vec

def find_gg_by_sig(db_len_grps, sig):
	# Add blocking to the templ find here
	_, templ_len, templ_scvo, igg, b_blocking = sig
	for igrp, len_grp in enumerate(db_len_grps):
		if len_grp.len() == templ_len:
			found_templ = len_grp.find_templ(templ_scvo)
			return True, found_templ, igg

	return False, None, None


def report_confirmers(db_len_grps, gg_confirmed_list, el_set_arr, def_article_dict, glv_dict):
	print('Rules found for confirmation:')
	for gg_confirm in gg_confirmed_list:
		b_found, found_templ, igg = find_gg_by_sig(db_len_grps, gg_confirm)
		if b_found:
			gg = found_templ.get_gg(igg)
			rule_grp = gg.get_rule_grp()
			out_str = ''
			out_str = els.print_phrase(rule_grp, rule_grp, out_str, def_article_dict, el_set_arr, glv_dict)
			gens_rec = gg.get_gens_rec()
			gens_str = ''
			gens_str = els.print_phrase(gens_rec, gens_rec, gens_str, def_article_dict, el_set_arr, glv_dict)
			print('PRECONDS:', out_str, '\nGENS:', gens_str)

def identical_rule_part(rule1, rule_ends1, rule2, rule_ends2):
	start1, end1 = rule_ends1
	start2, end2 = rule_ends2

	len1 = end1 - start1
	len2 = end2 - start2
	if len1 != len2:
		return False

	for iel in range(len1):
		el1 = rule1[start1 + iel]
		el2 = rule2[start2 + iel]
		if el1 != el2:
			return False

	return True


def duplicate_piece(src_rule, rule_ends, iel_sel, src_other_rule, other_rule_ends, iel_sel_other, b_make_var):
	rule = copy.deepcopy(src_rule)
	other_rule = copy.deepcopy(src_other_rule)
	start, end = rule_ends
	other_start, other_end = other_rule_ends
	piece = []
	for iel, el in enumerate(other_rule):
		if iel < other_start or iel > other_end:
			continue
		if b_make_var and iel == iel_sel_other:
			piece += [[rules.rec_def_type.var, iel_sel]]
		else:
			piece += [el]
	new_rule = rule[:end+1] + piece + rule[end+1:]
	insert_len = len(piece)
	for iel, el in enumerate(new_rule):
		if iel <= end + insert_len:
			continue
		if el[0] == rules.rec_def_type.var and el[1] > end:
			el[1] = el[1] + insert_len
			assert el[1] <= len(new_rule), 'duplicate piece created a reference beyond the end of the rule'

	return new_rule


def get_indef_likes(rule, rule_ends):
	start, end = rule_ends
	indefs_list = []
	for iel, el in enumerate(rule):
		if iel < start or iel > end:
			continue

		if el[0] == rules.rec_def_type.like and el[2] < (1.0 - config.c_cd_epsilon):
			indefs_list += [iel]

	return indefs_list

def create_start_end_listing(rule):
	listing = []
	el = rule[0]
	if el[0] == rules.rec_def_type.conn and not el[1] == rules.conn_type.AND:
		print('The purpose of this function is to process complex rules only')
		return listing
	start, end = -1, -1
	b_inside = False
	for iel, el in enumerate(rule):
		if iel == 0:
			continue

		if not b_inside and el[0] == rules.rec_def_type.conn and el[1] == rules.conn_type.start:
			b_inside = True
			start = iel
		elif b_inside and el[0] == rules.rec_def_type.conn and el[1] == rules.conn_type.end:
			b_inside = False
			end = iel
			listing.append([start, end])
	return listing


def make_scvo_arr(scvo):
	core = str(scvo)
	if scvo[0:2] =='ca':
		core = core[2:-2]
	core_arr = core.split('cs')[1:]
	core_arr = [acore[:-2] for acore in core_arr]
	return core_arr

def match_partial_scvo(perm_scvo, rule_scvo, rule_level):
	perm_arr = make_scvo_arr(perm_scvo)
	rule_arr = make_scvo_arr(rule_scvo)
	for ipiece, piece in enumerate(rule_arr):
		if ipiece > rule_level:
			return True
		if piece != perm_arr[ipiece]:
			return False
	return True

def make_rule_arr(rule):
	el = rule[0]
	core = copy.deepcopy(rule)
	if el[0] == rules.rec_def_type.conn and el[1] == rules.conn_type.AND:
		core = core[1:-1]
	start, end = -1, -1
	b_inside = False
	core_arr = []
	for iel, el in enumerate(core):
		if not b_inside and el[0] == rules.rec_def_type.conn and el[1] == rules.conn_type.start:
			b_inside = True
			start = iel
		elif b_inside and el[0] == rules.rec_def_type.conn and el[1] == rules.conn_type.end:
			b_inside = False
			end = iel
			core_arr.append(core[start:end])
	return core_arr

def match_rule_part(glv_dict, rule, perm):
	b_match = True
	for iel, el in enumerate(rule):
		if el[0] != rules.rec_def_type.like:
			continue
		vec_rule = glv_dict[el[1]]
		vec_perm = glv_dict[perm[iel][1]]
		cd = sum([vec_perm[i] * rule_val for i, rule_val in enumerate(vec_rule)])
		if cd < (el[2] - config.c_cd_epsilon):
			b_match = False
			break

	return b_match

def match_partial_rule(glv_dict, rule, perm, rule_level):
	perm_arr = make_rule_arr(perm)
	rule_arr = make_rule_arr(rule)
	for ipiece, piece in enumerate(rule_arr):
		if ipiece > rule_level:
			return True
		if not match_rule_part(glv_dict, piece, perm_arr[ipiece]):
			return False
	return True
	# b_match = True
	#
	# return b_match



	# 	if rule_scvo[0:2] == 'ca':
	# 	srule = str(rule_scvo[2:])
	# 	rule_arr = []
	# 	pieces = srule.split('cs')[1:]
	# 	if rule_level == 0 and perm_scvo[0:2] == 'cs':
	# 		return perm_scvo[2:] == pieces[0]
	# 	sperm = str(perm_scvo[2:])
	# 	perm_pieces = sperm.split('cs')[1:]
	# 	for ipiece, piece in enumerate(pieces):
	# 		if ipiece > rule_level:
	# 			return True
	# 		if piece != perm_pieces[ipiece]:
	# 			return False
	# 	return True
	# else:
	# 	if rule_level == 0:
	# 		return perm_scvo == rule_scvo

# A phrase list in this case is a list of els, each with just an obj
# Different phrases are simply structured as a list without start, end and AND
def make_rec_from_phrase_arr(phrase_list, b_force_AND = False):
	new_phrase_list = []

	if len(phrase_list) > 1 or b_force_AND:
		new_phrase_list.append([rec_def_type.conn, conn_type.AND])

	for phrase in phrase_list:
		new_phrase_list.append([rec_def_type.conn, conn_type.start])
		for iel, el in enumerate(phrase):
			new_phrase_list.append(el)
		new_phrase_list.append([rec_def_type.conn, conn_type.end])

	if len(phrase_list) > 1 or b_force_AND:
		new_phrase_list.append([rec_def_type.conn, conn_type.end])

	return new_phrase_list


def make_rec_from_phrase_list(phrase_list, b_force_AND = False):
	new_phrase_list, vars_dict = [], dict()

	if len(phrase_list) > 1 or b_force_AND:
		new_phrase_list.append([rec_def_type.conn, conn_type.AND])

	for phrase in phrase_list:
		new_phrase_list.append([rec_def_type.conn, conn_type.start])
		for iel, el in enumerate(phrase):
			if el[0] == rec_def_type.var:
				print('Error! Assuming the input phrase list contains no vars. Exiting!')
				exit()
			if el[0] != rec_def_type.obj:
				new_phrase_list.append(el)
				continue
			word = el[1]
			idx = vars_dict.get(word, None)
			if idx == None:
				# new_phrase_list.append([rec_def_type.var, len(vars_dict)])
				vars_dict[word] = len(new_phrase_list)
				new_phrase_list.append(el)
			else:
				new_phrase_list.append([rec_def_type.var, idx])
		new_phrase_list.append([rec_def_type.conn, conn_type.end])

	if len(phrase_list) > 1 or b_force_AND:
		new_phrase_list.append([rec_def_type.conn, conn_type.end])

	return new_phrase_list, vars_dict

def make_perm_preconds_rec(rule_base, one_perm, story_db, b_force_AND=False):
	phrase_list = list(rule_base)
	for iphrase in one_perm:
		phrase_list += [story_db[iphrase].phrase()]

	new_phrase_list, vars_dict = make_rec_from_phrase_list(phrase_list, b_force_AND)
	return new_phrase_list, vars_dict, phrase_list

	# move_rec =