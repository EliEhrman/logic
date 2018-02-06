from __future__ import print_function
# import sys
# import math
import numpy as np
# import random
# import itertools
import csv
from StringIO import StringIO

import config
import rules
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

	return new_conds, vars_dict

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
			if el[1] == rules.conn_type.start:
				lcvo += ['s']
			if el[1] == rules.conn_type.end:
				lcvo += ['e']
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
		elif lelf[0] == 'v':
			el = [rules.rec_def_type.var]
			el += [int(lelf[1])]
		elif lelf[0] == 'o':
			el = [rules.rec_def_type.obj]
			el += [lelf[1]]
		elif lelf[0] == 'l':
			el = [rules.rec_def_type.like]
			el += [lelf[1], float(lelf[2])]
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

def get_result_for_cvo_and_rec(preconds_rec, gens_rec):
	# sleft = str(scvo)
	#
	# while sleft != '':
	# 	c, sleft = sleft[0], sleft[1:]
	result = []
	for el in gens_rec:
		if el[0] == rules.rec_def_type.var:
			ivar = el[1]
			result.append(preconds_rec[ivar])
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

def make_rule_grp(glv_dict, rule_cluster):
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
		best_cd = -1.0
		best_name = 'not found!'
		for sym, one_vec in glv_dict.iteritems():
			cd = sum([one_vec[i] * set_val for i, set_val in enumerate(vec_avg)])
			if cd > best_cd:
				best_name, best_cd = sym, cd
		rule_phrase.append([rules.rec_def_type.like, best_name, min_cd])
		# el_set_arr.append([vec_avg, min_cd, vec_list])

	return rule_phrase


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
	_, templ_len, templ_scvo, igg = sig
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
