from __future__ import print_function
import sys
import math
import numpy as np
import random
import itertools

import config
import rules
import story
import cascade
from rules import conn_type
import els
import dmlearn
import ykmeans
import utils
import makerecs as mr

class cl_match_rec_grp(object):
	def __init__(self, first_rec):
		self.__success_eid_list = []
		self.__fail_eid_list = []
		self.__perm_list = [first_rec]
		self.__match_rec = first_rec

	def test(self, glv_dict, el_set_arr, preconds_rec):
		return mr.match_instance_to_rule(glv_dict, el_set_arr, self.__match_rec, preconds_rec)

	# This is only called after the new perm_rec was tested to be inside the match_rec
	def add_perm(self, preconds_rec):
		self.__perm_list.append(preconds_rec)

	def set_match_rec(self, match_rec):
		self.__match_rec = match_rec

	def get_perm_list(self):
		return self.__perm_list


class cl_gens_grp(object):
	def __init__(self, gens_rec, preconds_rec=None):
		if preconds_rec:
			self.__perm_list = [preconds_rec]
		self.__gens_rec = gens_rec
		# add a match rec whenever you add a gg
		# self.__mrg_list = [cl_match_rec_grp(preconds_rec)]

	def gens_matches(self, gens_rec):
		return mr.match_gens_phrase(self.__gens_rec, gens_rec)

	def add_perm(self, preconds_rec):
		self.__perm_list.append(preconds_rec)
		# mr.make_rule_grp(glv_dict, self.__perm_list, el_set_arr)

	def get_gens_rec(self):
		return self.__gens_rec

	def print_gens_rec(self, igg, def_article_dict):
		out_str = ''
		out_str = els.print_phrase(self.__gens_rec, self.__gens_rec, out_str, def_article_dict)
		print(str(igg), ':',out_str)

	# def test_mrg_list(self, preconds_rec, event_result):
	# 	for mrg in self.__mrg_list:
	# 		if mrg.test(preconds_rec, event_result):
	# def get_mrg_list(self):
	# 	return self.__mrg_list

class cl_prov_gens_grp(cl_gens_grp):
	def __init__(self, gens_rec, preconds_rec, eid):
		super(cl_prov_gens_grp, self).__init__(gens_rec, preconds_rec)
		self.__eid_set = set([eid])
		self.__b_graduated = False

	def add_perm(self, preconds_rec, eid):
		super(cl_prov_gens_grp, self).add_perm(preconds_rec)
		self.__eid_set.add(eid)
		if not self.__b_graduated and len(self.__eid_set) > config.c_gg_graduate_len:
			self.__b_graduated = True
			return True

		return False

	def print_gens_rec(self, igg, def_article_dict):
		super(cl_prov_gens_grp, self).print_gens_rec(igg, def_article_dict)
		print('eid set:', self.__eid_set)

class cl_templ_grp(object):
	# __slots__='__len', '__templ_grp_list'
	glv_dict = []
	glv_len = -1

	def __init__(self, templ_len, scvo, preconds_rec, gens_rec, event_result, eid):
		self.__templ_len = templ_len
		self.__scvo = scvo
		self.__olen = mr.get_olen(scvo)
		pgg = cl_prov_gens_grp(gens_rec=gens_rec, preconds_rec=preconds_rec, eid=eid)
		self.__pgg_list = [pgg]
		self.__perm_list = [preconds_rec]
		self.__perm_result_list = [event_result]
		self.__perm_eid_list = [eid]
		self.__perm_ipgg_arr = [0]
		self.__perm_vec_list = [mr.make_vec(self.glv_dict, preconds_rec, self.__olen, self.glv_len)]
		var_scope = 'templ'+str(templ_len)+scvo
		vec_len = self.__olen * self.glv_len
		# ph_input, v_W, t_y = dmlearn.build_templ_nn(var_scope, vec_len, b_reuse=False)
		# op_train_step, t_err, v_r1, v_r2, op_r1, op_r2, ph_numrecs, ph_o = dmlearn.create_tmpl_dml_tensors(t_y, var_scope)
		self.__nn_params = []
		self.__nn_params += dmlearn.build_templ_nn(var_scope, vec_len, b_reuse=False)
		self.__nn_params += dmlearn.create_tmpl_dml_tensors(self.__nn_params[2], var_scope)
		# self.__nn_params = [ph_input, v_W, t_y, op_train_step, t_err, v_r1, v_r2, op_r1, op_r2, ph_numrecs, ph_o]
		self.__db_valid = False
		self.__b_db_graduated = False
		self.__gg_list = []

	def get_nn_params(self):
		return self.__nn_params

	def scvo(self):
		return self.__scvo

	def find_pgg(self, gens_rec):
		for ipgg, pgg in enumerate(self.__pgg_list):
			if pgg.gens_matches(gens_rec):
				return pgg, ipgg
		return None, -1

	# unusually, the following returns the index of the newly added gg
	# def add_pgg(self, gens_grp, preconds_rec, perm_result):
	def add_pgg(self, gens_grp):
		ipgg = len(self.__pgg_list)
		self.__pgg_list.append(gens_grp)
		# self.__perm_list.append(preconds_rec)
		# self.__perm_vec_list.append(mr.make_vec(self.glv_dict, preconds_rec, self.__olen, self.glv_len))
		# self.__perm_result_list.append(perm_result)
		# self.__perm_ipgg_arr.append(ipgg)
		return ipgg

	def add_perm(self, preconds_rec, gens_rec, perm_result, eid):
		self.__perm_list.append(preconds_rec)
		self.__perm_result_list.append(perm_result)
		self.__perm_vec_list.append(mr.make_vec(self.glv_dict, preconds_rec, self.__olen, self.glv_len))
		self.__perm_eid_list.append(eid)

		b_pgg_needs_graduating = False
		perm_pgg, perm_ipgg = self.find_pgg(gens_rec=gens_rec)
		if not perm_pgg:
			perm_pgg = cl_prov_gens_grp(gens_rec=gens_rec, preconds_rec=preconds_rec, eid=eid)
			perm_ipgg = self.add_pgg(perm_pgg)
		else:
			b_pgg_needs_graduating = perm_pgg.add_perm(preconds_rec, eid)
			# perm_templ.add_perm(preconds_rec=perm_preconds_list[iperm], gens_rec=perm_gens_list[iperm], igg=perm_igg)

		self.__perm_ipgg_arr.append(perm_ipgg)

		if b_pgg_needs_graduating:
			if not self.__b_db_graduated:
				gg_null = cl_gens_grp([])
				self.__gg_list.append(gg_null)
			gg = cl_gens_grp(perm_pgg.get_gens_rec())
			self.__gg_list.append(gg)
			self.recalib_ggs()
		else:
			if self.__b_db_graduated:
				self.__perm_igg_arr.append(self.assign_gg(preconds_rec, perm_result))

	def get_num_pggs(self):
		return len(self.__pgg_list)

	def get_num_perms(self):
		return len(self.__perm_pigg_arr)

	def get_match_score(self, preconds_rec, event_result, b_real_score=True):
		if self.__db_valid:
			if b_real_score:
				print('Calculating score:')
			perm_vec = mr.make_vec(self.glv_dict, preconds_rec, self.__olen, self.glv_len)
			return dmlearn.get_score(preconds_rec, perm_vec, self.__nd_W, self.__nd_db, self.__gg_list,
									 self.__perm_igg_arr, self.__perm_eid_list, event_result)
		else:
			return 0.0

		return 1.0

	def do_learn(self, sess):
		# if len(self.__gg_list) > 1 and len(self.__perm_igg_arr) > 5:
		if self.__b_db_graduated:
			self.__nd_W, self.__nd_db = dmlearn.do_templ_learn(sess, self.__nn_params, self.__perm_vec_list, self.__perm_igg_arr, self.__scvo)
			self.__db_valid = True

	def printout(self, def_article_dict):
		print('pggs in order:')
		for ipgg, pgg in enumerate(self.__pgg_list):
			pgg.print_gens_rec(ipgg, def_article_dict)
		if self.__b_db_graduated:
			print('ggs in order:')
			for igg, gg in enumerate(self.__gg_list):
				gg.print_gens_rec(igg, def_article_dict)
		print('all perm recs:')
		for irec, rec in enumerate(self.__perm_list):
			out_str = str(irec) + ' '
			out_str = els.print_phrase(rec, rec, out_str, def_article_dict)
			print(out_str)
			out_str = str(irec) + ' '
			out_str = els.print_phrase(rec, self.__perm_result_list[irec], out_str, def_article_dict)
			print(out_str)
			print('ipgg:', self.__perm_ipgg_arr[irec], 'eid:', self.__perm_eid_list[irec])
			if self.__b_db_graduated:
				print('assigned igg:', self.__perm_igg_arr[irec])

	def assign_gg(self, perm_rec, perm_result):
		b_success_found = False
		for igg, gg in enumerate(self.__gg_list):
			if igg == 0:
				continue
			generated_result = mr.get_result_for_cvo_and_rec(perm_rec,
															 gg.get_gens_rec())
			if mr.match_rec_exact(generated_result[1:-1], perm_result):
				b_success_found = True
				return igg
		return 0
		# if b_success_found:
		# 	self.__perm_igg_arr.append(igg)
		# else:
		# 	self.__perm_igg_arr.append(0)

	def recalib_ggs(self):
		self.__perm_igg_arr = []
		for irec, rec in enumerate(self.__perm_list):
			self.__perm_igg_arr.append(self.assign_gg(rec, self.__perm_result_list[irec]))
		self.__b_db_graduated = True



class cl_len_grp(object):
	# __slots__='__len', '__templ_grp_list'

	def __init__(self, init_len, first_scvo, preconds_rec, gens_rec, event_result, eid):
		self.__len = init_len
		# gg = cl_gens_grp(gens_rec, preconds_rec)
		self.__templ_grp_list = [cl_templ_grp(init_len, first_scvo, gens_rec=gens_rec, preconds_rec=preconds_rec,
											  event_result=event_result, eid=eid)]

	def add_templ(self, templ):
		self.__templ_grp_list.append(templ)

	def len(self):
		return self.__len

	def get_templ_grp_list(self):
		return self.__templ_grp_list

	def find_templ(self, scvo):
		for templ_grp in self.__templ_grp_list:
			if templ_grp.scvo() == scvo:
				return templ_grp
		return None

	def add_templ(self, templ_grp):
		self.__templ_grp_list.append(templ_grp)


