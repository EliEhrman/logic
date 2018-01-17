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
	def __init__(self, gens_rec, igg, templ_iperm=None):
		if templ_iperm == None:
			self.__iperm_list = []
		else:
			self.__iperm_list = [templ_iperm]
		self.__gens_rec = gens_rec
		self.__num_points = 0
		self.__b_confirmed = False
		self.__perm_match_list = []
		self.__igg = igg # index number of gg or pgg in template container
		self.__num_tests = 0.0
		self.__num_successes = 0.0
		self.__num_perm_adds_till_next_learn = 0

	def gens_matches(self, gens_rec):
		return mr.match_gens_phrase(self.__gens_rec, gens_rec)

	def add_perm(self, templ_iperm):
		self.__iperm_list.append(templ_iperm)
		self.__num_perm_adds_till_next_learn -= 1

	def mark_templ_perm_matched(self, b_match):
		self.__perm_match_list.append(b_match)

	def get_gens_rec(self):
		return self.__gens_rec

	def print_gens_rec(self, igg, def_article_dict):
		out_str = ''
		if self.__gens_rec != None:
			out_str = els.print_phrase(self.__gens_rec, self.__gens_rec, out_str, def_article_dict)
		print(str(igg), ':',out_str)

	def add_point(self):
		self.__num_points += 1
		if self.__num_points >= config.c_gg_confirm_thresh:
			if not self.__b_confirmed:
				print('gg now confirmed for igg:', self.__igg, 'gens rec:', self.__gens_rec)
				self.__b_confirmed = True

		return self.__b_confirmed

	def clear_perms(self):
		self.__perm_list = []
		self.__iperm_list = []
		self.__perm_match_list = []

	def set_match_limits(self, templ_perm_list, templ_perm_igg_arr, templ_len, glv_len, glv_dict):
		print('Testing for match limits for templ_olen, gg: ', templ_len, self.__igg)
		self.__thresh_cd = 1.0
		for one_iperm in self.__iperm_list:
			perm_vec = mr.make_vec(glv_dict, templ_perm_list[one_iperm], templ_len, glv_len)
			if config.c_b_nbns:
				perm_vec = dmlearn.modify_vec_for_success(perm_vec)
			min_cd = dmlearn.get_score_stats(one_iperm, perm_vec, self.__nd_W, self.__nd_db, templ_perm_igg_arr)
			if min_cd < self.__thresh_cd:
				self.__thresh_cd = min_cd
		print('Thresh set at:', self.__thresh_cd)
		print('Comparing to the unsorted:')
		for one_iperm in range(len(templ_perm_list)):
			print(one_iperm, ': is', (one_iperm in self.__iperm_list), '.', templ_perm_list[one_iperm])
			perm_vec = mr.make_vec(glv_dict, templ_perm_list[one_iperm], templ_len, glv_len)
			if config.c_b_nbns:
				perm_vec = dmlearn.modify_vec_for_success(perm_vec)
			dmlearn.get_score_stats(one_iperm, perm_vec, self.__nd_W, self.__nd_db, templ_perm_igg_arr)

	def init_for_learn(self, vec_len, templ_scvo, igg):
		var_scope = 'gg_'+str(vec_len).rjust(5, '0')+str(igg).rjust(3, '0')+templ_scvo
		self.__nn_params = []
		self.__nn_params += dmlearn.build_templ_nn(var_scope, vec_len, b_reuse=False)
		self.__nn_params += dmlearn.create_tmpl_dml_tensors(self.__nn_params[2], var_scope)

	def do_learn(self, sess, templ_perm_vec_list, templ_perm_list, templ_olen, glv_len, glv_dict, el_set_arr):
		if self.__num_perm_adds_till_next_learn > 0:
			print('gg cant learn yet. Still have', self.__num_perm_adds_till_next_learn, ' add_perms to go.')

		print('Learning for gg:', self.__igg)
		self.__nd_W, self.__nd_db, self.__b_lrn_success = \
			dmlearn.do_templ_learn(sess, self.__nn_params, templ_perm_vec_list, self.__perm_match_list)
		if self.__b_lrn_success:
			self.set_match_limits(templ_perm_list, self.__perm_match_list, templ_olen, glv_len, glv_dict)
			self.__rule_grp = self.make_rule_grp(glv_dict, templ_perm_list, el_set_arr)
		self.__num_perm_adds_till_next_learn = config.c_gg_learn_every_num_perms
		# for one_gg in self.__gg_list:
		# 	one_gg.set_match_limits(self.__nd_W, self.__nd_db, self.__perm_list, self.__perm_igg_arr, self.__olen, self.glv_len, self.glv_dict)
		# self.__db_valid = True

	def make_rule_grp(self, glv_dict, templ_perm_list, el_set_arr):
		return mr.make_rule_grp(glv_dict, [templ_perm_list[iperm] for iperm in self.__iperm_list], el_set_arr)

	def get_perm_match_list(self):
		return self.__perm_match_list

	def get_perm_match_val(self, iperm):
		if len(self.__perm_match_list) <= iperm:
			return 'NA'
		return self.__perm_match_list[iperm]

	def get_igg(self):
		return self.__igg

	def get_match_score(self, preconds_rec, perm_vec, event_result_list, event_result_score_list,
						templ_len, templ_scvo, result_confirmed_list, gg_confirmed_list,
						expected_but_not_found_list):
		if not self.__b_lrn_success:
			print('Cannot provide score for for failed learning igg:', self.__igg, 'match list:', self.__perm_match_list)
			return event_result_score_list

		print('Calculating score for igg:', self.__igg, 'match list:', self.__perm_match_list)
		b_hit, b_success, expected_but_not_found = \
			dmlearn.get_gg_score(	preconds_rec, perm_vec, self.__nd_W, self.__nd_db ,
									self.__igg, self.__perm_match_list,
									self.__thresh_cd, self.get_gens_rec(), event_result_list,
									event_result_score_list, templ_len, templ_scvo,
									self.__b_confirmed, result_confirmed_list, gg_confirmed_list,
									self.__num_successes / self.__num_tests if self.__num_tests else 0.0)
		if b_hit:
			self.__num_tests += 1.0
			if b_success:
				self.__num_successes += 1.0
			else:
				expected_but_not_found_list.append(expected_but_not_found)
		return event_result_score_list, expected_but_not_found_list

	def get_rule_grp(self):
		return self.__rule_grp


	# def test_mrg_list(self, preconds_rec, event_result):
	# 	for mrg in self.__mrg_list:
	# 		if mrg.test(preconds_rec, event_result):
	# def get_mrg_list(self):
	# 	return self.__mrg_list

class cl_prov_gens_grp(cl_gens_grp):
	def __init__(self, gens_rec, templ_iperm, igg, eid):
		# super(cl_prov_gens_grp, self).__init__(gens_rec, igg, templ_iperm)
		cl_gens_grp.__init__(self, gens_rec, igg, templ_iperm)
		self.__eid_set = set([eid])
		self.__b_graduated = False

	def add_perm(self, templ_iperm, eid):
		super(cl_prov_gens_grp, self).add_perm(templ_iperm)
		self.__eid_set.add(eid)
		if not self.__b_graduated and len(self.__eid_set) > config.c_gg_graduate_len:
			self.__b_graduated = True
			print('gg graduated. igg:', self.get_igg())
			return True

		return False

	def print_gens_rec(self, igg, def_article_dict):
		super(cl_prov_gens_grp, self).print_gens_rec(igg, def_article_dict)
		print('eid set:', self.__eid_set)

class cl_templ_grp(object):
	# __slots__='__len', '__templ_grp_list'
	glv_dict = []
	glv_len = -1

	def __init__(self, templ_len, scvo, preconds_rec, gens_rec_list, event_result_list, eid):
		self.__templ_len = templ_len
		self.__scvo = scvo
		self.__olen = mr.get_olen(scvo)
		pgg_list = []
		for one_gens in gens_rec_list:
			pgg_list.append(cl_prov_gens_grp(gens_rec=one_gens, igg=len(pgg_list), templ_iperm=0, eid=eid))
		self.__pgg_list = pgg_list
		self.__perm_list = [preconds_rec]
		self.__perm_result_list = [event_result_list]
		self.__perm_eid_list = [eid]
		self.__perm_ipgg_arr = [range(len(pgg_list))] # list in list here
		self.__perm_vec_list = [mr.make_vec(self.glv_dict, preconds_rec, self.__olen, self.glv_len)]
		# var_scope = 'templ'+str(templ_len)+scvo
		# vec_len = self.__olen * self.glv_len
		# self.__nn_params = []
		# self.__nn_params += dmlearn.build_templ_nn(var_scope, vec_len, b_reuse=False)
		# self.__nn_params += dmlearn.create_tmpl_dml_tensors(self.__nn_params[2], var_scope)
		# self.__nn_params = [ph_input, v_W, t_y, op_train_step, t_err, v_r1, v_r2, op_r1, op_r2, ph_numrecs, ph_o]
		self.__db_valid = False
		self.__b_db_graduated = False
		self.__gg_list = []
		self.__b_confirmed = False
		self.__num_perm_adds_till_next_learn = -1

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

	def add_perm(self, preconds_rec, gens_rec_list, perm_result_list, eid):
		iperm = len(self.__perm_list)
		self.__perm_list.append(preconds_rec)
		self.__perm_result_list.append(perm_result_list)
		self.__perm_vec_list.append(mr.make_vec(self.glv_dict, preconds_rec, self.__olen, self.glv_len))
		self.__perm_eid_list.append(eid)
		self.__perm_ipgg_arr.append([])

		if gens_rec_list == []:
			# skipping reducing count of adds till next learn
			return

		b_needs_recalib = False

		for igens, one_gens in enumerate(gens_rec_list):
			b_pgg_needs_graduating = False
			perm_pgg, perm_ipgg = self.find_pgg(gens_rec=one_gens)
			if not perm_pgg:
				perm_pgg = cl_prov_gens_grp(gens_rec=one_gens, igg=len(self.__pgg_list), templ_iperm=iperm, eid=eid)
				perm_ipgg = self.add_pgg(perm_pgg)
			else:
				b_pgg_needs_graduating = perm_pgg.add_perm(iperm, eid)
				# perm_templ.add_perm(preconds_rec=perm_preconds_list[iperm], gens_rec=perm_gens_list[iperm], igg=perm_igg)

			# if igens == 0:
			# 	self.__perm_ipgg_arr.append([perm_ipgg])
			# else:
			# 	self.__perm_ipgg_arr[-1].append(perm_ipgg)
			self.__perm_ipgg_arr[-1].append(perm_ipgg)

			if b_pgg_needs_graduating:
				print('gg graduated in template:', self.__scvo, 'len:', self.__templ_len)
				if len(self.__gg_list) == 0:
					gg_null = cl_gens_grp([], 0)
					self.__gg_list = [gg_null]
				gg = cl_gens_grp(perm_pgg.get_gens_rec(), len(self.__gg_list))
 				gg.init_for_learn(self.__olen * self.glv_len, self.__scvo, len(self.__gg_list))
				self.__gg_list.append(gg)
				b_needs_recalib = True
				self.__num_perm_adds_till_next_learn = 0

		if b_needs_recalib:
			self.recalib_ggs()
		else:
			if self.__b_db_graduated:
				self.assign_gg(iperm, preconds_rec, perm_result_list)
				self.__num_perm_adds_till_next_learn -= 1
				# self.__perm_igg_arr.append(self.assign_gg(iperm, preconds_rec, perm_result_list))

	def get_num_pggs(self):
		return len(self.__pgg_list)

	def get_num_perms(self):
		return len(self.__perm_pigg_arr)

	def get_match_score(self, def_article_dict, preconds_rec, event_result_list, event_result_score_list,
						result_confirmed_list, gg_confirmed_list, b_real_score=True):
		# rewrite for result list and gg list
		if self.__db_valid:
			if b_real_score:
				print('Calculating score for all ggs in template:', self.__scvo, 'len:', self.__templ_len)
				self.printout(def_article_dict)
				score_list = event_result_score_list
			else:
				score_list = [[] for _ in event_result_score_list]

			expected_but_not_found_list = []
			perm_vec = mr.make_vec(self.glv_dict, preconds_rec, self.__olen, self.glv_len)
			for one_gg in self.__gg_list[1:]:
				one_gg.get_match_score(preconds_rec, perm_vec, event_result_list, event_result_score_list,
									   self.__templ_len, self.__scvo, result_confirmed_list, gg_confirmed_list,
									   expected_but_not_found_list)
			# 	one_gg.get_match_score(sess, self.__perm_vec_list, self.__perm_list, self.__olen, self.glv_len, self.glv_dict)
			# return dmlearn.get_score(preconds_rec, perm_vec, self.__nd_W, self.__nd_db, self.__gg_list,
			# 						 self.__perm_igg_arr, self.__perm_eid_list, event_result_list,
			# 						 score_list, self.__templ_len, self.__scvo)
		else:
			return event_result_score_list, []

		return event_result_score_list, expected_but_not_found_list

	def do_learn(self, def_article_dict, sess, el_set_arr):
		# if len(self.__gg_list) > 1 and len(self.__perm_igg_arr) > 5:
		if not self.__b_db_graduated:
			return

		if self.__num_perm_adds_till_next_learn > 0:
			print('Not learning yet. Still have', self.__num_perm_adds_till_next_learn, ' add_perms to go for ', self.__scvo)
			return


		print('Learning all ggs in template:', self.__scvo, 'len:', self.__templ_len)
		self.printout(def_article_dict)
		for one_gg in self.__gg_list[1:]:
			one_gg.do_learn(sess, self.__perm_vec_list, self.__perm_list, self.__olen, self.glv_len, self.glv_dict, el_set_arr)
		# for one_gg in self.__gg_list:
		# 	one_gg.set_match_limits(self.__nd_W, self.__nd_db, self.__perm_list, self.__perm_igg_arr, self.__olen, self.glv_len, self.glv_dict)
		self.__num_perm_adds_till_next_learn = config.c_templ_learn_every_num_perms
		self.__db_valid = True

	def printout(self, def_article_dict):
		if not self.__b_db_graduated:
			return

		print('printout for template:', self.__scvo, 'len:', self.__templ_len)
		print('pggs in order:')
		for ipgg, pgg in enumerate(self.__pgg_list):
			pgg.print_gens_rec(ipgg, def_article_dict)
		if self.__b_db_graduated:
			print('ggs in order:')
			for igg, gg in enumerate(self.__gg_list):
				gg.print_gens_rec(igg, def_article_dict)
				print('perm_match_list: ', gg.get_perm_match_list())
		print('all perm recs:')
		for irec, rec in enumerate(self.__perm_list):
			out_str = str(irec) + ' '
			out_str = els.print_phrase(rec, rec, out_str, def_article_dict)
			print(out_str)
			for perm_result in self.__perm_result_list[irec]:
				out_str =  ''
				out_str = els.print_phrase(rec, perm_result, out_str, def_article_dict)
				print('\t', str(irec), out_str)
			print('\tipgg:', self.__perm_ipgg_arr[irec], 'eid:', self.__perm_eid_list[irec])
			if self.__b_db_graduated:
				print('assigned igg:', [[igg, gg.get_perm_match_val(irec)] for igg, gg in enumerate(self.__gg_list)])


	def assign_gg(self, perm_irec, perm_rec, perm_result_list):
		# igg_list = []
		# igg_arr = [0.0 for _ in self.__gg_list]
		# nd_igg = np.zeros([len(self.__gg_list)], np.float32)
		# Each gg must have its own record of whether the record matched or not
		for igg, gg in enumerate(self.__gg_list):
			if igg == 0:
				continue
			generated_result = mr.get_result_for_cvo_and_rec(perm_rec,
															 gg.get_gens_rec())
			b_success_found = False
			for one_result in perm_result_list:
				if mr.match_rec_exact(generated_result[1:-1], one_result):
					b_success_found = True
					# igg_list.append(igg)
					# nd_igg[igg] = 1.0
					break
			gg.mark_templ_perm_matched(b_success_found)
			if b_success_found:
				gg.add_perm(perm_irec)
		# 	# igg_list.append(0)
			# 	nd_igg[0] = 1.0
		# nd_igg = dmlearn.l2_norm_arr(nd_igg)
		# return igg_list
		# return nd_igg
		# if b_success_found:
		# 	self.__perm_igg_arr.append(igg)
		# else:
		# 	self.__perm_igg_arr.append(0)

	def recalib_ggs(self):
		self.__perm_igg_arr = []
		for one_gg in self.__gg_list:
			one_gg.clear_perms()

		for irec, rec in enumerate(self.__perm_list):
			self.assign_gg(irec, rec, self.__perm_result_list[irec])
			# self.__perm_igg_arr.append(self.assign_gg(irec, rec, self.__perm_result_list[irec]))
		self.__b_db_graduated = True

	def add_point(self, igg):
		if self.__gg_list[igg].add_point():
			print('gg confirmed in templ group with scvo, len;', self.__scvo, 'len:', self.__templ_len)
			self.__b_confirmed = True

	def get_gg(self, igg):
		if igg >= len((self.__gg_list)):
			return None
		return self.__gg_list[igg]

class cl_len_grp(object):
	# __slots__='__len', '__templ_grp_list'

	def __init__(self, init_len, first_scvo, preconds_rec, gens_rec_list, event_result_list, eid):
		self.__len = init_len
		# gg = cl_gens_grp(gens_rec, preconds_rec)
		self.__templ_grp_list = [cl_templ_grp(init_len, first_scvo, gens_rec_list=gens_rec_list, preconds_rec=preconds_rec,
											  event_result_list=event_result_list, eid=eid)]

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


