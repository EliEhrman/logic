from __future__ import print_function
import csv
import sys
import math
import numpy as np
import random
import itertools
from StringIO import StringIO

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

def get_next(ldata):
	# return next(ldata)
	return ldata.pop(0)

class cl_gens_grp(object):
	glv_len = -1

	def __init__(self, b_from_load, igg, gens_rec=None, b_blocking=None, templ_iperm=None, eid=-1):
		if templ_iperm == None:
			self.__iperm_list = []
		else:
			self.__iperm_list = [templ_iperm]
		self.__gens_rec = gens_rec
		self.__num_points = 0
		self.__b_validated = False
		self.__b_confirmed = False
		self.__perm_match_list = []
		self.__igg = igg # index number of gg or pgg in template container
		self.__num_tests = 0.0
		self.__num_successes = 0.0
		self.__num_perm_adds_till_next_learn = 0
		self.__scoring_eid_set = set()
		self.__b_lrn_success = False
		self.__penalty = config.c_gg_starting_penalty
		self.__last_eid = -1
		self.__thresh_cd = -1.0
		self.__rule_grp = None
		self.__b_blocking = b_blocking
		self.__cont_id = -1
		# self.__eid_last_assign = -1
		self.__eid_last_add = eid
		self.__eid_last_learn = -1
		self.__eid_last_score = -1

	def gens_matches(self, gens_rec, b_blocking):
		if b_blocking != self.__b_blocking:
			return False
		return mr.match_gens_phrase(self.__gens_rec, gens_rec)

	def add_perm(self, templ_iperm, eid):
		if eid == self.__eid_last_add:
			return False

		self.__eid_last_add = eid

		# if self.__b_lrn_success and not utils.prob_for_penalty(self.__penalty):
		if not utils.prob_for_penalty(self.__penalty):
			return False

		self.__iperm_list.append(templ_iperm)
		if eid != self.__last_eid:
			self.__num_perm_adds_till_next_learn -= 1
			self.__last_eid = eid

		return True

	def mark_templ_perm_matched(self, b_match):
		self.__perm_match_list.append(b_match)

	def get_gens_rec(self):
		return self.__gens_rec

	def get_b_blocking(self):
		return self.__b_blocking

	def set_b_blocking(self, b_blocking):
		self.__b_blocking = b_blocking

	def get_eid_last_add(self):
		return self.__eid_last_add

	def apply_penalty(self, penalty):
		if penalty > 0 and self.__num_points > 0:
			self.__num_points -= 1
			return
		self.__penalty += penalty
		if self.__penalty < config.c_gg_starting_penalty:
			self.__penalty = config.c_gg_starting_penalty

	def print_gens_rec(self, igg, def_article_dict):
		out_str = ''
		if self.__gens_rec != None:
			out_str = els.print_phrase(self.__gens_rec, self.__gens_rec, out_str, def_article_dict)
		print(str(igg), ':', ('blocking' if self.__b_blocking else ''), out_str)

	def print_stats(self, def_article_dict):
		print('Learn status:', self.__b_lrn_success, '. Scoring eid set:', self.__scoring_eid_set,
			  '\n Num perms till next learn:', self.__num_perm_adds_till_next_learn,
			  '\n Confirmed:', self.__b_confirmed,
			  '\n Num successes:', self.__num_successes,
			  '\n Num Tests:', self.__num_tests,
			  '\n Num Points:', self.__num_points,
			  '\n CD Thresh:', self.__thresh_cd,
			  '\n penalty:', self.__penalty,
			  '\n Last eid:', self.__last_eid,
			  )
		if self.__rule_grp != None:
			out_str = ''
			if self.__gens_rec != None:
				out_str = els.print_phrase(self.__rule_grp, self.__gens_rec, out_str, def_article_dict)
			print(('Blocking gens:' if self.__b_blocking else ''), out_str)

	def add_point(self):
		if self.__penalty > 0:
			self.apply_penalty(-config.c_points_penalty_value)
			return
		self.__num_points += 1
		if self.__num_points >= config.c_gg_validate_thresh:
			if not self.__b_validated:
				print('gg now validated for igg:', self.__igg, 'gens rec:', self.__gens_rec)
				self.__b_validated = True
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
		# do I have to add templ result_blocking?
		print('Testing for match limits for templ_olen, gg: ', templ_len, self.__igg)
		self.__thresh_cd = 1.0
		for one_iperm in self.__iperm_list:
			perm_vec = mr.make_vec(glv_dict, templ_perm_list[one_iperm], templ_len, glv_len)
			if config.c_b_nbns:
				perm_vec = dmlearn.modify_vec_for_success(perm_vec)
			min_cd = dmlearn.get_score_stats(one_iperm, perm_vec, self.__nd_W, self.__nd_db, templ_perm_igg_arr)
			if min_cd < self.__thresh_cd:
				self.__thresh_cd = min_cd
		self.__thresh_cd = config.c_fudge_thresh_cd * self.__thresh_cd
		print('Thresh set at:', self.__thresh_cd)
		# print('Comparing to the unsorted:')
		# for one_iperm in range(len(templ_perm_list)):
		# 	if random.random() > 0.2:
		# 		continue
		# 	print(one_iperm, ': is', (one_iperm in self.__iperm_list), '.', templ_perm_list[one_iperm])
		# 	perm_vec = mr.make_vec(glv_dict, templ_perm_list[one_iperm], templ_len, glv_len)
		# 	if config.c_b_nbns:
		# 		perm_vec = dmlearn.modify_vec_for_success(perm_vec)
		# 	dmlearn.get_score_stats(one_iperm, perm_vec, self.__nd_W, self.__nd_db,
		# 							templ_perm_igg_arr, b_always_print=True)

	def init_for_learn(self, vec_len, templ_scvo, igg):
		var_scope = 'gg_'+('B' if self.__b_blocking else 'N')+str(vec_len).rjust(5, '0')+str(igg).rjust(3, '0')+templ_scvo
		print('Creating var scope:', var_scope)
		self.__nn_params = []
		self.__nn_params += dmlearn.build_templ_nn(var_scope, vec_len, b_reuse=False)
		self.__nn_params += dmlearn.create_tmpl_dml_tensors(self.__nn_params[2], var_scope)

	def get_will_learn(self, eid):
		if self.__igg == 0:
			return False

		if self.__b_lrn_success:
			return False

		if self.__num_perm_adds_till_next_learn > 0:
			return False

		if self.__cont_id >= 0:
			return False

		if eid == self.__eid_last_learn:
			return False

		return True

	def get_will_score(self, eid):
		if self.__igg == 0:
			return False

		if not self.__b_lrn_success:
			return False

		# if self.__cont_id >= 0:
		# 	return False
		#
		if eid == self.__eid_last_score:
			return False

		return True


	def do_learn(self, b_do_learn, sess, templ_perm_vec_list, templ_perm_list,
				 templ_olen, glv_len, glv_dict, el_set_arr, curr_cont, eid):
		if self.__igg == 0:
			return

		if self.__num_perm_adds_till_next_learn > 0:
			print('gg', self.__igg, 'cant learn yet. Still have', self.__num_perm_adds_till_next_learn, ' add_perms to go.')
			return

		if b_do_learn:
			if eid == self.__eid_last_learn:
				return

			self.__eid_last_learn = eid

			print('Learning for gg:', self.__igg)
			self.__nd_W, self.__nd_db, self.__b_lrn_success = \
				dmlearn.do_gg_learn(sess, self.__nn_params, templ_perm_vec_list,
									self.__perm_match_list, b_must_learn=(self.__num_points > 1))
			if self.__b_lrn_success:
				self.set_match_limits(templ_perm_list, self.__perm_match_list, templ_olen, glv_len, glv_dict)
				self.__rule_grp = self.make_rule_grp(glv_dict, templ_perm_list, curr_cont)

		self.__num_perm_adds_till_next_learn = config.c_gg_learn_every_num_perms


	def make_rule_grp(self, glv_dict, templ_perm_list, curr_cont):
		return mr.make_rule_grp(glv_dict, [templ_perm_list[imatch] for imatch, bmatch in enumerate(self.__perm_match_list) if bmatch], self.glv_len, curr_cont)
		# return mr.make_rule_grp(glv_dict, [templ_perm_list[iperm] for iperm in self.__iperm_list], self.glv_len, curr_cont)


	def get_perm_match_list(self):
		return self.__perm_match_list

	def get_perm_match_val(self, iperm):
		if len(self.__perm_match_list) <= iperm:
			return 'NA'
		return self.__perm_match_list[iperm]

	def get_igg(self):
		return self.__igg

	def get_gens_rec(self):
		return self.__gens_rec

	def set_gens_rec(self, gens_rec):
		self.__gens_rec = gens_rec

	def get_b_learn_success(self):
		return self.__b_lrn_success

	def get_penalty(self):
		return self.__penalty

	def get_match_score(self, preconds_rec, perm_vec, perm_phrases, eid, event_result_list,
						b_blocking, event_result_score_list,
						templ_len, templ_scvo, result_confirmed_list, gg_confirmed_list,
						expected_but_not_found_list):
		if not self.__b_lrn_success:
			print('Cannot provide score for for failed learning igg:', self.__igg, 'match list:', self.__perm_match_list)
			return event_result_score_list, expected_but_not_found_list

		if eid == self.__eid_last_score:
			return event_result_score_list, expected_but_not_found_list

		self.__eid_last_score = eid

		print('Calculating score for igg:', self.__igg, 'of templ grp', templ_scvo,
			  'eid set len', len(self.__scoring_eid_set)) # , 'match list:', self.__perm_match_list)
		b_hit, b_success, expected_but_not_found = \
			dmlearn.get_gg_score(	preconds_rec, perm_vec, perm_phrases, self.__nd_W, self.__nd_db ,
									self.__igg, self.__perm_match_list,
									self.__thresh_cd, self.get_gens_rec(), event_result_list, b_blocking,
									event_result_score_list, templ_len, templ_scvo, self.__b_blocking,
									self.__b_confirmed, result_confirmed_list, gg_confirmed_list,
									self.__num_successes / self.__num_tests if self.__num_tests else 0.0,
									len(self.__scoring_eid_set) > config.c_gg_scoring_eid_thresh)
		if b_hit:
			self.__scoring_eid_set.add(eid)
			self.__num_tests += 1.0
			if b_success:
				self.__num_successes += 1.0
			else:
				if self.__b_validated:
					print('Adding unexpected for ', self.__igg, expected_but_not_found)
					expected_but_not_found_list.append(expected_but_not_found)
				# expected_but_not_found_list.append((preconds_rec, self.get_gens_rec()))
		return event_result_score_list, expected_but_not_found_list

	def get_rule_grp(self):
		return self.__rule_grp

	def get_b_valid(self):
		return self.__b_validated

	def set_cont_id(self, cont_id):
		self.__cont_id = cont_id

	def get_cont_stats(self, templ_len, templ_scvo, igg):
		rule_str = mr.gen_rec_str(self.__rule_grp)
		return [templ_len, templ_scvo, self.__b_blocking, igg, self.__num_successes, self.__num_tests,
				rule_str, self.__cont_id, self]

	def add_to_W_dict(self, cont_id, templ_len, templ_scvo, W_dict):
		if self.__b_lrn_success and self.__nd_W != None:
			W_dict[(cont_id, templ_len, templ_scvo, self.__igg)] = self.__nd_W

	def set_W_from_file(self, nd_W, templ_perm_vec_list):
		self.__nd_W = nd_W
		self.__nd_db = dmlearn.create_db_from_W(nd_W, templ_perm_vec_list, self.__perm_match_list)
		if self.__nd_db == None:
			self.__nd_W = None
			self.__b_lrn_success = False
			return

		self.__b_lrn_success = True

	def save(self, db_csvr):
		db_csvr.writerow(['tgg', 'confirmed', self.__b_confirmed, 'gg id', self.__igg, 'num points', self.__num_points,
						  'num successes', self.__num_successes, 'num tests', self.__num_tests,
						  'penalty points', self.__penalty, 'cd thresh', self.__thresh_cd,
						  'rule grp', mr.gen_rec_str(self.__rule_grp),
						  'gens rec', mr.gen_rec_str(self.get_gens_rec()),
						  'blocking', self.__b_blocking, 'cont id', self.__cont_id])

	def load(self, db_csvr):
		_, _, sb_confirmed, _, sigg, _, snum_points, _, snum_successes, _, snum_tests, \
		_, spenalty, _, s_thresh_cd, _, srule_rec, _, sgens_rec, _, sb_blocking, _, s_cont_id = get_next(db_csvr)
		self.__b_confirmed, self.__igg, self.__num_points = sb_confirmed == 'True', int(sigg), int(snum_points)
		self.__num_successes, self.__num_tests, self.__penalty = float(snum_successes), float(snum_tests), int(spenalty)
		self.__thresh_cd = float(s_thresh_cd)
		self.__gens_rec = mr.extract_rec_from_str(sgens_rec)
		self.__rule_grp = mr.extract_rec_from_str(srule_rec)
		self.__num_perm_adds_till_next_learn = config.c_gg_num_perms_till_learn_on_load
		self.__b_lrn_success, self.__b_blocking = False, sb_blocking == 'True'
		self.__cont_id = int(s_cont_id)
		if self.__num_points >= config.c_gg_validate_thresh:
			if not self.__b_validated:
				print('gg validated on load for igg:', self.__igg, 'gens rec:', self.__gens_rec)
				self.__b_validated = True

	# def test_mrg_list(self, preconds_rec, event_result):
	# 	for mrg in self.__mrg_list:
	# 		if mrg.test(preconds_rec, event_result):
	# def get_mrg_list(self):
	# 	return self.__mrg_list

class cl_prov_gens_grp(cl_gens_grp):
	def __init__(self, b_from_load, igg, gens_rec=None, b_blocking=None, templ_iperm=None, eid=None):
		# super(cl_prov_gens_grp, self).__init__(gens_rec, igg, templ_iperm)
		if b_from_load:
			cl_gens_grp.__init__(self, b_from_load=True, igg=igg )
			self.__eid_set = set()
		else:
			cl_gens_grp.__init__(self, b_from_load=False, igg=igg, gens_rec=gens_rec,
								 b_blocking=b_blocking, templ_iperm=templ_iperm, eid=eid)
			self.__eid_set = set([eid])
		self.__b_graduated = False
		# self.__b_blocking = b_blocking

	def is_graduated(self):
		return self.__b_graduated

	def add_perm(self, templ_iperm, eid, db_len_grps=None, templ_len=None, templ_b_cont_blocking=True, b_perms_from_file=False):
		if not b_perms_from_file and self.__b_graduated:
			return False

		b_success = super(cl_prov_gens_grp, self).add_perm(templ_iperm, eid)
		if not b_success:
			return False

		self.__eid_set.add(eid)
		if not self.__b_graduated and len(self.__eid_set) > config.c_gg_graduate_len:
			if templ_b_cont_blocking and not self.get_b_blocking():
				print('Will not graduate a non-blocking pgg in a templ grp that has cont blocked')
				return False

			self.__b_graduated = True
			print('gg graduated. igg:', self.get_igg())
			return True

		return False

	def print_gens_rec(self, igg, def_article_dict):
		super(cl_prov_gens_grp, self).print_gens_rec(igg, def_article_dict)
		print('eid set:', self.__eid_set)

	def test_for_better_set(self, eid_set):
		if self.__eid_set >= eid_set:
			return True
		return False

	def save(self, db_csvr):
		db_csvr.writerow(['pgg', 'graduated', self.__b_graduated, 'gens rec', mr.gen_rec_str(self.get_gens_rec()),
						  'is blocking', self.get_b_blocking()])

	def load(self, db_csvr):
		_, _, sb_graduated, _, srec, _, sb_blocking = get_next(db_csvr)
		self.__b_graduated = sb_graduated == 'True'
		gens_rec = mr.extract_rec_from_str(srec)
		self.set_gens_rec(gens_rec)
		self.set_b_blocking(sb_blocking == 'True')

class cl_templ_grp(object):
	# __slots__='__len', '__templ_grp_list'
	glv_dict = []
	glv_len = -1
	c_score_loser_penalty = config.c_score_loser_penalty
	c_score_winner_bonus = config.c_score_winner_bonus
	c_target_gens = None

	def __init__(self, b_from_load, templ_len=None, scvo=None, preconds_rec=None, gens_rec_list=None,
				 event_result_list=None, eid=None, b_blocking=None, b_cont_blocking=None):
		pgg_list = []
		if not b_from_load:
			self.__templ_len = templ_len
			self.__scvo = scvo
			self.__olen = mr.get_olen(scvo)
			for one_gens in gens_rec_list:
				pgg_list.append(cl_prov_gens_grp(	b_from_load=False, igg=len(pgg_list), gens_rec=one_gens,
													b_blocking=b_blocking, templ_iperm=0, eid=eid))
			self.__pgg_list = pgg_list
			self.__perm_list = [preconds_rec]
			self.__perm_result_list = [event_result_list]
			self.__perm_result_blocked_list = [b_blocking]
			self.__perm_eid_list = [eid]
			self.__perm_ipgg_arr = [range(len(pgg_list))] # list in list here
			self.__perm_vec_list = [mr.make_vec(self.glv_dict, preconds_rec, self.__olen, self.glv_len)]
		else:
			self.__templ_len = None
			self.__scvo = None
			self.__olen = None
			self.__pgg_list = pgg_list
			self.__perm_list = []
			self.__perm_result_list = []
			self.__perm_eid_list = []
			self.__perm_ipgg_arr = []  # list in list here
			self.__perm_vec_list = []
			self.__perm_result_blocked_list = []
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
		self.__b_cont_blocking = b_cont_blocking

	def get_nn_params(self):
		return self.__nn_params

	def scvo(self):
		return self.__scvo

	def find_pgg(self, gens_rec, b_blocking):
		for ipgg, pgg in enumerate(self.__pgg_list):
			if pgg.gens_matches(gens_rec, b_blocking):
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

	def add_perm(self, preconds_rec, gens_rec_list, perm_result_list, perm_result_blocked, eid,
				 db_len_grps=None, b_perms_from_file=False ):
		if self.__scvo == 'cacsoooooocecsv02v03ooov03v07cece':
			print('stop here')

		if not b_perms_from_file and self.__b_db_graduated:
			# sanity test
			# for one_gg in self.__gg_list[1:]:
			# 	if b_perms_from_file:
			# 		continue
			# 	assert one_gg.get_b_learn_success(), 'if a gg exists, it has learn success'

			b_add_valid = False
			b_needs_pgg = False
			for igens, one_gens in enumerate(gens_rec_list):
				perm_pgg, perm_ipgg = self.find_pgg(gens_rec=one_gens, b_blocking=perm_result_blocked)
				if perm_pgg:
					if not perm_pgg.is_graduated():
						b_add_valid = True
				else:
					b_needs_pgg = True
			if not b_add_valid and not b_needs_pgg:
				return

			# for one_gg in self.__gg_list[1:]:
			# 	if utils.prob_for_penalty(one_gg.get_penalty()):
			# 		b_add_valid = True
			# if not b_add_valid:
			# 	return

		iperm = len(self.__perm_list)
		self.__perm_list.append(preconds_rec)
		self.__perm_result_list.append(perm_result_list)
		self.__perm_result_blocked_list.append(perm_result_blocked)
		self.__perm_vec_list.append(mr.make_vec(self.glv_dict, preconds_rec, self.__olen, self.glv_len))
		self.__perm_eid_list.append(eid)
		self.__perm_ipgg_arr.append([])

		if gens_rec_list == []:
			# skipping reducing count of adds till next learn
			print('Warning! check why this function returned here. Cannot find original reason for code')
			return

		b_needs_recalib = False

		for igens, one_gens in enumerate(gens_rec_list):
			b_pgg_needs_graduating = False
			perm_pgg, perm_ipgg = self.find_pgg(gens_rec=one_gens, b_blocking=perm_result_blocked)
			if not perm_pgg:
				perm_pgg = cl_prov_gens_grp(b_from_load=False, igg=len(self.__pgg_list), gens_rec=one_gens,
											b_blocking=perm_result_blocked, templ_iperm=iperm, eid=eid)
				perm_ipgg = self.add_pgg(perm_pgg)
			else:
				b_pgg_needs_graduating = perm_pgg.add_perm(iperm, eid, db_len_grps=db_len_grps,
														   templ_len=self.__templ_len,
														   templ_b_cont_blocking=self.__b_cont_blocking,
														   b_perms_from_file=b_perms_from_file)
				# perm_templ.add_perm(preconds_rec=perm_preconds_list[iperm], gens_rec=perm_gens_list[iperm], igg=perm_igg)

			# if igens == 0:
			# 	self.__perm_ipgg_arr.append([perm_ipgg])
			# else:
			# 	self.__perm_ipgg_arr[-1].append(perm_ipgg)
			self.__perm_ipgg_arr[-1].append(perm_ipgg)

			if b_pgg_needs_graduating:
				print('gg graduated in template:', self.__scvo, 'len:', self.__templ_len)
				if len(self.__gg_list) == 0:
					gg_null = cl_gens_grp(b_from_load = False, igg=0, gens_rec=[], b_blocking=False)
					self.__gg_list = [gg_null]
				gg = cl_gens_grp(b_from_load = False, igg=len(self.__gg_list),
								 gens_rec=perm_pgg.get_gens_rec(), b_blocking=perm_pgg.get_b_blocking() )
 				gg.init_for_learn(self.__olen * self.glv_len, self.__scvo, len(self.__gg_list))
				self.__gg_list.append(gg)
				b_needs_recalib = True
				# self.__num_perm_adds_till_next_learn = 0

		if b_needs_recalib:
			self.recalib_ggs()
		else:
			if self.__b_db_graduated:
				self.assign_gg(iperm, preconds_rec, perm_result_list, eid)
				# self.__num_perm_adds_till_next_learn -= 1
				# self.__perm_igg_arr.append(self.assign_gg(iperm, preconds_rec, perm_result_list))

	def get_num_pggs(self):
		return len(self.__pgg_list)

	def get_num_perms(self):
		return len(self.__perm_pigg_arr)

	def get_match_score(self, def_article_dict, preconds_rec, perm_phrases, event_result_list, eid,
						b_blocking, event_result_score_list,
						result_confirmed_list, expected_but_not_found_list, gg_confirmed_list, b_real_score=True):
		# rewrite for result list and gg list
		b_score_valid = False
		gg_use_list = [False]
		if self.__db_valid:
			if b_real_score:
				for one_gg in self.__gg_list[1:]:
					gg_use_list.append(False)
					if not one_gg.get_will_score(eid): # get_b_learn_success():
						continue
					if utils.prob_for_penalty(one_gg.get_penalty()):
						gg_use_list[-1] = True
						b_score_valid = True
				if b_score_valid:
					print('Calculating score for all ggs in template:', self.__scvo, 'len:', self.__templ_len)
					self.printout(def_article_dict)
					score_list = event_result_score_list
				else:
					print('No gg valid for getting score for template:', self.__scvo, 'len:', self.__templ_len)
					if self.__scvo == 'cacsoooooocecsv02v03v07v05v06v04cece':
						print('stop here')
			else:
				score_list = [[] for _ in event_result_score_list]

		if b_score_valid:
			# expected_but_not_found_list = []
			perm_vec = mr.make_vec(self.glv_dict, preconds_rec, self.__olen, self.glv_len)
			for igg, one_gg in enumerate(self.__gg_list):
				if gg_use_list[igg]:
					one_gg.get_match_score(preconds_rec, perm_vec, perm_phrases, eid,
										   event_result_list, b_blocking,
										   event_result_score_list, self.__templ_len, self.__scvo,
										   result_confirmed_list, gg_confirmed_list,
										   expected_but_not_found_list)
			# 	one_gg.get_match_score(sess, self.__perm_vec_list, self.__perm_list, self.__olen, self.glv_len, self.glv_dict)
			# return dmlearn.get_score(preconds_rec, perm_vec, self.__nd_W, self.__nd_db, self.__gg_list,
			# 						 self.__perm_igg_arr, self.__perm_eid_list, event_result_list,
			# 						 score_list, self.__templ_len, self.__scvo)
		else:
			return event_result_score_list, []

		return event_result_score_list

	def do_learn(self, def_article_dict, sess, el_set_arr, curr_cont, eid):
		# if len(self.__gg_list) > 1 and len(self.__perm_igg_arr) > 5:
		if not self.__b_db_graduated:
			return

		# if self.__num_perm_adds_till_next_learn > 0:
		# 	print(self.__scvo, 'not learning yet. Still have', self.__num_perm_adds_till_next_learn, ' add_perms to go for ', self.__scvo)
		# 	return

		b_learn_valid = False
		gg_use_list = [False]
		for one_gg in self.__gg_list[1:]:
			gg_use_list.append(False)
			if one_gg.get_will_learn(eid) and utils.prob_for_penalty(one_gg.get_penalty()):
				gg_use_list[-1] = True
				b_learn_valid = True
		if not b_learn_valid:
			print('No gg in a position to learn for template:', self.__scvo, 'len:', self.__templ_len)
			return


		print('Learning all ggs in template:', self.__scvo, 'len:', self.__templ_len)
		self.printout(def_article_dict)
		for igg, one_gg in enumerate(self.__gg_list):
			one_gg.do_learn(gg_use_list[igg], sess, self.__perm_vec_list, self.__perm_list, self.__olen, self.glv_len,
							self.glv_dict, el_set_arr, curr_cont, eid)
		# for one_gg in self.__gg_list:
		# 	one_gg.set_match_limits(self.__nd_W, self.__nd_db, self.__perm_list, self.__perm_igg_arr, self.__olen, self.glv_len, self.glv_dict)
		# self.__num_perm_adds_till_next_learn = config.c_templ_learn_every_num_perms
		self.__db_valid = True

	def printout(self, def_article_dict):
		return

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
				gg.print_stats(def_article_dict)
		print('all perm recs:')
		for irec, rec in enumerate(self.__perm_list):
			if random.random() > 0.02:
				continue
			out_str = str(irec) + ' '
			out_str = els.print_phrase(rec, rec, out_str, def_article_dict)
			print(out_str)
			for perm_result in self.__perm_result_list[irec]:
				out_str =  ''
				out_str = els.print_phrase(rec, perm_result, out_str, def_article_dict)
				print('\t', str(irec), out_str)
			print('\tipgg:', self.__perm_ipgg_arr[irec],
				  ('result blocked' if self.__perm_result_blocked_list[irec] else ''),
				  'eid:', self.__perm_eid_list[irec])
			if self.__b_db_graduated:
				print('assigned igg:', [[igg, gg.get_perm_match_val(irec)] for igg, gg in enumerate(self.__gg_list)])


	def assign_gg(self, perm_irec, perm_rec, perm_result_list, eid):
		# igg_list = []
		# igg_arr = [0.0 for _ in self.__gg_list]
		# nd_igg = np.zeros([len(self.__gg_list)], np.float32)
		# Each gg must have its own record of whether the record matched or not
		for igg, gg in enumerate(self.__gg_list):
			if igg == 0:
				continue

			generated_result = mr.replace_vars_in_phrase(perm_rec,
														 gg.get_gens_rec())
			b_success_found = False
			for one_result in perm_result_list:
				if self.__perm_result_blocked_list[perm_irec] != gg.get_b_blocking():
					continue
				if mr.match_rec_exact(generated_result[1:-1], one_result):
					b_success_found = True
					# igg_list.append(igg)
					# nd_igg[igg] = 1.0
					break
			gg.mark_templ_perm_matched(b_success_found)

			if eid == gg.get_eid_last_add():
				continue

			if b_success_found:
				gg.add_perm(perm_irec, eid)
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
			self.assign_gg(irec, rec, self.__perm_result_list[irec], self.__perm_eid_list[irec])
			# self.__perm_igg_arr.append(self.assign_gg(irec, rec, self.__perm_result_list[irec]))
		self.__b_db_graduated = True

	def add_point(self, igg):
		if self.__gg_list[igg].add_point():
			print('gg confirmed in templ group with scvo, len;', self.__scvo, 'len:', self.__templ_len)
			self.__b_confirmed = True

	def apply_penalty(self, igg, bwinner):
		self.__gg_list[igg].apply_penalty(-self.c_score_winner_bonus if bwinner else self.c_score_loser_penalty)

	def get_gg(self, igg):
		if igg >= len((self.__gg_list)):
			return None
		return self.__gg_list[igg]

	def test_for_better_set(self, eid_set):
		for pgg in self.__pgg_list:
			if pgg.test_for_better_set(eid_set):
				return True
		return False

	def get_valid_ggs(self, templ_len):
		valid_gg_list = []
		for igg, gg in enumerate(self.__gg_list):
			if gg.get_b_valid():
				valid_gg_list.append(gg.get_cont_stats(templ_len, self.__scvo, igg))

		return valid_gg_list

	def get_num_data_rows(self):
		return 1 + len(self.__pgg_list) + len(self.__gg_list)

	def add_to_perm_dict(self, cont_id, templ_len, perm_dict):
		perm_dict[(cont_id, templ_len, self.__scvo)] = [self.__perm_list, self.__perm_result_list, self.__perm_result_blocked_list, self.__perm_eid_list]

	def add_to_W_dict(self, cont_id, templ_len, W_dict):
		for igg, gg in enumerate(self.__gg_list):
			if gg.get_b_learn_success():
				gg.add_to_W_dict(cont_id, templ_len, self.__scvo, W_dict)

	def set_W_from_file(self, igg, nd_W):
		if igg >= len(self.__gg_list):
			return

		self.__gg_list[igg].set_W_from_file(nd_W, self.__perm_vec_list)

	def save(self, db_csvr):
		db_csvr.writerow(['templ grp', 'confirmed', self.__b_confirmed, 'graduated', self.__b_db_graduated,
						  'valid', self.__db_valid, 'num obj fields', self.__olen,
						  'templ len', self.__templ_len, 'num pggs', len(self.__pgg_list),
						  'num ggs', len(self.__gg_list), 'scvo', self.__scvo])
		for pgg in self.__pgg_list:
			pgg.save(db_csvr)
		for gg in self.__gg_list:
			gg.save(db_csvr)

	def load(self, db_csvr, b_cont_blocking):
		_, _, sb_confirmed, _, sb_db_graduated, _, sdb_valid, _, solen, \
			_, stempl_len, _, num_pggs, _, num_ggs, _, self.__scvo = get_next(db_csvr)
		self.__b_confirmed, self.__b_db_graduated = sb_confirmed == 'True', sb_db_graduated == 'True'
		self.__db_valid, self.__olen = sdb_valid == 'True', int(solen)
		self.__templ_len  = int(stempl_len)
		self.__b_cont_blocking = b_cont_blocking

		for ipgg in range(int(num_pggs)):
			pgg = cl_prov_gens_grp(b_from_load=True, igg=ipgg)
			pgg.load(db_csvr)
			self.__pgg_list.append(pgg)

		for igg in range(int(num_ggs)):
			gg = cl_gens_grp(b_from_load=True, igg=igg)
			gg.load(db_csvr)
			if igg > 0:
				gg.init_for_learn(self.__olen * self.glv_len, self.__scvo, len(self.__gg_list))
			self.__gg_list.append(gg)

	# def sload(self, ldata):
	# 	_, _, sb_confirmed, _, sb_db_graduated, _, sdb_valid, _, solen, \
	# 		_, stempl_len, _, num_pggs, _, num_ggs, _, self.__scvo = ldata.pop(0)
	# 	self.__b_confirmed, self.__b_db_graduated = sb_confirmed == 'True', sb_db_graduated == 'True'
	# 	self.__db_valid, self.__olen = sdb_valid == 'True', int(solen)
	# 	self.__templ_len  = int(stempl_len)
	#
	# 	for ipgg in range(int(num_pggs)):
	# 		pgg = cl_prov_gens_grp(b_from_load=True, igg=ipgg)
	# 		pgg.load(ldata)
	# 		self.__pgg_list.append(pgg)
	#
	# 	for igg in range(int(num_ggs)):
	# 		gg = cl_gens_grp(b_from_load=True, igg=igg)
	# 		gg.load(ldata)
	# 		if igg > 0:
	# 			gg.init_for_learn(self.__olen * self.glv_len, self.__scvo, len(self.__gg_list))
	# 		self.__gg_list.append(gg)

class cl_len_grp(object):
	# __slots__='__len', '__templ_grp_list'

	def __init__(self, b_from_load, init_len=None, first_scvo=None, preconds_rec=None, gens_rec_list=None,
				 event_result_list=None, eid=None, b_blocking=None, b_cont_blocking=None):
		# gg = cl_gens_grp(gens_rec, preconds_rec)
		if b_from_load:
			self.__templ_grp_list = []
			self.__len = None
		else:
			self.__len = init_len
			self.__templ_grp_list = [cl_templ_grp(b_from_load=False, templ_len=init_len, scvo=first_scvo,
												  gens_rec_list=gens_rec_list, preconds_rec=preconds_rec,
												  event_result_list=event_result_list, eid=eid, b_blocking=b_blocking,
												  b_cont_blocking=b_cont_blocking)]

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

	def test_for_better_set(self, eid_set):
		for templ_grp in self.__templ_grp_list:
			if templ_grp.test_for_better_set(eid_set):
				return True
		return False

	def get_valid_ggs(self):
		valid_gg_list = []
		for templ_grp in self.__templ_grp_list:
			valid_gg_list += templ_grp.get_valid_ggs(self.__len)

		return valid_gg_list

	def get_num_data_rows(self):
		num_data_rows = 1
		for templ_grp in self.__templ_grp_list:
			num_data_rows += templ_grp.get_num_data_rows()
		return num_data_rows

	def add_to_perm_dict(self, cont_id, perm_dict):
		for templ_grp in self.__templ_grp_list:
			templ_grp.add_to_perm_dict(cont_id, self.len(), perm_dict)

	def add_to_W_dict(self, cont_id, W_dict):
		for templ_grp in self.__templ_grp_list:
			templ_grp.add_to_W_dict(cont_id, self.len(), W_dict)

	def save(self, db_csvr):
		db_csvr.writerow(['len grp', self.__len, 'num templates', len(self.__templ_grp_list)])
		for templ_grp in self.__templ_grp_list:
			templ_grp.save(db_csvr)

	def load(self, db_csvr, b_cont_blocking):
		_, slen, _, num_tmpl_grps = get_next(db_csvr)
		self.__len = int(slen)
		for i_templ_grp in range(int(num_tmpl_grps)):
			templ_grp = cl_templ_grp(b_from_load=True)
			templ_grp.load(db_csvr, b_cont_blocking)
			self.__templ_grp_list.append(templ_grp)

	# def sload(self, ldata):
	# 	_, slen, _, num_tmpl_grps = ldata.pop(0)
	# 	self.__len = int(slen)
	# 	for i_templ_grp in range(int(num_tmpl_grps)):
	# 		templ_grp = cl_templ_grp(b_from_load=True)
	# 		templ_grp.load(ldata)
	# 		self.__templ_grp_list.append(templ_grp)




