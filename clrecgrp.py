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
	def __init__(self, gens_rec, preconds_rec):
		self.__perm_list = [preconds_rec]
		self.__gens_rec = gens_rec
		# add a match rec whenever you add a gg
		self.__mrg_list = [cl_match_rec_grp(preconds_rec)]

	def gens_matches(self, gens_rec):
		return mr.match_gens_phrase(self.__gens_rec, gens_rec)

	def add_perm(self, preconds_rec):
		self.__perm_list.append(preconds_rec)
		# mr.make_rule_grp(glv_dict, self.__perm_list, el_set_arr)

	# def test_mrg_list(self, preconds_rec, event_result):
	# 	for mrg in self.__mrg_list:
	# 		if mrg.test(preconds_rec, event_result):
	def get_mrg_list(self):
		return self.__mrg_list




class cl_templ_grp(object):
	# __slots__='__len', '__templ_grp_list'

	def __init__(self, scvo, gens_grp, preconds_rec):
		self.__scvo = scvo
		self.__gg_list = [gens_grp]
		self.__perm_list = [preconds_rec]

	def scvo(self):
		return self.__scvo

	def find_gg(self, gens_rec):
		for igg, gg in enumerate(self.__gg_list):
			if gg.gens_matches(gens_rec):
				return gg, igg
		return None, -1

	# unusually, the following returns the index of the newly added gg
	def add_gg(self, gens_grp, preconds_rec):
		self.__gg_list.append(gens_grp)
		self.__perm_list.append(preconds_rec)
		return len(self.__gg_list) - 1

	def add_perm(self, preconds_rec):
		self.__perm_list.append(preconds_rec)

	def get_gg_list(self):
		return self.__gg_list


class cl_len_grp(object):
	# __slots__='__len', '__templ_grp_list'

	def __init__(self, init_len, first_scvo, preconds_rec, gens_rec):
		self.__len = init_len
		gg = cl_gens_grp(gens_rec, preconds_rec)
		self.__templ_grp_list = [cl_templ_grp(first_scvo, gens_grp=gg, preconds_rec=preconds_rec)]

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


