from time import gmtime, strftime
import sys
import math
import logging
import collections
import random

ulogger = None

def init_logging():
	global ulogger
	if ulogger != None:
		return

	ulogger = logging.getLogger('logic')
	ch = logging.StreamHandler(stream=sys.stdout)
	ch.setLevel(logging.DEBUG)
	ulogger.addHandler(ch)
	ulogger.setLevel(logging.DEBUG)
	ulogger.info('Starting at: %s', strftime("%Y-%m-%d %H:%M:%S", gmtime()))



if ulogger == None:
	init_logging()

def combine_sets(l_sets):
	# sym_set = [sym for one_set in l_sets for sym in one_set[2]]
	new_set = [[], 0, []]
	for one_set in l_sets:
		new_set[0] += one_set[0]
		new_set[1] += one_set[1]
		new_set[2] += one_set[2]

	return new_set

def set_from_l(ll, els_dict):
	return [[els_dict[l] for l in ll], len(ll), ll]

nt_el_sets = collections.namedtuple('nt_el_sets', 'names, objects, places, actions')

def unpack_els_sets(els_sets):
	return els_sets.names, els_sets.objects, els_sets.places, els_sets.actions

# to be used for choosing for prioritizing some rules over others
def weighted_choice(choices):
   total = sum(w for c, w in choices)
   r = random.uniform(0, total)
   upto = 0
   for c, w in choices:
      if upto + w >= r:
         return c
      upto += w
   assert False, "Shouldn't get here"

def get_avg_min_cd(vec_list, veclen):
	num_rules = 0.0
	vec_sum = [0.0 for _ in range(veclen)]
	for vec in vec_list:
		vec_sum = [vec_sum[i] + vec[i] for i in range(veclen)]
		num_rules += 1.0

	vec_avg = [vec_sum[i] / num_rules for i in range(veclen)]
	min_cd = 1.0
	for ivec, vec in enumerate(vec_list):
		cd = sum([vec_avg[i] * vec[i] for i in range(veclen)])
		if cd < min_cd:
			min_cd = cd

	return vec_avg, min_cd

def vec_norm(vec):
	sq = math.sqrt(sum([el * el for el in vec]))
	return [el/sq for el in vec]

def test_for_better_eid_set(db_len_grps, len, eid_set):
	for len_grp in db_len_grps:
		if len_grp.len() >= len:
			continue
		if len_grp.test_for_better_set(eid_set):
			return True
	return False