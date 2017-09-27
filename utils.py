from time import gmtime, strftime
import sys
import logging

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
