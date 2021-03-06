from __future__ import print_function
import random
import itertools

import config
import rules
import story
from rules import conn_type
import els
import utils

def all_combinations(src_set, max_len=None):
	results = []
	if max_len == None:
		max_len = len(src_set)
	for i in range(max_len):
		results += itertools.combinations(src_set, i+1)
	return results

def get_cascade_combs(els_sets, story_db, event_phrase, seed):
	story_els_set = utils.combine_sets([els_sets.objects, els_sets.places, els_sets.names])
	story_els = story_els_set[2]
	cascade_db = [event_phrase]
	phrase_idx_set = set()
	all_perms = []

	recurse_combos(story_els, story_db, cascade_db, seed, phrase_idx_set, phrase_idx_set, all_perms,
				   recursions_left = config.c_cascade_level)

	limited_all_perms = []
	for one_perm in all_perms:
		if len(one_perm) <= config.c_cascade_max_phrases:
			limited_all_perms.append(one_perm)
	all_perms = limited_all_perms

	return [[]] + all_perms


def get_obj_cascade(els_sets, target_phrase, story_db, event_phrase, b_for_query):
	story_els_set = utils.combine_sets([els_sets.objects, els_sets.places, els_sets.names])
	story_els = story_els_set[2]
	if b_for_query:
		cascade_db = [event_phrase]
	else:
		cascade_db = [target_phrase, event_phrase]
	phrase_idx_set = set()
	all_perms = []

	recurse_combos(story_els, story_db, cascade_db, phrase_idx_set, phrase_idx_set, all_perms,
				   recursions_left = config.c_cascade_level)

	limited_all_perms = []
	for one_perm in all_perms:
		if len(one_perm) <= config.c_cascade_max_phrases:
			limited_all_perms.append(one_perm)
	all_perms = limited_all_perms

	if b_for_query:
		all_perms = [k for j in [list(itertools.permutations(i, len(i))) for i in all_perms] for k in j]

	return [[]] + all_perms

def recurse_combos(story_els, story_db, cascade_db, seed, phrase_idx_set, phrase_idx_used, all_perms, recursions_left):
	recursions_left -= 1
	if recursions_left <= 0:
		return

	new_phrase_idx_set = level_cascade(story_els, story_db, cascade_db, phrase_idx_used, seed)
	if not new_phrase_idx_set:
		return
	all_combos = all_combinations(new_phrase_idx_set)
	new_phrase_idx_used = set(phrase_idx_used).union(new_phrase_idx_set)
	for one_perm in all_combos:
		new_phrase_idx_set = []
		new_cascade_db = list(cascade_db + [story_db[iphrase].phrase() for iphrase in one_perm])
		perm_phrase_idx_set = set(phrase_idx_set).union(one_perm)
		recurse_combos(story_els, story_db, new_cascade_db, seed, perm_phrase_idx_set, new_phrase_idx_used, all_perms,
					   recursions_left)
		all_perms += [list(perm_phrase_idx_set)]

def level_cascade(story_els, story_db, cascade_db, phrase_idx_set, seed):
	targets = set([seed])
	new_phrase_idx_set = set()
	# new_cascade_db = cascade_db
	for phrase in cascade_db:
		for phrase_el in phrase:
			if phrase_el[1] in story_els:
				targets.add(phrase_el[1])

	for iphrase, phrase in enumerate(story_db):
		phrase_story_els = [el[1] for el in phrase.phrase() if el[1] in story_els]
		inter = set(targets).intersection(phrase_story_els)
		if inter and iphrase not in phrase_idx_set and iphrase not in new_phrase_idx_set:
			# new_cascade_db.append(phrase.phrase())
			new_phrase_idx_set.add(iphrase)

	return new_phrase_idx_set

def recurse_phrase_combos(story_els, story_db, cascade_db, seed, phrase_idx_used, recursions_left):
	recursions_left -= 1
	if recursions_left <= 0:
		return phrase_idx_used

	new_phrase_idx_set = level_cascade(story_els, story_db, cascade_db, phrase_idx_used, seed)
	if not new_phrase_idx_set:
		return phrase_idx_used

	new_cascade_db = list(cascade_db + [story_db[iphrase].phrase() for iphrase in new_phrase_idx_set])
	total_phrase_idx_set = set(new_phrase_idx_set).union(phrase_idx_used)
	total_phrase_idx_set = recurse_phrase_combos(	story_els, story_db, new_cascade_db, seed,
													total_phrase_idx_set, recursions_left)
	return total_phrase_idx_set

	# all_combos = all_combinations(new_phrase_idx_set)
	# new_phrase_idx_used = set(phrase_idx_used).union(new_phrase_idx_set)
	# for one_perm in all_combos:
	# 	new_phrase_idx_set = []
	# 	new_cascade_db = list(cascade_db + [story_db[iphrase].phrase() for iphrase in one_perm])
	# 	perm_phrase_idx_set = set(phrase_idx_set).union(one_perm)
	# 	recurse_combos(story_els, story_db, new_cascade_db, perm_phrase_idx_set, new_phrase_idx_used, all_perms, recursions_left)
	# 	all_perms += [list(perm_phrase_idx_set)]

def get_phrase_cascade(els_sets, story_db, event_phrase, seed):
	story_els_set = utils.combine_sets([els_sets.objects, els_sets.places, els_sets.names])
	story_els = story_els_set[2]
	return get_ext_phrase_cascade(story_els, story_db, event_phrase, seed)

def get_ext_phrase_cascade(cascase_els, story_db, event_phrases, seed, num_recurse_levels=-1, max_num_phrases = -1):
	cascade_db = event_phrases
	phrase_idx_set = set()
	all_perms = []
	if num_recurse_levels == -1:
		num_recurse_levels = config.c_cascade_level
	if max_num_phrases == -1:
		max_num_phrases = config.c_cascade_max_phrases

	phrase_idx_set = recurse_phrase_combos(	cascase_els, story_db, cascade_db, seed, phrase_idx_set,
											recursions_left = num_recurse_levels)

	all_perms = all_combinations(phrase_idx_set, max_num_phrases)

	return [[]] + all_perms

def get_ext_phrase_cascade2(cascase_els, story_db, event_phrases, seed, num_recurse_levels=-1, max_num_phrases = -1):
	cascade_db = event_phrases
	phrase_idx_set = set()
	all_perms = []
	if num_recurse_levels == -1:
		num_recurse_levels = config.c_cascade_level
	if max_num_phrases == -1:
		max_num_phrases = config.c_cascade_max_phrases

	phrase_idx_set = recurse_phrase_combos(	cascase_els, story_db, cascade_db, seed, phrase_idx_set,
											recursions_left = num_recurse_levels)

	all_perms = all_combinations(phrase_idx_set, max_num_phrases)

	return all_perms
