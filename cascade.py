from __future__ import print_function
import random
import itertools

import config
import rules
import story
from rules import conn_type
import els
import utils

def all_combinations(src_set):
	results = []
	for i in range(len(src_set)):
		results += itertools.combinations(src_set, i+1)
	return results

def get_cascade_combs(els_sets, story_db, event_phrase):
	story_els_set = utils.combine_sets([els_sets.objects, els_sets.places, els_sets.names])
	story_els = story_els_set[2]
	cascade_db = [event_phrase]
	phrase_idx_set = set()
	all_perms = []

	recurse_combos(story_els, story_db, cascade_db, phrase_idx_set, phrase_idx_set, all_perms,
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

def recurse_combos(story_els, story_db, cascade_db, phrase_idx_set, phrase_idx_used, all_perms, recursions_left):
	recursions_left -= 1
	if recursions_left <= 0:
		return

	new_phrase_idx_set = level_cascade(story_els, story_db, cascade_db, phrase_idx_used)
	if not new_phrase_idx_set:
		return
	all_combos = all_combinations(new_phrase_idx_set)
	new_phrase_idx_used = set(phrase_idx_used).union(new_phrase_idx_set)
	for one_perm in all_combos:
		new_phrase_idx_set = []
		new_cascade_db = list(cascade_db + [story_db[iphrase].phrase() for iphrase in one_perm])
		perm_phrase_idx_set = set(phrase_idx_set).union(one_perm)
		recurse_combos(story_els, story_db, new_cascade_db, perm_phrase_idx_set, new_phrase_idx_used, all_perms, recursions_left)
		all_perms += [list(perm_phrase_idx_set)]

def level_cascade(story_els, story_db, cascade_db, phrase_idx_set):
	targets = set()
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