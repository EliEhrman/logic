import random
from enum import Enum
import collections

import config
import rules
from rules import df_type
from rules import dm_type
from rules import conn_type
import els



def create_story(els_dict, def_article, els_arr, story_rules, gen_rules):
	story = []
	story_db = []
	flds_arr = []

	b_gen_for_learn = False

	for igen, rule in enumerate(story_rules):
		if not rule.story_based:
			src_recs, recs = rules.gen_for_rule(els_dict, b_gen_for_learn, rule)

			# The first value in the list is the story mod. It is in int format, so translate to enum value (which is 1-based index)
			for rec in recs:
				if rec[0]+1 == dm_type.Insert.value:
					story_db.append(rec[1:])

	print 'Current state of story DB'
	for phrase in story_db:
		out_str = ''
		out_str = els.output_phrase(def_article, els_arr, out_str, phrase)
		print out_str

	for i_story_step in range(config.c_story_len):
		for igen, rule in enumerate(story_rules):
			if rule.story_based:
				src_recs, recs = rules.gen_from_story(els_dict, els_arr, rule, story_db)
				out_str = 'Next story step: *** '
				out_str = els.output_phrase(def_article, els_arr, out_str, recs[0][1:])
				out_str += ' **** '
				print out_str
				mod_phrases, search_markers = rules.apply_rules(els_dict, gen_rules, recs[0][1:])
				story_db = rules.apply_mods(story_db, mod_phrases, search_markers)
				print 'New state of story DB'
				for phrase in story_db:
					out_str = ''
					out_str = els.output_phrase(def_article, els_arr, out_str, phrase)
					print out_str

	return story