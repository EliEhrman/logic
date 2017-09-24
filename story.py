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
			# For now, we only support story generation of simple one-branch tree
			for rec in recs:
				if rec[0][1] == conn_type.start:
					if rec[1][1] == conn_type.Insert:
						story_db.append(rec[2:-1])
				else:
					print 'Error! for now, story insertions may only be simple single-branch phrases stripped of start/end'
					exit()

	print 'Current state of story DB'
	for phrase in story_db:
		out_str = ''
		out_str = els.output_phrase(def_article, els_dict, out_str, phrase)
		print out_str

	for i_story_step in range(config.c_story_len):
		for i_story_rule, story_rule in enumerate(story_rules):
			if story_rule.story_based:
				src_recs, recs = rules.gen_from_story(els_dict, els_arr, story_rule, story_db)
				if recs == []:
					break
				new_phrase = recs[0][2:-1]
				out_str = 'Next story step: *** '
				out_str = els.output_phrase(def_article, els_dict, out_str, new_phrase)
				out_str += ' **** '
				print out_str
				story_with_step = story_db + [new_phrase]
				for igen, gen_rule in enumerate(gen_rules):
					src_recs, recs = rules.gen_from_story(els_dict, els_arr, gen_rule, story_with_step, gen_by_last=True)
					if recs == []:
						continue
					mod_phrase = recs[0][1:-1]
					story_db = rules.apply_mods(story_db, [mod_phrase])
					print '-------- New state of story DB -----'
					for phrase in story_db:
						out_str = ''
						out_str = els.output_phrase(def_article, els_dict, out_str, phrase)
						print out_str

	return story