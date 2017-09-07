from enum import Enum
import collections

import rules
from rules import df_type
from rules import dm_type
from rules import conn_type
import els

def create_story(els_dict, def_article, els_arr, story_rules):
	story = []
	flds_arr = []

	b_gen_for_learn = False

	for igen, rule in enumerate(story_rules):
		flds_arr
		if not rule.story_based:
			src_recs, recs = rules.gen_for_rule(els_dict, b_gen_for_learn, rule)

		# The first value in the list is the sotry mod. It is in int format, so translate to enum value (which is 1-based index)
		for rec in recs:
			if rec[0]+1 == dm_type.Insert.value:
				story.append(rec[1:])

	for phrase in story:
		out_str = ''
		out_str = els.output_phrase(def_article, els_arr, out_str, phrase)
		print out_str

	return story