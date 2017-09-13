import random
from enum import Enum
import collections
import utils
from utils import ulogger as logger



# varmod is a var whose object was in the input and is the only value to be modified
# mod is a field, normally seen in output, that tells you how to modify the database
# conn is a connective such as AND or OR
# var says this is a var, actual value defined in varobj
# varobj is a field with both a var id and an object, this providing the data and giving it a var name
# obj is just an object. It will not be repeated or used in a var

# mod is one of dm_type and says how to change the database
df_type = Enum('df_type', 'bool var obj varobj varmod mod conn')
dm_type = Enum('dm_type', 'Insert Remove Modify')
conn_type = Enum('conn_type', 'AND OR')

rule_fld = collections.namedtuple('rule_fld', 'els_set, df_type, sel_el, var_id, rand_sel')
rule_fld.__new__.__defaults__ = (None, None, None, False)
rule_parts = collections.namedtuple('rule_parts', 'gens, preconds, story_based, b_db, b_story')
rule_parts.__new__.__defaults__ = (None, False, True, False)

# def old_make_fld(els_set, df_type, sel_el=None, var_id=None):
# 	return [els_set, df_type, sel_el, var_id]
#
# def make_fld(els_set, df_type, sel_el=None, var_id=None):
# 	return rule_fld(els_set, df_type, sel_el, var_id)
#
def init_story_rules(name_set, object_set, place_set, action_set):
	story_rules = []
	objects_start = rule_parts(	gens = [
		rule_fld(els_set=[], df_type=df_type.mod, sel_el=dm_type.Insert),
		rule_fld(els_set=object_set, df_type=df_type.obj),
		rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
		rule_fld(els_set=place_set, df_type=df_type.obj, rand_sel=True)])
	story_rules.append(objects_start)
	people_start = rule_parts(	gens = [
		rule_fld(els_set=[], df_type=df_type.mod, sel_el=dm_type.Insert),
		rule_fld(els_set=name_set, df_type=df_type.obj),
		rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
		rule_fld(els_set=place_set, df_type=df_type.obj, rand_sel=True)])
	story_rules.append(people_start)
	picukup_rule = rule_parts(	preconds= [
		rule_fld(els_set=name_set, df_type=df_type.obj),
		rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
		rule_fld(els_set=place_set, df_type=df_type.obj),
		rule_fld(els_set=[], df_type=df_type.conn, sel_el=conn_type.AND),
		rule_fld(els_set=object_set, df_type=df_type.obj),
		rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
		rule_fld(els_set=[], df_type=df_type.var, var_id=2)],
			gens = [
		rule_fld(els_set=[], df_type=df_type.mod, sel_el=dm_type.Insert),
		rule_fld(els_set=[], df_type=df_type.var, var_id=0),
		rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='picked up'),
		rule_fld(els_set=[], df_type=df_type.var, var_id=4)],
			story_based = True, b_db=False, b_story=True
	)
	story_rules.append(picukup_rule)
	return story_rules

def init_rules(name_set, object_set, place_set, action_set):
	src_recs = []
	gen_rules = []
	gen_rule_has_and_went =	rule_parts(	preconds = [
						rule_fld(els_set=name_set, df_type=df_type.obj),
						rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='has'),
						rule_fld(els_set=object_set, df_type=df_type.obj),
						rule_fld(els_set=[], df_type=df_type.conn, sel_el=conn_type.AND),
						rule_fld(els_set=[], df_type=df_type.var, var_id=0),
						rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='went to'),
						rule_fld(els_set=place_set, df_type=df_type.obj)],
								gens = [
						rule_fld(els_set=[], df_type=df_type.mod, sel_el=dm_type.Modify),
						rule_fld(els_set=[], df_type=df_type.var, var_id=2),
						rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
						rule_fld(els_set=[], df_type=df_type.varmod, var_id=6),
						 ])
	gen_rules.append(gen_rule_has_and_went)
	gen_rule_picked_up =	rule_parts(	preconds = [
						rule_fld(els_set=name_set, df_type=df_type.obj),
						rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='picked up'),
						rule_fld(els_set=object_set, df_type=df_type.obj)],
								gens=[
						rule_fld(els_set=[], df_type=df_type.mod, sel_el=dm_type.Insert),
						rule_fld(els_set=[], df_type=df_type.var, var_id=0),
						rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='has'),
						rule_fld(els_set=[], df_type=df_type.var, var_id=2),
						 ])
	gen_rules.append(gen_rule_picked_up)
	gen_rule_put_down = 	rule_parts(	preconds = [
						rule_fld(els_set=name_set, df_type=df_type.obj),
						rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='put down'),
						rule_fld(els_set=object_set, df_type=df_type.obj)],
								gens=[
						rule_fld(els_set=[], df_type=df_type.mod, sel_el=dm_type.Remove),
						rule_fld(els_set=[], df_type=df_type.var, var_id=0),
						rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='has'),
						rule_fld(els_set=[], df_type=df_type.var, var_id=2),
						 ])
	gen_rules.append(gen_rule_put_down)
	gen_rule_went = rule_parts(	preconds = [
						rule_fld(els_set=name_set, df_type=df_type.obj),
						rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='went to'),
						rule_fld(els_set=place_set, df_type=df_type.obj)],
								gens=[
						rule_fld(els_set=[], df_type=df_type.mod, sel_el=dm_type.Modify),
						rule_fld(els_set=[], df_type=df_type.var, var_id=0),
						rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
						rule_fld(els_set=[], df_type=df_type.varmod, var_id=2),
						 ])
	gen_rules.append(gen_rule_went)
	print gen_rules[1].preconds[0].els_set
	print gen_rule_picked_up.preconds[0].els_set[2]
	return gen_rules
	# end function init_story_rules

"""
So far we have three functions
gen_for_rule creates examples from a set of rules. It is used both by story to generate a (n initial) database state
and by the oracle to create instances the learning network uses to learn rules.
gen_for_story is similar to this but does not generate an exhaustive set. Rather it looks at a story and tries
to find set of phrases from the story that matches the rule. It stops when it has found just on new inference
apply rules takes a single phrase, a set of rules and tries to generate all the inferences for that rule

"""

def gen_for_rule(els_dict, b_gen_for_learn, rule):
	gen_part = 'preconds'
	b_gen_from_conds = True
	if rule.preconds == None:
		gen_part = 'gens'
		b_gen_from_conds = False

	src_recs = None
	numrecs = 1
	for rule_part_name, rule_part in rule._asdict().iteritems():
		if rule_part_name == gen_part:  # i rule
			for fld_rule in rule_part:
				els_set, df_type, sel_el, var_id, rand_sel = fld_rule
				if sel_el == None and els_set != [] and not rand_sel:
					numrecs *= els_set[1]
				else:
					numrecs *= 1

	for ipart in range(2):
		if b_gen_from_conds:
			if ipart == 0:
				rule_part_name = 'preconds'
				rule_part = rule.preconds
			else:
				rule_part_name = 'gens'
				rule_part = rule.gens
		else:
			if ipart == 0:
				rule_part_name = 'gens'
				rule_part = rule.gens
			else:
				continue
		# for rule_part_name, rule_part in rule._asdict().iteritems():
		recs = [[] for i in range(numrecs)]
		for ifrule, fld_rule in enumerate(rule_part):
			els_set, df_type, sel_el, var_id, rand_sel = fld_rule
			if rule_part_name == gen_part and sel_el == None and els_set != [] and not rand_sel:
				numrecdiv = 1
				for ifrcont in range(ifrule + 1, len(rule_part)):
					els_set2, df_type2, sel_el2, var_id2, rand_sel2 = rule_part[ifrcont]
					if sel_el2 == None and els_set2 != [] and not rand_sel2:
						numrecdiv *= els_set2[1]
				recval = -1
				for irec in range(numrecs):
					# recval = (irec % numrecmod) / numrecdiv
					if irec % numrecdiv == 0:
						recval += 1
					if recval == els_set[1]:
						recval = 0
					recs[irec].append(els_set[0][recval])
			# following if applies to both input and output
			elif df_type == df_type.obj:
				if rand_sel:
					for irec in range(numrecs):
						i_sel_el = random.choice(els_set[0])
						recs[irec].append(i_sel_el)
				else:
					for irec in range(numrecs):
						recs[irec].append(els_dict[sel_el])
			elif df_type == df_type.bool:
				for irec in range(numrecs):
					recs[irec].append(int(sel_el))
			elif df_type == df_type.mod or df_type == df_type.conn:
				for irec in range(numrecs):
					recs[irec].append(sel_el.value - 1)
			elif df_type == df_type.var or df_type == df_type.varmod:
				# if we are genreating records for learning the rule, we don't want the explicit value
				# just to learn the var id
				if b_gen_for_learn:
					for irec in range(numrecs):
						recs[irec].append(var_id)
				else:
					# otherwise, it depends whether we are in the src, in which case the var is part of the same record
					if not b_gen_from_conds or (rule_part_name == gen_part and b_gen_from_conds):
						for irec in range(numrecs):
							recs[irec].append(recs[irec][var_id])
					else:
						for irec in range(numrecs):
							recs[irec].append(src_recs[irec][var_id])
			else:
				logger.error('Invalid field def for rec generation. Exiting')
				exit()

		if rule_part_name == gen_part and b_gen_from_conds:
			src_recs = recs

	return src_recs, recs
	# end function gen_for_rule

def apply_rules(els_dict, rules, phrase):
	mod_phrases = []
	search_markers = []
	for igen, rule in enumerate(rules):
		conds = rule.preconds
		field_id = -1
		b_hit = True
		for ifrule, fld_rule in enumerate(conds):
			field_id += 1
			els_set, df_type, sel_el, var_id, rand_sel = fld_rule
			if df_type == df_type.obj:
				if sel_el != None:
					iel = els_dict[sel_el]
					if iel == phrase[field_id]:
						continue
					else:
						b_hit = False
						break
				elif els_set != None and len(els_set) > 0 and els_set[0] != None and len(els_set[0]) > 0:
					if phrase[field_id] in els_set[0]:
						continue
					else:
						b_hit = False
						break
		if not b_hit:
			continue

		new_phrase = []
		new_markers = []
		gens = rule.gens
		for ifrule, fld_rule in enumerate(gens):
			field_id += 1
			els_set, df_type, sel_el, var_id, rand_sel = fld_rule
			if df_type == df_type.obj:
				if sel_el != None:
					new_phrase.append(els_dict[sel_el])
					new_markers.append(True)
				else:
					print 'Error! Dont know what to add.'
					exit()
			elif df_type == df_type.var:
				new_phrase.append(phrase[var_id])
				new_markers.append(True)
			elif df_type == df_type.mod and sel_el != None:
				new_phrase.append(sel_el.value - 1)
				new_markers.append(True)
			elif df_type == df_type.varmod and sel_el != None:
				new_phrase.append(sel_el.value - 1)
				new_markers.append(False)
			elif df_type == df_type.conn:
				continue
		# end loop over fld_rule of gens
		mod_phrases.append(new_phrase)
		search_markers.append(new_markers)

	return mod_phrases, search_markers

def apply_mods(story_db, mod_phrases, search_markers):
	for imod, mod_phrase in enumerate(mod_phrases):
		mod_type = mod_phrase[0]
		if mod_type == dm_type.Insert.value - 1:
			story_db += [mod_phrase[1:]]
			continue
		for phrase in story_db:
			b_match = True
			for iel, el in enumerate(phrase):
				if el != mod_phrase[iel + 1] and search_markers[imod][iel+1]:
					b_match = False
					break
			if b_match:
				if mod_type == dm_type.Remove.value - 1:
					story_db.remove(phrase)
				elif mod_type == dm_type.Remove.value - 1:
					story_db.remove(phrase)
					story_db += [mod_phrase[1:]]

	return story_db

def gen_from_story(els_dict, els_arr, rule, story):
	conds = rule.preconds

	old_hits = [[-1]]
	new_hits = []
	# b_still_in = True
	old_cands = range(len(story))
	hit_old_cands = [old_cands]
	new_cands = []
	field_id = -1
	var_locs = []

	# old_hits = [[-1], [-1]]
	# hit_old_cands = [[5, 6], [6, 7]]
	# old_hits = [[-1, 5], [-1, 6], [-1, 7], [-1, 8], [-1, 9]]
	# hit_old_cands = [[0, 1], [], [0, 1], [0, 1], []]

	def expand_hits(old_hits, hit_old_cands):
		nh1, old_hits = old_hits, []
		for ihit, hit in enumerate(nh1):
			for icand in hit_old_cands[ihit]:
				nh2 = hit
				# [nh2.extend(h) for h in nh1]
				nh3 = nh2 + [icand]
				old_hits.append(nh3)
		return old_hits
	# old_hits = expand_hits(old_hits, hit_old_cands)

	for ifrule, fld_rule in enumerate(conds):
		field_id += 1
		els_set, df_type, sel_el, var_id, rand_sel = fld_rule
		if df_type == df_type.conn and sel_el == conn_type.AND:
			field_id = -1
			old_hits = expand_hits(old_hits, hit_old_cands)
			# nh1, old_hits = old_hits, []
			# for icand in old_cands:
			# 	nh2 = []
			# 	[nh2.extend(h) for h in nh1]
			# 	nh2 += [icand]
			# 	old_hits.append(nh2)
			var_locs.append((len(new_hit), field_id)) # this cannot be accessed for an AND field so keep at -1 - just a placeholder
			hit_old_cands = [range(len(story)) for hit in old_hits]
			continue # move on to next field. This field needs no more parsing
			# end if fld == AND

		for ihit, hit in enumerate(old_hits):
			new_hit = hit
			old_cands = hit_old_cands[ihit]
			for iphrase in old_cands:
				phrase = story[iphrase]
				# if not b_still_in:
				# 	break
				if df_type == df_type.obj:
					if sel_el != None:
						if phrase[field_id] == els_dict[sel_el]:
							new_cands.append(iphrase)
							# new_hit.append([iphrase])
						# else:
						# 	b_still_in = False
					elif els_set != None and els_set[0] != None and len(els_set[0]) > 0:
						if phrase[field_id] in els_set[0]:
							new_cands.append(iphrase)
							# new_hit.append([iphrase])
						# else:
						# 	b_still_in = False
				elif df_type == df_type.var:
					var_phrase, var_field_id = var_locs[var_id]
					if var_phrase >= len(hit):
						req = phrase[var_field_id]
					else:
						 req = story[hit[var_phrase]][var_field_id]
					if phrase[field_id] == req:
						new_cands.append(iphrase)
				# end switch over df_type

			# end loop of iphrase over old_cands
			hit_old_cands[ihit] = new_cands
			new_cands = []

		# end loop of ihit, hit over old_hits
		var_locs.append((len(new_hit), field_id))
	# end loop of fld over conds part of rule
	old_hits = expand_hits(old_hits, hit_old_cands)

	sel_hit = random.choice(old_hits)

	new_conds = []
	# new_conds.append(rule_fld(els_set=[], df_type=df_type.mod, sel_el=dm_type.Insert))
	for iphrase in sel_hit[1:]:
		phrase = story[iphrase]
		for el in phrase:
			obj_el = els_arr[el]
			new_conds.append(rule_fld(els_set=[], df_type=df_type.obj, sel_el=obj_el))
		new_conds.append(rule_fld(els_set=[], df_type=df_type.conn, sel_el=conn_type.AND))
	new_conds = new_conds[:-1]
	new_rule = rule_parts(preconds=new_conds, gens=rule.gens)
	src_recs, recs = gen_for_rule(els_dict, b_gen_for_learn=False, rule=new_rule)
	return src_recs, recs

	# end of function gen_from_story