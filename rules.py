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
rule_parts = collections.namedtuple('rule_parts', 'gens, preconds, story_based')
rule_parts.__new__.__defaults__ = (None, False)

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
		rule_fld(els_set=place_set, df_type=df_type.obj)])
	story_rules.append(objects_start)
	people_start = rule_parts(	gens = [
		rule_fld(els_set=[], df_type=df_type.mod, sel_el=dm_type.Insert),
		rule_fld(els_set=name_set, df_type=df_type.obj),
		rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
		rule_fld(els_set=place_set, df_type=df_type.obj)])
	story_rules.append(people_start)
	picukup_rule = rule_parts(	preconds= [
		rule_fld(els_set=name_set, df_type=df_type.obj),
		rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
		rule_fld(els_set=place_set, df_type=df_type.obj),
		rule_fld(els_set=[], df_type=df_type.conn, sel_el=conn_type.AND),
		rule_fld(els_set=object_set, df_type=df_type.obj),
		rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
		rule_fld(els_set=[], df_type=df_type.var, var_id=1)],
			gens = [
		rule_fld(els_set=[], df_type=df_type.mod, sel_el=dm_type.Insert),
		rule_fld(els_set=[], df_type=df_type.var, var_id=0),
		rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='picked up'),
		rule_fld(els_set=[], df_type=df_type.var, var_id=1)],
			story_based = True
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
				if sel_el == None and els_set != []:
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
			if rule_part_name == gen_part and sel_el == None and els_set != []:
				numrecdiv = 1
				for ifrcont in range(ifrule + 1, len(rule_part)):
					els_set2, df_type2, sel_el2, var_id2, rand_sel2 = rule_part[ifrcont]
					if sel_el2 == None and els_set2 != []:
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
					if not b_gen_from_conds or (rule_part_name != gen_part and b_gen_from_conds):
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