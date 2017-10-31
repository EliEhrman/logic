from __future__ import print_function
import random
from enum import Enum
import collections
import config
import utils
# import els
from utils import ulogger as logger
# import els

# varmod is a var whose object was in the input and is the only value to be modified
# mod is a field, normally seen in output, that tells you how to modify the database
# conn is a connective such as AND or OR
# var says this is a var, actual value defined in varobj
# varobj is a field with both a var id and an object, this providing the data and giving it a var name
# obj is just an object. It will not be repeated or used in a var

# mod is one of dm_type and says how to change the database
df_type = Enum('df_type', 'bool var obj varobj varmod mod conn')
dm_type = Enum('dm_type', 'Insert Remove Modify')
conn_type = Enum('conn_type', 'single AND OR start end Insert Remove Modify')
rec_def_type = Enum('rec_def_type', 'obj conn var error')

# nt_rule_fld is the rule for one field (normally a word in the phrase). Consists of:
# els_set. A definition of a elements set. Itself consisting of a set of el_ids, a size and a set of the els
# df_type. either: obj, var, mod, conn
# sel_el. If not none, means the field must contain exactly this value
# var_id. If df_type == var, this is the id of the var. This uses a count of all objects in the rule to reference an earlier object
# rand_sel specifies that the object is selected from the set by random. Just one
# replace_by_next. When modifying, the first field must match exactly for record to count and the following field is inserted.
# Again, only applies when applying a mod to the story db
nt_rule_fld = collections.namedtuple('nt_rule_fld', 'els_set, df_type, sel_el, var_id, rand_sel, replace_by_next')
nt_rule_fld.__new__.__defaults__ = (None, None, None, False, False)
nt_rule_parts = collections.namedtuple('nt_rule_parts', 'gens, preconds, story_based, b_db, b_story')
nt_rule_parts.__new__.__defaults__ = (None, False, True, False)
nt_tree_junct = collections.namedtuple('nt_tree_junct', 'logic single branches')
nt_tree_junct.__new__.__defaults__ = (conn_type.single, [], [])
nt_phrase_rec = collections.namedtuple('nt_phrase_rec', 'time, phrase')
nt_phrase_rec.__new__.__defaults__ = (0, [])

class C_phrase_rec(object):
	# __slots__='time', '_phrase'

	def __init__(self, init_phrase=None, init_time=0):
		self.time = init_time
		if init_phrase:
			self.__phrase = init_phrase
		else:
			self.__phrase = []

	def append(self, el):
		self.__phrase.append(el)

	def append_replace_next(self, replace_next=True):
		self.__phrase[-1].append(replace_next)

	def phrase(self):
		return self.__phrase


# rec_def = collections.namedtuple('rec_def', 'obj, conn')

def init_blocking_rules(els_sets, els_dict):
	name_set, object_set, place_set, action_set = utils.unpack_els_sets(els_sets)
	blocking_rules = []

	gave_to_block_rule = nt_rule_parts(
		preconds = nt_tree_junct(single = [
			nt_rule_fld(els_set=name_set, df_type=df_type.obj),
			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='gave to'),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
			nt_rule_fld(els_set=object_set, df_type=df_type.obj)	]),
		gens = nt_tree_junct(single=[
			nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Remove)]))

	# if gave_to_block_rule.__class__ == nt_rule_parts:
	# 	print(str(gave_to_block_rule.__class__))
	blocking_rules.append(gave_to_block_rule)

	went_to_block_rule = nt_rule_parts(
		preconds = nt_tree_junct(logic=conn_type.AND, branches = [
			nt_tree_junct(single=[
				nt_rule_fld(els_set=name_set, df_type=df_type.obj),
				nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='is located in'),
				nt_rule_fld(els_set=place_set, df_type=df_type.obj)]),
			nt_tree_junct(single=[
				nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
				nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='went to'),
				nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2)])]),
		gens = nt_tree_junct(single=[
			nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Remove)]))

	blocking_rules.append(went_to_block_rule)

	return blocking_rules

def init_knowledge_query_rules(els_sets, els_dict, name):
	name_set, object_set, place_set, action_set = utils.unpack_els_sets(els_sets)
	query_rules = []

	where_know_3_rule = nt_rule_parts(
		preconds = nt_tree_junct(logic=conn_type.AND, branches = [
			nt_tree_junct(single=[
				nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='what does'),
				nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el=name),
				nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='know')
			]),
			nt_tree_junct(single=[
				nt_rule_fld(els_set=[], df_type=df_type.var, var_id=1),
				nt_rule_fld(els_set=utils.set_from_l(config.knowledge_action, els_dict), df_type=df_type.obj),
				nt_rule_fld(els_set=utils.combine_sets([name_set, object_set]), df_type=df_type.obj),
				nt_rule_fld(els_set=action_set, df_type=df_type.obj),
				nt_rule_fld(els_set=utils.combine_sets([name_set, object_set, place_set]), df_type=df_type.obj),
			])
		]),
		gens = nt_tree_junct(single=[
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=5),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=6),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=7)]),
						story_based = True, b_db=False, b_story=True
	)
	query_rules.append(where_know_3_rule)

	return query_rules

def init_ask_rules(els_sets, els_dict):
	name_set, object_set, place_set, action_set = utils.unpack_els_sets(els_sets)
	query_rules = []

	ask_where_object_rule = nt_rule_parts(
		preconds = nt_tree_junct(logic=conn_type.AND, branches = [
			nt_tree_junct(single=[
				nt_rule_fld(els_set=name_set, df_type=df_type.obj),
				nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='asked'),
				nt_rule_fld(els_set=name_set, df_type=df_type.obj),
				nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='where is'),
				nt_rule_fld(els_set=object_set, df_type=df_type.obj)
			]),
			nt_tree_junct(single=[
				nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
				nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
				nt_rule_fld(els_set=place_set, df_type=df_type.obj)]),
			nt_tree_junct(single=[
				nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2),
				nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
				nt_rule_fld(els_set=[], df_type=df_type.var, var_id=7)]),
		]),
		gens = nt_tree_junct(single=[
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=1),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=3),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=4),
			]),
						story_based = True, b_db=False, b_story=True
	)
	query_rules.append(ask_where_object_rule)

	return query_rules

def init_ask_blocking_rules(els_sets, els_dict):
	name_set, object_set, place_set, action_set = utils.unpack_els_sets(els_sets)
	blocking_rules = []

	ask_self_block_rule = nt_rule_parts(
		preconds = nt_tree_junct(logic=conn_type.AND, branches = [
			nt_tree_junct(single=[
				nt_rule_fld(els_set=name_set, df_type=df_type.obj),
				nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='asked'),
				nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
				nt_rule_fld(els_set=action_set, df_type=df_type.obj),
				nt_rule_fld(els_set=utils.combine_sets([name_set, object_set, place_set]), df_type=df_type.obj)
			]),
		]),
		gens = nt_tree_junct(single=[
			nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Remove),
			]),
							story_based = True, b_db=False, b_story=True
	)
	blocking_rules.append(ask_self_block_rule)

	return blocking_rules


def init_query_rules(els_sets, els_dict):
	name_set, object_set, place_set, action_set = utils.unpack_els_sets(els_sets)
	query_rules = []

	where_object_rule = nt_rule_parts(
			preconds = nt_tree_junct(logic=conn_type.AND, branches = [
				nt_tree_junct(single=[
		nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='where is'),
		nt_rule_fld(els_set=object_set, df_type=df_type.obj)]),
				nt_tree_junct(single=[
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=1),
		nt_rule_fld(els_set=utils.set_from_l(config.object_place_action, els_dict), df_type=df_type.obj),
		nt_rule_fld(els_set=place_set, df_type=df_type.obj)])]),
			gens = nt_tree_junct(single=[
		nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=1),
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=3),
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=4)]),
			story_based = True, b_db=False, b_story=True
	)
	query_rules.append(where_object_rule)
	# return query_rules

	where_person_rule = nt_rule_parts(
			preconds = nt_tree_junct(logic=conn_type.AND, branches = [
				nt_tree_junct(single=[
		nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='where is'),
		nt_rule_fld(els_set=name_set, df_type=df_type.obj)]),
				nt_tree_junct(single=[
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=1),
		nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='is located in'),
		nt_rule_fld(els_set=place_set, df_type=df_type.obj)])]),
			gens = nt_tree_junct(single=[
		nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=1),
		nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='is in'),
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=4)]),
			story_based = True, b_db=False, b_story=True
	)
	query_rules.append(where_person_rule)

	what_person_have_rule = nt_rule_parts(
			preconds = nt_tree_junct(logic=conn_type.AND, branches = [
				nt_tree_junct(single=[
		nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='what does'),
		nt_rule_fld(els_set=name_set, df_type=df_type.obj),
		nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='have')]),
				nt_tree_junct(single=[
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=1),
		nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='has'),
		nt_rule_fld(els_set=object_set, df_type=df_type.obj)])]),
			gens = nt_tree_junct(single=[
		nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=1),
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=4),
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=5)]),
			story_based = True, b_db=False, b_story=True
	)
	query_rules.append(what_person_have_rule)

	what_person_rule = nt_rule_parts(
			preconds = nt_tree_junct(logic=conn_type.AND, branches = [
				nt_tree_junct(single=[
		nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='what has'),
		nt_rule_fld(els_set=name_set, df_type=df_type.obj),
		nt_rule_fld(els_set=utils.set_from_l(config.person_object_dynamic_action, els_dict), df_type=df_type.obj)]),
				nt_tree_junct(single=[
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=1),
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2),
		nt_rule_fld(els_set=object_set, df_type=df_type.obj)])]),
			gens = nt_tree_junct(single=[
		nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=1),
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2),
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=5)]),
			story_based = True, b_db=False, b_story=True
	)
	query_rules.append(what_person_rule)

	what_place_rule = nt_rule_parts(preconds = nt_tree_junct(logic=conn_type.AND, branches = [
			nt_tree_junct(single=[
		nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='what'),
		nt_rule_fld(els_set=utils.set_from_l(config.object_place_action, els_dict), df_type=df_type.obj),
		nt_rule_fld(els_set=place_set, df_type=df_type.obj)]),
		# nt_rule_fld(els_set=utils.combine_sets([place_set, object_set]), df_type=df_type.obj)]),
			nt_tree_junct(single=[
		nt_rule_fld(els_set=object_set, df_type=df_type.obj),
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=1),
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2)])]),
									gens = nt_tree_junct(single=[
		nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=3),
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=1),
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2)]),
									story_based = True, b_db=False, b_story=True
									)
	query_rules.append(what_place_rule)

	who_obj_rule = nt_rule_parts(preconds = nt_tree_junct(logic=conn_type.AND, branches = [
			nt_tree_junct(single=[
		nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='who'),
		nt_rule_fld(els_set=utils.set_from_l(config.person_object_action, els_dict), df_type=df_type.obj),
		nt_rule_fld(els_set=object_set, df_type=df_type.obj)]),
			nt_tree_junct(single=[
		nt_rule_fld(els_set=name_set, df_type=df_type.obj),
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=1),
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2)])]),
								 gens = nt_tree_junct(single=[
		nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=3),
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=1),
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2)]),
								 story_based = True, b_db=False, b_story=True
								 )
	query_rules.append(who_obj_rule)

	who_place_rule = nt_rule_parts(preconds = nt_tree_junct(logic=conn_type.AND, branches = [
			nt_tree_junct(single=[
		nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='who'),
		nt_rule_fld(els_set=utils.set_from_l(config.person_place_action, els_dict), df_type=df_type.obj),
		nt_rule_fld(els_set=place_set, df_type=df_type.obj)]),
		# nt_rule_fld(els_set=utils.combine_sets([place_set, object_set]), df_type=df_type.obj)]),
			nt_tree_junct(single=[
		nt_rule_fld(els_set=name_set, df_type=df_type.obj),
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=1),
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2)])]),
								   gens = nt_tree_junct(single=[
		nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=3),
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=1),
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2)]),
								   story_based = True, b_db=False, b_story=True
								   )
	query_rules.append(who_place_rule)

	return query_rules

def init_story_rules(els_sets):
	name_set, object_set, place_set, action_set = utils.unpack_els_sets(els_sets)
	story_rules = []
	objects_start = nt_rule_parts(	gens = nt_tree_junct(single = [
		nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
		nt_rule_fld(els_set=object_set, df_type=df_type.obj),
		nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is free in'),
		nt_rule_fld(els_set=place_set, df_type=df_type.obj, rand_sel=True)]))
	story_rules.append(objects_start)

	people_start = nt_rule_parts(	gens = nt_tree_junct(single = [
		nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
		nt_rule_fld(els_set=name_set, df_type=df_type.obj),
		nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
		nt_rule_fld(els_set=place_set, df_type=df_type.obj, rand_sel=True)]))
	story_rules.append(people_start)

	gave_rule = nt_rule_parts(
		preconds = nt_tree_junct(logic=conn_type.AND, branches = [
			nt_tree_junct(single=[
				nt_rule_fld(els_set=name_set, df_type=df_type.obj),
				nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
				nt_rule_fld(els_set=place_set, df_type=df_type.obj)]),
			nt_tree_junct(single=[
				nt_rule_fld(els_set=name_set, df_type=df_type.obj),
				nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
				nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2)]),
			nt_tree_junct(single=[
				nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
				nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='has'),
				nt_rule_fld(els_set=object_set, df_type=df_type.obj)])]),
		gens = nt_tree_junct(single=[
			nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='gave to'),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=3),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=8)
		]),
		story_based = True, b_db=False, b_story=True
	)
	story_rules.append(gave_rule)

	pickup_rule = nt_rule_parts(preconds = nt_tree_junct(logic=conn_type.AND, branches = [
			nt_tree_junct(single=[
		nt_rule_fld(els_set=name_set, df_type=df_type.obj),
		nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
		nt_rule_fld(els_set=place_set, df_type=df_type.obj)]),
			nt_tree_junct(single=[
		nt_rule_fld(els_set=object_set, df_type=df_type.obj),
		nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is free in'),
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2)])]),
								gens = nt_tree_junct(single=[
		nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
		nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='picked up'),
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=3)]),
								story_based = True, b_db=False, b_story=True
								)
	story_rules.append(pickup_rule)

	putdown_rule = nt_rule_parts(preconds = nt_tree_junct(logic=conn_type.AND, branches = [
			nt_tree_junct(single=[
		nt_rule_fld(els_set=name_set, df_type=df_type.obj),
		nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
		nt_rule_fld(els_set=place_set, df_type=df_type.obj)]),
			nt_tree_junct(single=[
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
		nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='has'),
		nt_rule_fld(els_set=object_set, df_type=df_type.obj)])]),
								 gens = nt_tree_junct(single=[
		nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
		nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='put down'),
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=5)]),
								 story_based = True, b_db=False, b_story=True
								 )
	story_rules.append(putdown_rule)

	went_rule = nt_rule_parts(
			preconds=nt_tree_junct(single=[
		nt_rule_fld(els_set=name_set, df_type=df_type.obj),
		nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
		nt_rule_fld(els_set=place_set, df_type=df_type.obj)]),
			gens = nt_tree_junct(single = [
		nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
		nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='went to'),
		nt_rule_fld(els_set=place_set, df_type=df_type.obj, rand_sel=True)]),
			story_based = True, b_db = False, b_story = True
	)
	story_rules.append(went_rule)

	return story_rules

def instatiate_query(query_rule, rec):
	return nt_rule_parts(preconds=nt_tree_junct(logic=conn_type.AND, branches=[
		nt_tree_junct(single=[
			query_rule.preconds.branches[0].single[0],
			nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el=rec[3][1]),
			nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el=rec[4][1])
		]),
		nt_tree_junct(single=query_rule.preconds.branches[1])]),
						 gens=query_rule.gens)

def extract_query_gen(query_rule):
	return nt_rule_parts(gens = query_rule.preconds.branches[0])

def init_rules(els_sets, els_dict):
	name_set, object_set, place_set, action_set = utils.unpack_els_sets(els_sets)
	gen_rules = []
	gen_rule_picked_up =	nt_rule_parts(	preconds = nt_tree_junct(single=[
						nt_rule_fld(els_set=name_set, df_type=df_type.obj),
						nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='picked up'),
						nt_rule_fld(els_set=object_set, df_type=df_type.obj)]),
								gens= nt_tree_junct(single=[
						nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
						nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
						nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='has'),
						nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2),
						 ]))
	gen_rules.append(gen_rule_picked_up)

	gen_rule_went_from =	nt_rule_parts(
		preconds = nt_tree_junct(logic=conn_type.AND, branches=[
			nt_tree_junct(single=[
				nt_rule_fld(els_set=name_set, df_type=df_type.obj),
				nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='gave to'),
				nt_rule_fld(els_set=name_set, df_type=df_type.obj),
				nt_rule_fld(els_set=object_set, df_type=df_type.obj)]),
			nt_tree_junct(single=[
				nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
				nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='has'),
				nt_rule_fld(els_set=[], df_type=df_type.var, var_id=3)])]),
		gens=nt_tree_junct(single=[
			nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Remove),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='has'),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=3),
						 ]))
	gen_rules.append(gen_rule_went_from)


	gen_rule_gave_to =	nt_rule_parts(	preconds = nt_tree_junct(single=[
						nt_rule_fld(els_set=name_set, df_type=df_type.obj),
						nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='gave to'),
						nt_rule_fld(els_set=name_set, df_type=df_type.obj),
						nt_rule_fld(els_set=object_set, df_type=df_type.obj)]),
								gens= nt_tree_junct(single=[
						nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
						nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2),
						nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='has'),
						nt_rule_fld(els_set=[], df_type=df_type.var, var_id=3),
						 ]))
	gen_rules.append(gen_rule_gave_to)

	gen_rule_picked_up_free =	nt_rule_parts(
								preconds = nt_tree_junct(logic=conn_type.AND, branches=[
									nt_tree_junct(single=[
						nt_rule_fld(els_set=name_set, df_type=df_type.obj),
						nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='picked up'),
						nt_rule_fld(els_set=object_set, df_type=df_type.obj)]),
									nt_tree_junct(single=[
						nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2),
						nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is free in'),
						nt_rule_fld(els_set=place_set, df_type=df_type.obj)])]),
								gens=nt_tree_junct(single=[
						nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Modify),
						nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2),
						nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is free in', replace_by_next=True),
						nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
						nt_rule_fld(els_set=[], df_type=df_type.var, var_id=5),
						 ]))
	gen_rules.append(gen_rule_picked_up_free)

	gen_rule_put_down =	nt_rule_parts(	preconds = nt_tree_junct(single=[
						nt_rule_fld(els_set=name_set, df_type=df_type.obj),
						nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='put down'),
						nt_rule_fld(els_set=object_set, df_type=df_type.obj)]),
								gens= nt_tree_junct(single=[
						nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Remove),
						nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
						nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='has'),
						nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2),
						 ]))
	gen_rules.append(gen_rule_put_down)

	gen_rule_put_down_free =	nt_rule_parts(
								preconds = nt_tree_junct(logic=conn_type.AND, branches=[
									nt_tree_junct(single=[
						nt_rule_fld(els_set=name_set, df_type=df_type.obj),
						nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='put down'),
						nt_rule_fld(els_set=object_set, df_type=df_type.obj)]),
									nt_tree_junct(single=[
						nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2),
						nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
						nt_rule_fld(els_set=place_set, df_type=df_type.obj)])]),
								gens=nt_tree_junct(single=[
						nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Modify),
						nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2),
						nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in', replace_by_next=True),
						nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is free in'),
						nt_rule_fld(els_set=[], df_type=df_type.var, var_id=5),
						 ]))
	gen_rules.append(gen_rule_put_down_free)

	gen_rule_knows_dynamic_action =	nt_rule_parts(
		preconds = nt_tree_junct(logic=conn_type.AND,
			branches=[
				nt_tree_junct(single=[
					nt_rule_fld(els_set=name_set, df_type=df_type.obj),
					nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
					nt_rule_fld(els_set=place_set, df_type=df_type.obj)]),
				nt_tree_junct(single=[
					nt_rule_fld(els_set=name_set, df_type=df_type.obj),
					nt_rule_fld(els_set=utils.set_from_l(config.person_object_dynamic_action, els_dict), df_type=df_type.obj),
					nt_rule_fld(els_set=object_set, df_type=df_type.obj)]),
				nt_tree_junct(single=[
					nt_rule_fld(els_set=[], df_type=df_type.var, var_id=3),
					nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
					nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2)]),
			]),
		gens=nt_tree_junct(single=[
			nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
			nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='knows that'),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=3),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=4),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=5)
								]))
	gen_rules.append(gen_rule_knows_dynamic_action)


	# make sure this rule is before the place modification of 'went to'
	gen_rule_knows_went_from =	nt_rule_parts(
		preconds = nt_tree_junct(logic=conn_type.AND,
			branches=[
				nt_tree_junct(single=[
					nt_rule_fld(els_set=name_set, df_type=df_type.obj),
					nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
					nt_rule_fld(els_set=place_set, df_type=df_type.obj)]),
				nt_tree_junct(single=[
					nt_rule_fld(els_set=name_set, df_type=df_type.obj),
					nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='went to'),
					nt_rule_fld(els_set=place_set, df_type=df_type.obj)]),
				nt_tree_junct(single=[
					nt_rule_fld(els_set=[], df_type=df_type.var, var_id=3),
					nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
					nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2)]),
			]),
		gens=nt_tree_junct(single=[
			nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
			nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='knows that'),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=3),
			nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='went to'),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=5)
								]))
	gen_rules.append(gen_rule_knows_went_from)

	gen_rule_knows_went_to =	nt_rule_parts(
		preconds = nt_tree_junct(logic=conn_type.AND,
			branches=[
				nt_tree_junct(single=[
					nt_rule_fld(els_set=name_set, df_type=df_type.obj),
					nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
					nt_rule_fld(els_set=place_set, df_type=df_type.obj)]),
				nt_tree_junct(single=[
					nt_rule_fld(els_set=name_set, df_type=df_type.obj),
					nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='went to'),
					nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2)])
			]),
		gens=nt_tree_junct(single=[
			nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
			nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='knows that'),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=3),
			nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='went to'),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2)
								]))
	gen_rules.append(gen_rule_knows_went_to)

	# for now, the order matters. The first rule removes the located in and leaves no located in
	# The following rule looks only at the went to story step and created a new located in
	# The reliance on rule ordering seems poor. A better method would either allow a rule that
	# DOES NOT match any story db phrase or a specific disagreement between two vars (so you don't
	# remove a previously inserted 'is located in'
	print('replace the following with a modify!')
	gen_rule_went_from =	nt_rule_parts(
								preconds = nt_tree_junct(logic=conn_type.AND, branches=[
									nt_tree_junct(single=[
						nt_rule_fld(els_set=name_set, df_type=df_type.obj),
						nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
						nt_rule_fld(els_set=place_set, df_type=df_type.obj)]),
									nt_tree_junct(single=[
						nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
						nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='went to'),
						nt_rule_fld(els_set=place_set, df_type=df_type.obj)])]),
								gens=nt_tree_junct(single=[
						nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Remove),
						nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
						nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
						nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2),
						 ]))
	gen_rules.append(gen_rule_went_from)

	gen_rule_went =	nt_rule_parts(	preconds = nt_tree_junct(single=[
						nt_rule_fld(els_set=name_set, df_type=df_type.obj),
						nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='went to'),
						nt_rule_fld(els_set=place_set, df_type=df_type.obj)]),
								gens= nt_tree_junct(single=[
						nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
						nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
						nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
						nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2),
						 ]))
	gen_rules.append(gen_rule_went)

	gen_rule_has_and_went =	nt_rule_parts(
								preconds = nt_tree_junct(logic=conn_type.AND, branches=[
									nt_tree_junct(single=[
						nt_rule_fld(els_set=name_set, df_type=df_type.obj),
						nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='has'),
						nt_rule_fld(els_set=object_set, df_type=df_type.obj)]),
									nt_tree_junct(single=[
						nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2),
						nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
						nt_rule_fld(els_set=place_set, df_type=df_type.obj)]),
									nt_tree_junct(single=[
						nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
						nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='went to'),
						nt_rule_fld(els_set=place_set, df_type=df_type.obj)])]),
								gens=nt_tree_junct(single=[
						nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Modify),
						nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2),
						nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
						nt_rule_fld(els_set=[], df_type=df_type.var, var_id=5, replace_by_next=True),
						nt_rule_fld(els_set=[], df_type=df_type.var, var_id=8),
								]))
	gen_rules.append(gen_rule_has_and_went)

	gen_rule_knows_has =	nt_rule_parts(
		preconds = nt_tree_junct(logic=conn_type.AND,
			branches=[
				nt_tree_junct(single=[
					nt_rule_fld(els_set=name_set, df_type=df_type.obj),
					nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
					nt_rule_fld(els_set=place_set, df_type=df_type.obj)]),
				nt_tree_junct(single=[
					nt_rule_fld(els_set=name_set, df_type=df_type.obj),
					nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
					nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2)]),
				nt_tree_junct(single=[
					nt_rule_fld(els_set=[], df_type=df_type.var, var_id=3),
					nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='has'),
					nt_rule_fld(els_set=object_set, df_type=df_type.obj)])
			]),
		gens=nt_tree_junct(single=[
			nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
			nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='knows that'),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=3),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=7),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=8)
								]))
	gen_rules.append(gen_rule_knows_has)

	gen_rule_knows_location =	nt_rule_parts(
		preconds = nt_tree_junct(logic=conn_type.AND,
			branches=[
				nt_tree_junct(single=[
					nt_rule_fld(els_set=name_set, df_type=df_type.obj),
					nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='is located in'),
					nt_rule_fld(els_set=place_set, df_type=df_type.obj)]),
				nt_tree_junct(single=[
					nt_rule_fld(els_set=object_set, df_type=df_type.obj),
					nt_rule_fld(els_set=utils.set_from_l(config.object_place_action, els_dict), df_type=df_type.obj),
					nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2)]),
			]),
		gens=nt_tree_junct(single=[
			nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
			nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='knows that'),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=3),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=4),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2)
								]))
	gen_rules.append(gen_rule_knows_location)


	# print gen_rules[1].preconds[0].els_set
	# print gen_rule_picked_up.preconds[0].els_set[2]
	return gen_rules
	# end function init_story_rules

"""
So far we have three functions
gen_for_rule creates examples from a set of rules. It is used both by story to generate a (n initial) database state
and by the oracle to create instances the learning network uses to learn rules.
gen_for_story is similar to this but does not generate all combinations of the input sets. Rather it looks at a story and tries
to find set of phrases from the story that matches the rule. Sometimes, it stops when it has found just one new inference
apply rules takes a single phrase, a set of rules and tries to generate all the inferences for that rule

"""
def gen_for_rule(b_gen_for_learn, rule, time_stamp=0):
	gen_part = 'preconds'
	b_gen_from_conds = True
	if not rule.preconds:
		gen_part = 'gens'
		b_gen_from_conds = False

	src_recs = None
	numrecs = 1
	recdivarr = []

	def count_numrecs(tree, numrecs, recdivarr):
		if tree.logic == conn_type.single:
			for fld_rule in tree.single:
				if fld_rule.__class__ != nt_rule_fld:
					continue
				els_set, _, sel_el, _, rand_sel, _ = fld_rule
				if not sel_el and els_set and not rand_sel:
					numrecs *= els_set[1]
					recdivarr.append(els_set[1])
				else:
					numrecs *= 1
					recdivarr.append(1)
		else: # AND or OR
			for branch in tree.branches:
				numrecs, recdivarr = count_numrecs(branch, numrecs, recdivarr)
		return numrecs, recdivarr

	for rule_part_name, rule_part in rule._asdict().iteritems():
		if rule_part_name == gen_part:  # i rule
			numrecs, recdivarr = count_numrecs(rule_part, numrecs, recdivarr)

	def build_recs(tree, rule_part_name, recs, vars_dict, src_recs, recdivarr, field_id, obj_num):
		numrecs = len(recs)
		# field_id = 0
		# obj_num = 0
		if tree.logic == conn_type.single:
			for irec in range(numrecs):
				# recs[irec].append(rec_def_type.conn.value - 1)
				# recs[irec].append(conn_type.start.value - 1)
				recs[irec].append([rec_def_type.conn, conn_type.start])
			field_id += 1
			fld_defs = tree.single
			for ifrule, fld_rule in enumerate(fld_defs):
				if fld_rule.__class__ != nt_rule_fld:
					continue
				if rule_part_name == gen_part:
					recdivarr = recdivarr[1:]
				els_set, df_type, sel_el, var_id, rand_sel, replace_by_next = fld_rule
				if rule_part_name == gen_part and not sel_el and els_set and not rand_sel:
					numrecdiv = 1
					for irecdiv in recdivarr:
						numrecdiv *= irecdiv
					# for ifrcont in range(ifrule + 1, len(fld_defs)):
					# 	els_set2, df_type2, sel_el2, var_id2, rand_sel2 = fld_defs[ifrcont]
					# 	if sel_el2 == None and els_set2 != [] and not rand_sel2:
					# 		numrecdiv *= els_set2[1]
					recval = -1
					for irec in range(numrecs):
						# recval = (irec % numrecmod) / numrecdiv
						if irec % numrecdiv == 0:
							recval += 1
						if recval == els_set[1]:
							recval = 0
						# recs[irec].append(rec_def_type.obj.value - 1)
						# recs[irec].append(els_set[0][recval])
						recs[irec].append([rec_def_type.obj, els_set[2][recval]])
					vars_dict[obj_num] = field_id
					obj_num += 1
					field_id += 1
				# following if applies to both input and output
				elif df_type == df_type.obj:
					if rand_sel:
						for irec in range(numrecs):
							# recs[irec].append(rec_def_type.obj.value - 1)
							# i_sel_el = random.choice(els_set[0])
							# recs[irec].append(i_sel_el)
							r_sel_el = random.choice(els_set[2])
							recs[irec].append([rec_def_type.obj, r_sel_el])
					else:
						for irec in range(numrecs):
							# recs[irec].append(rec_def_type.obj.value - 1)
							# recs[irec].append(els_dict[sel_el])
							recs[irec].append([rec_def_type.obj, sel_el])
					if rule_part_name == gen_part:
						vars_dict[obj_num] = field_id
					obj_num += 1
					field_id += 1
				elif df_type == df_type.bool:
					print('Error! No code implementation for df.type == bool')
					exit()
					# for irec in range(numrecs):
					# 	recs[irec].append(int(sel_el))
				elif df_type == df_type.mod:
					for irec in range(numrecs):
						# recs[irec].append(rec_def_type.conn.value - 1)
						# recs[irec].append(sel_el.value - 1)
						recs[irec].append([rec_def_type.conn, sel_el])
					field_id += 1
				elif df_type == df_type.conn:
					print('Error! df.type == conn should not exits any more')
					exit()
				elif df_type == df_type.varmod:
					print('Error! df.type == varmod should not exits any more')
					exit()
				elif df_type == df_type.var:
					# if we are genreating records for learning the rule, we don't want the explicit value
					# just to learn the var id
					if b_gen_for_learn:
						for irec in range(numrecs):
							# recs[irec].append(rec_def_type.var.value - 1)
							# recs[irec].append(var_id)
							try:
								recs[irec].append([rec_def_type.var, vars_dict[var_id]])
							except:
								print('key error')
						obj_num += 1 # increment because its easier to specify rules counting vars, but no add to dict
						field_id += 1
					else:
						# otherwise, it depends whether we are in the src, in which case the var is part of the same record
						if not b_gen_from_conds or (rule_part_name == gen_part and b_gen_from_conds):
							for irec in range(numrecs):
								# recs[irec].append(rec_def_type.obj.value - 1)
								# recs[irec].append(recs[irec][var_id])
								recs[irec].append([rec_def_type.obj, (recs[irec].phrase())[vars_dict[var_id]][1]])
						else:
							for irec in range(numrecs):
								# recs[irec].append(rec_def_type.obj.value - 1)
								# recs[irec].append(src_recs[irec][var_id])
								recs[irec].append([rec_def_type.obj, (src_recs[irec].phrase())[vars_dict[var_id]][1]])
						if rule_part_name == gen_part:
							vars_dict[obj_num] = field_id
						obj_num += 1 # increment because its easier to specify rules counting vars, but no add to dict
						field_id += 1
				else:
					logger.error('Invalid field def for rec generation. Exiting')
					exit()
				if replace_by_next:
					for irec in range(numrecs):
						recs[irec].append_replace_next()
			for irec in range(numrecs):
				# recs[irec].append(rec_def_type.conn.value - 1)
				# recs[irec].append(conn_type.end.value - 1)
				recs[irec].append([rec_def_type.conn, conn_type.end])
			field_id += 1

		else:
			for irec in range(numrecs):
				# recs[irec].append(rec_def_type.conn.value - 1)
				# recs[irec].append(tree.logic.value - 1)
				recs[irec].append([rec_def_type.conn, tree.logic])
			field_id += 1
			for branch in tree.branches:
				recs, recdivarr, field_id, obj_num = \
					build_recs(branch, rule_part_name, recs, vars_dict, src_recs, recdivarr, field_id, obj_num)
			for irec in range(numrecs):
				# recs[irec].append(rec_def_type.conn.value - 1)
				# recs[irec].append(conn_type.end.value - 1)
				recs[irec].append([rec_def_type.conn, conn_type.end])
			field_id += 1

		return recs, recdivarr, field_id, obj_num

	vars_dict = {}
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
		recs = [C_phrase_rec(init_time=time_stamp) for i in range(numrecs)]
		field_id = 0
		obj_num = 0
		recs, _, _, _ = build_recs(rule_part, rule_part_name, recs, vars_dict, src_recs, recdivarr, field_id, obj_num)

		if rule_part_name == gen_part and b_gen_from_conds:
			src_recs = recs
	# end for part over preconds and gens
	del vars_dict

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
			els_set, df_type, sel_el, _, _, _ = fld_rule
			if df_type == df_type.obj:
				if sel_el is not None:
					iel = els_dict[sel_el]
					if iel == phrase[field_id]:
						continue
					else:
						b_hit = False
						break
				elif els_set is not None and len(els_set) > 0 and els_set[0] is not None and len(els_set[0]) > 0:
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
			_, df_type, sel_el, var_id, _, _ = fld_rule
			if df_type == df_type.obj:
				if sel_el != None:
					new_phrase.append(els_dict[sel_el])
					new_markers.append(True)
				else:
					print('Error! Dont know what to add.')
					exit()
			elif df_type == df_type.var:
				new_phrase.append(phrase[var_id])
				new_markers.append(True)
			elif df_type == df_type.mod and sel_el:
				new_phrase.append(sel_el.value - 1)
				new_markers.append(True)
			elif df_type == df_type.varmod and sel_el:
				new_phrase.append(sel_el.value - 1)
				new_markers.append(False)
			elif df_type == df_type.conn:
				continue
		# end loop over fld_rule of gens
		mod_phrases.append(new_phrase)
		search_markers.append(new_markers)

	return mod_phrases, search_markers

def apply_mods(story_db, mod_phrases, time_id):
	for imod, mod_phrase_rec in enumerate(mod_phrases):
		mod_phrase = mod_phrase_rec.phrase()
		mod_type = mod_phrase[0][1]
		b_phrase_found = False
		for iphrase, phrase_rec in enumerate(story_db):
			phrase = phrase_rec.phrase()
			new_phrase = []
			b_match = True
			iel = 0
			for el in phrase:
				iel += 1
				if el[1] != mod_phrase[iel][1]:
					b_match = False
					break
				if len(mod_phrase[iel]) > 2 and mod_phrase[iel][2]:
					iel += 1
				new_phrase.append(mod_phrase[iel])
			if b_match:
				b_phrase_found = True
				if mod_type == conn_type.Remove or mod_type == conn_type.Insert:
					story_db.pop(iphrase)
				elif mod_type == conn_type.Modify:
					story_db.pop(iphrase)
					story_db += [C_phrase_rec(new_phrase, time_id)]
		# Only insert if the phrase does not exist identically in DB already
		# implemented by removing the previous iteration and re-inserting but with a new time stamp
		if mod_type == conn_type.Insert:
			story_db += [C_phrase_rec(mod_phrase[1:], time_id)]

	return story_db

def gen_from_story(els_dict, els_arr, rule, story, gen_by_last=False, multi_ans=False):
	conds = rule.preconds

	old_hits = [[-1]]
	new_hits = []
	old_cands = range(len(story))
	hit_old_cands = [old_cands]
	field_id = -1

	def expand_hits(old_hits, hit_old_cands):
		nh1, old_hits = old_hits, []
		for ihit, hit in enumerate(nh1):
			for icand in hit_old_cands[ihit]:
				nh2 = hit
				nh3 = nh2 + [icand]
				old_hits.append(nh3)
		return old_hits

	def search_by_rules(tree, rule_part_name, field_id, obj_num, hit_old_cands, old_hits, vars_dict):
		if tree.logic == conn_type.single:
			new_cands = []
			field_id = 0
			fld_defs = tree.single
			for ifrule, fld_rule in enumerate(fld_defs):
				if fld_rule.__class__ != nt_rule_fld:
					continue
				els_set, df_type, sel_el, var_id, _, _ = fld_rule
				for ihit, hit in enumerate(old_hits):
					old_cands = hit_old_cands[ihit]
					for iphrase in old_cands:
						phrase_rec = story[iphrase]
						phrase = phrase_rec.phrase()
						if df_type == df_type.obj:
							if phrase[field_id][0] == rec_def_type.obj:
								if sel_el:
									if phrase[field_id][1] == sel_el:
										new_cands.append(iphrase)
								elif els_set != None and els_set[2] != None and len(els_set[2]) > 0:
									if phrase[field_id][1] in els_set[2]:
										new_cands.append(iphrase)
						elif df_type == df_type.var:
							var_phrase, var_field_id = vars_dict[var_id]
							if var_phrase >= len(hit):
								req = phrase[var_field_id][1]
							else:
								req = (story[hit[var_phrase]].phrase())[var_field_id][1]
							if phrase[field_id][1] == req:
								new_cands.append(iphrase)
					# end switch over df_type

					# end loop of iphrase over old_cands
					hit_old_cands[ihit] = new_cands
					new_cands = []

				# end loop of ihit, hit over old_hits
				if not old_hits:
					break
				vars_dict[obj_num] = (len(old_hits[0]), field_id)
				field_id += 1
				obj_num += 1
				# end loop over hits
			old_hits = expand_hits(old_hits, hit_old_cands)
			hit_old_cands = [range(len(story)) for hit in old_hits]
			# # end loop of fld over conds part of rule

		else:
			for branch in tree.branches:
				if not old_hits:
					break
				old_hits, field_id, obj_num, hit_old_cands, vars_dict \
					= search_by_rules(branch, rule_part_name, field_id, obj_num, hit_old_cands, old_hits, vars_dict)

		return old_hits, field_id, obj_num, hit_old_cands, vars_dict

	rule_part_name = 'preconds'
	rule_part = rule.preconds
	field_id = 0
	obj_num = 0
	vars_dict = {}

	old_hits = \
		search_by_rules(rule_part, rule_part_name, field_id, obj_num, hit_old_cands, old_hits, vars_dict)[0]

	if gen_by_last:
		ilast = len(story) - 1
		good_hits = []
		for hit in old_hits:
			if ilast >= 0 and ilast in hit:
				good_hits.append(hit)
	else:
		good_hits = old_hits

	if len(good_hits) == 0:
		return [], []

	ret_src_recs, ret_recs = [], []

	for igood in range(len(good_hits)):
		if multi_ans:
			sel_hit = good_hits[igood][1:]
		else:
			sel_hit = (random.choice(good_hits))[1:]

		new_conds = []
		# new_conds.append(nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert))
		# What we should do here is process the tree recursively using the original rule
		# However, besides the fact that I don't even know why I should have a second level of recursion
		# in this case it's not real. We are simply combining the phrases found in the story as if its a rule
		# So if there's jut one phrase, it's a single otherwise we do an AND
		time_for_rec = 0
		if len(sel_hit) == 1:
			phrase = (story[sel_hit[0]]).phrase()
			time_for_rec = story[sel_hit[0]].time
			rule_fields = []
			for el in phrase:
				# obj_el = els_arr[el]
				rule_fields.append(nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el=el[1]))
			new_conds = nt_tree_junct(single = rule_fields)
		else:
			branches = []
			for hit in sel_hit:
				phrase = (story[hit]).phrase()
				rule_fields = []
				for el in phrase:
					# obj_el = els_arr[el]
					rule_fields.append(nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el=el[1]))
				branches.append(nt_tree_junct(single=rule_fields))
				# Don't know if it makes sense to have multiple sel_hit
				hit_time = (story[hit]).time
				if hit_time > time_for_rec:
					# print('time calc not clear')
					time_for_rec = hit_time
			new_conds = nt_tree_junct(branches=branches, logic=conn_type.AND)

		new_rule = nt_rule_parts(preconds=new_conds, gens=rule.gens)
		src_recs, recs = gen_for_rule(b_gen_for_learn=False, rule=new_rule, time_stamp=time_for_rec)
		ret_src_recs.extend(src_recs)
		ret_recs.extend(recs)

		if not multi_ans:
			break

	return ret_src_recs, ret_recs

	# end of function gen_from_story

def print_rule(rule, out_str):
	def process_single_or_branch(tree, rule_part_name, out_str, i_fld_rule):
		num_fld_defs = 0
		if tree.logic == conn_type.single:
			fld_defs = tree.single
			num_fld_defs += len(fld_defs)
			for ifrule, fld_rule in enumerate(fld_defs):
				if fld_rule.__class__ != nt_rule_fld:
					continue
				out_str += 'fld rule # ' + str(ifrule + i_fld_rule) + ': '
				els_set, df_type, sel_el, var_id, rand_sel, replace_by_next = fld_rule
				if not sel_el and els_set:
					out_str += 'els_set: [['
					out_str += ''.join(str(e) + ', ' for e in els_set[0])
					out_str += '], '
					out_str += str(els_set[1])
					out_str += ', ['
					out_str += ''.join(e + ', ' for e in els_set[2])
					out_str += ']], '
				if df_type:
					out_str += 'df_type = ' + str(df_type) + ', '
				if sel_el:
					if df_type and df_type == df_type.mod:
						out_str += 'sel_el = ' + str(sel_el) + ', '
					else:
						out_str += 'sel_el = ' + sel_el + ', '
				out_str += 'rand_sel = ' + str(rand_sel) + ','
				if var_id is not None:
					out_str += 'var_id = ' + str(var_id) + ', '
				out_str += 'replace_by_next = ' + str(replace_by_next) + '. \n'


		else:
			out_str += str(tree.logic) + ' START\n'
			for branch in tree.branches:
				out_str += '{\n'
				out_str, i_fld_rule = process_single_or_branch(branch, rule_part_name, out_str, i_fld_rule)
				out_str += '}\n'
			out_str += str(tree.logic) + ' END\n'

		return  out_str, i_fld_rule+num_fld_defs

	if rule.preconds:
		rule_part_name = 'preconds'
		rule_part = rule.preconds
		out_str += 'preconds: \n'
		out_str, _ = process_single_or_branch(rule_part, rule_part_name, out_str, 0)
	if rule.gens:
		rule_part_name = 'gens'
		rule_part = rule.gens
		out_str += 'gens: \n'
		out_str, _ = process_single_or_branch(rule_part, rule_part_name, out_str, 0)
	return out_str
