from rules import nt_rule
from rules import nt_tree_junct
from rules import nt_rule_fld
from rules import rule_type
from rules import df_type
from rules import conn_type
from rules import rule_type
from rules import rule_type
import utils
import config

person_to_person_ask_rule_names = []

def is_this_a_person_to_person_ask_rule(rule_name):
	return rule_name in person_to_person_ask_rule_names

def init_decide_rules(els_sets, els_dict, name):
	name_set, object_set, place_set, action_set = utils.unpack_els_sets(els_sets)
	all_rules = []

	pickup_decide_rule = nt_rule(
		gens = nt_tree_junct(single = [
			# nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
			nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el=name),
			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='decided to'),
			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='pick up'),
			nt_rule_fld(els_set=object_set, df_type=df_type.obj, rand_sel=False)
		]),
		story_based=False, type=rule_type.story_start, name='pickup_decide_rule'
	)
	all_rules.append(pickup_decide_rule)

	goto_decide_rule = nt_rule(
		gens = nt_tree_junct(single = [
			# nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
			nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el=name),
			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='decided to'),
			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='go to'),
			nt_rule_fld(els_set=place_set, df_type=df_type.obj, rand_sel=False)
		]),
		story_based=False, type=rule_type.story_start, name='goto_decide_rule'
	)
	all_rules.append(goto_decide_rule)

	ask_where_decide_rule = nt_rule(
		gens = nt_tree_junct(single = [
			# nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
			nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el=name),
			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='decided to'),
			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='ask'),
			nt_rule_fld(els_set=name_set, df_type=df_type.obj, rand_sel=False),
			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='where is'),
			nt_rule_fld(els_set=object_set, df_type=df_type.obj, rand_sel=False)
		]),
		story_based=False, type=rule_type.story_start, name='ask_where_decide_rule'
	)
	all_rules.append(ask_where_decide_rule)

	tell_where_decide_rule = nt_rule(
		gens = nt_tree_junct(single = [
			# nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
			nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el=name),
			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='decided to'),
			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='tell'),
			nt_rule_fld(els_set=name_set, df_type=df_type.obj, rand_sel=False),
			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='where is'),
			nt_rule_fld(els_set=object_set, df_type=df_type.obj, rand_sel=False)
		]),
		story_based=False, type=rule_type.story_start, name='tell_where_decide_rule'
	)
	all_rules.append(tell_where_decide_rule)

	ask_give_decide_rule = nt_rule(
		gens = nt_tree_junct(single = [
			# nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
			nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el=name),
			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='decided to'),
			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='ask'),
			nt_rule_fld(els_set=name_set, df_type=df_type.obj, rand_sel=False),
			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='for'),
			nt_rule_fld(els_set=object_set, df_type=df_type.obj, rand_sel=False)
		]),
		story_based=False, type=rule_type.story_start, name='ask_give_decide_rule'
	)
	all_rules.append(ask_give_decide_rule)

	give_decide_rule = nt_rule(
		gens = nt_tree_junct(single = [
			# nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
			nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el=name),
			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='decided to'),
			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='give'),
			nt_rule_fld(els_set=name_set, df_type=df_type.obj, rand_sel=False),
			nt_rule_fld(els_set=object_set, df_type=df_type.obj, rand_sel=False)
		]),
		story_based=False, type=rule_type.story_start, name='give_decide_rule'
	)
	all_rules.append(give_decide_rule)

	return all_rules


def init_adv_rules(els_sets, els_dict):
	global person_to_person_ask_rule_names
	name_set, object_set, place_set, action_set = utils.unpack_els_sets(els_sets)
	all_rules = []

	# if gave_to_block_rule.__class__ == nt_rule_parts:
	# 	print(str(gave_to_block_rule.__class__))

	gave_to_block_rule = nt_rule(
		preconds = nt_tree_junct(single = [
			nt_rule_fld(els_set=name_set, df_type=df_type.obj),
			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='gave to'),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
			nt_rule_fld(els_set=object_set, df_type=df_type.obj)	]),
		gens = nt_tree_junct(single=[
			nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Remove)]),
		type=rule_type.block_event, name='gave_to_block_rule'
	)

	all_rules.append(gave_to_block_rule)

	want_dont_give_block_rule = nt_rule(
		# preconds = nt_tree_junct(single = [
		# 	nt_rule_fld(els_set=name_set, df_type=df_type.obj),
		# 	nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='gave to'),
		# 	nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
		# 	nt_rule_fld(els_set=object_set, df_type=df_type.obj)	]),
		preconds = nt_tree_junct(logic=conn_type.AND, branches = [
			nt_tree_junct(single=[
				nt_rule_fld(els_set=name_set, df_type=df_type.obj),
				nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='gave to'),
				nt_rule_fld(els_set=name_set, df_type=df_type.obj),
				nt_rule_fld(els_set=object_set, df_type=df_type.obj),
			]),
			nt_tree_junct(single=[
				nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
				nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='wants'),
				nt_rule_fld(els_set=[], df_type=df_type.var, var_id=3)])]),
		gens = nt_tree_junct(single=[
			nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Remove)]),
		type=rule_type.block_event, name='want_dont_give_block_rule'
	)

	all_rules.append(want_dont_give_block_rule)

	went_to_block_rule = nt_rule(
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
			nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Remove)]),
		type = rule_type.block_event, name = 'went_to_block_rule'
	)

	all_rules.append(went_to_block_rule)

	ask_self_block_rule = nt_rule(
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
		story_based = True, type=rule_type.block_event, name = 'ask_self_block_rule'
	)
	all_rules.append(ask_self_block_rule)

	# return all_rules

#
# def init_query_rules(els_sets, els_dict):
# 	name_set, object_set, place_set, action_set = utils.unpack_els_sets(els_sets)
# 	all_rules = []

	where_object_rule = nt_rule(
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
		story_based = True, type=rule_type.query, name='where_object_rule'
	)
	all_rules.append(where_object_rule)
	# return all_rules

	where_person_rule = nt_rule(
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
		story_based = True, type = rule_type.query, name = 'where_person_rule'
	)
	all_rules.append(where_person_rule)

	what_person_have_rule = nt_rule(
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
		story_based = True, type = rule_type.query, name = 'what_person_have_rule'
	)
	all_rules.append(what_person_have_rule)

	what_person_rule = nt_rule(
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
		story_based=True, type=rule_type.query, name='what_person_rule'
	)
	all_rules.append(what_person_rule)

	what_place_rule = nt_rule(
		preconds = nt_tree_junct(logic=conn_type.AND, branches = [
			nt_tree_junct(single=[
				nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='what'),
				nt_rule_fld(els_set=utils.set_from_l(config.object_place_action, els_dict), df_type=df_type.obj),
				nt_rule_fld(els_set=place_set, df_type=df_type.obj)]),
			nt_tree_junct(single=[
				nt_rule_fld(els_set=object_set, df_type=df_type.obj),
				nt_rule_fld(els_set=[], df_type=df_type.var, var_id=1),
				nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2)])]),
		gens = nt_tree_junct(single=[
			nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=3),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=1),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2)]),
		story_based=True, type=rule_type.query, name='what_place_rule'
	)
	all_rules.append(what_place_rule)

	who_obj_rule = nt_rule(
		preconds = nt_tree_junct(logic=conn_type.AND, branches = [
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
		story_based=True, type=rule_type.query, name='who_obj_rule'
	)
	all_rules.append(who_obj_rule)

	who_place_rule = nt_rule(
		preconds = nt_tree_junct(logic=conn_type.AND, branches = [
			nt_tree_junct(single=[
				nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='who'),
				nt_rule_fld(els_set=utils.set_from_l(config.person_place_action, els_dict), df_type=df_type.obj),
				nt_rule_fld(els_set=place_set, df_type=df_type.obj)]),
			nt_tree_junct(single=[
				nt_rule_fld(els_set=name_set, df_type=df_type.obj),
				nt_rule_fld(els_set=[], df_type=df_type.var, var_id=1),
				nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2)])]),
		gens = nt_tree_junct(single=[
			nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=3),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=1),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2)]),
		story_based=True, type=rule_type.query, name='who_place_rule'
	)
	all_rules.append(who_place_rule)

# 	return all_rules
#
# def init_story_rules(els_sets, els_dict):
# 	name_set, object_set, place_set, action_set = utils.unpack_els_sets(els_sets)
# 	all_rules = []
	objects_start = nt_rule(
		gens = nt_tree_junct(single = [
			nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
			nt_rule_fld(els_set=object_set, df_type=df_type.obj),
			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is free in'),
			nt_rule_fld(els_set=place_set, df_type=df_type.obj, rand_sel=True)]),
		story_based=False, type=rule_type.story_start, name='objects_start'
	)
	all_rules.append(objects_start)

	people_start = nt_rule(
		gens = nt_tree_junct(single = [
			nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
			nt_rule_fld(els_set=name_set, df_type=df_type.obj),
			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
			nt_rule_fld(els_set=place_set, df_type=df_type.obj, rand_sel=True)]),
		story_based=False, type=rule_type.story_start, name='people_start'
	)
	all_rules.append(people_start)

	people_want_start = nt_rule(
		gens = nt_tree_junct(single = [
			nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
			nt_rule_fld(els_set=name_set, df_type=df_type.obj),
			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='wants'),
			nt_rule_fld(els_set=object_set, df_type=df_type.obj, rand_sel=True)]),
		story_based=False, type=rule_type.story_start, name='people_want_start'
	)
	all_rules.append(people_want_start)

	ask_where_object_rule = nt_rule(
		preconds = nt_tree_junct(logic=conn_type.AND, branches = [
			nt_tree_junct(single=[
				nt_rule_fld(els_set=name_set, df_type=df_type.obj),
				nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='wants'),
				nt_rule_fld(els_set=object_set, df_type=df_type.obj),
			]),
			nt_tree_junct(single=[
				nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
				nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
				nt_rule_fld(els_set=place_set, df_type=df_type.obj)]),
			nt_tree_junct(single=[
				nt_rule_fld(els_set=name_set, df_type=df_type.obj),
				nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
				nt_rule_fld(els_set=[], df_type=df_type.var, var_id=5)]),
		]),
		gens = nt_tree_junct(single=[
			nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
			nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='asked'),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=6),
			nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='where is'),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2),
			]),
		story_based = True, type=rule_type.event_from_none, name='ask_where_object_rule', prob=2.0
	)
	person_to_person_ask_rule_names += ['ask_where_object_rule']
	all_rules.append(ask_where_object_rule)

	went_if_want_rule =	nt_rule(
		preconds = nt_tree_junct(logic=conn_type.AND,
			branches=[
				nt_tree_junct(single=[
					nt_rule_fld(els_set=name_set, df_type=df_type.obj),
					nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='knows that'),
					nt_rule_fld(els_set=object_set, df_type=df_type.obj),
					nt_rule_fld(els_set=utils.set_from_l(config.object_place_action, els_dict), df_type=df_type.obj),
					nt_rule_fld(els_set=place_set, df_type=df_type.obj)]),
				nt_tree_junct(single=[
					nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
					nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='wants'),
					nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2)]),
			]),
		gens=nt_tree_junct(single=[
			nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
			nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='went to'),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=4),
								]),
		story_based = True, type=rule_type.event_from_none, name='went_if_want_rule', prob=3.0
	)
	all_rules.append(went_if_want_rule)

	ask_to_give_rule =	nt_rule(
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
					nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='wants'),
					nt_rule_fld(els_set=object_set, df_type=df_type.obj)]),
				nt_tree_junct(single=[
					nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
					nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='has'),
					nt_rule_fld(els_set=[], df_type=df_type.var, var_id=8)]),
			]),
		gens=nt_tree_junct(single=[
			nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=3),
			nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='asked'),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
			nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='for'),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=8)
								]),
		story_based = True, type=rule_type.event_from_none, name='ask_to_give_rule', prob=5.0
	)
	all_rules.append(ask_to_give_rule)

	give_for_ask_rule =	nt_rule(
		preconds = nt_tree_junct(logic=conn_type.AND,
			branches=[
				nt_tree_junct(single=[
					nt_rule_fld(els_set=name_set, df_type=df_type.obj),
					nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='asked'),
					nt_rule_fld(els_set=name_set, df_type=df_type.obj),
					nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='for'),
					nt_rule_fld(els_set=object_set, df_type=df_type.obj)
				]),
				nt_tree_junct(single=[
					nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
					nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
					nt_rule_fld(els_set=place_set, df_type=df_type.obj)
				]),
				nt_tree_junct(single=[
					nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2),
					nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
					nt_rule_fld(els_set=[], df_type=df_type.var, var_id=7)]),
				nt_tree_junct(single=[
					nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2),
					nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='has'),
					nt_rule_fld(els_set=[], df_type=df_type.var, var_id=4)
				]),
				nt_tree_junct(single=[
					nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2),
					nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='likes'),
					nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0)]),
			]),
		gens=nt_tree_junct(single=[
			nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2),
			nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='gave to'),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=4)
								]),
		story_based = True, type=rule_type.event_from_event, name='give_for_ask_rule'
	)
	all_rules.append(give_for_ask_rule)


	gave_rule = nt_rule(
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
				nt_rule_fld(els_set=object_set, df_type=df_type.obj)]),
			nt_tree_junct(single=[
				nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
				nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='likes'),
				nt_rule_fld(els_set=[], df_type=df_type.var, var_id=8)])
		]),
		gens = nt_tree_junct(single=[
			nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='gave to'),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=3),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=8)
		]),
		story_based = True, type=rule_type.event_from_none, name='gave_rule', prob=3.0
	)
	all_rules.append(gave_rule)

	pickup_decide_rule = nt_rule(
		gens = nt_tree_junct(single = [
			# nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
			nt_rule_fld(els_set=name_set, df_type=df_type.obj),
			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='decided to'),
			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='pick up'),
			nt_rule_fld(els_set=object_set, df_type=df_type.obj, rand_sel=False)
		]),
		story_based=False, type=rule_type.story_start, name='pickup_decide_rule'
	)
	all_rules.append(pickup_decide_rule)

	pickup_rule = nt_rule(
		preconds = nt_tree_junct(logic=conn_type.AND, branches = [
			nt_tree_junct(single=[
				nt_rule_fld(els_set=name_set, df_type=df_type.obj),
				nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='decided to'),
				nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='pick up'),
				nt_rule_fld(els_set=object_set, df_type=df_type.obj)]),
			nt_tree_junct(single=[
				nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
				nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
				nt_rule_fld(els_set=place_set, df_type=df_type.obj)]),
			nt_tree_junct(single=[
				nt_rule_fld(els_set=[], df_type=df_type.var, var_id=3),
				nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is free in'),
				nt_rule_fld(els_set=[], df_type=df_type.var, var_id=6)])]),
		gens = nt_tree_junct(single=[
			nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='picked up'),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=3)]),
		story_based=True, type=rule_type.event_from_decide, name='pickup_rule'
								)
	all_rules.append(pickup_rule)

	putdown_rule = nt_rule(
		preconds = nt_tree_junct(logic=conn_type.AND, branches = [
			nt_tree_junct(single=[
				nt_rule_fld(els_set=name_set, df_type=df_type.obj),
				nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='decided to'),
				nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='put down'),
				nt_rule_fld(els_set=place_set, df_type=df_type.obj)]),
			nt_tree_junct(single=[
				nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
				nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
				nt_rule_fld(els_set=place_set, df_type=df_type.obj)]),
			nt_tree_junct(single=[
				nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
				nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='has'),
				nt_rule_fld(els_set=[], df_type=df_type.var, var_id=3)
			])]),
		gens = nt_tree_junct(single=[
			nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='put down'),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=3)]),
		story_based=True, type=rule_type.event_from_decide, name='putdown_rule', prob=0.05
								 )
	all_rules.append(putdown_rule)

	went_rule = nt_rule(
		preconds=nt_tree_junct(single=[
			nt_rule_fld(els_set=name_set, df_type=df_type.obj),
			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='decided to'),
			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='go to'),
			nt_rule_fld(els_set=place_set, df_type=df_type.obj)]),
		gens = nt_tree_junct(single = [
			nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='went to'),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=3)]),
		story_based=True, type=rule_type.event_from_decide, name='went_rule'
	)
	all_rules.append(went_rule)

# 	return all_rules
#
# def init_rules(els_sets, els_dict):
# 	name_set, object_set, place_set, action_set = utils.unpack_els_sets(els_sets)
# 	all_rules = []

	gen_rule_knows_when_told =	nt_rule(
		preconds = nt_tree_junct(logic=conn_type.AND,
			branches=[
				nt_tree_junct(single=[
					nt_rule_fld(els_set=name_set, df_type=df_type.obj),
					nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
					nt_rule_fld(els_set=place_set, df_type=df_type.obj)]),
				nt_tree_junct(single=[
					nt_rule_fld(els_set=name_set, df_type=df_type.obj),
					nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='told'),
					nt_rule_fld(els_set=name_set, df_type=df_type.obj),
					nt_rule_fld(els_set=utils.combine_sets([name_set, object_set, place_set]), df_type=df_type.obj),
					nt_rule_fld(els_set=utils.set_from_l(config.actions, els_dict), df_type=df_type.obj),
					nt_rule_fld(els_set=utils.combine_sets([name_set, object_set, place_set]), df_type=df_type.obj)]),
				nt_tree_junct(single=[
					nt_rule_fld(els_set=[], df_type=df_type.var, var_id=3),
					nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
					nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2)]),
			]),
		gens=nt_tree_junct(single=[
			nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
			nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='knows that'),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=6),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=7),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=8)
		]),
		type=rule_type.state_from_event, name='gen_rule_knows_when_told'
	)
	all_rules.append(gen_rule_knows_when_told)

	gen_likes_if_told =	nt_rule(
		preconds =
			nt_tree_junct(single=[
				nt_rule_fld(els_set=name_set, df_type=df_type.obj),
				nt_rule_fld(els_set=[], df_type=df_type.obj, sel_el='told'),
				nt_rule_fld(els_set=name_set, df_type=df_type.obj),
				nt_rule_fld(els_set=utils.combine_sets([name_set, object_set, place_set]), df_type=df_type.obj),
				nt_rule_fld(els_set=utils.set_from_l(config.actions, els_dict), df_type=df_type.obj),
				nt_rule_fld(els_set=utils.combine_sets([name_set, object_set, place_set]), df_type=df_type.obj)]),
		gens= nt_tree_junct(single=[
			nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2),
			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='likes'),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
		]),
		type=rule_type.state_from_event, name='gen_likes_if_told', prob=0.8
	)
	all_rules.append(gen_likes_if_told)

	gen_likes_if_gives =	nt_rule(
		preconds = nt_tree_junct(logic=conn_type.AND, branches=[
			nt_tree_junct(single=[
				nt_rule_fld(els_set=name_set, df_type=df_type.obj),
				nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='gave to'),
				nt_rule_fld(els_set=name_set, df_type=df_type.obj),
				nt_rule_fld(els_set=object_set, df_type=df_type.obj)]),
			nt_tree_junct(single=[
				nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2),
				nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='wants'),
				nt_rule_fld(els_set=[], df_type=df_type.var, var_id=3)])]),
		gens= nt_tree_junct(single=[
			nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2),
			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='likes'),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
		]),
		type=rule_type.state_from_event, name='gen_likes_if_gives', prob=1.0
	)
	all_rules.append(gen_likes_if_gives)


	gen_rule_picked_up =	nt_rule(
		preconds = nt_tree_junct(single=[
			nt_rule_fld(els_set=name_set, df_type=df_type.obj),
			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='picked up'),
			nt_rule_fld(els_set=object_set, df_type=df_type.obj)]),
		gens= nt_tree_junct(single=[
			nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='has'),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2),
		]),
		type=rule_type.state_from_event, name='gen_rule_picked_up'
	)
	all_rules.append(gen_rule_picked_up)

	gen_rule_gave_away =	nt_rule(
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
			nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Modify),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0, replace_by_next=True),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2),
			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='has'),
			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=3),
		]),
		type=rule_type.state_from_event, name='gen_rule_gave_away',
	)
	all_rules.append(gen_rule_gave_away)


	# gen_rule_gave_to =	nt_rule(
	# 	preconds = nt_tree_junct(single=[
	# 		nt_rule_fld(els_set=name_set, df_type=df_type.obj),
	# 		nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='gave to'),
	# 		nt_rule_fld(els_set=name_set, df_type=df_type.obj),
	# 		nt_rule_fld(els_set=object_set, df_type=df_type.obj)]),
	# 	gens= nt_tree_junct(single=[
	# 		nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
	# 		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2),
	# 		nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='has'),
	# 		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=3),
	# 	]),
	# 	type=rule_type.state_from_event, name='gen_rule_gave_to'
	# )
	# all_rules.append(gen_rule_gave_to)
	#
	gen_rule_picked_up_free =	nt_rule(
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
		]),
		type=rule_type.state_from_event, name='gen_rule_picked_up_free'
	)
	all_rules.append(gen_rule_picked_up_free)

	gen_rule_put_down =	nt_rule(
		preconds = nt_tree_junct(single=[
			nt_rule_fld(els_set=name_set, df_type=df_type.obj),
				nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='put down'),
				nt_rule_fld(els_set=object_set, df_type=df_type.obj)]),
		gens= nt_tree_junct(single=[
				nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Remove),
				nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
				nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='has'),
				nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2),
		]),
		type=rule_type.state_from_event, name='gen_rule_put_down'
	)
	all_rules.append(gen_rule_put_down)

	gen_rule_put_down_free =	nt_rule(
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
		]),
		type=rule_type.state_from_event, name='gen_rule_put_down_free'
	)
	all_rules.append(gen_rule_put_down_free)

	gen_rule_knows_dynamic_action =	nt_rule(
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
		]),
		type=rule_type.state_from_event, name='gen_rule_knows_dynamic_action'
	)
	all_rules.append(gen_rule_knows_dynamic_action)


	# make sure this rule is before the place modification of 'went to'
	gen_rule_knows_went_from =	nt_rule(
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
		]),
		type=rule_type.state_from_event, name='gen_rule_knows_went_from'
	)
	all_rules.append(gen_rule_knows_went_from)

	gen_rule_knows_went_to =	nt_rule(
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
								]),
		type=rule_type.state_from_event, name='gen_rule_knows_went_to'
	)
	all_rules.append(gen_rule_knows_went_to)

	# for now, the order matters. The first rule removes the located in and leaves no located in
	# The following rule looks only at the went to story step and created a new located in
	# The reliance on rule ordering seems poor. A better method would either allow a rule that
	# DOES NOT match any story db phrase or a specific disagreement between two vars (so you don't
	# remove a previously inserted 'is located in'
	# print('replace the following with a modify!')
	# gen_rule_went_from =	nt_rule(
	# 	preconds = nt_tree_junct(logic=conn_type.AND, branches=[
	# 		nt_tree_junct(single=[
	# 			nt_rule_fld(els_set=name_set, df_type=df_type.obj),
	# 			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
	# 			nt_rule_fld(els_set=place_set, df_type=df_type.obj)]),
	# 		nt_tree_junct(single=[
	# 			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
	# 			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='went to'),
	# 			nt_rule_fld(els_set=place_set, df_type=df_type.obj)])]),
	# 	gens=nt_tree_junct(single=[
	# 			nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Remove),
	# 			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
	# 			nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
	# 			nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2),
	# 	]),
	# 	type=rule_type.state_from_event, name='gen_rule_went_from'
	# )
	# all_rules.append(gen_rule_went_from)
	#
	# gen_rule_went =	nt_rule(
	# 	preconds = nt_tree_junct(single=[
	# 		nt_rule_fld(els_set=name_set, df_type=df_type.obj),
	# 		nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='went to'),
	# 		nt_rule_fld(els_set=place_set, df_type=df_type.obj)]),
	# 	gens= nt_tree_junct(single=[
	# 		nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Insert),
	# 		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
	# 		nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
	# 		nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2),
	# 	]),
	# 	type=rule_type.state_from_event, name='gen_rule_went'
	# )
	# all_rules.append(gen_rule_went)
	#
	gen_rule_went_from =	nt_rule(
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
				nt_rule_fld(els_set=[], df_type=df_type.mod, sel_el=conn_type.Modify),
				nt_rule_fld(els_set=[], df_type=df_type.var, var_id=0),
				nt_rule_fld(els_set=action_set, df_type=df_type.obj, sel_el='is located in'),
				nt_rule_fld(els_set=[], df_type=df_type.var, var_id=2, replace_by_next=True),
				nt_rule_fld(els_set=[], df_type=df_type.var, var_id=5),
		]),
		type=rule_type.state_from_event, name='gen_rule_went'
	)
	all_rules.append(gen_rule_went_from)

	gen_rule_has_and_went =	nt_rule(
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
		]),
		type=rule_type.state_from_event, name='gen_rule_has_and_went'
	)
	all_rules.append(gen_rule_has_and_went)

	gen_rule_knows_has =	nt_rule(
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
		]),
		type=rule_type.state_from_event, name='gen_rule_knows_has'
	)
	all_rules.append(gen_rule_knows_has)

	gen_rule_knows_location =	nt_rule(
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
		]),
		type=rule_type.state_from_event, name='gen_rule_knows_location'
	)
	all_rules.append(gen_rule_knows_location)


	# print all_rules[1].preconds[0].els_set
	# print gen_rule_picked_up.preconds[0].els_set[2]
	return all_rules
	# end function init_rules
