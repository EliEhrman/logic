import csv
import random
import config
import rules
from rules import df_type
from rules import dm_type
from rules import conn_type
from rules import rec_def_type
from utils import ulogger as logger

import numpy as np

num_conn_types = len(conn_type)
num_rec_def_types = len(rec_def_type)

# distinguishing between objs and objects. The former is generic. The latter are things you pick up, take etc

def init_els(els_dict, els_arr, def_article, fname=None, alist=None, new_def_article=False, cap_first=False, max_new=5):
	num_els = len(els_arr)
	if fname != None:
		fh_names = open(fname, 'rb')
		fr_names = csv.reader(fh_names, delimiter=',')
		if cap_first:
			new_els = [new_id.lower().title() for lname in fr_names for new_id in lname]
		else:
			new_els = [new_id.lower() for lname in fr_names for new_id in lname]

		random.shuffle(new_els)
		new_els = new_els[0:min(max_new, len(new_els))]
	else:
		new_els = alist

	els_arr += new_els
	for iel, el_name in enumerate(new_els):
		els_dict[el_name] = num_els + iel
	def_article += [new_def_article for id in new_els]
	num_new_els = len(new_els)
	new_els_range = range(num_els, num_els+num_new_els)
	num_els += num_new_els
	return [new_els_range, num_new_els, new_els], num_els



def init_objects():
	els_arr = []
	els_dict = {}
	def_article = []
	# name_ids, num_els, num_names, names = \
	name_set, num_els = \
		init_els(	fname='names.txt', cap_first=True,
					max_new=config.max_names, els_dict=els_dict, els_arr=els_arr, def_article=def_article)
	# object_ids, num_els, num_objects, objects = \
	object_set, num_els = \
		init_els(	fname='objects.txt', new_def_article=True,
					max_new=config.max_objects, els_dict=els_dict, els_arr=els_arr, def_article=def_article)
	place_set, num_els =\
		init_els(	fname='countries.txt', cap_first=True,
					max_new=config.max_countries, els_dict=els_dict, els_arr=els_arr, def_article=def_article)

	action_set, num_els =\
		init_els(	alist=config.actions,
					max_new=config.max_objects, els_dict=els_dict, els_arr=els_arr, def_article=def_article)

	return els_arr, els_dict, def_article, num_els, name_set, object_set, place_set, action_set


def output_phrase(def_article, els_dict, out_str, phrase, filled_phrase = None):
	for iel, el in enumerate(phrase):
		if el[0] == rec_def_type.error:
			out_str += '<Error!> '
			# return out_str
		else:
			if filled_phrase != None and not filled_phrase[iel]:
				out_str += '*'
			if def_article[els_dict[el[1]]]:
				out_str += 'the '
			out_str += el[1]
			if filled_phrase != None and not filled_phrase[iel]:
				out_str += '*'
			out_str += ' '

	return out_str

# def output_phrase(def_article, els_arr, out_str, phrase, filled_phrase = None):
# 	# rewrite this for new rec format
# 	for iel, el in enumerate(phrase):
# 		if el < 0:
# 			out_str += '<Error!> '
# 		else:
# 			if filled_phrase != None and not filled_phrase[iel]:
# 				out_str += '*'
# 			if def_article[el]:
# 				out_str += 'the '
# 			out_str += els_arr[el]
# 			if filled_phrase != None and not filled_phrase[iel]:
# 				out_str += '*'
# 			out_str += ' '
#
# 	return out_str

def complete_phrase(flds, src_phrase, out_phrase, out_str, def_article, els_arr):
	filled_phrase = []
	exact_mark = []
	for el, fld in enumerate(flds):
		if fld.df_type == df_type.varobj or fld.df_type == df_type.obj:
			filled_phrase += [out_phrase[el]]
			exact_mark += [True]
		elif fld.df_type == df_type.mod:
			if out_phrase[el] == dm_type.Insert.value - 1:
				out_str = 'Insert: '
			elif out_phrase[el] == dm_type.Remove.value - 1:
				out_str = 'Remove: '
			elif out_phrase[el] == dm_type.Modify.value - 1:
				out_str = 'Modify: '
		elif fld.df_type == df_type.conn:
			if out_phrase[el] == conn_type.AND.value - 1:
				out_str += 'AND '
			if out_phrase[el] == conn_type.OR.value - 1:
				out_str += 'OR '
		elif fld.df_type == df_type.bool:
			if out_phrase[el]:
				out_str = 'true that '
			else:
				out_str = 'false that '
		elif fld.df_type == df_type.var:
			# input_id = vars_dict.get(flds[el].var_id, None)
			input_id = out_phrase[el]
			if input_id == None or input_id >= len(src_phrase):
				filled_phrase += [-1]
			else:
				filled_phrase += [src_phrase[input_id]]
			exact_mark += [True]
		elif fld.df_type == df_type.varmod:
			input_id = out_phrase[el]
			if input_id == None or input_id >= len(src_phrase):
				filled_phrase += [-1]
			else:
				filled_phrase += [src_phrase[input_id]]
			exact_mark += [False]
		else:
			logger.error('Invalid field ID. Exiting')
			exit()

	return filled_phrase, exact_mark, out_str

def print_phrase(flds, src_phrase, out_phrase, out_str, def_article, els_arr):

	filled_phrase, exact_mark, out_str = complete_phrase(flds, src_phrase, out_phrase, out_str, def_article, els_arr)
	return output_phrase(def_article, els_arr, out_str, filled_phrase, exact_mark)


def make_vec(recs, els_dict):
	# numrecs = len(recs)
	num_els = len(els_dict)

	for irec, rec in enumerate(recs):
		for ifld, fld in enumerate(rec):
			subvec0 = np.zeros(num_rec_def_types)
			subvec0[fld[0].value - 1] = 1
			if fld[0] == rec_def_type.conn:
				subvec1 = np.zeros(num_conn_types)
				subvec1[fld[1].value - 1] = 1
			elif fld[0]  == rec_def_type.obj:
				subvec1 = np.zeros(num_els)
				subvec1[els_dict[fld[1]]] = 1
			elif fld[0] == rec_def_type.var:
				subvec1 = np.zeros(config.c_max_vars)
				subvec1[fld[1]] = 1

			subvec = np.concatenate((subvec0, subvec1), axis=0)

			if ifld == 0:
				vec = subvec
			else:
				vec = np.concatenate((vec, subvec), axis=0)

		if irec == 0:
			vecs = vec
		else:
			vecs = np.vstack((vecs, vec))
	return vecs

