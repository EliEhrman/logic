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
	if fname:
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

def output_phrase(def_article, els_dict, out_str, phrase):
	b_first = True
	for iel, el in enumerate(phrase):
		if el[0] == rec_def_type.error:
			out_str += '<Error!> '
			# return out_str
		else:
			if len(el) > 2 and el[2]:
				out_str += '{search for: '
			if el[0] == rec_def_type.obj and def_article[els_dict[el[1]]]:
				if b_first:
					out_str += 'The '
					b_first = False
				else:
					out_str += 'the '
			if b_first:
				out_str += el[1][:1].upper() + el[1][1:]
				b_first = False
			else:
				out_str += el[1]
			if len(el) > 2 and el[2]:
				out_str += '}'
			out_str += ' '

	return out_str


def complete_phrase(src_phrase,
					out_phrase,
					out_str):
	filled_phrase = []
	for el in out_phrase:
		if el[0] == rec_def_type.conn:
			if el[1] == conn_type.AND:
				filled_phrase.append([el[0], 'AND'])
			elif el[1] == conn_type.OR:
				filled_phrase.append([el[0], 'OR'])
			elif el[1] == conn_type.start:
				filled_phrase.append([el[0], '('])
			elif el[1] == conn_type.end:
				filled_phrase.append([el[0], ')'])
			elif el[1] == conn_type.Modify:
				out_str = 'Modify: '
			elif el[1] == conn_type.Insert:
				out_str = 'Insert: '
			elif el[1] == conn_type.Remove:
				out_str = 'Remove: '
		elif el[0] == rec_def_type.var:
			filled_phrase.append((src_phrase.phrase())[el[1]])
			if len(el) > 2:
				filled_phrase[-1].append(el[2])
		elif el[0] == rec_def_type.obj:
			filled_phrase.append(el)
	return filled_phrase, out_str

def print_phrase(src_phrase, out_phrase, out_str, def_article, els_dict):

	filled_phrase, out_str = complete_phrase(src_phrase, out_phrase, out_str)
	return output_phrase(def_article, els_dict, out_str, filled_phrase)

def make_vec(recs, els_dict):
	# numrecs = len(recs)
	num_els = len(els_dict)

	for irec, rec in enumerate(recs):
		for ifld, fld in enumerate(rec.phrase()):
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

def pad_ovec(vecs):
	numrecs = vecs.shape[0]
	pad_len = config.c_ovec_len - vecs.shape[1]
	vecs = np.concatenate((vecs, np.zeros((numrecs, pad_len), dtype=np.int32,)), axis=1)

	return vecs
