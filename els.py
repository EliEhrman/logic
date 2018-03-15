import csv
import random
import config
import utils
import rules
from rules import df_type
from rules import dm_type
from rules import conn_type
from rules import rec_def_type
from utils import ulogger as logger

import numpy as np

num_conn_types = len(conn_type)
num_rec_def_types = len(rec_def_type)

# this is not the right place for these functions. It is here because this module has the components it needs
def quality_of_el_set(glv_dict, els_set):
	vecs = []
	for name in els_set[2]:
		vecs.append(glv_dict[name])

	utils.get_avg_min_cd(vecs, len(vecs[0]))

def quality_of_els_sets(glv_dict, els_sets):
	# for ilen in range(5):
	# 	veclen = ilen * 100 + 4
	# 	vecs = [utils.vec_norm([random.uniform(-1.0, 1.0) for _ in range(veclen)]) for _ in range(20)]
	# 	utils.get_avg_min_cd(vecs, len(vecs[0]))
	#
	all_els = []
	for els_set in els_sets:
		quality_of_el_set(glv_dict, els_set)
		all_els += els_set[2][:5]

	vecs = []
	for name in all_els:
		vecs.append(glv_dict[name])

	utils.get_avg_min_cd(vecs, len(vecs[0]))


def create_actions_file():
	actions_fn = 'actions.txt'
	actions_fh = open(actions_fn, 'wb')
	actions_csvr = csv.writer(actions_fh, delimiter=',', quoting=csv.QUOTE_NONE)

	for action in config.actions:
		actions_csvr.writerow([str(action)])

	actions_fh.close()

# gels1 = []
# gels2 = []
# distinguishing between els and objects. The former is generic. The latter are things you pick up, take etc
def init_els(els_dict, els_arr, def_article, fname=None, alist=None, new_def_article=False, cap_first=False, max_new=5):
	num_els = len(els_arr)
	if fname:
		fh_names = open(fname, 'rb')
		fr_names = csv.reader(fh_names, delimiter=',')
		if cap_first:
			new_els = [lname[0].lower().title() for lname in fr_names]
		else:
			new_els = [lname[0].lower() for lname in fr_names]
		# fh_names.seek(0)
		# new_el_vecs = [[float(fs) for fs in glv_row[1:]] for glv_row in fr_names]

		if max_new < 0:
			max_new = len(new_els)

		# global gels1, gels2
		# if fname == 'objects.glv':
		# 	gels1 = new_el_vecs
		# elif fname == 'countries.glv':
		# 	gels2 = new_el_vecs

		shuffle_stick = range(len(new_els))
		random.shuffle(shuffle_stick)
		new_els = [new_els[i] for i in shuffle_stick]
		# new_el_vecs = [new_el_vecs[i] for i in shuffle_stick]
		new_els = new_els[0:min(max_new, len(new_els))]
		# new_el_vecs = new_el_vecs[0:min(max_new, len(new_els))]
		fh_names.close()
	else:
		new_els = alist

	els_arr += new_els
	for iel, el_name in enumerate(new_els):
		els_dict[el_name] = num_els + iel
		# glv_dict[el_name] = new_el_vecs[iel]
	def_article += [new_def_article for id in new_els]
	num_new_els = len(new_els)
	new_els_range = range(num_els, num_els+num_new_els)
	num_els += num_new_els
	return [new_els_range, num_new_els, new_els], num_els

def init_glv(fnames, cap_first_arr, def_article_arr, cascade_els_arr):
	# fnames = ['names.glv', 'objects.glv', 'countries.glv', 'actions.glv']
	# cap_first_arr = [True, False, True, False]
	# def_article_arr = [False, True, False, False]
	glv_dict = {}
	new_el_vecs = []
	new_els = []
	def_article_dict = {}
	cascade_dict = {}

	# return [new_els_range, num_new_els, new_els], num_els
	num_els = 0
	els_sets = []

	for ifname, fname in enumerate(fnames):
		fh_names = open(fname, 'rb')
		fr_names = csv.reader(fh_names, delimiter=',')
		if cap_first_arr[ifname]:
			all_names = [lname[0].lower().title() for lname in fr_names]
		else:
			all_names = [lname[0].lower() for lname in fr_names]
		num_new_els = len(all_names)
		# new_els_range = range(num_els, num_els + num_new_els)
		# num_els += num_new_els
		# els_sets.append([new_els_range, num_new_els, new_els])
		els_sets.append(all_names)

		new_els += all_names
		for name in all_names:
			def_article_dict[name] = def_article_arr[ifname]
			cascade_dict[name] = cascade_els_arr[ifname]

		fh_names.seek(0)
		new_el_vecs += [[float(fs) for fs in glv_row[1:]] for glv_row in fr_names]
		fh_names.close()

	for iitem, item in enumerate(new_el_vecs):
		glv_dict[new_els[iitem]] = item


	return glv_dict, def_article_dict, cascade_dict, els_sets


def init_objects():
	els_arr = []
	els_dict = {}
	def_article = []
	glv_dict = {}
	# name_ids, num_els, num_names, names = \
	name_set, num_els = \
		init_els(	fname='names.glv', cap_first=True, max_new=config.max_names,
					els_dict=els_dict, els_arr=els_arr, def_article=def_article)
	# object_ids, num_els, num_objects, objects = \
	object_set, num_els = \
		init_els(	fname='objects.glv', new_def_article=True, max_new=config.max_objects,
					els_dict=els_dict, els_arr=els_arr, def_article=def_article)
	place_set, num_els =\
		init_els(	fname='countries.glv', cap_first=True, max_new=config.max_countries,
					els_dict=els_dict, els_arr=els_arr, def_article=def_article)

	action_set, num_els =\
		init_els(	fname='actions.glv', max_new=-1,
					els_dict=els_dict, els_arr=els_arr, def_article=def_article)

	# utils.get_avg_min_cd(gels1+gels2, len(gels1[0]))

	els_sets = utils.nt_el_sets(names=name_set, objects=object_set, places=place_set, actions=action_set)
	# return els_arr, els_dict, def_article, num_els, name_set, object_set, place_set, action_set
	return els_arr, els_dict, def_article, num_els, els_sets

def init_ext_objects():
	els_arr = []
	els_dict = {}
	def_article = []
	glv_dict = {}
	# name_ids, num_els, num_names, names = \
	name_set, num_els = \
		init_els(	fname='names.glv', cap_first=True, max_new=config.max_names,
					els_dict=els_dict, els_arr=els_arr, def_article=def_article)
	# object_ids, num_els, num_objects, objects = \
	object_set, num_els = \
		init_els(	fname='objects.glv', new_def_article=True, max_new=config.max_objects,
					els_dict=els_dict, els_arr=els_arr, def_article=def_article)
	place_set, num_els =\
		init_els(	fname='countries.glv', cap_first=True, max_new=config.max_countries,
					els_dict=els_dict, els_arr=els_arr, def_article=def_article)

	action_set, num_els =\
		init_els(	fname='actions.glv', max_new=-1,
					els_dict=els_dict, els_arr=els_arr, def_article=def_article)

	# utils.get_avg_min_cd(gels1+gels2, len(gels1[0]))

	els_sets = utils.nt_el_sets(names=name_set, objects=object_set, places=place_set, actions=action_set)
	# return els_arr, els_dict, def_article, num_els, name_set, object_set, place_set, action_set
	return els_arr, els_dict, def_article, num_els, els_sets

def convert_list_to_phrases(statement_list):
	return [[[rec_def_type.obj, item] for item in statement] for statement in statement_list]

def make_obj_el_from_str(sobj):
	return [rec_def_type.obj, sobj]

def make_rec_list(phrase_list):
	return [rules.C_phrase_rec(phrase) for phrase in phrase_list]


def output_phrase(def_article_dict, out_str, phrase):
	b_first = True
	for iel, el in enumerate(phrase):
		if el[0] == rec_def_type.error:
			out_str += '<Error!> '
			# return out_str
		else:
			if len(el) > 2 and el[2]:
				out_str += '{search for: '
			if el[0] == rec_def_type.obj and def_article_dict.get(el[1], False):
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
					out_str, el_set_arr = None, glv_dict = None):
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
			# filled_phrase.append((src_phrase.phrase())[el[1]])
			filled_phrase.append([el[0], '{var:' + str(el[1]) + '}'])
			if len(el) > 2:
				filled_phrase[-1].append(el[2])
		elif el[0] == rec_def_type.obj:
			filled_phrase.append(el)
		elif el[0] == rec_def_type.set:
			# glv_dict must be passed as an actual value here and not default param
			# if you get an assert here, that is the reason why
			set_vec, set_cd, _ = el_set_arr[el[1]]
			best_cd = -1.0
			best_name = 'not found!'
			for sym, one_vec in glv_dict.iteritems():
				cd = sum([one_vec[i] * set_val for i, set_val in enumerate(set_vec)])
				if cd > best_cd:
					best_name, best_cd = sym, cd
			filled_phrase.append([el[0], '{set#' + str(len(filled_phrase)) + ':' + best_name + '-like (' + str(set_cd) + ')}'])
		elif el[0] == rec_def_type.like:
			filled_phrase.append([el[0], '{like#' + str(len(filled_phrase)) + ':' + el[1] + ' (' + str(el[2]) + ')}'])

	return filled_phrase, out_str

def print_phrase(src_phrase, out_phrase, out_str, def_article_dict, el_set_arr = None, glv_dict = None):

	filled_phrase, out_str = complete_phrase(src_phrase, out_phrase, out_str, el_set_arr, glv_dict)
	return output_phrase(def_article_dict, out_str, filled_phrase)

# This function may be one useful but it does no add starts, end and AND
def build_vars_dict(phrase_list):
	new_phrase_list, vars_dict = [], dict()
	for iel, el in enumerate(phrase_list):
		if el[0] == rec_def_type.var:
			print('Error! Assuming the input phrase list contains no vars. Exiting!')
			exit()
		if el[0] != rec_def_type.obj:
			new_phrase_list.append(el)
			continue
		word = el[1]
		idx = vars_dict.get(word, None)
		if idx == None:
			# new_phrase_list.append([rec_def_type.var, len(vars_dict)])
			new_phrase_list.append(el)
			vars_dict[word] = iel
		else:
			new_phrase_list.append([rec_def_type.var, idx])

	return new_phrase_list, vars_dict

def make_vec(recs, glv_dict):
	# numrecs = len(recs)
	# num_els = len(els_dict)

	for irec, rec in enumerate(recs):
		for ifld, fld in enumerate(rec.phrase()):
			subvec0 = np.zeros(num_rec_def_types)
			subvec0[fld[0].value - 1] = 1
			if fld[0] == rec_def_type.conn:
				subvec1 = np.zeros(num_conn_types)
				subvec1[fld[1].value - 1] = 1
			elif fld[0]  == rec_def_type.obj:
				# subvec1 = np.zeros(num_els)
				# subvec1[els_dict[fld[1]]] = 1
				subvec1 = np.asarray(glv_dict[fld[1]])
			elif fld[0] == rec_def_type.var:
				subvec1 = np.zeros(config.c_max_vars)
				if fld[1] >= config.c_max_vars:
					print('More vars in the rule than we planned for. Exiting!')
					exit()
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

	if irec == 0: # i.e. there was only one record
		vecs = np.expand_dims(vecs, axis=0) # we need the putput to be consistent with multiple record outputs

	return vecs

def make_vec_old(recs, els_dict):
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

	if irec == 0: # i.e. there was only one record
		vecs = np.expand_dims(vecs, axis=0) # we need the putput to be consistent with multiple record outputs

	return vecs

def pad_ovec(vecs):
	numrecs = vecs.shape[0]
	pad_len = config.c_ovec_len - vecs.shape[1]
	vecs = np.concatenate((vecs, np.zeros((numrecs, pad_len), dtype=np.int32,)), axis=1)

	return vecs

def match_rec_to_templ(rec, template):
	for i, v in enumerate(template):
		if v == '?':
			continue
		if rec[i][0] != rules.rec_def_type.obj:
			return False
		if rec[i][1] != v:
			return False
	return True
