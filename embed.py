"""
This module is a stand-alone pre-processor for the logic project

This module takes the ascii text file output by the parse module and builds a file with the original word or phrase
and a vector of floats as its embedding
The output file is given the extension .glv

For simple one-word el files such as places or names, it simply looks up the vector in the downloaded glove files
one-word here is taken to mean a combo such as 'United States' which has no need for substructure parsing

For more complex phrases the .nlp is taken from output of the parse module.
There was an earlier version that created a long vector that included all the dep tree information using one-hot
components to represent the dep relation and ref information

The current version simply adds the embedding of all the eords themselves in the dep tree but emphasis is given to
the root word. deps of the root are multiplied by a diminishing factor before element-wise addition of the vector
to that of the governor. deps of the deps are themselves added to with again diminished vectors
The source vector of ech word is again taken from the glove file.

The vectors are added pre-normalization which might give added weight to very common pronouns over less-common roots
This may be good and is left in. Consider chnaging
"""
from __future__ import print_function
import csv
import sys
import os
import math
from sklearn.preprocessing import LabelBinarizer

c_num_gloves = 400000
c_max_deps = 4
c_dep_vec_factor = 0.1

glove_fn = '../../data/glove/glove.6B.50d.txt'
els_fn = 's.txt'
els_glv_fn = 's.glv'
parsed_fn = 's.txt.nlp'
dep_names_fn = 'nlp_deps.txt'
simple_els = ['countrie', 'name']
parsed_els = ['object', 'action']
c_example_word  = 'the'

g_word_vec_len = 0

def load_word_dict():
	global g_word_vec_len
	glove_fh = open(glove_fn, 'rb')
	glove_csvr = csv.reader(glove_fh, delimiter=' ', quoting=csv.QUOTE_NONE)
	
	word_dict = {}
	for irow, row in enumerate(glove_csvr):
		word = row[0]
		vec = [float(val) for val in row[1:]]
		word_dict[word] = vec
		if irow > c_num_gloves:
			break
		# print(row)
	
	glove_fh.close()
	g_word_vec_len = len(word_dict[c_example_word])
	return word_dict

def vec_norm(vec):
	sq = math.sqrt(sum([el * el for el in vec]))
	return [el/sq for el in vec]

def create_simple_glv(el_name, word_dict):
	els_fh = open(el_name+els_fn, 'rb')
	els_csvr = csv.reader(els_fh, delimiter=' ', quoting=csv.QUOTE_NONE)
	
	els_glv_fh = open(el_name+els_glv_fn, 'wb')
	els_glv_csvr = csv.writer(els_glv_fh, delimiter=',')
	
	for row in els_csvr:
		if len(row) == 0:
			continue
		el_string = ' '.join(row)
		el_vec = word_dict.get(el_string.lower(), None)
		if el_vec:
			el_vec = vec_norm(el_vec)
			els_glv_csvr.writerow([el_string] + el_vec)
			print (row[0], el_vec)

	els_fh.close()
	els_glv_fh.close()

def create_combined_vec(refs_list_arr, word_vec_arr, iref):
	# if not refs_list_arr[iref]:
	# 	return word_vec_arr[iref]

	ret =  word_vec_arr[iref]
	for subref in refs_list_arr[iref]:
		factored = [c_dep_vec_factor * val for val in create_combined_vec(refs_list_arr, word_vec_arr, subref)]
		ret = [factored[i] + val for i, val in enumerate(ret)]
	"""
	The vectors are added pre-normalization which might give added weight to very common pronouns over less-common roots
	This may be good and is left in. Consider chnaging
	"""
	return ret

# Function makes a list of forward references from the word to the word dependent on it
def make_dep_tree(src_name, dep_arr, idx_ref_arr, word_arr):
	src_arr =src_name.split()
	word_dict = dict()
	for isrc, src_word in enumerate(src_arr):
		word_dict[src_word] = isrc

	pos_dict = dict()
	for iword, word in enumerate(word_arr):
		pos_dict[word_dict[word]] = iword

	refs_list_arr = [[] for _ in word_dict]
	for iidx, sidx in enumerate(idx_ref_arr):
		idx = int(sidx)
		if idx > 0:
			refs_list_arr[pos_dict[idx-1]].append(iidx)

	return refs_list_arr

def create_parsed_glv(el_name, word_dict, dep_enc, idx_enc):
	global g_word_vec_len
	els_fh = open(el_name + parsed_fn, 'rb')
	els_csvr = csv.reader(els_fh, delimiter=',', quoting=csv.QUOTE_NONE)

	els_glv_fh = open(el_name + els_glv_fn, 'wb')
	els_glv_csvr = csv.writer(els_glv_fh, delimiter=',')

	for row in els_csvr:
		if len(row) == 0:
			continue
		src_name = row[0]
		num_deps = int(row[1])
		if num_deps > c_max_deps:
			continue
		total_vec = []
		b_fail = False
		dep_arr, idx_ref_arr, word_arr, word_vec_arr = [[[] for _ in range(num_deps)] for _ in range(4)]
		for i_dep in range(num_deps):
			field_id = 2 + (i_dep * 3)
			dep_arr[i_dep], idx_ref_arr[i_dep], word_arr[i_dep] = row[field_id:field_id + 3]
			word_vec_arr[i_dep] = word_dict.get(word_arr[i_dep].lower(), None)
			if not word_vec_arr[i_dep]:
				b_fail = True
				break

		if not b_fail:
			refs_list_arr = make_dep_tree(src_name, dep_arr, idx_ref_arr, word_arr)
			total_vec = create_combined_vec(refs_list_arr, word_vec_arr, 0)
			total_vec = vec_norm(total_vec)
			els_glv_csvr.writerow([src_name] + total_vec)
			print(src_name, total_vec)

		# for i_dep in range(c_max_deps):
		# 	field_id = 2 + (i_dep * 3)
		# 	if i_dep < num_deps:
		# 		dep, idx_ref, word = row[field_id:field_id+3]
		# 		# Create a one-hot floating point vector for the dep
		# 		# The list comprehension is used to cast the els of the list to floats
		# 		# The [0] is because the transform produces its vector as a one-el vector of vectors
		# 		# so we just need to get rid of the outer array
		# 		dep_vec = [float(i) for i in list((dep_enc.transform([dep]))[0])]
		# 		word_vec = word_dict.get(word.lower(), None)
		# 		if not word_vec:
		# 			b_fail = True
		# 			break
		# 		idx_vec = [float(i) for i in list((idx_enc.transform([int(idx_ref)]))[0])]
		# 	else:
		# 		dep_vec = [float(i) for i in list((dep_enc.transform(['invalid']))[0])]
		# 		word_vec = [0.0] * g_word_vec_len
		# 		idx_vec = [float(i) for i in list((idx_enc.transform([int(c_max_deps+1)]))[0])]
		# 	total_vec += dep_vec + word_vec + idx_vec
		#
		# if not b_fail:
		# 	els_glv_csvr.writerow([src_name] + total_vec)
		# 	print(src_name, total_vec)

	els_fh.close()
	els_glv_fh.close()


word_dict = load_word_dict()
for el_name in simple_els:
	create_simple_glv(el_name, word_dict)

dep_names_fh = open(dep_names_fn, 'rb')
dep_names_csvr = csv.reader(dep_names_fh, delimiter=' ', quoting=csv.QUOTE_NONE)
dep_enc = LabelBinarizer()
list_dep_names = [' '.join(dep_name) for dep_name in dep_names_csvr] + ['invalid']
dep_enc.fit(list_dep_names)
idx_enc = LabelBinarizer()
idx_enc.fit(range(c_max_deps+2))

for el_name in parsed_els:
	create_parsed_glv(el_name, word_dict, dep_enc, idx_enc)

print('done')
