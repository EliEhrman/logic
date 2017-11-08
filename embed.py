from __future__ import print_function
import csv
import sys
import os
from sklearn.preprocessing import LabelBinarizer

c_num_gloves = 400000
c_max_deps = 4

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
			els_glv_csvr.writerow([el_string] + el_vec)
			print (row[0], el_vec)

	els_fh.close()
	els_glv_fh.close()


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
		for i_dep in range(c_max_deps):
			field_id = 2 + (i_dep * 3)
			if i_dep < num_deps:
				dep, idx_ref, word = row[field_id:field_id+3]
				dep_vec = [float(i) for i in list((dep_enc.transform([dep]))[0])]
				word_vec = word_dict.get(word.lower(), None)
				if not word_vec:
					b_fail = True
					break
				idx_vec = [float(i) for i in list((idx_enc.transform([int(idx_ref)]))[0])]
			else:
				dep_vec = [float(i) for i in list((dep_enc.transform(['invalid']))[0])]
				word_vec = [0.0] * g_word_vec_len
				idx_vec = [float(i) for i in list((idx_enc.transform([int(c_max_deps+1)]))[0])]
			total_vec += dep_vec + word_vec + idx_vec

		if not b_fail:
			els_glv_csvr.writerow([src_name] + total_vec)
			print(src_name, total_vec)

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
