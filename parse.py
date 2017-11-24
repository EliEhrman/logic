"""
This module is a stand-alone pre-processor for the logic project

It takes files such as lists of objects, with some properties and uses the Stanford parser to create a text file .nlp
that lists the dependencies tree of the phrase. Format of this file is:
1. word of phrase
2. Number of dependencies listed
3, List of dep tree elements:
	a. dep name
	b. which word in the phrase is its governor
	c. word itself

This file is used by the embed module to build a embedding for the phrase
"""

from __future__ import print_function
import csv
import sys
import os
import string
import subprocess
# from xml.etree.ElementTree import ElementTree as ET
import xml.etree.ElementTree as ET

b_init = True

NlpDir = '/guten/StanfordNLP36'
# NlpCmd = 'java'
NlpArgs = 'java -cp "*" -Xmx2g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse,dcoref -file /devlink2/dev/logic/<els>.dot.txt -outputDirectory /devlink2/dev/logic/ -outputExtension .xml; exit 0'

els_files = ['objects', 'actions']
els_fn = '.txt'
els_dot_fn = '.dot.txt'
els_xml_fn = '.dot.txt.xml'
els_nlp_fn = '.txt.nlp'
nlp_deps_fn = 'nlp_deps.txt'

def parse_els_file(el_name, depset):
	if b_init:
		els_fh = open(el_name+els_fn, 'rb')
		els_csvr = csv.reader(els_fh, delimiter=' ', quoting=csv.QUOTE_NONE)

		els_dot_fh = open(el_name+els_dot_fn, 'wb')
		els_dot_csvr = csv.writer(els_dot_fh, delimiter=' ')

		for irow, row in enumerate(els_csvr):
			els_dot_csvr.writerow(row + ['.'])

		els_dot_fh.close()
		els_fh.close()

		s_cwd = os.getcwd()
		os.chdir(NlpDir)
		# new_cwd = os.getcwd()
		NlpCmd = string.replace(NlpArgs, '<els>', el_name)
		ret_str = subprocess.check_output([NlpCmd], stderr=subprocess.STDOUT, shell=True)
		os.chdir(s_cwd)

	els_nlp_fh = open(el_name+els_nlp_fn, 'wb')
	els_nlp_csvr = csv.writer(els_nlp_fh, delimiter=',')

	tree = ET.parse(el_name+els_xml_fn)
	root = tree.getroot()
	for dep in root.iter('dependencies'):
		if dep.attrib['type'] != 'basic-dependencies':
			continue
		deplist = []
		phrase = [None] * len(dep.getchildren())
		if len(phrase) < 2:
			continue
		for onedep in dep.getchildren():
			deptype = onedep.attrib['type']
			if deptype == 'punct':
				continue
			depset.add(deptype)
			for govdep in onedep.getchildren():
				if govdep.tag == 'governor':
					govidx = govdep.attrib['idx']
				elif govdep.tag == 'dependent':
					deptext = govdep.text
					depidx = govdep.attrib['idx']
			try:
				phrase[int(depidx)] = deptext
			except:
				print('key error')
			deplist += [deptype, govidx, deptext]
		els_nlp_csvr.writerow([' '.join(phrase[1:])] + [str(len(phrase)-1)] + deplist)

	els_nlp_fh.close()

depset = set()
for name in els_files:
	parse_els_file(name, depset)

nlp_deps_fh = open(nlp_deps_fn, 'wb')
nlp_deps_csvr = csv.writer(nlp_deps_fh, delimiter=' ')
for depname in depset:
	nlp_deps_csvr.writerow([str(depname)])

nlp_deps_fh.close()

# sentences = root.getchildren()[0].getchildren()[0].getchildren()
# print(sentences)




