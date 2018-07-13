from __future__ import print_function
import sys
import csv

from rules import conn_type
from rules import rec_def_type
import makerecs as mr


class cl_fixed_rules(object):
	def __init__(self, fn):
		self.__l_rules = []
		self.__l_categories = []
		self.__l_names = []
		self.load_rules(fn)

	def load_rules(self, fn):
		try:
		# if True:
			with open(fn, 'rb') as fh:
				csvr = csv.reader(fh, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
				_, _, version_str = next(csvr)
				if int(version_str) != 1:
					raise ValueError('rules2 rules file', fn, 'version cannot be used. Starting from scratch')
				rule_start_marker = next(csvr)[0]
				if rule_start_marker != 'rules start':
					raise IOError('no rules start marker')
				# for irule in xrange(int(num_rules)):
				while True:
					trule = next(csvr)
					if trule[0] == 'rules end':
						break
					rule_name, category, srule = trule
					rule_pre = mr.extract_rec_from_str(srule)
					rule, var_dict = [], dict()
					for iel, el in enumerate(rule_pre):
						if el[0] in [rec_def_type.like, rec_def_type.obj] and len(el) > 3:
							var_dict[el[3]] = iel
							rule += [el[:3]]
						elif el[0] == rec_def_type.var:
							rule += [[el[0], var_dict[el[1]]]]
							if len(el) > 2:
								rule[-1] += [el[2]]
						elif el[0] == rec_def_type.conn \
								and el[1] in [	conn_type.Insert, conn_type.Modify, conn_type.Broadcast,
												conn_type.Remove, conn_type.Unique] \
								and len(el) > 2:
							rule += [el[:2] + [var_dict[e] for e in el[2:]]]
						else:
							rule += [el]

					self.__l_categories.append(category)
					self.__l_rules.append(rule)
					self.__l_names.append(rule_name)

		except ValueError as verr:
			print(verr.args)
		except IOError:
		# except:
			print('Could not open db_len_grps file! Starting from scratch.')
		except:
			print('Unexpected error:', sys.exc_info()[0])
			raise

		pass

	def add_to_bitvec_mgr(self, bitvec_mgr):
		for irule, rule in enumerate(self. __l_rules):
			bitvec_mgr.add_fixed_rule(rule, self.__l_categories[irule])

	def parse_phrase_for_mod(self, phrase):
		if phrase[0][0] != rec_def_type.conn:
			raise ValueError('Error. The first el in phrase for mod must be a rec_def_type.conn with insert, modify or remove')

		mod_type = phrase[0][1]
		# if len(phrase[0] > 2):
		# 	l_db_names = phrase[0][2:]
		# else:
		# 	l_db_names = ['main']

		remove_phrase, insert_phrase, m_unique_bels, b_insert_next = [], [], [], False
		if mod_type  == conn_type.Modify:
			for el in phrase[1:]:
				if el[0] != rec_def_type.obj:
					raise ValueError('Error. Badly formed phrase for mod', phrase)
				if b_insert_next:
					b_insert_next = False
				else:
					remove_phrase += [el[1]]
					m_unique_bels.append(True)
				insert_phrase += [el[1]]
				if len(el) > 2 and el[2] == True:
					b_insert_next = True
					insert_phrase.pop()
		if mod_type == conn_type.Unique:
			for el in phrase[1:]:
				if el[0] != rec_def_type.obj:
					raise ValueError('Error. Badly formed phrase for mod', phrase)
				insert_phrase += [el[1]]
				remove_phrase += [el[1]]
				# if len(el) > 2 and el[2] == True:
				m_unique_bels.append(len(el) > 2 and el[2] == True)
		elif mod_type == conn_type.Insert:
			insert_phrase = [el[1] for el in phrase[1:]]
		elif mod_type == conn_type.Remove:
			# remove_phrase = [el[1] for el in phrase[1:]]
			# m_unique_bels= [True for _ in phrase[1:]]
			for el in phrase[1:]:
				if el[0] != rec_def_type.obj:
					raise ValueError('Error. Badly formed phrase for mod', phrase)
				remove_phrase += [el[1]]
				# if len(el) > 2 and el[2] == True:
				m_unique_bels.append(len(el) > 2 and el[2] == True)

		return insert_phrase, remove_phrase , m_unique_bels

	def parse_phrase_for_mod_db(self, phrase):
		if phrase[0][1] not in [conn_type.Broadcast, conn_type.Insert,
										conn_type.Remove, conn_type.Modify, conn_type.Unique]:
			return None
		if len(phrase[0]) <= 2:
			return None
		db_name = phrase[0].pop()
		return db_name


