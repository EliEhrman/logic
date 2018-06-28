from __future__ import print_function
import sys
import csv

from rules import conn_type
from rules import rec_def_type
import makerecs as mr


class cl_fixed_rules(object):
	def __init__(self, fn):
		self.__l_rules = []
		self.load_rules(fn)

	def load_rules(self, fn):
		try:
		# if True:
			with open(fn, 'rb') as fh:
				csvr = csv.reader(fh, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
				_, _, version_str = next(csvr)
				if int(version_str) != 1:
					raise ValueError('rules2 rules file', fn, 'version cannot be used. Starting from scratch')
				_, num_rules = next(csvr)
				for irule in xrange(int(num_rules)):
					srule = next(csvr)
					rule_pre = mr.extract_rec_from_str(srule[0])
					rule, var_dict = [], dict()
					for iel, el in enumerate(rule_pre):
						if el[0] in [rec_def_type.like, rec_def_type.obj] and len(el) > 3:
							var_dict[el[3]] = iel
							rule += [el[:3]]
						elif el[0] == rec_def_type.var:
							rule += [[el[0], var_dict[el[1]]]]
						else:
							rule += [el]

					self.__l_rules.append(rule)

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
		for rule in self. __l_rules:
			bitvec_mgr.add_fixed_rule(rule)
