"""
mp stands for multi-player
Ths module provides support for multiple databases - often partial copies of each other
TYpically each player will have his own database representing his/her knowledge
Also, monte-carlo or tree-based search might use the db

"""

from __future__ import print_function
import numpy as np


class cl_mpdb_mgr(object):
	def __init__(self, bitvec_mgr, rule_mgr):
		self.__l_dbs = []
		self.__d_dn_names = dict()
		self.add_db('main')
		self.__bitvec_mgr = bitvec_mgr
		self.__rules_mgr = rule_mgr
		rule_mgr.add_to_bitvec_mgr(bitvec_mgr)
		pass

	def get_bitvec_mgr(self):
		return self.__bitvec_mgr

	def clear_dbs(self):
		self.__l_dbs = []
		self.__d_dn_names = dict()
		self.add_db('main')

	def add_db(self, db_name):
		idb = len(self.__l_dbs)
		self.__l_dbs.append([])
		self.__d_dn_names[db_name] = idb
		return idb

	def insert(self, l_db_names, phrase_ref):
		for db_name in l_db_names:
			idb = self.__d_dn_names.get(db_name, -1)
			if idb == -1:
				# print('Error. mpdb requested to insert into', db_name, 'which doesnt exist.')
				# continue
				idb = self.add_db(db_name)
			self.__l_dbs[idb].append(phrase_ref)
		pass

	def remove(self,  l_db_names, phrase_ref):
		for db_name in l_db_names:
			idb = self.__d_dn_names.get(db_name, -1)
			if idb == -1:
				print('Error. mpdb requested to remove from', db_name, 'which doesnt exist.')
				continue
			if phrase_ref not in self.__l_dbs[idb]:
				print('Error. mpdb requested to remove item', phrase_ref, 'from db', db_name, '.Item not found.')
				continue
			self.__l_dbs[idb].remove(phrase_ref)

	def infer(self, l_db_names_from, phase_data, l_rule_cats):
		results = []
		for db_name in l_db_names_from:
			idb = self.__d_dn_names.get(db_name, -1)
			if idb == -1:
				print('Error. mpdb requested to remove from', db_name, 'which doesnt exist.')
				continue
			for ilen, iphrase in self.__l_dbs[idb]:
				story_refs = list(self.__l_dbs[idb])
				story_refs.remove((ilen, iphrase))
				phrase = self.__bitvec_mgr.get_phrase(ilen, iphrase)
				pot_results = self.__bitvec_mgr.apply_rule(phrase, ilen, iphrase, phase_data, story_refs, l_rule_cats)
				if pot_results != []:
					results += pot_results

		return results

	def run_rule(self, stmt, phase_data, db_name, l_rule_cats):
		idb = self.__d_dn_names.get(db_name, -1)
		if idb == -1:
			print('Error. mpdb requested to run rule on db', db_name, 'which doesnt exist.')
			return None
		results = self.__bitvec_mgr.run_rule(stmt, phase_data, self.__l_dbs[idb], l_rule_cats)
		return [db_name for _ in results], results

	def learn_rule(self, stmt, l_results, phase_data, db_name):
		idb = self.__d_dn_names.get(db_name, -1)
		if idb == -1:
			print('Warning. mpdb requested to learn rule on db', db_name, 'which doesnt exist.')
			return
		return self.__bitvec_mgr.learn_rule(stmt, l_results, phase_data, self.__l_dbs[idb])

	def apply_mods(self, db_name, phrase, phase_data):
		insert_phrase, remove_phrase, m_unique_bels = self.__rules_mgr.parse_phrase_for_mod(phrase)
		# for db_name in l_db_names:
		idb = self.__d_dn_names.get(db_name, -1)
		if idb == -1:
			idb = self.add_db(db_name)
		if remove_phrase != []:
			for iphrase2, phrase2_ref in enumerate( self.__l_dbs[idb]):
				phrase2 = self.__bitvec_mgr.get_phrase(*phrase2_ref)
				if len(phrase2) != len(remove_phrase):
					continue
				bfound = True
				for breq, word, rword in zip(m_unique_bels, phrase2, remove_phrase):
					if breq and word != rword:
						bfound = False
						break
				if bfound:
					del self.__l_dbs[idb][iphrase2]
					break
		if insert_phrase != []:
			ilen, iphrase = self.__bitvec_mgr.add_phrase(insert_phrase, phase_data)
			self.__l_dbs[idb].append((ilen, iphrase))

	def show_dbs(self):
		for kdb_name, vidb in self.__d_dn_names.iteritems():
			print('db for', kdb_name)
			for iphrase, phrase_ref in enumerate( self.__l_dbs[vidb]):
				phrase = self.__bitvec_mgr.get_phrase(*phrase_ref)
				print(phrase)

	def extract_mod_db(self, l_dbs_to_mod, events_to_queue):
		for ievq, one_event_to_q in enumerate(events_to_queue):
			db_name = self.__rules_mgr.parse_phrase_for_mod_db(one_event_to_q)
			if db_name != None: l_dbs_to_mod[ievq] = db_name





