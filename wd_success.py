import os
from os.path import expanduser
import numpy as np
import csv
from StringIO import StringIO
import copy

import wdconfig

class cl_alliance_stats(object):
	def __init__(self, country_names_tbl, alliance_fnt, terr_fnt, unit_fnt, alliance_sel_mat_fnt):
		d_stmts = dict()
		for icountry, scountry in enumerate(country_names_tbl):
			if icountry == 0:
				continue
			for icountry2, scountry2 in enumerate(country_names_tbl):
				if icountry2 == 0 or icountry == icountry2:
					continue
				d_stmts[(scountry, 'allied', 'to', scountry2)] = len(d_stmts)

			for gtu in range(wdconfig.c_max_units_for_status):
				d_stmts[(scountry, 'has', str(gtu + 1), 'or', 'more', 'units')] = len(d_stmts)

		# l_phrases = [[] for _ in range(len(d_stmts))]
		# for kphrase, vidx in d_stmts.iteritems():
		# 	l_phrases[vidx] = kphrase
		# for iphrase, phrase in enumerate(l_phrases):
		# 	print(iphrase, ':', phrase)


		self.__d_alliance_stmts = d_stmts
		self.__d_terr_owns_stmts = None
		self.__d_unit_in_stmts = None


		self.__country_names_tbl = country_names_tbl
		self.__d_history = dict()
		self.__turn_data = []
		self.__l_nd_alliance_status = []
		self.__l_nd_terr_status = []
		self.__l_nd_unit_status = []
		self.__l_l_num_units = []
		self.__l_updated = []
		self.__l_diffs = []
		self.__l_saved = []
		# if os.path.isfile(fn):
		# 	copyfile(fn, fn + '.bak')
		self.__alliance_fn = expanduser(alliance_fnt)
		self.__unit_fn = expanduser(unit_fnt)
		self.__terr_fn = expanduser(terr_fnt)
		self.__alliance_sel_mat_fn = expanduser(alliance_sel_mat_fnt)
		self.__nd_diffs_db = []
		self.__nd_alliance_status_db = []
		self.__alli_sel_mat = []

		self.load()

	def init_terr_and_army_stmts(self, terr_type_tbl):
		if self.__d_terr_owns_stmts != None:
			return

		d_stmts = dict()
		for icountry, scountry in enumerate(self.__country_names_tbl):
			# if icountry == 0:
			# 	continue

			for kterr, vtype in terr_type_tbl.iteritems():
				if vtype == 'sea':
					continue
				d_stmts[(scountry, 'owns', kterr)] = len(d_stmts)

		# l_phrases = [[] for _ in range(len(d_stmts))]
		# for kphrase, vidx in d_stmts.iteritems():
		# 	l_phrases[vidx] = kphrase
		# for iphrase, phrase in enumerate(l_phrases):
		# 	print(iphrase, ':', phrase)
		self.__d_terr_owns_stmts = d_stmts

		d_stmts = dict()
		for icountry, scountry in enumerate(self.__country_names_tbl):
			if icountry == 0:
				continue

			for kterr, vtype in terr_type_tbl.iteritems():
				for unit_type in ['army', 'fleet']:
					if unit_type == 'army' and vtype == 'sea':
						continue
					if unit_type == 'fleet' and vtype == 'land':
						continue
					d_stmts[(scountry, 'owns', unit_type, 'in', kterr)] = len(d_stmts)

		# l_phrases = [[] for _ in range(len(d_stmts))]
		# for kphrase, vidx in d_stmts.iteritems():
		# 	l_phrases[vidx] = kphrase
		# for iphrase, phrase in enumerate(l_phrases):
		# 	print(iphrase, ':', phrase)
		self.__d_unit_in_stmts = d_stmts

	def create_bit_vector(self, statement_list, unit_owns_tbl):
		nd_alliance_status = np.zeros(len(self.__d_alliance_stmts), dtype=np.int32)
		nd_terr_status = np.zeros(len(self.__d_terr_owns_stmts), dtype=np.int32)
		nd_unit_status = np.zeros(len(self.__d_unit_in_stmts), dtype=np.int32)
		# status_stmts = []
		l_num_units = [0 for _ in self.__country_names_tbl]
		for stmt in statement_list:
			if 'allied' in stmt:
				nd_alliance_status[self.__d_alliance_stmts[tuple(stmt)]] = 1
			if 'owns' in stmt:
				if len(stmt) == 3:
					nd_terr_status[self.__d_terr_owns_stmts[tuple(stmt)]] = 1
				else:
					nd_unit_status[self.__d_unit_in_stmts[tuple(stmt)]] = 1
			# status_stmts.append(stmt)

		for icountry, scountry in enumerate(self.__country_names_tbl):
			l_units = unit_owns_tbl.get(scountry, [])
			num_units = min(len(l_units), wdconfig.c_max_units_for_status)
			l_num_units[icountry] = num_units
			for gtu in range(num_units):
				stmt = (scountry, 'has', str(gtu+1), 'or', 'more', 'units')
				nd_alliance_status[self.__d_alliance_stmts[stmt]] = 1
				# status_stmts.append(list(stmt))

		return nd_alliance_status, nd_terr_status, nd_unit_status, l_num_units

	def transform_bitvec_to_key(self, bitvec, sel_mat):
		key = []

		for iobit, sm_data in enumerate(sel_mat):
			sum = 0
			for iibit in sm_data[0]:
				sum += bitvec[iibit]
			key.append(1 if sum >= sm_data[1] else 0)

		return np.array(key)

	def predict_force_diffs(self, statement_list, unit_owns_tbl, modify_list = []):
		whatif_statement_list, whatif_unit_owns_tbl = copy.deepcopy(statement_list), copy.deepcopy(unit_owns_tbl)
		for amod in modify_list:
			mod_type, mod_del, mod_add = amod
			if mod_type == 0:
				if mod_del != [] and mod_del in whatif_statement_list:
					whatif_statement_list.remove(mod_del)
				if mod_add != []:
					whatif_statement_list.append(mod_add)
			else:
				print('Only supporting statement list modification for now')
				exit(1)

		bitvec, _, _, _ = self.create_bit_vector(whatif_statement_list, whatif_unit_owns_tbl)
		nd_key = self.transform_bitvec_to_key(bitvec, self.__alli_sel_mat)
		hd = np.sum(np.absolute(np.subtract(nd_key, self.__nd_alliance_status_db)), axis=1)
		num_k = wdconfig.c_alliance_prediction_k
		hd_winners = np.argpartition(hd, num_k)[:num_k]
		hd_nearest = hd[hd_winners]
		nd_nearest = self.__nd_diffs_db[hd_winners]
		return np.mean(nd_nearest, axis=0)

		# self.__l_nd_alliance_status_db = np.stack(l_nd_keys)

	def update_status_bitvec_data(self, gameID, turn_id, statement_list, unit_owns_tbl):
		if self.__d_history.get((gameID, turn_id), -1) != -1:
			return

		nd_alliance_status, nd_terr_status, nd_unit_status, l_num_units = self.create_bit_vector(statement_list, unit_owns_tbl)

		if turn_id >= wdconfig.c_alliance_stats_turn_delay:
			old_turn = turn_id - wdconfig.c_alliance_stats_turn_delay
			status_idx = self.__d_history.get((gameID, old_turn), -1)
			if status_idx >= 0:
				self.__l_diffs[status_idx] = [nu_new - nu_old for nu_new, nu_old in zip(l_num_units, self.__l_l_num_units[status_idx])]
				self.__l_updated[status_idx] = True
		# self.__status_history[(gameID, turn_id)] = [nd_alliance_status, l_num_units,
		self.__d_history[(gameID, turn_id)] = len(self.__l_nd_alliance_status)
		self.__turn_data = (gameID, turn_id)
		self.__l_nd_alliance_status.append(nd_alliance_status)
		self.__l_nd_terr_status.append(nd_terr_status)
		self.__l_nd_unit_status.append(nd_unit_status)
		self.__l_l_num_units.append(l_num_units)
		self.__l_updated.append(False)
		self.__l_saved.append(False)
		self.__l_diffs.append([])

		return

	def save(self):

		alli_fh = open(self.__alliance_fn, 'ab')
		terr_fh = open(self.__terr_fn, 'ab')
		unit_fh = open(self.__unit_fn, 'ab')
		alli_csvr = csv.writer(alli_fh, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
		terr_csvr = csv.writer(terr_fh, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
		unit_csvr = csv.writer(unit_fh, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
		# num_alli_bits, num_terr_bits, num_unit_bits = len(self.__d_alliance_stmts), len(self.__d_terr_owns_stmts), \
		# 											  len(self.__d_unit_in_stmts)
		for istatus, ndstatus in enumerate(self.__l_nd_alliance_status):
			if not self.__l_updated[istatus] or self.__l_saved[istatus]:
				continue
			s_alli_status = ''.join(['1' if b == 1 else '0' for b in self.__l_nd_alliance_status[istatus]])
			s_terr_status = ''.join(['1' if b == 1 else '0' for b in self.__l_nd_terr_status[istatus]])
			s_unit_status = ''.join(['1' if b == 1 else '0' for b in self.__l_nd_unit_status[istatus]])
			alli_csvr.writerow([self.__turn_data, s_alli_status, self.__l_diffs[istatus]])
			terr_csvr.writerow([self.__turn_data, s_terr_status, self.__l_diffs[istatus]])
			unit_csvr.writerow([self.__turn_data, s_unit_status, self.__l_diffs[istatus]])
			self.__l_saved[istatus] = True

		alli_fh.close()
		terr_fh.close()
		unit_fh.close()

	def load(self):
		with open(self.__alliance_sel_mat_fn, 'rb') as alli_sel_mat_fh:
			alli_sel_mat_csvr = csv.reader(alli_sel_mat_fh, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
			sel_mat = []
			for irow, row in enumerate(alli_sel_mat_csvr):
				s_sel_mat_row, sthresh = row
				f = StringIO(s_sel_mat_row[1:-1])
				scsv = csv.reader(f, delimiter=',')
				sel_mat.append([[int(d) for r in scsv for d in r], int(sthresh)])
			self.__alli_sel_mat = sel_mat
			alli_sel_mat_fh.close()

		with open(self.__alliance_fn, 'rb') as alli_fh:
			alli_csvr = csv.reader(alli_fh, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')

			# self.__l_nd_alliance_status_db, self.__l_diffs_db = [], []
			l_nd_keys, l_diffs_db = [], []
			for irow, row in enumerate(alli_csvr):
				sturn_data, sbitvec, sdiffs = row

				f = StringIO(sdiffs[1:-1])
				scsv = csv.reader(f, delimiter=',')
				l_diffs_db.append(np.array([int(d) for r in scsv for d in r]))
				bitvec = [1 if c == '1' else 0 for c in sbitvec]
				# key = []
				# for iobit, sm_data in enumerate(self.__alli_sel_mat):
				# 	sum = 0
				# 	for iibit in sm_data[0]:
				# 		sum += bitvec[iibit]
				# 	key.append(1 if sum >= sm_data[1] else 0)
				l_nd_keys.append(self.transform_bitvec_to_key(bitvec, self.__alli_sel_mat))

			self.__nd_alliance_status_db, self.__nd_diffs_db = np.stack(l_nd_keys), np.stack(l_diffs_db)
			alli_fh.close()
