from __future__ import print_function
import sys
from os.path import expanduser
import csv
import numpy as np

import addlearn
import config
import dmlearn

class cl_cont_stat(object):
	def __init__(self, cont):
		self.__cont = cont
		self.__match_list = []

	def get_cont(self):
		return self.__cont

	def add_match(self, bmatch):
		self.__match_list.append(bmatch)

	def get_match_list(self):
		return self.__match_list

	def set_match_list(self, match_list):
		self.__match_list = match_list


class cl_cont_stats_mgr(object):
	def __init__(self):
		self.__stats_list = []
		self.__b_cont_stats_initialized = False
		self.__cont_stats_list = []
		self.__match_list = []
		self.__nd_W = None

	def get_W(self):
		return self.__nd_W

	def set_W(self, nd_W):
		self.__nd_W = nd_W

	def is_cont_stats_initialized(self):
		return self.__b_cont_stats_initialized

	def set_cont_stats_intitalized(self, bval):
		self.__b_cont_stats_initialized = bval

	def get_cont_stats_list(self):
		return self.__cont_stats_list

	def set_cont_stats_list(self, stats_list):
		self.__cont_stats_list = stats_list

	def add_match(self, bmatch):
		self.__match_list.append(bmatch)

	def get_match_list(self):
		return self.__match_list

	def init_from_list(self, cont_list, thresh, exclude_list):
		for cont in cont_list:
			status = cont.get_status()
			if status not in exclude_list:
				if cont.get_initial_score() > thresh:
					self.__cont_stats_list.append(cl_cont_stat(cont))

	def do_learn(self):
		dmlearn.do_compare_conts_learn(self, self.__cont_stats_list)

	def predict_success_rate(self, cont_match_list):
		if self.__nd_W == None:
			print('Error for predict_success_rate. W not initialized yet.')
			return -1.0

		reclen = len(self.__cont_stats_list)
		mlist = self.get_match_list()
		mlist = [b == 'True' for b in mlist]
		numrecs = len(mlist)
		data = np.ndarray(shape=[reclen, numrecs], dtype=np.float32)
		for icont, cont_stat in enumerate(self.__cont_stats_list):
			data[icont, :] = [1.0 if b == 'True' else 0.0 for b in cont_stat.get_match_list()]
		matches = np.ndarray(shape=[numrecs], dtype=np.bool)
		matches[:] = mlist

		data = np.transpose(data)
		en = np.linalg.norm(data, axis=1)
		nz = np.nonzero(en)
		z = np.where(en == 0.0)
		zero_matches = matches[z].astype(np.float32)
		en, data, matches = en[nz], data[nz], matches[nz]
		numrecs = data.shape[0]
		data = data / en[:, None]

		query = np.ndarray(shape=[reclen], dtype=np.bool)
		query[:] = cont_match_list
		query = query.astype(np.float32)
		en = np.linalg.norm(query, axis=0)
		if abs(en) < 1.0e-5:
			zero_rate = np.average(zero_matches)
			print('predict_success_rate prediction. zero match success', zero_rate)
			return zero_rate
		test_vec = query / en
		test_vec = np.dot(test_vec, self.__nd_W)
		en = np.linalg.norm(test_vec, axis=0)
		test_vec = test_vec / en

		nd_keys = np.dot(data, self.__nd_W)
		en = np.linalg.norm(nd_keys, axis=1)
		nd_keys = nd_keys / en[:, None]

		cd = [[np.dot(test_vec, one_vec), ione] for ione, one_vec in enumerate(nd_keys)]
		cands = sorted(cd, key=lambda x: x[0], reverse=True)[1:config.c_cont_lrn_num_cd_winners+1]
		cd_winners = [cand[1] for cand in cands]
		winner_matches = (matches[cd_winners]).astype(np.float32)
		prediction = np.average(winner_matches)
		print('predict_success_rate prediction:', prediction)
		return prediction


	def save(self, fnt):
		fn = expanduser(fnt)
		fh = open(fn, 'wb')
		csvr = csv.writer(fh, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
		if self.__nd_W != None:
			num_rows_W = self.__nd_W.shape[0]
		else:
			num_rows_W = 0

		csvr.writerow(['version:', config.c_cont_file_version, 'Num conts:', len(self.__cont_stats_list),
					   'Num rows W', num_rows_W])
		csvr.writerow(self.__match_list)
		for cont_stat in self.__cont_stats_list:
			cont_stat.get_cont().save(csvr, b_write_grp_data=False)
			csvr.writerow(cont_stat.get_match_list())

		for irow in range(num_rows_W):
			csvr.writerow(self.__nd_W[irow])

		fh.close()

	def load(self, fnt):
		fn = expanduser(fnt)
		try:
			with open(fn, 'rb') as fh:
				csvr = csv.reader(fh, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
				_, version_str, _, num_conts, _, num_rows_W = next(csvr)
				if version_str != str(config.c_cont_file_version):
					raise IOError
				self.__match_list = next(csvr)
				for icont in range(int(num_conts)):
					gg_cont = addlearn.cl_add_gg(b_from_load=True)
					gg_cont.load(csvr, b_null=False)
					cont_stat = cl_cont_stat(gg_cont)
					cont_stat.set_match_list(next(csvr))
					self.__cont_stats_list.append(cont_stat)
				W = []
				num_rows_W = int(num_rows_W)
				for irow in range(num_rows_W):
					W.append([float(v) for v in next(csvr)])

				if num_rows_W > 0:
					self.__nd_W = np.asarray(W, dtype=np.float32)
		except IOError:
			print('Could not open cont stats file!')
			return False

		return True


def compare_conts():
	return