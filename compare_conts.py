from __future__ import print_function
import sys
import os.path
from os.path import expanduser
from shutil import copyfile
import csv
import numpy as np
import random

import addlearn
import config
import dmlearn
import makerecs as mr

class cl_cont_stat(object):
	def __init__(self, cont):
		self.__cont = cont
		self.__match_list = []

	def get_cont(self):
		return self.__cont

	def add_match(self, bmatch):
		self.__match_list.append(bmatch)

	def pop_match(self):
		self.__match_list.pop()

	def get_match_list(self):
		return self.__match_list

	def is_match_at_idx(self, idx):
		return self.__match_list[idx]

	def set_match_list(self, match_list):
		self.__match_list = match_list

	def remove_n_first_matches(self, n):
		self.__match_list = self.__match_list[n:]


class cl_cont_stat_pattern(object):
	def __init__(self, pattern):
		self.__pattern = pattern
		self.__idxs = []
		self.__success = 0.0
		self.__b_has_success = False
		self.__len = 0

	def add_idx(self, idx):
		b_added = False
		remove_idx = -1
		if self.__len < config.c_cont_stat_pattern_max_len:
			self.__idxs.append(idx)
			self.__len += 1
			b_added = True
		else:
			if random.random() < config.c_cont_stat_pattern_replace_prob:
				remove_idx_idx = random.randint(0, self.__len-1)
				# remove_idx = self.__idxs[remove_idx_idx]
				# del self.__idxs[remove_idx]
				remove_idx = self.__idxs.pop(remove_idx_idx)
				self.__idxs.append(idx)
				b_added = True
			else:
				b_added = False

		return b_added, remove_idx

	def calc_score(self, l_successes):
		if self.__len < config.c_cont_stat_pattern_min_for_score:
			return
		l_this_scores = [1.0 if l_successes[an_idx] else 0.0 for an_idx in self.__idxs]
		self.__success = np.mean(l_this_scores)
		# if self.__idxs[0] == 2:
		# 	print('gotit')
		self.__b_has_success = True

	def is_score_valid(self):
		return self.__b_has_success

	def get_success_score(self):
		if not self.__b_has_success:
			return -1.0

		return self.__success

class cl_cont_stats_mgr(object):
	def __init__(self):
		self.__stats_list = []
		self.__b_cont_stats_initialized = False
		self.__cont_stats_list = []
		self.__match_list = [] # match list means success list. Really got to fix this
		self.__nd_W = None
		self.__predictions_list = []
		self.__rejected_rule_list = []
		self.__nd_keys = None
		self.__nd_matches = None
		self.__nd_zero_matches = None
		self.__pattern_dict = dict()
		self.__removed_pattern_idxs = []

	def add_match_pattern(self, match_pattern):
		cs_pattern = self.__pattern_dict.get(tuple(match_pattern), None)
		if cs_pattern == None:
			cs_pattern = cl_cont_stat_pattern(match_pattern)
			self.__pattern_dict[tuple(match_pattern)] = cs_pattern

		b_added, remove_idx = cs_pattern.add_idx(len(self.__match_list))
		if remove_idx >= 0:
			self.__removed_pattern_idxs.append(remove_idx) # remove at save time
		return b_added

	def init_match_pattern_stats(self):
		match_list = list(self.__match_list)
		self.__match_list= []
		for imatch, bmatch in enumerate(match_list):
			match_pattern = []
			for cont_stat in self.__cont_stats_list:
				match_pattern.append(cont_stat.is_match_at_idx(imatch))

			_ = self.add_match_pattern(match_pattern)
			self.__match_list.append(bmatch)

		for _, cs_pattern in self.__pattern_dict.iteritems():
			cs_pattern.calc_score(self.__match_list)

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

	def add_match(self, bsuccess, match_pattern):
		b_added = self.add_match_pattern(match_pattern)
		if b_added:
			self.__match_list.append(bsuccess)
		else:
			if self.__predictions_list != []:
				self.__predictions_list.pop()
			for cstat in self.__cont_stats_list:
				cstat.pop_match()


	def add_prediction(self, prediction):
		self.__predictions_list.append(prediction)

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

	def remove_n_first_matches(self, n):
		self.__match_list = self.__match_list[n:]
		for icont, cont_stat in enumerate(self.__cont_stats_list):
			cont_stat.remove_n_first_matches(n)

	def prepare_dictance_keys(self):
		if self.__nd_W == None:
			print('Error for prepare_dictance_keys. W not initialized yet.')
			return

		reclen = len(self.__cont_stats_list)
		mlist = self.get_match_list()
		# mlist = [b == 'True' for b in mlist]
		numrecs = len(mlist)
		data = np.ndarray(shape=[reclen, numrecs], dtype=np.float32)
		for icont, cont_stat in enumerate(self.__cont_stats_list):
			data[icont, :] = [1.0 if b else 0.0 for b in cont_stat.get_match_list()]
		self.__nd_matches = np.ndarray(shape=[numrecs], dtype=np.bool)
		self.__nd_matches[:] = mlist

		data = np.transpose(data)
		en = np.linalg.norm(data, axis=1)
		nz = np.nonzero(en)
		z = np.where(en == 0.0)
		self.__nd_zero_matches = self.__nd_matches[z].astype(np.float32)
		en, data, matches = en[nz], data[nz], self.__nd_matches[nz]
		numrecs = data.shape[0]
		data = data / en[:, None]

		nd_keys = np.dot(data, self.__nd_W)
		en = np.linalg.norm(nd_keys, axis=1)
		self.__nd_keys = nd_keys / en[:, None]


	def predict_success_rate(self, cont_match_list):
		if self.__nd_W == None or self.__nd_matches == None or self.__nd_keys == None:
			print('Error for predict_success_rate. W not initialized yet.')
			return -1.0

		cs_pattern = self.__pattern_dict.get(tuple(cont_match_list), None)
		if cs_pattern != None and cs_pattern.is_score_valid():
			# print('Success rate prediction based on exact match:', cs_pattern.get_success_score())
			return cs_pattern.get_success_score()

		reclen = len(self.__cont_stats_list)
		# mlist = self.get_match_list()
		# # mlist = [b == 'True' for b in mlist]
		# numrecs = len(mlist)
		# data = np.ndarray(shape=[reclen, numrecs], dtype=np.float32)
		# for icont, cont_stat in enumerate(self.__cont_stats_list):
		# 	data[icont, :] = [1.0 if b else 0.0 for b in cont_stat.get_match_list()]
		# matches = np.ndarray(shape=[numrecs], dtype=np.bool)
		# matches[:] = mlist
		#
		# data = np.transpose(data)
		# en = np.linalg.norm(data, axis=1)
		# nz = np.nonzero(en)
		# z = np.where(en == 0.0)
		# zero_matches = matches[z].astype(np.float32)
		# en, data, matches = en[nz], data[nz], matches[nz]
		# numrecs = data.shape[0]
		# data = data / en[:, None]

		query = np.ndarray(shape=[reclen], dtype=np.bool)
		query[:] = cont_match_list
		query = query.astype(np.float32)
		en = np.linalg.norm(query, axis=0)
		if abs(en) < 1.0e-5:
			zero_rate = np.average(self.__nd_zero_matches)
			print('predict_success_rate prediction. zero match success', zero_rate)
			return zero_rate
		test_vec = query / en
		test_vec = np.dot(test_vec, self.__nd_W)
		en = np.linalg.norm(test_vec, axis=0)
		test_vec = test_vec / en

		# cd = [[np.dot(test_vec, one_vec), ione] for ione, one_vec in enumerate(self.__nd_keys)]
		# cands = sorted(cd, key=lambda x: x[0], reverse=True)[1:config.c_cont_lrn_num_cd_winners+1]
		# cd_winners = [cand[1] for cand in cands]
		cd = np.dot(self.__nd_keys, test_vec)
		cd_winners = np.argpartition(cd, -config.c_cont_lrn_num_cd_winners)[-config.c_cont_lrn_num_cd_winners:]
		top_cds = cd[cd_winners]
		winner_matches = (self.__nd_matches[cd_winners]).astype(np.float32)
		prediction = np.average(winner_matches)
		# print('predict_success_rate prediction:', prediction)
		return prediction


	def save(self, fnt):
		# clean up first

		fn = expanduser(fnt)
		if os.path.isfile(fn):
			copyfile(fn, fn + '.bak')
		fh = open(fn, 'wb')
		csvr = csv.writer(fh, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
		if self.__nd_W != None:
			num_rows_W = self.__nd_W.shape[0]
		else:
			num_rows_W = 0

		csvr.writerow(['version:', config.c_cont_file_version])
		b_has_predictions_row = len(self.__predictions_list) > 0
		csvr.writerow(['Num conts:', len(self.__cont_stats_list),
					   'Num rows W', num_rows_W, 'Has predictions row', b_has_predictions_row,
					   'Num rejected rules', len(self.__rejected_rule_list)])
		if b_has_predictions_row and len(self.__predictions_list) < len(self.__match_list):
			diff = len(self.__match_list) - len(self.__predictions_list)
			l_prediction_remove_idxs = [remove_idx - diff for remove_idx in self.__removed_pattern_idxs if remove_idx >= diff]
		else:
			l_prediction_remove_idxs = self.__removed_pattern_idxs
		match_list = [bmatch for imatch, bmatch in enumerate(self.__match_list) if imatch not in self.__removed_pattern_idxs]
		csvr.writerow(match_list)
		if b_has_predictions_row:
			predictions_list = [pred for ipred, pred in enumerate(self.__predictions_list)
								if ipred not in l_prediction_remove_idxs]
			csvr.writerow(predictions_list)
		for cont_stat in self.__cont_stats_list:
			cont_stat.get_cont().save(csvr, b_write_grp_data=False)
			cmatch_list = cont_stat.get_match_list()
			cmatch_list = [bmatch for imatch, bmatch in enumerate(cmatch_list) if imatch not in self.__removed_pattern_idxs]
			csvr.writerow(cmatch_list)
		for rej in self.__rejected_rule_list:
			srule = mr.gen_rec_str(rej)
			csvr.writerow([srule])
		for irow in range(num_rows_W):
			csvr.writerow(self.__nd_W[irow])

		fh.close()

	def load(self, fnt):
		fn = expanduser(fnt)
		try:
			with open(fn, 'rb') as fh:
				csvr = csv.reader(fh, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
				_, version_str = next(csvr)
				version_num = int(version_str)
				if version_num == config.c_cont_file_version:
					_, num_conts, _, num_rows_W, _, sb_has_predictions_row, _, s_num_rejected_rules = next(csvr)
					num_rejected_rules = int(s_num_rejected_rules)
				elif version_num == config.c_cont_file_version-1:
					_, num_conts, _, num_rows_W, _, sb_has_predictions_row = next(csvr)
					num_rejected_rules = 0
				else:
					raise IOError
				b_has_predictions_row = sb_has_predictions_row == 'True'
				self.__match_list = [sb == 'True' for sb in next(csvr)]
				if b_has_predictions_row:
					self.__predictions_list = [float(sv) for sv in next(csvr)]
				for icont in range(int(num_conts)):
					gg_cont = addlearn.cl_add_gg(b_from_load=True)
					gg_cont.load(csvr, b_null=False)
					cont_stat = cl_cont_stat(gg_cont)
					cont_stat.set_match_list([sb == 'True' for sb in next(csvr)])
					self.__cont_stats_list.append(cont_stat)
				for ireject in range(num_rejected_rules):
					rej_row = next(csvr)
					rej = mr.extract_rec_from_str(rej_row[0])
					self.__rejected_rule_list.append(rej)
				W = []
				num_rows_W = int(num_rows_W)
				for irow in range(num_rows_W):
					W.append([float(v) for v in next(csvr)])

				if num_rows_W > 0:
					self.__nd_W = np.asarray(W, dtype=np.float32)

			self.init_match_pattern_stats()

		except IOError:
			print('Could not open cont stats file!')
			return False

		return True

	def get_max_cont_id(self):
		max_cont_id = -1

		for icont, cont_stat in enumerate(self.__cont_stats_list):
			cont = cont_stat.get_cont()
			cont_id = cont.get_id()
			if cont_id > max_cont_id:
				max_cont_id = cont_id

		return max_cont_id

	def analyze(self, db_cont_mgr, b_modify):
		num_predictions = len(self.__predictions_list)
		if num_predictions < config.c_cont_stats_min_predictions:
			return

		num_matches = len(self.__match_list)
		diff = num_matches - num_predictions
		if diff > 0:
			self.remove_n_first_matches(diff)

		print('Performing analysis for', num_predictions, 'records')
		matches  = np.asarray([1.0 if b else 0.0 for b in self.get_match_list()], dtype=np.float32)
		match_list_list = []
		score_list = []
		id_list, parent_id_list = [], []

		for icont, cont_stat in enumerate(self.__cont_stats_list):
			cont = cont_stat.get_cont()
			print('Analysis for rule', icont, cont.get_rule_str(), 'blocking:', cont.is_blocking())
			data = np.asarray([1.0 if b else 0.0 for b in cont_stat.get_match_list()], dtype=np.float32)
			# nd_marker = np.asarray(cont_stat.get_match_list(), dtype=np.bool)
			all_zeros = np.all(data == 0.0)
			# if where_true[0].shape[0] > 0:
			cont_id = cont.get_id()
			id_list.append(cont_id)
			parent_id_list.append(cont.get_parent_id())

			if not all_zeros:
				where_true = np.nonzero(data)
				match_list_list.append(where_true[0].tolist())
				matches_where_true = matches[where_true]
				score = np.average(matches_where_true)
				print('score', score)
				score_list.append(score)
			else:
				match_list_list.append([])
				score_list.append(-1.0)


		def check_predict(bat, babove):
			for pval in range(13):
				plevel = float(pval - 1) / 10.0
				if bat:
					predictX = np.asarray([1.0 if (v > (plevel - 0.05) and v < (plevel + 0.05))  else 0.0
										   for v in self.__predictions_list], dtype=np.float32)
					prep = 'at'
				else:
					predictX = np.asarray([1.0 if (v > plevel if babove else v < plevel) else 0.0
										   for v in self.__predictions_list], dtype=np.float32)
					prep = 'above' if babove else 'below'
				X_all_zero = np.all(predictX==0.0)
				if X_all_zero:
					print('No items to score at predict level', prep, plevel)
				else:
					where_p = np.nonzero(predictX)
					match_where_p = matches[where_p]
					print('Score for', where_p[0].shape[0], 'items at predict level', prep, plevel, 'is', np.average(match_where_p))

		check_predict(True, True)
		check_predict(False, True)
		check_predict(False, False)

		strss = ["\t"]
		for i in range(len(match_list_list)):
			strss[0] += str(i) + "\t"

		if not b_modify:
			return

		delete_list = [False for ml in match_list_list]
		equal_list_list = [[] for ml in match_list_list]
		contains_list_list = [[] for ml in match_list_list]
		for iml, ml in enumerate(match_list_list):
			print('Analysis for match list', iml, 'length', len(ml))
			if ml == []:
				print('Marking cont', iml, 'for deletion due to zero match cases')
				delete_list[iml] = True
				continue
			strs = str(iml) + '\t'
			for iml2, ml2 in enumerate(match_list_list):
				if delete_list[iml]:
					break
				if iml == iml2:
					strs += '\t'
					continue
				print('Comparing cont', iml, 'to', iml2)
				b_sibling = (parent_id_list[iml] == parent_id_list[iml2])
				b_child = (parent_id_list[iml] == id_list[iml2])
				inlist = [v for v in ml if v in ml2]
				inlist_len = len(inlist)
				print(len(inlist), 'of match list in', iml2 )
				if b_sibling or b_child:
					if inlist_len == len(ml2) and len(ml) == len(ml2):
						print('cont', iml, 'seems identical to ', iml)
						equal_list_list[iml].append(iml2)
					else:
						inlist_fract =  float(inlist_len) / float(len(ml))
						if inlist_fract > config.c_cont_inlist_fract_thresh:
							if score_list[iml2] < config.c_cont_score_perfect_thresh:
								print('Marking cont', iml, 'for deletion because it is entirely included in cont', iml2,
									  'and the enclosing cont has perfect or almost perfect score')
								delete_list[iml] = True
							elif (1.0 - score_list[iml2]) < config.c_cont_score_perfect_thresh:
								print('Marking cont', iml, 'for deletion because it is entirely included in cont', iml2,
									  'and the enclosing cont has perfect or almost perfect block')
								delete_list[iml] = True
							elif score_list[iml] < score_list[iml2]:
								value_score = score_list[iml] / score_list[iml2]
								print('Marking cont', iml, 'because it is entirely included in cont', iml2,
									  'smaller with value score', value_score)
								contains_list_list[iml].append([iml2, value_score] )
							else:
								value_score = (1.0 - score_list[iml]) / (1.0 - score_list[iml2])
								print('Marking cont', iml, 'because it is entirely included in cont', iml2,
									  'larger with value score', value_score)
								contains_list_list[iml].append([iml2, value_score] )
						#end if equal or subset
				# end if sibling or child
				strs += str(len(inlist)) + '\t'
				if inlist != []:
					nd_inlist = np.asarray(inlist, dtype=np.int)
					matches_inlist = matches[nd_inlist]
					score = np.average(matches_inlist)
					print('inlist score ', score)
				else:
					print('No score for empty inlist')
			strss.append(strs)

		b_ipass_done = False
		for ipass in range(10): # keep a limited number of passes to keep things sane
			if b_ipass_done:
				break
			b_ipass_done = True
			for icont, contains_list in enumerate(contains_list_list):
				if contains_list == [] or delete_list[icont]:
					continue
				b_ipass_done = False
				b_this_pass = True
				for containing_data in contains_list:
					icontaining, value_score = containing_data
					if not delete_list[icontaining] and contains_list_list[icontaining] != []:
						b_this_pass = False
						break
				if not b_this_pass:
					continue

				b_keep_alive = False
				for icontaining in contains_list:
					icontaining, value_score = containing_data
					if value_score < config.c_cont_score_ratio:
						b_keep_alive = True
						break
				if b_keep_alive:
					contains_list_list[icont] = []
				else:
					delete_list[icont] = True





		for s in strss:
			print(s)

		for iequal, equal_list in enumerate(equal_list_list):
			if delete_list[iequal]:
				continue
			level = self.__cont_stats_list[iequal].get_cont().get_level()
			max_level = level
			peer_list = [iequal]
			for ipeer in equal_list:
				assert False, 'Never been tested'
				if delete_list[ipeer]:
					continue
				new_level = self.__cont_stats_list[iequal].get_cont().get_level()
				if new_level > max_level:
					print('Removing cont', ipeer, 'since it is equal to ', iequal, 'but from a higher level')
					delete_list[ipeer] = True
					continue
				if new_level < max_level:
					print('Removing cont', iequal, 'which started this list, since it is equal to ', ipeer, 'but from a higher level')
					delete_list[iequal] = True
					peer_list = []
					break
				peer_list.append(ipeer)
			isurvivor = random.choice(peer_list)
			for ipeer in peer_list:
				if ipeer != isurvivor:
					delete_list[ipeer] = True

		b_cont_list_changed = False
		for idelete, bdelete in reversed(list(enumerate(delete_list))):
			if not bdelete:
				continue
			self.__rejected_rule_list.append(self.__cont_stats_list[idelete].get_cont().get_rule())
			db_cont_mgr.delete_cont_by_rule(self.__cont_stats_list[idelete].get_cont().get_rule())
			del self.__cont_stats_list[idelete]
			b_cont_list_changed = True

		if b_cont_list_changed:
			self.__predictions_list = []
			self.remove_n_first_matches(len(self.__match_list))
			self.__nd_W = None

			# work on delete_list and equal_list_list

		return


			

	def create_new_conts(self, db_cont_mgr):
		num_predictions = len(self.__predictions_list)
		if num_predictions < config.c_cont_stats_min_predictions:
			return

		cand_list = []
		new_rule_list = []
		for icont, cont_stat in enumerate(self.__cont_stats_list):
			rule = cont_stat.get_cont().get_rule()
			rule_arr = mr.create_start_end_listing(rule)
			for rule_ends in rule_arr[1:]:
				indef_likes = mr.get_indef_likes(rule, rule_ends)
				if len(indef_likes) == 1:
					cand_list += [[icont, rule_ends, indef_likes[0]]]
		for imut in range(9):
			icont, rule_ends, iindef = random.choice(cand_list)
			cont = self.__cont_stats_list[icont].get_cont()
			rule = cont.get_rule()
			src_level = cont.get_level()
			src_gens_rec = cont.get_gens_rec()
			src_id = cont.get_id()
			if random.random() < 0.5:
				new_rule = mr.duplicate_piece(rule, rule_ends, iindef, rule, rule_ends, iindef,
											  b_make_var=random.random() > 0.5)
			else:
				other_icont, other_rule_ends, other_iindef = random.choice(cand_list)
				other_rule = self.__cont_stats_list[other_icont].get_cont().get_rule()
				if not mr.identical_rule_part(rule, [0, rule_ends[0]-1], other_rule, [0, other_rule_ends[0]-1]):
					continue
				new_rule = mr.duplicate_piece(rule, rule_ends, iindef, other_rule, other_rule_ends, other_iindef,
											  b_make_var=random.random() > 0.5)

			b_not_new = False
			for icont, cont_stat in enumerate(self.__cont_stats_list):
				rule = cont_stat.get_cont().get_rule()
				# In the following match we are ignoring the cd component of the like.
				# For now, I think that doesn't matter too much
				if mr.match_rec_exact(rule, new_rule):
					b_not_new = True
					break
			if b_not_new:
				continue
			for rej in self.__rejected_rule_list:
				if mr.match_rec_exact(rej, new_rule):
					b_not_new = True
					break
			if b_not_new:
				continue
			for other_new in new_rule_list:
				if mr.match_rec_exact(other_new[0], new_rule):
					b_not_new = True
					break
			if b_not_new:
				continue
			# new_rule_list.append(new_rule)
			new_rule_list.append([new_rule, src_level, src_gens_rec, src_id])

		b_cont_list_changed = False
		new_cont_list = []
		for new_rule in new_rule_list:
			b_cont_list_changed = True
			new_cont = db_cont_mgr.new_cont_by_rule(new_rule, db_cont_mgr.status.mutant)
			new_cont_list.append(new_cont)

		if b_cont_list_changed:
			self.__predictions_list = []
			self.remove_n_first_matches(len(self.__match_list))
			self.__nd_W = None

		for new_cont in new_cont_list:
			self.__cont_stats_list.append(cl_cont_stat(new_cont))

		return


