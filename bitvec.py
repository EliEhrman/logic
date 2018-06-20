"""
Descended from phrases.py

THis module seeks to add to the dictionary bitvec as new words come in

The skip representation will be replaced by a mask on the input for the unknown word

"""
from __future__ import print_function
import csv
import random
import sys
import copy
import os
from os.path import expanduser
from shutil import copyfile
import itertools

import numpy as np

import config
from rules import conn_type
from rules import rec_def_type
import els
import makerecs as mr


# fnt = 'orders_success.txt'
fnt = '~/tmp/adv_phrase_freq.txt'
fnt_dict = '~/tmp/adv_bin_dict.txt'


class Enum(set):
	def __getattr__(self, name):
		if name in self:
			return name
		raise AttributeError


rule_status = Enum([	'untried', 'initial', 'perfect', 'expands', 'perfect_block', 'blocks',
						'partial_expand', 'partial_block', 'irrelevant', 'mutant', 'endpoint'])

c_bitvec_size = 16
# c_min_bits = 3
# c_max_bits = 20
# c_num_replicate_missing = 5
c_bitvec_ham_winners_fraction = 32
# c_num_iters = 10000 # 300000
# c_num_skip_phrases_fraction = 0.1
# c_init_len = 4000
c_bitvec_move_rnd = 0.5
c_bitvec_move_rnd_change = 0.02
c_bitvec_min_frctn_change  = 0.001
c_bitvec_max_frctn_change  = 0.01
# c_b_init_db = False
# c_save_init_db_every = 100
c_bitvec_neibr_divider_offset = 5 # Closeness of neighbours factor
# c_add_batch = 400
# c_add_fix_iter = 20
c_bitvec_min_len_before_learn = 100

c_bitvec_gg_learn_min = 10 # must be even
c_bitvec_gg_stats_min = 10
c_bitvec_gg_initial_valid = 0.3
c_bitvec_gg_delta_on_parent = .1

assert c_bitvec_gg_learn_min % 2 == 0, 'c_bitvec_gg_learn_min must be even'

tdown = lambda l: tuple([tuple(li) for li in l])

class cl_bitvec_gg(object):
	def __init__(self, mgr, ilen, phrase_len, result, num_stages=1, l_wlist_vars = [],
				 phrase2_ilen = -1, phrase2_len = -1, parent_irule = -1):
		self.__l_phrases = []
		self.__ilen = ilen
		self.__phrase_len = phrase_len
		self.__result = result
		self.__num_since_learn = 0
		self.__mgr = mgr
		self.__l_els_rep = [[] for _ in range(phrase_len)]
		self.__l_hd_max = [c_bitvec_size for _ in range(phrase_len)]
		self.__hit_pct = 0.
		self.__b_formed = False
		self.__rule_rec = []
		self.__b_tested = False
		self.__status = rule_status.untried
		self.__num_stages = num_stages
		self.__parent_irule = parent_irule # rule and gg are interchangeable
		self.__child_rules = []
		self.__l_wlist_vars = l_wlist_vars
		self.__phrase2_ilen = phrase2_ilen
		self.__phrase2_len = phrase2_len
		self.__num_stats = 0
		self.__num_hits = 0
		self.__score = -1.0
		if num_stages == 2:
			self.__l_els_rep += [[] for _ in range(phrase2_len)]
			self.__l_hd_max += [c_bitvec_size for _ in range(phrase2_len)]

	def get_phrase2_ilen(self):
		return self.__phrase2_ilen

	def is_tested(self):
		return self.__b_tested

	def is_formed(self):
		return self.__b_formed

	def get_num_stages(self):
		return self.__num_stages

	def get_parent(self):
		return self.__mgr.get_rule(self.__parent_irule)

	def get_score(self):
		return self.__score

	def get_status(self):
		return self.__status

	def test_rule_match(self, l_wlist_vars, result, phrase2_ilen):
		if l_wlist_vars == self.__l_wlist_vars and result == self.__result and phrase2_ilen == self.__phrase2_ilen:
			return True
		return False

	def add_child_rule_id(self, child_rule_id):
		self.__child_rules.append(child_rule_id)

	def get_child_rule_ids(self):
		return self.__child_rules

	def add_phrase_stage2(self, iphrase, iphrase2):
		self.__l_phrases.append((iphrase, iphrase2))
		self.__num_since_learn += 1
		l_phrases1, l_phrases2 = [phrases[0] for phrases in self.__l_phrases], [phrases[1] for phrases in self.__l_phrases]
		if self.__num_since_learn <= c_bitvec_gg_learn_min:
			return
		# untested
			# test_set = [(self.__ilen, iphrase), (phrase2_ilen, iphrase2)]
		l_dest_var_pos = [(l_pos[2], l_pos[3]) for l_pos in self.__l_wlist_vars]
		prev_iel = 0
		rule_phrase = [[rec_def_type.conn, conn_type.AND]]
		var_pos_dict = dict()
		for istage, ilen, l_phrases in [(0, self.__ilen, l_phrases1), (1, self.__phrase2_ilen, l_phrases2)]:
			rule_phrase += [[rec_def_type.conn, conn_type.start]]
			len_phrase_bin_db = self.__mgr.get_phrase_bin_db(ilen)
			gg_bin_db = len_phrase_bin_db[l_phrases]
			phrase_len = gg_bin_db.shape[1] / c_bitvec_size
			m_match = np.ones(len(len_phrase_bin_db), dtype=bool)
			for iel in range(phrase_len):
				var_pos_dict[(istage, iel)] = len(rule_phrase)
				if (istage, iel) in l_dest_var_pos: # this is a var dest and so not compared on
					ivar = l_dest_var_pos.index((istage, iel))
					src_istage, src_ipos, _, _ = self.__l_wlist_vars[ivar]
					rule_phrase += [[rec_def_type.var, var_pos_dict[(src_istage, src_ipos)]]]
					continue
				el_bins = gg_bin_db[:, iel * c_bitvec_size:(iel + 1) * c_bitvec_size]
				els_rep = np.median(el_bins, axis=0)
				nd_diffs = np.not_equal(els_rep, el_bins)
				# nd_diffs = np.where(nd_diffs, np.ones_like(nd_phrase_bits_db), np.zeros_like(nd_phrase_bits_db))
				hd_max = np.max(np.sum(nd_diffs, axis=1))
				self.__l_els_rep[prev_iel + iel] = els_rep
				self.__l_hd_max[prev_iel + iel] = hd_max
				hd_rep = np.sum(np.not_equal(self.__mgr.get_el_db(), els_rep), axis=1)
				rep_word = self.__mgr.get_word_by_id(np.argmin(hd_rep))
				rule_phrase += [[rec_def_type.like, rep_word, hd_max]]
			prev_iel += phrase_len
			rule_phrase += [[rec_def_type.conn, conn_type.end]]

		self.__b_formed = True
		self.__rule_rec = rule_phrase
		self.__num_since_learn = 0
		if self.is_tested():
			self.__score = self.__num_hits / float(self.__num_stats)
			if self.__score > c_bitvec_gg_initial_valid:
				parent_score = self.get_parent().get_score()
				if parent_score > self.__score:
					if (self.__score - parent_score) / parent_score > c_bitvec_gg_delta_on_parent:
						self.__status = rule_status.blocking
					else:
						self.__status = rule_status.irrelevant
				else:
					if (self.__score - parent_score) / parent_score  > c_bitvec_gg_delta_on_parent:
						self.__status = rule_status.expands
					else:
						self.__status = rule_status.irrelevant
			print('status', self.__status, 'score', self.__score, 'for rule:', mr.gen_rec_str(self.__rule_rec) )

		pass

	def add_phrase(self, phrase):
		self.__l_phrases.append(phrase)
		self.__num_since_learn += 1
		if self.__num_since_learn > c_bitvec_gg_learn_min:
			len_phrase_bin_db = self.__mgr.get_phrase_bin_db(self.__ilen)
			gg_bin_db = len_phrase_bin_db[self.__l_phrases]
			# m_match = np.ones(len(len_phrase_bin_db), dtype=bool)
			rule_phrase = [[rec_def_type.conn, conn_type.start]]
			for iel in range(self.__phrase_len):
				el_bins = gg_bin_db[:, iel*c_bitvec_size:(iel+1)*c_bitvec_size]
				els_rep = np.median(el_bins, axis=0)
				nd_diffs = np.not_equal(els_rep, el_bins)
				# nd_diffs = np.where(nd_diffs, np.ones_like(nd_phrase_bits_db), np.zeros_like(nd_phrase_bits_db))
				hd_max =  np.max(np.sum(nd_diffs, axis=1))
				self.__l_els_rep[iel] = els_rep
				self.__l_hd_max[iel] = hd_max
				hd_rep = np.sum(np.not_equal(self.__mgr.get_el_db(), els_rep), axis=1)
				rep_word = self.__mgr.get_word_by_id(np.argmin(hd_rep))
				rule_phrase += [[rec_def_type.like, rep_word, hd_max]]
				# el_phrase_bins = len_phrase_bin_db[:, iel*c_bitvec_size:(iel+1)*c_bitvec_size]
				# nd_phrase_diffs = np.not_equal(els_rep, el_phrase_bins)
				# m_el_match = np.sum(nd_phrase_diffs, axis=1) <= hd_max
				# m_match = np.logical_and(m_match, m_el_match)
			# self.__hit_pct = float(len(self.__l_phrases)) / np.sum(m_match)
			self.__b_formed = True
			rule_phrase += [[rec_def_type.conn, conn_type.end]]
			self.__rule_rec = rule_phrase
			self.__num_since_learn = 0

			if self.is_tested():
				self.__score = self.__num_hits / float(self.__num_stats)
				if self.__score > c_bitvec_gg_initial_valid:
					self.__status = rule_status.initial
				print('status', self.__status, 'score', self.__score, 'for rule:', mr.gen_rec_str(self.__rule_rec) )

	def is_a_match_one_stage(self, iphrase):
		len_phrase_bin_db = self.__mgr.get_phrase_bin_db(self.__ilen)
		phrase_bin = len_phrase_bin_db[iphrase]
		for iel in range(self.__phrase_len):
			phrase_el_bins = phrase_bin[iel * c_bitvec_size:(iel + 1) * c_bitvec_size]
			hd = np.sum(np.not_equal(phrase_el_bins, self.__l_els_rep[iel]))
			if hd > self.__l_hd_max[iel]:
				return False
		return True

	def is_a_match(self, iphrase):
		if self.__num_stages == 1:
			return self.is_a_match_one_stage(iphrase)
		return False

	# assumes match has aaybeen confirmed. Justcheck the result
	def update_stats(self, phrase, l_results):
		self.__num_stats += 1
		if l_results == [] or l_results[0] == []:
			return
		_, new_result = els.replace_with_vars_in_wlist([phrase], l_results[0])
		if self.__result == new_result:
			self.__num_hits += 1
		if not self.__b_tested:
			if self.__num_stats > c_bitvec_gg_stats_min:
				self.__b_tested = True

	def find_matches(self, phrase_bin, story_bin):
		assert self.__num_stages > 1, 'function find_matches should only be called for stage 2+ rules'
		match_pat = list(self.__l_els_rep)
		l_hd_max = list(self.__l_hd_max)
		l_stagelens = [0, self.__phrase_len, self.__phrase_len + self.__phrase2_len]

		for src_istage, src_iel, dest_istage, dest_iel in self.__l_wlist_vars:
			# src_base_len = l_stagelens[dest_istage]
			dest_base_len = l_stagelens[dest_istage]
			if src_istage == 0:
				src_pat = phrase_bin[src_iel*c_bitvec_size:(src_iel+1)*c_bitvec_size].astype(float)
			else:
				assert False, 'Not coded yet phrases that depend on earlier els in phrase or earlier phrases'
			match_pat[dest_base_len+dest_iel] = src_pat
			l_hd_max[dest_base_len+dest_iel] = 0

		m_match = np.ones(story_bin.shape[0], dtype=bool)
		for iel in range(self.__phrase2_len):
			src_bin = match_pat[self.__phrase_len + iel]
			el_story_bins = story_bin[:, iel*c_bitvec_size:(iel+1)*c_bitvec_size]
			nd_el_diffs = np.not_equal(src_bin, el_story_bins)
			m_el_match = np.sum(nd_el_diffs, axis=1) <= l_hd_max[self.__phrase_len + iel]
			m_match = np.logical_and(m_match, m_el_match)

		return m_match

	def update_stats_stage_2(self, phrase, story_refs, m_matches, l_results):
		for imatch, bmatch in enumerate(m_matches.tolist()):
			if not bmatch:
				continue
			self.__num_stats += 1
			match_phrase = self.__mgr.get_phrase(self.__phrase2_ilen, story_refs[imatch])
			if l_results == [] or l_results[0] == []:
				return
			_, new_result = els.replace_with_vars_in_wlist([phrase, match_phrase], l_results[0])
			if self.__result == new_result:
				self.__num_hits += 1

		if not self.__b_tested:
			if self.__num_stats > c_bitvec_gg_stats_min:
				self.__b_tested = True
		pass


class cl_bitvec_mgr(object):
	def __init__(self):
		d_words, nd_bit_db, s_word_bit_db, l_els = load_word_db()
		# freq_tbl, s_phrase_lens = load_order_freq_tbl(fnt)
		# init_len = c_init_len  # len(freq_tbl) / 2
		# d_words, l_word_counts, l_word_phrase_ids = create_word_dict(freq_tbl, init_len)
		num_ham_winners = len(d_words) / c_bitvec_ham_winners_fraction
		score_hd_output_bits.num_ham_winners = num_ham_winners
		num_uniques = len(d_words)
		# nd_bit_db = np.zeros((num_uniques, c_bitvec_size), dtype=np.uint8)
		# s_word_bit_db = set()
		# self.__s_phrase_lens= s_phrase_lens
		self.__l_word_counts = [0 for _ in xrange(num_uniques)] # l_word_counts
		self.__l_word_phrase_ids = [[] for _ in xrange(num_uniques)]
		self.__l_word_change_db = [[[0.0 for _ in xrange(c_bitvec_size)], 0.0] for _ in self.__l_word_counts]
		self.__l_word_fix_num = [-1 for _ in xrange(num_uniques)]
		self.__d_lens = dict() # {phrase_len: ilen for ilen, phrase_len in enumerate(s_phrase_lens)}

		self.__l_phrases = [] # [[] for _ in s_phrase_lens]
		self.__d_words = d_words
		self.__phrase_bin_db = []
		self.__nd_el_bin_db = nd_bit_db
		self.__s_word_bit_db = s_word_bit_db
		self.__l_all_phrases = []
		self.__l_results = [] # alligned to all_phrases
		self.__d_gg = dict()
		self.__l_ggs = []
		self.__rule_stages = 1
		self.__d_gg2 = dict() # for two stage rules
		self.__l_els = l_els
		pass

	def increase_rule_stages(self):
		self.__rule_stages += 1

	def get_phrase_bin_db(self, ilen):
		return self.__phrase_bin_db[ilen]

	def get_phrase(self, ilen, iphrase):
		return self.__l_phrases[ilen][iphrase]

	def get_el_db(self):
		return self.__nd_el_bin_db

	def get_word_by_id(self, iel):
		return self.__l_els[iel]

	def add_phrase(self, phrase, phase_data):
		ilen, iphrase = self.__add_phrase(phrase, phase_data)
		self.__l_all_phrases.append((phase_data, ilen, iphrase))
		return ilen, iphrase

	def __add_phrase(self, phrase, phase_data):
		story_id, story_loop_stage, eid = phase_data
		self.__nd_el_bin_db, ilen, iphrase = \
			self.keep_going(phrase)
		return ilen, iphrase

	def get_rule(self, irule):
		return self.__l_ggs[irule]

	def learn_rule_one_stage(self, stmt, l_results, phase_data, l_story_db_event_refs):
		phrase = els.convert_phrase_to_word_list([stmt])[0]
		ilen, iphrase =  self.__add_phrase(phrase, phase_data)
		if l_results == []:
			return phrase, -1, ilen, iphrase
		_, vars_dict = els.build_vars_dict(stmt)
		self.__l_all_phrases.append((phase_data, ilen, iphrase))
		result = l_results[0]
		result_rec = mr.place_vars_in_phrase(vars_dict, result)
		self.__l_results.append(result_rec)
		tresult = (ilen, tdown(result_rec))
		igg = self.__d_gg.get(tresult, -1)
		if igg == -1:
			igg = len(self.__l_ggs)
			self.__d_gg[tresult] = igg
			self.__l_ggs.append(cl_bitvec_gg(self, ilen, len(phrase), result_rec))
		self.__l_ggs[igg].add_phrase(iphrase)
		return phrase, igg, ilen, iphrase


	def learn_rule_two_stages(self, stmt, l_results, phase_data, l_story_db_event_refs):
		phrase, igg1, ilen, iphrase = self.learn_rule_one_stage(stmt, l_results, phase_data, l_story_db_event_refs)
		if igg1 == -1:
			return phrase, igg1, ilen, iphrase
		gg1 = self.__l_ggs[igg1]
		if not gg1.is_tested() or gg1.get_status() == rule_status.irrelevant:
			return phrase, igg1, ilen, iphrase
		l_child_rule_ids = gg1.get_child_rule_ids()
		l_story_bins, l_story_refs = [], []
		for klen, vilen in self.__d_lens.iteritems():
			bin_dn = self.get_phrase_bin_db(vilen)
			story_refs = [tref[1] for tref in l_story_db_event_refs if tref[0] == vilen]
			if story_refs != []:
				l_story_bins.append(bin_dn[story_refs])
				l_story_refs.append((vilen, story_refs))
		# ilen, iphrase =  self.__add_phrase(phrase, phase_data)
		for iel, el in enumerate(phrase):
			el_bin = self.__nd_el_bin_db[self.__d_words[el]]
			for i_story_len, story_bin in enumerate(l_story_bins):
				db_len = story_bin.shape[1] / c_bitvec_size
				for iel2 in range(db_len):
					db_el_bin = story_bin[:, iel2*c_bitvec_size:(iel2+1)*c_bitvec_size]
					m_db = np.all(db_el_bin == el_bin, axis=1)
					if not np.any(m_db):
						continue
					for iref, bmatch in enumerate(m_db.tolist()):
						if not bmatch:
							continue
						phrase2_ilen, iphrase2 = l_story_refs[i_story_len][0], l_story_refs[i_story_len][1][iref]
						match_phrase = self.__l_phrases[phrase2_ilen][iphrase2]
						l_wlist_vars, new_result = els.replace_with_vars_in_wlist([phrase, match_phrase], l_results[0])
						# now create a rule with this information
						b_gg2_found = False
						for irule in l_child_rule_ids:
							gg2 = self.__l_ggs[irule]
							if gg2.test_rule_match(l_wlist_vars, new_result, phrase2_ilen):
								b_gg2_found = True
								break
						if not b_gg2_found:
							gg2 = cl_bitvec_gg(self, ilen, len(phrase), new_result, num_stages=2,
											   l_wlist_vars=l_wlist_vars, phrase2_ilen=phrase2_ilen,
											   phrase2_len=len(match_phrase), parent_irule=igg1)
							gg1.add_child_rule_id(len(self.__l_ggs))
							self.__l_ggs.append((gg2))
						gg2.add_phrase_stage2(iphrase, iphrase2)

					pass


		return phrase, igg1, ilen, iphrase

	learn_rule_fns = [learn_rule_one_stage, learn_rule_two_stages]

	def learn_rule(self, stmt, l_results, phase_data, l_story_db_event_refs):
		# self.learn_rule_fns[self.__rule_stages - 1](self, stmt, l_results, phase_data, l_story_db_event_refs)
		_, _, ilen, iphrase = self.learn_rule_two_stages(stmt, l_results, phase_data, l_story_db_event_refs)
		self.update_rule_stats(stmt, ilen, iphrase, l_results, l_story_db_event_refs)

	def update_rule_stats(self, stmt, ilen, iphrase, l_results, l_story_db_event_refs):
		phrase = els.convert_phrase_to_word_list([stmt])[0]
		# ilen, iphrase =  self.__add_phrase(phrase, phase_data)
		# self.__l_all_phrases.append((phase_data, ilen, iphrase))
		_, vars_dict = els.build_vars_dict(stmt)
		l_first_stage_ggs = []
		for igg, gg in enumerate(self.__l_ggs):
			if not gg.is_formed() or  gg.get_num_stages() != 1:
				continue
			if not gg.is_a_match(iphrase):
				continue
			gg.update_stats(phrase, l_results)
			if not gg.is_tested():
				continue
			l_first_stage_ggs.append(gg)

		if l_first_stage_ggs == []:
			return

		len_phrase_bin_db = self.get_phrase_bin_db(ilen)
		phrase_bin = len_phrase_bin_db[iphrase]

		# l_story_bins, l_story_refs = [], []
		d_story_bins = dict()
		for klen, vilen in self.__d_lens.iteritems():
			bin_dn = self.get_phrase_bin_db(vilen)
			story_refs = [tref[1] for tref in l_story_db_event_refs if tref[0] == vilen]
			if story_refs != []:
				d_story_bins[vilen] = (bin_dn[story_refs], story_refs)

		for gg in l_first_stage_ggs:

			l_child_rule_ids = gg.get_child_rule_ids()
			for irule in l_child_rule_ids:
				gg2 = self.__l_ggs[irule]
				if not gg2.is_formed():
					continue
				story_bin, story_refs = d_story_bins.get(gg2.get_phrase2_ilen(), ([], []))
				if story_bin == []:
					continue
				m_matches = gg2.find_matches(phrase_bin, story_bin)
				if not np.any(m_matches):
					continue
				gg2.update_stats_stage_2(phrase, story_refs, m_matches, l_results)



	def add_new_word(self, word, word_binvec):
		word_id = len(self.__d_words)
		self.__nd_el_bin_db = np.concatenate((self.__nd_el_bin_db, np.expand_dims(word_binvec, axis=0)), axis=0)
		self.__d_words[word] = word_id
		self.__l_word_change_db += [[[0.0 for ibit in xrange(c_bitvec_size)], 0.0]]
		self.__l_word_counts.append(1)
		self.__l_word_phrase_ids.append([])
		self.__l_els.append(word)
		# self.__nd_el_bin_db[word_id, :] = word_binvec
		self.__s_word_bit_db.add(tuple(word_binvec))
		self.__l_word_fix_num.append(0)

	def keep_going(self, phrase):
		phrase_bin_db, d_words, s_word_bit_db, d_lens, l_phrases, l_word_counts, l_word_phrase_ids =\
			self.__phrase_bin_db, self.__d_words, self.__s_word_bit_db, \
			self.__d_lens, self.__l_phrases, self.__l_word_counts, self.__l_word_phrase_ids
		# phrase_bin_db = build_phrase_bin_db(s_phrase_lens, l_phrases, nd_el_bin_db, d_words)
		# l_change_db = [[[0.0 for _ in xrange(c_bitvec_size)], 0.0] for _ in l_word_counts]
		num_changed = 0
		phrase_len = len(phrase)
		ilen = d_lens.get(phrase_len, -1)
		if ilen == -1:
			ilen = len(d_lens)
			d_lens[phrase_len] = ilen
			l_phrases.append([])
			phrase_bin_db.append([])
		iphrase = len(l_phrases[ilen])
		l_b_known = [True for _ in phrase]
		for iword, word in enumerate(phrase):
			word_id = d_words.get(word, -1)
			if word_id == -1:
				l_b_known[iword] = False
				while True:
					proposal = np.random.choice(a=[0, 1], size=(c_bitvec_size))
					tproposal = tuple(proposal.tolist())
					if tproposal in s_word_bit_db:
						continue
					break
				self.add_new_word(word, proposal)
				# self.__l_word_phrase_ids.append([(ilen, iphrase)])
			del word_id

		l_mbits = build_a_bit_mask(phrase_len)  # mask bits
		input_bits = create_input_bits(self.__nd_el_bin_db, d_words, phrase)
		for iskip in range(phrase_len):
			iword = d_words[phrase[iskip]]
			if iphrase > c_bitvec_min_len_before_learn:
				if self.__l_word_fix_num[iword] == 0:
					self.__nd_el_bin_db = add_new_words(self.__nd_el_bin_db, d_words, phrase_bin_db[ilen], phrase, input_bits,
												 s_word_bit_db, iskip)
					change_phrase_bin_db(phrase_bin_db, l_phrases, self.__nd_el_bin_db, d_words, iword, l_word_phrase_ids)
					self.__l_word_fix_num[iword] = 1
				else:
					score_hd_output_bits(	phrase_bin_db[ilen], input_bits,
										l_mbits[iskip], iskip, iword,
										self.__l_word_change_db, bscore=False)
					(l_bits_avg, num_hits), word_count = self.__l_word_change_db[iword], l_word_counts[iword]

					if num_hits * 2 > word_count:
						bchanged = change_bit(self.__nd_el_bin_db, s_word_bit_db, self.__nd_el_bin_db[iword], l_bits_avg, iword)
						if bchanged == 1:
							num_changed += 1
							change_phrase_bin_db(phrase_bin_db, l_phrases, self.__nd_el_bin_db, d_words, iword, l_word_phrase_ids)
						self.__l_word_change_db[iword] = [[0.0 for ibit in xrange(c_bitvec_size)], 0.0]
						if self.__l_word_fix_num[iword] != -1:
							self.__l_word_fix_num[iword] += 1
			l_word_counts[iword] += 1
			l_word_phrase_ids[iword].append((ilen, iphrase))

		l_phrases[ilen].append(phrase)
		if phrase_bin_db[ilen] == []:
			phrase_bin_db[ilen] = np.expand_dims(input_bits, axis=0)
		else:
			phrase_bin_db[ilen] = np.concatenate((phrase_bin_db[ilen], np.expand_dims(input_bits, axis=0)), axis=0)

		return self.__nd_el_bin_db, ilen, iphrase

def create_word_dict(phrase_list, max_process):
	d_els, l_presence, l_phrase_ids = dict(), [], []
	for iphrase, phrase in enumerate(phrase_list):
		if iphrase > max_process:
			break
		for iel, el, in enumerate(phrase):
			id = d_els.get(el, -1)
			if id == -1:
				d_els[el] = len(d_els)
				l_presence.append(1)
				l_phrase_ids.append([])
			else:
				l_presence[id] += 1
				# l_phrase_ids[id].append(iphrase)


	return d_els, l_presence, l_phrase_ids

def save_word_db(d_words, nd_bit_db):
	fn = expanduser(fnt_dict)

	if os.path.isfile(fn):
		copyfile(fn, fn + '.bak')
	fh = open(fn, 'wb')
	csvw = csv.writer(fh, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
	csvw.writerow(['Adv Dict', 'Version', '1'])
	csvw.writerow(['Num Els:', len(d_words)])
	for kword, virow in d_words.iteritems():
		csvw.writerow([kword, virow] + nd_bit_db[virow].tolist())

	fh.close()

def load_word_db():
	fn = expanduser(fnt_dict)
	# try:
	if True:
		with open(fn, 'rb') as o_fhr:
			csvr = csv.reader(o_fhr, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
			_, _, version_str = next(csvr)
			_, snum_els = next(csvr)
			version, num_els = int(version_str), int(snum_els)
			if version != 1:
				raise IOError
			d_words, s_word_bit_db, nd_bit_db = dict(), set(), np.zeros((num_els, c_bitvec_size), dtype=np.uint8)
			l_els = ['' for _ in xrange(num_els)]
			for irow, row in enumerate(csvr):
				word, iel, sbits = row[0], row[1], row[2:]
				d_words[word] = int(iel)
				l_els[int(iel)] = word
				bits = map(int, sbits)
				nd_bit_db[int(iel)] = np.array(bits, dtype=np.uint8)
				s_word_bit_db.add(tuple(bits))

	# except IOError:
	# 	raise ValueError('Cannot open or read ', fn)

	return d_words, nd_bit_db, s_word_bit_db, l_els

def load_order_freq_tbl(fnt):
	fn = expanduser(fnt)

	freq_tbl, d_words, s_phrase_lens = [], dict(), set()
	try:
		with open(fn, 'rb') as o_fhr:
			o_csvr = csv.reader(o_fhr, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
			_, _, version_str, _, snum_orders = next(o_csvr)
			version = int(version_str)
			if version != 1:
				raise IOError
			for iorder in range(int(snum_orders)):
				row = next(o_csvr)
				l_co_oids = next(o_csvr)
				phrase = row[2:]
				s_phrase_lens.add(len(phrase))
				# freq_tbl[tuple(phrase)] = row[0]
				freq_tbl.append(phrase)
				# for word in phrase:
				# 	id = d_words.get(word, -1)
				# 	if id == -1:
				# 		d_words[word] = len(d_words)
	except IOError:
		raise ValueError('Cannot open or read ', fn)

	# num_uniques = len(d_words)

	# return freq_tbl, num_uniques, d_words, s_phrase_lens
	# random.shuffle(freq_tbl)
	return freq_tbl, s_phrase_lens


def create_input_bits(nd_bit_db, d_words, phrase, l_b_known=[]):
	phrase_len = len(phrase)
	input_bits = np.zeros(phrase_len * c_bitvec_size, dtype=np.uint8)
	loc = 0
	for iword, word in enumerate(phrase):
		if l_b_known != [] and not l_b_known[iword]:
			input_bits[loc:loc + c_bitvec_size] = np.zeros(c_bitvec_size, dtype=np.uint8)
		else:
			input_bits[loc:loc + c_bitvec_size] = nd_bit_db[d_words[word]]
		loc += c_bitvec_size
	# missing_loc = phrase[1] * c_num_replicate_missing
	# input_bits[loc + missing_loc:loc + missing_loc + c_num_replicate_missing] = [1] * c_num_replicate_missing
	return np.array(input_bits)


def create_output_bits(sel_mat, input_bits):
	nd_bits = np.zeros(c_bitvec_size, dtype=np.int)

	for iobit in range(c_bitvec_size):
		sum = 0
		for iibit in sel_mat[iobit][0]:
			sum += input_bits[iibit]
		nd_bits[iobit] = 1 if sum >= sel_mat[iobit][1] else 0

	return nd_bits


def score_hd_output_bits(nd_phrase_bits_db, qbits, mbits, iskip, iword, change_db, bscore=True):
	numrecs = nd_phrase_bits_db.shape[0]
	hd_divider = np.array(range(c_bitvec_neibr_divider_offset, score_hd_output_bits.num_ham_winners + c_bitvec_neibr_divider_offset),
						  np.float32)
	hd_divider_sum = np.sum(1. / hd_divider)

	def calc_score(outputs):
		odiffs = np.logical_and(np.not_equal(qbits, outputs), np.logical_not(mbits))
		nd_diffs = np.where(odiffs, np.ones_like(outputs), np.zeros_like(outputs))
		divider = np.array(range(1, nd_diffs.shape[0] + 1), np.float32)
		return np.sum(np.divide(np.sum(nd_diffs, axis=1), divider))

	# nd_diffs = np.absolute(np.subtract(qbits, nd_phrase_bits_db))
	nd_diffs = np.logical_and(np.not_equal(qbits, nd_phrase_bits_db), mbits)
	nd_diffs = np.where(nd_diffs, np.ones_like(nd_phrase_bits_db), np.zeros_like(nd_phrase_bits_db))
	hd = np.sum(nd_diffs, axis=1)
	hd_winners = np.argpartition(hd, score_hd_output_bits.num_ham_winners)[:score_hd_output_bits.num_ham_winners]
	hd_of_winners = hd[hd_winners]
	iwinners = np.argsort(hd_of_winners)
	hd_idx_sorted = hd_winners[iwinners]
	winner_outputs = nd_phrase_bits_db[hd_idx_sorted]
	avg_outputs = nd_phrase_bits_db[np.random.randint(numrecs, size=hd_idx_sorted.shape[0])]
	obits = winner_outputs[:, iskip*c_bitvec_size:(iskip+1)*c_bitvec_size]
	bad_obits = avg_outputs[:, iskip*c_bitvec_size:(iskip+1)*c_bitvec_size]
	# ibits = qbits[iskip*c_bitvec_size:(iskip+1)*c_bitvec_size].astype(float)
	# obits_goal = np.where(np.average(obits, axis=0) > 0.5, np.ones_like(ibits), np.zeros_like(ibits))
	obits_goal = np.sum(obits.transpose() / hd_divider, axis=1) / hd_divider_sum
	obits_keep_away = np.average(bad_obits, axis=0)
	new_obits_goal = ((obits_goal + (np.ones(c_bitvec_size) - obits_keep_away)) / 2.0).tolist()
	if change_db[iword][1] == 0.0:
		change_db[iword][0] = new_obits_goal
	else:
		change_db[iword][0] = ((np.array(change_db[iword][0]) * change_db[iword][1]) + new_obits_goal) / (change_db[iword][1] + 1.0)
	change_db[iword][1] += 1.0
	if not bscore:
		return
	close_score, avg_score = calc_score(winner_outputs), calc_score(avg_outputs)
	return avg_score / (close_score + 10.0)


score_hd_output_bits.num_ham_winners = 0

def build_a_bit_mask(phrase_len):
	l_mbits = []
	for iskip in range(phrase_len):
		mbits = np.ones(phrase_len * c_bitvec_size, np.uint8)
		mbits[iskip*c_bitvec_size:(iskip+1)*c_bitvec_size] = np.zeros(c_bitvec_size, np.uint8)
		l_mbits.append(mbits)
	return l_mbits


def build_bit_masks(d_lens):
	l_l_mbits = [] # mask bits
	for phrase_len, ilen in d_lens.iteritems():
		l_mbits = []
		for iskip in range(phrase_len):
			mbits = np.ones(phrase_len * c_bitvec_size, np.uint8)
			mbits[iskip*c_bitvec_size:(iskip+1)*c_bitvec_size] = np.zeros(c_bitvec_size, np.uint8)
			l_mbits.append(mbits)
		l_l_mbits.append(l_mbits)
	return l_l_mbits

def change_bit(nd_bit_db, s_word_bit_db, l_bits_now, l_bits_avg, iword):
	# if random.random() < score_and_change_db.move_rnd:
	bchanged = False
	ibit = random.randint(0, c_bitvec_size - 1)
	bit_now, bit_goal = l_bits_now[ibit], l_bits_avg[ibit]
	proposal = np.copy(nd_bit_db[iword])
	if bit_now == 0 and bit_goal > 0.5:
		if random.random() < (bit_goal - 0.5):
			proposal[ibit] = 1
			bchanged = True
	elif bit_now == 1 and bit_goal < 0.5:
		if random.random() < (0.5 - bit_goal):
			proposal[ibit] = 0
			bchanged = True
	if bchanged:
		tproposal = tuple(proposal.tolist())
		if tproposal not in s_word_bit_db:
			tremove = tuple(nd_bit_db[iword].tolist())
			nd_bit_db[iword, :] = proposal
			s_word_bit_db.remove(tremove)
			s_word_bit_db.add(tproposal)
			return 1
	return 0

def score_and_change_db(s_phrase_lens, d_words, l_phrases, nd_bit_db, s_word_bit_db):
	num_uniques = len(d_words)
	l_change_db = [[[0.0 for ibit in xrange(c_bitvec_size)], 0.0] for _ in xrange(num_uniques)]
	bitvec_size = nd_bit_db.shape[1]
	num_scored = 0
	num_hits = 0
	l_l_mbits = build_bit_masks(s_phrase_lens) # mask bits

	phrase_bits_db = [np.zeros((len(l_len_phrases), bitvec_size * list(s_phrase_lens)[ilen]), dtype=np.int)
					  for ilen, l_len_phrases in enumerate(l_phrases)]
	score = 0.0
	for ilen, phrase_len in enumerate(s_phrase_lens):
		num_scored += len(l_phrases[ilen]) * phrase_len
		# sel_mat = d_phrase_sel_mats[phrase_len]
		for iphrase, phrase in enumerate(l_phrases[ilen]):
			# nd_bits = np.zeros(c_bitvec_size, dtype=np.int)
			input_bits = create_input_bits(nd_bit_db, d_words, phrase)
			phrase_bits_db[ilen][iphrase, :] = input_bits

		for iphrase, phrase in enumerate(l_phrases[ilen]):
			for iskip in range(phrase_len):
				score += score_hd_output_bits(	phrase_bits_db[ilen], phrase_bits_db[ilen][iphrase],
												l_l_mbits[ilen][iskip], iskip, d_words[phrase[iskip]],
												l_change_db)
	score /= num_scored

	num_changed = 0
	for iunique, bits_data in enumerate(l_change_db):
		l_bits_avg, _ = bits_data
		l_bits_now = nd_bit_db[iunique]
		num_changed += change_bit(nd_bit_db, s_word_bit_db, l_bits_now, l_bits_avg, iunique)
		# if random.random() < score_and_change_db.move_rnd:
		# 	bchanged = False
		# 	ibit = random.randint(0, c_bitvec_size-1)
		# 	bit_now, bit_goal = l_bits_now[ibit], l_bits_avg[ibit]
		# 	proposal = np.copy(nd_bit_db[iunique])
		# 	if bit_now == 0 and bit_goal > 0.5:
		# 		if random.random() < (bit_goal - 0.5):
		# 			proposal[ibit] = 1
		# 			bchanged = True
		# 	elif bit_now == 1 and bit_goal < 0.5:
		# 		if random.random() < (0.5 - bit_goal):
		# 			proposal[ibit] = 0
		# 			bchanged = True
		# 	if bchanged:
		# 		tproposal = tuple(proposal.tolist())
		# 		if tproposal not in s_word_bit_db:
		# 			tremove = tuple(nd_bit_db[iunique].tolist())
		# 			nd_bit_db[iunique, :] = proposal
		# 			s_word_bit_db.remove(tremove)
		# 			s_word_bit_db.add(tproposal)
		# 			num_changed += 1
	frctn_change = float(num_changed) / float(num_uniques * c_bitvec_size)
	if frctn_change < c_bitvec_min_frctn_change:
		score_and_change_db.move_rnd += c_bitvec_move_rnd_change
	elif frctn_change > c_bitvec_max_frctn_change:
		score_and_change_db.move_rnd -= c_bitvec_move_rnd_change
	print(num_changed, 'bits changed out of', num_uniques * c_bitvec_size, 'fraction:',
		  frctn_change, 'move_rnd = ', score_and_change_db.move_rnd)
	return score



score_and_change_db.move_rnd = c_bitvec_move_rnd

# def select_best(s_phrase_lens, d_words, l_phrases, l_objs, iiter, l_record_scores, l_record_objs, best_other, b_do_dbs):
# 	min_score, max_score = sys.float_info.max, -sys.float_info.max
# 	num_objs = len(l_objs)
# 	l_scores = []
# 	for iobj in range(num_objs):
# 		if b_do_dbs:
# 			nd_bit_db = l_objs[iobj]
# 			d_phrase_sel_mats = best_other
# 		else:
# 			nd_bit_db = best_other
# 			d_phrase_sel_mats = l_objs[iobj]
# 		score = score_db_and_sel_mat(s_phrase_lens, d_words, l_phrases, nd_bit_db, d_phrase_sel_mats)
# 		l_scores.append(score)
# 		if score > max_score:
# 			max_score = score
# 		if score < min_score:
# 			min_score = score
#
# 	# print('avg score:', np.mean(l_scores)) # , 'list', l_scores)
# 	print('iiter', iiter, 'avg score:', np.mean(l_scores), 'max score:', np.max(l_scores)) # , 'list', l_scores)
# 	if l_record_scores == [] or max_score > l_record_scores[0]:
# 		l_record_scores.insert(0, max_score)
# 		l_record_objs.insert(0, l_objs[l_scores.index(max_score)])
# 	else:
# 		l_objs[l_scores.index(min_score)] = l_record_objs[0]
# 		l_scores[l_scores.index(min_score)] = l_record_scores[0]
# 	# mid_score = (max_score + min_score) / 2.0
# 	mid_score = l_scores[np.array(l_scores).argsort()[c_mid_score]]
# 	if max_score == min_score:
# 			range_scores = max_score
# 			l_obj_scores = np.ones(len(l_scores), dtype=np.float32)
# 	elif mid_score == max_score:
# 		range_scores = max_score - min_score
# 		l_obj_scores = np.array([(score - min_score) / range_scores for score in l_scores])
# 	else:
# 		range_scores = max_score - mid_score
# 		l_obj_scores = np.array([(score - mid_score) / range_scores for score in l_scores])
# 	l_obj_scores = np.where(l_obj_scores > 0.0, l_obj_scores, np.zeros_like(l_obj_scores))
# 	sel_prob = l_obj_scores/np.sum(l_obj_scores)
# 	l_sel_dbs = np.random.choice(num_objs, size=num_objs, p=sel_prob)
# 	l_objs[:] = [copy.deepcopy(l_objs[isel]) for isel in l_sel_dbs]

# def mutate_dbs(l_dbs, num_uniques):
# 	num_flip_muts = int(c_db_num_flip_muts * num_uniques)
#
# 	for idb, nd_bit_db in enumerate(l_dbs):
# 		if random.random() < c_db_rnd_asex:
# 			for imut in range(num_flip_muts):
# 				allele, target = random.randint(0, c_bitvec_size - 1), random.randint(0, num_uniques-1)
# 				nd_bit_db[target][allele] = 1 if (nd_bit_db[target][allele] == 0) else 0
# 		elif random.random() < c_db_rnd_sex:
# 			partner_db = copy.deepcopy(random.choice(l_dbs))  # not the numpy function
# 			for allele in range(c_bitvec_size):
# 				for iun in range(num_uniques):
# 					if random.random() < 0.5:
# 						nd_bit_db[iun,:] = partner_db[iun,:]
#
# def mutate_sel_mats(l_d_phrase_sel_mats, s_phrase_lens):
# 	for isel, d_phrase_sel_mats in enumerate(l_d_phrase_sel_mats):
# 		for ilen, phrase_len in enumerate(s_phrase_lens):
# 			num_input_bits = ((phrase_len - 1) * c_bitvec_size) + (c_num_replicate_missing * phrase_len)
# 			sel_mat = d_phrase_sel_mats[phrase_len]
# 			if random.random() < c_rnd_asex:
# 				for imut in range(c_num_incr_muts):
# 					allele = random.randint(0, c_bitvec_size-1)
# 					num_bits = len(sel_mat[allele][0])
# 					if sel_mat[allele][1] < num_bits-2:
# 						sel_mat[allele][1] += 1
# 				for imut in range(c_num_incr_muts):
# 					allele = random.randint(0, c_bitvec_size-1)
# 					if sel_mat[allele][1] > 1:
# 						sel_mat[allele][1] -= 1
# 				for icmut in range(c_num_change_muts):
# 					allele = random.randint(0, c_bitvec_size-1)
# 					bit_list = sel_mat[allele][0]
# 					if random.random() < c_change_mut_prob_change_len:
# 						if len(bit_list) < c_max_bits:
# 							bit_list.append(random.randint(0, num_input_bits - 1))
# 					elif random.random() < c_change_mut_prob_change_len:
# 						if len(bit_list) > c_min_bits:
# 							bit_list.pop(random.randrange(len(bit_list)))
# 							if sel_mat[allele][1] >= len(bit_list) - 1:
# 								sel_mat[allele][1] -= 1
# 					else:
# 						for ichange in range(c_change_mut_num_change):
# 							bit_list[random.randint(0, len(bit_list)-1)] = random.randint(0, num_input_bits - 1)
# 			elif random.random() < c_rnd_sex:
# 				partner_sel_mat = copy.deepcopy(random.choice(l_d_phrase_sel_mats)[phrase_len]) # not the numpy function
# 				for allele in range(c_bitvec_size):
# 					if random.random() < 0.5:
# 						sel_mat[allele] = list(partner_sel_mat[allele])

def build_phrase_bin_db(s_phrase_lens, l_phrases, nd_el_bin_db, d_words):
	phrase_bits_db = [np.zeros((len(l_len_phrases), c_bitvec_size * list(s_phrase_lens)[ilen]), dtype=np.int)
					  for ilen, l_len_phrases in enumerate(l_phrases)]
	score = 0.0
	for ilen, phrase_len in enumerate(s_phrase_lens):
		for iphrase, phrase in enumerate(l_phrases[ilen]):
			# nd_bits = np.zeros(c_bitvec_size, dtype=np.int)
			input_bits = create_input_bits(nd_el_bin_db, d_words, phrase)
			phrase_bits_db[ilen][iphrase, :] = input_bits

	return phrase_bits_db

def change_phrase_bin_db(phrase_bits_db, l_phrases, nd_el_bin_db, d_words, iword, l_word_phrase_ids):
	score = 0.0
	for ilen, iphrase in l_word_phrase_ids[iword]:
		phrase = l_phrases[ilen][iphrase]
		input_bits = create_input_bits(nd_el_bin_db, d_words, phrase)
		phrase_bits_db[ilen][iphrase, :] = input_bits


# ilen is the index number of the list of phrase grouped by phrase len (not the length of the phrase)
# iphrase is index in that list of phrases of that length
def add_new_words(	nd_bit_db, d_words, nd_phrase_bits_db, phrase, phrase_bits, s_word_bit_db,
					iword):
	divider = np.array(range(c_bitvec_neibr_divider_offset, score_hd_output_bits.num_ham_winners + c_bitvec_neibr_divider_offset),
					   np.float32)
	divider_sum = np.sum(1. / divider)
	phrase_len = len(phrase)
	mbits = np.ones(phrase_len * c_bitvec_size, np.uint8)
	mbits[iword * c_bitvec_size:(iword + 1) * c_bitvec_size] = np.zeros(c_bitvec_size, np.uint8)

	nd_diffs = np.logical_and(np.not_equal(phrase_bits, nd_phrase_bits_db), mbits)
	nd_diffs = np.where(nd_diffs, np.ones_like(nd_phrase_bits_db), np.zeros_like(nd_phrase_bits_db))
	hd = np.sum(nd_diffs, axis=1)
	hd_winners = np.argpartition(hd, score_hd_output_bits.num_ham_winners)[:score_hd_output_bits.num_ham_winners]
	hd_of_winners = hd[hd_winners]
	iwinners = np.argsort(hd_of_winners)
	hd_idx_sorted = hd_winners[iwinners]
	winner_outputs = nd_phrase_bits_db[hd_idx_sorted]
	word_id = d_words[phrase[iword]]
	obits = winner_outputs[:, iword*c_bitvec_size:(iword+1)*c_bitvec_size]
	new_vals = np.sum(obits.transpose() / divider, axis=1) / divider_sum
	# round them all and if the pattern is already there switch the closest to 0.5
	new_bits = np.round_(new_vals).astype(np.uint8)
	s_word_bit_db.remove(tuple(nd_bit_db[word_id]))
	if tuple(new_bits) in s_word_bit_db:
		bfound = False
		while True:
			can_flip = np.argsort(np.square(new_vals - 0.5))
			for num_flip in range(1, c_bitvec_size):
				try_flip = can_flip[:num_flip]
				l = [list(itertools.combinations(try_flip, r)) for r in range(num_flip+1)]
				lp = [item for sublist in l for item in sublist]
				for p in lp:
					pbits = list(new_bits)
					for itf in try_flip:
						pbits[itf] = 1 if itf in p else 0
					if tuple(pbits) not in s_word_bit_db:
						new_bits = pbits
						bfound = True
						break
				if bfound:
					break
			if bfound:
				break

	s_word_bit_db.add(tuple(new_bits))
	nd_bit_db[word_id] = new_bits
	return nd_bit_db



def main():
	raise ValueError('why here')
	# success_orders_freq = dict()
	freq_tbl, s_phrase_lens = load_order_freq_tbl(fnt)
	init_len = c_init_len # len(freq_tbl) / 2
	d_words, l_word_counts, l_word_phrase_ids = create_word_dict(freq_tbl, init_len)
	num_ham_winners = len(d_words) / c_bitvec_ham_winners_fraction
	score_hd_output_bits.num_ham_winners= num_ham_winners
	num_uniques = len(d_words)
	nd_bit_db = np.zeros((num_uniques, c_bitvec_size), dtype=np.uint8)
	s_word_bit_db = set()
	for iunique in range(num_uniques):
		while True:
			proposal = np.random.choice(a=[0, 1], size=(c_bitvec_size))
			tproposal = tuple(proposal.tolist())
			if tproposal in s_word_bit_db:
				continue
			nd_bit_db[iunique, :] = proposal
			s_word_bit_db.add(tproposal)
			break
	l_change_db = [[[0.0 for ibit in xrange(c_bitvec_size)], 0.0] for _ in xrange(num_uniques)]
	# d_phrase_sel_mats, d_lens = dict(), dict()
	# for ilen, phrase_len in enumerate(s_phrase_lens):
	# 	num_input_bits = phrase_len * c_bitvec_size
	# 	sel_mat = []
	# 	for ibit in range(c_bitvec_size):
	# 		num_bits = random.randint(c_min_bits, c_max_bits)
	# 		l_sels = []
	# 		for isel in range(num_bits):
	# 			l_sels.append(random.randint(0, num_input_bits-1))
	# 		sel_mat.append([l_sels, random.randint(1, num_bits)])
	# 	d_phrase_sel_mats[phrase_len] = sel_mat
	# 	d_lens[phrase_len] = ilen
	d_lens = {phrase_len:ilen for ilen, phrase_len in enumerate(s_phrase_lens)}

	l_phrases = [[] for _ in s_phrase_lens]
	for iphrase, phrase in enumerate(freq_tbl):
		if iphrase >= init_len:
			break
		plen = len(phrase)
		l_phrases[d_lens[plen]].append(phrase)

	for ilen, phrases in enumerate(l_phrases):
		for iphrase, phrase in enumerate(phrases):
			for iel, el in enumerate(phrase):
				id = d_words[el]
				l_word_phrase_ids[id].append((ilen, iphrase))

	if c_b_init_db:
		for iiter in range(c_num_iters):
			score = score_and_change_db(s_phrase_lens, d_words, l_phrases, nd_bit_db, s_word_bit_db)
			print('iiter', iiter, 'score:', score)  # , 'list', l_scores)
			if iiter % c_save_init_db_every == 0:
				save_word_db(d_words, nd_bit_db)
		return
	else:
		d_words, nd_bit_db, s_word_bit_db, _ = load_word_db()

	add_start = c_init_len
	while (add_start < len(freq_tbl)):
		nd_bit_db = keep_going(	freq_tbl, d_words, nd_bit_db, s_word_bit_db, s_phrase_lens,
								l_phrases, l_word_counts, l_word_phrase_ids, add_start, c_add_batch)
		print('Added', c_add_batch, 'phrases after', add_start)
		add_start += c_add_batch
		for iiter in range(c_add_fix_iter):
			score = score_and_change_db(s_phrase_lens, d_words, l_phrases, nd_bit_db, s_word_bit_db)
			print('iiter', iiter, 'score:', score)  # , 'list', l_scores)




