from __future__ import print_function
from contextlib import closing
import MySQLdb
import urllib2
import string
import random
import time
import csv
import sys
import os
import collections

# from builtins import enumerate
from enum import Enum
import embed
import utils
import els
import dmlearn
import clrecgrp
import compare_conts
import wdconfig
from wdconfig import e_move_type
import wdlearn
import wd_imagine
import wd_admin
import wd_classicAI

# response = urllib2.urlopen("http://localhost/gamemaster.php?gameMasterSecret=")
nt_order_status = collections.namedtuple('nt_order_status', 'order_num, status, unitID, fromTerrID, toTerrID, iref')


def complete_game_init(db, cursor, gameID, l_humaan_countries):
	if l_humaan_countries != None and l_humaan_countries != []:
		print('function complete_game_init not designed for play against humans')
		return

	sqlGetStatus = string.Template('select userID, orderStatus from webdiplomacy.wD_Members where gameID = \'${gameID}\';')
	sql = sqlGetStatus.substitute(gameID=gameID)

	cursor.execute(sql)
	results = cursor.fetchall()
	l_present_uids = [row[0] for row in results]

	if len(l_present_uids) >= 7:
		print('cant help complete pre-game. All players present')
		return

	AddPlayers(db, cursor, gameID, l_except=l_present_uids)

def wait_to_play(db, cursor, gameID, l_humaans):
	sqlGetStatus = string.Template('select userID, orderStatus from webdiplomacy.wD_Members where gameID = \'${gameID}\';')
	sql = sqlGetStatus.substitute(gameID=gameID)
	# b_all_waiting = False
	while True:
		cursor.execute(sql)
		results = cursor.fetchall()
		l_humaan_country_ids = []
		b_waiting_for_AI = False
		for row in results:
			for row in results:
				if row[0] in l_humaans:
					continue

				if row[1] == '':
					b_waiting_for_AI = True

		if b_waiting_for_AI:
			break

		db.commit()
		time.sleep(5.0)


def get_humaan_countries(cursor, gameID, l_humaans):
	sqlFindCountries = string.Template('select userID, countryID from webdiplomacy.wD_Members where gameID = \'${gameID}\';')
	sql = sqlFindCountries.substitute(gameID=gameID)
	cursor.execute(sql)
	results = cursor.fetchall()
	l_humaan_country_ids = []
	for row in results:
		if row[0] in l_humaans:
			l_humaan_country_ids.append(row[1])

	return l_humaan_country_ids

def find_human_game(db, cursor):
	sqlGetGames = 'select id, name, phase from webdiplomacy.wD_Games;'
	sqlFindStarters = string.Template('select userID from webdiplomacy.wD_Members where gameID = \'${gameID}\';')
	sql = sqlGetGames
	# sql = sqlGetGame.substitute(gameID=gameID)
	cursor.execute(sql)
	results = cursor.fetchall()
	b_found_game = False
	for row in results:
		if row == None:
			continue
		gameID, gname, phase = row
		if gname[:len(wdconfig.c_gname_human_prefix)] != wdconfig.c_gname_human_prefix:
			continue
		if phase == 'Pre-game':
			# sqlStarters = sqlFindStarters.substitute(gameID=gameID)
			# cursor.execute(sqlStarters)
			# resultsStarters = cursor.fetchall()
			# for rowStarters in resultsStarters:
			# 	uid = rowStarters[0]
			# 	if uid in wdconfig.c_human_uids:
			AddPlayers(db, cursor, gameID, l_except=wdconfig.c_human_uids)
			b_found_game = True
			break
		elif phase == 'Finished':
			continue
		else:
			b_found_game = True
			break
	if b_found_game:
		return gameID, gname, wdconfig.c_human_uids

	return -1, '', []

	# gameID, gname, _ = result
	# l_except = []
	# sql = sqlFindStarters.substitute(gameID=gameID)
	# cursor.execute(sql)
	# results = cursor.fetchall()
	# for row in results:
	# 	l_except.append(row[0])
	#
	# if len(l_except) >= 7:
	# 	return -1, '', []
	#
	# AddPlayers(db, cursor, gameID, l_except)
	#
	# return gameID, gname, l_except


def get_game_name(cursor, gameID):
	sqlGetGame = string.Template('select id, name, phase from webdiplomacy.wD_Games where id = \'${gameID}\';')
	sql = sqlGetGame.substitute(gameID=gameID)
	cursor.execute(sql)
	result = cursor.fetchone()
	if result == None or result[0] != gameID:
		return None, False
	if result[2] == 'Finished':
		return result[1], True
	return result[1], False

def get_game_id(cursor, gname):
	sqlGetGame = string.Template('select id, phase, turn from webdiplomacy.wD_Games where name = \'${gname}\';')
	sql = sqlGetGame.substitute(gname=gname)
	cursor.execute(sql)
	result = cursor.fetchone()
	if result == None:
		return None, None, None
	return result[0], result[1], int(result[2])


def get_phase(cursor, gameID):
	sqlGetPhase = string.Template('SELECT phase FROM webdiplomacy.wD_Games WHERE id = ${gameID}')

	sql = sqlGetPhase.substitute(gameID=gameID)
	cursor.execute(sql)
	result = cursor.fetchone()
	return result[0]


def AddPlayers(db, cursor, gameID, l_except=[]):
	sqlAddPlayer = string.Template(
		'INSERT INTO webdiplomacy.wD_Members SET userID = ${userid}, countryID = 0, bet = 10, '
		'timeLoggedIn = 1516721310, gameID = ${gameID}, orderStatus = \'None,Completed,Ready\';')

	for i in range(7):
		uid = i + wdconfig.c_starting_user_id

		if uid in l_except:
			continue

		sql = sqlAddPlayer.substitute(gameID=gameID, userid=str(uid))
		cursor.execute(sql)

	db.commit()


def CreateGame(db, cursor, gname):
	sqlNewGame = string.Template('INSERT INTO webdiplomacy.wD_Games '
								 'SET '
								 'variantID=1, '
								 'name = \'${gname}\', '
								 'potType = \'Winner-takes-all\', '
								 'pot = 0, '
								 'minimumBet = 10, '
								 'anon = \'No\', '
								 'pressType = \'Regular\', '
								 'processTime = ${stime}, '
								 'phaseMinutes = 1440, '
								 'missingPlayerPolicy = \'Normal\', '
								 'drawType=\'draw-votes-public\', '
								 'minimumReliabilityRating=0 ')
	stime = str(int(time.time()))
	sql = sqlNewGame.substitute(gname=gname, stime=stime)
	cursor.execute(sql)
	db.commit()


def OwnsUnitsTbl(cursor, gameID, country_names_tbl, statement_list, orders_status_list):
	sqlUnitsOwned = string.Template('SELECT u.countryID, LOWER(u.type), LOWER(t.name), u.id, u.terrID '
									'FROM webdiplomacy.wD_Units AS u '
									'INNER JOIN webdiplomacy.wD_Territories AS t ON ('
									'u.terrID = t.id AND t.mapID = 1 '
									') '
									'WHERE u.gameID = ${gameID};')
	owns_tbl = dict()
	sql = sqlUnitsOwned.substitute(gameID=str(gameID))
	cursor.execute(sql)
	results = cursor.fetchall()
	unit_dict = dict()
	for row in results:
		db_str = country_names_tbl[row[0]] + ' owns ' + row[1]
		country = country_names_tbl[row[0]]
		owns_list = owns_tbl.get(country, None)
		if owns_list == None:
			owns_tbl[country] = []
		owns_tbl[country].append([row[1], row[2]])
		db_str += ' at ' + row[2]
		statement_list.append([country, 'owns', row[1], 'in', row[2]])
		unit_dict[row[3]] = row[4]
		# print(db_str)
	if len(results) > 0:
		del row, db_str, country, owns_list

	temp_orders_status_list = []
	for order_status in orders_status_list:
		uID, expected_terrID, status = order_status.unitID, order_status.toTerrID, order_status.status
		if expected_terrID == None or not status:
			temp_orders_status_list.append(order_status)
			continue
		terrID = unit_dict.get(uID, None)
		if terrID == None or terrID != expected_terrID:
			temp_orders_status_list.append(order_status._replace(status=False))
			continue
		temp_orders_status_list.append(order_status)

	if len(orders_status_list) > 0:
		del order_status, uID, expected_terrID, status

	new_orders_status_list = []
	for temp_status in temp_orders_status_list:
		iref = temp_status.iref
		if iref < 0:
			new_orders_status_list.append(temp_status)
			continue
		else:
			new_status = temp_orders_status_list[iref].status
			new_orders_status_list.append(temp_status._replace(status=new_status))


	return owns_tbl, new_orders_status_list

def OwnsTerrTbl(cursor, gameID, country_names_tbl, statement_list):
	sqlTerrOwned = string.Template('SELECT ts.countryID, LOWER(t.name), ts.occupyingUnitID '
								   'FROM webdiplomacy.wD_TerrStatus AS ts '
								   'INNER JOIN webdiplomacy.wD_Territories AS t ON ('
								   'ts.terrID = t.id AND mapID = 1 '
								   ')'
								   'where gameID = ${gameID};')
	owns_tbl = dict()
	sql = sqlTerrOwned.substitute(gameID=str(gameID))
	cursor.execute(sql)
	results = cursor.fetchall()
	for row in results:
		db_str = country_names_tbl[row[0]] + ' owns ' + row[1] + ' occupied by ' + str(row[2])
		country = country_names_tbl[row[0]]
		owns_list = owns_tbl.get(country, None)
		if owns_list == None:
			owns_tbl[country] = []
		owns_tbl[country].append([row[1], row[2]])
		if wdconfig.c_b_add_owns_to_phrases:
			statement_list.append([country, 'owns', row[1]])
		if row[2] == None:
			statement_list.append([row[1], 'is', 'unoccupied'])
		else:
			statement_list.append([row[1], 'is', 'occupied'])

		# print(db_str)
	return owns_tbl


def PassInsert(cursor, spass, type, can_pass_tbl, statement_list):
	sqlPass = string.Template('SELECT LOWER(tf.name), LOWER(tt.name) '
							  'FROM webdiplomacy.wD_Borders AS b '
							  'INNER JOIN webdiplomacy.wD_Territories AS tf ON ('
							  'b.fromTerrID = tf.id AND b.mapID = tf.mapID'
							  ') '
							  'INNER JOIN webdiplomacy.wD_Territories AS tt ON ('
							  'b.toTerrID = tt.id AND b.mapID = tt.mapID'
							  ') '
							  'where b.mapID = 1 AND b.$whichpass = \'Yes\';')
	sqlTPass = sqlPass.substitute(whichpass=spass)
	cursor.execute(sqlTPass)

	results = cursor.fetchall()
	for row in results:
		db_str = type + ' can move from ' + row[0] + ' to ' + row[1]
		if wdconfig.c_b_add_can_pass_to_phrases:
			statement_list.append([type, 'can', 'pass', 'from', row[0], 'to', row[1]])
		pass_list = can_pass_tbl.get(row[0], None)
		if pass_list == None:
			can_pass_tbl[row[0]] = []
			pass_list = []  # which is the same as None I think
		if row[1] not in pass_list:
			can_pass_tbl[row[0]].append(row[1])
		# print(db_str)

		# print(len(results), 'results returned.')


def create_terr_id_tbl(cursor, terr_id_tbl, country_names_tbl, statement_list):
	sqlGetTerrID = 'SELECT id, LOWER(name), LOWER(type), countryID FROM webdiplomacy.wD_Territories WHERE mapID = 1;'
	cursor.execute(sqlGetTerrID)
	results = cursor.fetchall()
	for row in results:
		terr_id_tbl[row[1]] = row[0]
		# statement_list.append([row[1], 'is', 'a', row[2]])
		if wdconfig.c_b_add_native_to_phrases:
			statement_list.append([row[1], 'is', 'a', 'native', 'of', country_names_tbl[row[3]]])


def create_supply_tbl(cursor, country_names_tbl, statement_list):
	sqlForSupplyTbl = ('SELECT countryID, LOWER(type), LOWER(name) FROM webdiplomacy.wD_Territories '
					   'WHERE mapID = 1 AND supply = \'Yes\';')

	supply_tbl = dict()
	cursor.execute(sqlForSupplyTbl)
	results = cursor.fetchall()
	for row in results:
		scountry = country_names_tbl[row[0]]
		supply_list = supply_tbl.get(scountry, None)
		if supply_list == None:
			supply_tbl[scountry] = []
		supply_tbl[scountry].append([row[1], row[2]])

		db_str = scountry + ' has a ' + row[1] + ' supply at ' + row[2]
		if wdconfig.c_b_add_supply_to_phrases:
			statement_list.append([row[2], 'is', 'a', 'supply'])
		# print(db_str)
	return supply_tbl

def create_terr_type_tbl(cursor, statement_list):
	sqlForTerrTypeTbl = ('SELECT LOWER(type), LOWER(name) FROM webdiplomacy.wD_Territories '
					   'WHERE mapID = 1;')

	terr_type_tbl = dict()
	cursor.execute(sqlForTerrTypeTbl)
	results = cursor.fetchall()
	for row in results:
		terr_type, terr_name = row
		terr_type_tbl[terr_name] = terr_type
		terr_stmt = [terr_name, 'is', 'a', terr_type]
		if wdconfig.c_b_add_terr_type_to_phrases:
			statement_list.append(terr_stmt)
		# print(' '.join(terr_stmt))
	return terr_type_tbl

def create_retreat_orders(db, cursor, gameID, l_humaan_countries, country_names_tbl, terr_id_tbl, supply_tbl, unit_owns_tbl, sqlOrderComplete):
	sqlGetBuildOrders = string.Template('SELECT countryID, LOWER(type), id from webdiplomacy.wD_Orders where gameID = ${gameID};')
	sqlBuildOrder = string.Template('UPDATE webdiplomacy.wD_Orders SET type=\'${scmd}\', '
										 'toTerrID = ${toTerrID} WHERE id = ${id} ;')

	sql = sqlGetBuildOrders.substitute(gameID=gameID)
	print(sql)
	cursor.execute(sql)
	results = cursor.fetchall()
	country_order_ids = set()
	for row in results:
		scountry = country_names_tbl[row[0]]
		country_order_ids.add(row[0])
		# if row[1] == 'Retreat':
		# 	unit_list = unit_owns_tbl.get(scountry, None)
		# 	if unit_list == None:
		# 		continue
		# 	supply_list = supply_tbl[scountry]
		# 	build_loc = random.choice(supply_list)
		# 	scmd = row[1]
		# 	if build_loc[0] == 'Coast' and random.random() > 0.5:
		# 		scmd = 'Build Fleet'
		# 	dest_id = terr_id_tbl[build_loc[1]]
		# 	sql = sqlBuildOrder.substitute(scmd=scmd, toTerrID=dest_id, id=row[2] )
		# else:
		# 	owns_list = unit_owns_tbl[scountry]
		# 	destroy_data = random.choice(owns_list)
		# 	dest_id = terr_id_tbl[destroy_data[1]]
		# 	sql = sqlBuildOrder.substitute(scmd=row[1], toTerrID=dest_id, id=row[2] )
		# print(sql)
		# cursor.execute(sql)

	for icountry in country_order_ids:
		if icountry in l_humaan_countries:
			continue
		sql = sqlOrderComplete.substitute(gameID=str(gameID), countryID=str(icountry), timeLoggedIn=str(int(time.time())))
		print(sql)
		cursor.execute(sql)

	db.commit()


def create_build_orders(db, cursor, gameID, l_humaan_countries, country_names_tbl, terr_id_tbl, supply_tbl,
						unit_owns_tbl, terr_owns_tbl, sqlOrderComplete):
	sqlGetBuildOrders = string.Template('SELECT countryID, LOWER(type), id from webdiplomacy.wD_Orders where gameID = ${gameID};')
	sqlBuildOrder = string.Template('UPDATE webdiplomacy.wD_Orders SET type=\'${scmd}\', '
										 'toTerrID = ${toTerrID} WHERE id = ${id} ;')

	build_opts = dict()
	for icountry in range(1, len(country_names_tbl)):
		scountry = country_names_tbl[icountry]
		terr_owns_list = terr_owns_tbl.get(scountry, None)
		if terr_owns_list == None:
			continue
		terr_unocc = [terr[0] for terr in terr_owns_list if terr[1] == None]
		supply_list = supply_tbl[scountry]
		supply_avail_list = []
		for supply in supply_list:
			if supply[1] in terr_unocc:
				supply_avail_list.append(supply)
		if supply_avail_list != []:
			build_opts[scountry] = supply_avail_list


	sql = sqlGetBuildOrders.substitute(gameID=gameID)
	print(sql)
	cursor.execute(sql)
	results = cursor.fetchall()
	country_order_ids = set()
	for row in results:
		if row[0] in l_humaan_countries:
			continue
		scountry = country_names_tbl[row[0]]
		country_order_ids.add(row[0])
		b_order_valid = False
		if row[1] == 'build army':
			build_opts_list = build_opts.get(scountry, None)
			if build_opts_list == None:
				continue
			build_loc = random.choice(build_opts_list)
			build_opts_list.remove(build_loc)
			scmd = 'Build Army'
			if build_loc[0] == 'coast' and random.random() > 0.5:
				scmd = 'Build Fleet'
			dest_id = terr_id_tbl[build_loc[1]]
			sql = sqlBuildOrder.substitute(scmd=scmd, toTerrID=dest_id, id=row[2] )
			b_order_valid = True
			dstr = scountry + ' ' + scmd + ' in ' + build_loc[1]
		else:
			owns_list = unit_owns_tbl.get(scountry, [])
			if owns_list != []:
				destroy_data = random.choice(owns_list)
				dest_id = terr_id_tbl[destroy_data[1]]
				sql = sqlBuildOrder.substitute(scmd='Destroy', toTerrID=dest_id, id=row[2] )
				b_order_valid = True
				dstr = scountry + ' destroy in ' + destroy_data[1]
		if b_order_valid:
			print(dstr)
			print(sql)
			cursor.execute(sql)

	for icountry in country_order_ids:
		if icountry in l_humaan_countries:
			continue
		sql = sqlOrderComplete.substitute(gameID=str(gameID), countryID=str(icountry), timeLoggedIn=str(int(time.time())))
		print(sql)
		cursor.execute(sql)

	db.commit()

def create_move_orders2(db, cursor, gameID, l_humaan_countries, sql_complete_order,
						sql_get_unit_id, l_sql_action_orders,
						unit_owns_tbl, terr_id_tbl,
						country_names_tbl, army_can_pass_tbl, fleet_can_pass_tbl,
						ref_orders_list, orders_status_list,
						init_db, status_db, db_cont_mgr, all_the_dicts,
						terr_owns_tbl, supply_tbl):
	# nt_order_status = collections.namedtuple('nt_order_status', 'order_num, status, unitID, fromTerrID, toTerrID')

	sql_move_order, sql_support_order, sql_support_hold_order, sql_convoy_order, sql_hold_order = l_sql_action_orders

	orders_list, orders_db, success_list, icountry_list = \
		wd_classicAI.create_move_orders2(	init_db, army_can_pass_tbl, fleet_can_pass_tbl, status_db,
										db_cont_mgr, country_names_tbl, unit_owns_tbl,
										all_the_dicts, terr_owns_tbl, supply_tbl, wdconfig.c_num_montes,
										wdconfig.c_preferred_nation, wdconfig.c_b_predict_success)

	move_template = ['?', 'in', '?', 'move', 'to', '?']
	convoy_move_template = ['?', 'in', '?', 'convoy', 'move', 'to', '?']
	support_template = ['?', 'in', '?', 'support', 'move', 'from', '?', 'to', '?']
	convoy_template = ['fleet', 'in', '?', 'convoy', 'army', 'in', '?', 'to', '?']
	hold_template = ['?', 'in', '?', 'hold']
	support_hold_template = ['?', 'in', '?', 'support', 'hold', 'in', '?']

	move_dict, convoy_move_dict, hold_dict, convoy_set = dict(), dict(), dict(), set()
	l_order_data = []

	for iorder, order in enumerate(orders_list):
		print('orig order', ' '.join(order))
		if utils.match_list_for_blanks(l_with_blanks=move_template, l_to_match=order):
			sutype, _, src_name, _, _, dest_name = order
			# unit_terr_id = terr_id_tbl[src_name]
			# move_type = e_move_type.move
			move_dict[(src_name, dest_name)] = iorder
			l_order_data.append([e_move_type.move, sutype, src_name, dest_name])
		elif utils.match_list_for_blanks(l_with_blanks=hold_template, l_to_match=order):
			sutype, _, src_name, _ = order
			# unit_terr_id = terr_id_tbl[src_name]
			# move_type = e_move_type.hold
			hold_dict[src_name] = iorder
			l_order_data.append([e_move_type.hold, sutype, src_name])
		elif utils.match_list_for_blanks(l_with_blanks=convoy_move_template, l_to_match=order):
			sutype, _, src_name, _, _, _, dest_name = order
			# unit_terr_id = terr_id_tbl[src_name]
			# move_type = e_move_type.convoy_move
			convoy_move_dict[(src_name, dest_name)] = iorder
			l_order_data.append([e_move_type.convoy_move, sutype, src_name, dest_name])
		elif utils.match_list_for_blanks(l_with_blanks=support_template, l_to_match=order):
			sutype, _, supporting_name, _, _, _, src_name, _, dest_name = order
			# unit_terr_id = terr_id_tbl[supporting_name]
			# from_terr_id = terr_id_tbl[src_name]
			# move_type = e_move_type.support
			hold_dict[supporting_name] = iorder
			l_order_data.append([e_move_type.support, sutype, supporting_name, src_name, dest_name])
		elif utils.match_list_for_blanks(l_with_blanks=support_hold_template, l_to_match=order):
			sutype, _, supporting_name, _, _, _, dest_name = order
			# unit_terr_id = terr_id_tbl[supporting_name]
			# move_type = e_move_type.support_hold
			hold_dict[supporting_name] = iorder
			l_order_data.append([e_move_type.support_hold, sutype, supporting_name, dest_name])
		elif utils.match_list_for_blanks(l_with_blanks=convoy_template, l_to_match=order):
			sutype, _, convoying_name, _, _, _, src_name, _, dest_name = order
			# unit_terr_id = terr_id_tbl[convoying_name]
			# from_terr_id = terr_id_tbl[src_name]
			# move_type = e_move_type.convoy
			hold_dict[convoying_name] = iorder
			convoy_set.add((src_name, dest_name))
			l_order_data.append([e_move_type.convoy, sutype, convoying_name, src_name, dest_name])
		else:
			print('create_move_orders. Unknown move template')
			move_type = e_move_type.none
			continue

	for i_order_data, order_data in enumerate(l_order_data):
		print('process order', ' '.join(orders_list[i_order_data]))
		b_do_sql = success_list[i_order_data]
		move_type, sutype, acting_name = order_data[0:3]
		iref = -1
		if move_type == e_move_type.move:
			src_name, dest_name = acting_name, order_data[-1]
		elif move_type == e_move_type.hold:
			dest_name = acting_name
		elif move_type == e_move_type.convoy_move:
			src_name, dest_name = acting_name, order_data[-1]
			if (src_name, dest_name) not in convoy_set:
				b_do_sql = False
		elif move_type == e_move_type.support:
			src_name, dest_name = order_data[3:]
			iref = move_dict.get((src_name, dest_name), -1)
			if iref == -1:
				b_do_sql = False
		elif move_type == e_move_type.support_hold:
			dest_name = order_data[-1]
			iref = hold_dict.get(dest_name, -1)
			if iref == -1:
				b_do_sql = False
		elif move_type == e_move_type.convoy:
			src_name, dest_name = order_data[3:]
			# if (src_name, dest_name) not in convoy_move_set:
			iref = convoy_move_dict.get((src_name, dest_name), -1)
			if iref == -1:
				b_do_sql = False

		if iref >= 0:
			print('target move finally happened!')

		unit_terr_id = terr_id_tbl[acting_name]

		sql_get = sql_get_unit_id.substitute(gameID=str(gameID), terrID=str(unit_terr_id))
		cursor.execute(sql_get)
		unit_id = cursor.fetchone()
		dest_id = terr_id_tbl[dest_name]
		src_id = terr_id_tbl[src_name]
		if move_type == e_move_type.move:
			order_status = nt_order_status(order_num=iorder, status=b_do_sql, unitID=unit_id[0],
										   fromTerrID=unit_terr_id, toTerrID=dest_id, iref=-1)
		elif move_type == e_move_type.convoy_move:
			order_status = nt_order_status(order_num=iorder, status=b_do_sql, unitID=unit_id[0],
										   fromTerrID=unit_terr_id, toTerrID=dest_id, iref=-1)
		elif move_type == e_move_type.hold:
			order_status = nt_order_status(order_num=iorder, status=success_list[iorder], unitID=unit_id[0],
										   fromTerrID=unit_terr_id, toTerrID=unit_terr_id, iref=-1)
		elif move_type == e_move_type.support or move_type == e_move_type.convoy \
				or move_type == e_move_type.support_hold:
			order_status = nt_order_status(order_num=iorder, status=(success_list[iorder] and iref >=0),
										   unitID=unit_id[0], fromTerrID=unit_terr_id, toTerrID=unit_terr_id,
										   iref = iref)

		orders_status_list.append(order_status)

		if success_list[iorder] and b_do_sql:
			if move_type == e_move_type.move:
				sql_order = sql_move_order.substitute(	toTerrID=str(dest_id), unitID=str(unit_id[0]),
														gameID=str(gameID), bConvoyed='No')
			elif move_type == e_move_type.hold:
				sql_order = sql_hold_order.substitute(unitID=str(unit_id[0]), gameID=str(gameID))
			elif move_type == e_move_type.convoy_move:
				sql_order = sql_move_order.substitute(	toTerrID=str(dest_id), unitID=str(unit_id[0]),
														gameID=str(gameID), bConvoyed='Yes')
			elif move_type == e_move_type.support:
				sql_order = sql_support_order.substitute(	toTerrID=str(dest_id), unitID=str(unit_id[0]),
															gameID=str(gameID), fromTerrID=str(src_id))
			elif move_type == e_move_type.support_hold:
				sql_order = sql_support_hold_order.substitute(toTerrID=str(dest_id), unitID=str(unit_id[0]),
															  gameID=str(gameID))
			elif move_type == e_move_type.convoy:
				sql_order = sql_convoy_order.substitute(	toTerrID=str(dest_id), unitID=str(unit_id[0]),
															gameID=str(gameID), fromTerrID=str(src_id))

			print(sql_order)
			cursor.execute(sql_order)

		dstr = ' '.join(orders_list[i_order_data])
		print(dstr)

	for icountry in icountry_list:
		sql_order = sql_complete_order.substitute(gameID=str(gameID), countryID=str(icountry), timeLoggedIn=str(int(time.time())))
		print(sql_order)
		cursor.execute(sql_order)

	ref_orders_list[:] = []
	ref_orders_list += orders_list
	db.commit()



def create_oracle_move_orders(db, cursor, gameID, l_humaan_countries, sql_complete_order,
							  sql_get_unit_id, l_sql_action_orders,
							  unit_owns_tbl, terr_id_tbl, terr_type_tbl,
							  country_names_tbl, army_can_pass_tbl, fleet_can_pass_tbl,
							  orders_list, orders_status_list):

	sql_move_order, sql_support_order, sql_support_hold_order, sql_convoy_order, sql_hold_order = l_sql_action_orders

	who_is_where_tbl = dict()
	for icountry in range(1, len(country_names_tbl)):
		scountry = country_names_tbl[icountry]
		unit_list = unit_owns_tbl.get(scountry, None)
		if unit_list == None or unit_list == []:
			continue
		random.shuffle(unit_list)
		if random.random() < wdconfig.c_oracle_fleet_first_prob:
			unit_list = sorted(unit_list, key=lambda x: x[0], reverse=True)
		unit_owns_tbl[scountry] = unit_list
		for iunit, unit_data in enumerate(unit_list):
			who_is_where_tbl[unit_data[1]] = [icountry, unit_data[0], iunit]


	for icountry in range(1, len(country_names_tbl)):
		if icountry in l_humaan_countries:
			continue
		scountry = country_names_tbl[icountry]
		print('Orders for ', scountry)
		unit_list = unit_owns_tbl.get(scountry, None)
		if unit_list == None:
			continue
		move_details_tbl = []

		l_units_avail = [True for u in unit_list]
		for iunit, unit_data in enumerate(unit_list):
			if not l_units_avail[iunit]:
				continue
			unit_terr_id = terr_id_tbl[unit_data[1]]
			# sqlIDbyTerr = string.Template('SELECT id FROM webdiplomacy.wD_Units where gameID = ${gameID} and terrID = ${terrID};')
			sql_get = sql_get_unit_id.substitute(gameID=str(gameID), terrID=str(unit_terr_id))
			cursor.execute(sql_get)
			unit_id = cursor.fetchone()

			nt_move_details = collections.namedtuple('nt_order_details', 'country, sutype, fromName, toName, bMove, iorder')
			# e_move_type = Enum('e_move_type', 'none move support')

			sutype, sfrom = unit_data
			this_can_pass_tbl = army_can_pass_tbl if sutype == 'army' else fleet_can_pass_tbl
			other_can_pass_tbl = army_can_pass_tbl if sutype == 'fleet' else fleet_can_pass_tbl

			def create_move(move_details_tbl):
				order_status = nt_order_status(order_num=len(orders_list), status=True, unitID=unit_id[0],
											   fromTerrID=unit_terr_id, toTerrID=None, iref=-1)
				pass_list = this_can_pass_tbl.get(sfrom, None)
				if pass_list == None:
					print(sutype, 'stuck where it is not supposed to be.')
					return False, e_move_type.none, -1
				b_dont_support = False
				if random.random() >= wdconfig.c_rnd_bad_move:
					for itry in range(20):
						if itry > 18:
							return False, e_move_type.none, -1
						dest_name = random.choice(pass_list)
						b_dest_blocked = False
						for dest_unit_data in unit_list:
							_, dest_from = dest_unit_data
							if dest_name == dest_from: # There's a guy there already
								b_dest_blocked = True
								b_dest_is_moving = False
								for one_move in move_details_tbl:
									if one_move.fromName == dest_name:
										b_dest_is_moving = True
										b_dont_support = True
										break # OK move but dont support
								if b_dest_blocked:
									break
						if not b_dest_blocked or b_dest_is_moving:
							break

				else:
					if random.random() > wdconfig.c_rnd_fleet_army_wrong:
						# These while loops really make sure that this is a bad move
						while True:
							dest_name = random.choice(terr_id_tbl.keys())
							if dest_name not in pass_list:
								break
					else:
						other_pass_list = other_can_pass_tbl.get(sfrom, None)
						if other_pass_list == None:
							return False, e_move_type.none, -1
						for itry in range(20):
							if itry > 18:
								return False, e_move_type.none, -1
							dest_name = random.choice(other_pass_list)
							if dest_name not in pass_list:
								break

				dest_id = terr_id_tbl[dest_name]
				dstr = scountry + ' move ' + sutype + ' from ' + sfrom + ' to ' + dest_name
				orders_list.append([sutype, 'in', sfrom, 'move', 'to', dest_name])
				order_status = order_status._replace(toTerrID=dest_id)

				print(dstr)
				# The webdip engine does not guarantee that the order is valid, so we have to do this
				if dest_name not in pass_list:
					order_status = order_status._replace(status=False)
				orders_status_list.append(order_status)
				if order_status.status and not b_dont_support:
					move_details_tbl.append(nt_move_details(country=scountry, sutype=sutype,
															fromName=unit_data[1], toName=dest_name,
															bMove=True, iorder=len(orders_list)-1))
				return order_status.status, e_move_type.move, dest_id

			def create_support(move_details_tbl, imove):
				move_details = move_details_tbl[imove]
				pass_list = this_can_pass_tbl.get(sfrom, [])
				if move_details.toName in pass_list:
					if move_details.bMove:
						lorder = [sutype, 'in', sfrom, 'support', 'move', 'from',  move_details.fromName, 'to', move_details.toName]
						move_type = e_move_type.support
					else:
						lorder = [sutype, 'in', sfrom, 'support', 'hold', 'in', move_details.toName]
						move_type = e_move_type.support_hold
					orders_list.append(lorder)
					print(' '.join(lorder))
					dest_id, from_id = terr_id_tbl[move_details.toName], terr_id_tbl[move_details.fromName]
					order_status = nt_order_status(order_num=len(orders_list), status=True, unitID=unit_id[0],
												   fromTerrID=unit_terr_id, toTerrID=unit_terr_id,
												   iref=move_details.iorder)
					orders_status_list.append(order_status)
					move_details_tbl.append(nt_move_details(country=scountry, sutype=sutype,
															fromName=sfrom, toName=sfrom,
															bMove=False, iorder=len(orders_list)-1))
					return True, move_type, dest_id, from_id
				return False, e_move_type.none, -1, -1

			def create_convoy(move_details_tbl):
				if sutype == 'fleet' and terr_type_tbl[sfrom] == 'sea':
					convoy_src, convoy_dest, i_unit_convoyed = '', '', -1
					for i_other_unit, other_unit_data in enumerate(unit_list):
						if not l_units_avail[i_other_unit]:
							continue
						other_type, other_from = other_unit_data
						if other_type == 'fleet':
							continue
						fleet_reach = fleet_can_pass_tbl[sfrom]
						if other_from not in fleet_reach:
							continue
						convoy_dest = ''
						random.shuffle(fleet_reach)
						for one_reach in fleet_reach:
							if one_reach == other_from:
								continue
							if terr_type_tbl[one_reach] != 'coast':
								continue
							army_reach = army_can_pass_tbl[other_from]
							if one_reach in army_reach:
								continue
							convoy_dest = one_reach
							break
						if convoy_dest == '':
							continue
						convoy_src = other_from
						i_unit_convoyed = i_other_unit
						break
					if convoy_src != '' and convoy_dest != '':
						b_convoy_err = (random.random() < wdconfig.c_oracle_convoy_err_prob)
						if not b_convoy_err:
							move_details_tbl.append(nt_move_details(country=scountry, sutype='fleet',
																	fromName=sfrom, toName=sfrom,
																	bMove=False, iorder=len(orders_list)))
							lcorder = ['fleet', 'in', sfrom, 'convoy', 'army', 'in', convoy_src, 'to', convoy_dest]
							orders_list.append(lcorder)
							print(' '.join(lcorder))
						move_details_tbl.append(nt_move_details(country=scountry, sutype='army',
																fromName=convoy_src, toName=convoy_dest,
																bMove=True, iorder=len(orders_list)))
						lmorder  = ['army', 'in', convoy_src, 'convoy', 'move', 'to', convoy_dest]
						orders_list.append(lmorder)
						l_units_avail[i_unit_convoyed] = False
						sea_id, from_id, dest_id = terr_id_tbl[sfrom], terr_id_tbl[convoy_src], terr_id_tbl[convoy_dest]

						sql_get = sql_get_unit_id.substitute(gameID=str(gameID), terrID=str(from_id))
						cursor.execute(sql_get)
						other_unit_id = cursor.fetchone()

						if not b_convoy_err:
							order_status = nt_order_status(order_num=len(orders_list), status=True, unitID=unit_id[0],
														   fromTerrID=sea_id, toTerrID=sea_id,
														   iref=move_details_tbl[-1].iorder)
							orders_status_list.append(order_status)

						order_status = nt_order_status(order_num=len(orders_list), status=True, unitID=other_unit_id[0],
													   fromTerrID=from_id, toTerrID=dest_id,
													   iref=-1)
						orders_status_list.append(order_status)

						return True, e_move_type.convoy, b_convoy_err, sea_id, from_id, dest_id, other_unit_id[0]
				return False, e_move_type.none, False, -1, -1, -1, -1

			def create_hold(move_details):
				order_status = nt_order_status(order_num=len(orders_list), status=True, unitID=unit_id[0],
											   fromTerrID=unit_terr_id, toTerrID=unit_terr_id,
											   iref=-1)
				orders_status_list.append(order_status)
				lorder = [sutype, 'in', sfrom, 'hold']
				orders_list.append(lorder)
				print(' '.join(lorder))
				move_details_tbl.append(nt_move_details(country=scountry, sutype=sutype,
														fromName=sfrom, toName=sfrom,
														bMove=False, iorder=len(orders_list)-1))
				return True, e_move_type.hold

			b_order_created = False
			b_order_created, move_type, b_convoy_err, sea_id, from_id, dest_id, convoyed_unit_id = create_convoy(move_details_tbl)

			if not b_order_created and random.random() < wdconfig.c_oracle_hold_prob:
				b_order_created, move_type = create_hold(move_details_tbl)

			if not b_order_created and len(move_details_tbl) > 0 and random.random() < wdconfig.c_oracle_support_prob:
				for imove, one_move in enumerate(move_details_tbl):
					b_order_created, move_type, dest_id, from_id = create_support(move_details_tbl, imove)
					if b_order_created:
						break

			if not b_order_created:
				b_order_created, move_type, dest_id = create_move(move_details_tbl)

			if b_order_created:
				l_units_avail[iunit] = False
				if move_type == e_move_type.convoy:
					if b_convoy_err:
						l_units_avail[iunit] = True
					else:
						sql_order = sql_convoy_order.substitute(toTerrID=str(dest_id), unitID=str(unit_id[0]),
																 gameID=str(gameID), fromTerrID=str(from_id))
					# print (sql_order)
						cursor.execute(sql_order)
					sql_order = sql_move_order.substitute(toTerrID=str(dest_id), unitID=str(convoyed_unit_id),
														  gameID=str(gameID), bConvoyed='Yes')
				elif move_type == e_move_type.hold:
					sql_order = sql_hold_order.substitute(unitID=str(unit_id[0]), gameID=str(gameID))
				elif move_type == e_move_type.support:
					sql_order = sql_support_order.substitute(toTerrID=str(dest_id), unitID=str(unit_id[0]),
															 gameID=str(gameID), fromTerrID=str(from_id))
				elif move_type == e_move_type.support_hold:
					sql_order = sql_support_hold_order.substitute(	toTerrID=str(dest_id), unitID=str(unit_id[0]),
																	gameID=str(gameID))
				elif move_type == e_move_type.move:
					sql_order = sql_move_order.substitute(toTerrID=str(dest_id), unitID=str(unit_id[0]),
														  gameID=str(gameID), bConvoyed='No')
				# print (sql_order)
				cursor.execute(sql_order)

		# end oreders for one unit
		sql_order = sql_complete_order.substitute(gameID=str(gameID), countryID=str(icountry), timeLoggedIn=str(int(time.time())))
		# print(sql_order)
		cursor.execute(sql_order)

	db.commit()

def create_results_db(orders_db, orders_status_list):
	if not orders_db:
		return []

	results_db = []

	for iorder, order in enumerate(orders_db):
		if orders_status_list[iorder].status:
			results_db.append(order+[els.make_obj_el_from_str('succeeded')])
		else:
			results_db.append(order+[els.make_obj_el_from_str('failed')])

	return results_db

def play_turn(	all_dicts, db_len_grps, db_cont_mgr, i_active_cont,  el_set_arr, sess, learn_vars,
				db, cursor, gname, l_humaans, country_names_tbl,
				terr_id_tbl, supply_tbl, terr_type_tbl, army_can_pass_tbl, fleet_can_pass_tbl,
				init_db, old_orders_status_list, old_status_db, old_orders_db, old_orders_list):
	sqlOrderComplete = string.Template(
		'UPDATE webdiplomacy.wD_Members SET timeLoggedIn = ${timeLoggedIn}, missedPhases = 0, orderStatus = \'Saved,Completed,Ready\' '
		'WHERE gameID = ${gameID} AND countryID = ${countryID};')
	sql_get_unit_id = string.Template(
		'SELECT id FROM webdiplomacy.wD_Units where gameID = ${gameID} and terrID = ${terrID};')
	sql_move_order = string.Template('UPDATE webdiplomacy.wD_Orders SET type=\'Move\', '
										 'toTerrID = ${toTerrID}, viaConvoy = \'${bConvoyed}\' WHERE unitID = ${unitID} AND gameID = ${gameID} ;')
	sql_support_order = string.Template('UPDATE webdiplomacy.wD_Orders SET type=\'Support move\', '
										 'toTerrID = ${toTerrID}, fromTerrID = ${fromTerrID} WHERE unitID = ${unitID} AND gameID = ${gameID} ;')
	sql_support_hold_order = string.Template('UPDATE webdiplomacy.wD_Orders SET type=\'Support hold\', '
										 'toTerrID = ${toTerrID} WHERE unitID = ${unitID} AND gameID = ${gameID} ;')
	sql_convoy_order = string.Template('UPDATE webdiplomacy.wD_Orders SET type=\'Convoy\', '
										 'toTerrID = ${toTerrID}, fromTerrID = ${fromTerrID} WHERE unitID = ${unitID} AND gameID = ${gameID} ;')
	sql_hold_order = string.Template('UPDATE webdiplomacy.wD_Orders SET type=\'Hold\' '
										 'WHERE unitID = ${unitID} AND gameID = ${gameID} ;')
	l_sql_action_orders = [sql_move_order, sql_support_order, sql_support_hold_order, sql_convoy_order, sql_hold_order]

	b_game_finished, b_orders_valid, b_reset_orders, b_stuck = False, False, False, False
	gameID, game_phase, game_turn = get_game_id(cursor, gname)
	if gameID == None:
		CreateGame(db, cursor, gname)
		gameID, game_phase, game_turn = get_game_id(cursor, gname)
		if gameID == None:
			print('Failed to create game. Exiting!')
			exit(1)
		AddPlayers(db, cursor, gameID)
		print('New game created.')
		b_reset_orders = True
		return gameID, b_game_finished, b_orders_valid, b_reset_orders, b_stuck, None, None, None, None

	try:
		if play_turn.turn == game_turn and play_turn.phase == game_phase:
			print('Game stuck!')
			b_game_finished, b_game_finished, b_stuck = False, False, True
			return gameID, b_game_finished, b_orders_valid, b_reset_orders, b_stuck, None, None, None, None
	except AttributeError:
		print('Naughty! You used a variable before we initialized it!')

	play_turn.turn, play_turn.phase = game_turn, game_phase
	print('Playing turn for game id:', gameID, 'phase:', game_phase, 'turn:', game_turn)


	# gameID = 28

	# sqlDone = 'UPDATE webdiplomacy.wD_Members SET orderStatus = \'Saved,Completed,Ready\' WHERE id = 20 ;'
	# cursor.execute(sqlDone)
	# db.commit()

	# CreateGame()
	# AddPlayers(gameID)

	statement_list = []

	terr_owns_tbl = OwnsTerrTbl(cursor, gameID, country_names_tbl, statement_list)
	unit_owns_tbl, updated_orders_status_list = OwnsUnitsTbl(cursor, gameID, country_names_tbl, statement_list, old_orders_status_list)
	for iorder, order_status in enumerate(updated_orders_status_list):
		print(' '.join(old_orders_list[iorder]), 'succeeded' if order_status.status else 'failed')

	results_db = create_results_db(old_orders_db, updated_orders_status_list)
	if results_db != None and len(results_db) > 0:
		b_reset_orders = True
		if wdconfig.c_b_save_orders:
			b_keep_working = wdlearn.create_order_freq_tbl(old_orders_list, updated_orders_status_list)
		if wdconfig.c_b_collect_cont_stats:
			b_keep_working = wdlearn.collect_cont_stats(init_db, old_status_db, old_orders_db, results_db,
										 all_dicts, db_cont_mgr)
		if wdconfig.c_b_add_to_db_len_grps:
			b_keep_working = wdlearn.learn_orders_success(init_db, old_status_db, old_orders_db, results_db,
										 all_dicts, db_len_grps, db_cont_mgr, i_active_cont,   el_set_arr, sess, learn_vars)
		if not b_keep_working:
			b_game_finished, b_orders_valid, = False, False
			return gameID, b_game_finished, b_orders_valid, b_reset_orders, b_stuck, [], [], [], []

	status_db = els.convert_list_to_phrases(statement_list)

	l_humaan_countries = get_humaan_countries(cursor, gameID, l_humaans)

	orders_list = []
	# Remebser how to do the following hack
	# orders_status_list[:] = [] # a hack so that the reference itself has the contents removed
	orders_status_list = []
	if game_phase == 'Builds':
		print('Creating build orders.')
		create_build_orders(db, cursor, gameID, l_humaan_countries, country_names_tbl, terr_id_tbl, supply_tbl,
							unit_owns_tbl, terr_owns_tbl, sqlOrderComplete)
	elif game_phase == 'Retreats':
		print('Creating retreat orders.')
		create_retreat_orders(db, cursor, gameID, l_humaan_countries, country_names_tbl, terr_id_tbl, supply_tbl,
							  unit_owns_tbl, sqlOrderComplete)
	elif game_phase == 'Diplomacy':
		b_orders_valid = True
		if wdconfig.c_b_play_from_saved:
			create_move_orders2(db, cursor, gameID, l_humaan_countries, sqlOrderComplete,
								sql_get_unit_id, l_sql_action_orders,
								unit_owns_tbl, terr_id_tbl,
								country_names_tbl, army_can_pass_tbl, fleet_can_pass_tbl,
								orders_list, orders_status_list,
								init_db, status_db, db_cont_mgr, all_dicts,
								terr_owns_tbl, supply_tbl)
		else:
			create_oracle_move_orders(	db, cursor, gameID, l_humaan_countries, sqlOrderComplete,
										sql_get_unit_id, l_sql_action_orders,
										unit_owns_tbl, terr_id_tbl, terr_type_tbl,
										country_names_tbl, army_can_pass_tbl, fleet_can_pass_tbl,
										orders_list, orders_status_list)

	elif game_phase == 'Pre-game':
		complete_game_init(db, cursor, gameID, l_humaan_countries)
	elif game_phase == 'Finished':
		b_game_finished, b_orders_valid, b_reset_orders = True, False, True
		return gameID, b_game_finished, b_orders_valid, b_reset_orders, b_stuck, None, None, None, None
	else:
		print('unknown phase:', game_phase)
		b_game_finished, b_orders_valid, b_reset_orders = False, False, True
		return gameID, b_game_finished, b_orders_valid, b_reset_orders, b_stuck, None, None, None, None

	orders_db = els.convert_list_to_phrases(orders_list)


	time.sleep(0.2)

	b_game_finished = False
	return gameID, b_game_finished, b_orders_valid, b_reset_orders, b_stuck, status_db, orders_db, orders_list, orders_status_list

def init(cursor, country_names_tbl):
	terr_id_tbl = dict()
	army_can_pass_tbl = dict()
	fleet_can_pass_tbl = dict()

	statement_list = []
	PassInsert(cursor, 'armysPass', 'army', army_can_pass_tbl, statement_list)
	PassInsert(cursor, 'fleetsPass', 'fleet', fleet_can_pass_tbl, statement_list)

	# def add_pass_statements(stype, tbl):
	# 	for src_name, pass_list in tbl.iteritems():
	# 		for dst_name in pass_list:
	# 			statement_list.append([stype, 'can', 'pass', 'from', src_name, 'to', dst_name])
	#
	# if wdconfig.c_include_pass_statements:
	# 	add_pass_statements('army', army_can_pass_tbl)
	# 	add_pass_statements('fleet', fleet_can_pass_tbl)

	supply_tbl = create_supply_tbl(cursor, country_names_tbl, statement_list)

	terr_type_tbl = create_terr_type_tbl(cursor, statement_list)

	create_terr_id_tbl(cursor, terr_id_tbl, country_names_tbl, statement_list)



	return terr_id_tbl, army_can_pass_tbl, fleet_can_pass_tbl, supply_tbl, terr_type_tbl, statement_list

def create_dict_files(terr_names_fn):
	# terr_names_fn = 'terrnames.txt'
	terr_names_fh = open(terr_names_fn, 'wb')
	# terr_names_csvr = csv.writer(terr_names_fh, delimiter='', quoting=csv.QUOTE_NONE, quotechar='',escapechar='\\')

	with closing(MySQLdb.connect("localhost","webdiplomacy","mypassword123","webdiplomacy" )) as db:
		# prepare a cursor object using cursor() method
		with closing(db.cursor()) as cursor:

			sqlTerrNames = 'SELECT LOWER(name) FROM webdiplomacy.wD_Territories where mapID = 1;'
			cursor.execute(sqlTerrNames)
			results = cursor.fetchall()
			for row in results:
				# terr_names_csvr.writerow(row[0])
				terr_names_fh.write(row[0]+'\n')
			return

def play(gameID, all_dicts, db_len_grps, db_cont_mgr, i_active_cont, el_set_arr, sess, learn_vars, b_create_dict_file=False):
	glv_dict, _, _ = all_dicts

	b_can_continue = True

	# if gameID == -1:
	# 	gname = 't' + str(int(time.time()))[-6:]

	country_names_tbl = ['neutral', 'england', 'france', 'italy', 'germany', 'austria', 'turkey', 'russia']


	# Open database connection
	with closing(MySQLdb.connect("localhost","webdiplomacy","mypassword123","webdiplomacy" )) as db:
		# prepare a cursor object using cursor() method
		with closing(db.cursor()) as cursor:

			if wdconfig.c_admin_action != None:
				if wdconfig.c_admin_action == 'DeleteGames':
					wd_admin.DeleteFinishedGames(db, cursor)
				b_can_continue= False
				return -1, b_can_continue

			terr_id_tbl, army_can_pass_tbl, fleet_can_pass_tbl, supply_tbl, terr_type_tbl, statement_list = \
				init(cursor, country_names_tbl)
			init_db = els.convert_list_to_phrases(statement_list)

			# if gameID != -1:
			l_humaans = []
			if wdconfig.c_b_play_human:
				while True:
					gameID, gname, l_humaans = find_human_game(db, cursor)
					if gameID != -1:
						break
					time.sleep(20.0)
			else:
				b_not_found = True
				while b_not_found:
					gname, b_game_over = get_game_name(cursor, gameID)
					if gname == None:
						gname = 't' + str(int(time.time()))[-6:]
						b_not_found = False
					elif b_game_over:
						gameID += 1
						continue
					else:
						b_not_found = False

			orders_status_list = []
			old_status_db, old_orders_db, old_orders_list, old_orders_status_list = [], [], [], []
			b_keep_working = True
			num_stuck = 0
			for iturn in range(wdconfig.c_num_turns_per_play):
				if not b_keep_working:
					# return gameID, b_can_continue
					break

				if gameID != -1:
					try:
						process_url = string.Template("http://localhost/board.php?gameID=${gameID}&process=Yes")
						time.sleep(0.2)
						response = urllib2.urlopen(process_url.substitute(gameID=gameID))
					except urllib2.HTTPError as err:
						print('HTTP Exception: ', err)
					except Exception as e:
						print(e)

					time.sleep(0.2)
				# else:
				# 	print('Explain why!')

				if wdconfig.c_b_play_human:
					wait_to_play(db, cursor, gameID, l_humaans)

				gameID, b_finished, b_orders_valid, b_reset_orders, b_stuck, status_db, orders_db, \
				orders_list, orders_status_list = \
					play_turn(	all_dicts, db_len_grps, db_cont_mgr, i_active_cont,  el_set_arr, sess, learn_vars,
								db, cursor, gname, l_humaans, country_names_tbl,
								terr_id_tbl, supply_tbl, terr_type_tbl, army_can_pass_tbl,
								fleet_can_pass_tbl, init_db, old_orders_status_list,
								old_status_db=old_status_db, old_orders_db=old_orders_db,
								old_orders_list=old_orders_list)
				if b_stuck:
					db.commit() # just in case
					time.sleep(1.0)
					num_stuck += 1
					if num_stuck > 100:
						b_keep_working = False

				if b_reset_orders:
					old_status_db, old_orders_db, old_orders_list, old_orders_status_list = [], [], [], []

				if b_orders_valid:
					old_status_db, old_orders_db, old_orders_list, old_orders_status_list \
						= status_db, orders_db, orders_list, orders_status_list
					num_stuck = 0

				if b_finished:
					print('Game Over!')
					gameID = -1
					# return -1, b_can_continue
					break
				elif status_db == []: # means turn finished early
					b_keep_working = False
				print('Next turn for game id:', gameID)

	if wdconfig.c_b_add_to_db_len_grps:
		b_can_continue = wdlearn.create_new_conts(glv_dict, db_cont_mgr, db_len_grps, i_active_cont)
		wdlearn.save_db_status(db_len_grps, db_cont_mgr)
		# if not b_keep_working:
		# 	break

	return gameID, b_can_continue

glv_file_list = ['wd_terrname', 'wd_order']
cap_first_arr = [False, False]
def_article_arr = [False, False]
cascade_els_arr = [True, False]

# def load_el_vecs():
# 	cwd = os.getcwd()
# 	os.chdir('..')
# 	os.chdir(cwd)
# 	cwdtest = os.getcwd()
# 	return

# load_el_vecs()

def logic_init():
	full_glv_list = [fname+'s.glv' for fname in glv_file_list]
	# cwd = os.getcwd()
	# os.chdir('..')
	glv_dict, def_article_dict, cascade_dict, _ = \
		els.init_glv(full_glv_list, cap_first_arr, def_article_arr, cascade_els_arr)
	# os.chdir(cwd)
	# cwdtest = os.getcwd()
	return [glv_dict, def_article_dict, cascade_dict]

def do_wd(gameID, all_dicts, el_set_arr, learn_vars):
	db_cont_mgr = None
	db_len_grps, i_active_cont = [], -1
	if wdconfig.c_b_load_cont_stats:
		db_cont_mgr = wdlearn.init_cont_stats_from_file()
	# if db_cont_mgr:
	# 	wdlearn.compare_conts_learn(db_cont_mgr)
	if wdconfig.c_b_load_cont_mgr:
		assert db_cont_mgr == None, 'If load cont mgr is selected, it must not be loaded from cont stats'
		db_cont_mgr = wdlearn.load_cont_mgr()

	if wdconfig.c_b_add_to_db_len_grps:
		assert wdconfig.c_b_load_cont_mgr and not wdconfig.c_b_load_cont_stats, 'Can only learn more db len grps if loaded cont mgr directly. Not from cont sats'
		db_len_grps, i_active_cont = wdlearn.sel_cont_and_len_grps(db_cont_mgr)
		if i_active_cont < 0:
			return -1, False
		db_cont_mgr.load_perm_dict(wdconfig.perm_fnt)
		max_eid = db_cont_mgr.apply_perm_dict_data(db_len_grps, i_active_cont)
		db_cont_mgr.load_W_dict(wdconfig.W_fnt)
		db_cont_mgr.apply_W_dict_data(db_len_grps, i_active_cont)
		learn_vars[0] = max_eid
	elif wdconfig.c_b_init_cont_stats_from_cont_mgr:
		assert wdconfig.c_b_load_cont_mgr, 'You cannot initialize cont stats without c_b_load_cont_mgr'
		db_len_grps, i_active_cont = [], -1
		if db_cont_mgr.get_cont_stats_mgr() == None:
			target_rule_list = wdlearn.add_success_to_targets(wdconfig.c_target_gens)

			db_cont_mgr.init_cont_stats_mgr(wdconfig.c_cont_stats_init_thresh, target_rule_list,
											all_dicts[0], # glv_dict - TBD avoid magic position
											wdconfig.c_cont_stats_init_exclude_irrelevant)

		# db_len_grps, blocked_len_grps = wdlearn.load_len_grps()
	sess = dmlearn.init_templ_learn()
	gameID, b_can_continue = play(gameID, all_dicts, db_len_grps, db_cont_mgr, i_active_cont, el_set_arr, sess, learn_vars)
	sess.close()
	dmlearn.learn_reset()

	if wdconfig.c_b_learn_conts:
		wdlearn.compare_conts_learn(db_cont_mgr)
	if wdconfig.c_b_cont_stats_save:
		wdlearn.cont_stats_save(db_cont_mgr, wdconfig.c_cont_stats_fnt)

	return gameID, b_can_continue

def main():
	# create_dict_files(glv_file_list[0] + 's.txt')
	# return
	# embed.create_ext(glv_file_list)
	# return

	gameID = 1187 # Set to -1 to restart
	all_dicts = logic_init()
	# db_len_grps = []
	el_set_arr = []
	event_step_id = -1
	learn_vars = [event_step_id]
	clrecgrp.cl_templ_grp.glv_dict = all_dicts[0]
	clrecgrp.cl_gens_grp.glv_len = clrecgrp.cl_templ_grp.glv_len = len(all_dicts[0][wdconfig.c_sample_el])
	wdlearn.set_target_gens()
	# clrecgrp.cl_templ_grp.c_target_gens = wdconfig.c_target_gens
	wdlearn.init_learn()
	# sess, saver_dict, saver = dmlearn.init_templ_learn()
	for iplay in range(wdconfig.c_num_plays):
		gameID, b_can_continue = do_wd(gameID, all_dicts, el_set_arr, learn_vars)
		if not b_can_continue:
			print('Program will not continue due to no conts being available or administrative action taken.')
			return

if __name__ == "__main__":
    main()
