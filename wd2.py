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
import embed
import els
import dmlearn
import clrecgrp
import wdlearn


c_rnd_bad_move = 0.6
c_rnd_fleet_army_wrong = 0.4

# response = urllib2.urlopen("http://localhost/gamemaster.php?gameMasterSecret=")

def get_game_name(cursor, gameID):
	sqlGetGame = string.Template('select id, name from webdiplomacy.wD_Games where id = \'${gameID}\';')
	sql = sqlGetGame.substitute(gameID=gameID)
	cursor.execute(sql)
	result = cursor.fetchone()
	if result == None or result[0] != gameID:
		return None
	return result[1]

def get_game_id(cursor, gname):
	sqlGetGame = string.Template('select id, phase from webdiplomacy.wD_Games where name = \'${gname}\';')
	sql = sqlGetGame.substitute(gname=gname)
	cursor.execute(sql)
	result = cursor.fetchone()
	if result == None:
		return None, None
	return result[0], result[1]


def get_phase(cursor, gameID):
	sqlGetPhase = string.Template('SELECT phase FROM webdiplomacy.wD_Games WHERE id = ${gameID}')

	sql = sqlGetPhase.substitute(gameID=gameID)
	cursor.execute(sql)
	result = cursor.fetchone()
	return result[0]


def AddPlayers(db, cursor, gameID):
	sqlAddPlayer = string.Template(
		'INSERT INTO webdiplomacy.wD_Members SET userID = ${userid}, countryID = 0, bet = 10, '
		'timeLoggedIn = 1516721310, gameID = ${gameID}, orderStatus = \'None,Completed,Ready\';')

	for i in range(7):
		sql = sqlAddPlayer.substitute(gameID=gameID, userid=str(i + 6))
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

	new_orders_status_list = []
	for order_status in orders_status_list:
		uID, expected_terrID, status = order_status.unitID, order_status.toTerrID, order_status.status
		if expected_terrID == None or not status:
			new_orders_status_list.append(order_status)
			continue
		terrID = unit_dict.get(uID, None)
		if terrID == None or terrID != expected_terrID:
			new_orders_status_list.append(order_status._replace(status=False))
			continue
		new_orders_status_list.append(order_status)

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
		statement_list.append([country, 'owns', row[1]])
		if row[2] == None:
			statement_list.append([row[1], 'is', 'unoccupied'])
		else:
			statement_list.append([row[1], 'is', 'occupied'])

		# print(db_str)
	return owns_tbl


def PassInsert(cursor, spass, type, can_pass_tbl):
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
		statement_list.append([row[1], 'is', 'a', row[2]])
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
		statement_list.append([row[2], 'is', 'a', 'supply'])
		# print(db_str)
	return supply_tbl


def create_retreat_orders(db, cursor, gameID, country_names_tbl, terr_id_tbl, supply_tbl, unit_owns_tbl, sqlOrderComplete):
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
		sql = sqlOrderComplete.substitute(gameID=str(gameID), countryID=str(icountry), timeLoggedIn=str(int(time.time())))
		print(sql)
		cursor.execute(sql)

	db.commit()


def create_build_orders(db, cursor, gameID, country_names_tbl, terr_id_tbl, supply_tbl,
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
		scountry = country_names_tbl[row[0]]
		country_order_ids.add(row[0])
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
			dstr = scountry + ' ' + scmd + ' in ' + build_loc[1]
		else:
			owns_list = unit_owns_tbl[scountry]
			destroy_data = random.choice(owns_list)
			dest_id = terr_id_tbl[destroy_data[1]]
			sql = sqlBuildOrder.substitute(scmd='Destroy', toTerrID=dest_id, id=row[2] )
			dstr = scountry + ' destroy in ' + destroy_data[1]
		print(dstr)
		print(sql)
		cursor.execute(sql)

	for icountry in country_order_ids:
		sql = sqlOrderComplete.substitute(gameID=str(gameID), countryID=str(icountry), timeLoggedIn=str(int(time.time())))
		print(sql)
		cursor.execute(sql)

	db.commit()



def create_move_orders(db, cursor, gameID, sql_complete_order,
					   unit_owns_tbl, terr_id_tbl,
					   country_names_tbl, army_can_pass_tbl, fleet_can_pass_tbl,
					   orders_list, orders_status_list):
	sql_get_unit_id = string.Template(
		'SELECT id FROM webdiplomacy.wD_Units where gameID = ${gameID} and terrID = ${terrID};')
	sql_make_order = string.Template('UPDATE webdiplomacy.wD_Orders SET type=\'Move\', '
										 'toTerrID = ${toTerrID} WHERE unitID = ${unitID} AND gameID = ${gameID} ;')

	for icountry in range(1, len(country_names_tbl)):
		scountry = country_names_tbl[icountry]
		print('Orders for ', scountry)
		unit_list = unit_owns_tbl.get(scountry, None)
		if unit_list == None:
			continue
		for unit_data in unit_list:
			unit_terr_id = terr_id_tbl[unit_data[1]]
			# sqlIDbyTerr = string.Template('SELECT id FROM webdiplomacy.wD_Units where gameID = ${gameID} and terrID = ${terrID};')
			sql_get = sql_get_unit_id.substitute(gameID=str(gameID), terrID=str(unit_terr_id))
			cursor.execute(sql_get)
			unit_id = cursor.fetchone()

			nt_order_status = collections.namedtuple('nt_order_status', 'order_num, status, unitID, fromTerrID, toTerrID')

			def create_move(sutype, this_can_pass_table, other_can_pass_table):
				order_status = nt_order_status(order_num=len(orders_list), status=True, unitID=unit_id[0],
											   fromTerrID=unit_terr_id, toTerrID=None)
				pass_list = this_can_pass_table.get(unit_data[1], None)
				if pass_list == None:
					print(sutype, 'stuck where it is not supposed to be.')
					return False, None
				if random.random() > c_rnd_bad_move:
					dest_name = random.choice(pass_list)
				else:
					if random.random() > c_rnd_fleet_army_wrong:
						dest_name = random.choice(terr_id_tbl.keys())
					else:
						other_pass_list = other_can_pass_table.get(unit_data[1], None)
						if other_pass_list == None:
							return False, None
						dest_name = random.choice(other_pass_list)
				dest_id = terr_id_tbl[dest_name]
				dstr = scountry + ' move ' + sutype + ' from ' + unit_data[1] + ' to ' + dest_name
				orders_list.append([sutype, 'in', unit_data[1], 'move', 'to', dest_name])
				order_status = order_status._replace(toTerrID=dest_id)
				print(dstr)
				# The webdip engine does not guarantee that the order is valid, so we have to do this
				if dest_name not in pass_list:
					order_status = order_status._replace(status=False)
				orders_status_list.append(order_status)
				return order_status.status, dest_id

			if unit_data[0] == 'army':
				b_order_created, dest_id = create_move('army', army_can_pass_tbl, fleet_can_pass_tbl)
			else:
				b_order_created, dest_id = create_move('fleet', fleet_can_pass_tbl, army_can_pass_tbl)

			if b_order_created:
				sql_order = sql_make_order.substitute(toTerrID=str(dest_id), unitID=str(unit_id[0]), gameID=str(gameID))
				print (sql_order)
				cursor.execute(sql_order)

		sql_order = sql_complete_order.substitute(gameID=str(gameID), countryID=str(icountry), timeLoggedIn=str(int(time.time())))
		print(sql_order)
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

def play_turn(	all_dicts, db_len_grps, el_set_arr, sess, learn_vars, db, cursor, gname, country_names_tbl,
				terr_id_tbl, supply_tbl, army_can_pass_tbl, fleet_can_pass_tbl,
				init_db, orders_status_list, old_status_db, old_orders_db):
	sqlOrderComplete = string.Template(
		'UPDATE webdiplomacy.wD_Members SET timeLoggedIn = ${timeLoggedIn}, missedPhases = 0, orderStatus = \'Saved,Completed,Ready\' '
		'WHERE gameID = ${gameID} AND countryID = ${countryID};')

	gameID, game_phase = get_game_id(cursor, gname)
	if gameID == None:
		CreateGame(db, cursor, gname)
		gameID, game_phase = get_game_id(cursor, gname)
		if gameID == None:
			print('Failed to create game. Exiting!')
			exit(1)
		AddPlayers(db, cursor, gameID)
		return gameID, False, None, None

	# gameID = 28

	# sqlDone = 'UPDATE webdiplomacy.wD_Members SET orderStatus = \'Saved,Completed,Ready\' WHERE id = 20 ;'
	# cursor.execute(sqlDone)
	# db.commit()

	# CreateGame()
	# AddPlayers(gameID)

	statement_list = []

	terr_owns_tbl = OwnsTerrTbl(cursor, gameID, country_names_tbl, statement_list)
	unit_owns_tbl, new_orders_status_list = OwnsUnitsTbl(cursor, gameID, country_names_tbl, statement_list, orders_status_list)
	results_db = create_results_db(old_orders_db, new_orders_status_list)
	if results_db != None and len(results_db) > 0:
		wdlearn.learn_orders_success(init_db, old_status_db, old_orders_db, results_db,
									 all_dicts, db_len_grps, el_set_arr, sess, learn_vars)

	status_db = els.convert_list_to_phrases(statement_list)

	orders_list = []
	orders_status_list[:] = [] # a hack so that the reference itself has the contents removed
	if game_phase == 'Builds':
		print('Creating build orders.')
		create_build_orders(db, cursor, gameID, country_names_tbl, terr_id_tbl, supply_tbl,
							unit_owns_tbl, terr_owns_tbl, sqlOrderComplete)
	elif game_phase == 'Retreats':
		print('Creating retreat orders.')
		create_retreat_orders(db, cursor, gameID, country_names_tbl, terr_id_tbl, supply_tbl,
							  unit_owns_tbl, sqlOrderComplete)
	elif game_phase == 'Diplomacy':
		create_move_orders(db, cursor, gameID, sqlOrderComplete,
						   unit_owns_tbl, terr_id_tbl,
						   country_names_tbl, army_can_pass_tbl, fleet_can_pass_tbl,
						   orders_list, orders_status_list)
	elif game_phase == 'Finished':
		return gameID, True, None, None
	else:
		return gameID, False, None, None

	orders_db = els.convert_list_to_phrases(orders_list)


	time.sleep(0.2)

	return gameID, False, status_db, orders_db

def init(cursor, country_names_tbl):
	terr_id_tbl = dict()
	army_can_pass_tbl = dict()
	fleet_can_pass_tbl = dict()

	PassInsert(cursor, 'armysPass', 'army', army_can_pass_tbl)
	PassInsert(cursor, 'fleetsPass', 'fleet', fleet_can_pass_tbl)

	statement_list = []
	def add_pass_statements(stype, tbl):
		for src_name, pass_list in tbl.iteritems():
			for dst_name in pass_list:
				statement_list.append([stype, 'can', 'pass', 'from', src_name, 'to', dst_name])

	add_pass_statements('army', army_can_pass_tbl)
	add_pass_statements('fleet', fleet_can_pass_tbl)

	supply_tbl = create_supply_tbl(cursor, country_names_tbl, statement_list)

	create_terr_id_tbl(cursor, terr_id_tbl, country_names_tbl, statement_list)



	return terr_id_tbl, army_can_pass_tbl, fleet_can_pass_tbl, supply_tbl, statement_list

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

def play(gameID, all_dicts, db_len_grps, el_set_arr, sess, learn_vars, b_create_dict_file=False):

	if gameID == -1:
		gname = 't' + str(int(time.time()))[-6:]

	country_names_tbl = ['neutral', 'england', 'france', 'italy', 'germany', 'austria', 'turkey', 'russia']


	# Open database connection
	with closing(MySQLdb.connect("localhost","webdiplomacy","mypassword123","webdiplomacy" )) as db:
		# prepare a cursor object using cursor() method
		with closing(db.cursor()) as cursor:

			terr_id_tbl, army_can_pass_tbl, fleet_can_pass_tbl, supply_tbl, statement_list = \
				init(cursor, country_names_tbl)
			init_db = els.convert_list_to_phrases(statement_list)

			if gameID != -1:
				gname = get_game_name(cursor, gameID)

			orders_status_list = []
			status_db, orders_db = [], []
			for iturn in range(300):
				if iturn > 0:
					gameID, b_finished, status_db, orders_db = \
						play_turn(	all_dicts, db_len_grps, el_set_arr, sess, learn_vars,
									db, cursor, gname, country_names_tbl,
									terr_id_tbl, supply_tbl, army_can_pass_tbl,
									fleet_can_pass_tbl, init_db, orders_status_list,
									old_status_db=status_db, old_orders_db=orders_db)
					if b_finished:
						print('Game Over!')
						return
					print('Next turn for game id:', gameID)
				if gameID != -1:
					process_url = string.Template("http://localhost/board.php?gameID=${gameID}&process=Yes")
					try:
						response = urllib2.urlopen(process_url.substitute(gameID=gameID))
					except urllib2.HTTPError as err:
						print('HTTP Exception: ', err)
					time.sleep(0.2)

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
	glv_dict, def_article_dict, cascade_dict = \
		els.init_glv(full_glv_list, cap_first_arr, def_article_arr, cascade_els_arr)
	# os.chdir(cwd)
	# cwdtest = os.getcwd()
	return [glv_dict, def_article_dict, cascade_dict]

def main():
	# create_dict_files(glv_file_list[0] + 's.txt')
	# return
	# embed.create_ext(glv_file_list)
	# return

	gameID = 34 # Set to -1 to restart
	all_dicts = logic_init()
	# db_len_grps = []
	el_set_arr = []
	event_step_id = -1
	learn_vars = [event_step_id]
	clrecgrp.cl_templ_grp.glv_dict = all_dicts[0]
	clrecgrp.cl_templ_grp.glv_len = len(all_dicts[0]['army'])
	# sess, saver_dict, saver = dmlearn.init_templ_learn()
	db_len_grps = wdlearn.load_len_grps()
	sess = dmlearn.init_templ_learn()
	play(gameID, all_dicts, db_len_grps, el_set_arr, sess, learn_vars)
	sess.close()

if __name__ == "__main__":
    main()
