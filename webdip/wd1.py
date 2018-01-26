from __future__ import print_function
from contextlib import closing
import MySQLdb
import urllib2
import string
import random
import time
import datetime

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


def OwnsUnitsTbl(cursor, gameID, country_names_tbl):
	sqlUnitsOwned = string.Template('SELECT u.countryID, u.type, t.name '
									'FROM webdiplomacy.wD_Units AS u '
									'INNER JOIN webdiplomacy.wD_Territories AS t ON ('
									'u.terrID = t.id AND t.mapID = 1 '
									') '
									'WHERE u.gameID = ${gameID};')
	owns_tbl = dict()
	sql = sqlUnitsOwned.substitute(gameID=str(gameID))
	cursor.execute(sql)
	results = cursor.fetchall()
	for row in results:
		db_str = country_names_tbl[row[0]] + ' owns ' + row[1]
		country = country_names_tbl[row[0]]
		owns_list = owns_tbl.get(country, None)
		if owns_list == None:
			owns_tbl[country] = []
		owns_tbl[country].append([row[1], row[2]])
		db_str += ' at ' + row[2]

		# print(db_str)
	return owns_tbl

def OwnsTerrTbl(cursor, gameID, country_names_tbl):
	sqlTerrOwned = string.Template('SELECT ts.countryID, t.name, ts.occupyingUnitID '
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

		# print(db_str)
	return owns_tbl


def PassInsert(cursor, spass, type, can_pass_tbl):
	sqlPass = string.Template('SELECT tf.name, tt.name '
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


def create_terr_id_tbl(cursor, terr_id_tbl):
	sqlGetTerrID = 'SELECT id, name FROM webdiplomacy.wD_Territories WHERE mapID = 1;'
	cursor.execute(sqlGetTerrID)
	results = cursor.fetchall()
	for row in results:
		terr_id_tbl[row[1]] = row[0]


def create_supply_tbl(cursor, country_names_tbl):
	sqlForSupplyTbl = ('SELECT countryID, type, name FROM webdiplomacy.wD_Territories '
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
		# print(db_str)
	return supply_tbl


def create_retreat_orders(db, cursor, gameID, country_names_tbl, terr_id_tbl, supply_tbl, unit_owns_tbl, sqlOrderComplete):
	sqlGetBuildOrders = string.Template('SELECT countryID, type, id from webdiplomacy.wD_Orders where gameID = ${gameID};')
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
	sqlGetBuildOrders = string.Template('SELECT countryID, type, id from webdiplomacy.wD_Orders where gameID = ${gameID};')
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
		if row[1] == 'Build Army':
			build_opts_list = build_opts.get(scountry, None)
			if build_opts_list == None:
				continue
			build_loc = random.choice(build_opts_list)
			build_opts_list.remove(build_loc)
			scmd = row[1]
			if build_loc[0] == 'Coast' and random.random() > 0.5:
				scmd = 'Build Fleet'
			dest_id = terr_id_tbl[build_loc[1]]
			sql = sqlBuildOrder.substitute(scmd=scmd, toTerrID=dest_id, id=row[2] )
			dstr = scountry + ' ' + scmd + ' in ' + build_loc[1]
		else:
			owns_list = unit_owns_tbl[scountry]
			destroy_data = random.choice(owns_list)
			dest_id = terr_id_tbl[destroy_data[1]]
			sql = sqlBuildOrder.substitute(scmd=row[1], toTerrID=dest_id, id=row[2] )
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
					   country_names_tbl, army_can_pass_tbl, fleet_can_pass_tbl):
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
			if unit_data[0] == 'Army':
				pass_list = army_can_pass_tbl.get(unit_data[1], None)
				if pass_list == None:
					print('Army stuck where it is not supposed to be.')
					continue
				dest_name = random.choice(pass_list)
				dest_id = terr_id_tbl[dest_name]
				dstr = scountry + ' move army from ' + unit_data[1] + ' to ' + dest_name
			# sqlMoveOrderUpdate = string.Template('UPDATE webdiplomacy.wD_Orders SET type=\'Move\', '
				# 									 'toTerrID = ${toTerrID} WHERE unitID = ${unitID} AND gameID = ${gameID} ;')
			else:
				pass_list = fleet_can_pass_tbl.get(unit_data[1], None)
				if pass_list == None:
					print('Fleet stuck where it is not supposed to be.')
					continue
				dest_name = random.choice(pass_list)
				dest_id = terr_id_tbl[dest_name]
				dstr = scountry + ' move fleet from ' + unit_data[1] + ' to ' + dest_name

			sql_order = sql_make_order.substitute(toTerrID=str(dest_id), unitID=str(unit_id[0]), gameID=str(gameID))
			print(dstr)
			print (sql_order)
			cursor.execute(sql_order)

		sql_order = sql_complete_order.substitute(gameID=str(gameID), countryID=str(icountry), timeLoggedIn=str(int(time.time())))
		print(sql_order)
		cursor.execute(sql_order)

	db.commit()


def play_turn(db, cursor, gname, country_names_tbl, terr_id_tbl, supply_tbl, army_can_pass_tbl, fleet_can_pass_tbl):
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
		return gameID, False

	# gameID = 28

	# sqlDone = 'UPDATE webdiplomacy.wD_Members SET orderStatus = \'Saved,Completed,Ready\' WHERE id = 20 ;'
	# cursor.execute(sqlDone)
	# db.commit()

	# CreateGame()
	# AddPlayers(gameID)

	terr_owns_tbl = OwnsTerrTbl(cursor, gameID, country_names_tbl)
	unit_owns_tbl = OwnsUnitsTbl(cursor, gameID, country_names_tbl)

	# elif get_phase(gameID) == 'Builds':
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
						   country_names_tbl, army_can_pass_tbl, fleet_can_pass_tbl)
	elif game_phase == 'Finished':
		return gameID, True
	else:
		return gameID, False

	time.sleep(0.2)

	return gameID, False

def init(cursor, country_names_tbl):
	terr_id_tbl = dict()
	army_can_pass_tbl = dict()
	fleet_can_pass_tbl = dict()

	PassInsert(cursor, 'armysPass', 'Army', army_can_pass_tbl)
	PassInsert(cursor, 'fleetsPass', 'Fleet', fleet_can_pass_tbl)

	supply_tbl = create_supply_tbl(cursor, country_names_tbl)

	create_terr_id_tbl(cursor, terr_id_tbl)

	return terr_id_tbl, army_can_pass_tbl, fleet_can_pass_tbl, supply_tbl

def play(gameID):

	if gameID == -1:
		gname = 't' + str(int(time.time()))[-6:]

	country_names_tbl = ['Neutral', 'England', 'France', 'Italy', 'Germany', 'Austria', 'Turkey', 'Russia']


	# Open database connection
	with closing(MySQLdb.connect("localhost","webdiplomacy","mypassword123","webdiplomacy" )) as db:
		# prepare a cursor object using cursor() method
		with closing(db.cursor()) as cursor:

			terr_id_tbl, army_can_pass_tbl, fleet_can_pass_tbl, supply_tbl = init(cursor, country_names_tbl)

			if gameID != -1:
				gname = get_game_name(cursor, gameID)

			for _ in range(300):
				gameID, b_finished = play_turn(	db, cursor, gname, country_names_tbl,
												terr_id_tbl, supply_tbl, army_can_pass_tbl,
												fleet_can_pass_tbl)
				print('Next turn for game id:', gameID)
				process_url = string.Template("http://localhost/board.php?gameID=${gameID}&process=Yes")
				try:
					response = urllib2.urlopen(process_url.substitute(gameID=gameID))
				except urllib2.HTTPError as err:
					print('HTTP Exception: ', err)
				time.sleep(0.2)
				if b_finished:
					print('Game Over!')
					return


def main():
	gameID = -1 # Set to -1 to restart
	play(gameID)

if __name__ == "__main__":
    main()
