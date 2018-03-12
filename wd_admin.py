from __future__ import print_function
import MySQLdb
import string


def DeleteFinishedGames(db, cursor):
	game_table_names = ['Members', 'Orders', 'TerrStatus', 'Units',
						'GameMessages', 'TerrStatusArchive', 'MovesArchive']
	sqlGetGameIDs = 'SELECT id FROM webdiplomacy.wD_Games where phase = \'Finished\';'
	cursor.execute(sqlGetGameIDs)
	results = cursor.fetchall()
	sqlDeleteFromGameTable = string.Template('DELETE FROM webdiplomacy.wD_${tablename} WHERE gameID = ${gameID}')
	sqlRemoveFromGames = string.Template('DELETE FROM webdiplomacy.wD_Games WHERE id = ${id}')

	gameIDs = [row[0] for row in results]

	for gameID in gameIDs:
		print('Delete game', gameID)
		for table_name in game_table_names:
			sql = sqlDeleteFromGameTable.substitute(tablename=table_name, gameID=gameID)
			num_rows_affected = cursor.execute(sql)
			print('Deleted', num_rows_affected, 'rows from table ', table_name, 'for game', gameID)
		sql = sqlRemoveFromGames.substitute(id=gameID)
		num_rows_affected = cursor.execute(sql)
		print('Deleted', num_rows_affected, 'rows from table Games for game', gameID)

	db.commit()

	# sqlAddPlayer = string.Template(
	# 	'INSERT INTO webdiplomacy.wD_Members SET userID = ${userid}, countryID = 0, bet = 10, '
	# 	'timeLoggedIn = 1516721310, gameID = ${gameID}, orderStatus = \'None,Completed,Ready\';')

"""
	static function wipeBackups()
	{
		global $DB;

		foreach(self::$gameTables as $tableName=>$idColName)
			$DB->sql_put("DROP TABLE IF EXISTS wD_Backup_".$tableName);
	}


	static private function backupTables()
	{
		global $DB;

		foreach(self::$gameTables as $tableName=>$idColName)
			$DB->sql_put("CREATE TABLE IF NOT EXISTS wD_Backup_".$tableName." LIKE wD_".$tableName);
	}

 sudo find . -name 'gamelog.txt' -type f -delete

"""