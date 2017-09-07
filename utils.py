from time import gmtime, strftime
import sys
import logging

ulogger = None

def init_logging():
	global ulogger
	if ulogger != None:
		return

	ulogger = logging.getLogger('logic')
	ch = logging.StreamHandler(stream=sys.stdout)
	ch.setLevel(logging.DEBUG)
	ulogger.addHandler(ch)
	ulogger.setLevel(logging.DEBUG)
	ulogger.info('Starting at: %s', strftime("%Y-%m-%d %H:%M:%S", gmtime()))



if ulogger == None:
	init_logging()

