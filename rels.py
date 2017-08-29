from enum import Enum
from time import gmtime, strftime
import sys
import logging

logger = logging.getLogger('logic')
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

logger.setLevel(logging.DEBUG)
logger.info('Starting at: %s', strftime("%Y-%m-%d %H:%M:%S", gmtime()))



Rels = Enum('Rels', 'parent child uncle aunt brother sister grandpa grandma')
l = list(Rels)
for rel in Rels:
	print rel

a = Rels(3)
b = l[3]
print a, b,  l
logger.info(l)
