vs = [range(2), range(3), range(3)]
n = 1
for k in vs:
	n *= len(k)
r = [[] for i in range(n)]
for i in range(len(vs)):
	v = vs[i]
	numrecdiv = 1
	for irest in range(i + 1, len(vs)):
		numrecdiv *= len(vs[irest])
	recval = -1
	for irec in range(n):
		if irec % numrecdiv == 0:
			recval += 1
		if recval == len(v):
			recval = 0
		r[irec].append(recval)

