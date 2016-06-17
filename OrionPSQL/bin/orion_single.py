'''
Copyright 2015 Arun Kumar

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import sys, imp, math, random
import psycopg2
from cStringIO import StringIO

VERBOSE = False
OUTPUT = True
ALGOS = set(['bgd', 'cgd', 'lbfgs']) #default m for lbfgs is 5 

def sigma(v) :
	if (v > 30) :
		return (1.0 / (1.0 + math.exp(-v)))
  	else :
  		return (math.exp(v) / (1.0 + math.exp(v)))

#return tuple with (loss, grad) 
def gradloss(w, tuples) :
	loss = 0
	ndims = len(tuples[0]) - 2
	rndims = range(ndims)
	grad = [0 for _ in rndims]
	for dat in tuples :
		#rid, label, x
		wx = 0.0
		for i in rndims :
			wx += w[i] * dat[2 + i]
		loss += math.log(1 + math.exp(-dat[1] * wx))
		c = -dat[1] * sigma(-dat[1] * wx);
		for i in rndims :
			grad[i] += c * dat[2 + i]
	return (loss, grad)

#alphak = line_search(stepsize, current_loss, phiprimeat0)
def line_search(alphabar, phiat0, phiprimeat0, wonly, ponly, regularizer, tuples) :
	alpha = alphabar
	c = 1e-4
	rho = 0.5
	count = 0
	while True :
		count += 1
		neww = [m + alpha * n for m, n in zip(wonly,ponly)]
		phiatalpha = gradloss(neww, tuples)[0]
		phiatalpha += regularizer * 0.5 * sum(map(lambda x:x*x, neww)) #L2 norm sq reglzn
		if VERBOSE :
			print 'phi(a) = ',  phiatalpha, ', phi(0) = ', phiat0, ', phipr(0) = ', phiprimeat0, ', rhs = ', (phiat0 + c * alpha * phiprimeat0)
		if phiatalpha <= (phiat0 + c * alpha * phiprimeat0) :
			if OUTPUT :
				print count, '\t', alpha
			return alpha * 1.2 #heuristic increase
		alpha = rho * alpha
	
def main() :
	# parameters from arguments; check orion_front.py for explanation of arguments
	if(len(sys.argv) < 9) :
		print >> sys.stderr, 'usage: algo table_t dS dR iters w_init stepsize regularizer [tolerance]'
		sys.exit(2)
	algo = sys.argv[1];
	if(algo not in ALGOS) :
		print >> sys.stderr, 'algo must be one of', ALGOS
		sys.exit(2)
	table_t = sys.argv[2] #filename
	file_t = open(table_t, "r")
	dS = int(sys.argv[3])
	dR = int(sys.argv[4])
	iters = int(sys.argv[5])
	w_init = float(sys.argv[6])
	stepsize = float(sys.argv[7])
	regularizer = float(sys.argv[8])
	tolerance = None
	if(len(sys.argv) > 9) :
		tolerance = float(sys.argv[9])

	if VERBOSE and OUTPUT :
		print 'attributes of model:'
		print 'algo', algo, 'table_t', table_t, 'dS', dS, 'dR', dR, 'iters', iters, 'w_init', w_init, 'stepsize', stepsize, 'regularizer', regularizer
		if(len(sys.argv) > 9) :
			print 'and tolerance', tolerance
	
	vndims = dS + dR - 1
	w0 = [w_init for _ in range(vndims)]
	
	#read file into arrays of schema: rid, label, x[]
	tuples = []
	for line in file_t :
		vs = line.split()
		t = []
		for v in vs :
			t.append(float(v))
		tuples.append(t)
	file_t.close()
	
	# main control block for gradient descent algos
	previous_loss = None
	previous_grad = None
	previous_p = None
	previous_w = None
	lbm = 5 #for lbfgs
	lbs = []
	lbt = []
	for i in range(lbm) :
		lbs.append(None)
		lbt.append(None)
	titlestr = 'Iteration\tLoss\tNorm of Gradient\n(If line search performned, number of iterations and chosen stepsize printed too)\n'
	print titlestr
	neww = w0
	for i in range(iters) : #max iters
		
		if(algo == 'bgd') :
			retlg = gradloss(neww, tuples)
			current_loss = retlg[0]
			wonly = neww
			gonly = retlg[1]
			current_loss += regularizer * 0.5 * sum(map(lambda x:x*x, wonly)) #L2 norm sq reglzn
			gonly = [n + regularizer * m for m, n in zip(wonly, gonly)] #L2 norm sq reglzn
			if OUTPUT :
				normg = math.sqrt(sum(map(lambda x:x*x,gonly)))
				strr = str (i + 1) + '\t' + str(current_loss) + '\t' + str(normg) #+ '\t' + "'{" + ','.join(map(str,wonly)) + "}'"
				print strr
			improvement = None
			if i > 0 and previous_loss is not None and previous_loss != 0.0 :
				improvement = (previous_loss - current_loss) / previous_loss
			if tolerance is not None and improvement <= tolerance :
				break

			ponly = map(lambda x:-x, gonly) #BGD direction
			#BGD update with line search
			phiprimeat0 = -1.0 * sum(map(lambda x:x*x,gonly))
			alphak = line_search(stepsize, current_loss, phiprimeat0, wonly, ponly, regularizer, tuples)
			
			neww = [m + alphak * n for m, n in zip(wonly, ponly)]
			previous_loss = current_loss
			stepsize = alphak #use as starting stepsize for next epoch
			
		elif(algo == 'cgd') :
			retlg = gradloss(neww, tuples)
			current_loss = retlg[0]
			wonly = neww
			gonly = retlg[1]
			current_loss += regularizer * 0.5 * sum(map(lambda x:x*x, wonly)) #L2 norm sq reglzn
			gonly = [n + regularizer * m for m, n in zip(wonly, gonly)] #L2 norm sq reglzn
			if OUTPUT :
				normg = math.sqrt(sum(map(lambda x:x*x,gonly)))
				strr = str (i + 1) + '\t' + str(current_loss) + '\t' + str(normg) #+ '\t' + "'{" + ','.join(map(str,wonly)) + "}'" + '\t' + "'{" + ','.join(map(str,gonly)) + "}'"
				print strr
			improvement = None
			if i > 0 and previous_loss is not None and previous_loss != 0.0 :
				improvement = (previous_loss - current_loss) / previous_loss
			if tolerance is not None and improvement <= tolerance :
				break

			if i == 0 :
				ponly = map(lambda x:-x, gonly) #CGD direction
			else :
				deltag = [m - n for m, n in zip(gonly, previous_grad)]
				gtdeltag = sum([m  * n for m, n in zip(gonly, deltag)])
				prevgnorm2 = sum(map(lambda x:x*x, previous_grad))
				if prevgnorm2 == 0.0 :
					print 'ERROR: prevgnorm2 is 0'
					prevgnorm2 = 1
				#betak = max(0, gtdeltag / prevgnorm2) #PR+
				
				#ptdeltag = sum([m  * n for m, n in zip(previous_p, deltag)]) #HS
				#betak = gtdeltag / ptdeltag
				
				gnorm2 = sum(map(lambda x:x*x, gonly)) #FR
				betak = gnorm2 / prevgnorm2
				
				#print 'betak = ', betak
				ponly = [-1.0 * m + betak * n for m, n in zip(gonly, previous_p)] #CGD direction
			
			#CGD update with line search
			phiprimeat0 = sum([m  * n for m, n in zip(gonly, ponly)])
			alphak = line_search(stepsize, current_loss, phiprimeat0, wonly, ponly, regularizer, tuples)
			
			neww = [m + alphak * n for m, n in zip(wonly, ponly)]
			previous_loss = current_loss
			previous_grad = gonly
			previous_p = ponly
			stepsize = alphak #use as starting stepsize for next epoch
			
		elif(algo == 'lbfgs') :
			retlg = gradloss(neww, tuples)
			current_loss = retlg[0]
			wonly = neww
			gonly = retlg[1]
			current_loss += regularizer * 0.5 * sum(map(lambda x:x*x, wonly)) #L2 norm sq reglzn
			gonly = [n + regularizer * m for m, n in zip(wonly, gonly)] #L2 norm sq reglzn
			if OUTPUT :
				normg = math.sqrt(sum(map(lambda x:x*x,gonly)))
				strr = str (i + 1) + '\t' + str(current_loss) + '\t' + str(normg) #+ '\t' + "'{" + ','.join(map(str,wonly)) + "}'" + '\t' + "'{" + ','.join(map(str,gonly)) + "}'"
				print strr
			improvement = None
			if i > 0 and previous_loss is not None and previous_loss != 0.0 :
				improvement = (previous_loss - current_loss) / previous_loss
			if tolerance is not None and improvement <= tolerance :
				break

			if i == 0 :
				ponly = map(lambda x:-x, gonly) #LBFGS direction
			else :
				deltaw = [m - n for m, n in zip(wonly, previous_w)]
				deltag = [m - n for m, n in zip(gonly, previous_grad)]
				#adv buffer entries and insert into its end
				for i in range(lbm - 1) :
					lbs[i] = lbs[i + 1]
					lbt[i] = lbt[i + 1]
				lbs[lbm - 1] = deltaw
				lbt[lbm - 1] = deltag
				#strr = 'lbs[' +  str(lbm - 1) + '] = ' + '\t' + "'{" + ','.join(map(str,deltaw)) + "}'"
				#print strr
				#strr = 'lbt[' +  str(lbm - 1) + '] = ' + '\t' + "'{" + ','.join(map(str,deltag)) + "}'"
				#print strr
 
				b0k = sum([m  * n for m, n in zip(deltaw, deltag)]) / sum(map(lambda x:x*x, deltag))
				if VERBOSE :
					print 'b0k = ' + str(b0k)
				q = gonly
				for i in range(lbm) :
					j = lbm - 1 - i
					if (lbs[j] != None) :
						coeff = sum([m  * n for m, n in zip(lbs[j], q)]) / sum([m  * n for m, n in zip(lbs[j], lbt[j])])
						q = [m - coeff * n for m, n in zip(q, lbt[j])]
					elif VERBOSE :
						print 'lbs[' + str(j) + '] is none'
				#strr = 'q = ' + '\t' + "'{" + ','.join(map(str,q)) + "}'"
				#print strr
				r = map(lambda x:b0k*x, q)
				#rescaling by 50x for R features
				#for i in range(18) :
				#	r[i] = r[i] * 0.02
				for i in range(lbm) :
					if (lbs[i] != None) :
						coeff = (sum([m  * n for m, n in zip(lbs[i], q)]) - sum([m  * n for m, n in zip(lbt[i], r)])) / sum([m  * n for m, n in zip(lbs[i], lbt[i])])
						r = [m + coeff * n for m, n in zip(r, lbs[i])]
					elif VERBOSE :
						print 'lbs[' + str(i) + '] is none'
				#strr = 'r = ' + '\t' + "'{" + ','.join(map(str,r)) + "}'"
				#print strr
				ponly = map(lambda x:-x, r) #LBFGS direction
			
			#LBFGS update with line search
			phiprimeat0 = sum([m  * n for m, n in zip(gonly, ponly)])
			#print 'before line search, ponly: ' + '\t' + "'{" + ','.join(map(str,ponly)) + "}'"
			#print 'before line search, gonly: ' + '\t' + "'{" + ','.join(map(str,gonly)) + "}'"
			#print 'before line search, descentpdt: ' + '\t' + str(phiprimeat0)
			alphak = line_search(1.0, current_loss, phiprimeat0, wonly, ponly, regularizer, tuples)
			
			neww = [m + alphak * n for m, n in zip(wonly, ponly)]
			previous_loss = current_loss
			previous_grad = gonly
			previous_p = ponly
			previous_w = wonly
			stepsize = alphak #use as starting stepsize for next epoch
	if OUTPUT :
		print "final w:", neww

main()
