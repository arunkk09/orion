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
F = 1.4
MODELS = ('dense_logit') #logistic regression
APPROACHES = ('m','s','f', 'sr') #Materialize, Stream, Factorize, Stream-Rruse
ALGOS = ('bgd', 'cgd', 'lbfgs') #default m for lbfgs is 4 
PARAMS = {
		# required
		'model' : None, #only dense logistic regression available now
		'approach' : None, #the learning over join approach
		'algo' : None, #the gradient method
		'model_id' : None, #the id of this model
		'table_s' : None, #name of the entity table S (the outer table for join)
		'table_r' : None, #name of the attribute table R (the inner table for join)
		'table_t' : None, #name for the denormalized table T
		'dS' : None, #number of features in S
		'dR' : None, #number of features in R
		'nR' : None, #number of tuples in R
		'srsplits' : None, #number of splits; only for Stream-Reuse
		# with default
		'col_xS' : 'xs', #name of feature vector column in S
		'col_xR' : 'xr', #name of feature vector column in R
		'col_x' : 'x', #name of feature vector column in T
		'col_label' : 'label', #name of target/label column in S
		'col_forkey' : 'forkey', #name of foreign key column in S
		'col_rid' : 'rid', #name of primary key column in R
		'col_sid' : 'sid', #name of primary key column in S
		'iters' : 20, #maximum number of iterations of gradient method
		'w_init' : 0.01, #starting coefficient for model vector
		'stepsize' : 1, #starting stepsize parameter (alpha)
		'regularizer' : 0.1, #L2 regularization parameter
		# optional
		'tolerance' : None, #fractional improvement in loss; if given, stop condition is (iters <= given iters) and (tolerance > given tolerance)
		'output_file' : None #
		}

class DBInterface(object) :
	def __init__(self) :
		# connect DB using default connect string
		self.conn = psycopg2.connect('')

	def __del__(self) :
		self.conn.commit()
		self.conn.close()

	def execute(self, query) :
		cursor = self.conn.cursor()
		if VERBOSE :
			print 'executing: ', query
		cursor.execute(query)
		cursor.close()

	def execute_and_fetch(self, query) :
		cursor = self.conn.cursor()
		if VERBOSE :
			print 'executing: ', query
		cursor.execute(query)
		ret = cursor.fetchall()
		cursor.close()
		return ret

DB = DBInterface()

class Model(object) :
	def __init__(self) :
		self.model = PARAMS['model']
		self.approach = PARAMS['approach']
		self.algo = PARAMS['algo']
		self.model_id = PARAMS['model_id']
		self.table_r = PARAMS['table_r']
		self.table_s = PARAMS['table_s']
		self.table_t = PARAMS['table_t']
		self.col_xS = PARAMS['col_xS']
		self.col_xR = PARAMS['col_xR']
		self.col_x = PARAMS['col_x']
		self.col_label = PARAMS['col_label']
		self.col_forkey = PARAMS['col_forkey']
		self.col_sid = PARAMS['col_sid']
		self.col_rid = PARAMS['col_rid']
		self.nR = PARAMS['nR']
		self.dS = PARAMS['dS']
		self.dR = PARAMS['dR']
		self.srsplits = PARAMS['srsplits']
		self.iters = PARAMS['iters']
		self.w_init = PARAMS['w_init']
		self.stepsize = PARAMS['stepsize']
		self.regularizer = PARAMS['regularizer']
		self.tolerance = PARAMS['tolerance']
		self.output_file = PARAMS['output_file']

	def create_model(self) :
		ndims = self.dS + self.dR - 1
		if (self.approach == 'f') :
			vnb = int(F * self.nR);
			DB.execute('SELECT {0}_fact_shmem_push({1}, {2}, {3}, {4});'.format(self.model, self.model_id, vnb, self.dS, self.dR))
			
		else :
			DB.execute('SELECT {0}_shmem_push({1}, {2});'.format(self.model, self.model_id, ndims))

	def put_model(self, wvec) :
		if (self.approach == 'f') :
			return DB.execute('SELECT {0}_fact_putmodel({1}, {2})'.format(self.model, self.model_id, "'{" + ','.join(map(str,wvec)) + "}'"))
		else :
			return DB.execute('SELECT {0}_putmodel({1}, {2})'.format(self.model, self.model_id, "'{" + ','.join(map(str,wvec)) + "}'"))

	def prep(self) :
		self.create_model()
		self.put_model(self.w)
		if (self.approach == 'm') : #create table t as select sid, label, xS || xR as x from r_rr_$rr, s where s.rid = r_rr_$rr.forkey
			DB.execute('DROP TABLE IF EXISTS {0}; CREATE TABLE {0} AS SELECT {1}, {2}, {3} || {4} AS {5} FROM {6}, {7} WHERE {7}.{8} = {6}.{9}'
				.format(self.table_t, self.col_sid, self.col_label, self.col_xS, self.col_xR, self.col_x, self.table_s, self.table_r, self.col_rid, self.col_forkey))

	def get_model(self) :
		if (self.approach == 'f') :
			return DB.execute_and_fetch('SELECT {0}_fact_getmodel({1})'.format(self.model, self.model_id))[0][0]
			#sys.exit(0)
		else :
			return DB.execute_and_fetch('SELECT {0}_getmodel({1})'.format(self.model, self.model_id))[0][0]

	def pop_model(self) :
		if (self.approach == 'f') :
			DB.execute('SELECT {0}_fact_shmem_pop({1})'.format(self.model, self.model_id))
		else :
			DB.execute('SELECT {0}_shmem_pop({1})'.format(self.model, self.model_id))

	def gradloss(self) :
		if (self.approach == 'm') :
			vloss = DB.execute_and_fetch('SELECT sum({0}_grad({1}, {2}, {3})) FROM {4}'
					.format(self.model, self.model_id, self.col_x, self.col_label, self.table_t))[0][0]
			return vloss
		elif (self.approach == 's') :
			vloss = DB.execute_and_fetch('SELECT sum({0}_grad({1}, {2} || {3}, {4})) FROM {5}, {6} where {6}.{7} = {5}.{8}'
					.format(self.model, self.model_id, self.col_xS, self.col_xR, self.col_label, self.table_s, self.table_r, self.col_rid, self.col_forkey))[0][0]
			return vloss
		elif (self.approach == 'sr') :
			if (self.srsplits == None) :
				print 'Error: Number of SR splits not given'
				sys.exit(0)
			vloss = 0
			for i in range(self.srsplits) :
				vloss += DB.execute_and_fetch('SELECT sum({0}_grad({1}, {2} || {3}, {4})) FROM {5}_{9}, {6}_{9} where {6}_{9}.{7} = {5}_{9}.{8}'.format
					(self.model, self.model_id, self.col_xS, self.col_xR, self.col_label, self.table_s, self.table_r, self.col_rid, self.col_forkey, str(i + 1)))[0][0]
			return vloss
		elif (self.approach == 'f') :
			DB.execute('SELECT count({0}_fact_func1({1}, {2}, {3})) FROM {4}'.format(self.model, self.model_id, self.col_rid, self.col_xR, self.table_r))
			vloss = DB.execute_and_fetch('SELECT sum({0}_fact_func2({1}, {2}, {3}, {4})) FROM {5}'
					.format(self.model, self.model_id, self.col_forkey, self.col_xS, self.col_label, self.table_s))[0][0]
			DB.execute('SELECT count({0}_fact_func3({1}, {2}, {3})) FROM {4}'.format(self.model, self.model_id, self.col_rid, self.col_xR, self.table_r))
			return vloss

	#alphak = model.line_search(model.stepsize, current_loss, phiprimeat0)
	def line_search(self, alphabar, phiat0, phiprimeat0, wonly, ponly) :
		alpha = alphabar
		c = 1e-4
		rho = 0.5
		count = 0
		while True :
			count += 1
			neww = [m + alpha * n for m, n in zip(wonly,ponly)]
			self.put_model(neww)
			phiatalpha = self.gradloss()
			phiatalpha += self.regularizer * 0.5 * sum(map(lambda x:x*x, neww)) #L2 norm sq reglzn
			if VERBOSE :
				print 'phi(a) = ',  phiatalpha, ', phi(0) = ', phiat0, ', phipr(0) = ', phiprimeat0, ', rhs = ', (phiat0 + c * alpha * phiprimeat0)
			if phiatalpha <= (phiat0 + c * alpha * phiprimeat0) :
				if OUTPUT :
					print count, '\t', alpha
				return alpha * 1.2 #heuristic increase
			alpha = rho * alpha
		

#fout = open(self.output_file, 'w')
#print >> fout, '\t'.join([str(_) for _ in self.w])
#fout.close()

class LinearModel(Model) :
	def __init__(self) :
		super(LinearModel, self).__init__()
		vndims = self.dS + self.dR - 1
		self.w = [self.w_init for _ in range(vndims)]

class dense_logit(LinearModel) :
	def __init__(self) :
		super(dense_logit, self).__init__()
		self.model = 'dense_logit'

def main() :
	# parameters from arguments
	spec = imp.load_source('dense-logit-spec.py', sys.argv[1])
	for k in PARAMS :
		if k in spec.__dict__ :
			PARAMS[k] = spec.__dict__[k]
	
	# verbosing
	global VERBOSE
	try :
		VERBOSE = spec.verbose
	except AttributeError :
		VERBOSE = False
	
	# build the object for the specified model class
	model = None
	if spec.model in MODELS :
		model = globals()[spec.model]()
	else :
		print >> sys.stderr, 'model', spec.model, 'is not available'
		sys.exit(2)
	if VERBOSE and OUTPUT :
		print 'attributes of model:'
		print [(k, v) for k, v in model.__dict__.items() if k != 'w']
	
	# main control block for batch gradient descent (BGD)
	model.prep()
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
	for i in range(model.iters) : #max iters
	
		if(model.algo == 'bgd') :
			current_loss = model.gradloss()
			wvec = model.get_model() #w, g
			wonly = wvec[0 : len(wvec)/2]
			gonly = wvec[len(wvec)/2 : len(wvec) + 1]
			current_loss += model.regularizer * 0.5 * sum(map(lambda x:x*x, wonly)) #L2 norm sq reglzn
			gonly = [n + model.regularizer * m for m, n in zip(wonly, gonly)] #L2 norm sq reglzn
			if OUTPUT :
				normg = math.sqrt(sum(map(lambda x:x*x,gonly)))
				strr = str (i + 1) + '\t' + str(current_loss) + '\t' + str(normg) #+ '\t' + "'{" + ','.join(map(str,wonly)) + "}'"
				print strr
			improvement = None
			if i > 0 and previous_loss is not None and previous_loss != 0.0 :
				improvement = (previous_loss - current_loss) / previous_loss
			if model.tolerance is not None and improvement <= model.tolerance :
				break

			ponly = map(lambda x:-x, gonly) #BGD direction
			#BGD update with line search
			phiprimeat0 = -1.0 * sum(map(lambda x:x*x,gonly))
			alphak = model.line_search(model.stepsize, current_loss, phiprimeat0, wonly, ponly)
			
			neww = [m + alphak * n for m, n in zip(wonly, ponly)]
			model.put_model(neww)
			previous_loss = current_loss
			model.stepsize = alphak #use as starting stepsize for next epoch
			
		elif(model.algo == 'cgd') :
			current_loss = model.gradloss()
			wvec = model.get_model() #w, g
			wonly = wvec[0 : len(wvec)/2]
			gonly = wvec[len(wvec)/2 : len(wvec) + 1]
			current_loss += model.regularizer * 0.5 * sum(map(lambda x:x*x, wonly)) #L2 norm sq reglzn
			gonly = [n + model.regularizer * m for m, n in zip(wonly, gonly)] #L2 norm sq reglzn
			if OUTPUT :
				normg = math.sqrt(sum(map(lambda x:x*x,gonly)))
				strr = str (i + 1) + '\t' + str(current_loss) + '\t' + str(normg) #+ '\t' + "'{" + ','.join(map(str,wonly)) + "}'" + '\t' + "'{" + ','.join(map(str,gonly)) + "}'"
				print strr
			improvement = None
			if i > 0 and previous_loss is not None and previous_loss != 0.0 :
				improvement = (previous_loss - current_loss) / previous_loss
			if model.tolerance is not None and improvement <= model.tolerance :
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
			alphak = model.line_search(model.stepsize, current_loss, phiprimeat0, wonly, ponly)
			
			neww = [m + alphak * n for m, n in zip(wonly, ponly)]
			model.put_model(neww)
			previous_loss = current_loss
			previous_grad = gonly
			previous_p = ponly
			model.stepsize = alphak #use as starting stepsize for next epoch
			
		elif(model.algo == 'lbfgs') :
			current_loss = model.gradloss()
			wvec = model.get_model() #w, g
			wonly = wvec[0 : len(wvec)/2]
			gonly = wvec[len(wvec)/2 : len(wvec) + 1]
			current_loss += model.regularizer * 0.5 * sum(map(lambda x:x*x, wonly)) #L2 norm sq reglzn
			gonly = [n + model.regularizer * m for m, n in zip(wonly, gonly)] #L2 norm sq reglzn
			if OUTPUT :
				normg = math.sqrt(sum(map(lambda x:x*x,gonly)))
				strr = str (i + 1) + '\t' + str(current_loss) + '\t' + str(normg) #+ '\t' + "'{" + ','.join(map(str,wonly)) + "}'" + '\t' + "'{" + ','.join(map(str,gonly)) + "}'"
				print strr
			improvement = None
			if i > 0 and previous_loss is not None and previous_loss != 0.0 :
				improvement = (previous_loss - current_loss) / previous_loss
			if model.tolerance is not None and improvement <= model.tolerance :
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
			alphak = model.line_search(1.0, current_loss, phiprimeat0, wonly, ponly)
			
			neww = [m + alphak * n for m, n in zip(wonly, ponly)]
			model.put_model(neww)
			previous_loss = current_loss
			previous_grad = gonly
			previous_p = ponly
			previous_w = wonly
			model.stepsize = alphak #use as starting stepsize for next epoch
	
	model.pop_model()

if __name__ == '__main__' :
	if len(sys.argv) != 2 :
		print >> sys.stderr, 'Usage: python orion_front.py [spec_file]'
	else :
		random.seed()
		main()

