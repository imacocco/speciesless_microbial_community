import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import integrate
import seaborn as sns;# sns.set()
import numpy.linalg as linalg

class Patch:

	def __init__(self, noise=0, n_phen=20, resources=np.ones(20), ind_per_phen=1, base_phenotypes=None, skip=0, n_flip=0, prob_flip=0,\
				 patch = None, mask = None, ekpyrosis = False):

		#Initialise the patch by loading the phenotypes and computing their costs

		self.M_ab = None 		# jacobian near equilibrium
		self.evc  = None 		# jacobian eigenvectors
		self.evl  = None 		# jacobian eqigenvalues
		self.n_steps = 5000 	# default number of integration steps for dynamics

		#create a new patch with active populations	from an evolved patch
		if ( ( patch is not None ) and ( mask is not None ) ):
			
			self.phenotypes  	   = patch.phenotypes[mask]
			self.populations 	   = patch.final_populations[mask]
			self.final_populations = self.populations
			self.costs       	   = patch.costs[mask]
			self.resources 		   = patch.resources
			self.n_steps		   = patch.n_steps
		
		#new patch from scratch
		else:
	
			#if no base phenotype is given we randomize all the phenotypes and set populations to 1
			if base_phenotypes is None:

				self.populations = np.ones(n_phen)
				self.resources = resources
				
				#put some condition on the phenotypes (ex: no individual with equal first and last)
				if ekpyrosis is True:

					self.phenotypes = []
					counter = 0
					while (counter < n_phen):
						appo = np.random.randint(0,2, size = (len(resources)))
						if appo[0] == appo[-1]:
							continue
						else:
							counter += 1
							self.phenotypes.append(appo)

				else: 		#fully randomizedd phenotypes
					self.phenotypes = np.random.randint( 0, 2, size=( n_phen,len(resources) ) )


			#if some core phenotypes are given we know which is the number of resources
			else:
				self.resources = np.ones(len(base_phenotypes[0]))
				self.populations = np.ones(len(base_phenotypes)*ind_per_phen)
				self.phenotypes = []
				#fill in phenotypes
				for i in range(len(base_phenotypes)):
					for j in range(ind_per_phen):
						new_pheno = self.load_competitor( base_phenotypes[i], skip, n_flip, prob_flip )
						self.phenotypes.append( new_pheno )

			self.final_populations = self.populations
			self.phenotypes = np.array(self.phenotypes)
			self.costs = self.compute_costs(noise)
	

	def load_competitor(self, base_phenotype, skip=0, n_flip=0, prob_flip=0):
	
		#load a new phenotype int the patch based on the base_phenotype
			#skip: skip the first n pathway from randomisation
			#n_flip: select a fixed amount of pathways to flip from the base
			#prob_flip: probability to flip each single pathway

		competitor = list.copy(base_phenotype)
		
		if (n_flip != 0):
			#select n_flip random pathways (after the fixed ones) to flip
			idx_flip =  np.random.choice( np.arange(skip,len(base_phenotype)), n_flip, replace = False) 
			for i in idx_flip:
				competitor[i] = FLIP(competitor[i])
			return competitor
		
		else:
			#randomly flip each pathway (after the fixed ones) with given prob_flip
			for i in range(skip,len(base_phenotype)):
				if(np.random.rand() > (1 - prob_flip) ):
					competitor[i] = FLIP(competitor[i])
			return competitor


	def compute_costs(self, noise=0.01):

		#compute the cost associated to each phenotype

		return np.array( [ sum(self.phenotypes[i])+np.random.normal()*noise for i in range(len(self.phenotypes)) ] )


	def print_patch(self):

		#print the heatmap of all phenotypes

		sns.heatmap(self.phenotypes)
		plt.show()


	def set_populations(self, pop):

		#reset the populations

		if len(pop) == 1:
			self.populations = np.ones(len(self.phenotypes))*pop

		elif len(pop) == len(self.phenotypes):
			self.populations = pop

		else:
			print('wrong format for new population')



	def set_resources(self, res):

		#reset the populations

		if len([res]) == 1:
			self.resources = np.ones(len(self.resources))*res

		elif len(res) == len(self.resources):
			self.resources = res

		else:
			print('wrong format for new resources')


	def compute_individuals_per_resource(self, pop = None):

		#compute how many individuals there are for each resource
		if pop is None:
			pop = self.populations

		return np.array( [sum(self.phenotypes[:,i]*pop) for i in range(len(self.phenotypes[0])) ] )


	def compute_surplus(self, pop = None, res = None):

		#compute the surplus for each phenotype

		if pop is None:
			pop = self.populations
		if res is None:
			res = self.resources

		ind_per_resource = self.compute_individuals_per_resource(pop)
		res_per_ind = res/ind_per_resource
		surplus = [ sum(self.phenotypes[i]*res_per_ind) - self.costs[i] for i in range(len(self.phenotypes)) ]
		return np.array(surplus)


	def compute_full_ADJ(self):

		#compute plain adjacency matrix

		adj = []
		for i in range(len(self.phenotypes)):
			adj.append( [ sum(self.phenotypes[i]*self.phenotypes[j]) for j in range(len(self.phenotypes)) ] )
		
		return np.array(adj)


	def compute_M_ab(self):

		#compute interaction matrix near fixed point

		M_ab = []
		T = self.compute_individuals_per_resource()
		#print(len(T),len(X),len(phenotypes[0]),len(resources))
		for i in range(len(self.phenotypes)):
			M_ab.append( [ -(self.populations[i]/self.costs[i])*sum(self.phenotypes[i]*self.phenotypes[j]*self.resources/T/T) for j in range(len(self.phenotypes)) ] )
		
		self.M_ab = np.array(M_ab)


	def compute_spectrum(self, verb = True):

		#compute eigenvalues and eigenvectors

		if self.M_ab is None:
			self.compute_M_ab()

		evl, evc = linalg.eig(self.M_ab)
		idx = evl.argsort()
		self.evl = evl[idx]
		self.evc = evc[:,idx]
#		T = evc.T
		if verb is True:
			self.plot_spectrum()


	def plot_spectrum(self):

		#plot the eigenvalues

		if self.evl is None:
			self.compute_spectrum()

		x = np.ones(len(self.evl))
		#plt.scatter(x,evl,linewidths=0)
		plt.errorbar(x,abs(self.evl),xerr=0.01,fmt='none', ecolor='k')
		plt.xticks([])
		plt.xlabel('')
		plt.ylabel('|$\lambda$|')
		plt.show()


	def print_eigenpatch(self):

		#print the eigenvector projected onto the pathways basis
		
		eig_cmap = sns.diverging_palette(264.5, 10.8, as_cmap=True, sep=40, s=100, center='light', l=50)
		
		if self.evc is None:
			self.compute_spectrum()

		sns.heatmap( np.dot( np.real(self.evc.T),self.phenotypes), cmap=eig_cmap);
		plt.show()


	def g(self, populations):

		#dn/dt = g(n), here we compute g(n)

		return populations*self.compute_surplus(populations)/self.costs

	g.terminal=True

	def g_for_solver(self, t, z):

		#same as g but to include in the ODE solver
		#NB z == populations

		return z*self.compute_surplus(z)/self.costs


	def check_stationarity(self, pops = None, threshold = 1e-4):

		#condition of equilibrium within the patch

		if pops is None:
			pops = self.final_populations
		
		abs_val =  np.array( [ abs( self.g( pops[i] ) ) for i in range(len(pops)) ] )
		cond = np.array( [ np.all( abs_val[i] < threshold ) for i in range(len(pops)) ] )

		if np.any( cond ):
			index = np.where(cond == True )[0][0]
			return index
		else:
			print('Not all the derivatives fell below the given threshold of', threshold, 'before the last integration step.')
			print('Be careful, the system may not have properly converged to a stationary state!')
#			sys.exit('Be careful, the system may have not properly converged to a stationary state!')
			return -1


	def evolve(self, n_steps = None, temp_resource = None, plot=True, verb=False, check = True, thres = 1e-4, traj = None):

		#let the system evolve solving the coupled differential equations

		if temp_resource is not None:
			self.set_resources(temp_resource)

		if verb is True:
			print(self.resources[0])

		if n_steps is None:
			n_steps = self.n_steps

		#solve the ODE system
		sol = integrate.solve_ivp(self.g_for_solver, [0, n_steps], self.populations, dense_output=True)

		#save and visualise the solution
		t = np.linspace(0, n_steps, int(n_steps/10))
		#access to full solution, save z if necessary
		z = sol.sol(t)

		if plot == True:
			[plt.plot(t, z.T[:,i],label='phen'+ str(i)) for i in range(len(z.T[0]))];
			plt.show()

		if check is True:
			index = self.check_stationarity(z.T, thres)

#			if index == -1:
#				self.n_steps *= 1.5
#			else:
#				self.n_steps = index*20

			self.final_populations =  z.T[index]

			if verb:
				print(index)
		else:
			self.final_populations =  z.T[-1]

		if traj == 'last' or traj == -1:
			return self.final_populations
		elif traj == 'full':
			return z.T


# external definitions of time evolution functions, maybe these are conceptually more appropriate as an implementation

def FLIP(before):
	if before==0:
		return 1
	else:
		return 0

def compute_mask(populations, threshold):
    masks = [ populations[i] > threshold for i in range(len(populations)) ]
    masks = np.array(masks)
    [ print(sum(masks[i])) for i in range(len(masks)) ]
    final_mask = [ np.all(masks[:,i]) for i in range(len(masks[0])) ]
    return final_mask


def ODE_system_ivp(t, z, patch):

	surplus = patch.compute_surplus(z)
	next_step = z*surplus/patch.costs
#    if np.all(abs(next_step) < 1e-5) :
#        print(t)
#        print(next_step)
 #   else:
	return next_step


def equilibrium(t, y):

	#condition of equilibrium within the patch
	if np.all(ODE_system_ivp(t,y,patch) < 1e-2):
		return 0
	else:
		return 1

equilibrium.terminal = True

def evolve(n_steps, patch, plot=True,verb=True):
	if verb==True:
		print(patch.resources[0])
	#solve the ODE system
	sol = integrate.solve_ivp(ODE_system_ivp, [0, n_steps], patch.populations, args=(patch,), dense_output=True)
	#save and visualise the solution
	t = np.linspace(0, n_steps, int(n_steps/10))
	z = sol.sol(t)
	if plot == True:
		[plt.plot(t, z.T[:,i],label='phen'+ str(i)) for i in range(len(z.T[0]))];
		plt.show()
	#return the final population to a list
	return z.T[-1]