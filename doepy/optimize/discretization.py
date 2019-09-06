
import numpy as np
from sympy import symbols, lambdify,exp

class Discretization:
	def __init__ (self, dop, model, disc=None, method=None):
		self.dop = dop
		self.model = model
		self.t_u = np.concatenate((self.dop.t_inputs,[self.dop.tf]))
		self.set_t_disc(disc)
		self.n_disc = len(self.t_disc)

		if method is None:
			self.eq = self.exp_euler
		else:
			self.eq = method

	#Discretization points
	def set_t_disc (self, disc):
		if disc is None:
			self.t_disc = self.t_u
		else:
			self.t_disc = np.unique(np.concatenate((disc,self.t_u)))
			self.t_disc.sort()


	def __call__ (self,u_flat):
		"""
		Return the discretized problem for the set of controls U
		"""
		return self.eq(u_flat)

	def exp_euler(self, u_flat):
		"""
		Discretize the optimization problem based on explicity Euler integration
		"""
		arg_symbols = symbols(['t:'+str(1), 'x:'+str(self.model.num_states),'u:'+str(self.model.num_inputs),'p:'+str(self.model.num_param)])
		sym_func = self.model.s_model(*arg_symbols)
		s_all = arg_symbols[1]

		leq = []
		snames = [0 for i in range(self.model.num_states)]

		n_x = 0
		states = []
		conststates = []
		functions = []
		scname = []

		for k, f in enumerate(sym_func):
			if f == 0:
				name = s_all[k].name +"_0"
				conststates.append(name)
				scname.append((s_all[k].name,name))
			else:
				states.append(s_all[k])
				functions.append(f)
				n_x +=1

		snames = [0 for i in range(n_x)]
		for name in scname:
			snames.append(name)
		
		varstates = []

		ku = 0

		for i in range(self.n_disc-1):
			for k,state in enumerate(states):
				varstates.append(state.name + "_" + str(i))
				snames[k] = (state.name, state.name + "_" + str(i))

			dt = self.t_disc[i+1] - self.t_disc[i]

			if self.t_disc[i] >= self.t_u[ku+1]:
				ku+=1

            
			for j in range(n_x):
				if type(sym_func[j]) == int:
					eq = str(states[j])+"_"+str(i+1) +" =e= "+ str(states[j])+ "_" +str(i) +"+("+ str(sym_func[j]*dt)+")"
				else:
					for k in range(self.model.num_inputs):
						sym_func[j] = sym_func[j].subs('u'+str(k),'u'+str(k) + '_' + str(ku))
					eq = str(states[j])+"_"+str(i+1) +" =e= "+ str(states[j])+ "_" +str(i) +"+(" + str(sym_func[j].subs(snames)*dt)+")"
					eq = eq.replace('t0','(t-'+str(self.t_u[ku])+')')

				leq.append(eq)

		return leq
		#raise NotImplementedError

	def imp_euler(self):
		"""
		Discretize the optimization problem based on implicity Euler integration
		"""
		print('implicity euler')
		raise NotImplementedError


	def gen_discr_vars(self):
		arg_symbols = symbols(['t:'+str(1), 'x:'+str(self.model.num_states),'u:'+str(self.model.num_inputs),'p:'+str(self.model.num_param)])
		sym_func = self.model.s_model(*arg_symbols)
		states = arg_symbols[1]
		inputs = arg_symbols[2]

		varstates = []
		varinputs = []
		functions = []


		
		for i in range(self.dop.num_control):
			for j in range(self.model.num_inputs):
				varinputs.append(inputs[j].name + "_" + str(i))

		ku = 0
		for i in range(self.n_disc):
			if self.t_disc[i] >= self.t_u[ku+1]:
				if ku + 1 < self.dop.num_control:
					ku += 1
			dt = self.t_disc[i] - self.t_u[ku]
			for k,state in enumerate(states):
				sy = sym_func[k]
				if type(sy) != int:
					sy = sy.subs('t0', dt)
					for j in range(self.model.num_inputs):
						sy = sy.subs(inputs[j], inputs[j].name + "_" + str(ku))

					for j in range(self.model.num_states):
						sy = sy.subs(states[j], states[j].name + "_" + str(i))
				functions.append(sy)
				varstates.append(state.name + "_" + str(i))

		return varstates, varinputs, functions

            

	def obj_fun(self, const_idx, bfunction=None):
		#Generate coefficients
		#
		states, controls, dx = self.gen_discr_vars()
		states = np.reshape(states, (self.n_disc, self.model.num_states))
		controls = np.reshape(controls, (self.dop.num_control, self.model.num_inputs))
		dx = np.reshape(dx, (self.n_disc, self.model.num_states))

		eqx = [0]*self.n_disc
		eqdx = [0]*self.n_disc
		leq = []
		obj = 0

		for i in range(self.n_disc):
		 	ku = 0
		 	fi = self.basis_function(bfunction)
		 	ti = self.t_disc[i]
		 	if i == self.n_disc-1:
		 		dt = self.t_disc[i]-self.t_disc[i-1]
		 	else:
		 		dt = self.t_disc[i+1]-self.t_disc[i]

		 	fi = fi.subs([('ti',ti), ('dt', dt), ('a', 'a_'+str(i)), ('b', 'b_'+str(i))])
		 	obj += fi
		 	dfi = fi.diff('t')

		 	for j in range(self.n_disc):
		 		eqx[j] = eqx[j] + fi.subs('t',self.t_disc[j])
		 		eqdx[j] = eqdx[j] + dfi.subs('t',self.t_disc[j])
		 	
		ku=0
		for i in range(self.n_disc):
			leq.append(str(eqx[i]) + ' =e= ' + str(states[i,const_idx]))
			
			leq.append(str(eqdx[i]) + ' =e= ' + str(dx[i,const_idx]))

		return leq, obj

	def basis_function(self, fbasis=None):
		# Interpolation basis function
		if fbasis is None:
			t, ti, dt, a, b = symbols(['t','ti','dt','a','b'])
			fbasis = (a*t + b)*exp(-(((t-ti)/dt)**2))
		return fbasis

