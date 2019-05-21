'''
===============================================================================================================
The Patient Model

(Amanda Brown Senior Thesis, 2019)

Implementation of the modified Bergman Minimal Model (Bergman et al. 1989) 
per the reformulation in Schiavon et al. 2014.

Notes:
Run code below to test the modified Bergman Patient Model for simple dosing strategies. When running the modified 
Bergman Minimal Model as a module within with the DosingDriverScript.py to implement/simulate a sequential 
decision-making policy, then comment out the section that begins with the comment "# TESTING CODE"
===============================================================================================================
'''

from collections import namedtuple
import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.dates as mdates

import matplotlib.pyplot as plt
from scipy.integrate import odeint

class BergmanDiabeticPatient():
	def __init__(self, meal_disturbance,insulin_decision,y0,y0_beliefs,ts,true_theta_idx,thetas, epsilon_sigma=0):
		
		# total_time_days: number of days to simulate over
		self.y0 = y0
		self.y0_beliefs = y0_beliefs
		self.d = meal_disturbance
		self.u = insulin_decision
		self.ts = ts
		self.thetas = thetas
		self.epsilon_sigma = epsilon_sigma

		def diabetic(y,t,u,d,theta):
			# u = Insulin infusion rate (micro-U/min)
			# d = Meal carbohydrate load (mg)

			# adjust i_type depending on insulin properties
			i_type = 0

			g = y[0]                # blood glucose (mg/dL)
			x = y[1]                # remote insulin (unitless)
			i = y[2]                # insulin (mU/L)
			m = y[3]				# plasma glucose appearance rate (mg)
			gga = y[4]				# gut glucose absorption (units)
			s1 = y[5]            	# amount of insulin in the primary compartment (units)
			s2 = y[6]            	# amount of insulin in the primary compartment (units)
			
			# Uncomment next line to peak into each compartment at each time step
			#print("g ", g, "x ",x,"i ",i, "m ", m, "gga ", gga, "s1 ", s1, "s2 ", s2)

			dydt = np.empty(7)
			
			dydt[0] = -theta['p1']*(g - theta['G_b']) - x*g  + m/theta['V_G']
			dydt[1] = -theta['p2']*x + theta['p3']*i
			dydt[2] = -(theta['n'])*(i-theta["I_b"]) + (s2/(theta['t_maxI']-i_type))/theta['V_I']
			dydt[3] = -(1/theta['t_maxG'])*m + (1/theta['t_maxG'])*gga
			dydt[4] = -(1/theta['t_maxG'])*gga + (theta['A_G']/theta['t_maxG'])*d 
			dydt[5] = u - (s1/(theta['t_maxI']-i_type))
			dydt[6] = +(s1/(theta['t_maxI']-i_type)) - (s2/(theta['t_maxI']-i_type))

			dydt = dydt*60
			return dydt
		
		# True patient state transfer
		true_theta = {key: thetas[key][true_theta_idx] for key in thetas.keys()}
		y = odeint(diabetic,self.y0,self.ts,args=(self.u,self.d, true_theta))
		self.y0 = y[-1]

		# Measurement of BG with error
		self.measured_glucose = y[-1][0] + np.random.normal(0,self.epsilon_sigma)

		K  = len(thetas['p1'])

		# Belief state transfer
		output_y0_beliefs = []
		for i in range(K):
			test_theta = {key: thetas[key][i] for key in thetas.keys()}
			y_test = odeint(diabetic,self.y0_beliefs[i],self.ts,args=(self.u,self.d, test_theta))

			output_y0_beliefs.append((list(y_test[-1])))

		self.y0_beliefs = output_y0_beliefs

#===============================================================================================================
# TESTING CODE

K = 10
# Use this dictionary of patient parameters (10 patients)
thetas = {
'G_b': np.array([119.66598565,  94.83500979, 129.03919567, 127.8548505 ,123.72347188, 113.34304235, 106.68958353, 118.63594336, 134.32311924, 124.24712293]), 
'I_b': np.array([0.10030511, 0.09945428, 0.09970168, 0.10179413, 0.10035751,0.09835811, 0.09915144, 0.10087668, 0.10167444, 0.09959593]), 
'p1': np.array([0.09721208, 0.10660655, 0.09346647, 0.08772787, 0.09448505,0.09911992, 0.09956666, 0.08654369, 0.10450155, 0.09678427]), 
'p2': np.array([0.03286145, 0.03489866, 0.03354611, 0.03394798, 0.03286051, 0.03223424, 0.03435307, 0.03327569, 0.0345879 , 0.0335957 ]), 
'p3': np.array([0.01272057, 0.01494392, 0.01412465, 0.01458635, 0.01362076, 0.01339473, 0.01286449, 0.01069876, 0.01280876, 0.01396004]), 
'n': np.array([0.14209142, 0.13946169, 0.14200921, 0.14107541, 0.14248514, 0.14147655, 0.14157267, 0.14164257, 0.14304488, 0.14133998]), 
'A_G': np.array([0.8532165 , 0.8570972 , 0.85370901, 0.86861943, 0.85703812, 0.8431248 , 0.85743703, 0.85033163, 0.85447995, 0.83129563]), 
't_maxG': np.array([39.74566118, 55.2151079 , 54.05251437, 37.40561983, 53.55175915, 56.53191891, 43.05356551, 38.23173577, 56.14042086, 59.5801979 ]), 
't_maxI': np.array([41.82891538, 51.68815919, 44.55646918, 50.30334456, 51.02513857, 59.70092016, 45.74594717, 54.84808715, 55.82262399, 65.62384024]), 
'V_I': np.array([11.30296157, 10.5520986 ,  9.3468061 , 16.8412927 ,  9.51182073, 12.23195233, 12.77892532, 13.65248102, 12.01387684, 13.82673879]), 
'V_G': np.array([10.20778191,  9.51816104, 12.44337107, 12.83219405, 10.47882316, 11.19817353, 10.22387081, 13.09765845, 11.14755103, 11.43251341])}



# Or generate your own set of patients by drawing from a normal distribution:

# Below is a parameter dictionary containing the arguments of a N distribution: (mu_i and sigma_i ) for each parameter element i  
'''

params = {

	'G_b'    : [120,      1e-8],          # basal Blood Glucose (mg/dL)
	'I_b'    : [0.1,        1e-8],          # basal Insulin (mU/mL)
	'p1'     : [0.1,   1e-8],          # (1/min)
	'p2'     : [3.33e-2,   1e-8],          # (1/min)
	'p3'     : [1.33e-2,   1e-8],          # (1/min)
	'n'      : [0.142,   1e-8],          # time constant for insulin disappearance (1/min)
	'A_G'    : [0.85,    1e-8],          # carbohydrate availability (unitless)
	't_maxG' : [50,       1e-8],          # time to maximum glucose absorption (1/min)
	't_maxI' : [50,       1e-8],          # time to maximum insulin absorption (1/min)
	'V_I'    : [12,      1e-8],          # volume of insulin distribution (L)
	'V_G'    : [12,      1e-8]           # gut volume (L)

}

thetas = {k: [] for k in params.keys()}

# Create a dictionary of thetas, each element drawn from its normal distribution
for key in thetas:
	thetas[key] = np.random.normal(params[key][0],params[key][1],K)
'''

# Create an array (size = K) which represents the probability of each candidate vector being the true parameter vector
# Initialize with a uniform prior distribution
init_p = np.ones(K)*(1/K)
init_y0 = np.tile(np.array([100, 0, 0,10,50,1,1]), (K,1))
# Final Time (hr)
tf = 24   # simulate for 24 hours
freq = 4 # evolution frequency of the simulation (how many times per hour)
ns = tf*4+1  # sample time = 15 min
f,axarr = plt.subplots(3,sharex = True)
for j in range(K):

	
	t = np.linspace(0,tf,ns) # incremental time interval (in minutes)

	G = [100] # initial glucose (mg/dL)
	I = [0] # initial insulin concentration in main compartment
	M = [0] # initial meal disturbance


	y0 = np.array([100, 0,0,10,50,1,1])

	for i in range(len(t)-1):
		
		# Insert Dosing Rule Here
		#if ((i+5)%20 == 0 and j==0) or ((i+7)%20 == 0 and j==1) or ((i+1)%20 == 0 and j==3):
		
			#insulin_decision =round(G[-1]*0.2/10)*10
		#if (i)%12 == 0:
		if G[-1]>90 and (i)%6 == 0:
			ISF = 15 # insulin sensitivity factor
			TargetG = 90 # target glucose (mg/dL)
			insulin_decision = (G[-1]-TargetG)/ISF

		else:
			insulin_decision = 0

		# Insert Meal Arrival Pattern Here
		if (i+10)%12 == 0:
			meal_disturbance = 1500
		else:
			meal_disturbance = 0


		print("Insulin Dosed",insulin_decision)

		ts = [t[i],t[i+1]] # select time interval to insert into the modified Bergman patient model
		epsilon_sigma = 10

		patient_status = BergmanDiabeticPatient(meal_disturbance,insulin_decision,y0,init_y0, ts,j,thetas, epsilon_sigma)
		print("y0", patient_status.y0)
		y0 = patient_status.y0
		G.append(y0[0])
		I.append(insulin_decision>0)
		M.append(meal_disturbance>0)
		
		print("Measured Glucose",patient_status.measured_glucose)


	# Plot Results

	f.suptitle("{}-hour Blood Glucose Evolution for {} Different Parameter Sets (Fixed Meal Consumption & Naive Insulin Dose Rule)".format(tf,K))

	axarr[0].plot(np.arange(len(t))/4, G, label="$\\theta_{%i}$" %(j+1))
	axarr[0].set_ylabel("mg/dL")
	axarr[0].set_title("Blood Glucose")
	axarr[0].axhline(110, c="k", linewidth=0.7)

	axarr[1].plot(np.arange(len(t))/4, M, c="k")
	axarr[1].set_title("Carbohydrate Load")
	axarr[1].set_ylabel("Meal")
	axarr[1].set_ylim(-1,2)
	axarr[1].set_yticks([0,1])
	axarr[1].set_yticklabels(["No", "Yes"])

	axarr[2].scatter(np.arange(len(t))/4, I, c="k")
	axarr[2].set_title("Insulin")
	axarr[2].set_ylabel("Dose")
	axarr[2].set_ylim(-1,2)
	axarr[2].set_yticks([0,1])
	axarr[2].set_yticklabels(["No", "Yes"])
	axarr[2].set_xlabel("Time (hours)")

	plt.subplots_adjust(left=0.12, bottom=0.12, right=0.89, top=0.78, wspace=0.31, hspace=0.65)
  
print(thetas)
plt.show()


