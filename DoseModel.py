
'''
===============================================================================================================
The Decision Process Model

(Amanda Brown Senior Thesis, 2019)

 Citations:
 Code package based on sequential decision making framework outlined in Powell 2019.
 Code structured similarly to Asset Selling package (Castle Labs) and adapted from code by Donghun Lee (c) 2018.
===============================================================================================================
'''

from collections import namedtuple
import numpy as np
from scipy.stats import norm
from BergmanDiabeticPatient import BergmanDiabeticPatient

class DoseModel():
    """
    Base class for model
    """
    def __init__(self, thetas, true_theta_idx, epsilon_sigma, state_variable, decision_variable, init_state,T, inc,meal_vector_param):
        """
        Initializes the model

        :param thetas: 2d - vector of patient parameter dictionary
        :param true_theta_idx: int - index of the true hidden patient in the parameter dictionary
        :param epsilon_sigma: float - measurement error
        :param T: int - total simulation time
        :param inc: int - time increments/granularity in the simulation per hour (e.g. inc = 4 means that one model step is 15 minutes)
        :param meal_vector_param: vector - baseline means of meal intake parameters, which will be disturbed by random (exogenous) information

        :param state_variable: list(str) - state variable dimension names
        :param decision_variable: list(str) - decision variable dimension names
        :param exog_info_fn: function - calculates relevant exogenous information
        :param transition_fn: function - takes in decision variables and exogenous information to describe how the state
               evolves
        :param objective_fn: function - calculates contribution at time t
        """


        self.initial_args = {'T': T}


        self.thetas = thetas
        self.epsilon_sigma = epsilon_sigma

        self.meal_vector_param = meal_vector_param

        # Parameters are drawn from a normal distribution (to represent slight meal intake variability among patients)
        self.a = np.random.normal(meal_vector_param[0],meal_vector_param[0]*.1,1)
        self.b = np.random.normal(meal_vector_param[1],meal_vector_param[1]*.1,1)
        self.c = np.random.normal(meal_vector_param[2],meal_vector_param[2]*.1,1)


        self.true_theta_idx = true_theta_idx
        self.state_variable = state_variable
        self.decision_variable = decision_variable
        self.State = namedtuple('State', state_variable)
        self.state = self.build_state(init_state)
        self.Decision = namedtuple('Decision', decision_variable)
        self.objective = 0.0
        self.meal_dist_vector = []
        self.exog_info = 0

        self.inc = inc

    def build_state(self, info):
        """
        this function gives a state containing all the state information needed

        :param info: dict - contains all state information
        :return: namedtuple - a state object
        """
        return self.State(*[info[k] for k in self.state_variable])

    def build_decision(self, info):
        """
        this function gives a decision

        :param info: dict - contains all decision info
        :return: namedtuple - a decision object
        """
        return self.Decision(*[info[k] for k in self.decision_variable])


    def exog_info_fn(self,t,inc):
        """
        this function gives the exogenous information that is dependent on a random process

        in the case of the blood glucose management problem, it is the disturbance in blood glucose due to consumption of carbohydrates

        :return: dict - updated glucose and carbohydrate queue
        """

        #parameters of the exponential meal model

        a = self.a
        b = self.b
        c = self.c

        
        new_meal_dist= (a)*np.exp(-((t+1+c*inc)%(c*inc))*b)
        
        return new_meal_dist

    def transition_fn(self, ts, decision, exog_info, state, t):
        """
        this function takes in the decision and exogenous information to update the state

        :param decision: namedtuple - contains all decision info
        :param exog_info: any exogenous info
        :param state: state variable information
        :param ts: time interval to model the patient evolution
        :param t: current time in the model
        :return: dict - updated glucose and posterior distribution on theta

        """
        y0 = state.y0
        y0_beliefs = state.y0_beliefs
        new_meal_dist = exog_info
        #print("transition dose: ", decision.dose)

        patient_status = BergmanDiabeticPatient(new_meal_dist,decision.dose,y0,y0_beliefs, ts, self.true_theta_idx, self.thetas, self.epsilon_sigma)
        
        # update patient status vector
        new_y0 = patient_status.y0
        
        new_y0_beliefs = patient_status.y0_beliefs
        
        #print("new_y0",new_y0 )
 
        # Update patient glucose history according to model
        next_glucose = np.array(new_y0[0])
        new_G = np.append(state.G,next_glucose)

        # Update history of glucose measurements if a decision is made to measure
        new_measure = patient_status.measured_glucose

        if decision.measure == 1:


            # Update beliefs about p by computing posterior distribution
            K = np.size(state.p)

            for k in range(K):
                print(state.p[k])
            
            new_p = np.ones(K)
            denom = 0
            for k in range(K): 
                numer = norm.pdf(new_measure,new_y0_beliefs[k][0],self.epsilon_sigma)*state.p[k]
                denom += numer
                new_p[k] = numer


                #Update candidate y0 vectors with the measured glucose (deterministic)
                new_y0_beliefs[k][0] = new_measure
                print("new_y0_beliefs we see",new_y0_beliefs)
            new_p = new_p/denom
            print('new_p', new_p)
            
            new_tau1 = 1

            # Update vector of measured blood glucose
            new_mG = np.append(state.mG,new_measure)
            new_mt = np.append(state.mt, state.t)
        else:
            new_p = state.p
            new_tau1 = state.tau[1] + 1
            new_mG = state.mG
            new_mt = state.mt


        # Update dosing history
        if decision.dose<=0:
            new_tau0 = state.tau[0] + 1



        else:
            new_tau0 = 1
        new_Dt = np.append(state.Dt, state.t)
        new_D = np.append(state.D,decision.dose)



        # Update time
        new_t = state.t + 1


        state = self.build_state({'p': new_p, 'G': new_G, 'mG': new_mG, 'mt': new_mt, 'D':new_D, 'Dt': new_Dt, 'tau': [new_tau0,new_tau1], 'y0': new_y0, 'y0_beliefs':new_y0_beliefs, 't': new_t})
        return state

    def objective_fn(self, decision, exog_info, state):
        """
        this function calculates the contribution, which depends on the decision and glucose levels

        :param decision: namedtuple - contains all decision info
        :param exog_info: any exogenous info
        :param state: state variable information
        :return: float - calculated contribution
        """

        a1, a2, a3, a4 = [0.4, 0.6, 10, 1]
        upper_bound = 100
        lower_bound = 80
        glucose = state.G[-1]
        measure_penalty = decision.measure
        dose_penalty = decision.dose

        # Pick from a variety of objective function contributions

        #obj_part = a1*max(glucose - upper_bound, 0) + a2*max(lower_bound - glucose,0) + a3*measure_penalty + a4*dose_penalty
        #obj_part = (glucose-90)**2
        #obj_part = a1*np.max([glucose - upper_bound, 0]) + a2*np.max([lower_bound - glucose,0])
        obj_part = np.abs(glucose-90)
        
        print("objective value1", a1*max(glucose - upper_bound, 0))
        print("objective value2", a2*max(lower_bound - glucose,0))
        
        return obj_part

    def step(self, decision,t,ts, state):
        """
        this function steps the process forward by one time increment by updating the sum of the contributions, the
        exogenous information and the state variable

        :param decision: namedtuple - contains all decision info
        :param t: float - contains time
        :param ts: namedtuple - contains time segment info
        :param state: namedtuple - contains all state variable info
        :return: none
        """

        self.exog_info = self.exog_info_fn(t, self.inc)
        self.state = self.transition_fn(ts, decision, self.exog_info, state, t)
        self.objective += self.objective_fn(decision, self.exog_info, state)

