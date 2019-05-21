

'''
===============================================================================================================
The Policies

(Amanda Brown Senior Thesis, 2019)

 Citations:
 Code package based on sequential decision making framework outlined in Powell 2019.
 Code structured similarly to Asset Selling package (Castle Labs) and adapted from code by Donghun Lee (c) 2018.
===============================================================================================================
'''

from collections import namedtuple
from DoseModel import DoseModel
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from copy import copy
import numpy as np
from scipy import stats
from BergmanDiabeticPatient import BergmanDiabeticPatient

class DosePolicy():
    """
    Base class for decision policy
    """

    def __init__(self, model, policy_names, tau_lim):
        """
        Initializes the policy

        :param model: the base blood glucose model (5 part framework) upon which the policy is implemented
        :param policy_names: list(str) - list of policies
        :param tau_lim: vector - parameters constraining the frequency of blood glucose measurement and insulin dosing
        """
        self.tau_lim = tau_lim 
        self.model = model
        self.policy_names = policy_names
        self.Policy = namedtuple('Policy', policy_names)

    def build_policy(self, info):
        """
        this function builds the policies depending on the parameters provided

        :param info: dict - contains all policy information
        :return: namedtuple - a policy object
        """
        return self.Policy(*[info[k] for k in self.policy_names])

    def lookahead_policy(self, state, info_tuple, time,t):
        """
        This function implements the direct look ahead policy, which includes a pseudo lookahead model 
        with a discretized & limited future dosing decision space

        :param state: namedtuple - the state of the model at a given time
        :param info_tuple: tuple - contains the parameters needed to run the policy
        :return: a decision made based on the policy
        """

        measured_glucose_history =np.array(state.mG)

        # Obtain the last measured glucose level
        last_measured_G = measured_glucose_history[-1]

        # Obtain the time at which the last measurement occurred
        tau0 = state.tau[0]
        upper_limit = info_tuple[1]
        

        # select highest probability patient parameter vector
        k_select = np.argmax(state.p)
        G_pred_now = state.y0_beliefs[k_select][0]
    
        # Weighted average of last measured glucose and predicted glucose with highest probability patient parameter vector
        G_est_now = (1/tau0)*last_measured_G + ((tau0-1)/tau0)*G_pred_now
        print("G_est",G_est_now)
        print("G_last",last_measured_G)
        print("G_pred_now using best", G_pred_now)
        
        # The predicted BG level can also be computed with an expected value (see code commented out below)

        #sum_val = 0
        #for j in range(len(state.p)):
            #sum_val+=state.y0_beliefs[j][0]*state.p[j]

        #print("G_pred_now using expected val", sum_val)

        # define lookahead horizon
        H = info_tuple[2]
        print("Horizon: ",H)

        # initialize the patient status in the lookahead model (same as the base model)
        sim_y0 = state.y0
        sim_y0_beliefs = state.y0_beliefs


        G_target = info_tuple[0] #BG target level
        factor = 15 # insulin sensitivity factor (this can be tuned as well)
        dose_now = max(G_est_now - G_target,0)/factor # implement simple dosing rule

        d = dose_now
        D = [ dose_now-1, dose_now-0.5, dose_now+0, dose_now+0.5, dose_now+1] # define a set of different dosing decisions to search over
        #print("Dose Now: ", d)
        #print("Real time: ", t)
        #new_meal_dist = exog_info
       
        obj_arr = []

        # search over all possible dosing quantities over the time horizon
        for d in D:
            obj = 0
            count = 0
            if H == 0:
                count =1
            sim_y0 = state.y0
            sim_y0_beliefs = state.y0_beliefs

            for h in range(H):

                # current time step
                if (t+1+h) < len(time):
                    
                    ts = [time[t+h], time[t+1+h]]
                    #print("Sim time ts: ", ts)
                    #sim_y0 = state.y0
                    #print(sim_y0)
                    #sim_y0_beliefs = state.y0_beliefs
                    #print(sim_y0_beliefs)

                    # In the lookahead model, use dterministic approximation of meal consumption based on mean parameters
                    a = self.model.meal_vector_param[0]
                    b = self.model.meal_vector_param[1]
                    c = self.model.meal_vector_param[2]
                    inc = self.model.inc
                    meal= (a)*np.exp(-((t+c*inc)%(c*inc))*b)
                    
                    
                    patient_status = BergmanDiabeticPatient(meal,d,sim_y0,sim_y0_beliefs, ts, 0, self.model.thetas)
                    sim_y0 = patient_status.y0
                    sim_y0_beliefs = patient_status.y0_beliefs
                    G_pred_future = 0
                    for j in range(len(state.p)):
                        G_pred_future+=sim_y0_beliefs[j][0]*state.p[j]
                    G_pred_future = sim_y0_beliefs[k_select][0]
                    obj += np.abs(G_pred_future-90)
                    #print("Difference", np.abs(G_pred_future-90))


                    d = 0 # assume no additional insulin input over the horizon
                    count +=1

            obj_arr.append(obj/count)   

        print(obj_arr)

        
        G_pred_future = 0

        for j in range(len(state.p)):
            G_pred_future+=sim_y0_beliefs[j][0]*state.p[j]

        discount = 1/(H+1) # apply a discounting factor to account for uncertainty of lookahead model in the future
        G_pred_future= sim_y0_beliefs[k_select][0]
        dose_future_adj = discount*max(G_pred_future - G_target,0)/factor
        dose_amount = dose_now + dose_future_adj
        dose_amount = D[np.argmin(obj_arr)]
        print("Future Decision: ",np.argmin(obj_arr)) # obtain a dosing decision based on which minimizes the objective function

        print("Dose Amount", dose_amount)
        print("Dose Now", dose_now)

        # Implement dosing decision at time t in the base model
        new_decision = {'dose': dose_amount} if (state.tau[0]>self.tau_lim[0]) and (dose_amount) else {'dose': 0}
        if state.tau[1]>self.tau_lim[1]:
            new_decision['measure'] = 1 
        else:
            new_decision['measure'] = 0

        return new_decision

   

    def dose_high_policy(self, state, info_tuple):
        """
        The "Myopic" or Policy Function Approximation (PFA)-based policy
        :param state: namedtuple - the state of the model at a given time
        :param info_tuple: tuple - contains the parameters needed to run the policy
        :return: a decision made based on the policy
        """

        measured_glucose_history =np.array(state.mG)

        # Obtain the last measured glucose level
        last_measured_G = measured_glucose_history[-1]

        # Obtain the time at which the last measurement occurred
        tau0 = state.tau[0]
        upper_limit = info_tuple[1]
        

        # select highest probability patient parameter vector
        k_select = np.argmax(state.p)
        #k_select = 5
    

        G_pred = state.y0_beliefs[k_select][0]
        print("G_pred",G_pred)

        # Weighted average of last measured glucose and predicted glucose with highest probability patient parameter vector
        G_est = (1/tau0)*last_measured_G + ((tau0-1)/tau0)*G_pred
        print("G_est",G_est)
        print("G_last",last_measured_G)
        
        G_target = info_tuple[0]
        diff = (G_est - G_target)

        # The below ISF is generally defined in the literature relating to the Bergman Minimal Model
        in_sens_index = self.model.thetas['V_G'][k_select]*self.model.thetas['p3'][k_select]/self.model.thetas['p2'][k_select]

        ISF = in_sens_index # Choose a model-based ISF value
        ISF = 5 # Or, choose a pre-set ISF value (which can be tuned)

        IOB = state.y0_beliefs[k_select][2]
        
        print("IOB", IOB)
        print("ISF", ISF)
        print("diff", diff)

        # Dosing calculator
        dose_amount = diff/ISF - IOB

        # OR select one of the below formulations of a dosing calculation, depending on constraints
        #dose_amount = round(diff/ISF - IOB) # useful when dosing amount is constrained by integer dispensing amounts
        #dose_amount = state.CHO/state.ICR + (G_curr-G_target)/state.ISF - state.IOB


        print("Dose Amount", dose_amount)

        # Implement dosing decision
        new_decision = {'dose': dose_amount} if (state.tau[0]>self.tau_lim[0]) and (dose_amount>1) else {'dose': 0}
        

        if state.tau[1]>self.tau_lim[1]:
            new_decision['measure'] = 1 
        else:
            new_decision['measure'] = 0

        return new_decision



    def high_low_policy(self, state, info_tuple):
        """
        This function implements the high-low policy (which doses when the blood glucose falls out of range)

        It is also a naive PFA policy.

        :param state: namedtuple - the state of the model at a given time
        :param info_tuple: tuple - contains the parameters needed to run the policy
        :return: a decision made based on the policy
        """
        measured_glucose_history =np.array(state.mG)
        last_measured_G = measured_glucose_history[-1]
        
        #dose_amount = state.CHO/state.ICR + (G_curr-G_target)/state.ISF - state.IOB

        dose_amount = 2
        lower_limit = info_tuple[0]
        upper_limit = info_tuple[1]
        new_decision = {'dose': dose_amount} if last_measured_G > upper_limit else {'dose': 0}

        if state.tau[1]>self.tau_lim[1]:
            new_decision['measure'] = 1 
        else:
            new_decision['measure'] = 0

        return new_decision


    def sugar_surfing_policy(self, state, info_tuple):
        """
        This function implements the heuristic sugar surfing policy, with inspiration from Dr. Ponder's dynamic BG management approach.
        It considers the slope or trend of recent blood glucose measurements to help the patient make a dosing decision.

        :param state: namedtuple - the state of the model at a given time
        :param info_tuple: tuple - contains the parameters needed to run the policy
        :return: a decision made based on the policy
        """
        measured_glucose_history =np.array(state.mG)
        G_curr = measured_glucose_history[-1]
        slope_max = info_tuple[1]
        upper_limit = info_tuple[0]
        print("upper_lim", upper_limit)

        N = len(measured_glucose_history)
        time = np.arange(N)

        slope, intercept, r_value, p_value, std_err = stats.linregress(time,measured_glucose_history)
        print("slope",slope)
        print("intercept", intercept)
        
        dose_decision = 0
        if (slope > 10) and (intercept > 150):
            dose_decision = 1
        elif (G_curr > upper_limit):
            dose_decision = 1

        # Dosing calculator

        #dose_amount = (state.CHO-1000)/state.ICR + (G_curr-G_target)/state.ISF - state.IOB
        dose_amount = 3
        print("Dose Amount",dose_amount)

        #print("CHO",state.CHO)
        print("Dose Decision",dose_decision)

        new_decision = {'dose': dose_amount} if dose_decision == 1 else {'dose': 0}
        if state.tau[1]>self.tau_lim[1]:
            new_decision['measure'] = 1 
        else:
            new_decision['measure'] = 0

        return new_decision

    def run_policy(self,  policy_info, policy, time):
        """
        This function runs the model with a selected policy.

        :param policy_info: dict - dictionary of policies and their associated parameters
        :param policy: str - the name of the chosen policy
        :param time: float - start time
        :return: float - calculated contribution
        """
        model_copy = copy(self.model)

        tf = model_copy.initial_args['T']   # simulate for T hours
        ns = tf*4+1  # sample time = 15 min

        # Time Interval (min)
        time = np.linspace(0,tf,ns)
        p_beliefs = model_copy.state.p
        p_wins = []
        meals = []

        for t in range(len(time)-1):
            ts = [time[t],time[t+1]]
            # build decision policy
            p = self.build_policy(policy_info)

            # make decision based on chosen policy
            if policy == "dose_high":
                decision = self.dose_high_policy(model_copy.state, p.dose_high)
            elif policy == "high_low":
                decision = self.high_low_policy(model_copy.state, p.high_low)
            elif policy == "sugar_surfing":
                decision = self.sugar_surfing_policy(model_copy.state, p.sugar_surfing)
            elif policy == "lookahead":
                decision = self.lookahead_policy(model_copy.state, p.lookahead, time, t)
 

            x = model_copy.build_decision(decision)
            print("time={}, obj={}, tau={} s.Glucose={}, x={}".format(t, model_copy.objective, model_copy.state.tau[1],model_copy.state.G[t], x))

            # step the model forward one iteration
            model_copy.step(x,t,ts,model_copy.state)
            p_beliefs = np.vstack((p_beliefs,model_copy.state.p))
            meals.append(model_copy.exog_info)
            p_wins.append(np.argmax(model_copy.state.p))

        # Dictionary of information which can be called in the driver script to plot results
        history_dict = {"G": model_copy.state.G,
                        "D": model_copy.state.D,
                        "mG": model_copy.state.mG,
                        "Dt": model_copy.state.Dt,
                        "mt": model_copy.state.mt,
                        "p_beliefs": p_beliefs,
                        "p_wins": p_wins,
                        "Obj": model_copy.objective,
                        'Meals': meals,
                        'Best_Param_Set': np.argmax(model_copy.state.p)}
        
        return history_dict





