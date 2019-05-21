
'''
===============================================================================================================
The Driver Script

(Amanda Brown Senior Thesis, 2019)

 Citations:
 Code package based on sequential decision making framework outlined in Powell 2019.
 Code structured similarly to Asset Selling package (Castle Labs) and adapted from code by Donghun Lee (c) 2018.
===============================================================================================================
'''

from collections import namedtuple
import itertools
import numpy as np
from DoseModel import DoseModel
from DosePolicy import DosePolicy
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from copy import copy
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import scipy.stats

if __name__ == "__main__":


    # Set number of simulation experiments to run 
    budget = 2

    # Parameter vector describing the frequency of glucose measurement (in 15 minute increments)
    tau_glucose = np.arange(4,13)

    # Parameter vector describing the frequency of insulin dosing (in 15 minute increments)
    tau_insulin = np.arange(4,13)


    tunable_params = [
        tau_glucose,
        tau_insulin
    ]

    #convert to hours if desired
    tunable_params_hr = [
        tau_glucose/4,
        tau_insulin/4
    ]
    combos = list(itertools.product(*tunable_params))
    combo_hours = list(itertools.product(*tunable_params_hr))


    m_tun_param,n_tun_param  = np.shape(combos)

    # Initialize an array to store beliefs about the various tunable parameters (mean and standard deviation)
    mu_hat = np.array([1500]*m_tun_param)
    sigma = 75
    rho = 0.8

    cov_hat = np.zeros((m_tun_param,m_tun_param))
    for i in range(m_tun_param):
        for j in range(i,m_tun_param):
            if i != j:
                cov_hat[i][j] = sigma*sigma*np.exp(-rho*(np.absolute(combos[i][0]-combos[j][0])+np.absolute(combos[i][1]-combos[j][1])))
                cov_hat[j][i] = cov_hat[i][j]
            else:
                cov_hat[i][j] = sigma*sigma

    x_star_init = 0

    batch_mean_list = []
    x_star = x_star_init
    chosen_param_list = []
    chosen_x_star = []


    # Look ahead horizon range (for comparing policies)
    H_range = [4,8]



    #f1,bigaxarr = plt.subplots(3,sharex = True)
    
    #fig1, axarr = plt.subplots(111,sharex = True)
    #fig20 = plt.figure(20)

    patient_obj = np.zeros((len(H_range), budget))
    patient_std = []

    patient_loss_obj = np.zeros((len(H_range), budget))
    patient_loss_std = []

    for z in range(budget):
        obj_val_array = []
        obj_array_all = np.zeros(len(H_range))
        std_array_all = np.zeros(len(H_range))

        # Comment this out when optimizing x_star
        x_star = 0

        # Maintain list of selected parameter combinations when tuning
        chosen_x_star.append(x_star)
        chosen_param_list.append(combos[x_star])

        chosen_param = combos[x_star]

        # Time horizon in hours
        T = 24

        # Increments (i.e. simulation evolution frequency per hour)
        inc = 4

        # Create a dictionary of candidate theta vectors, which will be used to learn the parameters of the diabetic patient

        # K: number of candidate parameter vectors to be sampled
        K = 10

        
        '''
        Uncomment this section to randomly draw a dictionary of patients
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
            print(key)
            thetas[key] = np.random.normal(params[key][0],params[key][1],K)
        for key in thetas:
            print(key, thetas[key])
        '''
        
        # Patient dictionary 

        thetas = {'G_b': np.array([119.66598565,  94.83500979, 129.03919567, 127.8548505 ,123.72347188, 113.34304235, 106.68958353, 118.63594336,
       134.32311924, 124.24712293]), 'I_b': np.array([0.10030511, 0.09945428, 0.09970168, 0.10179413, 0.10035751,
       0.09835811, 0.09915144, 0.10087668, 0.10167444, 0.09959593]), 'p1': np.array([0.09721208, 0.10660655, 0.09346647, 0.08772787, 0.09448505,
       0.09911992, 0.09956666, 0.08654369, 0.10450155, 0.09678427]), 'p2': np.array([0.03286145, 0.03489866, 0.03354611, 0.03394798, 0.03286051,
       0.03223424, 0.03435307, 0.03327569, 0.0345879 , 0.0335957 ]), 'p3': np.array([0.01272057, 0.01494392, 0.01412465, 0.01458635, 0.01362076,
       0.01339473, 0.01286449, 0.01069876, 0.01280876, 0.01396004]), 'n': np.array([0.14209142, 0.13946169, 0.14200921, 0.14107541, 0.14248514,
       0.14147655, 0.14157267, 0.14164257, 0.14304488, 0.14133998]), 'A_G': np.array([0.8532165 , 0.8570972 , 0.85370901, 0.86861943, 0.85703812,
       0.8431248 , 0.85743703, 0.85033163, 0.85447995, 0.83129563]), 't_maxG': np.array([39.74566118, 55.2151079 , 54.05251437, 37.40561983, 53.55175915,
       56.53191891, 43.05356551, 38.23173577, 56.14042086, 59.5801979 ]), 't_maxI': np.array([41.82891538, 51.68815919, 44.55646918, 50.30334456, 51.02513857,
       59.70092016, 45.74594717, 54.84808715, 55.82262399, 65.62384024]), 'V_I': np.array([11.30296157, 10.5520986 ,  9.3468061 , 16.8412927 ,  9.51182073,
       12.23195233, 12.77892532, 13.65248102, 12.01387684, 13.82673879]), 'V_G': np.array([10.20778191,  9.51816104, 12.44337107, 12.83219405, 10.47882316,
       11.19817353, 10.22387081, 13.09765845, 11.14755103, 11.43251341])}
       

        # Create an array (size = K) which represents the probability of each candidate vector being the true parameter vector
        # Initialize with a uniform prior distribution
        init_p = np.ones(K)*(1/K)

        # Initial measured glucose (mg/dL)
        init_mG = np.array([100])

        # Initial true model glucose (mg/dL)
        init_G = np.array([100])

        # Initial time glucose is measured
        init_mt = np.array([0])

        # Initial time insulin is dosed
        init_Dt = np.array([0])


        # Initialize diabetic patient model status
        init_y0 = np.tile(np.array([100, 0, 0,10,50,1,1]),(K,1))
        
        nIterations = 3
        printStep = 5
        printIterations = [0]
        printIterations.extend(list(reversed(range(nIterations-1,0,-printStep))))  
       
        # initialize the model and the policy
        policy_names = ['dose_high', 'high_low', 'sugar_surfing', 'lookahead']

        # p: probabilities associated with each candidate \theta vector
        # G: rolling 24hr glucose history
        # mG: vector of measured blood glucose values
        # mt: vector of times at which a BG measurement is taken
        # D: insulin dosing history
        # Dt: insulin dosing time vector (for plotting purposes)
        # tau: vector of time since last dose and glucose measurement
        # y0: diabetic model status
        # y0_beliefs: model status of different candidate patients (who might be in the simulator) 
        # t: time

        state_names= ['p','G','mG', 'mt', 'D','Dt','tau','y0','y0_beliefs','t']
        
        decision_names = ['dose','measure']

        a = 2000 # average meal size in carbohydrates (mg)
        b = 2
        c = 4 # how often the patient eats a full meal (hours)

        
        meal_vector_param = [a,b,c]
        init_state = {'p': init_p, 'G':init_G, 'mG': init_mG, 'mt': init_mt, 'D':[0], 'Dt': init_Dt,'tau': [1,1], 'y0':init_y0[0],'y0_beliefs':init_y0,'t': 0}

        # randomly select a patient from the dictionary to place into the simulator 
        true_theta_idx = np.random.randint(0,K-1,1)

        # glucose monitor error (standard deviation)
        epsilon_sigma = 20 # mg/dL


        #constraints on frequency of BG measurement & insulin dosing as prescribed by initial policy parameters set
        tau_lim=[chosen_param[1],chosen_param[0]]
    
        # Creat model and policy objects
        M = DoseModel(thetas, true_theta_idx[0], epsilon_sigma,state_names, decision_names, init_state, T, inc, meal_vector_param)
        P = DosePolicy(M, policy_names, tau_lim)
        t = 0


    ##############################################################################################################################

        policy_selected = 'lookahead'

        

        f1,bigaxarr = plt.subplots(3,sharex = True)
        q=0
        
        for h_param in H_range:
        #for z in range(budget):
        #for policy_selected in policies_selected:
            # make a policy_info dict object
            policy_info = {'dose_high': [90,150],
                       'high_low': [100,100], #[target, upper limit]
                       'sugar_surfing': [180,0.5],
                       'lookahead': [90,150, h_param], #[target, upper limit, time horizon]
                       }
            start = time.time()
            history_dict=[P.run_policy(policy_info, policy_selected, t) for ite in list(range(nIterations))]
            
            for i in range(nIterations):
                val2 = h_param/inc
                bigaxarr[0].plot(np.arange(T*inc)/inc, np.array(history_dict[i]['G'])[1:],linestyle='-', linewidth = 0.7)
                bigaxarr[0].scatter(np.array(history_dict[i]['mt'])/inc, history_dict[i]['mG'], s=3)
                bigaxarr[1].plot(np.arange(T*inc)/inc, np.array(history_dict[i]['Meals']),linestyle='-', linewidth = 0.7)
                #bigaxarr[2].plot(np.array(history_dict[i]['Dt'])/inc, history_dict[i]['D'],linestyle=':')
                bigaxarr[2].scatter(np.array(history_dict[i]['Dt'])/inc, history_dict[i]['D'], s=3, label="H = %.2f" %val2)
                
                val = chosen_param[0]/inc
            bigaxarr[0].set_title("Blood glucose level (measurement frequency parameter $\\tau_{measure,lim}$ = %.2f hours)" %val)
            
            bigaxarr[0].set_ylabel("mg/dL")
            #bigaxarr[0].set_ylim(0,300)
            bigaxarr[1].set_title("Meal Carbohydrate Load")
            bigaxarr[1].set_ylabel("mg")
            val1 = chosen_param[1]/inc
            #bigaxarr[2].set_title("Insulin dosing with lookahead policy (frequency parameter $\\tau_{dose,lim}$ = %.2f)" %(val1))
            bigaxarr[2].set_title("Insulin dosing")
            
            bigaxarr[2].set_title("Insulin dosing with lookahead policy (frequency parameter $\\tau_{dose,lim}$ = %.2f hours and horizon H = %.2f hours )" %(val1,val2))
            bigaxarr[2].set_xlabel("time (hours)")
            bigaxarr[2].set_ylabel("units")
            bigaxarr[2].set_ylim(0.3,7)
            f1.subplots_adjust(left=0.12, bottom=0.12, right=0.89, top=0.88, wspace=0.2, hspace=0.47)
            
            

            '''
            f, axarr = plt.subplots(2,2)
            for i in range(nIterations):
                axarr[0,0].scatter(np.array(history_dict[i]['mt'])/inc, history_dict[i]['mG'],label="Measured Glucose", s=3)
            axarr[0,0].set_title("Glucose plot")
            axarr[0,0].set_xlabel("Time (hours)")
            axarr[0,0].set_ylabel("Blood Glucose (mg/dL)")
            axarr[0,0].set_ylim(0,300)
            plt.legend()

            
            for i in range(nIterations):
                axarr[0,1].scatter(np.array(history_dict[i]['Dt'])/inc, history_dict[i]['D'], s=3)
            axarr[0,1].set_title("Dosing Schedule with Policy Parameters:( {} )".format([chosen_param[0],chosen_param[1],policy_selected]))
            axarr[0,1].set_xlabel("Time (hours)")
            axarr[0,1].set_ylabel("Insulin Dose Quantity (Units)")
            axarr[0,1].set_ylim(1,10)

            print("Shape Meals: ", np.shape(history_dict[0]['Meals']))
            print("Shape T*inc: ", T*inc)

            for i in range(nIterations):
                axarr[1,0].plot(np.arange(T*inc)/inc, history_dict[i]['Meals'],c="k")
            axarr[1,0].set_title("Meal")
            axarr[1,0].set_xlabel("Time (hours)")
            axarr[1,0].set_ylabel("Meal Carbohydrate Load (mg)")

            

            
            K = np.shape(history_dict[0]['p_beliefs'])[1]
            L = np.shape(history_dict[0]['p_beliefs'])[0]

            loss_function = 0
            for i in range(nIterations):
                for j in range(L):
                    axarr[1,1].plot(np.arange(K), history_dict[i]['p_beliefs'][j])
                axarr[1,1].axvline(history_dict[i]['Best_Param_Set'], color='k', linestyle='--')
                if (history_dict[i]['Best_Param_Set'] != true_theta_idx[0]):
                    loss_function += 1
                print(loss_function)
            axarr[1,1].axvline(true_theta_idx[0], color='r', linestyle='--', label = "True Parameter Set $\\theta^*$")

            axarr[1,1].set_title("Model Parameter Beliefs")
            axarr[1,1].set_xlabel("k")
            axarr[1,1].set_ylabel("Posterior Distribution Likelihood")
            '''

            
            K = np.shape(history_dict[0]['p_beliefs'])[1]
            L = np.shape(history_dict[0]['p_beliefs'])[0]
            print("BELIEFS", history_dict[0]['p_beliefs'])
            max_val = 0
            

            # Plot evolution of parameter beliefs (3D graph)
            fig4 = plt.figure()
                
            ax4 = fig4.add_subplot(111, projection='3d')
            for i in range(nIterations):
                
                for j in range(L):
                    ax4.plot(np.ones(K)*j/inc, np.arange(K), history_dict[i]['p_beliefs'][j])
                    prob = max(history_dict[i]['p_beliefs'][j])
                    max_val = max(max_val, prob)
                ax4.set_xlabel("Time (hours)")
                ax4.set_ylabel("$\\theta_k$")
                ax4.set_zlabel("Posterior Likelihood")
            X, Z = np.mgrid[0:T, 0:2*max_val:2*max_val/2]
            Y = true_theta_idx[0]*np.ones((T, 2))

            ax4.set_title("Beliefs Over Time")
            
            

            plt.tight_layout()
            plt.subplots_adjust(left=0.12, bottom=0.12, right=0.89, top=0.78, wspace=0.31, hspace=0.65)


            obj_array = []
            for i in range(nIterations):
                print("Obj: ",history_dict[i]['Obj']/(T*inc))
                obj_array.append(history_dict[i]['Obj']/(T*inc))
            
            print("objective for policy {}: ".format(policy_selected),obj_array)
            
            batch_mean = np.mean(obj_array)
            
            obj_array_all[q] += batch_mean
            
            batch_std = np.std(obj_array)
            
            std_array_all[q] += batch_std
            



            print("p_wins", history_dict[i]['p_wins'])
            M = len(history_dict[i]['p_wins'])
            loss_array = history_dict[i]['p_wins']-np.ones(M)*true_theta_idx

            patient_obj[q,z] += history_dict[i]['Obj']/(T*inc)
            patient_loss_obj[q,z] += np.count_nonzero(loss_array)/(T*inc)

            
            q+=1


            # Update Beliefs for Tuning Parameters
            print("Mean: ",batch_mean,", Std: ",batch_std)
            e = np.zeros(m_tun_param)
            e[x_star] = 1
            print("x_star", x_star)
            
            mu_hat = np.add(mu_hat,(batch_mean-mu_hat[x_star])/(batch_std**2+cov_hat[x_star][x_star])*np.matmul(cov_hat,np.transpose(e)))
            cov_hat = np.add(cov_hat,-cov_hat*np.outer(np.transpose(e),e)*cov_hat/(batch_std**2 +cov_hat[x_star][x_star]))
            print("mu_hat", mu_hat)
            print("cov_hat: ", cov_hat)
            

            # Figure for plotting tuning process
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            x = tau_glucose
            y = tau_insulin
            nX = len(x)
            nY = len(y)

            X, Y = np.meshgrid(x/inc,y/inc)
            Z = np.zeros((nX,nY))

            for i in range(nX):
                for j in range(nY):
                    print(nX*i + j)
                    print(tau_glucose[i],tau_insulin[j])
                    Z[i,j] = mu_hat[nX*i + j]
            surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                                   linewidth=0, antialiased=False)
            ax.set_xlabel('$\\tau_{measure,lim}$ (hours)')
            ax.set_ylabel('$\\tau_{dose,lim}$ hours')
            ax.set_zlabel('Objective Value ($\\bar{\\mu}$)')
            ax.set_title("Tuning Parameters for Blood Glucose Control")
            
            batch_mean_list.append(batch_mean)
            

            #plt.xlabel("Objective Function")
            
               
            x_star_new = np.argmin(mu_hat + 10*(np.diag(cov_hat)**0.5))
            print("mu_hat", mu_hat)
            print("mu_hat plus dev: " ,mu_hat + 0.1*(np.diag(cov_hat)**0.5))
            x_star_new = np.random.choice(m_tun_param,1)[0]
            x_star = x_star_new
            end = time.time()
            print("{} secs".format(end - start))
        #obj_val_array.append(obj_per_h/budget)
        #print(np.array(obj_array_all)/budget)
        
        fig = plt.figure()
        for z in range(len(obj_array_all)):
            if z==0:
                intervals = scipy.stats.norm.interval(0.95,obj_array_all[z], std_array_all[z]/np.sqrt(nIterations))
                plt.plot(np.array([H_range[z]/4,H_range[z]/4]),intervals, c='k', label="95% Confidence Interval")

            intervals = scipy.stats.norm.interval(0.95,obj_array_all[z], std_array_all[z]/np.sqrt(nIterations))
            plt.scatter(H_range[z]/4,obj_array_all[z], s=10, c='k')
            print(intervals)
            plt.plot(np.array([H_range[z]/4,H_range[z]/4]),intervals, c='k')
        plt.xlabel(" 'H' Parameter Value (hours)")
        plt.ylabel("Objective Function Value")
        plt.title("Blood Glucose Control vs. Look-ahead Policy Horizon Parameter 'H' ")
        plt.legend()
        
        
        N = budget
        print(patient_obj)
        diff = patient_obj[0]-patient_obj[1]
        print(np.mean(diff))
        mean = np.mean(diff)
        variance = (1/(N-1)*np.var(diff))
        print(variance)
        #print(np.std(diff))
        print(patient_obj[0]-patient_obj[1])
        print(scipy.stats.norm.interval(0.95,mean, variance**0.5))


        print("Loss Function: ")
        print(patient_loss_obj)
        diff_loss = patient_loss_obj[0]-patient_loss_obj[1]
        print(np.mean(diff_loss))
        mean_loss = np.mean(diff_loss)
        variance_loss = (1/(N-1)*np.var(diff_loss))
        print(variance_loss)
        #print(np.std(diff))
        print(patient_loss_obj[0]-patient_loss_obj[1])
        print(scipy.stats.norm.interval(0.95,mean_loss, variance_loss**0.5))
        

        
        bigaxarr[0].axhline(70, color='k', linestyle=':', linewidth = 0.5)
        bigaxarr[0].axhline(110, color='k', linestyle=':', linewidth = 0.5,label="Acceptable glucose range: 70 mg/dL - 110 mg/dL")
        bigaxarr[0].axhline(90, color='k', linestyle='-', linewidth = 0.7,label="Glucose target: 90 mg/dL")
        bigaxarr[0].legend()
        bigaxarr[2].legend()
        f1.suptitle("Patient #{}, Duration of each run: {} hours".format(true_theta_idx[0],T ) )
        

        print("Curr Obj Array", obj_array_all)

   

    
    #print("batch_mean_list: ", batch_mean_list)
    #print("chosen_param_list: ", chosen_param_list)
    
    '''
    f,ax = plt.subplots(1,sharex = True)
    ax.hist(chosen_x_star, m_tun_param)
    f.suptitle("Histogram of Parameter Combination Selection ($x^*$)")
    ax.set_xlabel('($\\tau_{measure,lim}$, $\\tau_{dose,lim}$) (hours)')
    ax.set_ylabel('Number of Times Selected')
    ax.set_xticks(np.arange(m_tun_param))
    ax.set_xticklabels(combo_hours, fontsize=6)
    ax.tick_params(axis='x', rotation=85)
    plt.subplots_adjust(left=0.12, bottom=0.17, right=0.89, top=0.78, wspace=0.2, hspace=0.2)
    '''

    plt.show()


    















