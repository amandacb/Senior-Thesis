================================================================================
Beat the Curve: Designing Adaptive Blood Glucose Management 
Strategies For Non-Pump Patients with Type 1 Diabetes
                       ---
                  Amanda Brown
             aclairebrown@hotmail.com
                     May 2019
                       ---


                   Senior Thesis
                Princeton University
Department of Operations Research and Financial Engineering 
              Advisor: Warren B. Powell
================================================================================

ABOUT THE CODE
-----------------

We implement a non-linear mathematical model of the T1D patient in Python, 
which generates blood glucose paths over time as a function of exogenous 
inputs (such as carbohydrate intake), insulin dose decisions, and underlying (or 
“hidden”) patient parameters.

To explore the stochastic dynamic optimization problem of blood glucose 
management for a Type 1 Diabetes patient without a pump (where the patient is 
required to makedosing decisions and determine when to measure blood glucose), 
we simulate blood glucose run paths with various decision policies, simulating 
over a number of random variables (including patient-specific parameters, 
measurement noise, and meal arrival uncertainty). 

Policies include a mix of simple rules-based policies and lookahead policies.


REQUIREMENTS
------------

 - Python 2.7
 - NumPy, SciPy
 - MatPlotLib
 - itertools, copy, time, collections

INSTALLATION
------------

No installations required. The Python scripts can be copy-and-pasted into 
a text editor and used directly.

GETTING STARTED
---------------

To begin, install Python 2.7 and the latest versions of SciPy, NumPy, etc. 

First run the simulated patient model script (BergmanMinimalModel.py). This
script contains test code to validate the outputs of the patient model. The
user can also modify the model to include different compartments to represent
different effects on the patient's blood glucose, such as exercise.


Once the patient model is out-putting results that align with the user's
expectation, the user should comment out the lines following the comment
"# TEST CODE", and then run the main driver script (DosingDriverScript.py). 

From here the user can implement different dosing policies, simulate over 
varying lengths of time, play with different tunable parameters, plot 
blood glucose run paths, etc.


WORKING WITH THE CODE
---------------------

 * DosingDriverScript.py: This is the main script that should be run by the 
 						  user in order to test policies and assumptions of 
 						  the underlying patient model and patient decision-
 						  making process.
 
 * DosePolicy.py: Contains functions for testing various dosing policies.

 * DoseModel.py: Contains functions that represent the five parts of a sequential 
 				 decision-making process: the decision variable, the state variable, 
 				 exogenous information, the transition function, and the objective 
 				 function.

 * BergmanDiabeticPatient.py: Contains functions for testing various blood glucose 
 							  management policies.


CITING THIS CODE
----------------

If you wish to use this code in a scientific paper, please cite Amanda Brown, who 
has implemented the policies and the patient model, which is based on the Bergman 
et al. 1989 patient model (and more specifically is adapted from the modified model 
contained in Schiavon et al. 2014). Please also cite Amanda's thesis advisor, 
Warren B. Powell, who has developed the five-part framework for sequential decision-
making under uncertainty, which has been applied here to the specific problem setting 
of a Type 1 Diabetes patient.

BUGS
----

Email me at aclairebrown@hotmail.com to let me know if there are problems 
with the code, or if you would like more information about the formalism
or implementation of the model.
