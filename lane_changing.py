import numpy as np
import pandas as pd
from scipy.stats import beta



class lane_changing:
    """
    Class to simulate road
    Args:
        l (int): road lenght
    """

    def __init__(self, params, y_A=None):

        self.m_values = params["m_values"] 
        self.a_values = params["a_values"] 
        self.states   = params["s_values"] 


        self.y_A = y_A
        self.prob_s = params["prob_s"]
        self.prob_s_aux = np.copy(self.prob_s)
        self.prob_s_aux[self.prob_s_aux == 0] += 0.0001
        self.k = params["k"]
        self.mean_prob_correct_sensor = params["mean_prob_correct_sensor"]

        self.weights_MV = params["weights_MV"]
        self.consequences_MV = params["consequences_MV"] 
        self.rho_MV = params["rho_MV"]

        self.weights_A = params["weights_A"]
        self.consequences_A = params["consequences_A"] 
        self.rho_A = params["rho_A"]

        self.N = params["N"]


        ##

    def generate_mv_utility(self):

        utility_MV = np.zeros([len(self.m_values), len(self.states)])

        w_MV   = np.random.dirichlet(self.weights_MV * self.k)
        rho_MV = np.random.uniform(self.rho_MV - 0.1, self.rho_MV + 0.1)

        for m in self.m_values:

            states = self.consequences_MV.iloc[
                    self.consequences_MV.index.get_level_values('Action') == m]

            z = np.dot(states, w_MV)
            utility_MV[m] = 1.0 - np.exp(z * rho_MV)

        return utility_MV
    


    def generate_mv_prob(self):

        # Posterior pavement state given sensor reading
        prob_theta_MV = np.zeros(2)

        a = self.k * self.mean_prob_correct_sensor
        b = self.k * (1 - self.mean_prob_correct_sensor)

        if self.y_A == 0:

            prob_theta_MV[0] = np.random.beta(a, b)
            prob_theta_MV[1] = 1 - prob_theta_MV[0]

        else:

            prob_theta_MV[1] = np.random.beta(a, b)
            prob_theta_MV[0] = 1 - prob_theta_MV[1]

        # Posterior over states given actions
        prob_s_MV = np.apply_along_axis(
                            lambda x: np.random.dirichlet(
                            x*self.k), 3, self.prob_s_aux)


        return prob_theta_MV, prob_s_MV

    def generate_mv_exp_utility(self, a):

        exp_utility_MV = np.zeros(len(self.m_values))

        prob_theta_MV, prob_s_MV = self.generate_mv_prob()
        utility_MV = self.generate_mv_utility()

        for m in self.m_values:
            p = np.dot(  prob_s_MV[:, a, m].T, prob_theta_MV )
            exp_utility_MV[m] = np.dot(p, utility_MV[m])

        return exp_utility_MV

    def compute_p_m_a(self):

        pam = np.zeros([ len(self.a_values), len(self.m_values) ])

        for i, a in enumerate(self.a_values):

            for j in range(self.N):

                EU = self.generate_mv_exp_utility(a)
                id_max = np.random.choice(np.flatnonzero(EU == EU.max()))
                #id_max = np.argmax(EU)
                pam[a, id_max] += 1/self.N

        return pam

    def utility_A(self, a):

        states = self.consequences_A.iloc[
                self.consequences_A.index.get_level_values('Action') == a]

        z = np.dot(states, self.weights_A)
        return 1.0 - np.exp(z * self.rho_A)


    def exp_utility_A(self, a, pam):

        p  = ( self.y_A*np.array([1. - self.mean_prob_correct_sensor , self.mean_prob_correct_sensor ]) + 
        (1-self.y_A)*np.array([self.mean_prob_correct_sensor , 1. - self.mean_prob_correct_sensor    ]) )

        p = np.dot(  self.prob_s[:, a, :].T, p ) # Dim SxM
        p = np.dot( p, pam[a] ) # Dim Sx1

        return np.dot( self.utility_A(a), p )

    def optimal_action_A(self):

        EU_A = np.zeros( len(self.a_values) )

        self.pam = self.compute_p_m_a() # Estimate probabilities MV

        for a in self.a_values:
            EU_A[a] = self.exp_utility_A(a, self.pam)

        a_opt = self.a_values[np.argmax(EU_A)]

        return a_opt, EU_A


    def utility_MV(self):

        utility_MV = np.zeros([len(self.m_values), len(self.states)])

        for m in self.m_values:

            states = self.consequences_MV.iloc[
                    self.consequences_MV.index.get_level_values('Action') == m]

            z = np.dot(states, self.weights_MV)
            utility_MV[m] = 1.0 - np.exp(z * self.rho_MV)

        return utility_MV


    def exp_utility_MV(self, a):

        exp_utility_MV = np.zeros(len(self.m_values))

        prob_theta_MV = ( self.y_A*np.array([1. - self.mean_prob_correct_sensor , self.mean_prob_correct_sensor ]) + 
        (1-self.y_A)*np.array([self.mean_prob_correct_sensor , 1. - self.mean_prob_correct_sensor    ]) )

        utility_MV = self.utility_MV()

        for m in self.m_values:
            p = np.dot(  self.prob_s[:, a, m].T, prob_theta_MV )
            exp_utility_MV[m] = np.dot(p, utility_MV[m])

        return exp_utility_MV


    def optimal_action_MV(self, a):

        EU_MV = self.exp_utility_MV(a)
        m_opt = self.m_values[np.argmax(EU_MV)]

        return m_opt, EU_MV

    def simulate(self, theta):

        results = {}

        p  = ( theta*np.array([1. - self.mean_prob_correct_sensor , self.mean_prob_correct_sensor ]) + 
        (1.-theta)*np.array([self.mean_prob_correct_sensor , 1. - self.mean_prob_correct_sensor    ]) )


        self.y_A = 0
        a_opt_0, _    = self.optimal_action_A()
        
        self.y_A = 1
        a_opt_1, _    = self.optimal_action_A()

        self.y_A = 0
        m_opt_00, _    = self.optimal_action_MV(a_opt_0)
        m_opt_01, _    = self.optimal_action_MV(a_opt_1)

        self.y_A = 1
        m_opt_10, _    = self.optimal_action_MV(a_opt_0)
        m_opt_11, _    = self.optimal_action_MV(a_opt_1)


        prob_s_def    = ( p[0]*p[0] * self.prob_s[theta, a_opt_0, m_opt_00,:] +
                          p[0]*p[1] * self.prob_s[theta, a_opt_0, m_opt_10,:] +
                          p[1]*p[0] * self.prob_s[theta, a_opt_1, m_opt_01,:] +
                          p[1]*p[1] * self.prob_s[theta, a_opt_1, m_opt_11,:] )


        results["prob_sensor_00"] = p[0]*p[0]
        results["prob_sensor_01"] = p[0]*p[1]
        results["prob_sensor_10"] = p[1]*p[0]
        results["prob_sensor_11"] = p[1]*p[1]
        results["a_opt_0"]       = a_opt_0
        results["a_opt_1"]       = a_opt_1
        ##
        results["m_opt_00"]       = m_opt_00
        results["m_opt_01"]       = m_opt_01
        results["m_opt_10"]       = m_opt_10
        results["m_opt_11"]       = m_opt_11
        results["prob_s"]         = prob_s_def

        return results









        
        


        
    
    def get_info(self):
        pass

    @staticmethod
    def normalize(arr):
        return arr / np.sum(arr)

    @staticmethod
    def normalize_arr(arr):
        return arr / np.sum(arr, axis=0)




