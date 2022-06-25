import numpy as np
import pandas as pd

#s0 major accident
#s1 minor accident
#s2 safely executed interaction
#s3 minor accident

#a0 change lane
#a1 remain 
#a2 emergency maneuver

#m0 accelerate
#m1 deccelerate
#m2 change lane

a_values = np.array([0, 1, 2])
m_values = np.array([0, 1, 2])

theta2 = 1.0 # People die in ADS
theta3 = 1.0 # People die in MV 
theta4 = 0.0 # Pedestrians die 

tuples =  [(0, 0),(0, 1),(0, 2),(1, 0),(1, 1),(1, 2),(2, 0),(2, 1),(2, 2),(3, 0),(3, 1),(3, 2)]
index = pd.MultiIndex.from_tuples(tuples, names=["State", "Action"])

consequences_A = np.array([ 
                        [0., theta2, 1., 0., theta3, 1., 0., 0.],
                        [0., theta2, 1., 0., theta3, 1., 0., 0.],
                        [0., theta2, 1., 0., theta3, 1., 0., 0.],
                        [theta2, 0., 0.5, theta3, 0., 0.5, 0, 0.],
                        [theta2, 0., 0.5, theta3, 0., 0.5, 0, 0.],
                        [theta2, 0., 0.5, theta3, 0., 0.5, 0, 0.],
                        [0., 0., 0., 0., 0., 0., 0., -1.],
                        [0., 0., 0., 0., 0., 0., 0., -0.5],
                        [0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., theta4, 0.],
                        [0., 0., 0., 0., 0., 0., theta4, 0.],
                        [0., 0., 0., 0., 0., 0., theta4, 0.]      
                        ])

consequences_df_A = pd.DataFrame(consequences_A, index = index)

consequences_MV = np.array([ 
                            [0., theta3, 1.,  0.],
                            [0., theta3, 1.,  0.],
                            [0., theta3, 1.,  0.],
                            [theta3, 0., 0.5, 0.],
                            [theta3, 0., 0.5, 0.],
                            [theta3, 0., 0.5, 0.],
                            [0.,     0., 0.,  -1.],
                            [0.,     0., 0.,  -0.5],
                            [0.,     0., 0.,  0.],
                            [0.,     0., 0.,  0.],
                            [0.,     0., 0.,  0.],
                            [0.,     0., 0.,  0.]
                        ])

consequences_df_MV = pd.DataFrame(consequences_MV, index = index)

prob_s = np.array([
                  [
                    [ [2/3,1/6,1/6,0.], [1/6,1/6,2/3,0.], [1/24,1/24,11/12,0.] ],
                    [ [0., 0., 1., 0.], [0., 0., 1., 0.], [1/3, 1/3, 1/3,  0.] ], 
                    [ [0., 0., 0., 1.], [0., 0., 0., 1.], [0.,  0.,  0.,   1.] ]
                  ],
                  [
                    [ [5/6,1/8,1/24,0.], [1/3,1/3,1/3,0.], [1/3,1/3,1/3,0.] ],
                    [ [0., 0., 1.,  0.], [0., 0., 1., 0.], [2/3,1/3,0., 0.] ], 
                    [ [0., 0., 0.,  1.], [0., 0., 0., 1.], [0.,  0.,0., 1.] ]
                  ]
                  ])



params = {}
params["prob_s"] = prob_s
params["mean_prob_correct_sensor"] = 0.95

params["consequences_MV"] = consequences_df_MV
params["weights_MV"]      = np.array([0.98, 0.005, 0.005, 0.005])
params["rho_MV"]          = 0.6


params["consequences_A"] = consequences_df_A
params["weights_A"] = np.array([0., 0., 0., 0., 0., 0., 0., 1.]) 
params["rho_A"] = 0.6


params["k"] = 100 #controls variance of beta and dirichlet
params["N"] = 1000