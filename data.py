import numpy as np
import pandas as pd

#s0 major accident
#s1 minor accident
#s2 safely executed interaction
#s3 pedestrian casualty

#a0 change lane
#a1 remain 
#a2 emergency maneuver

#m0 accelerate
#m1 deccelerate
#m2 change lane

# c1 Internal inj. c2 Internal fat. c3 Internal damage. c4 Ext inj. 
# c5 ext fat. c6 Ext damage. c7 Pedestrians. c8 Speed 


def build_params(theta2, theta3, theta4, rho_A, rho_MV,
                 weights_A=np.array([5/100, 25/100, 0, 5/100, 25/100, 0, 25/100, 15/100]) ,
                 weights_MV=np.array([0.1, 0.5, 0.05, 0.35])):

    a_values = np.array([0, 1, 2])
    m_values = np.array([0, 1, 2])
    s_values = np.array([0, 1, 2, 3, 4])

    # theta2 = 4.0 # People die in ADS
    # theta3 = 4.0 # People die in MV 
    # theta4 = 0.0 # Pedestrians die 

    tuples =  [(0, 0),(0, 1),(0, 2),(1, 0),(1, 1),(1, 2),(2, 0),(2, 1),(2, 2),(3, 0),(3, 1),
      (3, 2), (4, 0),(4, 1),(4, 2)]
    index = pd.MultiIndex.from_tuples(tuples, names=["State", "Action"])

    # consequences_A = np.array([ 
    #                         [0., theta2, 1., 0., theta3, 1., 0., 0.],
    #                         [0., theta2, 1., 0., theta3, 1., 0., 0.],
    #                         [0., theta2, 1., 0., theta3, 1., 0., 0.],
    #                         [theta2, 0., 0.5, theta3, 0., 0.5, 0, 0.],
    #                         [theta2, 0., 0.5, theta3, 0., 0.5, 0, 0.],
    #                         [theta2, 0., 0.5, theta3, 0., 0.5, 0, 0.],
    #                         [0., 0., 0., 0., 0., 0., 0., -1.],
    #                         [0., 0., 0., 0., 0., 0., 0., -0.5],
    #                         [0., 0., 0., 0., 0., 0., 0., 0.],
    #                         [0., 0., 0., 0., 0., 0., theta4, 0.],
    #                         [0., 0., 0., 0., 0., 0., theta4, 0.],
    #                         [0., 0., 0., 0., 0., 0., theta4, 0.]      
    #                         ])

    kk = 0.0
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
                            [-kk*theta2, -kk*theta2, -kk*1., -kk*theta3, -kk*theta3, -kk*1.0, theta4, 0.],
                            [-kk*theta2, -kk*theta2, -kk*1., -kk*theta3, -kk*theta3, -kk*1.0, theta4, 0.],
                            [-kk*theta2, -kk*theta2, -kk*1., -kk*theta3, -kk*theta3, -kk*1.0, theta4, 0.], 
                            [theta2, 0., 0.5, 0., 0., 0.0, theta4, 0.],
                            [theta2, 0., 0.5, 0., 0., 0.0, theta4, 0.],
                            [theta2, 0., 0.5, 0., 0., 0.0, theta4, 0.]
                            ])

    consequences_df_A = pd.DataFrame(consequences_A, index = index)

    # consequences_MV = np.array([ 
    #                             [0., theta3, 1.,  0.],
    #                             [0., theta3, 1.,  0.],
    #                             [0., theta3, 1.,  0.],
    #                             [theta3, 0., 0.5, 0.],
    #                             [theta3, 0., 0.5, 0.],
    #                             [theta3, 0., 0.5, 0.],
    #                             [0.,     0., 0.,  -1.],
    #                             [0.,     0., 0.,  -0.5],
    #                             [0.,     0., 0.,  0.],
    #                             [0.,     0., 0.,  0.],
    #                             [0.,     0., 0.,  0.],
    #                             [0.,     0., 0.,  0.]
    #                         ])
    kk_MV = 0.001
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
                                [0.,     0., 0.,  -1.],
                                [0.,     0., 0.,  -0.5],
                                [0.,     0., 0.,  0.],
                                [0.,     0., 0.,  -1.],
                                [0.,     0., 0.,  -0.5],
                                [0.,     0., 0.,  0.]
                            ])

    consequences_df_MV = pd.DataFrame(consequences_MV, index = index)

    prob_s = np.array([
                      [
                        [ [2/3,1/6,1/6,0.,0.], [1/6,1/6,2/3,0.,0.], [1/24,1/24,11/12,0.,0.] ],
                        [ [0., 0., 0.95, 0., 0.05], [0., 0., 0.95, 0., 0.05], [1/3, 1/3, 1/3,  0., 0.] ], 
                        [ [0., 0., 0., 1., 0.], [0., 0., 0., 1.,0.], [0.,  0.,  0.,   1.,0.] ]
                      ],
                      [
                        [ [5/6,1/8,1/24,0.,0.], [1/3,1/3,1/3,0.,0.], [1/3,1/3,1/3,0.,0.] ],
                        [ [0., 0., 0.7,  0., 0.3], [0., 0., 0.7, 0., 0.3], [2/3,1/3, 0., 0., 0.] ], 
                        [ [0., 0., 0.,  1., 0.], [0., 0., 0., 1., 0.], [0.,  0., 0., 1., 0.] ]
                      ]
                      ])



    params = {}
    params["a_values"] = a_values
    params["m_values"] = m_values
    params["s_values"] = s_values

    params["prob_s"] = prob_s
    params["mean_prob_correct_sensor"] = 0.95

    params["consequences_MV"] = consequences_df_MV
    params["weights_MV"]      = weights_MV
    params["rho_MV"]          = rho_MV


    params["consequences_A"] = consequences_df_A
    params["weights_A"] = weights_A
    params["rho_A"] = rho_A


    params["k"] = 100 #controls variance of beta and dirichlet
    params["N"] = 1000

    return params

      

