# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

WALL=[5]
Q = np.zeros((3,4))
R_DEF =-.04
R = [R_DEF for _ in range(12)]
R[3]=1
R[7]=-1
PI = np.zeros(12).astype(int)
PI[3]=9
PI[7]=9
PI[5]=9
PI_S=np.chararray((3,4))
V = np.zeros(12)
V[3]=1
V[7]=-1
NO_CALC_STATES=[3,5,7]
gamma=.9
def prepare_next_state_matrix():
    NEXT_STATE = np.zeros((12,4)).astype(int)
    for i in range(12):
        NEXT_STATE[i,:]=i
        if i not in [3,7,11]:
            NEXT_STATE[i,2] = i+1
        if i not in [0,4,8]:
            NEXT_STATE[i,0] = i-1
        if i > 3:
            NEXT_STATE[i,1] = i-4
        if i <8:
            NEXT_STATE[i,3] = i+4
    return NEXT_STATE
N_S = prepare_next_state_matrix()


def get_next(s,a):
    s_p = N_S[s,a]
    return s if s_p in WALL else s_p

def get_left(s,a):
    if a==0:
        return 3
    elif a==1:
        return 0
    elif a == 2:
        return 1
    elif a==3:
        return 2
def get_right(s,a):
    if a==0:
        return 1
    elif a==1:
        return 2
    elif a == 2:
        return 3
    elif a==3:
        return 0

# Eq: V(s) = R(s,a,s') + gamma * sum (T * V(s'))

def value_iteration():
    for i in range(200):
        PREV_V = np.copy(V)
        _value_calculation()
        if np.allclose(PREV_V,V):
            print("!!! Covergence at:{}".format(i))
            break
    print("V =>{}".format(V))
#    _evaluate_policy()
    global PI_S
    PI_S = np.vectorize(_action_depict)(PI).reshape((3,4))
        
def _value_calculation():
    global PI
    for s in range(12):
        if s in NO_CALC_STATES:
            continue
        E_V=[0 for _ in range(4)] # this is for 4 actions.
        
        for a in range(4):
            s_p = get_next(s,a)
            
            E_V[a] =E_V[a]+  .8*  V[s_p]
            # left
            a_left = get_left(s,a)
            s_p_l = get_next(s,a_left)
            E_V[a] = E_V[a]+ .1 *  V[s_p_l]
            # right
            a_right = get_right(s,a)
            s_p_r = get_next(s,a_right)
            E_V[a] =E_V[a]+ .1 *  V[s_p_r]
        V[s] = R[s] + gamma * max(E_V)
        PI[s] = np.argmax(E_V)
    
        
def _action_depict(a):
    if a==2:
        return '->'
    elif a==0:
        return '<-'
    elif a==1:
        return ' ^'
    elif a==3:
        return ' v'
    elif a==9:
        return '  '
    
    
value_iteration()
print("PI==>\n{}".format(PI_S))

def _evaluate_policy():
    global PI
    for s in range(12):
        if s in NO_CALC_STATES:
            continue
        E_V=[0 for _ in range(4)] # this is for 4 actions.
        
        for a in range(4):
            s_p = get_next(s,a)
            
            E_V[a] =E_V[a]+  .8*  V[s_p]
            # left
            a_left = get_left(s,a)
            s_p_l = get_next(s,a_left)
            E_V[a] = E_V[a]+ .1 *  V[s_p_l]
            # right
            a_right = get_right(s,a)
            s_p_r = get_next(s,a_right)
            E_V[a] =E_V[a]+ .1 *  V[s_p_r]
        PI[s] = np.argmax(E_V)





