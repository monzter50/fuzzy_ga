import random
import matplotlib.pyplot as plt
import math
import numpy as np
import time

def get_y_population(a,b,c,d,e,f,g,xi):
    if c == 0 or e == 0:
        return 0
    else:
        return  a * (b * math.sin(xi/c) + d * math.cos(xi/e)) + f * xi - g

def get_y_fuzzy(data_fuzzy, x,fuzzy_networks):

    bf = 0
    af = 0
    for i in range(fuzzy_networks):
        m = data_fuzzy[i*4]
        de = data_fuzzy[(i*4)+1] 
        p = data_fuzzy[(i*4)+2] 
        q = data_fuzzy[(i*4)+3] 
        
        if de == 0:
            mf = 0
        else:
            mf = math.exp((-math.pow((x-m), 2))/(2*math.pow(de, 2)))

        a = mf*(p*x+q)

        bf += mf
        af += a


    if bf == 0:
        y = 0
    else:
        y = af/bf

    return y  

def better_options(participants):
    better_option = participants[0]
    for i in participants:
        if(fa[better_option] > fa[i]):
            better_option = i

    return better_option