import numpy as np
import matplotlib.pyplot as plt


# Functions contained in this file:
# expand_array()
# generate_data()
# continue_data()


def expand_array(array, runs = 1):
    
    temp = np.copy(array)

    for i in range(runs):
        new_array = []
        for x in range(len(temp)-1):
            val_avg = (temp[x] + temp[x+1]) / 2
            new_array.append(temp[x])
            new_array.append(val_avg)

        temp = np.copy(new_array)

    return new_array

def generate_data(numVals, volatility, init_val=1):
    rng = np.random.default_rng()
    
    vals = [init_val]

    for x in range(numVals):
        rnd = rng.random()
        change_percent = 1.5 * volatility * rnd
        if change_percent > volatility:
            change_percent -= (2 * volatility)
        change_amount = vals[x] * change_percent
        vals.append(vals[x] + change_amount)

    return vals

def continue_data(numVals, volatility, last_vals, range_of_old_data=1):
    rng = np.random.default_rng()
    
    if range_of_old_data >= len(last_vals):
        range_of_old_data = len(last_vals)

    vals = last_vals[-range_of_old_data:]
    

    for x in range(numVals):
        rnd = rng.random()
        change_percent = 1.5 * volatility * rnd
        if change_percent > volatility:
            change_percent -= (2 * volatility)
        change_amount = vals[x] * change_percent
        vals.append(vals[x] + change_amount)

    return vals


