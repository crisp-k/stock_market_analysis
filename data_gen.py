import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import softmax
from datetime import timedelta


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


def custom_sample_distribution(dist, num_bins=500):

    plt.ioff()

    n, bins, _ = plt.hist(dist, bins=num_bins, density=True);

    values = np.delete(bins, 0)
    probabilities = softmax(n)

    return values, probabilities


def extract_distributions(data):
    dist_labels = ['o_percent', 'c_percent', 'h_percent', 'l_percent']

    dist_samples = []

    for label in dist_labels:
        dist = data[label]
        x_values, x_proba = custom_sample_distribution(dist)

        dist_samples.append([x_values, x_proba])

    dist_samples = np.asarray(dist_samples)

    print(dist_samples.shape)


def gen_new_val(old_val, percent_change):
        to_add = old_val * percent_change / 100
        new_val = old_val + to_add

        return new_val

def generate_high_and_low(data, distributions):
    mins, maxes = [], []
    high, low = [], []

    for ind in data.index:
        open = data['Open'][ind]
        close = data['Close'][ind]

        if open > close:
            maxes.append(open)
            mins.append(close)
        else:
            maxes.append(close)
            mins.append(open)

    for val in maxes:
        h_random_percent = np.random.choice(a=distributions[2][0], p=distributions[2][1])
        high.append(gen_new_val(val, h_random_percent))

    for val in mins:
        l_random_percent = np.random.choice(a=distributions[3][0], p=distributions[3][1])
        low.append(gen_new_val(val, l_random_percent))
        
    data['High'] = high
    data['Low'] = low

    return data

def generate_data(numVals, volatility, init_val=1, moving_vol=True, ochl_dist=None, random_seed=42):
    np.random.seed(random_seed)
    
    rng = np.random.default_rng(seed=random_seed)
    
    vals = [init_val]
    vols = [volatility]
    open, close = [0], [0]

    o_random_percent = np.random.choice(a=ochl_dist[0][0], p=ochl_dist[0][1])
    c_random_percent = np.random.choice(a=ochl_dist[1][0], p=ochl_dist[1][1])

    open[0] = (gen_new_val(vals[0], o_random_percent))
    close[0] = (gen_new_val(vals[0], c_random_percent))


    for x in range(numVals):
        rnd = rng.random()
        change_percent = 1.5 * volatility * rnd
        if change_percent > volatility:
            change_percent -= (2 * volatility)
        change_amount = vals[x] * change_percent
        vals.append(vals[x] + change_amount)

        if moving_vol:
            if volatility >= 0.2:
                # print("Volatility over 20%:    ", volatility)
                p = [0.80, 0.10, 0.10]
            elif volatility <= 0.1:
                # print("Volatility under 10%:   ", volatility)
                p = [0.10, 0.10, 0.80]
            else:
                p = [0.30, 0.40, 0.30]

            vol_change = rng.random()
            vol_direction = np.random.choice([-1, 0, 1], p=p)
            volatility += vol_change * vol_direction * 0.03
            vols.append(volatility)

        
        o_random_percent = np.random.choice(a=ochl_dist[0][0], p=ochl_dist[0][1])
        c_random_percent = np.random.choice(a=ochl_dist[1][0], p=ochl_dist[1][1])

        open.append(gen_new_val(vals[x], o_random_percent))
        close.append(gen_new_val(vals[x], c_random_percent))
    

    start_date = '1998-03-22'
    end_date = pd.to_datetime(start_date) + timedelta(days=numVals)

    dates = pd.date_range(start_date, end_date)
    

    data = [vals, open, close]
    data = np.array(data).T

    stock_data = pd.DataFrame(data=data, columns=['avg_price', 'Open', 'Close'], index=dates)

    stock_data = generate_high_and_low(stock_data, ochl_dist)

    return stock_data




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


def create_stocks(numStocks, numVals, startingVal, moving_vol=True):
    for x in range(numStocks):
        os.makedirs('fake_stocks/stock_start_{0}/moving_vol_{1}'.format(startingVal, moving_vol), exist_ok=True)
        stock_data = generate_data(numVals, volatility=0.15, init_val=startingVal, moving_vol=moving_vol)

        stock = pd.DataFrame(np.array(stock_data), columns=['Value'])
        stock.to_csv('fake_stocks/stock_start_{0}/moving_vol_{1}/stock{2}_numVal{3}.csv'.format(startingVal, moving_vol, x, numVals))


def generate_stock_market(num_stocks, num_vals, price_ranges, moving_vols):
    for price in price_ranges:
        for mv in moving_vols:
            create_stocks(num_stocks, num_vals, price, moving_vol=mv)
