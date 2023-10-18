import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.interpolate import make_interp_spline

# Functions contained in this file:
# plot_array_w_slope()
# alternateMerge()
# find_min_maxes()
# isolate_hills()
# SMA()
# smooth_line()
# extract_subsections()
# plot_4_sections()
# extract_hills()
# extract_valleys()
# extract_rises_falls()

# read_clean_data()
# identify_candle_color()
# distribution_analysis()
# candlestick_distributions()


def plot_array_w_slope(array):
    X = np.array(np.linspace(0, len(array), len(array), dtype=np.int64))
    
    slopes = np.gradient(array)

    plt.scatter(X, slopes, c='green', alpha=0.8)
    plt.scatter(X, array, c='red', alpha=0.8)

def alternateMerge(arr1, arr2, n1, n2):
    i, j = 0, 0

    newArr = []

    while i < n1 and j < n2:
        newArr.append(arr1[i])
        i += 1

        newArr.append(arr2[j])
        j += 1

    while i < n1:
        newArr.append(arr1[i])
        i += 1

    while j < n2:
        newArr.append(arr2[j])
        j += 1

    return newArr

def find_min_maxes(arr, print_info=False):
    first_der = np.diff(arr)
    signs = np.sign(first_der)
    second_der = np.diff(signs)

    mins = np.where(second_der > 0)
    maxes = np.where(second_der < 0)

    if print_info:
        print('arr:        ', arr)
        print('first_der:  ', first_der)
        print('signs:      ', signs)
        print('second_der: ', second_der)

        print(mins)
        print(maxes)
    
    return mins[0], maxes[0]

def isolate_hills(arr, plot_kind='scatter', print_info=True, 
                  buy_sell_regions=False, figure_size=(10, 6)):
    indices = np.array(np.linspace(0, len(arr), len(arr), dtype=np.int64))

    mins, maxes = find_min_maxes(arr, print_info)

    plt.figure(figsize=figure_size)

    if plot_kind == 'scatter':
        plt.scatter(indices, arr, c='mediumblue', alpha=0.8)    
    elif plot_kind == 'line':
        plt.plot(indices, arr, c='mediumblue', alpha=0.8)

    for val in np.nditer(mins):
        plt.axvline(x=val+1, c='darkred', alpha=0.7)

    for val in np.nditer(maxes):
        plt.axvline(x=val+1, c='lime', alpha=0.7)

    if buy_sell_regions:
        minFirst = True

        print('mins[0]', mins[0])
        print('maxes[0]', maxes[0])

        if mins[0] < maxes[0]:
            mergedArr = alternateMerge(mins, maxes, len(mins), len(maxes))
        else:
            minFirst = False
            mergedArr = alternateMerge(maxes, mins, len(maxes), len(mins))

        print(mergedArr)

        #  3   8   14    
        # min max min 
        #  0   1   2   
        if minFirst:
            c = 'green'
            plt.axvspan(0, mergedArr[0]+1, color=c, alpha=0.3)
            
            for x in range(len(mergedArr)-1):
                if x % 2 == 0:
                    c = 'red'
                else:
                    c = 'green'

                plt.axvspan(mergedArr[x]+1, mergedArr[x+1]+1, color=c, alpha=0.3)

            if len(mergedArr) % 2 == 0:
                plt.axvspan(mergedArr[-1]+1, len(indices), color='green', alpha=0.3)
            else:
                plt.axvspan(mergedArr[-1]+1, len(indices), color='red', alpha=0.3)
        else:
            c = 'red'
            plt.axvspan(0, mergedArr[0]+1, color=c, alpha=0.3)
            
            for x in range(len(mergedArr)-1):
                if x % 2 == 0:
                    c = 'green'
                else:
                    c = 'red'

                plt.axvspan(mergedArr[x]+1, mergedArr[x+1]+1, color=c, alpha=0.3)

            if len(mergedArr) % 2 == 0:
                plt.axvspan(mergedArr[-1]+1, len(indices), color='red', alpha=0.3)
            else:
                plt.axvspan(mergedArr[-1]+1, len(indices), color='green', alpha=0.3)

    plt.show()

def SMA(arr, sma_range):
    SMAs = []

    for x in range(len(arr)):
        if x == 0:
            SMAs.append(arr[0])
        elif x <= sma_range:
            SMAs.append(np.sum(arr[:x]) / x)
        else:
            SMAs.append(np.sum(arr[x-sma_range:x]) / sma_range)

    return SMAs

def smooth_lines(arr, indices):
    i_new = np.arange(0, len(indices), 0.1)
    spl = make_interp_spline(indices, arr, k=3)

    return spl(i_new)

def extract_subsections(data, indices):
    pairs = []
    sub_data_arr = []
    sub_indices_arr = []

    for x in range(len(indices) - 1):
        pairs.append([indices[x]+1, indices[x+1]+2])

    for pair in pairs:
        sub_x1, sub_x2 = pair[0], pair[1]

        distance = sub_x2 - sub_x1
        sub_indices = np.linspace(sub_x1, sub_x2, distance, dtype=np.int64)
        sub_data = data[sub_x1:sub_x2]

        sub_data_arr.append(sub_data)
        sub_indices_arr.append(sub_indices)

    return sub_data_arr, sub_indices_arr

def plot_4_sections(sub_indices, sub_data):
    fig, axs = plt.subplots(1, 4, figsize=(17, 3))


    axs[0].set_ylim(0.2, 2.2)
    axs[0].scatter(sub_indices[0], sub_data[0])

    axs[1].set_ylim(0.2, 2.2)
    axs[1].scatter(sub_indices[1], sub_data[1])

    axs[2].set_ylim(0.2, 2.2)
    axs[2].scatter(sub_indices[2], sub_data[2])

    axs[3].set_ylim(0.2, 2.2)
    axs[3].scatter(sub_indices[3], sub_data[3])

    plt.show()

def extract_hills(data):
    mins, _ = find_min_maxes(data)
    hill_data, hill_indices = extract_subsections(data, mins)

    return hill_data, hill_indices

def extract_valleys(data):
    _, maxes = find_min_maxes(data)
    valley_data, valley_indices = extract_subsections(data, maxes)

    return valley_data, valley_indices

def extract_rises_falls(data):
    mins, maxes = find_min_maxes(data)

    if mins[0] < maxes[0]:
        mins_maxes = alternateMerge(mins, maxes, len(mins), len(maxes))
    else:
        mins_maxes = alternateMerge(maxes, mins, len(maxes), len(mins))

    min_max_pairs, min_max_indices = extract_subsections(data, mins_maxes)

    return min_max_pairs, min_max_indices


def read_clean_data(file, ochl_only=False):
    file_path = 'data/' + file

    data = pd.read_csv(file_path)

    data = data.loc[::-1].reset_index(drop=True)

    filename = file.split('_')
    stock_label = filename[0]

    data.insert(0, 'Label', value=stock_label)
    
    vol = data.pop('Volume')
    data.insert(2, 'Volume', vol)

    ochl_columns = ['Open', 'Close/Last', 'High', 'Low']

    for column in ochl_columns:
        data[column] = data[column].str.replace('$', '').astype('float64')

    if ochl_only:
        return data[ochl_columns]

    return data


def identify_candle_color(data):
    color = []

    for ind in data.index:
        if data['Open'][ind] < data['Close/Last'][ind]:
            color.append('g')
        else:
            color.append('r')

    data['Color'] = color

    return data

def distribution_analysis(data):

    data['avg_all'] = data.mean(axis=1)

    data['avg_op_cl'] = data[['Open', 'Close/Last']].mean(axis=1)
    data['avg_hi_lo'] = data[['High', 'Low']].mean(axis=1)
    data['all_avg_to_hi'] = data[['avg_all', 'High']].mean(axis=1)
    data['all_avg_to_lo'] = data[['avg_all', 'Low']].mean(axis=1)

    data['dist_op_cl'] = data['Open'] - data['Close/Last']
    data['dist_hi_lo'] = data['High'] - data['Low']
    data['dist_avg_all_hi'] = data['High'] - data['avg_all']
    data['dist_avg_all_lo'] = data['avg_all'] - data['Low']
    data['dist_avg_all_op'] = data['Open'] - data['avg_all']
    data['dist_avg_all_cl'] = data['avg_all'] - data['Close/Last']

    data = identify_candle_color(data)

    greens = data[data['Color'] == 'g']
    reds = data[data['Color'] == 'r']

    data['g_dist_op_lo'] = greens['Open'] - greens['Low']
    data['g_dist_cl_hi'] = greens['High'] - greens['Close/Last']

    data['r_dist_op_hi'] = reds['High'] - reds['Open']
    data['r_dist_cl_lo'] = reds['Close/Last'] - reds['Low']

    colors = data.pop('Color')
    data['Color'] = colors

    return data


def extract_ochl(data):
    ochl_columns = ['Close/Last', 'Open', 'High', 'Low']
    ochl_data = data[ochl_columns]

    return ochl_data


def gather_stock_data(filenames, concat=True):
    data = []

    for file in filenames:
        data_set = read_clean_data(file)
        data.append(data_set)
    
    if concat:
        stock_market = pd.concat(data, axis=0, ignore_index=1)

        return stock_market
    
    return data

def ochl_percent_dist(data):
    data['avg_all'] = data.mean(axis=1)

    data['o_percent'] = (data['Open'] - data['avg_all']) / data['avg_all'] * 100
    data['c_percent'] = (data['Close/Last'] - data['avg_all']) / data['avg_all'] * 100
    data['h_percent'] = (data['High'] - data['avg_all']) / data['avg_all'] * 100 
    data['l_percent'] = (data['Low'] - data['avg_all']) / data['avg_all'] * 100

    return data


def binary_indicator(data):
    data['binary_indicator'] = np.where(data['Color'] == 'g', 1, -1)

    return data


def color_probabilities(data):
    colors = data['Color']
    
    numPairs = len(data) - 1
    gg, gr = 0, 0
    rg, rr = 0, 0

    for x in range(len(data)-1):
        if colors[x] == 'g':
            if colors[x+1] == 'g': gg += 1
            else: gr += 1
        elif colors[x] == 'r':
            if colors[x+1] == 'g': rg += 1
            else: rr += 1

    counts = [gg, gr, rg, rr]
    probabilities = []


    for sum in counts:
        probabilities.append(sum / numPairs)

    return probabilities

