import numpy as np
import scipy
import pandas as pd

def SMA_batch_compute(data_series,n_period_compute,comp_mode='loop'):
    '''
    :param data_series: The time series data with length [m*t]
    :param n_period_compute: The length n one would like to care for the computation.
            Should be a list, if it is this case return [m * len]. Even if the length of list is 1
    :param comp_mode: should be either 'loop' or 'vector', whether to use vectorized computation
                        recommend 'loop' when max(n_period_compute) is not large, but 'vector' when it's large
    :return: [m*len(n_period_compute)] numpy array for Simple Moving Average
    '''
    # check the input arguments
    assert type(data_series).__module__ == np.__name__
    if not isinstance(n_period_compute, list):
        raise ValueError('Batch computation method only permit list as input!')
    # hyper-parameters
    n_len_return = len(n_period_compute)
    nData = data_series.shape[0]
    nTime_max = data_series.shape[1]
    if max(n_period_compute)>nTime_max:
        raise ValueError('The n_period value could not be greater than the length of time series')
    # loop mode
    if comp_mode=='loop':
        # compute the simple moving average
        MA_return_mat = np.zeros([n_len_return, nData])  # placeholder
        current_ind = 0
        for n_period in n_period_compute:
            current_rst = np.reshape(np.mean(data_series[:,-n_period:],axis=1),[-1])  # [nData]
            MA_return_mat[current_ind,:] = current_rst
            current_ind =  current_ind + 1
    elif comp_mode == 'vector':
        # actually the for loop is inevitable from a view that vectorized indexing could not provide partial 'multi-col'
        # here we implement just 4 fun
        MA_compute_mat = np.tile(data_series, [n_len_return, 1, 1])  # [n*m*t]
        # use mask mat to avoid loop
        MA_mask_mat = np.zeros(MA_compute_mat.shape)
        for current_ind in range(len(n_period_compute)):
            MA_mask_mat[current_ind,:,-n_period_compute[current_ind]:] = 1
        # print(np.multiply(MA_compute_mat,MA_mask_mat))
        # use the mask matrix for multiplication
        MA_return_mat = np.sum(np.multiply(MA_compute_mat,MA_mask_mat),axis=-1)      # [n * nData(m)]
        MA_return_mat = np.divide(MA_return_mat,
                                  np.tile(np.reshape(np.array(n_period_compute),
                                                     [-1,1]),
                                          [1,nData]))
    else:
        raise ValueError('Computation node not recognized! Could only process \'loop\' or \'vector\', but got \''+
                         comp_mode+'\'')
    # vectorized mode
    MA_return_mat = MA_return_mat.T

    return MA_return_mat

def EMA_batch_computation(data_series,n_period_compute,comp_mode='loop'):
    '''
    :param data_series: The time series data with length [m*t]
    :param n_period_compute: The n-period one would like to care for the computation.
            Should be a list, if it is this case return [m * len]. Even if the length of list is 1
    :param comp_mode: should be either 'loop' or 'vector', whether to use vectorized computation
                        recommend 'loop' when max(n_period_compute) is not large, but 'vector' when it's large
    :return: [m*len(n_period_compute)] numpy array for Exponential Moving Average
    '''
    # check the input arguments
    assert type(data_series).__module__ == np.__name__
    if not isinstance(n_period_compute, list):
        raise ValueError('Batch computation method only permit list as input!')
    # hyper-parameters
    n_len_return = len(n_period_compute)
    nData = data_series.shape[0]
    nTime_max = data_series.shape[1]
    if max(n_period_compute) > nTime_max:
        raise ValueError('The n_period value could not be greater than the length of time series')
    # loop mode
    if comp_mode == 'loop':
        # in the exponential case, since the \alpha value for different ns are different, there're no overlapping
        # sub-problems, which means one need to either nest loops, or we could (see the vectorized mode comments)
        # compute the simple moving average
        MA_return_mat = np.zeros([n_len_return, nData])  # placeholder
        current_ind = 0
        for n_period in n_period_compute:
            # use simple moving average to initialize the EMA value
            prev_exp_ma = np.reshape(np.mean(data_series[:,-n_period:],axis=1),[nData])
            current_rst = np.copy(prev_exp_ma)
            # compute the alpha value
            current_alpha = 2/(1+n_period)
            # compute the current starting and ending index
            current_recursion_range = range(n_len_return-n_period,n_len_return)
            # print(current_recursion_range)
            # print(prev_exp_ma)
            for c_recurs_ind in current_recursion_range:
                if c_recurs_ind==0:
                    continue
                # compute the current sequence
                current_rst = current_alpha*np.reshape(data_series[:, c_recurs_ind], [-1])\
                              +(1-current_alpha)*prev_exp_ma                                   # [nData]
                # assign the value of the previous result
                prev_exp_ma = np.copy(current_rst)
            MA_return_mat[current_ind, :] = np.copy(current_rst)       # avoid passing address
            current_ind = current_ind + 1
    elif comp_mode == 'vector':
        # Or we could duplicate the time series n times, and assign n alpha values to them; then we run the loop for one
        # time, and store the results to the resulting matrix in the loop. Then one could save computational time on the
        # cost of memory (also because nested for loops are 'enemies of the people' in Python)
        MA_compute_mat = np.tile(data_series, [n_len_return, 1, 1])  # [n*m*t]
        # place-holder of the resulting matrix
        MA_return_mat = np.zeros([n_len_return, nData])
        # duplicate alpha values for further computation
        alpha_value_all = 2/(np.tile(np.reshape(np.array(n_period_compute),[n_len_return,1]),[1,nData])+1)
        # define the [n*m] prev matrix to store the info
        prev_exp_ma = np.zeros([n_len_return,nData])
        # use the simple moving average as the initialization
        for current_ind in range(n_len_return):
            this_exp_ma_value = np.reshape(np.mean(data_series[:, -n_period_compute[current_ind]:], axis=1), [nData])
            # avoid computation during recursion by assigning given values
            MA_compute_mat[current_ind,:,:n_len_return-(current_ind+1)] = np.tile(np.reshape(this_exp_ma_value,[-1,1]),
                                                                                    [1,n_len_return-(current_ind+1)])
            # prev_exp_ma value
            prev_exp_ma[current_ind,:] = this_exp_ma_value[:]
            # # assign the arg[0] value anyways
            # if current_ind == n_len_return-1:
            #     MA_compute_mat[current_ind,:,0] = this_exp_ma_value[:]
        # print(prev_exp_ma)
        # print(MA_compute_mat)
        current_rst = 0
        for current_ind in range(0,n_len_return-1):
            # retrieve the data in the current time step
            current_series_data = np.reshape(MA_compute_mat[:,:,current_ind+1],[n_len_return,nData])
            # compute the recursion for all of them
            current_rst = np.multiply(alpha_value_all,current_series_data) + np.multiply(1-alpha_value_all,prev_exp_ma)
            # store the 'previous results'
            prev_exp_ma = np.copy(current_rst)
        # assign the values to the resulting matrix
        MA_return_mat = np.copy(current_rst)
    else:
        raise ValueError('Computation node not recognized! Could only process \'loop\' or \'vector\', but got \'' +
                         comp_mode + '\'')
    # vectorized mode
    MA_return_mat = MA_return_mat.T

    return MA_return_mat

def STOCH_batch_compute(data_series,high_series,low_series,n_period_compute):
    '''
    Compute the Stochastic Oscillator, return in [nData * n_Length_Period] format
    :param data_series: The time series of the close price
    :param low_series: The time series of the lowest price
    :param high_series: The time series of the highest price
    :param n_period_compute: The n-period one would like to care for the computation.
                            Should be a list, if it is this case return [m * len]. Even if the length of list is 1
    :return: Two [nData * len] arrays, representing K and D lines respectively
    (To clarify the nData here is actually the 'lags' of time series -- n-stride+1)
    '''
    # check the input arguments
    assert type(data_series).__module__ == np.__name__
    if not isinstance(n_period_compute, list):
        raise ValueError('Batch computation method only permit list as input!')
    # hyper-parameters
    n_len_return = len(n_period_compute)
    nData = data_series.shape[0]
    nTime_max = data_series.shape[1]
    # the time_series must be long enough: nTime_max-3 because (1 for additional 'last period'), (2 for d=3MA(K) )
    if max(n_period_compute) > (nTime_max-3):
        raise ValueError('The n_period value could not be greater than the length of time series')
    if (data_series.shape!=high_series.shape) or (data_series.shape!=low_series.shape) or (high_series.shape!=low_series.shape):
        raise ValueError('The shape of the times series of close, high, and low prices must be the same!')
    # compute the metric, for loop is not evadable here
    # placeholders
    stoch_K_occi_mat = np.zeros([n_len_return, nData])
    stoch_D_occi_mat = np.zeros([n_len_return, nData])
    close_price = np.reshape(data_series[:,-1],[nData])
    # define a list to collect current K information -- and compute the D mat
    K_3_list = []
    for current_ind in range(n_len_return+2):
        if current_ind<=n_len_return-1:
            current_period = n_period_compute[current_ind]
        else:
            current_period = n_period_compute[-1]+current_ind-n_len_return+1
        # compute the lowest low of this period of time
        period_lowest_low = np.reshape(np.amin(low_series[:,-(current_period+1):],axis=1),[nData])
        # compute the highest high of this period
        period_highest_high = np.reshape(np.amax(high_series[:,-(current_period+1):],axis=1),[nData])
        # compute the metric ele-by-ele
        period_K_metric = 100*(close_price-period_lowest_low)/(period_highest_high-period_lowest_low)
        # assign the value
        if current_ind<=(n_len_return-1):
            stoch_K_occi_mat[current_ind,:] = period_K_metric[:]
        # collect the info
        K_3_list.append(period_K_metric[:])
        # compute the D metric (if it's now computable)
        if current_ind>=2:
            current_3_K_mat = np.flip(np.array(K_3_list).T,axis=1)   # 'flip' to arrange the newest to the rightmost
            current_D_mat = np.reshape(SMA_batch_compute(current_3_K_mat,[3]),[nData])
            stoch_D_occi_mat[current_ind-2,:] = current_D_mat[:]
            K_3_list.pop(0)         # pop out the 'previous' record and maintain the length of list
    # inverse the matrices and return
    stoch_K_occi_mat = stoch_K_occi_mat.T
    stoch_D_occi_mat = stoch_D_occi_mat.T

    return stoch_K_occi_mat, stoch_D_occi_mat

def ADX_batch_compute(data_series,high_series,low_series,n_period_compute):
    '''
    Compute the ADX (Average Directional Movement Index), return in [nData * n_Length_Period] format
    :param data_series: The time series of the close price
    :param low_series: The time series of the lowest price
    :param high_series: The time series of the highest price
    :param n_period_compute: The n-period one would like to care for the computation.
                            Should be a list, if it is this case return [m * len]. Even if the length of list is 1
    :return: Two [nData * len] arrays, representing the DI+ and DI- matrices, which are components of ADX
    '''
    # check the input arguments
    assert type(data_series).__module__ == np.__name__
    if not isinstance(n_period_compute, list):
        raise ValueError('Batch computation method only permit list as input!')
    # hyper-parameters
    n_len_return = len(n_period_compute)
    nData = data_series.shape[0]
    nTime_max = data_series.shape[1]
    # time series should be at least 1 unit longer than the maximum time period, for each n-period requires (n+1) days
    # of data
    if max(n_period_compute) > (nTime_max - 1):
        raise ValueError('The n_period value could not be greater than the length of time series')
    if (data_series.shape != high_series.shape) or (data_series.shape != low_series.shape) or (
        high_series.shape != low_series.shape):
        raise ValueError('The shape of the times series of close, high, and low prices must be the same!')
    # compute the metric, the 1-layer for loop is inevitable here, but one could vectorize it to avoid nested for loop
    # place holder of ADX mat
    ADX_indicator_mat = np.zeros([n_len_return, nData])
    # compute the differences representing 'upward' and 'downward' trends
    UpMove_mat = np.diff(high_series,n=1,axis=1)                  # (t+1)'s high - t's high
    DownMove_mat = np.flip(np.diff(np.flip(low_series,axis=1),n=1,axis=1),axis=1)    # t's low - (t+1)'s low
    # concatenate them with the 'primitive' elements so that the shape of the two array will remain unchanged
    UpMove_mat = np.concatenate([np.zeros([nData,1]),UpMove_mat],axis=1)
    DownMove_mat = np.concatenate([np.zeros([nData,1]),DownMove_mat],axis=1)
    # now one could compute +DM and -DM matrices (firstly compute the mask)
    DM_up_mask_mat = (UpMove_mat > DownMove_mat) & (UpMove_mat > 0)
    DM_down_mask_mat = (DownMove_mat > UpMove_mat) & (DownMove_mat > 0)
    # initialize zeros mat for both DM_up and DM_down matrices, and then assign values to the corresponding index
    # up matrix
    DM_up_mat = np.zeros(UpMove_mat.shape)
    DM_up_mat[DM_up_mask_mat] = UpMove_mat[DM_up_mask_mat]
    # down matrix
    DM_down_mat = np.zeros(DownMove_mat.shape)
    DM_down_mat[DM_down_mask_mat] = DownMove_mat[DM_down_mask_mat]
    # for loop inside the function calling of Exponential Moving Average
    DM_up_SMA = EMA_batch_computation(data_series=DM_up_mat,n_period_compute=n_period_compute,comp_mode='vector')
    DM_down_SMA = EMA_batch_computation(data_series=DM_down_mat, n_period_compute=n_period_compute, comp_mode='vector')
    # compute the matrix of 'Average True Range'
    TR_mat = np.zeros(DM_up_SMA.shape)          # [nData * n_len_Periods]
    # for loop required
    for current_ind in range(n_len_return):
        current_period = n_period_compute[current_ind]
        # compute the lowest low of this period of time
        # (different from the previous function -- because we use different methodology here)
        period_lowest_low = np.reshape(np.amin(low_series[:, -current_period:], axis=1), [nData])
        # compute the highest high of this period
        period_highest_high = np.reshape(np.amax(high_series[:, -current_period:], axis=1), [nData])
        # retrieve the previous close price
        prev_close_price = np.reshape(data_series[:,-(current_period + 1)],[nData])
        period_TR = np.maximum(np.maximum(period_highest_high-period_lowest_low,
                                          np.absolute(period_lowest_low-prev_close_price)),
                               np.absolute(period_highest_high-prev_close_price))
        # assign the values
        TR_mat[:,current_ind] = period_TR[:]
    # compute the overall matrix
    DI_up_mat = 100*np.divide(DM_up_SMA,TR_mat)
    DI_down_mat = 100*np.divide(DM_down_SMA,TR_mat)
    # simply return the two values instead of ADI to explore what the AI algorithm could get

    return DI_up_mat, DI_down_mat

def CCI_batch_compute(data_series,high_series,low_series,n_period_compute):
    '''
    Compute the CCI index (Commodity channel index), return in [nData * n_Length_Period] format
    :param data_series: The time series of the close price
    :param low_series: The time series of the lowest price
    :param high_series: The time series of the highest price
    :param n_period_compute: The n-period one would like to care for the computation.
                            Should be a list, if it is this case return [m * len]. Even if the length of list is 1
    :return: [nData * len] arrays, representing the CCI index for different time length
    '''
    # check the input arguments
    assert type(data_series).__module__ == np.__name__
    if not isinstance(n_period_compute, list):
        raise ValueError('Batch computation method only permit list as input!')
    # hyper-parameters
    n_len_return = len(n_period_compute)
    nData = data_series.shape[0]
    nTime_max = data_series.shape[1]
    # time series should be at least 10 unit longer than the maximum time period, because one will be needing 3-period
    # SMA of the typical price
    if max(n_period_compute) > (nTime_max - 10):
        raise ValueError('The n_period value could not be greater than the length of time series')
    if (data_series.shape != high_series.shape) or (data_series.shape != low_series.shape) or (
                high_series.shape != low_series.shape):
        raise ValueError('The shape of the times series of close, high, and low prices must be the same!')
    # placeholders
    typical_price_mat = np.zeros([n_len_return, nData])
    MA_tp_mat = np.zeros([n_len_return, nData])
    MA_MD_mat = np.zeros([n_len_return, nData])
    # define the close price matrix (the last column)
    close_price_mat = np.reshape(data_series[:,-1],[nData])
    # define the list to collect 3-period
    typical_price_3_list = []
    # for loop required
    for current_ind in range(n_len_return+9):
        if current_ind<=n_len_return-1:
            current_period = n_period_compute[current_ind]
        else:
            current_period = n_period_compute[-1]+current_ind-n_len_return+1
        # compute the current typical price
        current_highest_high = np.reshape(np.amax(high_series[:,-current_period:],axis=1),[nData])
        current_lowest_low = np.reshape(np.amin(high_series[:,-current_period:],axis=1),[nData])
        current_typical_price = (current_highest_high+current_lowest_low+close_price_mat)/3
        # assign the value to the corresponding position
        if current_ind<=(n_len_return-1):
            typical_price_mat[current_ind,:] = current_typical_price
        # collect the info of typical prices
        typical_price_3_list.append(current_typical_price)
        if current_ind>=9:
            current_3_period_tp = np.array(typical_price_3_list).T
            # compute Moving Average of the Typical Price
            current_MA_tp = SMA_batch_compute(current_3_period_tp,[10],comp_mode='vector')
            MA_tp_mat[current_ind-9,:] = np.reshape(current_MA_tp,[nData])
            # compute the Mean Abosolute Derivation of the Typical Price
            current_Mean_mat = np.tile(np.reshape(np.mean(current_3_period_tp,axis=1),[nData,1]),[1,10]) # tile for computetation
            current_MD_mat = np.mean(np.absolute(current_3_period_tp - current_Mean_mat),axis=1)
            # assign the value
            MA_MD_mat[current_ind-9,:] = np.reshape(current_MD_mat,[nData])
            # pop the first element
            typical_price_3_list.pop(0)
    # compute the final results of CCI
    CCI_rst_mat = (1/0.015)*np.divide(typical_price_mat-MA_tp_mat, MA_MD_mat)
    CCI_rst_mat = CCI_rst_mat.T

    return CCI_rst_mat

def Nan_batch_interpolation(original_array):
    '''
    The function to perform linear interpolation for NaN values in [nData * nDim] arrays, row-based
    :param original_array: The pre-interpolated array
    :return: The new [nData * nDim] array, in which the NaN values are all replaced by linear interpolations
    '''
    # check the input argument
    assert type(original_array).__module__ == np.__name__
    # must be 2-dim
    if (len(original_array.shape)!=2):
        raise ValueError('NaN interpolation only process 2-d array with [nData*nDim] format')
    # retrieve the hyper-parameters
    nData = original_array.shape[0]
    nDim = original_array.shape[1]
    # define the index-checking func
    func_ind_check = lambda x: np.nonzero(x)[0]
    # placeholder of rst array
    rst_array = np.zeros(original_array.shape)
    # loop over the instances to replace NaN
    for cData in range(nData):
        # current original sequence
        current_org_sequence = np.reshape(original_array[cData,:],[-1])
        # retrieve the NaN mask vector
        NaN_mask_vec = np.isnan(current_org_sequence)
        # interpolation
        current_org_sequence[NaN_mask_vec] = np.interp(func_ind_check(NaN_mask_vec),
                                                       func_ind_check(~NaN_mask_vec),
                                                       current_org_sequence[~NaN_mask_vec])
        # re-assign the sequence back to the array
        rst_array[cData,:] = np.reshape(current_org_sequence,[-1])

    return rst_array

def MA_up_down_check(price_series, MA_period=3, sig_days=5, MA_mode='simple'):
    '''
    :param price_series: The price series denoting the adjusted close prices
    :param MA_period: The period one would like to compute Moving Average
    :param sig_days: The days one could judge if the trend is upward/downward
    :param MA_mode: Which type of moving average to compute. Could be 'simple' or 'exponential'
    NB: The length of price should be no less than (>=) (MA_period+sig_days)
        The reason is that to compute at least sig_days of difference of moving averages, we'll be required to compute
        (sig_days+1) moving averages, which will be resulting in (sig_days+1+MA_period-1)  = (MA_period+sig_days) length
    :return: symbols:
            2: current uptrend  ((uptrend time - downtrend time)>= det_days)
            1: current downtrend ((downtrend time - uptrend time) >= det_days)
            0: current no significant trend
    '''
    # check the input argument
    assert type(price_series).__module__ == np.__name__
    # retrieve the length
    nlength = price_series.shape[0]
    # check if the length of the series array satisfies the conditions
    if nlength<(MA_period+sig_days):
        raise ValueError('The length of the time series array should be at least = (MA_period+sig_days) !')
    # compute the base moving average
    current_price_array = np.reshape(price_series[:MA_period],[1,-1])
    prev_MA_price = SMA_batch_compute(current_price_array,n_period_compute=[MA_period],comp_mode='vector')
    # define the uptrend and downtrend times
    uptrend_time = 0
    downtrend_time = 0
    # for loop inevitable: use for loop to see the price
    for start_ind in range(1,nlength-MA_period):
        current_price_array = np.reshape(price_series[start_ind:start_ind+MA_period], [1, -1])
        if MA_mode=='simple':
            current_MA_price = SMA_batch_compute(current_price_array,n_period_compute=[MA_period],comp_mode='vector')
        elif MA_mode=='exponential':
            current_MA_price = EMA_batch_computation(current_price_array, n_period_compute=[MA_period], comp_mode='vector')
        else:
            raise ValueError('The Moving Average computation mode unrecognized! Should be either \'simple\' or \'exponential\', '
                             'but got \''+str(MA_mode)+'\' !')
        # determine whether to increase up/down time counts
        if current_MA_price>prev_MA_price:
            uptrend_time = uptrend_time + 1
        elif current_MA_price<prev_MA_price:
            downtrend_time = downtrend_time + 1
        else:
            pass
        # assign the current value to prev
        prev_MA_price = current_MA_price
    # determine which to return
    if (uptrend_time-downtrend_time)>=sig_days:    # significant uptrend
        return 2
    elif (downtrend_time-uptrend_time)>=sig_days:  # significant downtrend
        return 1
    else:                                           # no significant trend
        return 0

def increase_decrease_flag_check(price_series, base_price, sig_level=5):
    '''
    :param price_series: The array of the series of price
    :param base_price: The price serving as the
    :param sig_level: n% level for us to determine it as 'significant'
    :return: symbols:
            2: reached +5% situation
            1: reached -5% situation
            0: no significant change in price
    '''
    # check the input argument
    assert type(price_series).__module__ == np.__name__
    # check the relative price change
    relative_price_change = np.divide(100*(price_series-base_price),base_price)
    # retrieve the maximum and minimum changes
    max_price_uptrend = np.amax(relative_price_change)
    max_price_downtrend = np.amin(relative_price_change)
    # check the situation and return the corresponding flags
    if np.absolute(max_price_uptrend)>=sig_level:
        return 2
    elif np.absolute(max_price_downtrend)>=sig_level:
        return 1
    else:
        return 0


