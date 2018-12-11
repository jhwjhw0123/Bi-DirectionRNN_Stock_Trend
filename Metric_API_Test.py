import numpy as np
import pandas as pd
from Metric_Computation import *
time_series = np.array([[1,2,4,6,7,8,9],[333,444,666,888,999,1111,976],[11,21,31,41,39,32,28]])
high_series = np.array([[1.2,2.1,4.3,6.4,7.2,8.1,9.5],[335,446,667,889,1002,1128,982],[11.8,22,33,43.6,40.5,33.1,29.3]])
low_series = np.array([[0.8,1.7,3.6,5.2,6.8,7.4,8.5],[331,443,664,883,992,1107,973],[10.2,20.1,29.8,39.7,38.4,31,26.5]])
n_ma_list = [1,2,3,4]
# time_series_SMA = SMA_batch_compute(time_series,n_ma_list,comp_mode='loop')
# print(time_series_SMA)
# time_series_EMA_loop = EMA_batch_computation(time_series,n_ma_list,comp_mode='loop')
# time_series_EMA_vector = EMA_batch_computation(time_series,n_ma_list,comp_mode='vector')
# print(time_series_EMA_loop)
# print(time_series_EMA_vector)
#
# def ema_ref(values, period):
#     values = np.array(values)
#     return pd.ewma(values, span=period, adjust=True)[-1]
#
#
# time_series_EMA_ref = []
# for ind_ma_len in n_ma_list:
#     time_series_EMA_ref.append(ema_ref([1,2,4,6],ind_ma_len))
# print(time_series_EMA_ref)

time_series_stoch_K, time_series_stoch_D = STOCH_batch_compute(time_series,high_series,low_series,n_ma_list)
print(time_series_stoch_K)
print(time_series_stoch_D)

time_series_DI_up, time_series_DI_down = ADX_batch_compute(time_series,high_series,low_series,n_ma_list)
print(time_series_DI_up)
print(time_series_DI_down)

# time_series_CCI = CCI_batch_compute(time_series,high_series,low_series,n_ma_list)
# print(time_series_CCI)