import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb

#2.1
#2.1.2
days_infectious = 7
gamma = 1 / float(days_infectious) #transition rate from infected to recovered

def gr2R(gr_infected):
    #R = 1 + (1 / gamma) * gr_infected
    R = 1 + days_infectious * gr_infected
    return R

def R2gr(R):
    gr_infected = gamma * (R - 1)
    return gr_infected

#TODO drop country column from everywhere
random_seed = 12345
np.random.seed(random_seed)

#init_date = '2020-03-01'
#end_date = '2021-06-30' 

population_DE = 83160000 # 83.16 millions in 2020
input_folder = '.'
output_folder = '.'

#file_name = 'DE_confirmed.csv'

def construct_dataset(file_name, var_name):
    """Convenience function for constructing
    a clean Pandas dataframe from the CSV
    files provided by JH CSSE on their Github
    repo
    
    Args:
        file_name (str): File name / URL of CSV file
        var_name (name): Variable name
    
    Returns:
        df: Dataframe
    """
    df = pd.read_csv(file_name)
    del df['Lat'], df['Long']
    
    # Melt to long format
    df = pd.melt(df, 
                 id_vars = ['Country/Region'], 
                 value_vars = list(df.columns.values[2:]))
    df.rename(columns = {'variable': 'Date',
                         'value': var_name},
              inplace = True)
    # For some countries, data are reported
    # by regions / states; aggregate to country level
    #pdb.set_trace()
    return df
    #return df.groupby(['Country/Region', 'Date']).sum().reset_index() #TODO probably not required

# Read in data on total cases
df = construct_dataset(file_name = '{}/DE_confirmed.csv'.format(input_folder), 
                       var_name = 'total_cases')

#TODO why is the first row dropped? 1 Mar 2020 
# Merge in recovered cases and deaths
for file_name, var_name in zip(['{}/DE_Recovered.csv'.format(input_folder), 
                                '{}/DE_Dead.csv'.format(input_folder)],
                               ['total_recovered', 'total_deaths']):
    df_temp = construct_dataset(file_name = file_name, 
                                var_name = var_name)
    df = pd.merge(df, df_temp, 
                  on = ['Country/Region', 'Date'], 
                  how = 'left')

#pdb.set_trace()
# Clean up the dataframe
df['Date'] = pd.to_datetime(df['Date'])
df.reset_index(inplace = True)
del df['index']

#pdb.set_trace()
# Sort by date
df.sort_values(by = ['Date'], ascending = True,
               inplace = True)

# Construct derived flow variables (new cases / 
# recoveries / deaths)
for var_name in ['cases', 'recovered', 'deaths']:
    df['new_{}'.format(var_name)] = (df['total_{}'.format(var_name)] 
                                     - df.shift()['total_{}'.format(var_name)])

# Construct number of infected
df['infected_{}'.format(days_infectious)] = np.nan
for country in df['Country/Region'].unique():
    mask = df['Country/Region'] == country
    df_country = df.loc[mask, ].copy().reset_index()
    T = df_country.shape[0]
    #pdb.set_trace()
    # Initialize number of infected
    infected = np.zeros(T) * np.nan
    infected[0] = df_country['total_cases'][0]

    # Main loop
    for tt in range(1, T):
        #gamma = 1 / float(days_infectious)

        # Calculate number of infected recursively;
        # In the JH CSSE dataset, there are some
        # data problems whereby new cases are occasionally
        # reported to be negative; in these case, take zero
        # when constructing time series for # of invected,
        # and then change values to NaN's later on
        infected[tt] = ((1 - gamma) * infected[tt - 1] 
                        + np.maximum(df_country['new_cases'][tt], 0.0))
    df.loc[mask, 'infected_{}'.format(days_infectious)] = infected

# TODO --can be removed
# In the original JH CSSE dataset, there are
# some inconsistencies in the data
# Replace with NaN's in these cases
mask = df['new_cases'] < 0
df.loc[mask, 'new_cases'] = np.nan
print('     Inconsistent observations in new_cases in JH CSSE dataset: {:}'.format(mask.sum()))
df.loc[mask, 'infected_{}'.format(days_infectious)] = np.nan

# Calculate growth rate of infected
df['gr_infected_{}'.format(days_infectious)] = ((df['infected_{}'.format(days_infectious)] 
    / df.groupby('Country/Region').shift(1)['infected_{}'.format(days_infectious)]) - 1)

mask = df.groupby('Country/Region').shift(1)['infected_{}'.format(days_infectious)] == 0.0
df.loc[mask, 'gr_infected_{}'.format(days_infectious)] = np.nan

# Remove initial NaN values for growth rates
for country in df['Country/Region'].unique():
  mask = df['Country/Region'] == country
  T = df.loc[mask, ].shape[0]
  df.loc[mask, 'days_since_min_cases'] = range(T)
mask = df['days_since_min_cases'] >= 1
df = df.loc[mask, ]
del df['days_since_min_cases']

'''
# Save final dataset
df.to_csv('{}/DE_Dataset.csv'.format(output_folder), index = False)
'''

#2.1.3
from scipy import signal, misc
import math

window_size = 3 #TODO tune this parameter
gr_infected = np.array(df['gr_infected_7'])
gr_infected_filtered = signal.medfilt(gr_infected, window_size)
df['gr_infected_filtered'] = gr_infected_filtered

var_gr_infected = np.var(gr_infected)
var_gr_infected_filtered = np.var(gr_infected_filtered)
var_epsilon = (2 * window_size) * (var_gr_infected - var_gr_infected_filtered) / (2 * window_size - math.pi) #0.012482567657577066
std_epsilon = math.sqrt(var_epsilon)

# Save final dataset
df.to_csv('{}/DE_Dataset.csv'.format(output_folder), index = False)

#2.1.4
#pdb.set_trace()
#plot time-series of R from the beginning to end date
#computed R - measurement, estimated R- prediction, should overlap
# x axis - R vs. date, possibly jump in data if reporting/testing changes
plt.rcParams.update({'font.size': 15})

def addNoise(cs, sigma):
    '''
    sigma: standard deviation of noise distribution
    ns: noisy output signal
    '''
    #pdb.set_trace()
    rn = np.random.normal(0.0, sigma)
    ns = cs + rn
    return ns


def calcP_prior(P, A, Q):
    '''
    P: old process covariance matrix
    A: state transition matrix
    Q: process noise(perturbation) covariance matrix
    Pn: new process covariance matrix 
    '''
    #pdb.set_trace()
    row, col = np.shape(A)
    Pn = np.zeros((row, col))
    PAt = np.matmul(P, np.transpose(A))
    Pn = np.matmul(A, PAt) + Q
    return Pn


def calcK(P, C, R):
    '''
    Calculate Kalman gain to mix process with measurement
    
    Parameters: 
        P: apriori process covariance matrix
        C: transformation matrix to map parameters to measurment domain
        R: measurement noise(error) covariance matrix
    
    Returns:
        Pn: new process covariance matrix 
    '''
    row, col = np.shape(P)
    K = np.zeros((row, col))
    PCt = np.matmul(P, np.transpose(C))
    CPCtpR = np.matmul(C, PCt) + R
    K = np.matmul(PCt, np.linalg.inv(CPCtpR)) 
    return K

def calcEst(mdl, mes, K, C): 
    '''
    Calculate estimated state as a mix of process and measurement
    mdl: predicted/model state vector (according to theoretical formulas)
    mes: measurement state vector (containing measurement error)
    C: transformation matrix to map state parameters to measurement domain
    xh: xhat estimated state vector based on Kalman gain
    '''
    z = mes - np.matmul(C, mdl)
    Kz = np.matmul(K, z)
    xh = mdl + Kz
    return xh

def calcP_posterior(P, K, C):
    '''
    P: old process covariance matrix
    C: transformation matrix to map state parameters to measurement domain
    Pn: new process covariance matrix 
    '''
    row, col = np.shape(P)
    Pn = np.zeros((row, col))
    CP = np.matmul(C, P)
    Pn = P - np.matmul(K, CP)
    return Pn

# Generate state equation to obtain the state transition matrix A
#pdb.set_trace()

R_measured = gr2R(gr_infected) # Measurement of R TODO does this include the noise??
N = R_measured.shape[0] #TODO initialize it very early 

R_estimate = [] # R
mu_gr_infected_0, sigma_gr_infected_0 = 0.25, 0.15
var_gr_infected_0 = sigma_gr_infected_0
gr_infected_0 = np.random.normal(mu_gr_infected_0, sigma_gr_infected_0)
R_0 = gr2R(gr_infected_0)
R_t = R_0
std_eta = 0.5 * std_epsilon
var_eta = std_eta ** 2

for _ in range(N):
    R_t = addNoise(R_t, std_eta)
    R_estimate.append(R_t)
R_estimate = np.array(R_estimate)
#pdb.set_trace()
plt.figure(figsize=(16, 8))        
#plt_R = plt.subplot(2, 2, 1)
plt.plot(R_measured, label = "Measurement")
plt.plot(R_estimate, label = "Estimate")
#plt_R.plot(R_optimal, label = "Optimal") # Optimal estimation, output of Kalman filter

plt.ylabel("R")
plt.xlabel("Number of days") # TODO or date?

#plt.title("2-D Kalman Filter with eta = {} and sigma = {}".format(var_eta, var_gamma))
#Optimalcd 
# TODO Kalman init function
A = np.array([[1]]) # also referrred as F
C = np.array([[1]]) # also referred as H
Q =  np.array([[var_eta]])
R =  np.array([[var_epsilon]])
#pdb.set_trace()
var_R_0 = (1 / gamma) * var_gr_infected_0
P = np.array([[var_R_0]])
R_estimate = R_estimate[:, np.newaxis]
R_measured = R_measured[:, np.newaxis]
R_optimal = []

for i in range(N):
    P = calcP_prior(P, A, Q)
    K = calcK(P, C, R)
    xhat = calcEst(R_estimate[i], R_measured[i], K, C)
    R_optimal.append(xhat)
    P = calcP_posterior(P, K, C)

R_optimal = np.squeeze(np.array(R_optimal))

#2.1.5
plt.plot(R_optimal, label = "Optimal")
plt.legend()
plt.savefig('filter.png')

#TODO plot kalman gain coefficients for varying noises
#2.2
