import os
import sys
import numpy as np
import pandas as pd
import pdb

#2.1
#2.1.2
def gr2r(gamma, gr_infected):
    return 1 + 1 / (gamma) * gr_infected

#TODO drop country column from everywhere

days_infectious = 7
gamma = 1 / float(days_infectious) #transition rate from infected to recovered
random_seed = 12345
np.random.seed(random_seed)

#init_date = '2020-03-01'
#end_date = '2021-06-30' 
mu, sigma = 0.25, 0.15
gr_infected = np.random.normal(mu, sigma)
r0 = gr2r(gamma, gr_infected)
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

# TODO --can be removed
# Deal with potential consecutive zeros in the number of infected
mask = (df['infected_{}'.format(days_infectious)] == 0.0) & (df.groupby('Country/Region').shift(1)['infected_{}'.format(days_infectious)] == 0.0)
df.loc[mask, 'gr_infected_{}'.format(days_infectious)] = - (1 / days_infectious)
if mask.sum() > 0:
    print('     Number of observations with zero infected (with {} infectious days) over two consecutive days: {:}'.format(days_infectious, mask.sum()))

# TODO --can be removed
# Set to NaN observations with very small
# number of cases but very high growth rates
# to avoid these observations acting as
# large outliers
mask = (df['new_cases'] <= 25) & (df['gr_infected_{}'.format(days_infectious)] >= gamma * (5 - 1)) # Implicit upper bound on R
df.loc[mask, ['infected_{}'.format(days_infectious),
            'gr_infected_{}'.format(days_infectious)]] = np.nan

# TODO --can be removed
# Set to NaN observations implausibly
# high growth rates that are likely
# due to data issues 
gamma = 1 / float(days_infectious)
mask = (df['gr_infected_{}'.format(days_infectious)] >= gamma * (10 - 1)) # Implicit upper bound on R
df.loc[mask, ['infected_{}'.format(days_infectious),
                'gr_infected_{}'.format(days_infectious)]] = np.nan

# Remove initial NaN values for growth rates
for country in df['Country/Region'].unique():
  mask = df['Country/Region'] == country
  T = df.loc[mask, ].shape[0]
  df.loc[mask, 'days_since_min_cases'] = range(T)
mask = df['days_since_min_cases'] >= 1
df = df.loc[mask, ]
del df['days_since_min_cases']

# Save final dataset
df.to_csv('{}/DE_Dataset.csv'.format(output_folder), index = False)

#2.1.3



#2.1.4
#2.1.5

#2.2