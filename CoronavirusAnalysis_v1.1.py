import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
import seaborn as sns
from bs4 import BeautifulSoup
import requests
import re
import easygui

""" Created by Andrew Gibson a.gibson1@hotmail.co.uk """

def remove_symbols(string):
   """ Removes unwanted symbols from a string """
   symbol_present = len(re.findall("[^a-zA-Z0-9 ]", string)) > 0
   if symbol_present != 0:
       first_symbol = re.findall("[^a-zA-Z0-9 ]", string)[0]
       first_symbol_loc = string.find(first_symbol)
       string = string[:first_symbol_loc]
   string = string.strip()
   return string


def get_percent(child_col, parent_col):
   """ Calculates the child column as a percent of the parent column """
   if parent_col == 0:
       percent_col = 0
   else:
       percent_col = child_col * 100 / parent_col
   return percent_col


def get_r_squared(col1, col2):
   """ Calculates the r squared value for the least squares best fit calculation """
   try:
       diff_col = (col1 - col2) ** 2
   except OverflowError:
       diff_col = float('inf')
   return diff_col


def get_pop():
   """ Takes the country & population table from Wikipedia """
   print("Getting pop %s" % str(datetime.datetime.now()))
   pop_url = r'https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_population'
   pop_table = requests.get(pop_url).text
   soup = BeautifulSoup(pop_table, features="lxml")
   table = soup.find('table', class_='wikitable sortable')
   rows = table.find_all('td')

   i = 1
   line = []
   columns = []

   # Builds the table, removing unwanted symbols
   for row in rows:
       if i in [2, 3]:
           text = row.text.replace(',', '').replace('\xa0', '')
           text = remove_symbols(text)
           line.append(text)
       if i == 6:
           columns.append(line)
           line = []
           i = 0
       i += 1

   columns = np.array(columns)
   pop_df = pd.DataFrame(columns)
   pop_df.columns = ['location', 'population']
   pop_df['population'] = pop_df['population'].astype('int')
   world_pop = sum(pop_df['population'])
   pop_df = pop_df.append({'location': 'World', 'population': world_pop}, ignore_index=True)
   print("Got pop %s" % str(datetime.datetime.now()))
   return pop_df


def merge_dfs(who_df, pop_df, col):
   """ Merges the population table with the covid table to get populations with country's cases """

   print("Merging pop %s" % str(datetime.datetime.now()))
   merged_df = pd.merge(who_df, pop_df, on=col)
   merged_df['total_cases%'] = merged_df.apply(lambda x: get_percent(x['total_cases'], x['population']), axis=1)
   merged_df['total_deaths%'] = merged_df.apply(lambda x: get_percent(x['total_deaths'], x['population']), axis=1)
   merged_df['death_rate%'] = merged_df.apply(lambda x: get_percent(x['total_deaths'], x['total_cases']), axis=1)
   merged_df = merged_df.sort_values('date')
   merged_df = merged_df.loc[merged_df['total_cases'] > 100]
   merged_df['date'] = merged_df['date'].astype('datetime64[ns]')
   merged_df['day_diff'] = (merged_df.groupby('location')['date'].diff()).dt.days
   merged_df['days_since_first'] = (merged_df.groupby('location')['day_diff']).cumsum()
   print("Merged pop %s" % str(datetime.datetime.now()))
   return merged_df


def filter_country(merged_df, country):
   """ Filters the merged table into the country chosen by user """

   print("Filtering by country %s" % str(datetime.datetime.now()))
   country_df = merged_df
   country_df = country_df.loc[(country_df['location'] == country) & (country_df['total_cases'] > 100)]
   country_pop = country_df['population'].min()
   country_df.insert(len(country_df.columns), 'infected_model', country_df['total_cases'].min() /
                      country_pop)
   country_df.insert(len(country_df.columns), 'susceptible_model', 1)
   country_df.fillna(0, inplace=True)
   print("Filtered by country %s" % str(datetime.datetime.now()))
   return country_df, country_pop


def apply_model(modelling_df, to_present, k, B, population):
    """ Takes the filtered table and a pair of k and B values and calculates the number of infected people at each day.
    Then it compares the real data to the modelled data and calculates the sum of r squared (variance) for that pair."""

    loop_start = datetime.datetime.now()
    filtered_df['susceptible_model'] = 1  # Initialising the model columns (SIR model is in percent of population)
    filtered_df['infected_model'] = filtered_df['total_cases'].min() / population
    empty_df = pd.DataFrame(columns=['days_since_first', 'infected_model', 'susceptible_model'], index=[0])

    # If the function is called in the modelling stage then to_present
    # is True and the model is only done up to the current date.
    # If the best pair of k and B values has already been found and the
    # function is now being called to plot the future model, then to_present
    # equals False.
    if to_present:
        days_forward = 0
    else:
        days_forward = 200

    for row in range(len(modelling_df) + days_forward):
        """ Cycle through each row in the filtered dataframe and carry out the calculation for the model.
        The function used to calculate the model is here:
        https://www.maa.org/press/periodicals/loci/joma/the-sir-model-for-spread-of-disease-eulers-method-for-systems
        (it is Euler's method and iterates through the rows using the previous row's value) """

        if row == 0:
            # Since there is no previous row, the first row has to use the initialised values
            prev_inf = modelling_df.iloc[row, modelling_df.columns.get_loc('infected_model')]
            prev_susc = modelling_df.iloc[row, modelling_df.columns.get_loc('susceptible_model')]
            day_difference = modelling_df.iloc[row, modelling_df.columns.get_loc('day_diff')]

        elif row >= len(modelling_df):
            # If the row is after the length of the data i.e. it's in the future, append a new row and start making
            # future entries
            prev_inf = modelling_df.iloc[row - 1, modelling_df.columns.get_loc('infected_model')]
            prev_susc = modelling_df.iloc[row - 1, modelling_df.columns.get_loc('susceptible_model')]
            day_difference = 1
            modelling_df = modelling_df.append(empty_df, ignore_index=True)
            modelling_df.iloc[row, modelling_df.columns.get_loc('days_since_first')] = \
                modelling_df.iloc[row - 1, modelling_df.columns.get_loc('days_since_first')] + 1
            modelling_df.iloc[row, modelling_df.columns.get_loc('day_diff')] = 1

        else:
            # Otherwise, carry out the model on data up to the present day
            prev_inf = modelling_df.iloc[row - 1, modelling_df.columns.get_loc('infected_model')]
            prev_susc = modelling_df.iloc[row - 1, modelling_df.columns.get_loc('susceptible_model')]
            day_difference = modelling_df.iloc[row, modelling_df.columns.get_loc('day_diff')]

        # Carrying out the Euler's method to calculate the current infected and susceptible numbers
        # INFECTED
        modelling_df.iloc[row, modelling_df.columns.get_loc('infected_model')] = prev_inf + (
                (B * prev_susc * prev_inf) - (k * prev_inf)
        ) * day_difference
        # SUSCEPTIBLE
        modelling_df.iloc[row, modelling_df.columns.get_loc('susceptible_model')] = prev_susc - (
                B * prev_susc * prev_inf * day_difference
        )

    # Multiply by population to get real numbers
    modelling_df.loc[:, ['infected_model']] = modelling_df.loc[:, ['infected_model']] * population
    modelling_df.loc[:, ['susceptible_model']] = modelling_df.loc[:, ['susceptible_model']] * population

    # Remove bad data, round up to integer (round up because even 0.2 humans is actually 1 human)
    modelling_df['infected_model'] = modelling_df['infected_model'].dropna().apply(np.ceil)
    modelling_df['susceptible_model'] = modelling_df['susceptible_model'].dropna().apply(np.ceil)

    # Apply r squared function
    modelling_df['r_squared'] = modelling_df.apply(lambda x: get_r_squared(x['infected_model'], x['total_cases']),
                                                  axis=1)
    sum_r_squared = modelling_df['r_squared'].sum()

    # Put params in array
    params = [k, B, sum_r_squared]
    print('k = %.2f, B = %.2f, r^2 = %.0f' % (k, B, sum_r_squared))  # Current set of k and B values being tested
    loop_end = datetime.datetime.now()
    loop_time = loop_end - loop_start

    return modelling_df, params, loop_time


def plot_figures(
        model_df, real_df, country, population, k, B,r_squared, first_day, peak_cases, peak_percent, peak_day):
   """ Creates the graphs and plots data. Also saves graphs as image file. """

   print("Plotting figures %s" % str(datetime.datetime.now()))
   filename = 'Covid Model for %s (k=%.2f, B=%.2f, R^2=%.0f) %s' % \
              (country, k, B, r_squared, str(datetime.datetime.now().strftime('%d-%b-%Y %Hh%Mm%Ss')))
   real_df.to_csv(filename + '.csv')
   fig = plt.figure(figsize=(15, 8))
   chart = fig.add_subplot(111)
   model_chart = fig.add_subplot(211)
   real_chart = fig.add_subplot(212)
   sns.lineplot(x=model_df['days_since_first'], y=model_df['infected_model'],
                         label='k=%.3f, B=%.3f' % (k, B), ls='--', ax=model_chart)
   sns.lineplot(x=real_df['days_since_first'], y=real_df['total_cases'].astype(float), label='Cases', ax=model_chart)
   sns.lineplot(x=real_df['days_since_first'], y=real_df['total_deaths'].astype(float), label='Deaths', ax=model_chart)
   sns.lineplot(x=model_df['days_since_first'], y=model_df['infected_model'],
                label='k=%.3f, B=%.3f' % (k, B), ls='--', ax=real_chart)
   sns.lineplot(x=real_df['days_since_first'], y=real_df['total_cases'].astype(float),
                label='Cases', ax=real_chart, markers=True)
   sns.lineplot(x=real_df['days_since_first'], y=real_df['total_deaths'].astype(float),
                label='Deaths', ax=real_chart, markers=True)
   chart.set_title('Location = %s, Population = %s\n100th Case = %s, Peak Cases = %i (%.2f%%), Peak Date = %s' % (
   country, population, first_day.strftime("%d/%m/%y"), int(peak_cases), peak_percent, peak_day.strftime("%d/%m/%y")))
   real_chart.set_ylabel('Number of People')
   real_chart.set_xlabel('Days Since 100th Case')
   model_chart.set_ylabel('Number of People')
   model_chart.set_xlabel('Days Since 100th Case')
   real_chart.set_xlim(1, len(real_df['total_cases']))
   real_chart.set_ylim(0, max(real_df['total_cases']) * 1.1)
   chart.get_xaxis().set_visible(False)
   chart.get_yaxis().set_visible(False)
   model_chart.legend()
   model_chart.grid()
   real_chart.grid()
   real_chart.get_legend().remove()
   plt.savefig(filename + '.png')
   print("Figures ready %s" % str(datetime.datetime.now()))
   plt.show()


# Location of WHO data
file = r'https://covid.ourworldindata.org/data/ecdc/full_data.csv'
population_df = get_pop()
covid_df = pd.read_csv(file)

# Initialise range of values for parameters to iterate through
# k (or gamma in scientific studies) = Recovery rate
# B = Rate of disease transmission
# These can be between 0.2 and 3 depending on how effective measures are at restricting spread

k_array = np.concatenate((np.arange(0.3, 0.9, 0.01), np.arange(0.9, 3, 0.1)))
B_array = np.concatenate((np.arange(0.3, 0.9, 0.01), np.arange(0.9, 3, 0.1)))
iters = len(k_array) * len(B_array)
print(iters)
params_array = []

# Merge the covid data with population
merge_df = merge_dfs(covid_df, population_df, 'location')

while True:
    # Keep running until issue or cancel

    country = easygui.enterbox('Which country?')

    if country == None or country == '':
        print("User did not enter anything")
        break
    elif country not in merge_df.location.unique():
        print("Country not found")
        continue

    # Filter merged dataframe by country
    filtered_df, population = filter_country(merge_df, country)

    # Cycle through parameters, appending to array where the best will be chosen based on r squared values
    i = 0
    for k in k_array:
        for B in B_array:
            curr_test_df, curr_test_params, loop_time = apply_model(filtered_df, True, k, B, population)
            total_time = loop_time * iters
            time_remaining = loop_time * (iters - i)
            loop_percent = i * 100 / iters
            print("Previous loop took %s. Total time will take ~%s.\n"
                  "Time remaining = %s (%.2f%% complete)." %
                  (str(loop_time), str(total_time), str(time_remaining), loop_percent))
            params_array.append(curr_test_params)
            i += 1

    # Convert to dataframe, assign columns, apply mask, and pick k and B with lowest r squared
    params_df = pd.DataFrame(params_array)
    params_df.columns = ['k', 'B', 'r_squared']
    mask = params_df['r_squared'] == (params_df['r_squared'].min())
    params_df = params_df.loc[mask]
    print(params_df.sort_values('r_squared'))
    best_k = params_df.loc[:, 'k'].iloc[0]
    best_B = params_df.loc[:, 'B'].iloc[0]
    r_squared = params_df.loc[:, 'r_squared'].iloc[0]

    # Now apply model using the best parameters, modelling into the future
    model_df, best_params, loop_time = apply_model(filtered_df, False, best_k, best_B, population)
    real_df, params, loop_time = apply_model(filtered_df, True, best_k, best_B, population)

    # Some calculations on numbers, dates etc
    peak = max(model_df['infected_model'])
    peak_percent = peak * 100 / population
    peak_loc = model_df.loc[model_df['infected_model'] == peak]
    first_day = model_df.iloc[0, model_df.columns.get_loc('date')]
    peak_day = first_day + timedelta(days=peak_loc.iloc[0, peak_loc.columns.get_loc('days_since_first')])

    # Plot the figures
    plot_figures(
        model_df, real_df, country, population, best_k, best_B, r_squared, first_day, peak, peak_percent, peak_day)
