# Covid
Coronavirus analysis and modelling

(It's only my second Python project ever so I'm sure there's lots of room for improvement but please I'd love some criticism)

The way it works is it plots the number of cases per day, generates a best fit statistical model, then extrapolates the model into the distant future to show how the outbreak might unfold for a chosen country.

The model I used was the SIR model which describes how the susceptible (S), infected (I) and recovered (R) populations change with time through an outbreak. The two defining parameters in the SIR model are the transmission rate, B, and the recovery rate, k. This model describes those populations as percentages as shown below.

The populations of each country are taken from Wikipedia and multiplied by the percentages given by the SIR model to show real numbers. My program cycles through every possible set of values of B & k and finds the pair which have the best least squares fit with the real data. It then takes these values and extrapolates into the future by 200 days to show the entire outbreak. This is shown in the upper graph of the figure, whereas the lower graph is a zoomed in view of the real data.

All of it is automatic where the only user input is to select a country. The program extracts Covid-19 data from an online source (https://ourworldindata.org/coronavirus), as well as population data from Wikipedia (https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_population).


Improvements I'd like to do but would probably waste far too much time doing:

Improving the regression fit since it sometimes doesn't seem to work perfectly, especially for the countries heavily flattening their curves

Understand the theoretical side of the SIR model more to get a proper function instead of using Euler's method

Speed up the loops

Add a slider for the user to change the values of B and k in real-time and see how the graphs look

Show error bars

Plot more info like multiple countries? Death rates? etc
