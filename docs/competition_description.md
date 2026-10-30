## Overview
Amalthea x SCALE present Blind Forecast.

Somewhere on Earth there are ten weather stations. You get 75 years of daily temperature, humidity and wind for each of them, but no names, no coordinates and no dates. For each test window you see ten days of weather at one station; forecast the next five.

You cannot look the answer up. The only thing that works is a model that has learned how weather behaves. One of the ten stations is IIT Gandhinagar; which one is revealed when the contest ends.

Three days is short on purpose. The data is clean, the starter notebook runs in seconds, and a good idea beats a long grind.

## Description
Amalthea x SCALE present Blind Forecast.

Somewhere on Earth there are ten weather stations. We are not telling you where
they are. We are not telling you which years this data covers. You will never see
a date, a latitude, or a place name.

What you get is this: for each of about 890 windows, ten consecutive days of
observed weather at one of the stations: daily high, daily low, humidity, wind.
Your job is to forecast the next five days.

You cannot Google this. No weather app knows the forecast for an unknown place
in an unknown year. No archive helps when you don't know what to look up. The
only thing that works is a model that has actually learned how weather moves:
how a heatwave decays, how a humidity spike precedes a cold front, how a desert
and a coastline recover from the same anomaly differently.

To learn that, you get about 75 years of daily history for each of the ten
stations. That is roughly 270,000 station-days, enough for real machine learning,
and you will need it, because the benchmark you have to beat is not a straw man.

Three of the ten stations are climates most Indians would recognise instantly.
One is a cold desert at 3,500 m. One barely has seasons. One of them is IIT Gandhinagar. Which code is home? Guess in the forum. The map comes out when
the contest ends.

The benchmark problem
Weather forecasting has a brutal baseline: just predict the historical average
for that station and day of the year. It is free, it needs no model, and at five
days out it is almost impossible to beat.

Every score on this leaderboard is relative to that baseline: 1.000 means
you matched climatology exactly, 0.700 means your errors are 30% smaller.
Lower is better. Everything interesting happens between 0.60 and 1.00.

Three entries are already on the board for you to climb past:

Leaderboard entry	Score	What it is
Benchmark: climatology	1.000	The historical average. The floor, and what the starter notebook gives you.
Benchmark: persistence	0.822	Tomorrow = today.
Benchmark: SCALE	0.768	The bar. A tuned statistical forecast. Method revealed when the contest ends.
The starter notebook gives you the data pipeline, a local validation setup, and the
climatology baseline at 1.000. Every model below that line is yours to build.

Who can enter
Individual entry, open to IIT Gandhinagar students. Any department, any year.
No Kaggle experience needed: the starter notebook loads the data, validates,
and writes a valid submission in one run.

Prizes
Place (private leaderboard)	Prize
1st	Mechanical keyboard (worth ₹1,500)
2nd	Game controller (worth ₹1,000)
3rd	Mouse (worth ₹500)
Timeline
When	What
Fri 30 Oct	Competition opens: data, starter notebook, benchmarks on the leaderboard
Sun 1 Nov, 11:59 PM	Final deadline: Kaggle submission + code + write-up
After the deadline	Organisers re-run the top entries' code, then results, the station map, and how the benchmark works
Three days is short on purpose. The data is clean, the starter notebook runs in
seconds, and a good idea beats a long grind.

Evaluation
The metric
Mean absolute error on the scaled value column of your submission.

The scaling is what makes four different variables comparable. Humidity errors run
around 10%, minimum-temperature errors around 2°C; score them in raw units and
humidity decides the competition by accident of units. So every value (yours and
the ground truth's) is divided by that variable's constant in scales.csv, which
is climatology's own error on that variable.

The result is that your score equals

the average, over the four variables, of (your MAE) ÷ (climatology's MAE)

so 1.000 means you matched the historical average, and lower is better. The
starter notebook does the conversion for you in one function call.

Submission format
One row per window, forecast day and variable:

id,value
w0000_1_rh,4.68591
w0000_1_tmax,14.1138
w0000_1_tmin,8.75121
w0000_1_wind,3.42502
w0000_2_rh,4.62962
...
id is {window_id}_{step}_{variable}. value is your prediction for that
variable divided by its constant in scales.csv (so a 33°C forecast for tmax
becomes about 15). There are 17,800 rows.

The starter notebook's submit() builds this from your forecasts in real units
and checks the row count, the ids and for NaNs before writing. Use it.

Public and private leaderboard
30% of the windows form the public leaderboard, stratified so every station and
every season code is represented in proportion. The other 70% decide the result.

The public board is noisy and the gaps between good entries are small. Build a
validation split out of train.csv (same shape, 10 days in, 5 days out) and
trust it more than the public score. The starter notebook shows how. This is the
single most common way people lose competitions of this shape, and we are telling
you up front.

Submission limits
5 submissions per day. Select 2 final submissions for private
scoring before the deadline.