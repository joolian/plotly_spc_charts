""" Example of creating and Individual Moving Range chart"""
import pandas as pd
from spc_charts import IndividualMR
import string

if __name__ == '__main__':
    # Get some data
    values = [39, 41, 41, 41, 43, 44, 41, 42, 40, 41, 44, 40, 41, 43, 47]
    labels = list(string.ascii_uppercase)[: len(values)]
    chart = IndividualMR(title='Individuals chart', x_title='Values', r_title='Moving range')
    # Calculate the control limits
    chart.fit(values=values, labels=labels)
    # Plot the chart
    chart.plot()
    # Calculate the ranges and means for new data
    chart.predict(values=values, labels=labels)
    # plot the chart
    chart.plot()
    # Save the chart as an SVG file
    chart.save_chart('individual.svg')
    # Get the values that are outside the control limits
    print(chart.out_of_control)
    # Get the means and ranges of the data plotted on the chart
    means, ranges = chart.x_r_values
    print(means)
    print(ranges)
    # Save to a json file, the parameters required for a chart such as control limits.
    chart.save('model_params.json')
    # create a new chart object and load the model parameters from a json  file
    another_chart = IndividualMR()
    another_chart.load('model_params.json')
    another_chart.predict(values=values, labels=labels)
    another_chart.plot()
