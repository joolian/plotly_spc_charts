""" Example of creating Average and Range chart for subgroups"""
import pandas as pd
from spc_charts import XbarR
import json

if __name__ == '__main__':
    # Get some data
    data = pd.read_csv('spc_example_data_cooling_water.csv')
    labels = data['Date'].to_numpy()
    values = data[[col for col in data.columns.values if 'Pressure' in col]].to_numpy()
    # Create the chart object
    chart = XbarR(title='Water cooling pressure', x_title='Subgroup mean', r_title='Subgroup range')
    # Calculate the control limits
    chart.fit(values=values[:12, :], labels=labels[:12])
    # Plot the chart
    chart.plot()
    # Calculate the ranges and means for new data
    chart.predict(values=values, labels=labels)
    # plot the chart
    chart.plot()
    # Save the chart as an SVG file
    chart.save_chart('Water_chart.svg')
    # Get the values that are outside the control limits
    print(chart.out_of_control)
    # Get the means and ranges of the data plotted on the chart
    means, ranges = chart.averages_ranges
    print(means)
    print(ranges)
    # Save to a json file, the parameters required for a chart such as control limits.
    chart.save('model_params.json')
    # create a new chart object and load the model parameters from a json  file
    another_chart = XbarR()
    another_chart.load('model_params.json')
    another_chart.predict(values=values, labels=labels)
    another_chart.plot()