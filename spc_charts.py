"""Plotly charts for Statistical Process control"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def array_missing_values(values):
    """
    Raises a ValueError if there are missing values in values.
    """
    if np.any(pd.isna(values)):
        raise ValueError('There are missing values.')


class Constants:
    """
    Represents a table of constants

    Methods:
        constant()
    """

    def __init__(self, path):
        """
        :param path: The path and filename of the constants CSV file.
        """
        self._path = path
        self._factors = None
        self._load()

    def _load(self):
        """
        loads the constants csv file
        :raises: Exception: if the constants file is not found.
        """
        try:
            self._factors = pd.read_csv(self._path, index_col=0)
        except FileNotFoundError as e:
            raise Exception(f'Error: cannot find {self._path.name} at {self._path.parent}')

    def constant(self, n, name):
        """
        Returns the value of the constant given the constant's name and the value of n.
        :param n: the value of n
        :param name: the name of the constant
        :return: the value of the constant
        :raises: Exception: if the constant is not found
        """
        try:
            return self._factors.loc[n][name]
        except KeyError as e:
            raise Exception(f'Cannot find {name} for n={n}')


class XRChart:
    """
    Plotly chart for X and R

    Methods:
        draw()
        save()
    """

    def __init__(self, x_values, labels, x_center, x_status, x_upper_limit, x_lower_limit, r_values, r_status,
                 r_lower_limit, r_upper_limit, r_center, title, x_title, r_title, width, height, show_r_lower_limit):
        """
        :param x_values: the values to be plotted on the X chart
        :param labels: the labels for the values plotted on the X and R charts
        :param x_center: The center line of the X chart
        :param x_status: Boolean array: whether the values in x_values are in control.
        :param x_upper_limit: the upper limit for the X chart
        :param x_lower_limit: the lower limit for the X chart
        :param r_values: the values to be plotted on the R chart
        :param r_status: Boolean array: whether the values in x_values are in control.
        :param r_lower_limit: the lower limit of the R chart
        :param r_upper_limit: the upper limit of the R chart
        :param r_center: the center line of the R chart
        :param title: the title for the chart
        :param x_title: the title of the y-axis of the X chart
        :param r_title: the title of the y-axis of the R chart
        :param width: the width of the chart in pixels
        :param height: the height of the chart in pixels
        """
        self._x_values = x_values
        self._labels = labels
        self._y = np.arange(1, self._x_values.shape[0] + 1, 1)
        self._x_center = x_center
        self._x_status = x_status
        self._r_status = r_status
        self._x_upper_limit = x_upper_limit
        self._x_lower_limit = x_lower_limit
        self._r_values = r_values
        self._r_upper_limit = r_upper_limit
        self._r_lower_limit = r_lower_limit
        self._r_center = r_center
        self._title = title
        self._x_title = x_title
        self._r_title = r_title
        self._width = width
        self._height = height
        self._show_r_lower_limit = show_r_lower_limit

        self._fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        self.draw()

    @staticmethod
    def _marker_colors(status):
        """ Sets the color of the markers based on whether they are in or out of control"""
        return np.where(status, 'green', 'red')

    def _control_limit_trace(self, limit, name):
        """
        Creates a trace for control limits
        :param limit: the control limit
        :param name: the name to give the trace
        """
        control_limit_x = [min(self._y), max(self._y)]
        control_limit_y = [limit] * 2
        return go.Scatter(
            x=control_limit_x,
            y=control_limit_y,
            hoverinfo='skip',
            name=name,
            mode='lines',
            line=dict(
                color='black',
                dash='dash',
                width=1
            )
        )

    def _mean_trace(self, value):
        """
        Creates a trace for center lines
        :param value: the value of the center line
        """
        control_limit_x = [min(self._y), max(self._y)]
        control_limit_y = [value] * 2
        return go.Scatter(
            x=control_limit_x,
            y=control_limit_y,
            mode='lines',
            hoverinfo='skip',
            line=dict(color='black', width=1),
            name='Mean'
        )

    def _value_trace(self, values, status, marker_labels):
        """
        Creates a trace to plot the values
        :param values: the values to plot
        :param status: whether the vales are in or out of control
        :param marker_labels: the labels for the values
        """
        return go.Scatter(
            x=self._y,
            y=values,
            name='',
            mode='markers+lines',
            line=dict(color='gray'),
            marker=dict(
                size=10,
                opacity=0.7,
                color=self._marker_colors(status)
            ),
            customdata=marker_labels,
            hovertemplate='<b>%{customdata[0]}</b><br>%{customdata[1]}',
        )

    def _marker_labels(self, axis_title, values):
        """
        Creates the hover text for each marker
        :param axis_title: the title of the y-axis
        :param values: the values for each marker
        """
        return list(
            zip(
                self._labels,
                [f'{axis_title}: ' + str(value) for value in values]
            )
        )

    def draw(self):
        """Creates a plotly chart"""
        self._fig.add_trace(
            self._value_trace(
                values=self._x_values,
                status=self._x_status,
                marker_labels=self._marker_labels(self._x_title, self._x_values)
            ),
            row=1, col=1
        )
        self._fig.add_trace(self._control_limit_trace(self._x_lower_limit, 'LCL'), row=1, col=1)
        self._fig.add_trace(self._control_limit_trace(self._x_upper_limit, 'UCL'), row=1, col=1)
        self._fig.add_trace(self._mean_trace(self._x_center), row=1, col=1)

        self._fig.add_trace(
            self._value_trace(
                values=self._r_values,
                status=self._r_status,
                marker_labels=self._marker_labels(self._r_title, self._r_values)
            ),
            row=2, col=1
        )
        self._fig.add_trace(self._control_limit_trace(self._r_upper_limit, 'UCL'), row=2, col=1)
        if self._show_r_lower_limit:
            self._fig.add_trace(self._control_limit_trace(self._r_lower_limit, 'LCL'), row=2, col=1)
        self._fig.add_trace(self._mean_trace(self._r_center), row=2, col=1)

        self._fig.update_layout(title=self._title, template='simple_white', showlegend=False, width=self._width,
                                height=self._height)
        self._fig.update_xaxes(showgrid=False, row=1, col=1)
        self._fig.update_yaxes(title_text=self._x_title, showgrid=False, row=1, col=1)
        self._fig.update_yaxes(title_text=self._r_title, showgrid=False, row=2, col=1)
        self._fig.show()

    def save(self, path):
        """Saves the chart as an image file"""
        self._fig.write_image(path)


class XbarR:
    """
    Shewart Average and Range chart

    Methods:
        fit()
        predict()
        plot()
        save()
        load()
        save_chart()

    Properties:
        out_of_control
        averages_ranges
    """

    has_r_lower_limit = True
    chart_type = 'XBarR'

    def __init__(self, title='', r_title='Range', x_title='Average', chart_width=800, chart_height=600):
        """
        :param title: The chart title
        :param r_title: The title for the y-axis of the range chart
        :param x_title: The title for the y-axis of the mean chart
        :param chart_width: The chart width in pixels, defaults to 800.
        :param chart_height: The chart height in pixels, defaults to 600.
        """
        self._title = title
        self._x_title = x_title
        self._r_title = r_title
        self._chart_width = chart_width
        self._chart_height = chart_height

        self._constants = Constants(path=Path(__file__).parent / 'constants/factor_values_for_shewart_charts.csv')
        """Instance variable for the constants object"""
        self._n = None
        """Number of values in each subgroup"""
        self._x_center_line = None
        """The values of the center line for the subgroup averages chart"""
        self._x_sigma = None
        """The value of sigma for the subgroup averages chart"""
        self._x_upper_limit = None
        """The upper limit for the subgroup averages chart"""
        self._x_lower_limit = None
        """The lower limit for the subgroup averages chart"""
        self._r_center_line = None
        """The center line for the subgroup range chart"""
        self._r_upper_limit = None
        """The upper limit for the subgroup range chart"""
        self._r_lower_limit = None
        """The lower limit for the subgroup range chart"""
        self._fitted = False
        """True the model has been fitted, False if it has not."""
        # self._subgroup_ranges = None
        self._r_values = None
        """The range of each subgroup of the values to be plotted"""
        # self._subgroup_means = None
        self._x_values = None
        """The means of each subgroup calculated by the predict method"""
        self._labels = None
        """The labels given to the subgroups by the user"""
        self._r_in_limits = None
        """Whether each subgroup range is within the range chart control limits"""
        self._x_in_limits = None
        """Whether each subgroup mean is within the mean chart control limits"""
        self._chart = None
        """True is the chart has been plotted"""

    @staticmethod
    def _subgroup_range_mean(values):
        """
        Calculates the subgroup ranges and subgroup means
        :param values: numpy.Array each row is a subgroup and each column is a value of the subgroup.
        :return: The subgroup means and the subgroup ranges.
        """
        """Calculates the mean and the range of each subgroup of measurements"""
        subgroup_means = np.mean(values, axis=1)
        subgroup_ranges = np.abs(np.max(values, axis=1) - np.min(values, axis=1))
        return subgroup_means, subgroup_ranges

    def fit(self, values, labels):
        """
        Calculates the control limits and center lines for the mean and range charts.
        :param values: The values to be used to calculate the control limits. The subgroup size must be a least 2.
        :type values: list, tuple or array
        :params labels: The labels for each subgroup
        :type labels: list, tuple or array of strings
        :type values: numpy.Array where each row is a subgroup and each column is a value in a subgroup.
        :raises:
            ValueError: if there are missing values in values.
            ValueError: if there are less than 2 columns in values.

        """
        values = np.array(values)
        labels = np.array(labels)
        array_missing_values(values)
        subgroups, n = values.shape
        if n < 2:
            raise ValueError('The number of samples per subgroup must be greater than one.')
        self._n = n
        A2 = self._constants.constant(n=self._n, name='A2')
        D3 = self._constants.constant(n=self._n, name='D3')
        D4 = self._constants.constant(n=self._n, name='D4')
        subgroup_means, subgroup_ranges = self._subgroup_range_mean(values)
        mean_range = np.mean(subgroup_ranges)
        self._x_center_line = np.mean(values)
        self._x_sigma = A2 * mean_range
        self._x_upper_limit = self._x_center_line + self._x_sigma
        self._x_lower_limit = self._x_center_line - self._x_sigma
        self._r_center_line = mean_range
        self._r_upper_limit = mean_range * D4
        self._r_lower_limit = mean_range * D3
        self._fitted = True
        self.predict(values, labels)

    def predict(self, values, labels):
        """
        Transform subgroup values into mean and moving range values for plotting
        on a chart. Calculates whether the means and ranges are outside their respective limits.
        :params values: Each row is a subgroup and each column is a value of the subgroup.
        :type values: list, tuple or array
        :params labels: The labels for each subgroup
        :type labels: list, tuple or array of strings
        :raises:
            Exception: if the control limits have not been calculated.
            ValueError: if there are missing values in values.
            ValueError: if the number of values in the subgroups used to calculate the control limits is different
                        to the number of subgroups in values.

        """
        if not self._fitted:
            raise Exception('Error: chart has not been fitted')
        values = np.array(values)
        array_missing_values(values)
        if not values.shape[1] == self._n:
            raise ValueError(
                f'Error: the number of subgroups must be the same as that used to calculate the control limits ({self._n})')
        self._labels = np.array(labels)
        self._x_values, self._r_values = self._subgroup_range_mean(values)
        self._r_in_limits = self._in_control(self._r_values, self._r_upper_limit, self._r_lower_limit)
        self._x_in_limits = self._in_control(self._x_values, self._x_upper_limit, self._x_lower_limit)

    def plot(self):
        """
        Plots the mean and range values on Plotly chart
        """
        self._chart = XRChart(
            x_values=self._x_values, # same
            labels=self._labels,
            x_center=self._x_center_line,
            x_status=self._x_in_limits,
            x_upper_limit=self._x_upper_limit,
            x_lower_limit=self._x_lower_limit,
            r_values=self._r_values, # same
            r_status=self._r_in_limits,
            r_upper_limit=self._r_upper_limit,
            r_lower_limit=self._r_lower_limit,
            r_center=self._r_center_line,
            title=self._title,
            # x_title='\u0078\u0304', # X with bar
            x_title=self._x_title,
            r_title=self._r_title,
            width=self._chart_width,
            height=self._chart_height,
            show_r_lower_limit=type(self).has_r_lower_limit
        )

    def _update_chart(self):
        pass

    def save_chart(self, path):
        """
        Saves an image of the chart to a file.
        :params path: The full path and filename to save to. The file type is
            automatically determined by the filename extension. Allowed files
            types are PNG, JPEG, WebP, SVG and PDF.
        """
        if self._chart:
            self._chart.save(path)
        else:
            raise Exception('Error: the chart must be plotted before it can be saved')

    @staticmethod
    def _in_control(values, upper_limit, lower_limit):
        """
        Is each value within the control limits?
        :param values: array of values
        :param upper_limit: the upper control limit
        :param lower_limit: the lower control limit
        :return: boolean array
        """
        return (values >= lower_limit) & (values <= upper_limit)

    def _params_to_dict(self):
        """Converts the chart parameters to a dictionary"""
        return {
            'type': type(self).chart_type,
            'n': self._n,
            'x_upper_limit': self._x_upper_limit,
            'x_lower_limit': self._x_lower_limit,
            'x_center_line': self._x_center_line,
            'r_upper_limit': self._r_upper_limit,
            'r_lower_limit': self._r_lower_limit,
            'r_center_line': self._r_center_line,
            'title': self._title,
            'x_title': self._x_title,
            'r_title': self._r_title
        }

    def _dict_to_params(self, params):
        """
        Sets the values of the chart parameters from a dictionary.
        :param params: The values of the chart parameters as a dictionary
        """
        self._n = params['n']
        self._x_upper_limit = params['x_upper_limit']
        self._x_lower_limit = params['x_lower_limit']
        self._x_center_line = params['x_center_line']
        self._r_upper_limit = params['r_upper_limit']
        self._r_lower_limit = params['r_lower_limit']
        self._r_center_line = params['r_center_line']
        self._title = params['title']
        self._x_title = params['x_title']
        self._r_title = params['r_title']

    def save(self, path):
        """
        Saves the chart parameters to a json file.
        :param path: The full path and filename.
        """
        if not self._fitted:
            raise Exception('Error: the chart must be fitted before it can be saved')
        params = self._params_to_dict()
        with open(path, 'w') as fp:
            json.dump(params, fp)

    def load(self, path):
        """
        Loads the chart parameters from a JSON file.
        :param path: The full path and filename of the file to load
        """
        with open(path, 'r') as fp:
            params = json.load(fp)
        self._dict_to_params(params)
        self._fitted = True

    @property
    def params(self):
        """Returns the chart parameters"""
        return self._params_to_dict()

    @params.setter
    def params(self, params):
        """
        Sets the chart parameters
        :param params: The chart parameters
        :type params: dictionary
        :return:
        """
        self._dict_to_params(params)
        self._fitted = True

    @property
    def out_of_control(self):
        """pandas.DataFrame listing the subgroups where the ranges or means are out of control"""
        df = pd.DataFrame(
            data={
                'labels': self._labels,
                'mean': self._x_values,
                'range': self._r_values,
                'mean_within_limits': self._x_in_limits,
                'range_within_limits': self._r_in_limits
            }
        )
        return df[~df['mean_within_limits'] | ~df['range_within_limits']]

    @property
    def x_r_values(self):
        """
        The predicted subgroup means and ranges.
        """
        return self._x_values, self._r_values


class IndividualMR(XbarR):
    """
    Shewart Individual and Moving Range chart

    Methods:
        fit()
        predict()
        plot()
        save()
        load()
        save_chart()

    Properties:
        out_of_control
        averages_ranges
    """

    has_r_lower_limit = False
    chart_type = 'IndividualMR'

    def fit(self, values, labels):
        """
        Calculates the control limits and center lines for the mean and range charts.
        :param values: The values to be used to calculate the control limits. The subgroup size must be a least 2.
        :type values: list, tuple or array
        :params labels: The labels for each value
        :type labels: list, tuple or array of strings
        :type values: numpy.Array where each row is a subgroup and each column is a value in a subgroup.
        :raises:
            ValueError: if there are missing values in values.
            ValueError: if there are more than one column in values.

        """
        values = np.array(values)
        labels = np.array(labels)
        array_missing_values(values)
        # subgroups, n = values.shape
        if values.ndim > 1:
            raise ValueError('The number of samples per subgroup must be one.')
        self._n = 1
        self._x_center_line = np.mean(values)
        average_moving_range = np.mean(self.moving_ranges(values))
        self._x_sigma = self._constants.constant(n=2, name='d2') * average_moving_range
        self._x_upper_limit = self._x_center_line + (3 * self._x_sigma)
        self._x_lower_limit = self._x_center_line - (3 * self._x_sigma)
        self._r_upper_limit = self._constants.constant(n=2, name='D4') * average_moving_range
        self._r_center_line = average_moving_range
        self._r_lower_limit = 0
        self._fitted = True
        self.predict(values, labels)

    def predict(self, values, labels):
        """
        Calculates the moving range values for plotting on the Range chart.
        Calculates whether the values and moving ranges are outside their respective limits.
        :params values: the values
        :type values: list, tuple or array
        :params labels: The labels for each value
        :type labels: list, tuple or array of strings
        :raises:
            Exception: if the control limits have not been calculated.
            ValueError: if there are missing values in values.
            ValueError: if there are more than one column in values.

        """
        if not self._fitted:
            raise Exception('Error: chart has not been fitted')
        values = np.array(values)
        array_missing_values(values)
        if values.ndim > 1:
            raise ValueError('The number of samples per subgroup must be one.')
        self._x_values = values
        self._labels = np.array(labels)
        self._r_values = np.concatenate(([np.NAN], self.moving_ranges(self._x_values)))
        self._x_in_limits = self._in_control(self._x_values, self._x_upper_limit, self._x_lower_limit)
        self._r_in_limits = self._in_control(self._r_values, self._r_upper_limit, self._r_lower_limit)

    @staticmethod
    def moving_ranges(values):
        """
        Calculates the moving ranges for a single dimension array
        :param values: single dimension array of values
        :return: numpy array
        """
        return np.abs(np.ediff1d(values))


