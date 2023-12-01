from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json


def is_array(values):
    if not isinstance(values, np.ndarray):
        raise TypeError(f'Values must be a numpy array, not {type(values)}')


def array_missing_values(values):
    if np.any(pd.isna(values)):
        raise ValueError('There are missing values.')


class Constants:
    """Represents a table of constants"""

    def __init__(self, path):
        self._path = path
        self._factors = None
        self._load()

    def _load(self):
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
        """
        try:
            return self._factors.loc[n][name]
        except KeyError as error:
            raise Exception(f'Cannot find {name} for n={n}')


class MRChart:
    """Plotly chart for X and R"""

    def __init__(self, x_values, labels, x_center, x_status, x_upper_limit, x_lower_limit, r_values, r_status,
                 r_lower_limit, r_upper_limit, r_center, title, x_title, r_title, width, height):
        self._x_values = x_values
        self._labels = labels
        self._y = np.arange(1, self._x_values.shape[0]+1, 1)
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

        self._fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        self._draw()

    @staticmethod
    def _marker_colors(status):
        return np.where(status, 'green', 'red')

    def _control_limit_trace(self, limit, name):
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
        return list(
            zip(
                self._labels,
                [f'{axis_title}: ' + str(value) for value in values]
            )
        )

    def _draw(self):
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
        self._fig.add_trace(self._control_limit_trace(self._r_lower_limit, 'LCL'), row=2, col=1)
        self._fig.add_trace(self._mean_trace(self._r_center), row=2, col=1)

        self._fig.update_layout(title=self._title, template='simple_white', showlegend=False, width=self._width,
                                height=self._height)
        self._fig.update_xaxes(showgrid=False, row=1, col=1)
        self._fig.update_yaxes(title_text=self._x_title, showgrid=False, row=1, col=1)
        self._fig.update_yaxes(title_text=self._r_title, showgrid=False, row=2, col=1)
        self._fig.show()

    def save(self, path):
        self._fig.write_image(path)


class XbarR:
    """For the production of Average and Range Shewart charts from subgroups of data."""
    def __init__(self, title='', r_title='Range', x_title='Average',chart_width=800, chart_height=600):
        self._title = title
        self._x_title = x_title
        self._r_title = r_title
        self._chart_width = chart_width
        self._chart_height = chart_height
        self._constants = Constants(path=Path(__file__).parent / 'constants/factor_values_for_shewart_charts.csv')

        self._n = None  # Number of values in each subgroup
        self._x_center_line = None
        self._x_sigma = None
        self._x_upper_limit = None
        self._x_lower_limit = None
        self._r_center_line = None
        self._r_upper_limit = None
        self._r_lower_limit = None
        self._fitted = False
        self._subgroup_ranges = None
        self._subgroup_means = None
        self._labels = None  # TODO this is unsatisfactory
        self._r_in_limits = None
        self._x_in_limits = None
        self._chart = None

    @staticmethod
    def subgroup_range_mean(values):
        """Calculates the mean and the range of each subgroup of measurements"""
        subgroup_means = np.mean(values, axis=1)
        subgroup_ranges = np.abs(np.max(values, axis=1) - np.min(values, axis=1))
        return subgroup_means, subgroup_ranges

    def fit(self, values):
        """Calculates the control limits for average and range charts"""
        # if not isinstance(values, np.ndarray):
        #     raise Exception(f'Error: values must be a numpy array, not {type(values)}')
        is_array(values)
        array_missing_values(values)
        subgroups, n = values.shape
        if n < 2:
            raise ValueError('The number of samples per subgroup must be greater than one.')
        self._n = n
        A2 = self._constants.constant(n=self._n, name='A2')
        D3 = self._constants.constant(n=self._n, name='D3')
        D4 = self._constants.constant(n=self._n, name='D4')
        subgroup_means, subgroup_ranges = self.subgroup_range_mean(values)
        mean_range = np.mean(subgroup_ranges)
        self._x_center_line = np.mean(values)
        self._x_sigma = A2 * mean_range
        self._x_upper_limit = self._x_center_line + self._x_sigma
        self._x_lower_limit = self._x_center_line - self._x_sigma
        self._r_center_line = mean_range
        self._r_upper_limit = mean_range * D4
        self._r_lower_limit = mean_range * D3
        self._fitted = True

    def set_model(self, n, x_upper_limit, x_lower_limit, x_center_line, r_upper_limit, r_lower_limit, r_center_line, title, x_title, r_title):
        """ Sets the parameters for the chart"""
        self._n = n
        self._x_upper_limit = x_upper_limit
        self._x_lower_limit = x_lower_limit
        self._x_center_line = x_center_line
        self._r_upper_limit = r_upper_limit
        self._r_lower_limit = r_lower_limit
        self._r_center_line = r_center_line
        self._title = title
        self._x_title = x_title
        self._r_title = r_title
        self._fitted = True

    def predict(self, values, labels):
        """
        Transform subgroup data into average and moving range values for plotting
        on a chart. Labels which data is out of control.
        """
        if not self._fitted:
            raise Exception('Error: chart has not been fitted')
        is_array(values)
        is_array(labels)
        array_missing_values(values)
        if not values.shape[1] == self._n:
            raise ValueError(f'Error: the number of subgroups must be the same as that used to calculate the control limits ({self._n})')
        subgroups, n = values.shape
        if n < 2:
            raise Exception('Error: the number of samples per subgroup must be greater than one.')
        self._labels = labels
        self._subgroup_means, self._subgroup_ranges = self.subgroup_range_mean(values)
        self._r_in_limits = self.within_limits(self._subgroup_ranges, self._r_upper_limit, self._r_lower_limit)
        self._x_in_limits = self.within_limits(self._subgroup_means, self._x_upper_limit, self._x_lower_limit)

    def plot(self):
        """
        Plots the average and range values on a chart
        """
        self._chart = MRChart(
            x_values=self._subgroup_means,
            labels=self._labels,
            x_center=self._x_center_line,
            x_status=self._x_in_limits,
            x_upper_limit=self._x_upper_limit,
            x_lower_limit=self._x_lower_limit,
            r_values=self._subgroup_ranges,
            r_status=self._r_in_limits,
            r_upper_limit=self._r_upper_limit,
            r_lower_limit=self._r_lower_limit,
            r_center=self._r_center_line,
            title=self._title,
            # x_title='\u0078\u0304', # X with bar
            x_title=self._x_title,
            r_title=self._r_title,
            width=self._chart_width,
            height=self._chart_height
        )

    def _update_chart(self):
        pass

    def save_chart(self, path):
        """Saves an image of the chart to a file"""
        if self._chart:
            self._chart.save(path)
        else:
            raise Exception('Error: the chart must be plotted before it can be saved')

    @staticmethod
    def within_limits(values, upper_limit, lower_limit):
        """
        Is each value within the control limits?
        :param values: array of values
        :param upper_limit: the upper control limit
        :param lower_limit: the lower control limit
        :return: boolean array
        """
        return (values > lower_limit) & (values < upper_limit)

    def save(self, path):
        if not self._chart:
            raise Exception('Error: the chart must be fitted before it can be saved')
        params = {
            'n': self._n,
            'x_upper_limit': self._x_upper_limit,
            'x_lower_limit': self._x_lower_limit,
            'x_center_line': self._x_center_line,
            'r_upper_limit': self._r_upper_limit,
            'r_lower_limit': self._r_lower_limit,
            'r_center_line': self._r_center_line,
            'title':self._title,
            'x_title': self._x_title,
            'r_title': self._r_title
        }
        with open(path, 'w') as fp:
            json.dump(params, fp)

    @property
    def control_limits(self):
        """Returns the values of the control limits and means"""
        return {
            'X center line': self._x_center_line,
            'X sigma': self._x_sigma,
            'X upper control limit': self._x_upper_limit,
            'X lower limit': self._x_lower_limit,
            'R center line': self._r_center_line,
            'R upper limit': self._r_upper_limit,
            'R lower limit': self._r_lower_limit
        }

    @property
    def out_of_control(self):
        """Returns a DataFrame listing the values that are out of control"""
        df = pd.DataFrame(
            data={
                'labels': self._labels,
                'mean': self._subgroup_means,
                'range': self._subgroup_ranges,
                'mean_within_limits': self._x_in_limits,
                'range_within_limits': self._r_in_limits
            }
        )
        return df[~df['mean_within_limits'] | ~df['range_within_limits']]

    @property
    def averages_ranges(self):
        return self._subgroup_means, self._subgroup_ranges


# class IndividualMR:
#     """Calculates the data required to create an Average and moving range chart for individual values."""
#     # Constants for subgroups of size n=2
#     d2 = 1.128
#     d4 = 3.268
#
#     def __init__(self):
#         self._control_x = None
#         self._control_y = None
#         self._mean_x = None
#         self._x_upper_limit = None
#         self._x_lower_limit = None
#         self._r_upper_limit = None
#         self._r_center = None
#         self._x = None
#         self._y = None
#         self._r_values = None
#         self._x_status = None
#         self._r_status = None
#         self._fitted = False
#
#     def fit(self, control_y, control_x):
#         """
#         Calculates the control chart limits.
#
#         :param control_y: Array of the values of the quality
#         :param control_x: Integer index of the values
#         """
#         self._control_x = control_x
#         self._control_y = control_y
#         self._calculate_limits()
#         self._fitted = True
#
#
#     def predict(self, x, y):
#         """
#         Calculates the maving ranges for the values and determins which values
#         are out of control
#         """
#         if not self._fitted:
#             raise Exception('Not fitted')
#         self._x = x
#         self._y = y
#         self._x_moving_range = np.concatenate(([np.NAN], self.moving_ranges(self._x)))
#         self._x_status = self.x_in_control(x, self._x_upper_limit, self._x_lower_limit)
#         self._r_status = self.mr_in_control(self._x_moving_range, self._r_upper_limit)
#
#     @staticmethod
#     def x_in_control(values, upper_limit, lower_limit):
#         """
#         Is each value within the control limits?
#         :param values: array of values
#         :param upper_limit: the upper control limit
#         :param lower_limit: the lower control limit
#         :return: boolean array
#         """
#         return (values > lower_limit) & (values < upper_limit)
#
#     @staticmethod
#     def mr_in_control(values, upper_limit):
#         """Is each value of the moving range less than the upper control limit"""
#         return (values < upper_limit) | (np.isnan(values))
#
#     @staticmethod
#     def moving_ranges(values):
#         """
#         Calculates the moving ranges for a single dimension array
#         :param values: single dimension array of values
#         :return: numpy array
#         """
#         return np.abs(np.ediff1d(values))
#
#     def _calculate_limits(self):
#         """
#         Calculates the control limits and centers for Individual and
#         Moving Range Charts.
#         """
#         self._mean_x = np.mean(self._control_y)
#         average_moving_range = np.mean(self.moving_ranges(self._control_y))
#         process_limit = 3 / IndividualMR.d2 * average_moving_range
#         self._x_upper_limit = self._mean_x + process_limit
#         self._x_lower_limit = self._mean_x - process_limit
#         self._r_upper_limit = IndividualMR.d4 * average_moving_range
#         self._mean_r = average_moving_range
#
#     @property
#     def mean_x(self):
#         return self._mean_x
#
#     @property
#     def x_upper_limit(self):
#         return self._x_upper_limit
#
#     @property
#     def x_lower_limit(self):
#         return self._x_lower_limit
#
#     @property
#     def r_upper_limit(self):
#         return self._r_upper_limit
#
#     @property
#     def mean_r(self):
#         return self._mean_r
#
#     @property
#     def x_moving_ranges(self):
#         return self._x_moving_range
#
#     @property
#     def x_status(self):
#         return self._x_status
#
#     @property
#     def r_status(self):
#         return self._r_status
#
#     @property
#     def out_of_control_x(self):
#         df = pd.DataFrame(
#             data={
#                 'x': self._x,
#                 'y': self._y,
#                 'x_status': self._x_status,
#                 'r_status': self._r_status
#             }
#         )
#         return df[~df['x_status'] | ~df['mr_status']]
