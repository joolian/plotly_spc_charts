"""Plotly charts for Statistical Process control"""

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from importlib_resources import files
from plotly.subplots import make_subplots


def array_missing_values(values):
    """
    Check an array for missing values.

    :param values: array of values
    :raises: ValueError: if there is a missing value
    """
    if np.any(pd.isna(values)):
        raise ValueError('There are missing values.')


class Constants:
    """
    Represents a table of constants for values of n from 2 to 100:
     A, A2, A3, B3, B4, B5, B6, c4, d2, d3, D1, D2, D3, D4, E2

    Methods:
        constant()
    """

    def __init__(self):
        self._path = files('spc_charts').joinpath('factor_values_for_shewart_charts.csv')
        self._factors = None
        self._load()

    def _load(self):
        """
        Loads the constants from a csv file

        :raises: Exception: if the constants file is not found.
        """
        try:
            self._factors = pd.read_csv(self._path, index_col=0)
        except FileNotFoundError as e:
            raise Exception(f'Error: cannot find {self._path.name} at {self._path}')

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


class RunChart:
    """
    Plotly run chart

    Methods:
        save()
    """

    def __init__(self, x_values, labels, x_center, title, x_title, width, height):
        """
        :param x_values: The values to plot. Must be numpy.array like
        :param labels: The labels for each of the x_values.
        :param x_center: The value of the charts center line.
        :param title: The title of the chart
        :param x_title: The title for the y-axis
        :param width: The width of the chart in pixels
        :param height: The height of the chart in pixels
        """
        self._x_values = np.array(x_values)
        self._labels = np.array(labels)
        self._x = np.arange(1, self._x_values.shape[0] + 1, 1)
        self._x_center = x_center
        self._title = title
        self._x_title = x_title
        self._width = width
        self._height = height
        self._fig = None
        self._draw()

    def _marker_labels(self, axis_title, values):
        """
        Creates the hover text for each marker.

        :param axis_title: the title of the y-axis
        :param values: the values for each marker
        """
        return list(
            zip(
                self._labels,
                [f'{axis_title}: ' + str(value) for value in values]
            )
        )

    def _draw(self):
        """Draws the chart and shows it."""
        center_line_x = [min(self._x), max(self._x)]
        center_line_y = [self._x_center] * 2
        self._fig = go.Figure()
        self._fig.add_trace(
            go.Scatter(
                x=self._x,
                y=self._x_values,
                name='',
                mode='markers+lines',
                line=dict(color='gray'),
                marker=dict(
                    size=10,
                    opacity=0.7,
                    color='green'
                ),
                customdata=self._marker_labels(self._x_title, self._x_values),
                hovertemplate='<b>%{customdata[0]}</b><br>%{customdata[1]}',
            )
        )
        self._fig.add_trace(
            go.Scatter(
                x=center_line_x,
                y=center_line_y,
                mode='lines',
                customdata=[f'Median: {self._x_center}'] * 2,
                hovertemplate='<b>%{customdata}',
                line=dict(
                    color='black',
                    dash='dash',
                    width=1
                ),
                name=''
            )
        )
        self._fig.update_layout(
            title=self._title,
            template='simple_white',
            showlegend=False,
            width=self._width,
            height=self._height
        )
        self._fig.update_xaxes(showgrid=False)
        self._fig.update_yaxes(title_text=self._x_title, showgrid=False)

    def save(self, path):
        """
        Saves the chart as an image file

        :param path: the full path and filename to save the file to.
        """
        self._fig.write_image(path)


class XRChart:
    """
    Plotly chart for X and R

    Methods:
        draw()
        save()
        update()
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
        self._x = labels
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
        self._fig_widget = None
        self._create()

    @staticmethod
    def _marker_colors(status):
        """ Sets the color of the markers based on whether they are in or out of control"""
        return np.where(status, 'green', 'red')

    def _marker_labels(self, axis_title, values):
        """
        Creates the hover text for each marker.

        :param axis_title: the title of the y-axis
        :param values: the values for each marker
        """
        return list(
            zip(
                self._x,
                [f'{axis_title}: ' + str(value) for value in values]
            )
        )

    def _limit_labels(self, value, name):
        """ Creates the text for the hover labels for the center or control limit lines."""
        return np.repeat(f'{name}: {value}', self._x.size)

    def _limit_trace(self, limit, name, kind):
        """
        Creates a trace for ceneter line or control limits

        :param limit: the control limit
        :param name: the name of the trace to be used in hover text
        :param kind: 'center' for a center line or 'limit' for a control limit
        """
        if kind == 'center':
            line = dict(color='black', width=1)
        else:
            line = dict(color='black', dash='dash', width=1)
        return go.Scatter(
            x=self._x,
            y=np.repeat(limit, self._x.size),
            # customdata=np.repeat(f'{name}: {limit}', self._x.size),
            customdata=self._limit_labels(limit, name),
            hovertemplate='<b>%{customdata}</b>',
            name='',
            mode='lines',
            line=line
        )

    def _value_trace(self, values, status, marker_labels):
        """
        Creates a trace to plot the values.

        :param values: the values to plot
        :param status: whether the values are in or out of control
        :param marker_labels: the labels for the values
        """
        return go.Scatter(
            x=self._x,
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
            showlegend=False
        )

    def _create(self):
        """Creates a plotly chart and shows it."""
        self._fig.add_trace(
            self._value_trace(
                values=self._x_values,
                status=self._x_status,
                marker_labels=self._marker_labels(self._x_title, self._x_values)
            ),
            row=1, col=1
        )
        self._fig.add_trace(self._limit_trace(self._x_upper_limit, 'UCL', 'limit'), row=1, col=1)
        self._fig.add_trace(self._limit_trace(self._x_center, 'CL', 'center'), row=1, col=1)
        self._fig.add_trace(self._limit_trace(self._x_lower_limit, 'LCL', 'limit'), row=1, col=1)

        self._fig.add_trace(
            self._value_trace(
                values=self._r_values,
                status=self._r_status,
                marker_labels=self._marker_labels(self._r_title, self._r_values)
            ),
            row=2, col=1
        )
        self._fig.add_trace(self._limit_trace(self._r_upper_limit, 'UCL', 'limit'), row=2, col=1)
        self._fig.add_trace(self._limit_trace(self._r_center, 'CL', 'center'), row=2, col=1)
        if self._show_r_lower_limit:
            self._fig.add_trace(self._limit_trace(self._r_lower_limit, 'LCL', 'limit'), row=2, col=1)

        self._fig.update_layout(title=self._title, template='simple_white', showlegend=False, width=self._width,
                                height=self._height)
        self._fig.update_xaxes(showgrid=False, row=1, col=1)
        self._fig.update_yaxes(title_text=self._x_title, showgrid=False, row=1, col=1)
        self._fig.update_yaxes(title_text=self._r_title, showgrid=False, row=2, col=1)

    def save(self, path):
        """
        Saves the chart as an image file.

        :param path: Path and filename to save the chart to.
        """
        self._fig.write_image(path)

    @property
    def figure(self):
        """Returns the chart as a Figure object"""
        return self._fig

    @property
    def widget(self):
        """
        Returns the chart as a go.FigureWidget object. This is used to create
        charts in Jupyter Notebooks where it is necessary to update the chart's
        data after the chart has been displayed.

        :return: Plotly go.FigureWidget
        """
        self._fig_widget = go.FigureWidget(self._fig)
        return self._fig_widget

    def update(self, x_values, labels, x_status, x_center, x_upper_limit, x_lower_limit, r_values, r_status, r_upper_limit, r_lower_limit, r_center):
        """
        If the chart exists, it is updated with new values for the X and R data.

        :param x_values: the values to be plotted on the X chart
        :param labels: the labels for the values plotted on the X and R charts
        :param x_status: Boolean array: whether the values of the X chart are in control.
        :param x_center: the values of the center line for the X chart
        :param x_upper_limit: the value of the upper control limit for the X chart
        :param x_lower_limit: the value of the lower control limit for the X chart
        :param r_values: the values to be plotted on the R chart
        :param r_status: Boolean array: whether the values in of the R chart are in control.
        :param r_upper_limit: the value of the upper control limit for the R chart
        :param r_lower_limit: the value of the lower control limit for the R chart
        :param r_center: the value of th center line for th R chart
        """
        self._x_values = x_values
        self._x = labels
        self._x_center = x_center
        self._x_status = x_status
        self._x_upper_limit = x_upper_limit
        self._x_lower_limit = x_lower_limit
        self._r_values = r_values
        self._r_status = r_status
        self._r_upper_limit = r_upper_limit
        self._r_lower_limit = r_lower_limit
        self._r_center = r_center

        for chart in [self._fig, self._fig_widget]:
            if chart is None:
                continue
            # x chart values
            chart.data[0]['y'] = self._x_values
            chart.data[0]['x'] = self._x
            chart.data[0]['marker']['color'] = self._marker_colors(self._x_status)
            chart.data[0]['customdata'] = self._marker_labels(self._x_title, self._x_values)
            # x chart upper limit
            chart.data[1]['x'] = self._x
            chart.data[1]['y'] = np.repeat(self._x_upper_limit, self._x.size)
            chart.data[1]['customdata'] = self._limit_labels(self._x_upper_limit, 'UCL')
            # x chart center line
            chart.data[2]['x'] = self._x
            chart.data[2]['y'] = np.repeat(self._x_center, self._x.size)
            chart.data[2]['customdata'] = self._limit_labels(self._x_center, 'CL')
            # x chart lower limit
            chart.data[3]['x'] = self._x
            chart.data[3]['y'] = np.repeat(self._x_lower_limit, self._x.size)
            chart.data[3]['customdata'] = self._limit_labels(self._x_lower_limit, 'LCL')
            # r chart values
            chart.data[4]['x'] = self._x
            chart.data[4]['y'] = self._r_values
            chart.data[4]['marker']['color'] = self._marker_colors(self._r_status)
            chart.data[4]['customdata'] = self._marker_labels(self._r_title, self._r_values)
            # r chart upper limit
            chart.data[5]['x'] = self._x
            chart.data[5]['y'] = np.repeat(self._r_upper_limit, self._x.size)
            chart.data[5]['customdata'] = self._limit_labels(self._r_upper_limit, 'UCL')
            # r chart center line
            chart.data[6]['x'] = self._x
            chart.data[6]['y'] = np.repeat(self._r_center, self._x.size)
            chart.data[6]['customdata'] = self._limit_labels(self._r_center, 'CL')
            # r chart lower limit
            if self._show_r_lower_limit:
                chart.data[7]['x'] = self._x
                chart.data[7]['y'] = np.repeat(self._r_lower_limit, self._x.size)
                chart.data[7]['customdata'] = self._limit_labels(self._r_lower_limit, 'LCL')


class Run:
    """
    Run chart

    methods:
        save()
        plot()
    """

    def __init__(self, values, labels, title='', x_title='Average', chart_width=800, chart_height=600):
        """

        :param values: The values to plot. Must be numpy.array like
        :param labels: The labels for the values. Must be numpy.array like
        :param title: The chart title
        :param x_title: The title for the y-axis
        :param chart_width: The chart width in pixels, defaults to 800.
        :param chart_height: The chart height in pixels, defaults to 600.
        """
        self._x_values = np.array(values)
        self._labels = np.array(labels)
        self._title = title
        self._x_title = x_title
        self._chart_width = chart_width
        self._chart_height = chart_height
        self._x_center_line = None
        self._fitted = False
        self._chart = None
        self._fit()

    def _fit(self):
        if self._x_values.ndim > 1:
            raise ValueError('Values has more than one column.')
        self._x_center_line = np.median(self._x_values)
        self._fitted = True

    def plot(self):
        """
        Plots the chart.

        :raises: ValueError: if there is no data to plot
        """
        if self._x_values is None:
            raise ValueError('Error: there is no data to plot')
        self._chart = RunChart(
            x_values=self._x_values,  # same
            labels=self._labels,
            x_center=self._x_center_line,
            title=self._title,
            x_title=self._x_title,
            width=self._chart_width,
            height=self._chart_height
        )

    @property
    def centre_line(self):
        """Returns the value of the center line (median)."""
        return self._x_center_line

    def save_chart(self, path):
        """
        Saves an image of the chart to a file.

        :param path: The full path and filename to save to. The file type is automatically determined
        by the filename extension. Allowed files types are PNG, JPEG, WebP, SVG and PDF.
        """
        self._chart.save(path)


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
        params
        predicted
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
        self._constants = Constants()  # Instance variable for the constants object
        self._n = None  # Number of values in each subgroup
        self._x_center_line = None  # The values of the center line for the subgroup averages chart
        self._x_sigma = None  # The value of sigma for the subgroup averages chart
        self._x_upper_limit = None  # The upper limit for the subgroup averages chart
        self._x_lower_limit = None  # The lower limit for the subgroup averages chart
        self._r_center_line = None  # The center line for the subgroup range chart
        self._r_upper_limit = None  # The upper limit for the subgroup range chart
        self._r_lower_limit = None  # The lower limit for the subgroup range chart
        self._fitted = False  # True the model has been fitted, False if it has not
        self._r_values = None  # The range of each subgroup of the values to be plotted
        self._x_values = None  # The means of each subgroup calculated by the predict method
        self._labels = None  # The labels given to the subgroups by the user
        self._r_in_limits = None  # Whether each subgroup range is within the range chart control limits
        self._x_in_limits = None  # Whether each subgroup mean is within the mean chart control limits
        self._chart = None  # True is the chart has been plotted

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

        :param values: The values to be used to calculate the control limits. The subgroup size must be at least 2.
        :type values: list, tuple or array
        :param labels: The labels for each subgroup
        :type labels: list, tuple or array of strings
        :type values: numpy.Array where each row is a subgroup and each column is a value in a subgroup.
        :raises: ValueError: if there are missing values in values.
        :raises: ValueError: if there are less than 2 columns in values.
        """
        values = np.array(values)
        labels = np.array(labels)
        array_missing_values(values)
        subgroups, n = values.shape
        if n < 2:
            raise ValueError('The number of samples per subgroup must be greater than one.')
        self._n = n
        subgroup_means, subgroup_ranges = self._subgroup_range_mean(values)
        mean_range = np.mean(subgroup_ranges)
        self._x_center_line = np.mean(values)
        self._x_sigma = self._constants.constant(n=self._n, name='A2') * mean_range
        self._x_upper_limit = self._x_center_line + self._x_sigma
        self._x_lower_limit = self._x_center_line - self._x_sigma
        self._r_center_line = mean_range
        self._r_upper_limit = mean_range * self._constants.constant(n=self._n, name='D4')
        self._r_lower_limit = mean_range * self._constants.constant(n=self._n, name='D3')
        self._fitted = True
        self.predict(values, labels)

    def predict(self, values, labels):
        """
        Transform subgroup values into mean and moving range values for plotting
        on a chart. Calculates whether the means and ranges are outside their respective limits.

        :param values: Each row is a subgroup and each column is a value of the subgroup.
        :type values: list, tuple or array
        :param labels: The labels for each subgroup
        :type labels: list, tuple or array of strings
        :raises: Exception: if the control limits have not been calculated.
        :raises: ValueError: if there are missing values in values.
        :raises: ValueError: if the number of values in the subgroups used to calculate the control limits
         is different to the number of subgroups in values.
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
        if self._chart is None:
            self._create_chart()
        else:
            self._update_chart()

    def _create_chart(self):
        """
        Plots the mean and range values on Plotly chart.

        :raises: ValueError: if there is no data to plot
        """
        if self._x_values is None:
            raise ValueError('Error: there is no data to plot')
        self._chart = XRChart(
            x_values=self._x_values,  # same
            labels=self._labels,
            x_center=self._x_center_line,
            x_status=self._x_in_limits,
            x_upper_limit=self._x_upper_limit,
            x_lower_limit=self._x_lower_limit,
            r_values=self._r_values,  # same
            r_status=self._r_in_limits,
            r_upper_limit=self._r_upper_limit,
            r_lower_limit=self._r_lower_limit,
            r_center=self._r_center_line,
            title=self._title,
            x_title=self._x_title,
            r_title=self._r_title,
            width=self._chart_width,
            height=self._chart_height,
            show_r_lower_limit=type(self).has_r_lower_limit
        )

    def _update_chart(self):
        self._chart.update(
            x_values=self._x_values,
            labels=self._labels,
            x_status=self._x_in_limits,
            x_center=self._x_center_line,
            x_upper_limit=self._x_upper_limit,
            x_lower_limit=self._x_lower_limit,
            r_values=self._r_values,
            r_status=self._r_in_limits,
            r_upper_limit=self._r_upper_limit,
            r_lower_limit=self._r_lower_limit,
            r_center=self._r_center_line,
        )

    @property
    def chart(self):
        return self._chart

    def save_chart(self, path):
        """
        Saves an image of the chart to a file.

        :param path: The full path and filename to save to. The file type is automatically
         determined by the filename extension. Allowed files types are PNG, JPEG, WebP, SVG and PDF.
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
        return ((values >= lower_limit) & (values <= upper_limit)) | (pd.isna(values))

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
        """
        Return or set the chart parameters as a dictionary:

        * n : the number of values in each subgroup
        * x_upper_limit: the upper limit for the X chart
        * x_lower_limit: the lower limit for the X chart
        * x_center_line: the center line of the X chart
        * r_upper_limit: the upper limit of the R chart
        * r_lower_limit: the lower limit of the R chart
        * r_center_line: the center line of the R chart
        * title: the chart title
        * x_title: the title for the y-axis of the X chart
        * r_title: the title for the y-axis of the R chart

        :returns: dictionary
        """
        return self._params_to_dict()

    @params.setter
    def params(self, params):
        """
        Sets the chart parameters.

        :param params: The chart parameters
        :type params: dictionary
        :return:
        """
        self._dict_to_params(params)
        self._fitted = True

    @property
    def predicted(self):
        """
        A pandas DataFrame of the predicted data. Columns are:

        * labels: the labels for each value plotted.
        * x_values: the values plotted on the X chart.
        * r_values: the values plotted on the R chart.
        * x_in_control: True if the value plotted on the Averages chart is within the control limits.
        * r_in_control: True if the value plotted on the Range chart in within the control limits.

        :return: pandas.DataFrame
        """
        return pd.DataFrame(
            data={
                'labels': self._labels,
                'x_values': self._x_values,
                'r_values': self._r_values,
                'x_in_control': self._x_in_limits,
                'r_in_control': self._r_in_limits
            }
        )


class IndividualMR(XbarR):
    """
    Shewart Individual and Moving Range chart

    Methods:
        fit()
        predict()
        save()
        load()
        save_chart()

    Properties:
        params
        predicted
    """

    has_r_lower_limit = False
    chart_type = 'IndividualMR'

    def fit(self, values, labels):
        """
        Calculates the control limits and center lines for the average and range charts.
        It then runs the predict method and

        :param values: The values to be used to calculate the control limits. The subgroup size must be at least 2.
        :type values: list, tuple or array
        :param labels: The labels for each value
        :type labels: list, tuple or array of strings
        :type values: numpy.Array where each row is a subgroup and each column is a value in a subgroup.
        :raises: ValueError: if there are missing values in values.
        :raises: ValueError: if there are more than one column in values.

        """
        values = np.array(values)
        labels = np.array(labels)
        array_missing_values(values)
        if values.ndim > 1:
            raise ValueError('The number of samples per subgroup must be one.')
        self._n = 1
        self._x_center_line = np.mean(values)
        average_moving_range = np.mean(self.moving_ranges(values))
        self._x_sigma = average_moving_range / self._constants.constant(n=2, name='d2')
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

        :param values: the values
        :type values: list, tuple or array
        :param labels: The labels for each value
        :type labels: list, tuple or array of strings
        :raises: Exception: if the control limits have not been calculated.
        :raises: ValueError: if there are missing values in values.
        :raises: ValueError: if there are more than one column in values.

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
        if self._chart is None:
            self._create_chart()
        else:
            self._update_chart()

    @staticmethod
    def moving_ranges(values):
        """
        Calculates the moving ranges for a single dimension array

        :param values: single dimension array of values
        :return: numpy array
        """
        return np.abs(np.ediff1d(values))
