import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from spc_charts import Constants
from pprint import pprint


class MRChart:
    """Average and range chart"""

    def __init__(self, x, labels, mean_x, x_status, upper_limit_x, lower_limit_x, x_moving_range, mr_status, lower_limit_r,
                 upper_limit_r, mean_r, title, x_title, r_title, width, height):
        self._x = x
        self._labels = labels
        self._y = np.arange(1, x.shape[0], 1)
        self._mean_x = mean_x
        self._x_status = x_status
        self._mr_status = mr_status
        self._upper_limit_x = upper_limit_x
        self._lower_limit_x = lower_limit_x
        self._x_moving_range = x_moving_range
        self._upper_limit_r = upper_limit_r
        self._lower_limit_r = lower_limit_r
        self._mean_r = mean_r
        self._title = title
        self._x_title = x_title
        self._r_title = r_title
        self._width = width
        self._height = height

        self._fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        self._draw()

    @staticmethod
    def _marker_colors(status):
        things = np.where(status, 'green', 'red')
        return things

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
                values=self._x,
                status=self._x_status,
                marker_labels=self._marker_labels(self._x_title, self._x)
            ),
            row=1, col=1
        )
        self._fig.add_trace(self._control_limit_trace(self._lower_limit_x, 'LCL'), row=1, col=1)
        self._fig.add_trace(self._control_limit_trace(self._upper_limit_x, 'UCL'), row=1, col=1)
        self._fig.add_trace(self._mean_trace(self._mean_x), row=1, col=1)

        self._fig.add_trace(
            self._value_trace(
                values=self._x_moving_range,
                status=self._mr_status,
                marker_labels=self._marker_labels(self._r_title, self._x_moving_range)
            ),
            row=2, col=1
        )
        self._fig.add_trace(self._control_limit_trace(self._upper_limit_r, 'UCL'), row=2, col=1)
        self._fig.add_trace(self._control_limit_trace(self._lower_limit_r, 'LCL'), row=2, col=1)
        self._fig.add_trace(self._mean_trace(self._mean_r), row=2, col=1)
        self._fig.update_layout(title=self._title, template='simple_white', showlegend=False, width=self._width, height=self._height)
        self._fig.update_xaxes(showgrid=False, row=1, col=1)
        self._fig.update_yaxes(title_text=self._x_title, showgrid=False, row=1, col=1)
        self._fig.update_yaxes(title_text=self._r_title, showgrid=False, row=2, col=1)
        self._fig.show()

    def save(self, path):
        self._fig.write_image(path)


class IndividualMR:
    """Calculates the data required to create an Average and moving range chart for individual values."""
    # Constants for subgroups of size n=2
    d2 = 1.128
    d4 = 3.268

    def __init__(self):
        self._control_x = None
        self._control_y = None
        self._mean_x = None
        self._upper_limit_x = None
        self._lower_limit_x = None
        self._upper_limit_r = None
        self._mean_r = None
        self._x = None
        self._y = None
        self._x_moving_range = None
        self._x_status = None
        self._mr_status = None
        self._fitted = False

    def fit(self, control_y, control_x):
        """
        Calculates the control chart limits.

        :param control_y: Array of the values of the quality
        :param control_x: Integer index of the values
        """
        self._control_x = control_x
        self._control_y = control_y
        self._calculate_limits()
        self._fitted = True

    def predict(self, x, y):
        """
        Calculates the maving ranges for the values and determins which values
        are out of control
        """
        if not self._fitted:
            raise Exception('Not fitted')
        self._x = x
        self._y = y
        self._x_moving_range = np.concatenate(([np.NAN], self.moving_ranges(self._x)))
        self._x_status = self.x_in_control(x, self._upper_limit_x, self._lower_limit_x)
        self._mr_status = self.mr_in_control(self._x_moving_range, self._upper_limit_r)

    @staticmethod
    def x_in_control(values, upper_limit, lower_limit):
        """
        Is each value within the control limits?
        :param values: array of values
        :param upper_limit: the upper control limit
        :param lower_limit: the lower control limit
        :return: boolean array
        """
        return (values > lower_limit) & (values < upper_limit)

    @staticmethod
    def mr_in_control(values, upper_limit):
        """Is each value of the moving range less than the upper control limit"""
        return (values < upper_limit) | (np.isnan(values))

    @staticmethod
    def moving_ranges(values):
        """
        Calculates the moving ranges for a single dimension array
        :param values: single dimension array of values
        :return: numpy array
        """
        return np.abs(np.ediff1d(values))

    def _calculate_limits(self):
        """
        Calculates the control limits and centers for Individual and
        Moving Range Charts.
        """
        self._mean_x = np.mean(self._control_y)
        average_moving_range = np.mean(self.moving_ranges(self._control_y))
        process_limit = 3 / IndividualMR.d2 * average_moving_range
        self._upper_limit_x = self._mean_x + process_limit
        self._lower_limit_x = self._mean_x - process_limit
        self._upper_limit_r = IndividualMR.d4 * average_moving_range
        self._mean_r = average_moving_range

    @property
    def mean_x(self):
        return self._mean_x

    @property
    def upper_limit_x(self):
        return self._upper_limit_x

    @property
    def lower_limit_x(self):
        return self._lower_limit_x

    @property
    def upper_limit_r(self):
        return self._upper_limit_r

    @property
    def mean_r(self):
        return self._mean_r

    @property
    def x_moving_ranges(self):
        return self._x_moving_range

    @property
    def x_status(self):
        return self._x_status

    @property
    def mr_status(self):
        return self._mr_status

    @property
    def out_of_control_x(self):
        df = pd.DataFrame(
            data={
                'x': self._x,
                'y': self._y,
                'x_status': self._x_status,
                'mr_status': self._mr_status
            }
        )
        return df[~df['x_status'] | ~df['mr_status']]


class MR:

    def __init__(self, chart_width=800, chart_height=600):
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
        if not isinstance(values, np.ndarray):
            raise Exception(f'Values must be a numpy array, not {type(values)}')
        subgroups, n = values.shape
        if n < 2:
            raise Exception('The number of samples per subgroup must be greater than one.')
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

    def predict(self, values, labels):
        if not self._fitted:
            raise Exception('Chart has not been fitted')
        self._labels = labels
        self._subgroup_means, self._subgroup_ranges = self.subgroup_range_mean(values)
        self._r_in_limits = self.within_limits(self._subgroup_ranges, self._r_upper_limit, self._r_lower_limit)
        self._x_in_limits = self.within_limits(self._subgroup_means, self._x_upper_limit, self._x_lower_limit)
        # if self._chart:
        #     self._chart.update()

    def plot(self):
        """
        For the predict method:
        if the chart does not exist then create it
        if the chart does exist then update it

        """
        self._chart = MRChart(
            x=self._subgroup_means,
            labels=labels,
            mean_x=self._x_center_line,
            x_status=self._x_in_limits,
            upper_limit_x=self._x_upper_limit,
            lower_limit_x=self._x_lower_limit,
            x_moving_range=self._subgroup_ranges,
            mr_status=self._r_in_limits,
            upper_limit_r=self._r_upper_limit,
            lower_limit_r=self._r_lower_limit,
            mean_r=self._r_center_line,
            title='Thickness',
            # x_title='\u0078\u0304', # X with bar
            x_title='Average',
            r_title='Range',
            width=self._chart_width,
            height=self._chart_height
        )

    def _update_chart(self):
        pass

    def save_chart(self, path):
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

    @property
    def control_limits(self):
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
        df = pd.DataFrame(
            data={
                'labels': self._labels,
                'subgroup_mean': self._subgroup_means,
                'subgroup_range': self._subgroup_ranges,
                'subgroup_mean_within_limits': self._x_in_limits,
                'subgroup_range_within_limits': self._r_in_limits
            }
        )
        return df[~df['x_status'] | ~df['mr_status']]


if __name__ == '__main__':
    test_x = np.array([39, 41, 41, 41, 43, 55, 41, 42, 40, 41, 44, 40])
    test_y = np.array(np.arange(1, np.shape(test_x)[0] + 1, 1, dtype=int))
    ic = IndividualMR()
    ic.fit(test_x, test_y)
    ic.predict(test_x, test_y)
    # spc_chart = XmRChart(
    #     x=test_x,
    #     y=test_y,
    #     mean_x=ic.mean_x,
    #     x_status=ic.x_status,
    #     mr_status=ic.mr_status,
    #     upper_limit_x=ic.upper_limit_x,
    #     lower_limit_x=ic.lower_limit_x,
    #     x_moving_range=ic.x_moving_ranges,
    #     upper_limit_r=ic.upper_limit_r,
    #     mean_r=ic.mean_r,
    #     title='Individual and Moving Range Chart for Rail Cars',
    # )

data = pd.read_excel('control chart data.xlsx')
labels = data['labels'].to_numpy()
n = 4
values = data.iloc[:, 1:n + 1:1].to_numpy()
xr = MR(chart_width=1000, chart_height=800)
xr.fit(values=values)
print(xr.control_limits)
xr.predict(values, labels)
xr.plot()
xr.save_chart(Path('chart.svg'))
