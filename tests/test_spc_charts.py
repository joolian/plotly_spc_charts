import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from spc_charts import Constants, XbarR


class TestConstants:

    @pytest.fixture
    def constants_path(self):
        return Path(__file__).parent.parent / 'constants/factor_values_for_shewart_charts.csv'

    def test__load(self):
        filename = 'factor_values_for_shewart_charts.csv'
        with pytest.raises(Exception, match=f'Error: cannot find {filename}'):
            Constants(Path(__file__).parent / filename)

    def test_constants(self, constants_path):
        f = Constants(constants_path)
        assert f.constant(n=50, name='c4') == 0.9949113047

    @pytest.mark.parametrize('n, name', [(2, 'thing'), (101, 'A')])
    def test_constant_error(self, constants_path, n, name):
        f = Constants(constants_path)
        with pytest.raises(Exception, match='[Cannot find thing for]'):
            f.constant(n=n, name=name)


class TestXbarR:

    @pytest.fixture
    def test_data(self):
        return pd.read_csv('XbarR_test_data.csv')

    @pytest.fixture
    def values_labels(self, test_data):
        labels = test_data['Date'].to_numpy()
        values = test_data[[col for col in test_data.columns.values if 'Pressure' in col]].to_numpy()
        return {'values': values, 'labels': labels}

    @pytest.fixture
    def expected_params(self):
        return {
            "n": 5,
            "x_upper_limit": 62.6012796460125,
            "x_lower_limit": 51.882053687320834,
            "x_center_line": 57.24166666666667,
            "r_upper_limit": 19.647221223220832,
            "r_lower_limit": 0.0,
            "r_center_line": 9.291666666666666,
            "title": "Water cooling pressure",
            "x_title": "Subgroup mean",
            "r_title": "Subgroup range"
        }

    @pytest.fixture
    def expected_out_of_control(self):
        return pd.read_csv('XbarR_expected_out_of_control.csv', )

    @pytest.fixture
    def expected_means_ranges(self, test_data):
        means = test_data['Average'].to_numpy()
        ranges = test_data['Range'].to_numpy()
        return {'means': means, 'ranges': ranges}

    @pytest.fixture
    def fitted_chart(self, values_labels):
        chart = XbarR(title='Water cooling pressure', x_title='Subgroup mean', r_title='Subgroup range')
        chart.fit(values_labels['values'], values_labels['labels'])
        return chart

    def test_fit(self, fitted_chart, expected_params):
        fitted_chart.save('model_params.json')
        with open('model_params.json', 'r') as fp:
            chart_params = json.load(fp)
        assert chart_params == expected_params

    def test_fit_size_error(self):
        with pytest.raises(ValueError, match='The number of samples per subgroup must be greater than one.'):
            chart = XbarR()
            chart.fit(values=np.array([[1], [2]]), labels=['a', 'b'])

    def test_fit_missing_values(self):
        with pytest.raises(ValueError, match='There are missing values.'):
            chart = XbarR()
            chart.fit(values=np.array([[None, 2, 3], [1, 2, np.nan]]), labels=['a', 'b', 'c'])

    def test_predict(self, values_labels, expected_means_ranges):
        chart = XbarR()
        chart.fit(values_labels['values'], values_labels['labels'])
        chart.predict(values_labels['values'], values_labels['labels'])
        subgroup_means, subgroup_ranges = chart.averages_ranges
        assert_array_equal(subgroup_means, expected_means_ranges['means'])
        assert_array_equal(subgroup_ranges, expected_means_ranges['ranges'])

    def test_predict_missing_values(self, fitted_chart):
        with pytest.raises(ValueError, match='There are missing values.'):
            fitted_chart.predict(values=np.array([[None, 2, 3], [1, 2, np.nan]]), labels=np.array(['A', 'B', 'C']))

    def test_predict_n(self, fitted_chart):
        with pytest.raises(
                ValueError,
                match='Error: the number of subgroups must be the same as that used to calculate the control limits \(5\)'
        ):
            fitted_chart.predict(
                np.array([[1, 2, 3]]), np.array(['A'])
            )

    def test_out_of_limits(self, fitted_chart, values_labels, expected_out_of_control):
        fitted_chart.predict(values_labels['values'], values_labels['labels'])
        assert_frame_equal(fitted_chart.out_of_control.reset_index(drop=True), expected_out_of_control)
