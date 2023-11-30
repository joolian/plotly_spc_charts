from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

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
        workbook = pd.ExcelFile('control_chart_example_data.xlsx')
        return pd.read_excel(workbook, 'Data')

    @pytest.fixture
    def values_labels(self, test_data):
        n = 4
        values = test_data.iloc[:, 1:n + 1:1].to_numpy()
        labels = test_data['labels'].to_numpy()
        return {'values': values, 'labels': labels}

    @pytest.fixture
    def expected_control_limits(self):
        return {
            'X center line': 4.7625,
            'X sigma': 2.950818602895,
            'X upper control limit': 7.713318602895001,
            'X lower limit': 1.811681397105,
            'R center line': 4.05,
            'R upper limit': 9.24230882367,
            'R lower limit': 0.0
        }

    @pytest.fixture
    def expected_means_ranges(self, test_data):
        means = test_data['Average'].to_numpy()
        ranges = test_data['Range'].to_numpy()
        return {'means': means, 'ranges': ranges}

    @pytest.fixture
    def fitted_chart(self, values_labels):
        chart = XbarR()
        chart.fit(values_labels['values'])
        return chart

    def test_fit(self, values_labels):
        expected = {
            'X center line': 4.7625,
            'X sigma': 2.950818602895,
            'X upper control limit': 7.713318602895001,
            'X lower limit': 1.811681397105,
            'R center line': 4.05,
            'R upper limit': 9.24230882367,
            'R lower limit': 0.0
        }
        chart = XbarR()
        chart.fit(values=values_labels['values'])
        assert chart.control_limits == expected

    def test_fit_type_error(self):
        with pytest.raises(TypeError, match="Values must be a numpy array, not <class 'list'>"):
            chart = XbarR()
            chart.fit(values=[1, 2, 3])

    def test_fit_size_error(self):
        with pytest.raises(ValueError, match='The number of samples per subgroup must be greater than one.'):
            chart = XbarR()
            chart.fit(values=np.array([[1], [2]]))

    def test_fit_missing_values(self):
        with pytest.raises(ValueError, match='There are missing values.'):
            chart = XbarR()
            chart.fit(values=np.array([[None, 2, 3], [1, 2, np.nan]]))

    def test_predict(self, values_labels, expected_means_ranges):
        chart = XbarR()
        chart.fit(values_labels['values'])
        chart.predict(values_labels['values'], values_labels['labels'])
        subgroup_means, subgroup_ranges = chart.averages_ranges
        assert_array_equal(subgroup_means, expected_means_ranges['means'])
        assert_array_equal(subgroup_ranges, expected_means_ranges['ranges'])

    @pytest.mark.parametrize(
        'p_values, p_labels',
        [
            (np.array([1, 2, 3]), ['A', 'B', 'C']),
            ([1, 2, 3], np.array(['A', 'B', 'C']))
        ]
    )
    def test_predict_type_error(self, fitted_chart, p_values, p_labels):
        with pytest.raises(TypeError, match="Values must be a numpy array, not <class 'list'>"):
            fitted_chart.predict(p_values, p_labels)

    def test_predict_missing_values(self, fitted_chart):
        with pytest.raises(ValueError, match='There are missing values.'):
            fitted_chart.predict(values=np.array([[None, 2, 3], [1, 2, np.nan]]), labels=np.array(['A', 'B', 'C']))

    def test_predict_n(self, fitted_chart):
        with pytest.raises(
                ValueError,
                match='Error: the number of subgroups must be the same as that used to calculate the control limits \(4\)'
        ):
            fitted_chart.predict(
                np.array([[1, 2, 3]]), np.array(['A'])
            )

    def test_out_of_limits(self, fitted_chart, values_labels):
        values = np.vstack([values_labels['values'], np.array([20, 10, 10, 10])])
        labels = np.append(values_labels['labels'], 'Z')
        fitted_chart.predict(values, labels)
        df = fitted_chart.out_of_control
        assert not df.iloc[0][3]
        assert not df.iloc[0][4]
        assert df.iloc[0][0] == 'Z'
