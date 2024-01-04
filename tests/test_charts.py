import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from spc_charts.spc_charts import Constants, XbarR, IndividualMR, Run


class TestConstants:

    def test_constants(self, ):
        f = Constants()
        assert f.constant(n=50, name='c4') == 0.9949113047

    @pytest.mark.parametrize('n, name', [(2, 'thing'), (101, 'A')])
    def test_constant_error(self, n, name):
        f = Constants()
        with pytest.raises(Exception, match='[Cannot find thing for]'):
            f.constant(n=n, name=name)


class TestXbarR:

    @pytest.fixture
    def test_data_path(self):
        path = Path(Path(__file__).parent / 'XbarR_test_data')
        return path

    @pytest.fixture
    def test_data(self, test_data_path):
        return pd.read_csv(test_data_path / 'XbarR_test_data.csv')

    @pytest.fixture
    def values(self, test_data):
        return test_data[[col for col in test_data.columns.values if 'Pressure' in col]].to_numpy()

    @pytest.fixture
    def labels(self, test_data):
        return test_data['Date'].to_numpy()

    @pytest.fixture
    def expected_params(self):
        return {
            "type": "XBarR",
            "n": 5,
            "x_upper_limit": 62.786297673183334,
            "x_lower_limit": 52.98036899348333,
            "x_center_line": 57.88333333333333,
            "r_upper_limit": 17.97324273335,
            "r_lower_limit": 0.0,
            "r_center_line": 8.5,
            "title": "XbarR chart",
            "x_title": "Subgroup mean",
            "r_title": "Subgroup range"
        }

    @pytest.fixture
    def expected_out_of_control(self, test_data_path):
        return pd.read_csv(test_data_path / 'XbarR_expected_out_of_control.csv', )

    @pytest.fixture
    def expected_predicted(self, test_data):
        expected = test_data[['Date', 'averages', 'ranges', 'x_in_control', 'r_in_control']]
        expected = expected.rename(columns={
            'Date': 'labels',
            'averages': 'x_values',
            'ranges': 'r_values'
        })
        return expected

    @pytest.fixture
    def expected_means_ranges(self, test_data):
        means = test_data['averages'].to_numpy()
        ranges = test_data['ranges'].to_numpy()
        return {'means': means, 'ranges': ranges}

    @pytest.fixture
    def fitted_chart(self, values, labels):
        chart = XbarR(title='XbarR chart', x_title='Subgroup mean', r_title='Subgroup range')
        chart.fit(values[:12, :], labels[:12])
        return chart

    def test_fit(self, fitted_chart, expected_params):
        assert fitted_chart.params == expected_params

    def test_fit_size_error(self):
        with pytest.raises(ValueError, match='The number of samples per subgroup must be greater than one.'):
            chart = XbarR()
            chart.fit(values=np.array([[1], [2]]), labels=['a', 'b'])

    def test_fit_missing_values(self):
        with pytest.raises(ValueError, match='There are missing values.'):
            chart = XbarR()
            chart.fit(values=np.array([[None, 2, 3], [1, 2, np.nan]]), labels=['a', 'b', 'c'])

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

    def test_predict(self, fitted_chart, expected_predicted, values, labels):
        fitted_chart.predict(values, labels)
        assert_frame_equal(fitted_chart.predicted, expected_predicted)

    def test_save(self, fitted_chart, tmp_path, expected_params):
        file_name = 'model_params.json'
        fitted_chart.save(tmp_path / file_name)
        with open(tmp_path / file_name, 'r') as fp:
            assert json.load(fp) == expected_params

    def test_load(self, fitted_chart, test_data_path, expected_params):
        fitted_chart.load(test_data_path / 'XbarR_expected_params.json')
        assert fitted_chart.params == expected_params

    def test_get_params(self, fitted_chart, expected_params):
        assert fitted_chart.params == expected_params

    def test_set_params(self, fitted_chart, expected_params):
        fitted_chart.params = expected_params
        assert fitted_chart.params == expected_params

    def test_save_chart(self, fitted_chart, tmp_path):
        file_path = tmp_path / 'chart.png'
        # fitted_chart.plot()
        fitted_chart.save_chart(file_path)
        assert file_path.exists()

    def test_save_chart_error(self, fitted_chart, tmp_path):
        chart = XbarR()
        file_path = tmp_path / 'IndividualMR_chart.png'
        with pytest.raises(Exception, match='Error: the chart must be plotted before it can be saved'):
            chart.save_chart(file_path)


class TestIndividualMR:
    @pytest.fixture
    def data(self):
        return pd.read_csv(
            Path(__file__).parent / 'IndividualMR_test_data/IndividualMR_test_data.csv'
        )

    @pytest.fixture
    def values(self, data):
        return data['values'].to_numpy()

    @pytest.fixture
    def labels(self, data):
        return data['labels'].to_numpy()

    @pytest.fixture
    def expected_params(self):
        return {
            'type': 'IndividualMR',
            'n': 1,
            'x_upper_limit': 46.00893346217633,
            'x_lower_limit': 36.824399871157,
            'x_center_line': 41.416666666666664,
            'r_upper_limit': 5.642191496972727,
            'r_lower_limit': 0,
            'r_center_line': 1.7272727272727273,
            'title': 'Individuals chart',
            'x_title': 'Values',
            'r_title': 'Moving range'
        }

    @pytest.fixture
    def expected_predicted(self, data):
        expected = data[['labels', 'values', 'moving_range', 'x_in_control', 'r_in_control']]
        expected = expected.rename(columns={'values': 'x_values', 'moving_range': 'r_values'})
        return expected

    @pytest.fixture
    def fitted_chart(self, values, labels):
        chart = IndividualMR(
            title='Individuals chart',
            r_title='Moving range',
            x_title='Values'
        )
        chart.fit(values=values[:12], labels=labels[:12])
        return chart

    def test_fit(self, fitted_chart, expected_params):
        assert fitted_chart.params == expected_params

    @pytest.mark.parametrize(
        'values, labels', [
            (np.array([None, 2, 3]), np.array(['A', 'B', 'C'])),
            (np.array([1, 2, np.nan]), np.array(['A', 'B', 'C']))
        ]
    )
    def test_fit_missing_values(self, values, labels):
        with pytest.raises(ValueError, match='There are missing values.'):
            chart = IndividualMR()
            chart.fit(values=values, labels=labels)

    def test_fit_ndim_error(self):
        with pytest.raises(ValueError, match='The number of samples per subgroup must be one.'):
            values = np.array([[1, 2, 3], [4, 5, 6]])
            labels = np.array(['A', 'B'])
            chart = IndividualMR()
            chart.fit(values=values, labels=labels)

    def test_predict(self, fitted_chart, expected_predicted, values, labels):
        fitted_chart.predict(values=values, labels=labels)
        assert_frame_equal(fitted_chart.predicted, expected_predicted)

    def test_predict_not_fitted(self, values, labels):
        with pytest.raises(Exception, match='Error: chart has not been fitted'):
            chart = IndividualMR()
            chart.predict(values=values, labels=labels)

    @pytest.mark.parametrize(
        'input_values, input_labels', [
            (np.array([None, 2, 3]), np.array(['A', 'B', 'C'])),
            (np.array([1, 2, np.nan]), np.array(['A', 'B', 'C']))
        ]
    )
    def test_predict_missing_values(self, fitted_chart, input_values, input_labels):
        with pytest.raises(ValueError, match='There are missing values.'):
            fitted_chart.predict(values=input_values, labels=input_labels)

    def test_predict_ndim_error(self, fitted_chart):
        values = np.array([[1, 2, 3], [4, 5, 6]])
        labels = np.array(['A', 'B'])
        with pytest.raises(ValueError, match='The number of samples per subgroup must be one.'):
            fitted_chart.predict(values=values, labels=labels)


class TestRun:

    @pytest.fixture
    def data(self):
        return pd.read_csv(
            Path(__file__).parent / 'Run_test_data/Run_test_data.csv'
        )

    @pytest.fixture
    def values(self, data):
        return data['values'].to_numpy()

    @pytest.fixture
    def labels(self, data):
        return data['labels'].to_numpy()

    @pytest.fixture
    def fitted_chart(self, values, labels):
        return Run(values=values, labels=labels, title='Run chart', x_title='Values', chart_height=500, chart_width=700)

    def test_ndim_error(self):
        values = np.array([[1, 2, 3], [4, 5, 6]])
        labels = np.array(['A', 'B'])
        with pytest.raises(ValueError, match='Values has more than one column.'):
            Run(values=values, labels=labels)

    def test_centre_line(self, fitted_chart, values):
        assert fitted_chart.centre_line == np.median(values)

    def test_save_chart(self, fitted_chart, tmp_path):
        file_path = tmp_path / 'run_chart.png'
        fitted_chart.plot()
        fitted_chart.save_chart(file_path)
        assert file_path.exists()
