import pytest
from pathlib import Path
from spc_charts import Constants


@pytest.fixture
def constants_path():
    return Path(__file__).parent.parent / 'constants/factor_values_for_shewart_charts.csv'


class TestConstants:

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
