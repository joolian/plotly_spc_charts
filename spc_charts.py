import pandas as pd


class Constants:
    """Represents a table of constants"""

    def __init__(self, path):
        self._path = path
        self._factors = None
        self._load()

    def _load(self):
        try:
            self._factors = pd.read_csv(self._path, index_col=0)
        except FileNotFoundError as error:
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
