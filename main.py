from pathlib import Path

import numpy as np
import pandas as pd

from spc_charts import XbarR

if __name__ == '__main__':
    data = pd.read_excel(Path(__file__).parent / 'tests/control_chart_example_data.xlsx', sheet_name='Data')
    labels = data['labels'].to_numpy()
    n = 4
    values = data.iloc[:, 1:n + 1:1].to_numpy()
    xr = XbarR(chart_width=1000, chart_height=800)
    xr.fit(values=values)
    print(xr.control_limits)
    xr.predict(values, labels)
    xr.plot()
    xr.save_chart(Path('chart.svg'))
    print(xr.out_of_control)

