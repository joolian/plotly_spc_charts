from pathlib import Path

import numpy as np
import pandas as pd

from spc_charts import XbarR, IndividualMR

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

