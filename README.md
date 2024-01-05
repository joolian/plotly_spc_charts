# Plotly charts for statistical process control

[![Documentation Status](https://readthedocs.org/projects/plotly-spc-charts/badge/?version=latest)](https://plotly-spc-charts.readthedocs.io/en/latest/?badge=latest)

## Installation
This project is in progress, so constantly changing. You can find the package at TestPyPI:
[plotly-spc-charts](https://test.pypi.org/project/plotly-spc-charts)

To install:

`pip install -i https://test.pypi.org/simple/ plotly-spc-charts`

## Shewart Average and Range Chart
Creates an interactive Average and Range chart using Plotly.  
* Subgroups are classed as out of control if they are greater than the upper control limit or less than the lower control limit.  
* Each subgroup can be labelled and the labels will be shown on the chart when the mouse pointer is near the marker.  
* The control limits can be calculated from a set of data and the chart plotted.  
* New data can be plotted using the calculated control limits.  
* The chart can be saved as PNG, JPEG, WebP, SVG and PDF.  
* The control limits can be saved to JSON, and  loaded from JSON.  
* The control limits can be set and retrieved as a dictionary.  
* The subgroup labels, averages, ranges and status (in or out of control) can be retrieved as a pandas Dataframe.

For an example of usage see: [XbarR_example.ipynb](https://github.com/joolian/plotly_spc_charts/blob/main/examples/XbarR/XbarR_example.ipynb)

![XbarR_chart.svg](https://github.com/joolian/plotly_spc_charts/blob/package/examples/XbarR/chart_image.svg)


### Calculation of control limits
Calculations use the method given in: <em>Wheeler, D.J. and Chambers, D.S. (1992) Understanding statistical process control. SPC Press, Incorporated, p. 44.</em>  

&nbsp;&nbsp;&nbsp;&nbsp; $UCL_{\bar{X}} = \bar{\bar{X}} + A_{2}\bar{R}$  
  
&nbsp;&nbsp;&nbsp;&nbsp; $LCL_{\bar{X}} = \bar{\bar{X}} - A_{2}\bar{R}$  

&nbsp;&nbsp;&nbsp;&nbsp; $CL_{\bar{X}} = \bar{\bar{X}}$
  
&nbsp;&nbsp;&nbsp;&nbsp; $UCL_{R} = D_{4}\bar{R}$  
  
&nbsp;&nbsp;&nbsp;&nbsp; $CL_{R} = \bar{R}$  
  
&nbsp;&nbsp;&nbsp;&nbsp; $LCL_{R} = D_{3}\bar{R}$

Where:

&nbsp;&nbsp;&nbsp;&nbsp; $\bar{\bar{X}}$ is the average of all the values in the subgroups.  
&nbsp;&nbsp;&nbsp;&nbsp; $\bar{R}$ is the average of the subgroup ranges.  
&nbsp;&nbsp;&nbsp;&nbsp; $A_{2}$ is the value of the constant corresponding to the subgroup size.  
&nbsp;&nbsp;&nbsp;&nbsp; $D_{3}$ is the value of the constant corresponding to the subgroup size.  
&nbsp;&nbsp;&nbsp;&nbsp; $D_{4}$ is the value of the constant corresponding to the subgroup size.

## Shewart Individual and Moving Range Chart
Creates an interactive individual Moving Range chart using Plotly.  
* Individual values and moving ranges are classed as out of control if they are greater than the upper control limit or less than the lower control limit.  
* Each value can be labelled and the labels will be shown on the chart when the mouse pointer is near the marker.  
* The control limits can be calculated from a set of data and the chart plotted.  
* New data can be plotted using the calculated control limits.  
* The chart can be saved as PNG, JPEG, WebP, SVG and PDF.  
* The control limits can be saved to JSON, and  loaded from JSON.  
* The control limits can be set and retrieved as a dictionary.  
* The labels, individual values, moving ranges and status (in or out of control) can be retrieved as a pandas Dataframe.

For an example of usage see: [IndividualMR_example.ipynb](https://github.com/joolian/plotly_spc_charts/blob/main/examples/IndividualMR/IndividualMR_example.ipynb)

![chart_image.png](https://github.com/joolian/plotly_spc_charts/blob/main/examples/IndividualMR/chart_image.png)

### Calculation of control limits
Calculations use the method given in: <em>Wheeler, D.J. and Chambers, D.S. (1992) Understanding statistical process control. SPC Press, Incorporated, pp. 48-49.</em>  

&nbsp;&nbsp;&nbsp;&nbsp; $UNPL_{X} = \bar{X} + \displaystyle\frac{3\bar{mR}}{d_{2}}$

&nbsp;&nbsp;&nbsp;&nbsp; $LNPL_{X} = \bar{\bar{X}} - \displaystyle\frac{3\bar{mR}}{d_{2}}$

&nbsp;&nbsp;&nbsp;&nbsp; $CL_{X} = \bar{X}$

&nbsp;&nbsp;&nbsp;&nbsp; $UCL_{R} = D_{4}\bar{mR}$  

&nbsp;&nbsp;&nbsp;&nbsp; $CL_{R} = \bar{mR}$  

Where:  

&nbsp;&nbsp;&nbsp;&nbsp; $\bar{X}$ is the average of the individual values.

&nbsp;&nbsp;&nbsp;&nbsp; $\bar{mR}$ is the average of the moving ranges. 

&nbsp;&nbsp;&nbsp;&nbsp; $d_{2}$ is the value of the constant corresponding to n=2.  

&nbsp;&nbsp;&nbsp;&nbsp; $D_{4}$ is the value of the constant corresponding to n=2.  

 ## Run Chart
Creates an interactive run chart using Plotly.  
* Each value can be labelled and the labels will be shown on the chart when the mouse pointer is near the marker.  
* The chart can be saved as PNG, JPEG, WebP, SVG and PDF.  
* The value of the median can be retrieved.  

For an example of usage see: [Run_example.ipynb](https://github.com/joolian/plotly_spc_charts/blob/main/examples/Run/Run_example.ipynb)  

![chart_image.svg](https://github.com/joolian/plotly_spc_charts/blob/main/examples/Run/chart_image.svg)

## Constants
Constants for the computing of control limits are listed in [factor_values_for_shewart_charts.csv](https://github.com/joolian/plotly_spc_charts/blob/main/spc_charts/factor_values_for_shewart_charts.csv).  
Values where extracted from: 
<em>[A Note on the Factor Values of Three Common Shewhart Variables Control Charts. Henry H. Bi. 2015. Communications in Statistics - Theory and Methods (0361-0926). 44(13): 2655-2673. DOI:10.1080/03610926.2014.968732](https://www.researchgate.net/publication/275236350_A_Note_on_the_Factor_Values_of_Three_Common_Shewhart_Variables_Control_Charts_Henry_H_Bi_2015_Communications_in_Statistics_-_Theory_and_Methods_0361-0926_4413_2655-2673_httpdxdoiorg1010800361092620149)</em>

## TODO
Allow the use of Western Electric rules.  
See: <em>[ Western Electric Company, Statistical Quality Control Handbook., Indianapolis, Indiana: Western Electric Co](https://www.westernelectric.com/library#technical)</em>

Add np-chart, p-chart, c-chart and u-chart, EWMA-chart, CUSUM-chart, pareto-chart and multi-variate charts.
