# Plotly charts for statistical process control

In progress

## Shewart Average and Range Chart
Creates an interactive Average and Range chart using Plotly.  
The control limits can be calculated using one set of data and the chart plotted.  
New data can be plotted using the calculated control limits.  
The chart can be saved as PNG, JPEG, WebP, SVG and PDF.  
The control limits can be saved to JSON, and  loaded into another chart.  
Out of control subgroups are shown with red markers.
Subgroups are classed as out of control if they are greater or less than the control limits.  
Each subgroup can be labelled and the labels will be shown on the chart.  
The subgroups for each chart that are out of control can be returned as a pandas Dataframe.

For an example of usage see: [/examples/example_x_bar_r.py](/examples/example_x_bar_r.py)

![Water_chart.svg](examples%2FWater_chart.svg)


### Calculation of control limits
Calculations use the method given in: <em>Wheeler, D.J. and Chambers, D.S. (1992) Understanding statistical process control. SPC Press, Incorporated, p. 44.</em>  

$UCL_{\bar{X}} = \bar{\bar{X}} + A_{2}\bar{R}$  
  
$LCL_{\bar{X}} = \bar{\bar{X}} - A_{2}\bar{R}$  
  
$UCL_{R} = D_{4}\bar{R}$  
  
$CL_{R} = \bar{R}$  
  
$LCL_{R} = D_{3}\bar{R}$

Where:

$\bar{\bar{X}}$ is the average of all the values in the subgroups.  
$\bar{R}$ is the average of the subgroup ranges.  
$A_{2}$ is the value of the constant corresponding to the subgroup size.  
$D_{3}$ is the value of the constant corresponding to the subgroup size.  
$D_{4}$ is the value of the constant corresponding to the subgroup size.
### Factor values
Constants for the computing of control limits are listed in [factor_values_for_shewart_charts.csv](constants/factor_values_for_shewart_charts.csv).  
Values where extracted from: 
<em>[A Note on the Factor Values of Three Common Shewhart Variables Control Charts. Henry H. Bi. 2015. Communications in Statistics - Theory and Methods (0361-0926). 44(13): 2655-2673. DOI:10.1080/03610926.2014.968732](https://www.researchgate.net/publication/275236350_A_Note_on_the_Factor_Values_of_Three_Common_Shewhart_Variables_Control_Charts_Henry_H_Bi_2015_Communications_in_Statistics_-_Theory_and_Methods_0361-0926_4413_2655-2673_httpdxdoiorg1010800361092620149)</em>

## TODO
Allow the use of Western Electric rules.
