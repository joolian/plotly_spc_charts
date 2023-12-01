### Plotly charts for statistical process control

In progress

### Factor values
Factor values for the computing of control limits are listed in [factor_values_for_shewart_charts.csv](constants/factor_values_for_shewart_charts.csv).  
Values where extracted from: 
[A Note on the Factor Values of Three Common Shewhart Variables Control Charts. Henry H. Bi. 2015. Communications in Statistics - Theory and Methods (0361-0926). 44(13): 2655-2673. DOI:10.1080/03610926.2014.968732](https://www.researchgate.net/publication/275236350_A_Note_on_the_Factor_Values_of_Three_Common_Shewhart_Variables_Control_Charts_Henry_H_Bi_2015_Communications_in_Statistics_-_Theory_and_Methods_0361-0926_4413_2655-2673_httpdxdoiorg1010800361092620149)


Calculation of control limits for an Average and range chart for subgroups:  
$ A_{2}$ is the value of the constant corresponding to the subgroup size.  
$D_{3}$ is the value of the constant corresponding to the subgroup size.  
$D_{4}$ is the value of the constant corresponding to the subgroup size.  
$UCL_{\bar{X}} = \bar{\bar{X}} + A_{2}\bar{R}$  
$LCL_{\bar{X}} = \bar{\bar{X}} - A_{2}\bar{R}$  
$UCL_{R} = D_{4}\bar{R}$  
$CL_{R} = \bar{R}$  
$LCL_{R} = D_{3}\bar{R}$