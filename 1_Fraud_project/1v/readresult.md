PS D:\demo> python fraud.py

## 训练集形状：(1296675, 23)

## 测试集形状：(555719, 23)

## 原始特征数量：22
特征列表：['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'category', 'amt', 'first', 'last', 'gender', 'street', 'city', 'state', 'zip', 'lat', 'long', 'city_pop', 'job', 'dob', 'trans_num', 'unix_time', 'merch_lat', 'merch_long']

## 训练集缺失值统计：
Series([], dtype: int64)
--训练集无缺失值

## 测试集缺失值统计：
Series([], dtype: int64)
--测试集无缺失值

D:\demo\fraud.py:48: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.     

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df[col].fillna(df[col].median(), inplace=True)
D:\demo\fraud.py:50: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.     

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df[col].fillna(df[col].mode()[0], inplace=True)

## 训练集目标变量分布：
is_fraud
0    1289169
1       7506
Name: count, dtype: int64
Checking whether there is an H2O instance running at http://localhost:54321..... not found.
Attempting to start a local H2O server...
; Java HotSpot(TM) 64-Bit Server VM (build 25.441-b07, mixed mode)
  Starting server from C:\Users\lenovo\anac\Lib\site-packages\h2o\backend\bin\h2o.jar
  Ice root: C:\Users\lenovo\AppData\Local\Temp\tmpgyjl5xe8
  JVM stdout: C:\Users\lenovo\AppData\Local\Temp\tmpgyjl5xe8\h2o_lenovo_started_from_python.out
  JVM stderr: C:\Users\lenovo\AppData\Local\Temp\tmpgyjl5xe8\h2o_lenovo_started_from_python.err
  Server is running at http://127.0.0.1:54321
Connecting to H2O server at http://127.0.0.1:54321 ... successful.
--------------------------  -----------------------------
H2O_cluster_uptime:         04 secs
H2O_cluster_timezone:       Asia/Shanghai
H2O_data_parsing_timezone:  UTC
H2O_cluster_version:        3.46.0.9
H2O_cluster_version_age:    3 months and 1 day
H2O_cluster_name:           H2O_from_python_lenovo_xlcex3
H2O_cluster_total_nodes:    1
H2O_cluster_free_memory:    7.103 Gb
H2O_cluster_total_cores:    0
H2O_cluster_allowed_cores:  0
H2O_cluster_status:         locked, healthy
H2O_connection_url:         http://127.0.0.1:54321
H2O_connection_proxy:       {"http": null, "https": null}
H2O_internal_security:      False
Python_version:             3.13.5 final
--------------------------  -----------------------------
Parse progress: |████████████████████████████████████████████████████████████████ (done)| 100%
Parse progress: |████████████████████████████████████████████████████████████████ (done)| 100%

开始 H2O AutoML 训练...
AutoML progress: |                                                               |   0%
18:21:14.53: AutoML: XGBoost is not available; skipping it.
18:21:14.75: _train param, Dropping bad and constant columns: [trans_num]

AutoML progress: |█████████████████████████▏                                     |  40%
18:25:02.987: _train param, Dropping bad and constant columns: [trans_num]

AutoML progress: |████████████████████████████████████████████████▊              |  77%
18:28:55.276: _train param, Dropping unused columns: [trans_num]

AutoML progress: |█████████████████████████████████████████████████▌             |  78%
18:29:02.182: _train param, Dropping bad and constant columns: [trans_num]

AutoML progress: |████████████████████████████████████████████████████▎          |  83%
18:29:29.246: _train param, Dropping bad and constant columns: [trans_num]

AutoML progress: |██████████████████████████████████████████████████████▊        |  87%
18:29:53.690: _train param, Dropping bad and constant columns: [trans_num]

AutoML progress: |█████████████████████████████████████████████████████████▏     |  90%
18:30:17.356: _train param, Dropping bad and constant columns: [trans_num]

AutoML progress: |███████████████████████████████████████████████████████████▌   |  94%
18:30:40.973: _train param, Dropping unused columns: [trans_num]

AutoML progress: |████████████████████████████████████████████████████████████▎  |  95%
18:30:48.45: _train param, Dropping unused columns: [trans_num]

AutoML progress: |█████████████████████████████████████████████████████████████▎ |  97%
18:30:57.497: _train param, Dropping bad and constant columns: [trans_num]

AutoML progress: |██████████████████████████████████████████████████████████████▏|  98%
18:31:06.537: _train param, Dropping bad and constant columns: [trans_num]

AutoML progress: |███████████████████████████████████████████████████████████████ (done)| 100%

18:31:12.138: _train param, Dropping bad and constant columns: [trans_num]


AutoML 排行榜（按 AUC 排序）：
model_id                                                      auc    logloss      aucpr    mean_per_class_error       rmse         mse
StackedEnsemble_AllModels_1_AutoML_1_20260225_182114     0.956278  0.0122574  0.728072                 0.154287  0.0502616  0.00252623
StackedEnsemble_BestOfFamily_2_AutoML_1_20260225_182114  0.955067  0.012373   0.7299                   0.15823   0.0502848  0.00252856
GBM_1_AutoML_1_20260225_182114                           0.952288  0.0230068  0.723354                 0.165209  0.0697577  0.00486614
StackedEnsemble_BestOfFamily_1_AutoML_1_20260225_182114  0.952128  0.0127272  0.722531                 0.166783  0.050916   0.00259243
GBM_3_AutoML_1_20260225_182114                           0.947206  0.0268185  0.490289                 0.228214  0.0747635  0.00558958
GBM_2_AutoML_1_20260225_182114                           0.941936  0.0267941  0.473285                 0.229732  0.0746848  0.00557782
GBM_4_AutoML_1_20260225_182114                           0.933355  0.0280748  0.535719                 0.180594  0.0750347  0.0056302
DRF_1_AutoML_1_20260225_182114                           0.907128  0.053177   0.600988                 0.218846  0.0708785  0.00502376
GLM_1_AutoML_1_20260225_182114                           0.831594  0.033055   0.18977                  0.278229  0.0762161  0.0058089
GBM_5_AutoML_1_20260225_182114                           0.824508  0.0347216  0.132187                 0.453623  0.075797   0.00574519
XRT_1_AutoML_1_20260225_182114                           0.553131  0.171922   0.0478583                0.438209  0.0770337  0.0059342
[11 rows x 7 columns]


最佳模型：
Model Details
=============
H2OStackedEnsembleEstimator : Stacked Ensemble
Model Key: StackedEnsemble_AllModels_1_AutoML_1_20260225_182114


Model Summary for Stacked Ensemble:
key                                   value
------------------------------------  ----------------
Stacking strategy                     cross_validation
Number of base models (used / total)  3/6
# GBM base models (used / total)      2/4
# DRF base models (used / total)      1/1
# GLM base models (used / total)      0/1
Metalearner algorithm                 GLM
Metalearner fold assignment scheme    Random
Metalearner nfolds                    5
Metalearner fold_column
Custom metalearner hyperparameters    None

ModelMetricsBinomialGLM: stackedensemble
** Reported on train data. **

MSE: 0.00025021751323191834
RMSE: 0.01581826517769627
LogLoss: 0.0019166564952520762
AUC: 1.0
AUCPR: 1.0
Gini: 1.0
Null degrees of freedom: 10050
Residual degrees of freedom: 10047
Null deviance: 754.8118324670592
Residual deviance: 38.52862886755723
AIC: 46.52862886755723

Confusion Matrix (Act/Pred) for max f1 @ threshold = 0.8786031602035829
       0     1    Error    Rate
-----  ----  ---  -------  -------------
0      9989  0    0        (0.0/9989.0)
1      0     62   0        (0.0/62.0)
Total  9989  62   0        (0.0/10051.0)

Maximum Metrics: Maximum metrics at their respective thresholds
metric                       threshold    value     idx
---------------------------  -----------  --------  -----
max f1                       0.878603     1         55
max f2                       0.878603     1         55
max f0point5                 0.878603     1         55
max accuracy                 0.878603     1         55
max precision                0.995538     1         0
max recall                   0.878603     1         55
max specificity              0.995538     1         0
max absolute_mcc             0.878603     1         55
max min_per_class_accuracy   0.878603     1         55
max mean_per_class_accuracy  0.878603     1         55
max tns                      0.995538     9989      0
max fns                      0.995538     61        0
max fps                      0.000442701  9989      399
max tps                      0.878603     62        55
max tnr                      0.995538     1         0
max fnr                      0.995538     0.983871  0
max fpr                      0.000442701  1         399
max tpr                      0.878603     1         55

Gains/Lift Table: Avg response rate:  0.62 %, avg score:  0.74 %
group    cumulative_data_fraction    lower_threshold    lift     cumulative_lift    response_rate    score        cumulative_response_rate  cumulative_score    capture_rate    cumulative_capture_rate    gain     cumulative_gain    kolmogorov_smirnov
-------  --------------------------  -----------------  -------  -----------------  ---------------  -----------  --------------------------  ------------------  --------------  -------------------------  -------  -----------------  --------------------
1        0.0100488                   0.0211117          99.5149  99.5149            0.613861         0.644409     0.613861                    0.644409            1               1                          9851.49  9851.49            0.996096
2        0.0200975                   0.00544097         0        49.7574            0                0.00896815   0.306931                    0.326689            0               1                          -100     4875.74            0.985985
3        0.0300468                   0.00398389         0        33.2815            0                0.00455717   0.205298                    0.220022            0               1                          -100     3228.15            0.975974
4        0.0400955                   0.00313818         0        24.9404            0                0.00348334   0.153846                    0.165753            0               1                          -100     2394.04            0.965862
5        0.0500448                   0.00257496         0        19.9821            0                0.00281852   0.12326                     0.133361            0               1                          -100     1898.21            0.955851
6        0.10009                     0.00113478         0        9.99105            0                0.00157522   0.0616302                   0.067468            0               1                          -100     899.105            0.905496
7        0.150035                    0.00092814         0        6.66512            0                0.00101231   0.0411141                   0.0453455           0               1                          -100     566.512            0.855241
8        0.20008                     0.00082397         0        4.99801            0                0.000872475  0.0308304                   0.0342217           0               1                          -100     399.801            0.804885
9        0.30007                     0.000722147        0        3.33256            0                0.000767059  0.020557                    0.0230738           0               1                          -100     233.256            0.704275
10       0.40006                     0.000650304        0        2.49963            0                0.000680658  0.015419                    0.0174769           0               1                          -100     149.963            0.603664
11       0.50005                     0.000611688        0        1.9998             0                0.000629068  0.0123359                   0.014108            0               1                          -100     99.9801            0.503053
12       0.60004                     0.000611468        0        1.66656            0                0.000611604  0.0102802                   0.011859            0               1                          -100     66.6556            0.402443
13       0.70003                     0.000609534        0        1.42851            0                0.000610835  0.00881182                  0.0102524           0               1                          -100     42.8511            0.301832
14       0.80002                     0.0006046          0        1.24997            0                0.000607075  0.00771048                  0.00904684          0               1                          -100     24.9969            0.201221
15       0.90001                     0.000596259        0        1.1111             0                0.000601014  0.00685386                  0.00810852          0               1                          -100     11.1099            0.100611
16       1                           0.000442061        0        1                  0                0.000581326  0.00616854                  0.00735588          0               1                          -100     0                  0

ModelMetricsBinomialGLM: stackedensemble
** Reported on cross-validation data. **

MSE: 0.0025262287790052394
RMSE: 0.0502616034265247
LogLoss: 0.012257397915743254
AUC: 0.9562783912665239
AUCPR: 0.7280716214929632
Gini: 0.9125567825330478
Null degrees of freedom: 1296675
Residual degrees of freedom: 1296672
Null deviance: 92309.76495935638
Residual deviance: 31787.747399588603
AIC: 31795.747399588603

Confusion Matrix (Act/Pred) for max f1 @ threshold = 0.30684058645349627
       0            1     Error    Rate
-----  -----------  ----  -------  ------------------
0      1.28726e+06  1915  0.0015   (1915.0/1289170.0)
1      2305         5201  0.3071   (2305.0/7506.0)
Total  1.28956e+06  7116  0.0033   (4220.0/1296676.0)

Maximum Metrics: Maximum metrics at their respective thresholds
metric                       threshold    value        idx
---------------------------  -----------  -----------  -----
max f1                       0.306841     0.711394     213
max f2                       0.104211     0.722494     282
max f0point5                 0.655886     0.76637      121
max accuracy                 0.548204     0.996914     150
max precision                0.998828     1            0
max recall                   6.65402e-05  1            399
max specificity              0.998828     1            0
max absolute_mcc             0.413172     0.710837     187
max min_per_class_accuracy   0.00249363   0.908056     377
max mean_per_class_accuracy  0.00745022   0.922065     356
max tns                      0.998828     1.28917e+06  0
max fns                      0.998828     7415         0
max fps                      6.65402e-05  1.28917e+06  399
max tps                      6.65402e-05  7506         399
max tnr                      0.998828     1            0
max fnr                      0.998828     0.987876     0
max fpr                      6.65402e-05  1            399
max tpr                      6.65402e-05  1            399

Gains/Lift Table: Avg response rate:  0.58 %, avg score:  0.58 %
group    cumulative_data_fraction    lower_threshold    lift       cumulative_lift    response_rate    score        cumulative_response_rate    cumulative_score    capture_rate    cumulative_capture_rate    gain      cumulative_gain    kolmogorov_smirnov
-------  --------------------------  -----------------  ---------  -----------------  ---------------  -----------  --------------------------  ------------------  --------------  -------------------------  --------  -----------------  --------------------
1        0.0100002                   0.0376461          80.1344    80.1344            0.46387          0.459685     0.46387                     0.459685            0.801359        0.801359                   7913.44   7913.44            0.795966
2        0.0200004                   0.00844005         5.43555    42.785             0.0314645        0.0163533    0.247667                    0.238019            0.0543565       0.855715                   443.555   4178.5             0.840581
3        0.0300006                   0.00595332         1.19902    28.923             0.0069407        0.00695916   0.167425                    0.160999            0.0119904       0.867706                   19.9019   2792.3             0.842583
4        0.0400007                   0.00490057         0.919248   21.9221            0.0053212        0.00537392   0.126899                    0.122093            0.00919265      0.876898                   -8.07524  2092.21            0.84177
5        0.0500002                   0.00423563         0.786084   17.6951            0.00455036       0.00455253   0.102431                    0.0985862           0.00786038      0.884759                   -21.3916  1669.51            0.839619
6        0.1                         0.0022394          0.532905   9.11401            0.0030848        0.00320428   0.0527578                   0.0508952           0.0266454       0.911404                   -46.7095  811.401            0.816128
7        0.15                        0.0013822          0.42366    6.21723            0.00245242       0.00166179   0.0359893                   0.0344841           0.0211831       0.932587                   -57.634   521.723            0.787143
8        0.200002                    0.00117025         0.141216   4.69819            0.000817447      0.00126463   0.0271962                   0.026179            0.00706102      0.939648                   -85.8784  369.819            0.743953
9        0.30001                     0.00103082         0.0812618  3.15914            0.000470396      0.00107447   0.0182872                   0.0178105           0.00812683      0.947775                   -91.8738  215.914            0.651536
10       0.400061                    0.0009561          0.0732378  2.38739            0.000423948      0.00100221   0.0138198                   0.0136069           0.00732747      0.955103                   -92.6762  138.739            0.558274
11       0.500003                    0.00091491         0.0479892  1.91979            0.000277793      0.000937114  0.011113                    0.0110744           0.00479616      0.959899                   -95.2011  91.9786            0.462573
12       0.600013                    0.00075457         0.103907   1.61712            0.000601481      0.000830885  0.00936092                  0.00936704          0.0103917       0.97029                    -89.6093  61.7117            0.372434
13       0.700003                    0.00073037         0.0359747  1.39126            0.000208245      0.000741682  0.00805353                  0.00813497          0.00359712      0.973888                   -96.4025  39.1262            0.275479
14       0.800003                    0.0006831          0.0559551  1.22435            0.000323904      0.000716954  0.00708732                  0.00720772          0.00559552      0.979483                   -94.4045  22.4349            0.180525
15       0.900022                    5.907e-05          0.181154   1.10842            0.00104864       0.00016764   0.00641625                  0.00642536          0.0181188       0.997602                   -81.8846  10.8419            0.098148
16       1                           2.723e-05          0.0239861  1                  0.000138847      5.5531e-05   0.00578865                  0.00578852          0.00239808      1                          -97.6014  0                  0

Cross-Validation Metrics Summary:
                      mean          sd              cv_1_valid    cv_2_valid    cv_3_valid    cv_4_valid    cv_5_valid
--------------------  ------------  --------------  ------------  ------------  ------------  ------------  ------------
accuracy              0.9968673     0.00010640798   0.99686295    0.99699557    0.99690104    0.9967012     0.9968759
aic                   6365.5493     188.69879       6362.64       6191.635      6295.324      6684.939      6293.2095
auc                   0.9567096     0.0042284545    0.95737314    0.960968      0.95813525    0.957458      0.94961363
err                   0.003132679   0.00010640798   0.003137073   0.0030044143  0.0030989705  0.0032988403  0.0031240971
err_count             812.4         27.097971       812.0         780.0         804.0         855.0         811.0
f0point5              0.7398343     0.007213865     0.7407966     0.7507553     0.74109226    0.733398      0.73312926
f1                    0.7133876     0.0041321097    0.70749277    0.7182081     0.71630204    0.7122181     0.712717
f2                    0.6888349     0.0068878997    0.6770546     0.68836564    0.6931166     0.6922272     0.6934105
lift_top_group        80.10356      0.48699796      80.07527      80.432816     80.12884      79.3137       80.56717
loglikelihood         0.0           0.0             0.0           0.0           0.0           0.0           0.0
---                   ---           ---             ---           ---           ---           ---           ---
mean_per_class_error  0.16391619    0.004752132     0.17149828    0.16565606    0.16138813    0.16093498    0.16010344
mse                   0.0025262733  8.050958e-05    0.0025345741  0.002472528   0.0024856161  0.002663159   0.0024754896
null_deviance         18461.953     328.8428        18361.205     18288.021     18409.35      19035.365     18215.824
pr_auc                0.72823477    0.0050077066    0.725689      0.7263697     0.7349859     0.72248024    0.7316489
precision             0.7586335     0.011324096     0.7647975     0.7741433     0.75859493    0.74823195    0.7473997
r2                    0.5611122     0.005165822     0.5577395     0.5649575     0.56643665    0.55400485    0.5624225
recall                0.67341727    0.009581081     0.65817696    0.6698113     0.6784759     0.6795119     0.6811104
residual_deviance     6357.5493     188.69879       6354.64       6183.635      6287.324      6676.939      6285.2095
rmse                  0.050257023   0.0007943593    0.050344553   0.04972452    0.04985595    0.051605806   0.04975429
specificity           0.9987503     0.000104613064  0.9988265     0.9988766     0.99874777    0.9986181     0.9986828
[24 rows x 8 columns]


测试集评估结果：
ModelMetricsBinomialGLM: stackedensemble
** Reported on test data. **

MSE: 0.003990992554629233
RMSE: 0.06317430296116636
LogLoss: 0.0261871404714476
AUC: 0.689426192616552
AUCPR: 0.0712794491319213
Gini: 0.37885238523310405
Null degrees of freedom: 555718
Residual degrees of freedom: 555715
Null deviance: 28528.97556194064
Residual deviance: 29105.383031304784
AIC: 29113.383031304784

Confusion Matrix (Act/Pred) for max f1 @ threshold = 0.10620789162091057
       0       1     Error    Rate
-----  ------  ----  -------  -----------------
0      552429  1145  0.0021   (1145.0/553574.0)
1      1841    304   0.8583   (1841.0/2145.0)
Total  554270  1449  0.0054   (2986.0/555719.0)

Maximum Metrics: Maximum metrics at their respective thresholds
metric                       threshold    value     idx
---------------------------  -----------  --------  -----
max f1                       0.106208     0.169171  259
max f2                       0.0848813    0.15275   270
max f0point5                 0.84738      0.195299  38
max accuracy                 0.879756     0.99625   27
max precision                0.994407     1         0
max recall                   0.000555669  1         399
max specificity              0.994407     1         0
max absolute_mcc             0.876827     0.180696  28
max min_per_class_accuracy   0.000689243  0.651748  394
max mean_per_class_accuracy  0.000763353  0.685406  392
max tns                      0.994407     553574    0
max fns                      0.994407     2134      0
max fps                      0.000555669  553574    399
max tps                      0.000555669  2145      399
max tnr                      0.994407     1         0
max fnr                      0.994407     0.994872  0
max fpr                      0.000555669  1         399
max tpr                      0.000555669  1         399

Gains/Lift Table: Avg response rate:  0.39 %, avg score:  0.21 %
group    cumulative_data_fraction    lower_threshold    lift      cumulative_lift    response_rate    score        cumulative_response_rate    cumulative_score    capture_rate    cumulative_capture_rate    gain      cumulative_gain    kolmogorov_smirnov
-------  --------------------------  -----------------  --------  -----------------  ---------------  -----------  --------------------------  ------------------  --------------  -------------------------  --------  -----------------  --------------------
1        0.0100015                   0.00952636         16.3146   16.3146            0.0629723        0.130687     0.0629723                   0.130687            0.16317         0.16317                    1531.46   1531.46            0.153762
2        0.0200011                   0.00427092         1.86487   9.0904             0.00719813       0.00591784   0.0350877                   0.0683081           0.018648        0.181818                   86.4866   809.04             0.162444
3        0.0300008                   0.00319943         0.932433  6.37124            0.00359906       0.00364646   0.0245921                   0.0467555           0.00932401      0.191142                   -6.75672  537.124            0.161766
4        0.0400004                   0.00261067         0.27973   4.84843            0.00107972       0.00288006   0.0187143                   0.0357872           0.0027972       0.193939                   -72.027   384.843            0.154535
5        0.0500001                   0.00199509         1.21216   4.1212             0.00467878       0.00236674   0.0159073                   0.0291033           0.0121212       0.206061                   21.2163   312.12             0.156665
6        0.1                         0.00102077         1.85547   2.98834            0.00716188       0.00127372   0.0115346                   0.0151885           0.0927739       0.298834                   85.5475   198.834            0.199605
7        0.15                        0.000873283        1.72494   2.56721            0.00665803       0.000937959  0.00990907                  0.0104383           0.0862471       0.385082                   72.4939   156.721            0.235992
8        0.2                         0.000789674        2.39627   2.52447            0.00924926       0.000827553  0.00974412                  0.00803564          0.119814        0.504895                   139.627   152.447            0.306076
9        0.300001                    0.000699839        1.38927   2.14607            0.00536241       0.000743844  0.00828355                  0.00560504          0.138928        0.643823                   38.9275   114.607            0.345155
10       0.400001                    0.000640333        0.158508  1.64918            0.000611819      0.000664146  0.00636562                  0.00436982          0.0158508       0.659674                   -84.1492  64.9181            0.260679
11       0.500001                    0.000611646        0.172494  1.35384            0.000665803      0.000620547  0.00522565                  0.00361996          0.0172494       0.676923                   -82.7506  35.3844            0.177608
12       0.599999                    0.000611276        0.624719  1.23232            0.00241133       0.000611537  0.00475661                  0.00311857          0.0624709       0.739394                   -37.5281  23.2325            0.139935
13       0.699999                    0.000608441        1.19347   1.22677            0.00460664       0.00061029   0.00473518                  0.00276024          0.119347        0.858741                   19.3471   22.6774            0.159357
14       0.8                         0.000603293        0.755243  1.16783            0.00291514       0.00060598   0.00450768                  0.00249096          0.0755245       0.934266                   -24.4757  16.7833            0.134786
15       0.9                         0.000592964        0.414918  1.08418            0.00160153       0.000598834  0.00418477                  0.00228072          0.0414918       0.975758                   -58.5082  8.41753            0.0760513
16       1                           0.00034598         0.242424  1                  0.000935723      0.000576509  0.00385986                  0.0021103           0.0242424       1                          -75.7576  0                  0

## 最佳模型已保存至：D:\demo\h2o_models\StackedEnsemble_AllModels_1_AutoML_1_20260225_182114
H2O session _sid_abe5 closed.

运行完成！


## 描述
H2O AutoML 运行日志：模型训练已完成，但测试集上的表现不佳（AUC ≈ 0.689，AUCPR ≈ 0.071），与交叉验证结果（AUC ≈ 0.956）差距较大，存在明显的过拟合。

---

##  运行结果总结

| 指标                | 训练集（交叉验证） | 测试集     |
|---------------------|-------------------|------------|
| AUC                 | 0.956             | 0.689      |
| AUCPR               | 0.728             | 0.071      |
| 欺诈样本召回率      | 69.3% (交叉验证)  | 14.2%      |
| 欺诈样本精确率      | 75.9% (交叉验证)  | 21.0%      |

- **最佳模型**：`StackedEnsemble_AllModels_1`（堆叠集成模型）
- **训练数据**：1,296,675 笔交易，其中欺诈样本 7,506 笔（0.58%）
- **测试数据**：555,719 笔交易，其中欺诈样本 2,145 笔（0.39%）
- **运行时间**：10 分钟

---

##  关键问题分析

### 1. 过拟合
- 交叉验证 AUC 高达 0.956，测试集 AUC 为 0.689，模型在训练数据上表现优异，但无法泛化到新数据。
- 测试集 AUCPR 仅为 0.071，远低于交叉验证的 0.728，表明模型对少数类（欺诈）的预测能力极弱。

### 2. 测试集上欺诈检测效果差
- 混淆矩阵显示：2,145 个真实欺诈中仅检出 304 个，召回率仅 14.2%；而误报为欺诈的正常交易有 1,145 笔。
- 漏掉大量欺诈且误报较多。

### 3. 可能的原因
#### a) **训练/测试集划分不当**
   - 数据集按时间顺序划分（`fraudTrain.csv` 是早期交易，`fraudTest.csv` 是后期交易）。直接混合训练导致 **时间泄露**：模型可能学到了时间相关的模式（如特定季节的欺诈行为）。
   - 日志中未对 `trans_date_trans_time` 做任何处理，模型可能利用了时间信息进行过拟合。

#### b) **特征中包含未来信息或标识符**
   - `trans_num` 被 H2O 自动丢弃（“Dropping bad and constant columns”），但 `Unnamed: 0`（行号）被保留，可能成为过拟合的源头。
   - `cc_num`（卡号）可能泄露客户行为模式，测试集中新卡号会导致模型失效。
   - `dob`（出生日期）等个人信息可能导致模型学习到训练集中特定人群的规律，但测试集中人群分布可能不同。

#### c) **数据分布差异**
   - 训练集欺诈率 0.58%，测试集欺诈率 0.39%。
   - 测试集上的平均预测分数（0.21%）远低于训练集（0.58%），模型对测试集的置信度整体偏低。

#### d) **AutoML 默认配置的限制**
   - 使用了 `balance_classes=True` 对少数类进行上采样，但过采样可能加剧过拟合，尤其是当训练集和测试集分布不一致时。
   - 交叉验证为随机 5 折，未考虑时间顺序，导致验证集上的表现过于乐观。

#### e) **未做特征工程**
   - 某些特征（如时间戳）需要分解为更有意义的成分（如小时、是否节假日）才能帮助模型泛化。
   - 分类特征（如 `merchant`、`category`、`job`）基数很高，过拟合到训练集中的具体商户/职业。

---

##  第一版改进做法

### 1. **验证并调整训练/测试划分方式**
   - 两个文件是按照时间分割的，使用 **时间序列交叉验证**，而非随机划分。
   - 在 H2O AutoML 中可通过设置 `fold_column` 指定自定义的交叉验证折（例如按月份分折）。

### 2. **剔除易导致过拟合的特征**
   - 删除 `Unnamed: 0`（行号）、`trans_num`（交易唯一 ID）、`cc_num`（卡号）等标识性特征。
   - 高基数的分类变量（如 `merchant`、`job`）可考虑进行目标编码或频数编码，但需在交叉验证中避免泄露。

### 3. **时间特征工程**
   - 从 `trans_date_trans_time` 提取：小时、星期几、是否周末、月份等。
   - 这些特征有助于模型捕捉欺诈行为的时间模式，且可能泛化到未来数据。

### 4. **调整 AutoML 参数**
   - 增加 `nfolds` 到 10（基于时间）。
   - 尝试关闭 `balance_classes`，或使用更温和的采样比例（如 `max_after_balance_size=2.0`）。
   - 限制模型复杂度：例如排除某些易过拟合的算法（如 `GBM` 或 `DRF` 的深度树），或设置 `max_runtime_secs` 较短以防止过度训练。
   - 使用 `stopping_metric="AUC"` 并设置早停轮数。

### 5. **尝试更稳健的模型**
   - 虽然堆叠模型在交叉验证中表现最佳，但在测试集上过拟合。尝试单独的 GBM 或 GLM 作为基线。
   - GLM 在测试集上 AUC 仅 0.83，但仍远低于交叉验证值，说明简单模型同样过拟合。

### 6. **检查数据泄露**
   - 训练集和测试集中应当没有重叠的交易。若存在，应确保按时间分割后，同一卡号的交易不跨文件出现。

### 7. **增加数据量**
   - 使用更大规模的训练数据（合并两个文件并按时间重新划分训练/验证/测试）。

---

## 后续步骤

1. **重新划分数据**：重新确定时间分割点。
2. **简化特征**：移除标识列，处理高基数分类变量。
3. **基础特征工程**：从时间戳提取周期特征。
4. **重新运行 AutoML**，并监控验证集上的 AUC 变化。
5. **若仍不理想**，可考虑使用 LightGBM 或 XGBoost 手动调参，并配合时间序列交叉验证。
