|                                                                |     MAE |    RMSE |   MAPE |     bias |       n |   coverage_raw |   coverage_conformal |   width_conformal |
|:---------------------------------------------------------------|--------:|--------:|-------:|---------:|--------:|---------------:|---------------------:|------------------:|
| ('2021 pre-crisis', 'SARIMAX baseline')                        |  78.795 | 106.601 |  3.001 |   -9.492 | 362.000 |          0.826 |              nan     |           nan     |
| ('2021 pre-crisis', 'SARIMAX')                                 |  75.233 | 103.983 |  2.830 |  -10.068 | 362.000 |          0.823 |                0.931 |           371.199 |
| ('2021 pre-crisis', 'LightGBM')                                | 153.438 | 209.099 |  5.186 |   36.534 | 362.000 |        nan     |              nan     |           nan     |
| ('2021 pre-crisis', 'Seasonal naive')                          | 440.350 | 655.487 | 14.656 |  -12.795 | 362.000 |        nan     |              nan     |           nan     |
| ('2021 pre-crisis', 'Combination (equal)')                     |  96.562 | 129.901 |  3.369 |   13.233 | 362.000 |        nan     |              nan     |           nan     |
| ('2021 pre-crisis', 'Combination (inv_mse)')                   |  79.670 | 107.559 |  2.890 |    3.037 | 362.000 |        nan     |              nan     |           nan     |
| ('2022 crisis', 'SARIMAX baseline')                            |  74.542 |  97.891 |  3.511 |   -8.037 | 365.000 |          0.847 |              nan     |           nan     |
| ('2022 crisis', 'SARIMAX')                                     |  71.338 |  94.158 |  3.291 |   -7.921 | 365.000 |          0.838 |                0.929 |           372.223 |
| ('2022 crisis', 'LightGBM')                                    | 175.234 | 248.221 |  7.573 | -138.445 | 365.000 |        nan     |              nan     |           nan     |
| ('2022 crisis', 'Seasonal naive')                              | 357.972 | 531.936 | 14.249 |  -19.982 | 365.000 |        nan     |              nan     |           nan     |
| ('2022 crisis', 'Combination (equal)')                         | 106.194 | 143.986 |  4.640 |  -73.183 | 365.000 |        nan     |              nan     |           nan     |
| ('2022 crisis', 'Combination (inv_mse)')                       |  79.472 | 104.085 |  3.575 |  -39.479 | 365.000 |        nan     |              nan     |           nan     |
| ('  2022 Jan-Aug', 'SARIMAX baseline')                         |  69.311 |  93.375 |  3.272 |  -10.573 | 243.000 |          0.860 |              nan     |           nan     |
| ('  2022 Jan-Aug', 'SARIMAX')                                  |  67.851 |  90.879 |  3.162 |  -10.691 | 243.000 |          0.856 |                0.938 |           377.372 |
| ('  2022 Jan-Aug', 'LightGBM')                                 |  90.917 | 121.870 |  4.170 |  -36.622 | 243.000 |        nan     |              nan     |           nan     |
| ('  2022 Jan-Aug', 'Seasonal naive')                           | 322.848 | 482.390 | 12.808 |  -62.139 | 243.000 |        nan     |              nan     |           nan     |
| ('  2022 Jan-Aug', 'Combination (equal)')                      |  71.475 |  94.748 |  3.288 |  -23.657 | 243.000 |        nan     |              nan     |           nan     |
| ('  2022 Jan-Aug', 'Combination (inv_mse)')                    |  67.758 |  89.474 |  3.137 |  -17.590 | 243.000 |        nan     |              nan     |           nan     |
| ('  2022 Sep-Dec (demand reduction)', 'SARIMAX baseline')      |  84.963 | 106.316 |  3.987 |   -2.985 | 122.000 |          0.820 |              nan     |           nan     |
| ('  2022 Sep-Dec (demand reduction)', 'SARIMAX')               |  78.284 | 100.372 |  3.550 |   -2.403 | 122.000 |          0.803 |                0.910 |           361.968 |
| ('  2022 Sep-Dec (demand reduction)', 'LightGBM')              | 343.178 | 393.386 | 14.353 | -341.256 | 122.000 |        nan     |              nan     |           nan     |
| ('  2022 Sep-Dec (demand reduction)', 'Seasonal naive')        | 427.934 | 618.915 | 17.119 |   63.986 | 122.000 |        nan     |              nan     |           nan     |
| ('  2022 Sep-Dec (demand reduction)', 'Combination (equal)')   | 175.348 | 210.108 |  7.332 | -171.830 | 122.000 |        nan     |              nan     |           nan     |
| ('  2022 Sep-Dec (demand reduction)', 'Combination (inv_mse)') | 102.804 | 128.322 |  4.447 |  -83.080 | 122.000 |        nan     |              nan     |           nan     |
