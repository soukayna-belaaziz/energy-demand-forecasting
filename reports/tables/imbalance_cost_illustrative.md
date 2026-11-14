| model                 |   days |   mean_abs_error_MWh |   mean_cost_EUR_per_day |   total_cost_EUR |
|:----------------------|-------:|---------------------:|------------------------:|-----------------:|
| SARIMAX               |    424 |                  777 |                    4811 |          2039722 |
| SARIMAX baseline      |    424 |                  805 |                    4948 |          2097812 |
| Combination (inv_mse) |    424 |                  806 |                    6049 |          2564707 |
| Combination (equal)   |    424 |                  969 |                    8342 |          3537044 |
| LightGBM              |    424 |                 1475 |                   14374 |          6094663 |
| Seasonal naive        |    424 |                 3477 |                   24773 |         10503632 |
