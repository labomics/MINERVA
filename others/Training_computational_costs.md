| Dataset | Sampling Ratio (%)| Training Cells Num |  SSL Task Number  |  Training Time (min) |   knowledge Transfer Time(s)   |
| :-----: |       :---:       |       :----:       |       :----:      |         :----:       |              :----:            |
|    DM   |         10        |         664        |         1         |          2.13        |               5.11             |
|    DM   |         20        |        1,336       |         1         |          3.10        |               5.04             |
|    DM   |         50        |        3,346       |         1         |          3.70        |               6.32             |
|    DM   |        100        |        6,697       |         1         |          5.00        |               6.54             |
|         |                   |                    |                   |                      |                                |
|   SLN   |         10        |        2,339       |         3         |          6.03        |              13.26             |
|   SLN   |         20        |        4,678       |         3         |         12.03        |              13.65             |
|   SLN   |         50        |       11,731       |         3         |         15.57        |              16.65             |
|   SLN   |        100        |       23,470       |         3         |         43.60        |              19.81             |
|         |                   |                    |                   |                      |                                |
|  BMMC   |         10        |        5,893       |         4         |         17.63        |              35.09             |
|  BMMC   |         20        |       17,840       |         4         |         36.83        |              36.31             |
|  BMMC   |         50        |       29,975       |         4         |        112.83        |              40.44             |
|  BMMC   |        100        |       60,155       |         4         |        201.73        |		           49.98             |		

**Each cell undergoes _K_ distinct SSL strategies tasks, increasing the number of training samples per epoch from _N_ (original input) to _KN_ ( additional with SSL-constructed data).**
