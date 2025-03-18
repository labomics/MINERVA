| Dataset | Sampling Ratio (%)| Training Cells Num |  SSL Task Number  |   Training Time(h)   | **knowledge Transfer Time(s)** |
| :-----: |       :---:       |       :----:       |       :----:      |         :----:       |              :----:            |
|    DM   |         10        |         664        |         1         |          0.92        |             **5.11**           |
|    DM   |         20        |        1,336       |         1         |          1.09        |             **5.04**           |
|    DM   |         50        |        3,346       |         1         |          1.58        |             **6.32**           |
|    DM   |        100        |        6,697       |         1         |          2.66        |             **6.54**           |
|         |                   |                    |                   |                      |                                |
|   SLN   |         10        |        2,339       |         3         |          4.51        |            **13.26**           |
|   SLN   |         20        |        4,678       |         3         |          8.05        |            **13.65**           |
|   SLN   |         50        |       11,731       |         3         |         13.06        |            **16.65**           |
|   SLN   |        100        |       23,470       |         3         |         26.50        |            **19.81**           |
|         |                   |                    |                   |                      |                                |
|  BMMC   |         10        |        5,893       |         4         |         10.11        |            **35.09**           |
|  BMMC   |         20        |       17,840       |         4         |         18.89        |            **36.31**           |
|  BMMC   |         50        |       29,975       |         4         |         46.27        |            **40.44**           |
|  BMMC   |        100        |       60,155       |         4         |         97.63        |		         **49.98**           |		

**Each cell undergoes _K_ distinct SSL strategies tasks, increasing the number of training samples per epoch from _N_ (original input) to _KN_ ( additional with SSL-constructed data).**
