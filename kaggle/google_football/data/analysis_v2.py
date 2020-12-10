import numpy as np

# if not work, do deeper analysis of step when opponent take it vs steps when no body takes it

43505
long_steps_data = [9, 8, 6, 9, 9, 9, 8, 4, 8, 9, 9, 1, 1, 1, 9, 8, 1, 2, 2, 8, 3, 8, 1, 2, 1, 7, 9, 4, 4, 9, 9, 2, 7, 9, 10, 9, 6, 11, 3, 2, 1, 9, 8, 8, 8, 1, 5, 6, 1, 3, 4, 9, 9, 9, 9, 9, 6, 7, 9, 9, 9, 10, 1, 3, 9, 8, 9, 2, 4, 9, 10, 1, 7, 8, 9, 7, 4, 3, 9, 9, 9, 7, 9, 2, 9, 8, 1, 8, 1, 2, 9, 9, 9, 2, 7, 1, 9, 7, 3, 4, 9, 2, 8, 6, 1, 11, 9, 9, 1, 3, 10, 1, 12, 7, 9, 9, 1, 1, 8, 9, 8, 8, 9, 9, 2, 9, 1, 3, 9, 9, 7, 3, 2, 1, 9, 1, 9, 7, 7, 9, 7, 9, 10, 9, 2, 4, 9, 1, 3, 11, 9, 1, 1, 9, 1, 7, 5, 1, 6, 12, 4, 4, 9, 10, 9, 5, 8, 1, 7, 9, 9, 9, 2, 2, 1, 9, 7, 10, 9, 8, 9, 3, 9, 10, 1, 11, 5, 6, 3, 3, 9, 9, 9, 10, 10, 3, 10, 10, 3, 5, 8, 9, 1, 1, 1, 7, 5, 9, 4, 8, 8, 2, 1, 1, 1, 7, 10, 9, 5, 3, 2, 2, 3, 1, 7, 1, 1, 1, 2, 1, 4, 1, 5, 8, 9, 9, 1, 9, 8, 9, 9, 9, 7, 7, 10, 10, 9, 9, 8, 9, 7, 9, 6, 7, 9, 9, 2, 2, 5, 6, 9, 7, 1, 1, 7, 1, 5, 8, 7, 3, 3, 8, 7, 9, 4, 9, 10, 9, 5, 1, 9, 6, 6, 7, 9, 9, 2, 9, 7, 9, 8, 2, 1, 6, 2, 8, 8, 1, 3, 1, 9, 8, 1, 6, 6, 7, 8, 2, 3, 6, 4, 1, 9, 2, 9, 3, 5, 2, 9, 1, 7, 8, 8, 8, 9, 8, 8, 8, 9, 1, 5, 3, 4, 6, 2, 9, 1, 9, 9, 9, 6, 9, 8, 8, 9, 8, 1, 1, 1, 9, 5, 1, 1, 2, 9, 2, 2, 9, 9, 9, 7, 9, 1, 9, 3, 9, 8, 9, 1, 8, 3, 8, 2, 1, 5, 1, 9, 8, 4, 4, 1, 9, 8, 7, 8, 9, 1, 8, 2, 1, 2, 4, 2, 7, 9, 8, 1, 2, 9, 7, 9, 9, 1, 8, 9, 2, 8, 2, 3, 7, 9, 5, 3, 1, 9, 10, 8, 9, 9, 10, 9, 8, 1, 3, 8, 9, 9, 10, 9, 10, 4, 1, 2, 1, 4, 4, 3, 9, 10, 8, 1, 2, 7, 10, 1, 9, 8, 1, 8, 1, 8, 8, 9, 1, 1, 5, 1, 1, 1, 1, 2, 9, 9, 6, 5, 1, 2, 9, 7, 9, 9, 8, 1, 3, 9, 8, 6, 12, 8, 9, 9, 1, 1, 8, 2, 8, 1, 3, 9, 7, 8, 9, 10, 2, 9, 3, 1, 1, 9, 3, 2, 7, 9, 9, 1, 2, 3, 9, 4, 3, 9, 5, 8, 9, 1, 8, 8, 1, 1, 7, 4, 9, 7, 2, 9, 4, 1, 9, 1, 9, 1, 6, 5, 9, 10, 1, 2, 9, 9, 8, 2, 10, 5, 5, 6, 3, 9, 4, 2, 4, 7, 6, 1, 1, 6, 2, 2, 7, 10, 9, 5, 6, 9, 1, 9, 4, 3, 9, 8, 9, 9, 9, 9, 10, 8, 9, 9, 10, 3, 10, 1, 1, 9, 10, 7, 8, 4, 2, 1, 10, 9, 8, 1, 2, 1, 3, 5, 9, 10, 1, 9, 9, 9, 2, 1, 10, 9, 3, 5, 1, 1, 7, 4, 6, 5, 8, 1, 7, 8, 1, 1, 10, 13, 1, 2, 1, 9, 9, 1, 5, 1, 1, 8, 10, 1, 9, 9, 9, 8, 9, 7, 9, 9, 9, 9, 8, 1, 3, 4, 9, 9, 8, 7, 9, 1, 1, 4, 5, 1, 2, 10, 7, 9, 9, 11, 7, 3, 1, 9, 1, 5, 2, 8, 3, 8, 9, 9, 1, 2, 10, 1, 9, 4, 9, 10, 1, 9, 9, 7, 10, 8, 1, 1, 10, 7, 3, 1, 9, 10, 8, 7, 9, 1, 9, 1, 8, 11, 9, 1, 2, 1, 9, 9, 8, 2, 5, 9, 1, 4, 7, 9, 8, 9, 9, 7, 10, 9, 9, 1, 9, 8, 3, 2, 6, 8, 9, 6, 2, 1, 7, 2, 6, 1, 1, 3, 9, 10, 1, 4, 5, 9, 10, 7, 9, 8, 11, 8, 10, 9, 9, 1, 2, 1, 10, 6, 6, 5, 4, 1, 8, 1, 9, 2, 9, 1, 3, 2, 1, 2, 7, 2, 3, 9, 9, 9, 9, 9, 9, 9, 9, 8, 4, 8, 9, 8, 1, 9, 9, 8, 7, 1, 1, 9, 9, 7, 2, 1, 2, 9, 8, 8, 9, 2, 7, 1, 3, 9, 9, 9, 9, 9, 9, 1, 9, 1, 3, 1, 3, 1, 7, 8, 1, 8, 1, 9, 9, 9, 3, 2, 8, 9, 4, 3, 8, 1, 3, 9, 3, 8, 1, 7, 7, 2, 6, 7, 5, 3, 3, 10, 10, 9, 1, 9, 5, 7, 9, 8, 9, 9, 9, 10, 9, 9, 10, 8, 9, 1, 9, 9, 4, 3, 1, 3, 1, 9, 1, 1, 1, 8, 9, 10, 1, 5, 2, 4, 4, 9, 5, 4, 9, 10, 1, 9, 9, 10, 9, 7, 1, 1, 4, 1, 9, 8, 1, 1, 9, 9, 9, 7, 9, 3, 8, 8, 9, 9, 9, 4, 3, 9, 9, 3, 3, 9, 9, 9, 2, 11, 4, 3, 9, 9, 8, 1, 1, 9, 9, 2, 9, 6, 10, 5, 2, 2, 9, 7, 9, 2, 4, 9, 1, 1, 8, 4, 3, 4, 6, 9, 8, 9, 10, 10, 9, 8, 2, 5, 9, 9, 2, 10, 7, 10, 2, 5, 9, 8, 3, 3, 2, 9, 9, 3, 8, 4, 2, 8, 9, 9, 9, 1, 2, 2, 9, 9, 9, 9, 4, 8, 9, 9, 9, 9, 9, 9, 11, 8, 9, 8, 4, 7, 2, 4, 4, 8, 2, 5, 8, 8, 2, 3, 9, 1, 1, 8, 4, 2, 4, 6, 2, 7, 9, 5, 2, 2, 1, 9, 10, 9, 9, 8, 6, 7, 7, 11, 1, 10, 9, 7, 4, 3, 10, 9, 4, 9, 8, 9, 5, 1, 7, 9, 9, 1, 8, 9, 5, 5, 1, 9, 9, 1, 9, 7, 8, 9, 2, 9, 9, 8, 1, 8, 1, 2, 9, 1, 2, 9, 3, 9, 3, 2, 1, 2, 9, 9, 7, 8, 1, 8, 10, 3, 3, 2, 9, 9, 8, 7, 8, 1, 8, 9, 2, 8, 2, 9, 9, 1, 9, 9, 3, 9, 9, 10, 8, 1, 8, 9, 9, 9, 8, 9, 9, 7, 10, 1, 3, 8, 8, 1, 9, 10, 7, 3, 2, 7, 8, 4, 7, 9, 2, 1, 1, 10, 8, 1, 3, 1, 8, 2, 2, 1, 9, 1, 4, 4, 2, 7, 1, 3, 2, 9, 10, 10, 9, 9, 10, 9, 9, 8, 8, 9, 13, 3, 8, 9, 10, 9, 2, 9, 3, 4, 2, 7, 4, 3, 8, 9, 8, 9, 8, 1, 10, 9, 7, 3, 2, 16, 9, 9, 10, 9, 9, 9, 5, 9, 8, 7, 2, 1, 7, 7, 1, 9, 8, 2, 2, 7, 4, 8, 2, 9, 3, 4, 2, 5, 1, 12, 5, 2, 9, 4, 3, 2, 9, 3, 2, 12, 2, 5, 1, 9, 9, 10, 9, 10, 10, 9, 8, 9, 1, 2, 1, 1, 8, 10, 8, 7, 1, 1, 8, 9, 4, 1, 1, 7, 1, 10, 1, 9, 10, 4, 8, 9, 9, 8, 1, 2, 2, 9, 9, 8, 9, 9, 1, 3, 8, 9, 1, 3, 8, 9, 1, 5, 7, 9, 8, 8, 1, 8, 1, 2, 8, 9, 9, 9, 4, 8, 1, 1, 9, 8, 9, 10, 8, 1, 2, 3, 9, 1, 8, 1, 2, 3, 7, 10, 9, 1, 11, 9, 15, 9, 6, 3, 8, 9, 9, 3, 1, 1, 8, 3, 5, 1, 3, 9, 8, 9, 8, 1, 5, 1, 4, 9, 4, 4, 8, 2, 3, 1, 1, 1, 9, 9, 2, 9, 1, 9, 9, 9, 9, 2, 1, 7, 9, 3, 1, 1, 10, 9, 9, 1, 2, 9, 7, 3, 9, 9, 9, 1, 2, 2, 9, 3, 1, 9, 8, 9, 8, 1, 1, 9, 3, 8, 1, 8, 3, 3, 2, 1, 1, 1, 9, 9, 7, 9, 3, 1, 2, 4, 2, 2, 9, 2, 3, 1, 8, 1, 3, 2, 8, 1, 8, 7, 9, 2, 1, 9, 4, 4, 1, 9, 3, 9, 10, 3, 10, 1, 9, 10, 3, 9, 9, 7, 9, 9, 8, 1, 1, 9, 9, 8, 9, 10, 9, 8, 9, 9, 7, 9, 9, 7, 9, 9, 9, 8, 2, 9, 8, 2, 3, 9, 4, 4, 9, 2, 9, 8, 1, 8, 5, 8, 10, 2, 9, 9, 9, 7, 16, 9, 5, 2, 8, 7, 8, 2, 7, 2, 4, 3, 1, 9, 9, 1, 8, 9, 2, 12, 9, 9, 10, 1, 11, 8, 10, 9, 9, 3, 9, 2, 9, 9, 16, 1, 9, 9, 6, 7, 4, 4, 5, 8, 9, 8, 9, 1, 5, 7, 9, 1, 10, 9, 9, 8, 2, 4, 2, 9, 1, 7, 2, 12, 9, 9, 1, 1, 1, 5, 9, 4, 2, 2, 1, 2, 9, 2, 7, 9, 8, 2, 9, 1, 9, 9, 9, 9, 3, 4, 5, 4, 8, 9, 3, 2, 9, 2, 4, 1, 1, 3, 7, 9, 1, 8, 9, 1, 8, 9, 1, 1, 3, 8, 5, 8, 9, 8, 6, 2, 7, 5, 8, 8, 2, 2, 9, 6, 1, 2, 6, 8, 9, 6, 1, 9, 9, 7, 1, 1, 4, 10, 11, 3, 1, 1, 9, 9, 9, 9, 8, 5, 1, 4, 2, 7, 1, 8, 4, 9, 3, 3, 1, 9, 9, 7, 1, 1, 3, 2, 8, 1, 9, 1, 9, 6, 8, 1, 8, 8, 4, 4, 9, 9, 1, 1, 9, 1, 8, 9, 8, 3, 4, 1, 1, 2, 9, 9, 9, 3, 4, 10, 7, 7, 8, 5, 8, 9, 9, 9, 9, 9, 9, 10, 9, 1, 1, 3, 1, 5, 9, 10, 2, 3, 9, 7, 9, 10, 9, 8, 2, 9, 1, 1, 1, 2, 9, 9, 5, 1, 1, 5, 3, 9, 11, 9, 2, 9, 1, 4, 7, 9, 3, 3, 6, 8, 7, 4, 9, 9, 6, 9, 4, 3, 5, 9, 9, 3, 2, 9, 4, 1, 9, 7, 2, 9, 9, 8, 9, 9, 5, 9, 10, 9, 7, 8, 1, 2, 1, 9, 9, 1, 10, 5, 10, 9, 9, 6, 4, 10, 9, 7, 8, 8, 2, 1, 1, 8, 2, 2, 1, 1, 8, 8, 1, 2, 8, 7, 4, 5, 9, 9, 9, 4, 9, 9, 9, 9, 2, 10, 3, 9, 9, 5, 6, 4, 9, 23, 8, 9, 1, 9, 7, 9, 2, 9, 9, 8, 1, 9, 2, 2, 1, 9, 9, 9, 11, 13, 2, 9, 9, 8, 1, 4, 3, 8, 9, 4, 4, 7, 1, 2, 8, 9, 8, 9, 1, 1, 19, 9, 9, 9, 9, 8, 2, 1, 4, 6, 9, 9, 2, 4, 10, 10, 9, 8, 9, 9, 9, 8, 7, 9, 9, 9, 9, 8, 10, 9, 9, 9, 9, 9, 1, 5, 7, 1, 7, 1, 3, 1, 8, 2, 9, 2, 1, 7, 8, 9, 1, 5, 1, 8, 9, 9, 7, 7, 9, 9, 9, 9, 9, 9, 9, 9, 7, 9, 1, 1, 1, 8, 7, 9, 9, 1, 2, 7, 10, 9, 6, 9, 9, 1, 9, 5, 2, 8, 9, 1, 1, 10, 10, 1, 4, 3, 5, 1, 9, 1, 9, 1, 5, 3, 8, 2, 8, 1, 2, 7, 2, 9, 9, 1, 10, 8, 9, 9, 9, 9, 9, 9, 9, 11, 9, 9, 8, 8, 9, 10, 9, 9, 9, 8, 9, 1, 9, 3, 9, 9, 9, 9, 8, 9, 2, 8, 9, 9, 9, 9, 9, 9, 1, 9, 2, 1, 10, 10, 9, 9, 9, 10, 7, 9, 8, 1, 8, 12, 7, 9, 9, 9, 8, 3, 2, 8, 9, 9, 1, 4, 8, 8, 8, 2, 9, 2, 1, 1, 1, 1, 5, 7, 1, 6, 9, 8, 7, 3, 8, 9, 9, 1, 8, 9, 9, 1, 9, 2, 8, 9, 9, 2, 9, 7, 11, 9, 9, 1, 9, 8, 11, 1, 4, 6, 9, 4, 2, 9, 9, 1, 7, 9, 4, 4, 7, 9, 8, 10, 10, 7, 1, 1, 8, 10, 1, 7, 9, 4, 3, 9, 10, 9, 9, 7, 3, 6, 9, 8, 1, 11, 1, 5, 3, 1, 8, 8, 9, 1, 2, 8, 8, 10, 3, 9, 7, 6, 9, 2, 7, 1, 7, 8, 8, 1, 8, 9, 1, 10, 9, 8, 9, 7, 8, 9, 8, 7, 1, 9, 1, 4, 10, 2, 3, 9, 2, 9, 8, 4, 3, 9, 1, 2, 9, 2, 3, 2, 1, 2, 4, 9, 8, 6, 7, 4, 1, 2, 2, 2, 1, 9, 2, 2, 1, 1, 2, 9, 5, 9, 9, 8, 1, 1, 9, 8, 1, 8, 4, 3, 8, 11, 7, 9, 1, 8, 9, 9, 1, 1, 1, 8, 8, 9, 8, 3, 5, 7, 9, 1, 1, 4, 1, 2, 12, 3, 4, 9, 2, 1, 9, 11, 3, 3, 1, 2, 7, 12, 1, 1, 7, 10, 8, 9, 1, 9, 9, 3, 7, 8, 7, 2, 9, 1, 3, 10, 9, 7, 9, 7, 6, 10, 8, 8, 2, 4, 1, 3, 7, 1, 9, 6, 1, 8, 9, 10, 8, 8, 1, 9, 8, 9, 9, 2, 3, 2, 1, 3, 9, 8, 9, 8, 9, 1, 9, 9, 10, 5, 7, 9, 9, 9, 1, 6, 9, 8, 9, 4, 3, 9, 1, 3, 9, 9, 9, 9, 7, 9, 8, 9, 9, 9, 9, 1, 9, 8, 6, 4, 3, 2, 4, 9, 2, 1, 1, 1, 1, 9, 9, 9, 11, 2, 4, 1, 7, 9, 1, 3, 9, 8, 1, 8, 9, 8, 9, 10, 10, 1, 4, 2, 8, 5, 9, 9, 9, 9, 6, 9, 5, 1, 8, 4, 9, 4, 4, 10, 7, 4, 7, 4, 2, 1, 8, 1, 8, 7, 1, 3, 1, 7, 10, 1, 9, 8, 1, 7, 4, 9, 4, 1, 5, 9, 9, 9, 9, 1, 8, 6, 3, 10, 1, 4, 3, 9, 9, 8, 1, 9, 8, 7, 9, 1, 9, 9, 9, 8, 9, 13, 9, 9, 1, 9, 1, 6, 9, 7, 9, 1, 2, 2, 9, 7, 9, 9, 8, 7, 8, 8, 2, 2, 9, 8, 7, 2, 1, 9, 10, 10, 9, 10, 9, 1, 9, 1]
long_speed_data = [6.36259991e-02, 1.50005381e-02, 2.40687699e-02, 2.56818302e-02, 4.35050000e+04]

39393
high_steps_data = [7, 2, 3, 1, 3, 1, 8, 1, 1, 8, 8, 2, 4, 2, 1, 2, 8, 8, 1, 8, 10, 8, 8, 4, 3, 9, 8, 8, 7, 2, 10, 7, 1, 2, 1, 3, 8, 3, 2, 7, 8, 1, 2, 9, 11, 1, 2, 1, 1, 4, 1, 9, 9, 8, 1, 9, 1, 24, 8, 7, 4, 1, 11, 8, 1, 1, 9, 8, 4, 3, 3, 5, 8, 1, 1, 2, 1, 7, 4, 1, 8, 7, 11, 8, 9, 3, 3, 9, 8, 3, 1, 2, 2, 1, 3, 8, 1, 1, 1, 10, 3, 1, 1, 1, 1, 8, 3, 2, 1, 8, 7, 1, 2, 8, 2, 8, 8, 1, 7, 8, 8, 8, 1, 5, 5, 8, 8, 1, 8, 5, 7, 1, 6, 8, 4, 7, 10, 4, 3, 9, 8, 3, 11, 8, 8, 7, 1, 2, 8, 1, 1, 2, 2, 8, 9, 1, 8, 1, 1, 3, 2, 9, 7, 2, 1, 1, 1, 10, 1, 1, 5, 1, 3, 3, 1, 8, 8, 2, 2, 1, 3, 14, 3, 1, 2, 8, 7, 8, 1, 10, 2, 4, 2, 8, 8, 9, 1, 5, 3, 4, 9, 8, 1, 1, 3, 8, 3, 8, 3, 8, 1, 1, 3, 8, 2, 1, 4, 5, 8, 7, 7, 2, 1, 1, 1, 1, 9, 8, 7, 2, 11, 1, 8, 3, 1, 8, 8, 4, 8, 1, 2, 7, 7, 1, 7, 9, 3, 1, 7, 7, 9, 8, 9, 10, 3, 1, 3, 1, 8, 1, 7, 1, 2, 3, 1, 24, 8, 3, 7, 2, 1, 9, 1, 8, 1, 8, 6, 9, 4, 7, 4, 1, 1, 4, 2, 3, 4, 8, 8, 1, 8, 6, 2, 8, 8, 9, 4, 24, 8, 8, 8, 1, 8, 9, 1, 8, 8, 7, 2, 4, 2, 8, 8, 3, 3, 8, 3, 10, 2, 1, 8, 8, 3, 2, 7, 7, 8, 8, 8, 2, 2, 8, 2, 8, 9, 7, 10, 1, 8, 8, 3, 1, 3, 2, 8, 5, 1, 5, 3, 4, 3, 5, 4, 4, 2, 1, 1, 1, 1, 8, 8, 8, 7, 8, 8, 2, 9, 1, 1, 7, 7, 8, 3, 6, 1, 1, 7, 1, 1, 1, 1, 2, 4, 7, 1, 9, 4, 1, 7, 11, 1, 10, 7, 7, 11, 3, 3, 8, 1, 4, 5, 8, 1, 7, 6, 9, 1, 8, 8, 8, 8, 9, 1, 1, 1, 8, 1, 3, 2, 1, 1, 1, 8, 8, 9, 1, 1, 8, 1, 8, 8, 1, 3, 2, 4, 8, 8, 1, 5, 2, 6, 2, 1, 1, 2, 5, 2, 5, 10, 8, 1, 9, 11, 2, 1, 8, 1, 14, 8, 10, 2, 2, 5, 8, 1, 2, 2, 2, 2, 2, 2, 2, 1, 10, 1, 1, 7, 7, 10, 1, 2, 2, 9, 1, 10, 4, 7, 7, 8, 2, 10, 7, 5, 7, 4, 8, 8, 2, 1, 8, 1, 8, 8, 3, 3, 8, 2, 1, 9, 3, 4, 7, 6, 8, 3, 3, 4, 7, 7, 1, 8, 8, 8, 2, 1, 8, 8, 8, 8, 7, 3, 1, 1, 8, 3, 2, 2, 2, 10, 3, 1, 8, 3, 8, 1, 1, 1, 4, 7, 1, 2, 2, 8, 9, 3, 2, 9, 5, 24, 7, 7, 5, 8, 1, 1, 8, 1, 8, 3, 9, 1, 3, 9, 1, 8, 5, 2, 2, 1, 7, 1, 8, 2, 10, 3, 8, 5, 9, 1, 1, 1, 2, 2, 24, 1, 24, 1, 7, 2, 1, 9, 1, 3, 2, 2, 1, 11, 4, 9, 1, 1, 3, 3, 3, 9, 8, 10, 1, 1, 9, 1, 1, 9, 4, 8, 8, 8, 3, 1, 4, 8, 1, 9, 1, 1, 3, 3, 1, 4, 1, 2, 24, 2, 1, 8, 8, 8, 7, 2, 5, 10, 9, 8, 1, 3, 1, 1, 4, 2, 2, 9, 7, 7, 8, 2, 1, 8, 9, 8, 9, 8, 1, 3, 3, 1, 7, 1, 1, 1, 1, 10, 2, 1, 4, 10, 2, 8, 9, 8, 8, 5, 1, 8, 8, 17, 7, 1, 1, 1, 1, 4, 2, 1, 1, 8, 2, 7, 1, 5, 1, 4, 3, 1, 10, 7, 10, 2, 3, 5, 8, 1, 9, 4, 2, 4, 7, 1, 2, 3, 1, 6, 3, 3, 4, 2, 9, 8, 2, 1, 8, 4, 3, 8, 1, 2, 7, 1, 1, 4, 8, 8, 10, 1, 6, 7, 7, 8, 5, 10, 2, 8, 1, 7, 3, 2, 8, 1, 8, 1, 3, 1, 8, 1, 9, 8, 8, 1, 1, 24, 3, 8, 7, 7, 2, 9, 1, 1, 7, 1, 2, 2, 8, 8, 8, 1, 4, 9, 8, 2, 2, 2, 1, 2, 8, 3, 9, 2, 2, 8, 8, 1, 9, 1, 7, 8, 1, 8, 2, 1, 1, 1, 8, 8, 2, 8, 9, 1, 2, 2, 3, 1, 4, 8, 4, 2, 7, 1, 3, 2, 1, 2, 2, 1, 7, 8, 5, 7, 1, 1, 7, 7, 3, 8, 10, 2, 4, 9, 7, 8, 1, 3, 1, 7, 10, 8, 3, 3, 7, 1, 4, 1, 8, 1, 1, 4, 3, 3, 2, 12, 2, 7, 7, 8, 4, 1, 4, 1, 8, 1, 3, 3, 2, 7, 9, 2, 8, 1, 7, 9, 8, 8, 9, 8, 5, 1, 3, 1, 2, 1, 3, 9, 1, 1, 8, 1, 9, 1, 3, 2, 24, 8, 8, 1, 2, 1, 3, 3, 1, 8, 8, 1, 1, 3, 24, 8, 1, 8, 2, 9, 2, 1, 1, 1, 1, 8, 9, 1, 9, 2, 1, 4, 1, 1, 9, 1, 6, 8, 2, 1, 2, 1, 8, 2, 2, 4, 1, 7, 5, 1, 9, 2, 3, 7, 1, 2, 2, 4, 3, 1, 2, 1, 8, 2, 1, 1, 1, 1, 1, 11, 2, 1, 8, 8, 7, 1, 1, 3, 1, 2, 1, 9, 8, 7, 2, 8, 3, 2, 2, 1, 1, 7, 8, 4, 4, 1, 9, 1, 8, 9, 1, 9, 6, 6, 6, 4, 8, 8, 9, 8, 8, 2, 1, 1, 1, 7, 8, 1, 8, 9, 1, 5, 1, 1, 8, 9, 8, 6, 2, 1, 8, 1, 1, 7, 7, 1, 9, 8, 9, 1, 1, 1, 1, 1, 6, 1, 8, 9, 1, 7, 1, 9, 2, 1, 8, 9, 8, 8, 8, 1, 2, 4, 1, 9, 2, 1, 8, 8, 8, 2, 8, 1, 24, 2, 2, 6, 2, 1, 1, 8, 7, 8, 8, 3, 7, 4, 4, 3, 4, 9, 10, 5, 2, 9, 1, 2, 1, 8, 1, 1, 1, 10, 9, 1, 3, 8, 7, 1, 4, 9, 7, 2, 6, 2, 1, 1, 10, 8, 7, 7, 8, 7, 1, 3, 8, 8, 1, 2, 1, 2, 1, 6, 1, 10, 8, 8, 1, 1, 24, 1, 10, 8, 9, 3, 4, 9, 9, 8, 8, 1, 8, 7, 2, 8, 8, 1, 1, 1, 1, 2, 2, 1, 9, 8, 7, 8, 9, 7, 2, 1, 1, 1, 2, 7, 1, 1, 8, 1, 11, 1, 1, 1, 7, 4, 5, 1, 1, 7, 1, 8, 9, 2, 3, 3, 7, 8, 1, 7, 8, 4, 3, 2, 1, 2, 2, 6, 8, 8, 1, 1, 1, 7, 1, 3, 1, 3, 2, 1, 3, 1, 9, 1, 12, 3, 8, 1, 5, 4, 3, 7, 9, 2, 7, 2, 1, 6, 10, 1, 8, 1, 1, 1, 11, 5, 13, 8, 2, 1, 24, 8, 9, 6, 1, 4, 4, 8, 1, 8, 5, 9, 3, 2, 9, 8, 1, 7, 2, 8, 8, 1, 19, 7, 7, 7, 7, 8, 3, 1, 8, 5, 7, 1, 2, 1, 8, 11, 9, 24, 2, 9, 2, 1, 1, 3, 9, 4, 3, 2, 1, 1, 1, 7, 3, 6, 7, 2, 3, 4, 2, 2, 3, 4, 11, 1, 5, 1, 4, 7, 5, 1, 8, 1, 6, 1, 2, 1, 3, 2, 3, 1, 1, 4, 2, 7, 9, 3, 5, 1, 2, 1, 3, 4, 8, 8, 12, 7, 7, 9, 1, 1, 2, 1, 2, 2, 5, 4, 8, 1, 24, 8, 1, 3, 3, 9, 8, 8, 3, 1, 9, 8, 8, 1, 1, 3, 2, 3, 9, 7, 9, 3, 9, 1, 10, 9, 2, 1, 1, 1, 3, 1, 1, 2, 7, 1, 4, 6, 3, 3, 7, 3, 1, 3, 5, 1, 10, 1, 10, 3, 5, 9, 7, 1, 6, 9, 1, 2, 7, 3, 4, 8, 1, 8, 24, 2, 9, 1, 8, 8, 1, 1, 8, 10, 3, 8, 7, 1, 2, 1, 3, 10, 8, 1, 1, 10, 1, 1, 4, 10, 1, 2, 1, 1, 1, 5, 3, 8, 3, 2, 2, 1, 8, 7, 2, 3, 4, 1, 6, 8, 1, 3, 8, 8, 9, 6, 8, 1, 1, 10, 1, 1, 2, 8, 8, 8, 1, 8, 3, 2, 7, 1, 3, 2, 1, 7, 1, 2, 7, 1, 8, 8, 8, 3, 4, 2, 2, 1, 8, 1, 1, 1, 10, 8, 9, 9, 10, 8, 4, 1, 1, 8, 1, 7, 8, 2, 1, 1, 1, 8, 8, 3, 3, 2, 1, 7, 7, 1, 2, 9, 5, 5, 8, 7, 9, 8, 1, 7, 8, 4, 9, 1, 1, 2, 2, 1, 2, 6, 1, 24, 8, 1, 2, 2, 11, 2, 4, 2, 1, 8, 2, 4, 1, 4, 1, 5, 4, 7, 8, 3, 7, 8, 9, 1, 1, 7, 1, 3, 2, 1, 2, 7, 1, 2, 3, 7, 3, 1, 2, 4, 8, 3, 22, 1, 5, 6, 3, 1, 8, 8, 9, 9, 7, 2, 3, 8, 8, 4, 1, 7, 7, 8, 9, 3, 1, 9, 5, 2, 3, 1, 2, 9, 7, 2, 4, 8, 10, 8, 7, 8, 8, 8, 8, 8, 7, 1, 11, 1, 3, 8, 8, 7, 1, 1, 7, 8, 24, 8, 1, 7, 8, 1, 3, 2, 2, 9, 7, 1, 1, 1, 10, 9, 2, 1, 4, 7, 4, 1, 7, 7, 1, 8, 3, 3, 9, 3, 2, 9, 3, 2, 2, 8, 3, 2, 1, 9, 9, 7, 6, 9, 8, 1, 8, 1, 8, 1, 1, 1, 7, 1, 1, 9, 7, 9, 24, 8, 2, 1, 1, 3, 1, 2, 8, 1, 10, 7, 2, 10, 8, 2, 2, 1, 8, 3, 3, 6, 1, 8, 9, 9, 9, 2, 1, 3, 1, 5, 2, 2, 1, 6, 4, 7, 2, 7, 2, 3, 1, 8, 1, 1, 11, 9, 3, 1, 9, 1, 9, 1, 2, 2, 1, 5, 4, 4, 4, 7, 1, 5, 3, 3, 1, 3, 3, 4, 8, 2, 8, 8, 7, 8, 5, 24, 8, 1, 7, 8, 10, 2, 2, 11, 1, 10, 8, 6, 9, 8, 6, 2, 7, 1, 11, 8, 3, 1, 4, 2, 2, 7, 7, 9, 5, 9, 1, 9, 8, 7, 8, 7, 8, 8, 9, 9, 3, 1, 2, 3, 7, 8, 3, 9, 8, 14, 8, 7, 1, 1, 1, 1, 1, 9, 7, 2, 1, 2, 7, 1, 9, 7, 3, 3, 6, 2, 4, 1, 7, 8, 7, 2, 11, 2, 9, 8, 4, 1, 8, 24, 4, 6, 1, 8, 8, 10, 1, 12, 7, 6, 8, 8, 1, 2, 1, 1, 8, 1, 2, 7, 1, 7, 3, 2, 9, 1, 10, 8, 1, 1, 24, 8, 8, 1, 3, 1, 1, 8, 8, 2, 2, 3, 1, 8, 1, 1, 7, 8, 7, 9, 1, 2, 8, 8, 8, 7, 1, 1, 4, 3, 1, 1, 2, 8, 8, 2, 8, 5, 10, 1, 1, 2, 1, 2, 24, 2, 4, 1, 1, 2, 5, 1, 9, 7, 9, 5, 8, 1, 3, 10, 1, 2, 9, 3, 8, 8, 8, 11, 1, 8, 7, 1, 9, 6, 6, 2, 4, 3, 1, 8, 8, 1, 8, 8, 7, 8, 8, 4, 8, 9, 1, 1, 1, 8, 4, 2, 9, 8, 8, 8, 9, 2, 7, 24, 1, 3, 3, 1, 1, 9, 8, 10, 7, 3, 3, 1, 9, 8, 9, 8, 8, 7, 2, 1, 4, 4, 1, 24, 7, 1, 2, 8, 1, 8, 7, 5, 3, 9, 3, 5, 8, 8, 1, 1, 1, 7, 9, 1, 7, 1, 8, 2, 8, 1, 1, 5, 8, 1, 8, 8, 3, 7, 2, 11, 8, 1, 1, 8, 24, 1, 2, 2, 2, 3, 1, 8, 8, 6, 7, 1, 1, 7, 10, 8, 2, 9, 1, 2, 8, 8, 1, 3, 2, 2, 2, 4, 4, 1, 8, 3, 1, 9, 4, 3, 7, 8, 1, 10, 4, 5, 8, 7, 6, 3, 2, 4, 5, 7, 3, 3, 2, 8, 1, 3, 4, 3, 1, 2, 2, 1, 7, 1, 8, 1, 1, 7, 8, 1, 9, 7, 3, 1, 7, 8, 2, 6, 8, 2, 8, 1, 1, 3, 4, 7, 2, 7, 2, 7, 1, 1, 1, 8, 1, 4, 1, 3, 1, 7, 1, 5, 2, 5, 3, 1, 1, 1, 5, 2, 3, 8, 8, 8, 8, 1, 8, 1, 4, 3, 2, 1, 1, 3, 7, 9, 7, 8, 3, 1, 1, 2, 2, 9, 1, 2, 7, 1, 8, 1, 1, 7, 1, 6, 6, 9, 1, 8, 2, 1, 1, 8, 24, 9, 1, 7, 2, 8, 2, 1, 8, 6, 1, 10, 9, 1, 5, 8, 3, 1, 1, 4, 2, 2, 7, 5, 2, 7, 2, 8, 1, 3, 2, 2, 1, 1, 7, 1, 2, 3, 3, 1, 2, 7, 8, 1, 1, 1, 8, 9, 8, 1, 7, 1, 6, 1, 7, 1, 1, 2, 8, 2, 2, 1, 9, 1, 6, 9, 5, 7, 6, 7, 3, 9, 1, 6, 4, 3, 7, 1, 8, 1, 3, 8, 8, 1, 3, 24, 2, 5, 2, 3, 3, 1, 8, 2, 3, 4, 1, 6, 8, 7, 8, 2, 8, 8, 8, 2, 2, 3, 1, 1, 2, 3, 5, 2, 1, 2, 4, 3, 2, 8, 9, 8, 10, 1, 8, 1, 24, 8, 7, 2, 1, 4, 1, 7, 11, 8, 5, 1, 8, 24, 8, 3]
high_speed_data = [7.67913801e-02, 1.50002051e-02, 2.30151514e-02, 2.55020106e-02, 3.93930000e+04]

46059
short_steps_data = [1, 1, 9, 15, 2, 1, 3, 8, 3, 6, 2, 2, 4, 8, 11, 9, 6, 5, 1, 9, 2, 5, 9, 9, 9, 9, 7, 9, 2, 7, 9, 8, 9, 9, 9, 9, 1, 1, 9, 9, 7, 1, 9, 1, 9, 2, 1, 6, 1, 8, 9, 1, 9, 9, 9, 9, 9, 7, 1, 9, 9, 9, 9, 9, 8, 9, 9, 9, 9, 9, 9, 7, 6, 4, 4, 9, 1, 8, 42, 3, 1, 9, 9, 1, 7, 2, 1, 1, 7, 8, 9, 9, 9, 3, 7, 11, 3, 9, 9, 7, 9, 10, 9, 8, 1, 1, 1, 3, 7, 7, 2, 3, 8, 8, 8, 6, 7, 9, 4, 3, 2, 3, 8, 2, 1, 10, 9, 9, 1, 9, 1, 3, 8, 8, 8, 9, 9, 9, 10, 9, 7, 7, 11, 1, 2, 9, 9, 8, 7, 2, 9, 1, 7, 7, 1, 9, 10, 2, 3, 1, 9, 2, 1, 2, 3, 1, 6, 1, 3, 2, 10, 10, 9, 6, 1, 4, 5, 2, 3, 9, 4, 4, 9, 1, 7, 5, 3, 9, 7, 9, 9, 9, 9, 9, 7, 8, 8, 9, 2, 2, 1, 9, 9, 8, 8, 8, 9, 7, 9, 3, 6, 9, 5, 9, 4, 8, 9, 1, 8, 9, 1, 2, 8, 9, 4, 1, 1, 3, 1, 1, 8, 3, 3, 9, 1, 8, 9, 8, 1, 2, 3, 1, 8, 7, 1, 1, 9, 9, 9, 9, 1, 1, 9, 8, 9, 9, 7, 10, 9, 9, 1, 9, 8, 9, 10, 9, 2, 9, 9, 7, 8, 9, 9, 9, 9, 9, 9, 9, 9, 11, 4, 1, 2, 1, 3, 4, 4, 9, 6, 9, 7, 9, 9, 9, 9, 9, 7, 10, 9, 1, 8, 1, 9, 9, 8, 9, 9, 9, 1, 8, 9, 9, 8, 7, 1, 9, 9, 9, 1, 1, 2, 9, 2, 9, 13, 9, 8, 2, 10, 12, 9, 7, 1, 2, 9, 9, 9, 10, 9, 2, 1, 6, 1, 12, 2, 8, 6, 9, 9, 1, 9, 9, 11, 1, 1, 8, 8, 4, 9, 9, 10, 9, 9, 7, 1, 8, 1, 2, 5, 7, 9, 8, 9, 9, 9, 1, 8, 9, 9, 9, 1, 7, 9, 8, 8, 9, 13, 9, 1, 3, 9, 9, 8, 9, 3, 3, 8, 7, 9, 2, 1, 9, 9, 2, 8, 9, 9, 9, 8, 9, 8, 8, 7, 9, 5, 3, 9, 9, 2, 10, 9, 9, 8, 9, 9, 5, 9, 9, 9, 9, 9, 9, 4, 2, 8, 9, 5, 9, 9, 9, 9, 9, 1, 1, 2, 7, 1, 2, 1, 9, 5, 9, 2, 6, 10, 8, 2, 9, 9, 8, 1, 1, 1, 6, 1, 4, 2, 8, 2, 3, 6, 9, 3, 9, 9, 9, 9, 10, 6, 7, 1, 9, 1, 5, 2, 3, 4, 9, 9, 4, 3, 9, 9, 9, 8, 5, 8, 1, 8, 9, 9, 1, 10, 9, 9, 9, 9, 12, 1, 9, 9, 9, 7, 1, 1, 1, 7, 8, 7, 9, 3, 2, 3, 9, 7, 9, 9, 9, 10, 7, 1, 9, 9, 10, 7, 8, 4, 1, 2, 1, 2, 9, 7, 9, 9, 9, 2, 4, 1, 4, 5, 4, 1, 1, 2, 9, 9, 7, 9, 9, 1, 6, 1, 5, 2, 7, 7, 10, 8, 8, 9, 9, 1, 3, 9, 7, 1, 7, 8, 8, 9, 6, 11, 8, 9, 8, 9, 9, 9, 8, 9, 4, 7, 9, 7, 5, 9, 2, 1, 9, 1, 3, 3, 9, 3, 7, 1, 1, 8, 1, 4, 1, 2, 1, 12, 8, 9, 9, 10, 9, 8, 9, 9, 9, 7, 9, 2, 6, 12, 9, 9, 9, 2, 9, 2, 9, 8, 2, 1, 8, 1, 1, 7, 8, 9, 2, 10, 7, 1, 3, 9, 9, 9, 7, 9, 9, 9, 9, 9, 5, 1, 8, 8, 5, 1, 1, 1, 1, 7, 8, 9, 9, 8, 9, 9, 3, 3, 4, 8, 9, 8, 9, 6, 9, 9, 9, 8, 6, 1, 8, 1, 7, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1, 2, 7, 9, 9, 9, 1, 9, 9, 9, 2, 7, 6, 9, 7, 9, 8, 9, 8, 2, 6, 9, 9, 8, 8, 8, 9, 10, 9, 9, 9, 13, 4, 9, 9, 9, 8, 7, 9, 9, 9, 1, 1, 1, 10, 9, 9, 9, 8, 9, 9, 8, 1, 10, 1, 7, 7, 9, 9, 2, 1, 9, 1, 9, 9, 8, 6, 9, 4, 1, 9, 9, 9, 9, 9, 2, 1, 1, 8, 10, 5, 1, 8, 1, 4, 1, 1, 8, 3, 1, 9, 9, 9, 7, 9, 4, 7, 1, 9, 9, 1, 8, 9, 9, 1, 9, 6, 4, 1, 8, 6, 1, 2, 9, 1, 8, 8, 12, 4, 7, 3, 4, 1, 1, 8, 8, 9, 9, 9, 6, 9, 1, 9, 9, 8, 9, 8, 9, 9, 9, 3, 4, 7, 2, 2, 5, 9, 1, 1, 8, 4, 1, 8, 8, 9, 9, 9, 9, 9, 9, 2, 1, 9, 9, 9, 9, 8, 7, 9, 9, 9, 3, 9, 9, 9, 3, 1, 3, 1, 1, 4, 9, 9, 2, 8, 8, 9, 8, 1, 9, 8, 10, 9, 6, 9, 9, 9, 4, 9, 4, 3, 5, 4, 1, 10, 4, 1, 2, 8, 1, 2, 9, 9, 9, 4, 1, 5, 9, 6, 7, 9, 9, 9, 9, 9, 9, 9, 2, 8, 8, 9, 8, 9, 9, 2, 3, 1, 9, 9, 9, 1, 8, 3, 8, 2, 7, 9, 9, 9, 9, 9, 9, 4, 9, 9, 4, 1, 9, 9, 9, 3, 8, 7, 9, 9, 1, 9, 9, 10, 9, 9, 9, 2, 8, 1, 1, 1, 5, 5, 9, 2, 2, 9, 9, 9, 8, 9, 9, 9, 9, 9, 1, 9, 7, 1, 7, 2, 1, 3, 9, 1, 8, 9, 9, 5, 9, 9, 9, 9, 9, 9, 1, 1, 9, 9, 10, 9, 8, 8, 10, 9, 9, 1, 1, 1, 1, 2, 9, 5, 1, 10, 9, 9, 9, 6, 9, 4, 9, 4, 1, 1, 1, 1, 10, 6, 9, 9, 9, 9, 10, 1, 1, 1, 1, 9, 9, 9, 8, 7, 4, 8, 9, 9, 12, 9, 9, 9, 2, 9, 9, 10, 9, 9, 9, 1, 1, 2, 2, 3, 2, 9, 7, 11, 8, 1, 1, 2, 2, 6, 6, 7, 4, 3, 9, 9, 1, 1, 3, 3, 8, 8, 6, 1, 3, 8, 8, 1, 1, 6, 9, 4, 9, 9, 8, 9, 9, 9, 9, 1, 9, 9, 8, 9, 9, 10, 9, 9, 1, 9, 8, 9, 1, 2, 2, 5, 9, 9, 9, 9, 15, 8, 1, 9, 8, 9, 9, 9, 9, 9, 9, 9, 1, 4, 8, 8, 9, 3, 1, 1, 3, 1, 5, 4, 5, 1, 7, 9, 6, 1, 9, 7, 5, 2, 8, 9, 9, 36, 7, 9, 7, 8, 2, 6, 8, 13, 7, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 9, 9, 9, 9, 6, 7, 5, 1, 6, 2, 9, 9, 9, 1, 8, 8, 8, 10, 1, 9, 9, 9, 9, 9, 5, 2, 3, 2, 1, 7, 9, 1, 1, 8, 9, 9, 8, 9, 7, 10, 9, 9, 9, 9, 9, 9, 9, 3, 7, 8, 8, 1, 3, 3, 6, 9, 4, 1, 1, 9, 9, 7, 9, 8, 1, 2, 1, 5, 2, 5, 8, 8, 9, 9, 9, 9, 8, 2, 3, 9, 9, 7, 5, 8, 3, 2, 1, 4, 9, 9, 8, 1, 7, 9, 6, 2, 1, 8, 7, 4, 1, 9, 10, 1, 7, 10, 9, 8, 10, 9, 1, 4, 9, 9, 9, 8, 1, 7, 3, 3, 9, 9, 9, 1, 1, 11, 9, 9, 8, 5, 6, 8, 1, 5, 9, 1, 9, 5, 2, 9, 1, 9, 9, 9, 9, 9, 1, 6, 9, 9, 1, 5, 1, 2, 9, 7, 1, 2, 8, 10, 1, 1, 9, 9, 9, 2, 2, 9, 8, 9, 3, 2, 2, 6, 9, 12, 9, 1, 21, 7, 9, 1, 2, 8, 8, 5, 1, 8, 9, 9, 3, 7, 7, 2, 7, 9, 1, 9, 10, 9, 9, 4, 10, 2, 1, 6, 8, 9, 2, 8, 9, 9, 8, 9, 9, 9, 9, 7, 1, 10, 9, 9, 9, 7, 1, 1, 8, 9, 9, 9, 9, 9, 9, 1, 9, 15, 7, 8, 3, 9, 2, 1, 8, 5, 8, 11, 9, 9, 9, 1, 1, 1, 2, 7, 6, 9, 9, 8, 8, 8, 9, 9, 11, 1, 7, 6, 9, 9, 9, 7, 9, 9, 1, 1, 1, 8, 9, 9, 9, 7, 1, 9, 2, 9, 9, 9, 9, 9, 9, 1, 2, 1, 10, 9, 9, 1, 1, 9, 9, 9, 1, 9, 9, 8, 9, 9, 9, 9, 2, 1, 9, 9, 9, 9, 7, 9, 9, 9, 9, 8, 9, 8, 9, 9, 6, 1, 8, 8, 6, 2, 6, 2, 9, 8, 9, 9, 9, 4, 1, 5, 9, 8, 8, 7, 1, 10, 9, 6, 10, 1, 1, 1, 8, 1, 8, 6, 9, 9, 9, 6, 9, 9, 1, 5, 9, 9, 8, 10, 9, 8, 9, 2, 9, 9, 5, 8, 9, 9, 9, 9, 8, 1, 7, 9, 3, 9, 9, 5, 8, 1, 12, 9, 8, 3, 3, 1, 2, 8, 8, 9, 9, 3, 1, 9, 1, 9, 1, 2, 2, 9, 8, 1, 7, 8, 9, 9, 9, 9, 9, 9, 9, 9, 3, 9, 9, 9, 9, 9, 9, 8, 9, 9, 1, 2, 8, 1, 6, 1, 5, 6, 9, 9, 9, 9, 1, 9, 2, 2, 9, 1, 8, 8, 2, 8, 9, 9, 7, 9, 1, 1, 6, 2, 3, 8, 9, 7, 2, 1, 9, 9, 1, 4, 2, 4, 1, 9, 8, 10, 9, 10, 2, 1, 2, 3, 9, 5, 8, 2, 9, 8, 9, 9, 9, 1, 9, 9, 9, 9, 9, 7, 9, 9, 1, 3, 9, 1, 1, 9, 8, 8, 3, 9, 10, 10, 1, 9, 9, 9, 9, 9, 2, 5, 9, 9, 9, 9, 9, 4, 2, 4, 9, 1, 10, 9, 9, 9, 9, 9, 9, 1, 7, 9, 9, 1, 8, 7, 8, 9, 9, 9, 7, 1, 1, 1, 9, 9, 2, 7, 9, 2, 3, 3, 1, 9, 9, 1, 8, 9, 7, 3, 8, 9, 9, 1, 9, 9, 9, 9, 9, 9, 9, 4, 4, 2, 5, 12, 9, 3, 9, 9, 9, 9, 9, 1, 4, 1, 4, 9, 2, 9, 9, 9, 3, 5, 8, 2, 6, 9, 5, 1, 2, 7, 7, 4, 1, 2, 9, 2, 1, 3, 2, 9, 4, 2, 4, 8, 4, 2, 1, 10, 9, 9, 9, 5, 1, 9, 9, 9, 9, 2, 2, 9, 9, 1, 1, 1, 3, 2, 1, 6, 1, 9, 9, 9, 7, 9, 9, 9, 10, 9, 9, 9, 9, 9, 9, 1, 9, 8, 9, 9, 9, 3, 1, 1, 9, 9, 1, 1, 2, 1, 2, 8, 10, 4, 43, 9, 9, 6, 1, 2, 9, 7, 9, 9, 9, 10, 9, 10, 9, 10, 1, 8, 9, 9, 1, 3, 9, 9, 9, 9, 9, 9, 9, 10, 2, 9, 2, 9, 9, 6, 9, 2, 2, 1, 9, 1, 3, 1, 9, 9, 9, 9, 9, 8, 9, 1, 9, 9, 9, 8, 9, 9, 9, 8, 2, 8, 9, 11, 1, 3, 6, 9, 9, 8, 8, 1, 3, 1, 6, 8, 7, 6, 1, 5, 2, 2, 2, 4, 9, 9, 9, 8, 9, 9, 8, 8, 9, 9, 9, 9, 9, 9, 10, 4, 9, 1, 4, 4, 9, 1, 1, 3, 1, 1, 6, 4, 9, 9, 1, 3, 12, 1, 3, 5, 1, 1, 1, 9, 2, 1, 9, 9, 9, 4, 3, 9, 9, 9, 9, 9, 3, 3, 8, 9, 4, 4, 9, 9, 9, 9, 9, 8, 5, 9, 8, 5, 9, 9, 9, 1, 5, 2, 1, 1, 1, 10, 1, 2, 9, 8, 9, 9, 7, 8, 9, 9, 9, 1, 1, 8, 3, 5, 4, 2, 2, 9, 10, 1, 9, 9, 2, 9, 9, 10, 9, 9, 9, 7, 9, 9, 1, 9, 9, 7, 9, 9, 9, 9, 8, 9, 7, 2, 1, 8, 9, 8, 9, 9, 9, 9, 9, 9, 9, 8, 9, 9, 9, 9, 8, 4, 1, 5, 7, 9, 9, 9, 12, 1, 9, 9, 1, 1, 7, 9, 9, 2, 1, 9, 11, 9, 9, 1, 1, 3, 4, 8, 1, 2, 1, 9, 9, 9, 1, 4, 8, 4, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1, 6, 9, 9, 9, 9, 1, 9, 9, 2, 9, 9, 9, 9, 9, 9, 9, 4, 2, 1, 7, 9, 9, 9, 1, 1, 2, 1, 2, 4, 7, 9, 9, 1, 3, 9, 9, 9, 9, 9, 9, 9, 9, 3, 2, 3, 3, 9, 9, 9, 9, 9, 7, 9, 9, 1, 7, 1, 4, 7, 1, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 6, 8, 6, 1, 9, 9, 8, 1, 8, 9, 1, 2, 1, 9, 9, 1, 9, 13, 1, 1, 9, 9, 1, 9, 9, 10, 9, 9, 9, 9, 8, 9, 2, 5, 1, 3, 4, 9, 3, 5, 14, 5, 8, 6, 2, 9, 9, 9, 9, 9, 5, 9, 9, 4, 8, 9, 8, 9, 9, 6, 9, 9, 1, 1, 9, 9, 9, 8, 7, 1, 7, 1, 1, 1, 8, 7, 9, 9, 3, 1, 2, 9, 9, 9, 9, 3, 20, 13, 9, 9, 9, 9, 9, 9, 8, 9, 6, 5, 1, 9, 9, 2, 6, 5, 1, 9, 8, 9, 9, 14, 10, 9, 9, 1, 8, 9, 9, 9, 9, 4, 9, 9, 9, 9, 9, 9, 9, 10, 9, 9, 9, 1, 1, 9, 10, 9, 2, 2, 10, 8, 9, 1, 1, 8, 8, 3, 3, 5, 4, 2, 10, 1, 7, 4, 1, 9, 9, 5, 1, 1, 1, 9, 9, 8, 9, 9, 9, 9, 7, 7, 7, 3, 10, 8, 8, 9, 7, 1, 2, 1, 2, 9, 8, 9, 2, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1, 8, 3, 3, 2, 11, 9, 9, 9, 9, 8, 9, 1, 1, 2, 7, 2, 4, 9, 8, 6, 7, 4, 6, 9, 9, 9, 9, 8, 9, 1, 9, 8, 4, 4, 3, 1, 9, 9, 9, 9, 9, 9, 6, 1, 1, 11, 9, 9, 9, 9, 9, 9, 9, 7, 9, 1, 6, 3, 2, 1, 9, 6, 8, 9, 3, 1, 6, 4, 8, 1, 1, 9, 3, 6, 10, 2, 4, 1, 9, 3, 6, 9, 9, 9, 1, 9, 9, 9, 8, 1, 1, 6, 8, 1, 1, 7, 9, 8, 8, 8, 9, 9, 9, 5, 10, 7, 9, 9, 9, 15, 9, 9, 9, 2, 9, 9, 9, 1, 9, 9, 9, 9, 1, 9, 7, 7, 3, 5, 6, 9, 1, 1, 4, 5, 4, 7, 5, 9, 9, 8, 9, 9, 9, 9, 9, 7, 2, 9, 9, 9, 1, 4, 2, 1, 2, 1, 4, 1, 9, 9, 8, 9, 8, 7, 1, 8, 9, 3, 1, 5, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 4, 9, 9, 9, 9, 2, 3, 1, 9, 9, 9, 1, 8, 4, 9, 8, 9, 9, 1, 9, 9, 9, 9, 9, 1, 5, 1, 9, 9, 2, 1, 6, 8, 8, 9, 8, 9, 9, 9, 1, 9, 7, 1, 6, 9, 9, 9, 2, 8, 6, 3, 9, 5, 3, 2, 1, 2, 2, 10, 9, 11, 9, 9]
short_speed_data = [6.39355596e-02, 1.50005879e-02, 2.40355699e-02, 2.55815518e-02, 4.60590000e+04]

37957
shot_steps_data = [9, 3, 2, 2, 8, 3, 2, 2, 9, 9, 8, 26, 10, 33, 9, 1, 1, 4, 8, 27, 1, 3, 1, 1, 8, 4, 2, 20, 7, 8, 17, 1, 8, 9, 9, 32, 1, 42, 11, 1, 27, 8, 9, 1, 17, 4, 5, 5, 9, 19, 54, 8, 14, 4, 16, 3, 5, 9, 8, 9, 8, 9, 2, 9, 7, 1, 5, 8, 9, 3, 6, 42, 37, 33, 1, 2, 2, 4, 8, 2, 2, 3, 3, 3, 2, 5, 1, 2, 1, 7, 7, 7, 4, 3, 3, 6, 9, 3, 1, 8, 42, 2, 1, 1, 5, 4, 4, 6, 8, 9, 1, 2, 10, 1, 8, 3, 1, 8, 1, 3, 1, 3, 2, 2, 1, 8, 8, 8, 2, 3, 3, 7, 3, 8, 19, 5, 1, 2, 1, 8, 30, 11, 9, 1, 1, 6, 9, 9, 43, 4, 2, 2, 9, 9, 13, 9, 9, 13, 18, 1, 139, 2, 2, 2, 4, 6, 29, 8, 9, 8, 5, 3, 1, 10, 1, 10, 7, 8, 8, 1, 1, 1, 6, 1, 9, 8, 1, 9, 8, 24, 43, 8, 8, 8, 5, 1, 2, 33, 1, 9, 39, 1, 10, 15, 8, 32, 10, 9, 42, 1, 7, 6, 43, 11, 4, 41, 2, 30, 1, 8, 9, 7, 43, 3, 1, 43, 9, 28, 9, 21, 8, 8, 10, 42, 22, 10, 1, 9, 4, 1, 8, 8, 4, 10, 16, 1, 44, 13, 10, 1, 8, 5, 2, 29, 8, 8, 11, 8, 29, 8, 5, 10, 8, 5, 3, 4, 8, 2, 24, 1, 1, 4, 1, 7, 26, 9, 7, 1, 23, 5, 13, 6, 42, 8, 4, 2, 1, 8, 10, 8, 10, 24, 34, 8, 8, 2, 22, 22, 1, 10, 8, 8, 3, 1, 5, 1, 16, 98, 42, 16, 8, 8, 24, 14, 7, 3, 1, 1, 9, 6, 23, 8, 1, 8, 1, 4, 8, 4, 6, 24, 9, 11, 43, 17, 19, 10, 2, 9, 14, 1, 8, 8, 11, 43, 9, 3, 4, 4, 2, 2, 9, 8, 1, 2, 7, 7, 8, 7, 8, 35, 9, 18, 3, 8, 28, 25, 9, 16, 2, 39, 8, 2, 1, 5, 24, 9, 1, 9, 11, 6, 10, 2, 8, 8, 8, 8, 1, 7, 8, 1, 8, 2, 2, 10, 9, 4, 8, 2, 1, 2, 21, 1, 8, 1, 8, 1, 1, 1, 24, 4, 6, 1, 8, 7, 1, 3, 9, 8, 9, 8, 2, 10, 2, 9, 9, 8, 1, 8, 8, 1, 10, 25, 3, 9, 1, 8, 1, 2, 2, 1, 5, 9, 1, 10, 1, 8, 10, 1, 8, 24, 38, 3, 29, 13, 27, 10, 1, 6, 9, 1, 8, 8, 8, 2, 8, 1, 8, 8, 8, 39, 10, 10, 1057, 34, 8, 9, 24, 32, 2, 8, 5, 1, 1, 3, 4, 25, 1, 43, 56, 9, 7, 2, 1, 8, 8, 1, 3, 9, 10, 5, 8, 1, 1, 292, 8, 7, 6, 2, 3, 3, 2, 9, 8, 8, 8, 1, 16, 2, 6, 29, 8, 28, 7, 8, 10, 9, 1377, 2, 5, 9, 9, 32, 21, 11, 26, 9, 29, 15, 9, 1, 10, 43, 13, 8, 1, 2, 8, 5, 13, 3, 1, 42, 2, 1, 16, 3, 13, 9, 10, 16, 9, 1, 9, 1, 10, 5, 9, 8, 4, 4, 3, 33, 36, 1, 1, 43, 10, 9, 10, 2, 4, 21, 2, 2, 8, 5, 9, 1, 1, 1, 36, 13, 9, 1, 43, 2, 37, 1, 6, 8, 43, 25, 9, 9, 1, 8, 8, 8, 16, 10, 9, 40, 1, 1, 3, 1, 8, 8, 2, 1, 2, 18, 30, 637, 4, 8, 1, 8, 8, 1, 2, 4, 12, 2, 3, 2, 1, 3, 2, 8, 25, 1, 1, 3, 7, 5, 1, 8, 1, 5, 1, 8, 5, 2, 29, 9, 2, 14, 8, 7, 9, 12, 9, 1, 1, 10, 23, 3, 8, 8, 4, 1, 9, 3, 2, 9, 9, 24, 8, 30, 2, 18, 8, 27, 10, 32, 1, 1, 8, 8, 8, 2, 38, 10, 8, 3, 3, 9, 3, 1, 6, 24, 9, 8, 3, 1, 9, 24, 9, 1, 9, 2, 1, 6, 30, 1, 1, 10, 2, 2, 2, 13, 3, 1, 1, 8, 1, 1, 8, 7, 3, 24, 2, 6, 34, 1, 9, 71, 3, 1, 2, 10, 9, 10, 43, 1, 22, 8, 42, 16, 1, 1, 19, 2, 10, 10, 1, 6, 7, 8, 23, 1, 1, 4, 52, 40, 8, 1, 1, 6, 4, 4, 8, 3, 8, 8, 9, 1, 8, 8, 8, 4, 29, 42, 12, 9, 6, 3, 1, 2, 1, 24, 9, 8, 13, 2, 1, 9, 1, 4, 1, 1, 2, 1, 3, 2, 8, 9, 8, 2, 2, 1, 8, 1, 4, 4, 24, 1, 8, 13, 1, 1, 6, 1, 1, 24, 2, 27, 10, 11, 8, 9, 3, 8, 8, 11, 1, 1, 8, 9, 3, 8, 9, 43, 10, 8, 2, 1, 32, 8, 8, 9, 42, 1, 8, 13, 32, 9, 42, 1, 3, 9, 9, 10, 8, 8, 10, 42, 10, 10, 6, 5, 1, 8, 9, 1, 2, 1, 3, 5, 36, 8, 34, 7, 8, 2, 4, 2, 8, 8, 8, 7, 1, 1, 8, 5, 3, 9, 4, 3, 1, 8, 9, 1, 8, 1, 1, 1, 24, 5, 10, 43, 8, 24, 8, 9, 8, 16, 10, 9, 1, 6, 2, 1, 5, 3, 3, 1, 21, 11, 42, 10, 17, 3, 9, 9, 2, 8, 6, 19, 8, 9, 21, 80, 8, 4, 2, 2, 10, 1, 1, 2, 3, 3, 2, 8, 8, 9, 2, 11, 4, 3, 1, 9, 2, 7, 10, 1, 9, 1, 11, 5, 7, 8, 10, 4, 7, 1, 6, 42, 9, 8, 2, 10, 9, 8, 16, 79, 19, 8, 1, 8, 6, 1, 2, 10, 7, 8, 11, 43, 3, 7, 24, 2, 2, 9, 1, 9, 9, 33, 8, 1, 8, 8, 26, 9, 2, 1, 8, 8, 1, 3, 6, 1, 2, 7, 1, 9, 1, 8, 8, 8, 24, 10, 8, 11, 8, 29, 29, 3, 1, 2, 14, 1, 9, 8, 3, 1, 5, 29, 13, 9, 35, 9, 3, 9, 8, 8, 1, 6, 1, 8, 3, 3, 9, 7, 1, 8, 4, 3, 8, 3, 1, 8, 4, 7, 4, 1, 4, 18, 4, 4, 10, 19, 15, 10, 1, 17, 1, 8, 8, 10, 25, 6, 6, 2, 2, 8, 7, 1, 9, 22, 12, 9, 8, 5, 8, 1, 4, 9, 1, 10, 7, 1, 9, 1, 8, 10, 10, 2, 30, 9, 1, 8, 8, 9, 8, 8, 9, 9, 8, 1, 1, 52, 10, 9, 1, 2, 1, 3, 8, 9, 3, 2, 10, 2, 19, 2, 11, 1, 9, 2, 5, 10, 1, 5, 1, 8, 8, 1, 8, 40, 6, 10, 6, 9, 42, 2, 8, 9, 8, 3, 8, 29, 22, 1, 4, 2, 7, 14, 9, 15, 9, 8, 9, 1, 7, 1, 2, 10, 4, 10, 5, 28, 1, 9, 23, 8, 1, 3, 5, 2, 11, 9, 9, 7, 8, 47, 1, 10, 8, 6, 8, 8, 38, 8, 3, 25, 43, 1, 9, 11, 12, 2, 3, 5, 10, 3, 10, 4, 9, 2, 20, 9, 1, 4, 8, 27, 9, 8, 3, 2, 1, 3, 3, 2, 3, 1, 8, 8, 5, 10, 8, 10, 1, 1, 8, 32, 3, 1, 2, 2, 8, 1, 1, 2, 8, 8, 8, 10, 44, 3, 5, 18, 9, 8, 7, 29, 10, 8, 32, 5, 2, 1, 4, 8, 9, 10, 1, 9, 2, 8, 2, 7, 1, 8, 1, 7, 1, 4, 10, 3, 4, 1, 3, 8, 1, 8, 8, 1, 3, 1, 3, 3, 1, 8, 7, 55, 1, 43, 30, 42, 12, 3, 2, 1, 13, 9, 9, 8, 39, 6, 8, 10, 4, 15, 3, 1, 2, 1, 42, 38, 1, 10, 9, 8, 1, 8, 8, 8, 1, 2, 8, 1, 2, 1, 4, 14, 34, 4, 35, 10, 31, 8, 9, 9, 35, 3, 2, 5, 8, 2, 2, 10, 3, 1, 19, 8, 8, 10, 9, 8, 3, 2, 1, 1, 1, 5, 2, 9, 8, 8, 5, 9, 1, 30, 8, 1, 2, 9, 9, 9, 9, 5, 42, 11, 2, 3, 1, 14, 10, 43, 3, 2, 13, 7, 9, 8, 1, 2, 2, 4, 2, 8, 8, 1, 1, 2, 21, 9, 1, 3, 9, 11, 4, 6, 1, 8, 24, 8, 43, 1, 16, 147, 15, 4, 4, 1, 39, 2, 4, 1, 9, 3, 3, 2, 1, 8, 8, 10, 9, 8, 5, 9, 7, 13, 9, 8, 8, 8, 8, 9, 8, 2, 8, 1, 2, 9, 10, 9, 1, 8, 9, 9, 9, 1, 1, 6, 9, 10, 1, 3, 8, 3, 5, 24, 2, 7, 11, 28, 1, 2, 10, 8, 3, 4, 8, 2, 4, 8, 23, 9, 31, 2, 5, 33, 8, 2, 2, 8, 8, 2, 2, 16, 1, 16, 1, 1, 2, 38, 5, 18, 16, 2, 8, 1155, 20, 2, 14, 1, 5, 8, 8, 5, 3, 1, 2, 32, 8, 15, 3, 1, 1, 2, 1, 33, 8, 8, 8, 9, 8, 1, 1, 17, 8, 8, 1, 3, 8, 1, 2, 8, 28, 5, 1, 7, 15, 7, 3, 1, 1, 9, 8, 14, 1, 13, 8, 6, 8, 7, 9, 50, 26, 45, 8, 42, 8, 9, 8, 4, 8, 24, 1, 8, 28, 4, 8, 6, 6, 3, 24, 8, 31, 1, 2, 6, 9, 2, 7, 11, 27, 3, 8, 21, 8, 9, 1, 9, 8, 19, 24, 42, 3, 6, 4, 3, 8, 9, 1, 9, 9, 2, 9, 1, 1, 8, 7, 3, 10, 8, 11, 1, 24, 3, 1, 13, 9, 15, 6, 9, 37, 1, 1, 1, 9, 2, 17, 9, 8, 8, 9, 9, 10, 9, 9, 1, 1, 8, 4, 11, 8, 1, 4, 7, 2, 12, 2, 1, 8, 1, 1, 1, 9, 35, 1, 4, 18, 14, 627, 8, 1, 42, 5, 2, 9, 1, 5, 8, 33, 2, 8, 8, 8, 8, 9, 2, 3, 3, 1, 3, 2, 9, 1, 25, 4, 8, 1, 2, 5, 2, 6, 13, 3, 3, 8, 4, 2, 10, 34, 3, 1, 1, 1, 9, 9, 43, 9, 7, 2, 6, 3, 7, 4, 4, 5, 13, 2, 31, 8, 42, 8, 50, 8, 25, 1, 9, 2, 1, 2, 5, 1, 2, 11, 10, 11, 28, 7, 17, 3, 9, 10, 8, 8, 8, 8, 1, 24, 8, 5, 9, 1, 10, 1, 1, 8, 8, 8, 32, 43, 11, 8, 2, 7, 7, 8, 13, 15, 7, 10, 8, 1, 2, 4, 3, 8, 26, 10, 9, 9, 2, 10, 9, 3, 3, 16, 9, 12, 1, 2, 9, 1, 1, 3, 7, 9, 9, 2, 1, 3, 1, 2, 8, 24, 8, 10, 6, 6, 1, 5, 4, 1, 14, 8, 6, 3, 1, 4, 1, 7, 8, 24, 9, 24, 8, 86, 2, 15, 4, 2, 8, 9, 11, 11, 9, 13, 1, 8, 5, 1, 2, 7, 8, 8, 1, 5, 9, 8, 7, 8, 1, 8, 8, 8, 8, 1, 9, 38, 8, 5, 4, 9, 8, 8, 10, 8, 8, 1, 9, 9, 9, 25, 9, 2, 2, 9, 8, 9, 2, 9, 8, 7, 1, 7, 38, 9, 24, 9]
shot_speed_data = [7.45146682e-02, 1.50001623e-02, 2.98254678e-02, 3.11145930e-02, 3.79570000e+04]

16563
high_speed_y_data1 = [1.97921467e+00, 1.50024053e-02, 2.25985318e-01, 2.77630997e-01, 1.65630000e+04]
high_speed_y_data2 = [1.59930253e+00, 1.50077222e-02, 2.33815990e-01, 2.85041042e-01, 1.59800000e+04]

print(a := (high_speed_y_data1[2]+high_speed_y_data2[2])/2)
print(b := (high_speed_y_data1[3]+high_speed_y_data2[3])/2)
print((shot_speed_data[3]*2 + shot_speed_data[2])/3)

steps_data = [step for step in steps_data if step < 15 and step > 1]

def reject_outliers(data, m=2):
    centralised_data = data - np.mean(data)
    comparison = abs(centralised_data) < m * np.std(data)
    return comparison

# tmp = steps_data
tmp = np.array(steps_data)[reject_outliers(steps_data, m=1)]

result = [np.max(tmp), 
        np.min(tmp), 
        np.median(tmp),
        np.average(tmp),
        len(tmp)]

print(result)

long_steps = 8.2
# long_steps = 8.16
# long_speed = 0.02 - 0.025 = 0.0225

high_steps = 6.9
# high_steps = 6.89
# high_speed = 0.02 - 0.025

short_steps = 8.2
# short_steps = 8.35
# short_speed = 0.02 - 0.025

shot_steps = 7.5
# shot_steps = 7.45

pass_speed = 0.025
shot_speed = 0.031
pass_y_speed = 0.265

# acceleration = 0.001 ~ 0.0015