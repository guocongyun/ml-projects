import numpy as np

data = [[[], [5, 5, 5, 27, 11, 11, 11, 11, 11, 2, 3, 47, 5, 11, 10, 11, 7, 5, 11, 6, 6, 7, 6], [6, 6, 5, 3, 5, 2, 2, 7, 8, 10, 8, 8, 4, 1, 8, 1, 8, 5, 9, 4, 1, 8, 7], [1, 10, 3, 3, 1, 13, 10, 7, 11, 7, 23, 23, 4, 3, 5]],
[[], [5, 11, 5, 11, 5, 9, 11, 10, 3, 12, 5, 2, 8, 11], [2, 1, 13, 1, 6, 2, 1, 2, 8, 1, 8, 8, 9], [25, 23, 7, 6, 2, 35, 3, 4, 11]],
[[], [11, 7, 11, 1, 11, 3, 5, 11], [8, 8], [6, 23, 8]],
[[], [11, 12, 2, 9], [9], []],
[[], [2, 23, 8, 11, 9, 2, 2, 4, 8, 24, 1, 8], [8, 8, 9, 9], [3, 2, 23, 44, 1, 20, 2, 8]],
[[], [12, 1, 11, 9, 11, 11, 10, 11, 6, 11, 1, 7, 11, 11], [8, 8], [6, 45]],
[[], [11, 11, 10, 12, 11, 11, 11, 11, 11, 11, 10, 11, 11, 10, 11, 11, 11], [8, 6, 5, 8], [8, 7, 24, 4, 8, 4, 23, 23, 7, 1, 7, 1, 1, 23, 23, 2, 9, 44, 10]],
[[], [10, 11], [9, 2, 2, 6], [17, 2, 45]],
[[], [9, 5, 44, 11, 11, 11, 9, 11], [2], [9, 38, 4, 11, 6]],
[[], [10, 5, 4, 11, 10, 11, 7, 8, 3, 8, 8, 11, 1, 7, 12, 11, 7, 10, 10, 3], [9, 9, 9, 10, 2, 3], [11, 3, 7, 7, 23, 5, 1, 10, 44, 7, 1]],
[[], [10], [], [4, 3]],
[[], [], [1, 3], []],
[[], [11], [1, 9, 8, 7, 3], [10]],
[[], [44, 12, 2, 1, 1, 6], [1, 6, 9], [1, 1, 8, 10, 5, 14, 15, 9]],
[[], [11, 5, 1, 1, 7], [4, 1], [11, 11, 44]],
[[], [11, 11, 3, 5, 11, 1, 7, 24, 9, 1, 11], [8, 8, 8, 1, 10], [6, 4, 23, 7, 4, 44, 1, 4, 10, 2, 7, 3]],
[[], [2, 2, 1, 3, 1, 5, 3, 11, 5], [9, 8, 16], [1, 1, 1, 7, 23, 29, 9]],
[[], [], [5], [11, 14, 17]],
[[], [11, 11], [], [7]],
[[], [11, 7, 6, 1, 11, 1, 11, 13], [1], [6, 15]],
[[], [5, 10, 4], [10, 1, 6, 3, 9, 1, 3, 6, 1, 5, 5, 2], [52, 7, 6, 1]],
[[], [10, 9, 11, 11, 11, 1, 2, 8, 12, 9, 2, 10], [18, 8], [7, 7, 7, 1, 5, 4, 8, 1, 3, 2, 1, 10, 23, 16]],
[[], [1, 3, 11, 8, 12, 10, 12, 11, 5, 1, 11, 11, 11, 5, 4, 8, 9, 11, 5, 11, 11], [6, 5, 3, 5, 1, 9, 9, 1, 3, 3, 3, 1, 11, 3, 1, 8, 4, 2, 9, 7, 5, 12, 9], [22, 6, 4, 17, 16, 11, 6, 23, 1, 9, 9, 44]],
[[], [2, 1, 10], [], [3, 3, 15]],
[[], [23, 20, 11, 10, 2, 1, 23], [], [9, 10, 3]],
[[], [6, 5, 12, 10], [8], [23, 11]],
[[], [1], [8], [17]],
[[], [7, 2, 4, 10, 17, 1, 44, 33, 11], [9, 9, 10, 8], [7, 2, 1, 23, 3, 2, 8, 8, 9, 7, 2, 8, 10]],
[[], [10, 11, 3, 1, 6, 11, 11, 9, 11, 11, 11, 9, 11, 11, 11, 11, 12, 5, 25, 5, 9, 11, 2, 3, 8, 11, 11], [6, 3, 2, 2, 4, 12, 1, 3, 2, 2, 9, 8, 8, 3], [8, 3, 7, 9, 44, 1, 2, 7, 8, 1, 5, 2]],
[[], [5, 11, 12, 7, 6, 1, 5, 11, 10, 12, 5, 11, 11, 12, 1, 7, 2, 12, 1], [8, 8, 8], [1, 1, 15, 7, 10, 18, 2, 1, 8]],
[[], [8, 17, 6, 10, 5, 3, 2, 2, 1, 1, 3, 8], [8, 3, 1, 1, 3, 10, 2, 3, 5, 8, 5, 9, 10, 8, 10, 1, 9], [6, 2, 2, 23]],
[[], [9, 5], [1, 8, 7, 8, 2], [10, 23, 59, 10, 1, 5]],
[[], [11, 44, 33], [], [9, 44]],
[[], [11, 10, 12, 11, 10, 11, 12, 23, 11, 11, 8, 5, 4, 12, 9, 11], [4, 1, 4, 2, 8, 3, 1, 8, 4, 1, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [1, 8, 7, 8]],
[[], [2, 6, 10, 4, 11, 1, 11, 18, 20], [9], [47]],
[[], [3, 12, 5, 12, 8, 12, 7, 12, 11, 2, 1, 11, 3, 11, 2, 7, 6, 11, 3, 3, 4, 1, 24, 1, 17], [3, 3, 1, 1, 1, 2, 1, 8, 5, 5, 4, 6, 4, 1, 9, 4], [11, 8, 1, 24, 7, 12, 5, 7, 10, 21, 23]],
[[], [4, 11, 3, 7, 11, 2, 2, 7, 10, 2, 5, 11, 11, 11], [8, 2, 1, 7], [1, 7, 6, 2, 7, 3, 10, 3]],
[[], [45, 11, 11, 1, 11, 12, 11, 11], [2, 5, 7], [24, 3, 10, 23, 1]],
[[], [5, 1, 7, 1, 11, 11, 11, 1], [8, 8, 1, 10, 1], [9, 10, 8, 3, 8, 3, 2, 7, 8, 24, 1, 47]],
[[], [3, 7, 5, 11, 11, 11], [], [8, 2, 2, 13, 1, 23, 1, 23, 3, 1, 5, 7]],
[[], [1], [8], [7, 7]],
[[], [9, 6, 11, 3, 11, 4, 6, 10, 5, 10], [4, 1, 2, 1, 1, 10, 8, 7, 1, 8, 8, 1, 8], [7, 20, 6, 1, 8, 5, 8, 8, 3, 7, 8, 11]],
[[], [9, 24, 1, 5, 11, 4], [8, 9, 14, 4], [7, 10, 24, 1, 1]],
[[], [2, 8], [], [9, 2, 1]],
[[], [11, 11, 11, 5, 11, 12, 11, 12, 6, 44, 5], [1, 8, 4, 1, 2, 2, 4, 1, 3, 8, 3, 1, 1, 3, 1, 2, 1, 1, 2, 4, 5, 3, 2], [7, 11, 4, 7, 10]],
[[], [5, 11, 8, 5, 5, 10, 6, 10, 5, 1, 11, 8, 1, 15], [2, 9, 3, 3, 4, 3, 3, 2, 1, 8, 7, 5, 1, 1, 1, 8, 8, 3, 8, 8, 2, 1, 3], [2, 3, 1, 14, 4, 17, 10, 8, 23, 16]],
[[], [11, 11, 8, 10, 8, 2, 10, 1, 3, 6, 44, 13, 2, 7, 6, 11], [8, 12, 4], [8, 9, 4, 8, 8, 14, 9]],
[[], [9, 11, 11, 2, 11, 7, 11, 12, 5, 7, 11, 5, 11, 12, 5, 1, 1, 10, 5, 7, 11, 6, 10, 11, 5, 11], [2, 1, 5, 9, 1, 1, 3, 6, 2, 1, 7, 1, 8, 8, 1, 7, 4, 2, 2, 3, 9, 6, 2], [6, 7, 7, 7, 9, 7, 7, 1, 24, 23]],
[[], [5, 11, 11, 26, 10, 1, 11, 11, 7, 11, 1, 2, 10], [8, 8], [23, 7, 3, 3, 8, 15, 3, 2, 23, 23, 1, 4, 5, 7, 8, 23]],
[[], [11, 11, 2, 3, 3, 44, 9, 12, 18], [8, 8], [12, 8, 7, 6]],
[[], [10], [1], [7, 23, 5, 7, 3]],
[[], [1, 5, 5, 12, 2, 11], [8, 8, 5], []],
[[], [10, 1, 7, 2, 44, 21, 6, 3, 11, 17, 11, 4, 1, 11, 11, 9, 9, 5, 2], [8, 9, 6, 8, 7, 9, 8], [8, 14, 1, 8, 7, 8, 5, 3, 23, 3, 1, 11]],
[[], [2, 5, 11, 11, 10, 12, 1, 8, 11, 11], [], [11, 13, 44, 2, 8]],
[[], [11, 11, 10, 17, 4, 6, 10, 44, 6, 5, 11, 11, 1, 8, 12, 10, 44, 24, 5], [1, 8, 9, 2, 9, 2, 3, 3, 1, 1, 1, 1, 4, 8, 6, 1, 3, 3, 7, 5, 8, 1, 1], [16, 23, 8, 23, 7, 2, 1, 8, 4, 44, 45]],
[[], [11, 11, 1, 1, 7, 12], [1], [4, 3, 13, 7, 7]],
[[], [3, 11, 1, 2, 3, 5, 10, 3, 6, 12, 9, 12, 7, 12, 11, 13, 11, 7], [4, 1, 3, 2, 8, 1, 4, 7, 3, 9, 3, 1, 2, 9, 2, 7, 1, 2, 1, 2], [3, 23, 8, 10, 2, 2, 18, 4, 7, 8]]]

data2 = [[], 
[2, 2, 1, 12, 7, 11, 11, 11, 10, 1, 5, 11, 11, 10, 1, 5, 5, 5, 10, 11, 11, 8, 5, 5, 8, 12, 7, 11, 8, 12, 9, 12, 11, 11, 5, 8, 1, 1, 4, 1, 2, 3, 13, 2, 1, 2, 2, 8, 11, 11, 11, 11, 11, 10, 5, 11, 9, 11, 10, 11, 1, 11, 3, 4, 5, 2, 2, 11, 1, 10, 11, 11, 3, 2, 2, 5, 10, 5, 4, 11, 9, 11, 1, 11, 6, 11, 11, 11, 11, 11, 11, 24, 5, 6, 5, 32, 2, 11, 5, 11, 11, 12, 1, 1, 11, 11, 2, 2, 5, 11, 2, 3, 3, 10, 11, 17, 10, 8, 11, 7, 11, 8, 12, 10, 5, 23, 11, 7, 12, 5, 5, 1, 4, 3, 1, 11, 9, 7, 10, 3, 22, 11, 6, 3, 2, 5, 11, 5, 14, 11, 5, 11, 11, 5, 11, 10, 4, 1, 10, 11, 5, 47, 5, 2, 9, 11, 11, 8, 11, 11, 11, 12, 11, 10, 1, 7, 11, 44, 6, 4, 7, 11, 11, 11, 6, 10, 10, 5, 5, 12, 7, 6, 11, 6, 11, 11, 7, 12, 11, 9, 10, 10, 10, 9, 5, 8, 12, 1, 6, 5, 5, 5, 12, 16, 9, 11, 6, 11, 5, 12, 5, 5, 8, 11, 12, 5, 5, 12, 11, 11, 10, 9, 11, 11, 11, 9, 3, 11, 4, 7, 2, 11, 44, 11, 3, 5, 5, 11, 11, 5, 4, 44, 9, 8, 12, 7, 1, 11, 7, 12, 6, 5, 11, 7, 11, 7, 10, 5, 1, 5, 1, 11, 6, 5, 6, 10, 5, 24, 1, 24, 44, 10, 8, 11, 9, 9, 6, 7, 5, 10, 11, 5, 7, 45, 3, 3, 10, 12, 6, 11, 11, 11, 13, 24, 11, 12, 4, 4, 1, 8, 11, 11, 5, 13, 12, 5, 4, 5, 3, 11, 5, 10, 5, 9, 6, 1, 1, 2, 9, 11, 12, 5, 6, 10, 1, 11, 1, 3, 11, 11, 2, 4, 11, 10, 11, 6, 8, 44, 8, 6, 3, 1, 5, 11, 6, 5, 1, 9, 11, 10, 11, 1, 1, 5, 24, 9, 11, 12, 11, 11, 11, 10, 11, 1, 1, 11, 11, 7, 1, 11, 11, 11, 11, 7, 5, 15, 16, 5, 4, 5, 11, 5, 11, 10, 2, 12, 14, 15, 8, 2, 1, 4, 44, 5, 10, 9, 6, 1, 11, 7, 11, 2, 11, 11, 11, 7, 12, 1, 11, 24, 13, 10, 8, 5, 5, 1, 12, 11, 23, 10, 2, 24, 5, 11, 11, 23, 10, 11, 2, 10, 1, 11, 5, 24, 11, 10, 11, 10, 5, 8, 11, 8, 6, 11, 5, 10, 19], 
[8, 8, 1, 7, 4, 8, 8, 7, 1, 3, 2, 2, 1, 1, 7, 3, 8, 7, 4, 4, 1, 6, 3, 3, 8, 9, 7, 2, 9, 1, 10, 6, 4, 1, 1, 2, 1, 3, 8, 11, 8, 8, 2, 1, 8, 8, 9, 2, 8, 8, 9, 8, 8, 9, 3, 8, 7, 2, 4, 3, 1, 1, 4, 9, 4, 2, 2, 1, 3, 2, 2, 8, 7, 2, 8, 9, 8, 2, 1, 8, 2, 7, 1, 10, 8, 8, 7, 10, 3, 2, 3, 3, 5, 2, 2, 4, 7, 1, 9, 7, 9, 8, 1, 1, 4, 1, 1, 8, 2, 8, 11, 8, 8, 1, 8, 2, 5, 1, 2, 6, 1, 1, 3, 4, 2, 1, 10, 2, 2, 7, 4, 4, 1, 8, 8, 1, 4, 9, 1, 2, 2, 1, 3, 2, 6, 1, 3, 3, 1, 2, 1, 3, 2, 8, 2, 1, 6, 8, 8, 12, 10, 9, 8, 8, 2, 2, 10, 4, 3, 2, 8, 1, 3, 3, 1, 1, 7, 3, 4, 5, 3, 4, 9, 2, 3, 1, 9, 5, 3, 2, 3, 1, 3, 8, 8, 1, 5, 1, 4, 1, 1, 3, 7, 3, 1, 8, 9, 1, 6, 1, 8, 1, 1, 5, 1, 1, 8, 2, 9, 2, 5, 1, 8, 2, 2, 1, 1, 2, 2, 7, 2, 3, 1, 1, 8, 1, 3, 1, 3, 3, 1, 1, 1, 6, 9, 3, 10, 6, 6, 13, 5, 5, 2, 4, 2, 5, 1, 4, 7, 5, 8, 2, 1, 2, 2, 1, 8, 4, 10, 8, 7, 7, 2, 7, 1, 5, 2, 7, 7, 2, 4, 5, 4, 9, 3, 2, 8, 10, 2, 8, 8, 8, 5, 1, 9, 8, 1, 4, 6, 1, 8, 7, 9, 8, 7, 2, 1, 8, 4, 9, 8, 2, 3, 3, 3, 3, 1, 4, 3, 3, 1, 12, 2, 1, 8, 6, 1, 8, 4, 2, 8, 7, 1, 1, 1, 1, 1, 9, 1, 4, 7, 2, 8, 3, 8, 1, 2, 4, 8, 1, 8, 3, 7, 8, 11, 3, 2, 8, 8, 1, 7, 1, 1, 2, 8, 8, 8, 1, 7, 8, 4, 8, 1, 7, 1, 1, 1, 1, 1, 8, 2, 2, 1, 8, 1, 1, 2, 7, 5, 1, 8, 9, 2, 8, 2, 6, 8, 1, 2, 2, 1, 6, 4, 2, 4, 2, 2, 9, 8], 
[9, 6, 44, 1, 2, 7, 10, 7, 4, 23, 3, 23, 10, 15, 7, 17, 13, 25, 1, 7, 45, 23, 1, 4, 4, 6, 25, 23, 1, 16, 23, 2, 14, 11, 7, 3, 7, 11, 4, 24, 23, 23, 23, 23, 23, 23, 23, 1, 1, 1, 12, 7, 7, 5, 1, 9, 4, 7, 4, 2, 26, 8, 9, 8, 2, 1, 2, 24, 3, 2, 8, 5, 7, 4, 1, 3, 23, 6, 7, 23, 6, 23, 8, 44, 2, 15, 7, 23, 8, 9, 10, 3, 1, 4, 23, 3, 44, 4, 1, 14, 45, 1, 44, 23, 47, 8, 7, 11, 9, 6, 24, 7, 8, 1, 5, 2, 9, 47, 4, 8, 45, 24, 3, 1, 7, 7, 2, 2, 1, 2, 23, 23, 1, 8, 2, 2, 9, 6, 3, 1, 8, 31, 1, 8, 10, 1, 9, 8, 2, 8, 4, 1, 24, 7, 6, 15, 2, 6, 24, 1, 9, 7, 24, 44, 10, 5, 9, 7, 2, 2, 7, 5, 1, 36, 3, 4, 7, 1, 6, 7, 1, 12, 2, 3, 2, 1, 6, 23, 3, 3, 7, 9, 1, 9, 1, 8, 7, 7, 6, 7, 1, 23, 9, 8, 2, 44, 2, 7, 3, 7, 1, 7, 9, 10, 44, 7, 3, 9, 23, 23, 23, 44, 1, 9, 4, 18, 5, 7, 10, 1, 24, 45, 7, 9, 7, 9, 7, 1, 7, 8, 1, 1, 3, 8, 3, 11, 8, 2, 38, 1, 1, 7, 4, 8, 1, 1, 3, 44, 7, 9, 17, 8, 2, 3, 7, 4, 8, 5, 7, 5, 1, 9, 4, 1, 3, 23, 3, 24]
]

data3 = [
[9, 2, 9, 6, 5, 24, 2, 11, 1, 4, 6, 1, 2, 1, 1, 1, 12], 
[1, 6, 9, 1, 11, 16, 5, 11, 18, 8, 7, 1, 11, 9, 8, 15, 11, 9, 10, 11, 4, 2, 1, 12, 2, 2, 2, 2, 9, 4, 10, 51, 7, 11, 1, 1, 1, 4, 5, 10, 9, 9, 10, 9, 9, 11, 12, 14, 2, 4, 16, 5, 10, 2, 2, 6, 7, 10, 9, 9, 6, 9, 9, 7, 7, 7, 7, 12, 8, 3, 2, 10, 14, 13, 11, 12, 3, 1, 11, 1, 4, 2, 9, 11, 14, 10, 10, 12, 11, 11, 9, 11, 12, 3, 2, 2, 2, 2, 1, 1, 10, 3, 1, 2, 11, 1, 3, 11, 9, 9, 13, 10, 10, 9, 9, 11, 10, 1, 8, 11, 11, 5, 4, 10, 2, 8, 2, 2, 1, 9, 7, 10, 38, 11, 10, 2, 3, 2, 9, 3, 4, 10, 16, 10, 10, 12, 7, 1, 1, 4, 1, 2, 2, 6, 14, 2, 9, 2, 1, 1, 7, 1, 3, 9, 9, 9, 19, 10, 10, 6, 13, 3, 1, 10, 11, 8, 13, 9, 8, 1, 2, 2, 2, 11, 10, 14, 10, 10, 1, 907, 12, 9, 2, 6, 14, 9, 15, 8, 6, 11, 7, 10, 1, 9, 3, 7, 10, 5, 2, 7, 13, 10, 7, 12, 8, 7, 9, 3, 11, 4, 1, 1, 6, 9, 3, 3, 14, 9, 8, 2, 2, 1, 8, 8, 8, 9, 8, 1, 8, 15, 4, 8, 10, 11, 2, 12, 2, 8, 5, 10, 4, 10, 17, 1, 9, 11, 10, 10, 8, 9, 10, 10, 7, 3, 9, 3, 9, 9, 7, 13, 6, 12, 91, 8, 9, 6, 11, 10, 7, 7, 11, 11, 2, 8, 10, 12, 5, 9, 9, 11, 6, 2, 2, 11, 5, 6, 7, 2],
[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], 
[7, 3, 3, 2, 1, 12, 10, 8, 6, 1, 3, 4, 3, 8, 5, 7, 10, 7, 9, 8, 7, 7, 7, 9, 13, 11, 10, 4, 8, 9, 8, 8, 10, 7, 10, 5, 8, 2, 7, 10, 9, 7, 9, 6, 2, 10, 10, 1, 7, 1, 9, 12, 9, 1, 4, 12, 5, 10, 4, 2, 3, 6, 8, 10, 7, 11, 11, 9, 7, 7, 8, 11, 5, 7, 7, 7, 7, 8, 9, 8, 2, 15, 7, 10, 3, 4, 9, 9, 9, 9, 11, 9, 9, 7, 9, 10, 3, 8, 1, 7, 8, 7, 3, 3, 11, 8, 2, 8, 9, 7, 9, 12, 2, 1, 1, 8, 1, 2, 1, 7, 7, 10, 7, 2, 7, 1, 2, 20, 2, 10, 7, 10, 11, 2, 4, 5, 9, 2, 1, 9, 7, 9, 4, 8, 7, 9, 8, 10, 7, 1, 3, 7, 11, 8, 8, 5, 8, 7, 1, 2, 11, 11, 2, 1, 7, 3, 10, 8, 12, 9, 7, 9, 10, 5, 7, 10, 1, 10, 7, 4, 4, 10, 3, 2, 8, 6, 4, 9, 10, 12, 11, 5, 9, 8, 5, 13, 5, 1, 4, 7, 8, 8, 10, 7, 8, 9, 8, 8, 7, 10, 8, 5, 1, 4, 1, 10, 2, 5, 1, 2, 2, 4, 3, 1, 7, 10, 4, 12, 12, 5, 10, 10, 1, 9, 8, 4, 8, 7, 8, 12, 10, 9, 11, 8, 7, 7, 4, 1, 11, 8, 7, 1, 5, 1, 4, 4, 1, 6, 5, 8, 13, 4, 6, 12, 7, 7, 7, 7, 9, 9, 10, 11, 8, 8, 13, 11, 7, 1, 2, 6, 3, 11, 10, 1, 8, 1, 8, 8, 2, 11, 9, 11, 9, 8, 9, 7, 8, 7, 3, 3, 7, 7, 3, 8, 9, 9, 9, 9, 9, 11, 23, 9, 9, 13, 1, 4, 9, 8, 9, 7, 6, 6, 1, 6, 1, 10, 9, 3, 9, 12, 8, 9, 9]
]
def reject_outliers(data, m=2):
    centralised_data = data - np.mean(data)
    comparison = abs(centralised_data) < m * np.std(data)
    return comparison

result = []
result2 = []
result3 = []
for action in range(1,4):
    tmp = []
    for step in np.array(data)[:,action]:
        tmp += step
    result.append([np.max(np.array(tmp)[reject_outliers(tmp, m=1)]), 
            np.min(np.array(tmp)[reject_outliers(tmp, m=1)]), 
            np.median(np.array(tmp)[reject_outliers(tmp, m=1)]),
            np.average(np.array(tmp)[reject_outliers(tmp, m=1)]),
            len(tmp)])
print(np.array(result))

for tmp in data2:
    if tmp != []:
        result2.append([np.max(np.array(tmp)[reject_outliers(tmp, m=1)]), 
            np.min(np.array(tmp)[reject_outliers(tmp, m=1)]), 
            np.median(np.array(tmp)[reject_outliers(tmp, m=1)]),
            np.average(np.array(tmp)[reject_outliers(tmp, m=1)]),
            len(tmp)])
print(np.array(result2))

for tmp in data3:
    tmp2 = tmp[:]
    for num in range(len(tmp)): 
        if tmp[num] > 20: 
            tmp2.remove(tmp[num])
    tmp = tmp2[:]
    
    if tmp != []:
        result3.append([np.max(tmp), 
            np.min(tmp), 
            np.median(tmp),
            np.average(tmp),
            len(tmp)])
print(np.array(result3))
