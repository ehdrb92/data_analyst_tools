import numpy as np

"""0차원 배열"""

# arr0 = np.array(42)

# print(arr0)  # 42
# print(type(arr0))  # <class 'numpy.ndarray'>

"""1차원 배열"""

# arr1 = np.array([1, 2, 3, 4, 5])

# print(arr1)  # [1 2 3 4 5]

"""2차원 배열"""

# arr2 = np.array([[1, 2, 3], [4, 5, 6]])

# print(arr2)
"""
[[1 2 3]
 [4 5 6]]
"""

"""3차원 배열"""

# arr3 = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

# print(arr3)
"""
[[[1 2 3]
  [4 5 6]]

 [[1 2 3]
  [4 5 6]]]
"""

"""배열의 차원 수 확인"""

# print(arr0.ndim)  # 0
# print(arr1.ndim)  # 1
# print(arr2.ndim)  # 2
# print(arr3.ndim)  # 3

"""특정 배열을 다른 차원으로 만들기"""

# arr5 = np.array([1, 2, 3, 4], ndmin=5)

# print(arr5)  # [[[[[1 2 3 4]]]]]
# print(arr5.ndim)  # 5

"""1차원 배열 데이터에 접근"""

# arr = np.array([1, 2, 3, 4])

# print(arr[0])  # 1
# print(arr[1])  # 2
# print(arr[2] + arr[3])  # 7

"""2차원 배열 데이터에 접근"""

# arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

# print(arr[0, 1])  # 2
# print(arr[1, 4])  # 10
# print(arr[1, -1])  # 10

"""3차원 배열 데이터에 접근"""

# arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

# print(arr[0, 1, 2])  # 6
# print(arr[-1, -1, -1])  # 12

"""1차원 배열의 슬라이스"""

# arr = np.array([1, 2, 3, 4, 5, 6, 7])

# print(arr[1:5])  # [2 3 4 5]
# print(arr[4:])  # [5 6 7]
# print(arr[:4])  # [1 2 3 4]

# # 네거티브 인덱스 사용
# print(arr[-3:-1])  # [5 6]

# # 슬라이스 스텝 설정
# print(arr[1:5:2])  # [2 4]

"""2차원 배열의 슬라이스"""

# arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

# print(arr[1, 1:4])  # [7 8 9]
# print(arr[0:2, 2])  # [3 8]
# print(arr[0:2, 1:4])
"""
[[2 3 4]
 [7 8 9]]
"""

"""Numpy의 데이터 타입"""

"""
i - integer
b - boolean
u - unsigned integer
f - float
c - complex float
m - timedelta
M - datetime
O - object
S - String
U - unicode string
V - fixed chunk of memory for other type
"""

# arr1 = np.array([1, 2, 3, 4])
# arr2 = np.array(["apple", "banana", "cherry"])

# print(arr1.dtype)  # int64
# print(arr2.dtype)  # <U6

"""배열의 데이터 타입 정의하여 생성"""

# arr1 = np.array([1, 2, 3, 4], dtype="S")

# print(arr1)  # [b'1' b'2' b'3' b'4']
# print(arr1.dtype)  # |S1

# # 4바이트 integer
# arr2 = np.array([1, 2, 3, 4], dtype="i4")

# print(arr2)  # [1 2 3 4]
# print(arr2.dtype)  # int32

"""기존 배열의 데이터 타입 변환"""

# arr1 = np.array([1.1, 2.1, 3.1])

# print(arr1.dtype)  # float64

# newarr1 = arr1.astype("i")

# print(newarr1)  # [1 2 3]
# print(newarr1.dtype)  # int32

# arr2 = np.array([1, 0, 3])

# newarr2 = arr2.astype(bool)

# print(newarr2)  # [ True False  True]
# print(newarr2.dtype)  # bool

"""copy vs view"""

# # copy는 깊은 복사
# arr1 = np.array([1, 2, 3, 4, 5])
# x = arr1.copy()
# arr1[0] = 42

# print(arr1)  # [42  2  3  4  5]
# print(x)  # [1 2 3 4 5]

# # view는 얕은 복사
# arr2 = np.array([1, 2, 3, 4, 5])
# x = arr2.view()
# arr2[0] = 42

# print(arr2)  # [42  2  3  4  5]
# print(x)  # [42  2  3  4  5]

# arr3 = np.array([1, 2, 3, 4, 5])
# x = arr3.view()
# x[0] = 31

# print(arr3)  # [31  2  3  4  5]
# print(x)  # [31  2  3  4  5]

# arr = np.array([1, 2, 3, 4, 5])

# x = arr.copy()
# y = arr.view()

# print(x.base)  # None
# print(y.base)  # [1 2 3 4 5]

"""배열의 차원 형태"""

# arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

# print(arr.shape)  # (2, 4)

# arr = np.array([1, 2, 3, 4], ndmin=5)

# print(arr)  # [[[[[1 2 3 4]]]]]
# print("shape of array :", arr.shape)  # shape of array : (1, 1, 1, 1, 4)

"""배열의 차원 재정의"""

# arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

# newarr = arr.reshape(4, 3)

# print(newarr)
# """
# [[ 1  2  3]
#  [ 4  5  6]
#  [ 7  8  9]
#  [10 11 12]]
# """

# arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

# newarr = arr.reshape(2, 3, 2)

# # reshape는 view이다.
# print(newarr.base)  # [ 1  2  3  4  5  6  7  8  9 10 11 12]
# print(newarr)
# """
# [[[ 1  2]
#   [ 3  4]
#   [ 5  6]]

#  [[ 7  8]
#   [ 9 10]
#   [11 12]]]
# """

# arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# newarr = arr.reshape(2, 2, -1)

# print(newarr)
# """
# [[[1 2]
#   [3 4]]

#  [[5 6]
#   [7 8]]]
# """

# arr = np.array([[1, 2, 3], [4, 5, 6]])

# newarr = arr.reshape(-1)

# print(newarr)  # [1 2 3 4 5 6]

"""배열의 반복자"""

# # 1차원 배열의 데이터를 하나씩 반복
# arr = np.array([1, 2, 3])

# for x in arr:
#     print(x)

# # 2차원 배열의 데이터를 2차원 수준에서 반복
# arr = np.array([[1, 2, 3], [4, 5, 6]])

# for x in arr:
#     print(x)

# # 2차원 배열의 데이터를 1차원 수준에서 반복
# arr = np.array([[1, 2, 3], [4, 5, 6]])

# for x in arr:
#     for y in x:
#         print(y)

# # 3차원 배열의 데이터를 3차원 수준에서 반복
# arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

# for x in arr:
#     print(x)

# # 3차원 배열의 데이터를 1차원 수준에서 반복
# arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

# for x in arr:
#     for y in x:
#         for z in y:
#             print(z)

"""nditer() 메서드 활용"""

# arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# # 1차원 수준에서 반복
# for x in np.nditer(arr):
#     print(x)

# # 반복하는 동안 데이터 자료형을 op_dtypes에 정의한 자료형으로 변경
# # 실제 데이터 자료형을 수정하지 않고, flags에 정의된 버퍼에서 작업
# for x in np.nditer(arr, flags=["buffered"], op_dtypes=["S"]):
#     print(x)

# arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

# for x in np.nditer(arr[:, ::2]):
#     print(x)
# """
# 1
# 3
# 5
# 7
# """

# """ndenumerate() 메서드 활용"""

# arr = np.array([1, 2, 3])

# for idx, x in np.ndenumerate(arr):
#     print(idx, x)

# """
# (0,) 1
# (1,) 2
# (2,) 3
# """

# arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

# for idx, x in np.ndenumerate(arr):
#     print(idx, x)

# """
# (0, 0) 1
# (0, 1) 2
# (0, 2) 3
# (0, 3) 4
# (1, 0) 5
# (1, 1) 6
# (1, 2) 7
# (1, 3) 8
# """

"""배열의 결합"""

# arr1 = np.array([1, 2, 3])

# arr2 = np.array([4, 5, 6])

# arr = np.concatenate((arr1, arr2))

# print(arr)  # [1 2 3 4 5 6]

# arr1 = np.array([[1, 2], [3, 4]])

# arr2 = np.array([[5, 6], [7, 8]])

# # 축을 1로 명시
# # 축을 명시하지 않으면 기본적으로 0
# arr = np.concatenate((arr1, arr2), axis=1)

# print(arr)
# """
# [[1 2 5 6]
#  [3 4 7 8]]
# """

# arr1 = np.array([1, 2, 3])

# arr2 = np.array([4, 5, 6])

# arr = np.stack((arr1, arr2), axis=1)

# print(arr)
# """
# [[1 4]
#  [2 5]
#  [3 6]]
# """

# arr = np.hstack((arr1, arr2))

# print(arr)  # [1 2 3 4 5 6]

# arr = np.vstack((arr1, arr2))

# print(arr)
# """
# [[1 2 3]
#  [4 5 6]]
# """

# arr = np.dstack((arr1, arr2))

# print(arr)
# """
# [[[1 4]
#   [2 5]
#   [3 6]]]
# """

"""배열 splitting"""


# arr = np.array([1, 2, 3, 4, 5, 6])

# newarr = np.array_split(arr, 3)

# print(newarr)  # [array([1, 2]), array([3, 4]), array([5, 6])]

# # 배열에 필요한 것보다 요소의 수가 적으면 끝에서부터 조정
# newarr = np.array_split(arr, 4)

# print(newarr)  # [array([1, 2]), array([3, 4]), array([5]), array([6])]
# print(newarr[0])  # [1 2]
# print(newarr[1])  # [3 4]
# print(newarr[2])  # [5]

# arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])

# newarr = np.array_split(arr, 3)

# print(newarr)
# """
# [array([[1, 2],
#        [3, 4]]), array([[5, 6],
#        [7, 8]]), array([[ 9, 10],
#        [11, 12]])]
# """

# arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])

# newarr = np.array_split(arr, 3, axis=1)

# print(newarr)

# newarr = np.hsplit(arr, 3)

# print(newarr)

"""배열 탐색"""


# arr = np.array([1, 2, 3, 4, 5, 4, 4])

# x = np.where(arr == 4)

# print(x)  # (array([3, 5, 6]),). 인덱스 3, 5, 6에 값 4가 존재

# arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# x = np.where(arr % 2 == 0)  # 짝수 찾기

# print(x)  # (array([1, 3, 5, 7]),)

"""배열 이진 탐색"""


# arr = np.array([6, 7, 8, 9])

# # 배열에 대해 이진 탐색을 먼저 수행
# # 주어진 값이 들어갈 인덱스 자리 반환
# x = np.searchsorted(arr, 7)

# print(x)  # 1

# x = np.searchsorted(arr, 7, side="right")

# print(x)  # 2

# arr = np.array([1, 3, 5, 7])

# x = np.searchsorted(arr, [2, 4, 6])

# print(x)  # [1 2 3]

"""배열의 정렬"""


# arr = np.array([3, 2, 0, 1])

# print(np.sort(arr))  # [0 1 2 3]

# arr = np.array(["banana", "cherry", "apple"])

# print(np.sort(arr))  # ["apple" "banana" "cherry"]

# arr = np.array([True, False, True])

# print(np.sort(arr))  # [False True True]

# arr = np.array([[3, 2, 4], [5, 0, 1]])

# print(np.sort(arr))
# """
# [[2 3 4]
#  [0 1 5]]
# """

"""배열의 필터링"""

arr = np.array([41, 42, 43, 44])

x = [True, False, True, False]

newarr = arr[x]

print(newarr)  # [41 43]

# 필터링 배열 생성해서 사용하기

filter_arr = []

for element in arr:
    if element > 42:
        filter_arr.append(True)
    else:
        filter_arr.append(False)

newarr = arr[filter_arr]

print(filter_arr)  # [False, False, True, True]
print(newarr)  # [43 44]

# 직접 필터 적용하기

filter_arr = arr > 42

newarr = arr[filter_arr]

print(filter_arr)  # [False, False, True, True]
print(newarr)  # [43 44]
