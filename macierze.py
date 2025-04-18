from typing import List 
from typing import Tuple
from typing import Callable

Matrix = List[List[float]]
Vector = List[float]

A = [[1, 2, 3],
     [4, 5, 6]]

B = [[1,2],
     [3,4],
     [5,6]]

def shape(A: Matrix) -> Tuple[int, int]:
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows,num_cols

# print(shape(A))

def get_row(A: Matrix, i: int) -> Vector:
    return A[i]

# print(get_row(A,0))

def get_col(A:Matrix, j: int) -> Vector:
    return [A_i[j]
            for A_i in A]

# print(get_col(A , 2))

def make_matrix(num_rows: int,
                num_cols: int,
                entry_fn: Callable[[int,int], float]) -> Matrix:
    return[[entry_fn(i,j)
            for j in range(num_cols)]
        for i in range(num_rows)]

def identity_matriox(n: int) -> Matrix:
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)

printed_matrix = identity_matriox(5)
# for row in printed_matrix:
    # print(row)


# x = [[(i+1,j+1) for j in range(5)]for i in range(5)]
# print(x)

# typ = [[print(type(z))
#        for z in c]
#             for c in x]

xs = [1,2,5,2,6,7,3,2]

def mean(xs: List[float]) -> float:
    return sum(xs)/len(xs)

def data_range(xs: List[float]) ->  float:
    return max(xs) - min(xs)

def de_mean(xs: List[float]) -> List[float]:
    x_bar = mean(xs)
    return [x - x_bar for x in xs]

def variance(xs: List[float])-> float:
    n = len(xs)
    deviation = de_mean(xs)
    # return sum_ofs

macierz = [[1,2,3],
               [1,2,3],
               [1,2,3],
               [2,3,4]]
    
def wymiar(x:Matrix) -> Tuple:
    num_rows = len(macierz)
    num_cols = len(macierz[0])
    
    return num_cols, num_rows

print(wymiar(macierz))
