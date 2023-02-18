import sys
import numpy as np
from numpy import exp, cos, sin


def __balance_row(matrix: np.ndarray, matrix_size: int, row: int):
    if matrix[row, row] == 0.0:
        for index in range(row, matrix_size):
            if matrix[index, row] != 0.0:
                matrix[[row, index]] = matrix[[index, row]]
                break
    matrix[row] /= matrix[row, row]
    for index in range(row + 1, matrix_size):
        matrix[index, :] -= matrix[index, row] * matrix[row, :]


def __calc_solution(matrix: np.ndarray, matrix_size: int, solution: np.ndarray):
    for diag_ind in range(matrix_size - 1, -1, -1):
        solution[diag_ind] = matrix[diag_ind, matrix_size]
        for col in range(diag_ind + 1, matrix_size):
            solution[diag_ind] -= matrix[diag_ind, col] * solution[col]


def get_matrix_keyboard() -> tuple:
    matrix_size = int(input("Type the size of matrix: "))

    # n * (n + 1) - rows * columns
    matrix = np.zeros((matrix_size, matrix_size + 1), float)

    print("Type matrix values, for example:\n", "1 2.4 -2\n", "0 22 2233333\n", "123 123 123", sep='')
    for row in range(matrix_size):
        matrix[row, :matrix_size] = [float(j) for j in input().split()]

    if (np.abs(np.linalg.det(matrix[:, :matrix_size]))) <= 1e-5:
        sys.exit("Matrix with zero determinant")

    print("Type right side of the equation, for example:\n", "1\n", "2.2\n", "-4.2", sep='')
    for row in range(matrix_size):
        matrix[row, matrix_size] = float(input())

    return matrix_size, matrix


def get_matrix_file(file_name: str) -> tuple:
    with open(file_name, 'r') as input_file:
        matrix_size = int(input_file.readline().strip('\n'))
        matrix = np.zeros((matrix_size, matrix_size + 1), float)
        for row in range(matrix_size):
            matrix[row, :matrix_size] = [float(j) for j in input_file.readline().strip('\n').split()]
        for row in range(matrix_size):
            matrix[row, matrix_size] = float(input_file.readline().strip('\n'))
        if (np.abs(np.linalg.det(matrix[:, :matrix_size]))) <= 1e-5:
            sys.exit("Matrix with zero determinant")
        return matrix_size, matrix


def get_matrix_formula(matrix_el_main_formula: str, matrix_el_sub_formula: str, side_el_formula: str, matrix_size: int) -> np.ndarray:
    matrix = np.zeros((matrix_size, matrix_size + 1), float)
    for i in range(1, matrix_size + 1):
        for j in range(1, matrix_size + 1):
            n = matrix_size
            if i == j:
                matrix[i - 1][j - 1] = eval(matrix_el_main_formula)
            else:
                matrix[i - 1][j - 1] = eval(matrix_el_sub_formula)
    for i in range(1, matrix_size + 1):
        n = matrix_size
        matrix[i - 1, matrix_size] = eval(side_el_formula)
    if (np.abs(np.linalg.det(matrix[:, :matrix_size]))) <= 1e-7:
        sys.exit("Matrix with zero determinant")
    return matrix


def gauss_elimination(matrix: np.ndarray, matrix_size: int, solution: np.ndarray):
    for diag_ind in range(matrix_size):
        __balance_row(matrix, matrix_size, diag_ind)
    __calc_solution(matrix, matrix_size, solution)


def gauss_elimination_with_main_element(matrix: np.ndarray, matrix_size: int, solution: np.ndarray):
    swap_indices = tuple()
    for diag_ind in range(matrix_size):
        max_index = np.abs(matrix[diag_ind, :matrix_size]).argmax()
        matrix[:, [diag_ind, max_index]] = matrix[:, [max_index, diag_ind]]
        swap_indices += ((diag_ind, max_index), )
        __balance_row(matrix, matrix_size, diag_ind)
    __calc_solution(matrix, matrix_size, solution)
    for indices in swap_indices[::-1]:
        solution[[indices[0], indices[1]]] = solution[[indices[1], indices[0]]]


def upper_relaxation_method(matrix: np.ndarray, matrix_size: int, omega: float, eps: float, solution: np.ndarray) -> int:
    first = np.zeros((matrix_size, ), float)
    last = np.zeros(matrix_size, float)
    for index in range(matrix_size):
        first[index] = 1
    num_iter = 0
    while np.linalg.norm(last - first) >= eps:
        first = last.copy()
        num_iter += 1
        for index in range(matrix_size):
            shift = matrix[index, matrix_size]
            for i in range(0, index):
                shift -= matrix[index, i] * last[i]
            for i in range(index, matrix_size):
                shift -= matrix[index, i] * first[i]
            shift *= omega / matrix[index, index]
            last[index] = shift + first[index]
    for i in range(matrix_size):
        solution[i] = last[i]
    return num_iter


def print_matrix(matrix: np.ndarray):
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            print(matrix[row][col], end=' ')
        print()


def print_solution(solution: np.ndarray, matrix_size: int):
    for index in range(matrix_size):
        print(f'x{index + 1} = {solution[index]:.4f}')
