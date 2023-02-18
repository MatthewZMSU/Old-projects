import sys
import numpy as np
import matrix_methods as mm


if __name__ == "__main__":
    # Get matrix:
    input_type = int(input('''Type the type number of input:
    1 -  get matrix from keyboard
    2 - get matrix from file
    3 - get matrix using formulas\n'''))
    if input_type == 1:
        matrix_size, matrix = mm.get_matrix_keyboard()
    elif input_type == 2:
        file_name = input("Type file path: ")
        matrix_size, matrix = mm.get_matrix_file(file_name)
    elif input_type == 3:
        matrix_size = int(input("Type the size of matrix: "))
        print("Formulas have to include only i (rows), j (columns) and n (matrix_size)")
        matrix_el_main_formula = input("Type formula for matrix diagonal elements (only left side): ")
        matrix_el_sub_formula = input("Type formula for matrix non-diagonal elements (only left side): ")
        side_el_formula = input("Type formula for right side vector: ")
        matrix = mm.get_matrix_formula(matrix_el_main_formula, matrix_el_sub_formula, side_el_formula, matrix_size)
    else:
        sys.exit("Not valid argument!")
    # Some information about the matrix:
    print('Your matrix has determinant:')
    print(np.linalg.det(matrix[:, :matrix_size]))
    print('The inverse matrix is:')
    inv_matrix = np.linalg.inv(matrix[:, :matrix_size])
    mm.print_matrix(inv_matrix)
    print('The number of conditionality:')
    print(np.linalg.norm(matrix[:, :matrix_size]) * np.linalg.norm(inv_matrix))
    # Solution by gauss elimination
    print("Solution using Gauss elimination:")
    gauss_solution = np.zeros(matrix_size, float)
    mm.gauss_elimination(matrix.copy(), matrix_size, gauss_solution)
    mm.print_solution(gauss_solution, matrix_size)
    # Solution by gauss elimination with main element:
    print("Solution by Gauss elimination with main element:")
    main_element_solution = np.zeros(matrix_size, float)
    mm.gauss_elimination_with_main_element(matrix.copy(), matrix_size, main_element_solution)
    mm.print_solution(main_element_solution, matrix_size)
    print("Solution by upper relaxation method:")
    omega = float(input("Type omega value (0;2): "))
    eps = float(input("Type epsilon value (>0): "))
    upper_relaxation_solution = np.zeros(matrix_size, float)
    upper_relaxation_iterations = mm.upper_relaxation_method(matrix, matrix_size, omega, eps, upper_relaxation_solution)
    mm.print_solution(upper_relaxation_solution, matrix_size)
    print(f'It took {upper_relaxation_iterations} iterations to calculate the solution')
