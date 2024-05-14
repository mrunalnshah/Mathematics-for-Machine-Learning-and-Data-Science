import numpy as np
from numpy import array

def test_row_echelon_form(target_function):

    successful_cases = 0
    failed_cases = []

    test_cases = [{'name': 'check_null_matrix',
  'input': {'A': array([[0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]]),
   'B': array([[0],
          [0],
          [0]])},
  'expected': 'Singular system'},
 {'name': 'check_identity_matrix',
  'input': {'A': array([[1., 0., 0.],
          [0., 1., 0.],
          [0., 0., 1.]]),
   'B': array([[0.],
          [0.],
          [0.]])},
  'expected': array([[1., 0., 0., 0.],
         [0., 1., 0., 0.],
         [0., 0., 1., 0.]])},
 {'name': 'check_matrix_1',
  'input': {'A': array([[1., 2., 3.],
          [5., 0., 2.],
          [1., 4., 5.]]),
   'B': array([[4.],
          [3.],
          [6.]])},
  'expected': array([[ 1.        ,  2.        ,  3.        ,  4.        ],
         [-0.        ,  1.        ,  1.3       ,  1.7       ],
         [-0.        , -0.        ,  1.        ,  2.33333333]])},
 {'name': 'check_matrix_2',
  'input': {'A': array([[ 1.,  5.,  6.],
          [ 3.,  1.,  4.],
          [ 2., -4., -2.]]),
   'B': array([[ 9.],
          [ 4.],
          [-5.]])},
  'expected': 'Singular system'}]

    successful_cases = 0
    failed_cases = []
    for test_case in test_cases:

        try:
            target_output = target_function(**test_case["input"])
        except Exception as e:
            print("\033[91m", f'An exception was thrown while running your function: {e}.\nInput matrix:\n{test_case["input"]}')
            return


        try:
            if isinstance(test_case['expected'],str):
                assert isinstance(target_output, str)
            else:
                assert np.allclose(target_output, test_case['expected'])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": target_output,
                }
            )
            print(
                f"Wrong output for test case {test_case['name']}. \n\tExpected:\n\t {failed_cases[-1].get('expected')}.\n\tGot:\n {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_back_substitution(target_function):

    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "check_null_matrix",
            "input": np.array(
                    [
                    [1, 0, 0, 5],
                    [0, 1, 0, 6],
                    [0, 0, 1, 7]
                    ]
                ),
            "expected": np.array([5, 6, 7])
        },
        {
            "name": "check_matrix_1",
            "input": np.array([
                [ 1.        ,  2.        ,  3.        ,  4.        ],
                [-0.        ,  1.        ,  1.3       ,  1.7       ],
                [-0.        , -0.        ,  1.        ,  2.33333333]]),
            "expected": np.array([-0.33333333, -1.33333333,  2.33333333])
        },

        {
            "name": "check_matrix_2",
            "input": np.array([
                [ 1.        ,  5.        ,  6.        ,  9.        ],
                [-0.        ,  1.        ,  1.        ,  1.64285714],
                [ 0.        ,  0.        ,  1.        ,  0.        ]]),
            "expected": np.array([0.7857143 , 1.64285714, 0.        ])
        },
        {
            "name": "check_matrix_3",
            "input": np.array([
                [ 1.        ,  8.        ,  6.        ,  9.        ],
                [0.        ,  1        ,  8        ,  6],
                [ 0.        ,  0.        ,  1.        ,  1.        ]]),
            "expected": np.array([19., -2.,  1.])
        }
    ]


    for test_case in test_cases:

        try:
            target_output = target_function(test_case["input"])
        except Exception as e:
            print("\033[91m", f'An exception was thrown while running your function: {e}.\nInput matrix:\n{test_case["input"]}')
            return


        try:
            assert np.allclose(target_output, test_case['expected'])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": target_output,
                }
            )
            print(
                f"Wrong output for test case {test_case['name']}. \n\tExpected:\n\t {failed_cases[-1].get('expected')}.\n\tGot:\n {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

def test_gaussian_elimination(target_function):

    successful_cases = 0
    failed_cases = []

    test_cases = [{'name': 'test_matrix_0',
  'input': {'A': array([[-9, -1, -5],
          [-9, -9, -8],
          [ 0, -3, -8]]),
   'B': array([[1],
          [4],
          [8]])},
  'expected': array([ 0.44444444,  0.        , -1.        ])},
 {'name': 'test_matrix_1',
  'input': {'A': array([[ 1,  7, -5, -8],
          [ 0, -5, -1,  9],
          [ 6, -6,  0,  5],
          [ 7, -6,  8,  9]]),
   'B': array([[-8],
          [ 4],
          [ 9],
          [ 5]])},
  'expected': array([ 0.0279965 , -2.07567804, -0.14129484, -0.72440945])},
 {'name': 'test_matrix_2',
  'input': {'A': array([[ -9,  -9,  -2,   3, -10],
          [  6,   5,   6,   2,   7],
          [ -6,  -2,   5,   5,   7],
          [ -4,  -5,  -6,  -9,   4],
          [  3,  -6,  -2,  -8,  -8]]),
   'B': array([[ 3],
          [-4],
          [ 1],
          [-2],
          [-3]])},
  'expected': array([-2.24012291,  2.29784899,  1.39727831, -1.46641791, -1.0713345 ])},
 {'name': 'test_matrix_3',
  'input': {'A': array([[ -8,  -9, -10,  -9,   7,  -4],
          [  2,   8,   8,   3,  -8,  -2],
          [ -4,   0,   3,   0,   2,   9],
          [  2,  -6,  -8,  -2,  -8,   9],
          [ -2,  -1,  -3,   3,   4,  -3],
          [-10,   1,  -5,  -6,   3,  -4]]),
   'B': array([[ 1],
          [-6],
          [-4],
          [-1],
          [-1],
          [ 9]])},
  'expected': array([ 1.66947817,  3.32226171, -2.22881748, -1.60512025,  1.48105316,
          0.71136209])},
 {'name': 'test_matrix_4',
  'input': {'A': array([[-10,   8,  -8,  -6,   4,   2,  -2],
          [  3,   1,   6,   2,  -3,  -5, -10],
          [  4,   9,  -9,   1,   2,  -9,  -7],
          [ -2,  -9,   1,   9,   6,  -2,  -2],
          [ -3,  -5, -10,   9,   4,   1, -10],
          [  6,   8,   7,  -3,   6,  -6,   8],
          [  4,  -8,   5,  -4,   1,   3,   6]]),
   'B': array([[ 0],
          [-5],
          [ 2],
          [-6],
          [ 5],
          [ 4],
          [-2]])},
  'expected': array([ 1.28049215,  1.12547489, -0.17505018,  0.74172351,  0.64510371,
          1.98786815, -0.14745546])}]

    for test_case in test_cases:

        try:
            target_output = target_function(**test_case["input"])
        except Exception as e:
            print("\033[91m", f'An exception was thrown while running your function: {e}.\nInput matrix:\n{test_case["input"]}')
            return


        try:
            assert np.allclose(target_output, test_case['expected'])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": target_output,
                }
            )
            print(
                f"Wrong output for test case {test_case['name']}. \n\tExpected:\n\t {failed_cases[-1].get('expected')}.\n\tGot:\n {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")