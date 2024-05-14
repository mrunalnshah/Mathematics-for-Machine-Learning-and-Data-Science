import numpy as np

def test_matrix(target_P, target_X0, target_X1):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "expected": {
                "P": np.array(
                    [
                        [0, 0.75, 0.35, 0.25, 0.85],
                        [0.15, 0, 0.35, 0.25, 0.05],
                        [0.15, 0.15, 0, 0.25, 0.05],
                        [0.15, 0.05, 0.05, 0, 0.05],
                        [0.55, 0.05, 0.25, 0.25, 0],
                    ]
                ),
                "X0": np.array([[0], [0], [0], [1], [0]]),
            },
        },
    ]

    for test_case in test_cases:

        try:
            assert target_P.shape == test_case["expected"]["P"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["P"].shape,
                    "got": target_P.shape,
                }
            )
            print(
                f"Wrong shape of matrix P. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            break

        try:
            assert np.allclose(np.diagonal(target_P), np.diagonal(test_case["expected"]["P"]))
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": np.nonzero(
                        np.logical_not(np.isclose(np.diagonal(target_P), np.diagonal(test_case["expected"]["P"])))
                    ),
                    "got": sum(target_P),
                }
            )
            print(
                f"Wrong matrix P. \nCheck the diagonal elements."
            )
            
        try:
            assert np.allclose(sum(target_P), sum(test_case["expected"]["P"]))
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": np.nonzero(
                        np.logical_not(np.isclose(sum(target_P), sum(test_case["expected"]["P"])))
                    ),
                    "got": sum(target_P),
                }
            )
            print(
                f"Wrong matrix P. \nCheck the elements in the column {failed_cases[-1].get('expected')[0][0] + 1}."
            )

        try:
            assert target_X0.shape == test_case["expected"]["X0"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["X0"].shape,
                    "got": target_X0.shape,
                }
            )
            print(
                f"Wrong shape of vector X0. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            break

        try:
            assert np.allclose(target_X0, test_case["expected"]["X0"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": np.nonzero(
                        np.logical_not(
                            np.isclose(target_X0, test_case["expected"]["X0"])
                        )
                    ),
                    "got": target_X0,
                }
            )
            print(
                f"Wrong array X0.\nCheck element {failed_cases[-1].get('expected')[0][0] + 1} in the vector X0."
            )

        expected_X1 = np.matmul(target_P, target_X0)
            
        try:
            assert target_X1.shape == expected_X1.shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": expected_X1.shape,
                    "got": target_X1.shape,
                }
            )
            print(
                f"Wrong shape of vector X1. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            break

        try:
            assert np.allclose(target_X1, expected_X1)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": expected_X1,
                    "got": target_X1,
                }
            )
            print(
                f"Wrong vector X1. Check if matrix multiplication was performed correctly."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_check_eigenvector(target_T):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "P": np.array(
                    [
                        [0, 0.75, 0.35, 0.25, 0.85],
                        [0.15, 0, 0.35, 0.25, 0.05],
                        [0.15, 0.15, 0, 0.25, 0.05],
                        [0.15, 0.05, 0.05, 0, 0.05],
                        [0.55, 0.05, 0.25, 0.25, 0],
                    ]
                ),
            },
        },
        {"name": "extra_check", "input": {"P": np.array([[2, 3], [2, 1],]),},},
    ]

    for test_case in test_cases:

        X_inf = np.linalg.eig(test_case["input"]["P"])[1][:, 0]

        target_X_check = target_T(test_case["input"]["P"], X_inf)
        expected_X_check = test_case["input"]["P"] @ X_inf

        try:
            assert target_X_check.shape == expected_X_check.shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": expected_X_check.shape,
                    "got": target_X_check.shape,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong shape of output matrix in the check_eigenvector function. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            break

        try:
            assert np.allclose(target_X_check, expected_X_check)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": expected_X_check,
                    "got": target_X_check,
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output matrix in the check_eigenvector function. Check if matrix multiplication was performed correctly."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

def test_center_data(target_center_data):
    successful_cases = 0
    failed_cases = []
    
    test_cases = [{'name': 'default_check',
                    'input': np.load('./support_files/imgs_flatten.npy'),
                    'expected': np.load('./support_files/expected_centered_data.npy')
                    },
                    ]
       
    for test_case in test_cases:
        try:
            target_result = target_center_data(test_case['input'])

        except Exception as e:
            print(
                f"There was an error evaluating the function. \nError: {e}")
            return
            
        try: 
            assert isinstance(target_result, np.ndarray)
            successful_cases += 1
        except:
            failed_cases.append(
            {
                "name": test_case["name"],
                "expected": np.ndarray,
                "got": type(target_result),
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output type on the center_data function. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            break

            
        try:
            assert np.allclose(target_result, test_case['expected'], atol=1e-2)
            successful_cases += 1
        except:
            failed_cases.append(
            {
                "name": test_case["name"],
                "expected": test_case['expected'],
                "got": target_result,
            })
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output on the center_data function. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")
        
def test_cov_matrix(target_cov_mat):
    successful_cases = 0
    failed_cases = []
    test_cases = [{'name': 'default_check',
                    'input': np.load('./support_files/expected_centered_data.npy'),
                    'expected': np.load('./support_files/expected_cov_mat.npy')
                    },
                  ]
   
    for test_case in test_cases:
        try:
            target_result = target_cov_mat(test_case['input'])
        except Exception as e:
            print(f"There was an error evaluating the function. \nError: {e}")
            return
        
        try:
            assert isinstance(target_result, np.ndarray)
            successful_cases += 1
        except:
            
            failed_cases.append(
                {
                'name': test_case["name"],
                'expected': np.ndarray,
                'got': type(target_result)
                
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output type on the get_cov_matrix function. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}.")
            break

        try:
            assert target_result.shape == test_case['expected'].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    'name': test_case["name"],
                    'expected': test_case['expected'].shape,
                    'got': target_result.shape
                }
            )
            print(
                    f"Test case \"{failed_cases[-1].get('name')}\". Wrong output shape on the get_cov_matrix function. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}.")

        try:
            assert np.allclose(target_result, test_case['expected'])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    'name': test_case["name"],
                    'expected': test_case['expected'],
                    'got': target_result
                }
            )
            print(
                    f"Test case \"{failed_cases[-1].get('name')}\". Wrong output on the get_cov_matrix function. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}.")
    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")
        
                
def test_check_PCA(target_pca):
    successful_cases = 0
    failed_cases = []
    test_cases = [{'name':'default_check_2',
                    'input': {'X':np.load('./support_files/expected_centered_data.npy'), 
                                    'eigenvecs': np.load('./support_files/expected_eigvecs.npy'),
                                    'k': 2},
                    'expected': np.load('./support_files/expected_pca2.npy')
                    },

                    {'name': 'default_check_12',
                    'input': {'X':np.load('./support_files/expected_centered_data.npy'), 
                            'eigenvecs': np.load('./support_files/expected_eigvecs.npy'),
                            'k': 12},
                     'expected': np.load('./support_files/expected_pca12.npy')
                    },]
        
    
    for test_case in test_cases:
        try:
            target_result = target_pca(**test_case['input'])
        except Exception as e:
            print(f"There was an error evaluating the function. \nError: {e}")
            return
        
        try: 
            assert isinstance(target_result, np.ndarray)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                'name': test_case['name'],
                'expected': np.ndarray,
                'got': type(target_result),
                }
            )
            print(
                    f"Test case \"{failed_cases[-1].get('name')}\". Wrong output type on the perform_PCA function. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}.")
                
        try:
            assert target_result.shape == test_case['expected'].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                'name': test_case['name'],
                'expected': test_case['expected'].shape,
                'got': target_result.shape
                }
            )
            print(
                f"Test case \"{failed_cases[-1].get('name')}\". Wrong output shape on the perform_PCA function. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}.")
                
        try:
            assert np.allclose(target_result, test_case['expected'], atol=1e-8)
            successful_cases += 1
        except:
            failed_cases.append(
            {
                'name': test_case['name'],
                'expected': test_case['expected'],
                'got': target_result
            }
            )
            print(
                        f"Test case \"{failed_cases[-1].get('name')}\". Wrong output on the perform_PCA function. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}.")
                
    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")
        