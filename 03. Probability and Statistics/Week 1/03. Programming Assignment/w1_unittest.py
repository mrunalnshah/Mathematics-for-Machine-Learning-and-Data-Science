import numpy as np
import pandas as pd
import joblib




def test_get_word_frequency(target):
    successful_cases = 0
    failed_cases = []

    test_cases = joblib.load("test_cases_get_word_frequency.joblib")

    for test_case in test_cases:

        output = target(**test_case['input'])
        expected = test_case['expected']

        ## Check same length
        if len(output) != len(expected):
            failed_cases.append({
                "name":test_case["name"],
                "expected":len(expected),
                "got":len(output)})
            print(f"Wrong output dictionary size for {failed_cases[-1]['name']}. Expected: {failed_cases[-1]['expected']}. Got: {failed_cases[-1]['got']}")
            continue

        ## Check if keys are the same
        keys_truth = set(expected.keys())
        keys_output = set(output.keys())
        if not (keys_truth.issubset(keys_output) and keys_output.issubset(keys_truth)):
            failed_cases.append({
                "name":test_case["name"],
                "expected":keys_truth,
                "got":keys_output})
            print(f"Wrong output dictionary keys for {failed_cases[-1]['name']}. Expected: {failed_cases[-1]['expected']}. Got: {failed_cases[-1]['got']}")
            continue      

        ## Check if, for every key, the counting is the same
        for key in output.keys():
            if len(output[key]) != len(expected[key]):
                failed_cases.append({
                "name":test_case["name"],
                "expected":len(expected[key]),
                "got":len(output[key])})
                print(f"Wrong output dictionary size for word {key} in {failed_cases[-1]['name']}. Expected: {failed_cases[-1]['expected']}. Got: {failed_cases[-1]['got']}")
                break

            if output[key] != expected[key]:
                failed_cases.append({
                "name":test_case["name"],
                "expected":expected[key],
                "got":output[key]})
                print(f"Wrong counting for word {key} in {failed_cases[-1]['name']}. Expected: {failed_cases[-1]['expected']}. Got: {failed_cases[-1]['got']}")
                break
        successful_cases+=1

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")                                   

def test_prob_word_given_class(target, word_frequency, class_frequency):

    successful_cases = 0
    failed_cases = []

    test_cases = [
        {'name': 'test_case_0',
  'input': {'word': 'compile', 'cls': 'ham'},
  'expected': 0.0025951557093425604},
 {'name': 'test_case_1',
  'input': {'word': 'doc', 'cls': 'spam'},
  'expected': 0.0026929982046678637},
 {'name': 'test_case_2',
  'input': {'word': 'nights', 'cls': 'ham'},
  'expected': 0.003748558246828143},
 {'name': 'test_case_3',
  'input': {'word': 'attached', 'cls': 'spam'},
  'expected': 0.008976660682226212},
 {'name': 'test_case_4',
  'input': {'word': 'hook', 'cls': 'ham'},
  'expected': 0.0025951557093425604},
 {'name': 'test_case_5',
  'input': {'word': 'projector', 'cls': 'spam'},
  'expected': 0.0017953321364452424},
 {'name': 'test_case_6',
  'input': {'word': 'also', 'cls': 'ham'},
  'expected': 0.2577854671280277},
 {'name': 'test_case_7',
  'input': {'word': 'requirements', 'cls': 'spam'},
  'expected': 0.018850987432675045},
 {'name': 'test_case_8',
  'input': {'word': 'dietary', 'cls': 'ham'},
  'expected': 0.0008650519031141869},
 {'name': 'test_case_9',
  'input': {'word': 'equipment', 'cls': 'spam'},
  'expected': 0.02064631956912029},
 {'name': 'test_case_10',
  'input': {'word': 'staying', 'cls': 'ham'},
  'expected': 0.008362168396770472},
 {'name': 'test_case_11',
  'input': {'word': 'find', 'cls': 'spam'},
  'expected': 0.09425493716337523},
 {'name': 'test_case_12',
  'input': {'word': 'reserve', 'cls': 'ham'},
  'expected': 0.019319492502883506},
 {'name': 'test_case_13',
  'input': {'word': 'several', 'cls': 'spam'},
  'expected': 0.04039497307001795},
 {'name': 'test_case_14',
  'input': {'word': 'university', 'cls': 'ham'},
  'expected': 0.13408304498269896},
 {'name': 'test_case_15',
  'input': {'word': 'shirley', 'cls': 'spam'},
  'expected': 0.0017953321364452424},
 {'name': 'test_case_16',
  'input': {'word': 'ca', 'cls': 'ham'},
  'expected': 0.03460207612456748},
 {'name': 'test_case_17',
  'input': {'word': 'enron', 'cls': 'spam'},
  'expected': 0.0008976660682226212},
 {'name': 'test_case_18',
  'input': {'word': 'thanks', 'cls': 'ham'},
  'expected': 0.41205305651672436},
 {'name': 'test_case_19',
  'input': {'word': 'soon', 'cls': 'spam'},
  'expected': 0.04039497307001795}
  ]

    for test_case in test_cases:

        output = target(word_frequency = word_frequency, class_frequency = class_frequency, **test_case['input'])
        expected = test_case['expected']

        if not np.isclose(output,expected):
            failed_cases.append(
                {
                    "name":test_case['name'],
                    "expected":expected,
                    "got":output
                }

            )
            print(f"Wrong value for P({test_case['input']['word']} | spam = {test_case['input']['spam']}) in {failed_cases[-1]['name']}. Expected: {failed_cases[-1]['expected']}. Got: {failed_cases[-1]['got']}")
            continue
        successful_cases+=1
    
    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")                                   
       

    
def test_prob_email_given_class(target, word_frequency, class_frequency):

    successful_cases = 0
    failed_cases = []

    test_cases = [{'name': 'test_case_0',
  'input': {'treated_email': ['ca',
    'enron',
    'find',
    'projector',
    'compile',
    'find',
    'attached',
    'staying',
    'soon'],
   'cls': 'ham'},
  'expected': 3.3983894210489835e-13},
 {'name': 'test_case_1',
  'input': {'treated_email': ['doc',
    'soon',
    'university',
    'nights',
    'attached',
    'nights',
    'equipment',
    'hook'],
   'cls': 'spam'},
  'expected': 7.069258091965318e-19},
 {'name': 'test_case_2',
  'input': {'treated_email': ['thanks',
    'ca',
    'university',
    'enron',
    'university',
    'several'],
   'cls': 'ham'},
  'expected': 9.77525599231039e-06},
 {'name': 'test_case_3',
  'input': {'treated_email': ['projector', 'also', 'also', 'ca', 'hook'],
   'cls': 'spam'},
  'expected': 6.013747036672614e-10},
 {'name': 'test_case_4',
  'input': {'treated_email': ['dietary',
    'find',
    'thanks',
    'staying',
    'shirley',
    'dietary',
    'attached',
    'thanks'],
   'cls': 'ham'},
  'expected': 3.4470299679032404e-12},
 {'name': 'test_case_5',
  'input': {'treated_email': ['several',
    'find',
    'staying',
    'staying',
    'also',
    'ca',
    'university',
    'equipment'],
   'cls': 'spam'},
  'expected': 4.397549817075224e-15},
 {'name': 'test_case_6',
  'input': {'treated_email': ['projector',
    'reserve',
    'attached',
    'staying',
    'university',
    'hook',
    'staying',
    'dietary'],
   'cls': 'ham'},
  'expected': 2.717031873714673e-16},
 {'name': 'test_case_7',
  'input': {'treated_email': ['thanks',
    'attached',
    'thanks',
    'equipment',
    'also',
    'staying',
    'several',
    'staying'],
   'cls': 'spam'},
  'expected': 4.973982358381553e-15},
 {'name': 'test_case_8',
  'input': {'treated_email': ['compile',
    'dietary',
    'requirements',
    'shirley',
    'several',
    'nights',
    'doc',
    'hook',
    'thanks'],
   'cls': 'ham'},
  'expected': 2.698275888443421e-16},
 {'name': 'test_case_9',
  'input': {'treated_email': ['enron',
    'hook',
    'staying',
    'staying',
    'doc',
    'equipment'],
   'cls': 'spam'},
  'expected': 1.444102224707258e-16}]
    
    for test_case in test_cases:

        got = target(word_frequency = word_frequency, class_frequency = class_frequency, **test_case["input"])
        expected = test_case['expected']
        
        if not np.isclose(got, expected):
            failed_cases.append(
                {
                    "name":test_case['name'],
                    "expected":expected,
                    "got":got
                }

            )
            print(f"Wrong value for email = {test_case['input']['treated_email']} and spam = {test_case['input']['spam']} in {failed_cases[-1]['name']}. Expected: {failed_cases[-1]['expected']}. Got: {failed_cases[-1]['got']}")
            continue
        successful_cases+=1
    
    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")                                   
       

def test_naive_bayes(target, word_frequency, class_frequency):
    successful_cases = 0
    failed_cases = []

    test_cases =[{'name': 'test_case_0',
  'input': {'treated_email': ['ca',
    'enron',
    'find',
    'projector',
    'compile',
    'find',
    'attached',
    'staying',
    'soon']},
  'expected': 0},
 {'name': 'test_case_1',
  'input': {'treated_email': ['doc',
    'soon',
    'university',
    'nights',
    'attached',
    'nights',
    'equipment',
    'hook']},
  'expected': 0},
 {'name': 'test_case_2',
  'input': {'treated_email': ['thanks',
    'ca',
    'university',
    'enron',
    'university',
    'several']},
  'expected': 0},
 {'name': 'test_case_3',
  'input': {'treated_email': ['projector', 'also', 'also', 'ca', 'hook']},
  'expected': 0},
 {'name': 'test_case_4',
  'input': {'treated_email': ['dietary',
    'find',
    'thanks',
    'staying',
    'shirley',
    'dietary',
    'attached',
    'thanks']},
  'expected': 0},
 {'name': 'test_case_5',
  'input': {'treated_email': ['several',
    'find',
    'staying',
    'staying',
    'also',
    'ca',
    'university',
    'equipment']},
  'expected': 0},
 {'name': 'test_case_6',
  'input': {'treated_email': ['projector',
    'reserve',
    'attached',
    'staying',
    'university',
    'hook',
    'staying',
    'dietary']},
  'expected': 0},
 {'name': 'test_case_7',
  'input': {'treated_email': ['thanks',
    'attached',
    'thanks',
    'equipment',
    'also',
    'staying',
    'several',
    'staying']},
  'expected': 0},
 {'name': 'test_case_8',
  'input': {'treated_email': ['compile',
    'dietary',
    'requirements',
    'shirley',
    'several',
    'nights',
    'doc',
    'hook',
    'thanks']},
  'expected': 0},
 {'name': 'test_case_9',
  'input': {'treated_email': ['enron',
    'hook',
    'staying',
    'staying',
    'doc',
    'equipment']},
  'expected': 0}]
    
    for test_case in test_cases:

        got = target(word_frequency = word_frequency, class_frequency = class_frequency, **test_case["input"])
        expected = test_case['expected']
        
        if not np.isclose(got, expected):
            failed_cases.append(
                {
                    "name":test_case['name'],
                    "expected":expected,
                    "got":got
                }

            )
            print(f"Wrong decision for email = {test_case['input']['treated_email']} in {failed_cases[-1]['name']}. Expected: {failed_cases[-1]['expected']}. Got: {failed_cases[-1]['got']}")
            continue
        successful_cases+=1
    
    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")                                   
       

