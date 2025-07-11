'''
This file handles loading of the following datasets 
- medicine: USMLE MCQs from BEA 2024 Shared Task Data (BEA_2024.csv)
- math: MCQs from the NeurIPS 2020 Education Challenge (NeurIPS_2020.csv)
- trivial: MCQs from the OpenTriviaQA (for-kids.txt, science-technology.txt)
'''

import csv
import os
import numpy as np
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_BEA_2024 = os.path.join(BASE_DIR,'raw', 'BEA_2024.csv')
PATH_TRIVIAL_EASY = os.path.join(BASE_DIR,'raw', 'trivial', 'for-kids.txt')
PATH_TRIVIAL_HARD = os.path.join(BASE_DIR,'raw', 'trivial', 'science-technology.txt')
PATH_MATH = os.path.join(BASE_DIR,'raw', 'math', 'NeurIPS_2020.csv')
PATH_MATH_TRAIN_1 = os.path.join(BASE_DIR,'raw', 'math', 'train_task_1_2.csv')
PATH_MATH_TRAIN_2 = os.path.join(BASE_DIR,'raw', 'math', 'train_task_3_4.csv')

def load_medicine() -> list[list]:
    """
    Handles loading of the BEA Shared Task 2024 Dataset

    Returns
    -------
    list of list
        A list where each element is a list containing:
        - question (str): The exam question.
        - choices (list of str): All possible answer choices.
        - correct_choice (str): The correct choice.
        - difficulty (float): The empirical calculated difficulty for this exam item.
    """
    
    data = []
    
    # read in the data from the corresponding csv-file
    # sort out the non-essential data for this specific problem domain
    print("Loading Dataset...")
    with open(PATH_BEA_2024, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        first_row = True
        for row in reader:
            if (first_row == False): # sort out row descriptions
                question = row[1] 
                choices = []
                for i in range(2,12): # place all possible choices in a list
                    choice = row[i]
                    if (choice != ''): # sort out empty fields
                        choices.append(choice)
                correct_choice = row[13]
                difficulty = float(row[16].replace(",",".")) # convert the difficulty from string to float
                data.append([question, choices, correct_choice, difficulty])
            first_row = False
    print(f'Dataset contains {len(data)} rows.')
    print("\n")
    return data

def load_math() -> list[list]:
    """
    Handles loading of the BEA Shared Task 2024 Dataset

    Returns
    -------
    list of list
        A list where each element is a list containing:
        - question (str): The exam question.
        - choices (list of str): All possible answer choices.
        - correct_choice (str): The correct choice.
        - difficulty (float): The empirical calculated difficulty for this exam item.
    """
    
    data = []

    with open(PATH_MATH, newline='', encoding='utf-8') as csvfile:
        counter = 1
        reader = csv.reader(csvfile, delimiter=';')
        first_row = True
        for row in reader:
            if (first_row == False):
                new_row = [row[3], row[4], row[5]]
                data.append(new_row)
            if (counter == 256): # only the first 254 rows are properly preprocessed
                break
            counter += 1
            first_row = False
    
    final_data = []
    for d in data:
        id = d[0]
        choices_text = d[1]
        question = d[2]
        temp = choices_text.split('#')
        if (len(temp) == 5):
            choices = [temp[1], temp[2], temp[3], temp[4]]
            new1 = temp[1][:-2]
            new2 = temp[2][:-2]
            new3 = temp[3][:-2]
            choices[0] = new1
            choices[1] = new2
            choices[2] = new3
        
            if (choices[0] != '' and choices[1] != '' and choices[2] != '' and choices[3] != ''):
                temp_entry = [id, question, choices] 
                final_data.append(temp_entry)
                
    ### Extract only the significant rows, by checking wheter or not the QuestionId is existing ###

    temp = []
    row_counter = 0
    for row in final_data:
        if row_counter == 0:
            row_counter += 1
            continue
        temp.append(row[0])

    existing_questions = set(temp)

    essential_user_data = []
    
    with open(PATH_MATH_TRAIN_1, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        row_counter = 0
        for row in reader:
            if row_counter == 0:
                row_counter += 1
                continue
            if row[0] in existing_questions:
                essential_user_data.append(row)

    with open(PATH_MATH_TRAIN_2, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        row_counter = 0
        for row in reader:
            if row_counter == 0:
                row_counter += 1
                continue
            if row[0] in existing_questions:
                essential_user_data.append(row)
    question_stats = []

    ## QuestionId (0), UserId (1), AnswerId (2), IsCorrect (3), CorrectAnswer (4), AnswerValue (5) ##

    question_ids = sorted(set(row[0] for row in essential_user_data))

    for id in question_ids:
        correct_answer = ""
        answered_in_total = 0
        answered_correctly = 0
        for row in essential_user_data:
            if row[0] == id:
                correct_answer = row[4]
                answered_in_total += 1
                answered_correctly += int(row[3])
        question_stats.append([id, answered_correctly, answered_in_total, correct_answer])
        
    # Add the correct answer to the final data
    
    correct_choices = {}
    p_values = {}

    for stat in question_stats:
        item_difficulty = stat[1] / stat[2]
        p_values[stat[0]] = round(item_difficulty,2)
        correct_choices[stat[0]] = stat[3]

    print(correct_choices)
    print(p_values)
    
    ## Calculate the p-value for each question ##
    return

def trivial_helper(lines: list, difficulty: float) -> list:
    
    questions = []
    i = 0
    while i < len(lines):
        if (lines[i][0] == "#"):
            question = []
            question.append(lines[i])
            for j in range(1,6):
                line = lines[i+j]
                if line[0] != "#":
                    question.append(line)
            questions.append(question)
        i += 1
    
    # remove Yes and No (True / False) questions for better comparability to other MCQ Datasets
    filtered_questions = []
    binary_question_count = 0
    for q in questions:
        if len(q) == 6:
            q[0] = q[0].replace('\n', '')
            q[0] = q[0].replace('#Q ', '')
            q[1] = q[1].replace('\n', '')
            q[1] = q[1].replace('^ ', '')
            q[2] = q[2].replace('\n', '')
            q[2] = q[2].replace('A ', '')
            q[3] = q[3].replace('\n', '')
            q[3] = q[3].replace('B ', '')
            q[4] = q[4].replace('\n', '')
            q[4] = q[4].replace('C ', '')
            q[5] = q[5].replace('\n', '')
            q[5] = q[5].replace('D ', '')
            filtered_questions.append(q)
        else:
            binary_question_count += 1
    
    final_questions = []

    for q in filtered_questions:
        question = q[0]
        choices = [q[2], q[3], q[4], q[5]]
        correct_choice = q[1]
        final = [question, choices, correct_choice, difficulty]
        final_questions.append(final)

    return final_questions

def load_trivial() -> list[list]:
    """
    Handles loading of a trivial dataset from OpenTriviaQA

    Returns
    -------
    list of list
        A list where each element is a list containing:
        - question (str): The question.
        - choices (list of str): All possible answer choices.
        - correct_choice (str): The correct choice.
        - difficulty (float): The empirical calculated difficulty for this exam item.
    """
    
    lines_easy = []
    lines_hard = []
    print("Loading Dataset...")
    easy_count = 0
    # read in the easy dataset
    with open(PATH_TRIVIAL_EASY, encoding="utf-8") as questions_easy:
        for line in questions_easy:
            lines_easy.append(line)
            easy_count += 1
    hard_count = 0
    # read in the hard dataset
    with open(PATH_TRIVIAL_HARD, encoding="utf-8") as questions_hard:
        for line in questions_hard:
            lines_hard.append(line)
            hard_count += 1
    
    final_easy = trivial_helper(lines_easy, -1)
    final_hard = trivial_helper(lines_hard, 1)
    
    final = final_easy + final_hard
    print("\n")
    return final

def analyze_dataset(data: list):
    print("Analyze Dataset...")
    print(f'Dataset contains {len(data)} rows.')
    
    difficulties = []
    for q in data:
        difficulties.append(q[3])
    
    print(f"Minimum Difficulty: {min(difficulties)}")
    print(f"Maximum Difficulty: {max(difficulties)}")
    print(f"Mean Difficulty: {np.mean(difficulties)}")
    print("\n")
    return