'''
This file handles loading of the following datasets 
- medicine: USMLE MCQs from BEA 2024 Shared Task Data (BEA_2024.csv)
- math: MCQs from the NeurIPS 2020 Education Challenge (NeurIPS_2020.csv)
- trivial: MCQs from the OpenTriviaQA (for-kids.txt, science-technology.txt)
- language: Translation Tasks from the Duolingo Dataverse Half-Life Regression (learning_traces.13m.csv)
'''

import csv
import os
import numpy as np
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import re

# Dynamic System Paths to the raw datasets

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_BEA_2024 = os.path.join(BASE_DIR,'raw', 'bea', 'BEA_2024.csv')
PATH_TRIVIAL_EASY = os.path.join(BASE_DIR,'raw', 'trivial', 'for-kids.txt')
PATH_TRIVIAL_HARD = os.path.join(BASE_DIR,'raw', 'trivial', 'science-technology.txt')
PATH_MATH = os.path.join(BASE_DIR,'raw', 'math', 'NeurIPS_2020.csv')
PATH_MATH_TRAIN_1 = os.path.join(BASE_DIR,'raw', 'math', 'train_task_1_2.csv')
PATH_MATH_TRAIN_2 = os.path.join(BASE_DIR,'raw', 'math', 'train_task_3_4.csv')
PATH_LANGUAGE = os.path.join(BASE_DIR,'raw', 'language', 'learning_traces.13m.csv')

# Define all loading functions for the datasets

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
    
    # this whole part is very inelegant, but it works regardless #
    
    temp_1 = []

    print("Loading Dataset...")
    with open(PATH_MATH, newline='', encoding='utf-8') as csvfile:
        counter = 1
        reader = csv.reader(csvfile, delimiter=';')
        first_row = True
        for row in reader:
            if (first_row == False):
                new_row = [row[3], row[4], row[5]]
                temp_1.append(new_row)
            if (counter == 256): # only the first 254 rows are properly preprocessed
                break
            counter += 1
            first_row = False
    
    temp_2 = []
    for d in temp_1:
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
                temp_2.append(temp_entry)

    temp = []
    row_counter = 0
    for row in temp_2:
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
    
    correct_choices = {}
    p_values = {}

    for stat in question_stats:
        item_difficulty = stat[1] / stat[2]
        p_values[stat[0]] = round(item_difficulty,2)
        correct_choices[stat[0]] = stat[3]

    data = []

    for i in temp_2:
        q_id = i[0]
        if q_id not in correct_choices or q_id not in p_values:
            continue
        question_string = i[1]
        possible_choices = i[2]
        correct_choice_idx = int(correct_choices[q_id]) - 1
        difficulty = p_values[q_id]
        
        data_entry = [question_string, possible_choices, possible_choices[correct_choice_idx], difficulty]
        data.append(data_entry)
    
    return data

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

def load_language() -> list[list]:
    """
    Handles loading of a language translation dataset from Duolingo Dataverse Half-Life Regression
    Returns
    -------
    list of list
        A list where each element is a list containing:
        - question (str): The question.
        - choices (list of str): All possible answer choices.
        - correct_choice (str): The correct choice.
        - difficulty (float): The calculated difficulty for this exam item.
    """

    data = []

    id_difficulty = defaultdict(lambda: [0, 0])
    userID_taskID_seen_and_correct = dict()
    id_question = dict()
    id_amount_of_data = defaultdict(lambda: 0)

    print("Loading dataset...")
    with open(PATH_LANGUAGE, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        next(reader)

        for row in reader:
            learning_language = row[4]
            if learning_language != "de": # reduce data complexity by only using german words
                continue
            user_id = row[3]
            task_id = row[6]
            question = row[7]
            seen = int(row[8])
            correct = int(row[9])
            id_question[task_id] = question
            
            entry_key = (user_id, task_id)
            
            # check whether or not entry already exists, if not create entry, if the seen in total amount of the new value is greater than the existing one update entry
            if entry_key not in userID_taskID_seen_and_correct or seen > userID_taskID_seen_and_correct[(user_id, task_id)][0]:
                userID_taskID_seen_and_correct[(user_id, task_id)] = (seen,correct)
    
    # count how often an item has been seen totally and correctly answered across all users
    for (user_id, task_id), (seen, correct) in userID_taskID_seen_and_correct.items():
        id_amount_of_data[task_id] += 1
        id_difficulty[task_id][0] += seen
        id_difficulty[task_id][1] += correct

    # calculate the proportion of the correct answers to seen in total
    task_difficulties = dict()
    for task_id, (seen_total, correct_total) in id_difficulty.items():
        difficulty = correct_total / seen_total
        task_difficulties[task_id] = difficulty
    
    '''
    # remove entries that have only 1,2 or 3 entries throughout
    for id, i in id_amount_of_data.items():
        if i == 1:
            del task_difficulties[id]
    '''
    
    # format data
    for task_id, difficulty in task_difficulties.items():
        if difficulty <= 1:
            data_entry = [id_question[task_id], [], "", difficulty]
            data.append(data_entry)
    
    # remove <Tags> in the question entries
    for d in data:
        cleaned = re.sub(r"<[^<>]*>", "", d[0])
        d[0] = cleaned
            
    print("\n")

    return data

# Data Analysis Functions

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

def analyse_training_data(y_train: list):
    
    # # Defining the fonts before plotting:
    plt.rcParams.update({
        'font.family': 'Courier New',  # monospace font
        'font.size': 20,
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'figure.titlesize': 20
    }) 
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.25)
    
    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle='-', linewidth=0.25, alpha=0.15)
    
    ax.set_axisbelow(True) # <-- Ensure grid is below data
    
    plt.title('Difficulty Distribution')
    ax.set_xlabel('Difficulty')
    ax.set_ylabel('Amount')
    
    plt.hist(y_train, bins=100)
    plt.show()
    return