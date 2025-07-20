'''
This file handles the preprocessing of the data to make it ready for embedding
'''

CHOICE_PREFIX = ['Choice A: ', ' Choice B: ',' Choice C: ',' Choice D: ',' Choice E: ',' Choice F: ',' Choice G: ',' Choice H: ',' Choice I: ',' Choice J: ']

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def get_detailed_input(data: list[list], task: str) -> list[str]:
    """
    Handels converting the raw data into strings ready for embedding
    
    Parameters
    ----------
    data: list of list
        Expects a list where each element is a list containing:
        - question (str): The exam question.
        - choices (list of str): All possible answer choices.
        - correct_choice (str): The correct choice.
        - difficulty (float): The empirical calculated difficulty for this exam item.
    task: str
        A short one-sentence description of the task for the embedding model
        
    Returns
    -------
    list of str
        A list where each element is a String representing the text to be embedded
    """
    print("Preprocessing Data...")
    
    detailed_inputs = []
    
    for row in data:
        question = row[0]
        choices = row[1]
        
        correct_choice = row[2]
        correct_choice_prefix = "\nCorrect Choice: "
        if correct_choice == "":
            correct_choice_prefix = ""
        
        choices_string = ''
        if len(choices) != 0:
            choice_count = 0
            for choice in choices:
                choices_string = choices_string + CHOICE_PREFIX[choice_count] + choice
                choice_count += 1

            
        detailed_input = question + "\n" + choices_string + correct_choice_prefix + correct_choice
        detailed_inputs.append(get_detailed_instruct(task, detailed_input))
    
    print("\n")
    return detailed_inputs

def create_embedding_difficulty_tuple(embeddings: list, items: list) -> tuple:
    """
    Creates the embedding-difficulty-tuples for regression learning
    
    Parameters
    ----------
    embeddings: list
        Expects a list where each element is an embedding vector.
    items: list
        Expects a list where each element is a list containing:
        - question (str): The exam question.
        - choices (list of str): All possible answer choices.
        - correct_choice (str): The correct choice.
        - difficulty (float): The empirical calculated difficulty for this exam item.
        
    Returns
    -------
    tuple of lists
        A tuple with 2 elements:
        X: the embedding vectors
        y: the difficulties of the corresponding items
    """
    difficulties = []
    for item in items:
        difficulties.append(item[3])
    
    return embeddings, difficulties