import data.datasets as ds
import data.data_preproc as preproc
import methods as methods
from utils.loggingUtils import print_X_y
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
import numpy as np


def bea_reproduction(determine_best_instruct = False):
    print("------- BEA Reproduction Pipeline -------")

    data = ds.load_medicine()
    ds.analyze_dataset(data)

    # the following instruct variations are generated with chatgpt 
    # Prompt: "the following phrase is the instruction for an multilingual-e5-large-instruct embedding: 
    # TEMP_TASK_BEA = 'Given a multiple choice question from the United States Medical Licensing Examination, estimate the difficulty on a scale from 0 (very hard) to 1.5 (very easy).'
    # give me 9 more semantically similar variations of this phrase."

    # Best Instruct
    if (determine_best_instruct == True):
        INSTRUCTS = ["",
                     "Given a multiple choice question from the United States Medical Licensing Examination, estimate the difficulty on a scale from 0 (very hard) to 1.5 (very easy).",
                    "Given a multiple-choice question from the US Medical Licensing Examination, predict its difficulty on a scale from 0 (very difficult) to 1.5 (very easy).",
                    "For the following USMLE multiple-choice question, estimate the ease of answering on a scale of 0 (hardest) to 1.5 (easiest).",
                    "Assess the difficulty of the provided USMLE multiple-choice question, scoring from 0 (extremely hard) to 1.5 (extremely easy).",
                    "Given a US Medical Licensing Examination multiple-choice question, assign a difficulty score between 0 (most difficult) and 1.5 (least difficult).",
                    "Evaluate the difficulty level of the given USMLE question using a scale where 0 = very hard and 1.5 = very easy.",
                    "Given a multiple-choice question from the United States Medical Licensing Examination, rate its difficulty from 0 (hardest) to 1.5 (easiest).",
                    "Score the difficulty of the following USMLE question on a scale ranging from 0 (maximum difficulty) to 1.5 (minimum difficulty).",
                    "Given a multiple-choice question from the US Medical Licensing Examination, determine its difficulty rating between 0 (very challenging) and 1.5 (very simple).",
                    "Estimate the difficulty of this USMLE multiple-choice question, using 0 as the hardest possible score and 1.5 as the easiest."]
        
        best = ("",100)

        c = 1
        for i in INSTRUCTS:
            print(f"Testing Instruct Number {c}:")
            c += 1
            preproc_data = preproc.get_detailed_input(data, i)
            embeddings = methods.e5_embed(preproc_data)
            X,y = preproc.create_embedding_difficulty_tuple(embeddings, data)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            # Ridge Regression
            ridge = Ridge(alpha=0.1).fit(X_train, y_train)
            y_pred_ridge = ridge.predict(X_test)
            rmse_ridge = root_mean_squared_error(y_test, y_pred_ridge)

            # RandomForest Regression
            random_forest = RandomForestRegressor(n_estimators=200, max_depth=20, n_jobs=-1, random_state=0).fit(X_train, y_train)
            y_pred_random_forest = random_forest.predict(X_test)
            rmse_random_forest = root_mean_squared_error(y_test, y_pred_random_forest)

            mean_rmse = np.mean(np.array([rmse_ridge, rmse_random_forest]))

            if mean_rmse < best[1]:
                best = (i,mean_rmse)

        print(f"The best performing Instruct is '{best[0]}' with an average RMSE of {best[1]}.\n")
    
    if (determine_best_instruct == True):
        best_instruct = best[0]
    else:
        # this was determined from the code above. Use determine_best_instruct if you want to confirm this claim (Long runtime even on GPU).
        best_instruct = "Given a multiple choice question from the United States Medical Licensing Examination, estimate the difficulty on a scale from 0 (very hard) to 1.5 (very easy)."  

    # Best Regressor

    results = []

    preproc_data = preproc.get_detailed_input(data, best_instruct)
    embeddings = methods.e5_embed(preproc_data)
    X,y = preproc.create_embedding_difficulty_tuple(embeddings, data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    ds.analyse_training_data("usmle", y_train)
    
    # Linear Regression
    linear = LinearRegression().fit(X_train, y_train)
    y_pred_linear = linear.predict(X_test)
    rmse_linear = root_mean_squared_error(y_test, y_pred_linear)

    results.append(("Linear Regression", rmse_linear, y_pred_linear))

    # Ridge Regression
    ridge = Ridge().fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    rmse_ridge = root_mean_squared_error(y_test, y_pred_ridge)

    results.append(("Ridge Regression", rmse_ridge, y_pred_ridge))

    # RandomForest Regression
    random_forest = RandomForestRegressor().fit(X_train, y_train)
    y_pred_random_forest = random_forest.predict(X_test)
    rmse_random_forest = root_mean_squared_error(y_test, y_pred_random_forest)

    results.append(("RandomForest Regression", rmse_random_forest, y_pred_random_forest))

    for r in results:
        print(f"RMSE of {r[0]}: {r[1]}")

    best = min(results, key=lambda r: r[1])

    methods.test_regressor("usmle", y_test, best[2])
    return

def trivial(determine_best_instruct = False):
    print("------- Started Trivial Pipeline -------")

    data = ds.load_trivial()
    ds.analyze_dataset(data)

    # the following instruct variations are generated with chatgpt 
    # Prompt: "the following phrase is the instruction for an multilingual-e5-large-instruct embedding: 
    # TEMP_TASK_TRIVIAL = 'Given a multiple choice question, estimate the difficulty on a scale from 0 (very hard) to 1 (very easy).' 
    # give me 9 more semantically similar variations of this phrase."

    # Best Instruct
    if (determine_best_instruct == True):
        INSTRUCTS = ["",
                    "Given a multiple choice question, estimate the difficulty on a scale from 0 (very hard) to 1 (very easy).",
                    "Given a multiple-choice question, predict its difficulty level from 0 (very difficult) to 1 (very easy).",
                    "For the following multiple-choice question, estimate the ease of answering on a scale of 0 (hardest) to 1 (easiest).",
                    "Given a multiple-choice question, assign a difficulty score between 0 (most difficult) and 1 (least difficult).",
                    "Evaluate the difficulty of the given multiple-choice question, scoring from 0 (extremely hard) to 1 (extremely easy).",
                    "Rate the given multiple-choice question on a scale from 0 (hardest) to 1 (easiest) based on difficulty.",
                    "Given a multiple-choice question, output a score from 0 (very challenging) to 1 (very simple) representing its difficulty.",
                    "Assess the difficulty of the provided multiple-choice question using a scale where 0 = very difficult and 1 = very easy.",
                    "Given a multiple-choice question, determine its difficulty rating on a scale from 0 (hard) to 1 (easy).",
                    "Score the following multiple-choice question's difficulty between 0 (maximum difficulty) and 1 (minimum difficulty)."]
        
        best = ("",100)

        c = 1
        for i in INSTRUCTS:
            print(f"Testing Instruct Number {c}:")
            c += 1
            preproc_data = preproc.get_detailed_input(data, i)
            embeddings = methods.e5_embed(preproc_data)
            X,y = preproc.create_embedding_difficulty_tuple(embeddings, data)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            # Ridge Regression
            ridge = Ridge(alpha=0.1).fit(X_train, y_train)
            y_pred_ridge = ridge.predict(X_test)
            rmse_ridge = root_mean_squared_error(y_test, y_pred_ridge)

            # RandomForest Regression
            random_forest = RandomForestRegressor(n_estimators=200, max_depth=20, n_jobs=-1, random_state=0).fit(X_train, y_train)
            y_pred_random_forest = random_forest.predict(X_test)
            rmse_random_forest = root_mean_squared_error(y_test, y_pred_random_forest)

            mean_rmse = np.mean(np.array([rmse_ridge, rmse_random_forest]))

            if mean_rmse < best[1]:
                best = (i,mean_rmse)

        print(f"The best performing Instruct is '{best[0]}' with an average RMSE of {best[1]}.\n")
    
    if (determine_best_instruct == True):
        best_instruct = best[0]
    else:
        best_instruct = "" # this was determined from the code above. Use determine_best_instruct if you want to confirm this claim (Long runtime even on GPU). 

    # Best Regressor

    results = []

    preproc_data = preproc.get_detailed_input(data, best_instruct)
    embeddings = methods.e5_embed(preproc_data)
    X,y = preproc.create_embedding_difficulty_tuple(embeddings, data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    ds.analyse_training_data("trivial", y_train)
    
    # Linear Regression
    linear = LinearRegression().fit(X_train, y_train)
    y_pred_linear = linear.predict(X_test)
    rmse_linear = root_mean_squared_error(y_test, y_pred_linear)

    results.append(("Linear Regression", rmse_linear, y_pred_linear))

    # Ridge Regression
    ridge = Ridge().fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    rmse_ridge = root_mean_squared_error(y_test, y_pred_ridge)

    results.append(("Ridge Regression", rmse_ridge, y_pred_ridge))

    # RandomForest Regression
    random_forest = RandomForestRegressor().fit(X_train, y_train)
    y_pred_random_forest = random_forest.predict(X_test)
    rmse_random_forest = root_mean_squared_error(y_test, y_pred_random_forest)

    results.append(("RandomForest Regression", rmse_random_forest, y_pred_random_forest))

    for r in results:
        print(f"RMSE of {r[0]}: {r[1]}")

    best = min(results, key=lambda r: r[1])

    methods.test_regressor("trivial", y_test, best[2])
    return

def math(determine_best_instruct = False):
    print("------- Started Math Pipeline -------")
    data = ds.load_math()
    
    ds.analyze_dataset(data)
    # the following instruct variations are generated with chatgpt 
    # Prompt: "the following phrase is the instruction for an multilingual-e5-large-instruct embedding: 
    # TEMP_TASK_MATH = 'Given a multiple choice question from a math exam, estimate the difficulty on a scale from 0 (very hard) to 1 (very easy).'
    # give me 9 more semantically similar variations of this phrase."

    # Best Instruct
    if (determine_best_instruct == True):
        INSTRUCTS = ["",
                    "Given a multiple choice question from a math exam, estimate the difficulty on a scale from 0 (very hard) to 1 (very easy).",
                    "Given a multiple-choice question from a mathematics exam, predict its difficulty on a scale from 0 (hardest) to 1 (easiest).",
                    "For the following math exam multiple-choice question, estimate its difficulty level where 0 means very hard and 1 means very easy.",
                    "Assess the difficulty of the provided mathematics multiple-choice question, rating it between 0 (most difficult) and 1 (least difficult).",
                    "Given a multiple-choice question from a math test, assign a difficulty score from 0 (very challenging) to 1 (very easy).",
                    "Evaluate the given math exam question and rate its difficulty on a scale from 0 (most difficult) to 1 (simplest).",
                    "Given a math multiple-choice question, determine its difficulty score where 0 is hardest and 1 is easiest.",
                    "Rate the following mathematics exam multiple-choice question on a scale from 0 (maximum difficulty) to 1 (minimum difficulty).",
                    "Given a math exam multiple-choice question, estimate the difficulty from 0 (very hard) to 1 (very easy).",
                    "Score the difficulty of this multiple-choice math question, using 0 for most difficult and 1 for easiest."]
        
        best = ("",100)

        c = 1
        for i in INSTRUCTS:
            print(f"Testing Instruct Number {c}:")
            c += 1
            preproc_data = preproc.get_detailed_input(data, i)
            embeddings = methods.e5_embed(preproc_data)
            X,y = preproc.create_embedding_difficulty_tuple(embeddings, data)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            # Ridge Regression
            ridge = Ridge(alpha=0.1).fit(X_train, y_train)
            y_pred_ridge = ridge.predict(X_test)
            rmse_ridge = root_mean_squared_error(y_test, y_pred_ridge)

            # RandomForest Regression
            random_forest = RandomForestRegressor(n_estimators=200, max_depth=20, n_jobs=-1, random_state=0).fit(X_train, y_train)
            y_pred_random_forest = random_forest.predict(X_test)
            rmse_random_forest = root_mean_squared_error(y_test, y_pred_random_forest)

            mean_rmse = np.mean(np.array([rmse_ridge, rmse_random_forest]))

            if mean_rmse < best[1]:
                best = (i,mean_rmse)

        print(f"The best performing Instruct is '{best[0]}' with an average RMSE of {best[1]}.\n")
    
    if (determine_best_instruct == True):
        best_instruct = best[0]
    else:
        # this was determined from the code above. Use determine_best_instruct if you want to confirm this claim (Long runtime even on GPU).
        best_instruct = "For the following math exam multiple-choice question, estimate its difficulty level where 0 means very hard and 1 means very easy."  

    # Best Regressor

    results = []

    preproc_data = preproc.get_detailed_input(data, best_instruct)
    embeddings = methods.e5_embed(preproc_data)
    X,y = preproc.create_embedding_difficulty_tuple(embeddings, data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    ds.analyse_training_data("neuroips", y_train)
    
    # Linear Regression
    linear = LinearRegression().fit(X_train, y_train)
    y_pred_linear = linear.predict(X_test)
    rmse_linear = root_mean_squared_error(y_test, y_pred_linear)

    results.append(("Linear Regression", rmse_linear, y_pred_linear))

    # Ridge Regression
    ridge = Ridge().fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    rmse_ridge = root_mean_squared_error(y_test, y_pred_ridge)

    results.append(("Ridge Regression", rmse_ridge, y_pred_ridge))

    # RandomForest Regression
    random_forest = RandomForestRegressor().fit(X_train, y_train)
    y_pred_random_forest = random_forest.predict(X_test)
    rmse_random_forest = root_mean_squared_error(y_test, y_pred_random_forest)

    results.append(("RandomForest Regression", rmse_random_forest, y_pred_random_forest))

    for r in results:
        print(f"RMSE of {r[0]}: {r[1]}")

    best = min(results, key=lambda r: r[1])

    methods.test_regressor("neuroips", y_test, best[2])
    return

def language(determine_best_instruct = False):
    print("------- Started Language Pipeline -------")
    data = ds.load_language()

    # side step to train a regression model over the word length

    print("Training Ridge Regression over word length")

    X_wl, y_wl = methods.word_length(data)
    X_train_wl, X_test_wl, y_train_wl, y_test_wl = train_test_split(X_wl, y_wl, test_size=0.2, random_state=0)

    ridge = Ridge().fit(X_train_wl, y_train_wl)
    y_pred_wl = ridge.predict(X_test_wl)
    methods.test_regressor("duolingo_wl", y_test_wl, y_pred_wl)

    # the following instruct variations are generated with chatgpt 
    # Prompt: "the following phrase is the instruction for an multilingual-e5-large-instruct embedding: 
    # TEMP_TASK_LANGUAGE = 'Given a word to translate, estimate the difficulty on a scale from 0 (very hard) to 1 (very easy).'
    # give me 9 more semantically similar variations of this phrase."

    # Best Instruct
    if (determine_best_instruct == True):
        INSTRUCTS = ["",
                    "Given a word to translate, estimate the difficulty on a scale from 0 (very hard) to 1 (very easy).",
                    "Given a word for translation, predict its difficulty on a scale from 0 (most difficult) to 1 (least difficult).",
                    "For the following word that needs to be translated, estimate its difficulty level where 0 means very hard and 1 means very easy.",
                    "Assess the difficulty of translating the provided word, rating it between 0 (hardest) and 1 (easiest).",
                    "Given a word to be translated, assign a difficulty score from 0 (very challenging) to 1 (very simple).",
                    "Evaluate the difficulty of translating the given word, scoring from 0 (extremely hard) to 1 (extremely easy).",
                    "Given a word for translation, determine its difficulty rating where 0 is hardest and 1 is easiest.",
                    "Rate the difficulty of translating the following word on a scale from 0 (maximum difficulty) to 1 (minimum difficulty).",
                    "Given a single word to translate, estimate the difficulty from 0 (very hard) to 1 (very easy).",
                    "Score how difficult it is to translate the given word, using 0 for hardest and 1 for easiest."]
        
        best = ("",100)

        c = 1
        for i in INSTRUCTS:
            print(f"Testing Instruct Number {c}:")
            c += 1
            preproc_data = preproc.get_detailed_input(data, i)
            embeddings = methods.e5_embed(preproc_data)
            X,y = preproc.create_embedding_difficulty_tuple(embeddings, data)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            # Ridge Regression
            ridge = Ridge(alpha=0.1).fit(X_train, y_train)
            y_pred_ridge = ridge.predict(X_test)
            rmse_ridge = root_mean_squared_error(y_test, y_pred_ridge)

            # RandomForest Regression
            random_forest = RandomForestRegressor(n_estimators=200, max_depth=20, n_jobs=-1, random_state=0).fit(X_train, y_train)
            y_pred_random_forest = random_forest.predict(X_test)
            rmse_random_forest = root_mean_squared_error(y_test, y_pred_random_forest)

            mean_rmse = np.mean(np.array([rmse_ridge, rmse_random_forest]))

            if mean_rmse < best[1]:
                best = (i,mean_rmse)

        print(f"The best performing Instruct is '{best[0]}' with an average RMSE of {best[1]}.\n")
    
    if (determine_best_instruct == True):
        best_instruct = best[0]
    else:
        # this was determined from the code above. Use determine_best_instruct if you want to confirm this claim (Long runtime even on GPU).
        best_instruct = "Assess the difficulty of translating the provided word, rating it between 0 (hardest) and 1 (easiest)."  

    # Best Regressor

    results = []

    preproc_data = preproc.get_detailed_input(data, best_instruct)
    embeddings = methods.e5_embed(preproc_data)
    X,y = preproc.create_embedding_difficulty_tuple(embeddings, data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    ds.analyse_training_data("duolingo", y_train)
    
    # Linear Regression
    linear = LinearRegression().fit(X_train, y_train)
    y_pred_linear = linear.predict(X_test)
    rmse_linear = root_mean_squared_error(y_test, y_pred_linear)

    results.append(("Linear Regression", rmse_linear, y_pred_linear))

    # Ridge Regression
    ridge = Ridge().fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    rmse_ridge = root_mean_squared_error(y_test, y_pred_ridge)

    results.append(("Ridge Regression", rmse_ridge, y_pred_ridge))

    # RandomForest Regression
    random_forest = RandomForestRegressor().fit(X_train, y_train)
    y_pred_random_forest = random_forest.predict(X_test)
    rmse_random_forest = root_mean_squared_error(y_test, y_pred_random_forest)

    results.append(("RandomForest Regression", rmse_random_forest, y_pred_random_forest))

    for r in results:
        print(f"RMSE of {r[0]}: {r[1]}")

    best = min(results, key=lambda r: r[1])

    methods.test_regressor("duolingo", y_test, best[2])
    return

# TODO add this to the other pipelines
def concat(task: str):
    
    # pipeline should be the same but preproc differs

    return