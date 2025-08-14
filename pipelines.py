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


def bea_reproduction(task: str):
    print("------- BEA Reproduction Pipeline -------")
    data = ds.load_medicine()
    ds.analyze_dataset(data)
    
    preproc_data = preproc.get_detailed_input(data, task)

    embeddings = methods.e5_embed(preproc_data)
    
    X,y = preproc.create_embedding_difficulty_tuple(embeddings, data)
    # print_X_y(X,y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    ds.analyse_training_data("bea", y_train)
    
    reg = Ridge(alpha=0.1).fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    methods.test_regressor("bea", y_test, y_pred)
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

def math(task: str):
    print("------- Started Math Pipeline -------")
    data = ds.load_math()
    
    ds.analyze_dataset(data)
    preproc_data = preproc.get_detailed_input(data, task)

    embeddings = methods.e5_embed(preproc_data)
    
    X,y = preproc.create_embedding_difficulty_tuple(embeddings, data)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    ds.analyse_training_data("math", y_train)
    
    reg = Ridge(alpha=0.1).fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    methods.test_regressor("math", y_test, y_pred)
    
    return

def language(task: str):
    print("------- Started Language Pipeline -------")
    data = ds.load_language()

    # side step to train a regression model over the word length

    X_wl, y_wl = methods.word_length(data)

    X_train_wl, X_test_wl, y_train_wl, y_test_wl = train_test_split(X_wl, y_wl, test_size=0.2, random_state=0)

    reg = Ridge(alpha=0.1).fit(X_train_wl, y_train_wl)
    y_pred_wl = reg.predict(X_test_wl)
    methods.test_regressor("language", y_test_wl, y_pred_wl)

    ds.analyze_dataset(data)
    preproc_data = preproc.get_detailed_input(data, task)

    embeddings = methods.e5_embed(preproc_data)
    
    X,y = preproc.create_embedding_difficulty_tuple(embeddings, data)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    ds.analyse_training_data("language", y_train)

    reg = Ridge(alpha=0.1).fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    methods.test_regressor("language", y_test, y_pred)
    
# TODO add this to the other pipelines
def concat(task: str):
    
    # pipeline should be the same but preproc differs

    return