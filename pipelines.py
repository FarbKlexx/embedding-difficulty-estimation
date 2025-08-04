import data.datasets as ds
import data.data_preproc as preproc
import methods as methods
from utils.loggingUtils import print_X_y
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

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

def trivial(task: str):
    print("------- Started Trivial Pipeline -------")
    data = ds.load_trivial()
    ds.analyze_dataset(data)
    preproc_data = preproc.get_detailed_input(data, task)

    embeddings = methods.e5_embed(preproc_data)
    
    X,y = preproc.create_embedding_difficulty_tuple(embeddings, data)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    ds.analyse_training_data("trivial", y_train)
    
    reg = Ridge(alpha=0.1).fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    methods.test_regressor("trivial", y_test, y_pred)
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
    methods.test_regressor(y_test_wl, y_pred_wl)

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