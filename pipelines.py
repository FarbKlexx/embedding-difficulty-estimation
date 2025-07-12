import data.datasets as ds
import data.data_preproc as preproc
import methods as methods
from utils.loggingUtils import print_X_y
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

def bea_reproduction(task: str):
    print("------- BEA Reproduction Pipeline -------")
    data = ds.load_medicine()
    ds.analyze_dataset(data)
    
    preproc_data = preproc.get_detailed_input(data, task)

    embeddings = methods.e5_embed(preproc_data)
    
    X,y = preproc.create_embedding_difficulty_tuple(embeddings, data)
    # print_X_y(X,y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    reg = Ridge(alpha=0.1).fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    methods.test_regressor(y_test, y_pred)
    return

def trivial(task: str):
    print("------- Started Trivial Pipeline -------")
    data = ds.load_trivial()
    ds.analyze_dataset(data)
    preproc_data = preproc.get_detailed_input(data, task)

    embeddings = methods.e5_embed(preproc_data)
    
    X,y = preproc.create_embedding_difficulty_tuple(embeddings, data)
    # print_X_y(X,y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    reg = Ridge(alpha=0.1).fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    methods.test_regressor(y_test, y_pred)
    return

# TODO finish math pipeline
def math(task: str):
    print("------- Started Math Pipeline -------")
    data = ds.load_math()
    '''
    ds.analyze_dataset(data)
    preproc_data = preproc.get_detailed_input(data, task)

    embeddings = methods.e5_embed(preproc_data)
    
    X,y = preproc.create_embedding_difficulty_tuple(embeddings, data)
    # print_X_y(X,y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    reg = Ridge(alpha=0.1).fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    methods.test_regressor(y_test, y_pred)
    '''
    return

# TODO finish concat pipeline
def concat(task: str):
    
    # pipeline should be the same but preproc differs
    return