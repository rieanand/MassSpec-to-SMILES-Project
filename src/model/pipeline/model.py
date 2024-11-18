'''Where we define and build the model. Train and save models we need.'''
from sklearn.model_selection import train_test_split, GridSearchCV
import pickle as pk
from preparation import prepare_data

def build_model():
    df = prepare_data()

    X, y = get_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = train_model(X_train, y_train)

    score = evaluate_model(rf, X_test, y_test)
    print(f'Models score = {score}')

    save_model(rf)

def save_model(model):
    pk.dump(model, open('models/model_name', 'wb'))

def evaluate_model(model, X_test, y_test):
   return model.score(X_test, y_test)
    
def train_model(X_train, y_train):
    grid_space = {'param1': test_values1, 'param2': test_values2}

    grid = GridSearchCV(regression_model,
                        param_grid=grid_space,
                        cv=5,
                        scoring='r2')
    
    model_grid = grid.fit(X_train, y_train)

    return model_grid.best_estimator_

def get_X_y(data,
            col_X = ['content'],
            col_y = 'content'):
    return data[col_X], data[col_y]