from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import pandas as pd

def labelencode_if_object(df_ml):
    for col in df_ml.columns:
        if df_ml[col].dtype == 'O':
            le = LabelEncoder()
            replacement_series = le.fit_transform(df_ml[col])
            #print("dropping", col)
            df_ml = df_ml.drop(columns=[col])
            df_ml[col] = replacement_series
    return df_ml

def discover(df, classifier_overrides=None, method="rf", score_method_class=None, score_method_reg = None, random_state=None):
    """TODO describe what we're doing here"""
    corr_methods = ["pearson", 'spearman', 'kendall']
    known_methods = corr_methods + ['rf']
    assert method in set(known_methods), f"Expecting method to be one of: {known_methods}"
   
    score_methods_class = ['accuracy','balanced_accuracy','neg_log_loss']
    assert score_method_class in set(score_methods_class + [None]), f"Expecting scoring method for class to be one of: {score_methods_class + [None]}"
    
    score_methods_reg = ['neg_mean_squared_error','r2']
    assert score_method_reg in set(score_methods_reg + [None]), f"Expecting scoring method for reg to be one of: {score_methods_reg + [None]}"
    
    score_mapping = {}   
    estimator_mapping = {}
    cols = df.columns
    if classifier_overrides is None:
        classifier_overrides = []
    for col in cols:
        if col in classifier_overrides:
            est = RandomForestClassifier(n_estimators=50, random_state=random_state)
            scorer = score_method_class
        else:
            est = RandomForestRegressor(n_estimators=50, random_state=random_state)
            scorer = score_method_reg
            
        estimator_mapping[col] = est
        score_mapping[col] = scorer

    ds = []
    for idx_Y, target in enumerate(cols):
        est = estimator_mapping[target]
        scorer = score_mapping[target]
        for idx_X, feature in enumerate(cols):
            if idx_X == idx_Y:
                continue

            df_ml = df[[feature, target]]
            rows_before_drop_na = df_ml.shape[0]
            df_ml = df_ml.dropna()
            rows_after_drop_na = df_ml.shape[0]
            #if rows_after_drop_na < rows_before_drop_na:
            #    print(feature, target)
            #    print(f"Dropped {rows_before_drop_na - rows_after_drop_na} rows")

            df_ml = labelencode_if_object(df_ml)

            df_X = df_ml[[feature]]
            df_y = df_ml[target]

            assert df_X.isnull().sum().sum() == 0
            assert df_y.isnull().sum() == 0

            #if False:
            #    # no cross validation
            #    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.33, random_state=0)
            #    #print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
            #    est.fit(X_train, y_train)
            #    score = est.score(X_test, y_test)

            score = 0.0
            if method=="rf":
                # cross validation
                scores = cross_val_score(est, df_X, df_y, scoring=scorer, cv=3)#, n_jobs=-1)
                score = scores.mean()
                if scorer in ['neg_log_loss','neg_mean_squared_error']:
                    score = -score
                if scorer == 'r2':
                    score = max(score,0.0)

            if method in set(corr_methods):
                pair = df_ml[[feature, target]]
                assert pair.shape[1] == 2
                score = pair.corr(method=method)[feature][target]

            d = {'feature': feature, 'target': target, 'score': score}
            ds.append(d)

    df_results = pd.DataFrame(ds)
    return df_results

if __name__ == "__main__":
    # simple test to make sure the code is running
    import numpy as np
    X = pd.DataFrame({'a': np.ones(10),
                      #'a': [1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
                      'b': np.arange(0, 10),
                      'c': np.arange(0, 20, 2)})
    df_results = discover(X)
    print(df_results)
    assert (df_results.query("feature=='b' and target=='a'")['score'].iloc[0]) == 1, "Expect b to predict a"

    df_results = discover(X, method="kendall")
    print(df_results)
    assert (df_results.query("feature=='b' and target=='c'")['score'].iloc[0]) >= 0.99, "Expect b to predict c"
