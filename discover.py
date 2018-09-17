from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.utils.multiclass import type_of_target
import numpy
import pandas as pd
import itertools

def labelencode_if_object(df_ml):
    for col in df_ml.columns:
        if df_ml[col].dtype == 'O':
            le = LabelEncoder()
            replacement_series = le.fit_transform(df_ml[col])
            #print("dropping", col)
            df_ml = df_ml.drop(columns=[col])
            df_ml[col] = replacement_series
    return df_ml

def discover(df, classifier_overrides=None, method="rf"):
    corr_methods = ["pearson", 'spearman', 'kendall']
    known_methods = corr_methods + ['rf','mutual_information']
    assert method in set(known_methods), f"Expecting method to be one of: {known_methods}"
    estimator_mapping = {}
    cols = df.columns
    if classifier_overrides is None:
        classifier_overrides = []
    for col in cols:
        if col in classifier_overrides:
            est = RandomForestClassifier(n_estimators=50)
        else:
            est = RandomForestRegressor(n_estimators=50)
        estimator_mapping[col] = est
    
    ds = []
    if method=='mutual_information':
        mi=mutual_information(df)
        for (feature,row) in mi.iterrows():
            for (target,score) in row.iteritems():
                ds.append({'feature': feature, 'target': target, 'score': score})
    else:
        for idx_Y, target in enumerate(cols):
            est = estimator_mapping[target]
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
                    scores = cross_val_score(est, df_X, df_y, cv=3)#, n_jobs=-1)
                    score = scores.mean()
                    #score = max(score, 0.0) # set negative r^2 to 0
                if method in set(corr_methods):
                    pair = df_ml[[feature, target]]
                    assert pair.shape[1] == 2
                    score = pair.corr(method=method)[feature][target]
                    
                d = {'feature': feature, 'target': target, 'score': score}
                ds.append(d)

    df_results = pd.DataFrame(ds)
    return df_results

def mutual_information(df):
    n_cols=df.shape[1]
    mi=pd.DataFrame(numpy.zeros((n_cols,n_cols)),
                    index=df.columns,
                    columns=df.columns)
    for column in df.columns:
        if type_of_target(df[column].values)=='continuous':
            mi[column]=mutual_info_regression(df.values,df[column].values)
        else:
            mi[column]=mutual_info_classif(df.values,df[column].values)
    return mi

def greedy_bayes(data,k):
    columns=set(data.colums)
    x=columns.pop()
    network={x:data[x].value_counts(normalize=True)}
    mi=mutual_information(data)
    while len(columns)>0:
        X=None
        parents=None
        score=0
        for i in range(1,min(len(network),k)):
            for subset in itertools.combinations(network.keys(),i):
                for candidate in columns:
                    H=mi.loc[candidate,subset].sum()
                    if H>score:
                        X=candidate
                        parents=subset
                        score=H
        network[X]=data.groupby(parents)[X].apply(lambda x: x.value_counts(normalize=True))
        columns.discard(X)
    return network
        
            
            
    

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