def remove_nan(X,y):
    temp=X.isna().any(axis=1)
    drop_indexes=temp[temp].index
    X.drop(index=drop_indexes,inplace=True)
    y.drop(index=drop_indexes,inplace=True)
    return X,y
