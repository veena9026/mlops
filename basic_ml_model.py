import pandas as pd
import numpy as np
import os
import argparse

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,accuracy_score,roc_auc_score
from sklearn.model_selection import train_test_split




def get_data():
    URl="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

    # read the data as df
    try:
        df=pd.read_csv(URl,sep=";")
        return df
    except Exception as e:
        raise e

def evaluate(y_test,y_pred,pred_prob):
    #mae=mean_absolute_error(y_test,y_pred)
    #mse=mean_squared_error(y_test,y_pred)
    #rmse=np.sqrt=(mean_squared_error(y_test,y_pred))
    #r2=r2_score(y_test,y_pred)
    accuracy=accuracy_score(y_test,y_pred)
    roc_score=roc_auc_score(y_test,pred_prob,multi_class="ovr")
    return accuracy,roc_score


def main(n_estimators,max_depth):
    df=get_data()
   
    train,test=train_test_split(df,test_size=0.25,random_state=24)
   
    X_train=train.drop(labels=["quality"],axis=1)
    X_test=test.drop(labels=["quality"],axis=1)
   
    y_train=train[["quality"]]
    y_test=test[["quality"]]

    #lr=ElasticNet()
    #lr.fit(X_train,y_train)
    #y_pred=lr.predict(X_test)
    #mae,mse,rmse,r2=evaluate(y_test,y_pred)
    #print(f"mae :{mae}, mse : {mse}, rmse : {rmse},r2 score : {r2}")

    with mlflow.start_run():
    
        rf=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
        rf.fit(X_train,y_train)
        y_pred=rf.predict(X_test)

        pred_prob=rf.predict_proba(X_test)
        accuracy,roc_score=evaluate(y_test,y_pred,pred_prob)

        mlflow.log_param("n_estimators",n_estimators)
        mlflow.log_param("max_Depth ",max_depth)

        mlflow.log_metric("accuracy :",accuracy)
        mlflow.log_metric("roc_score :",roc_score)


        mlflow.sklearn.log_model(rf,"random forest classifier")
        print(f"accuacy :{accuracy},roc_score :{roc_score}")



if __name__ =="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--n_estimators","-n",default=50,type=int)
    args.add_argument("--max_depth","-m",default=5 ,type=int)
    parse_arg= args.parse_args()

    #print(parse_arg.n_estimators,parse_arg.max_depth)    

    

    

    try:
        main(n_estimators=parse_arg.n_estimators,max_depth=parse_arg.max_depth)

    except Exception as e:
        raise e