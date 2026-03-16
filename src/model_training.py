import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import logging
import pickle

log_dir="./LOG"
os.makedirs(log_dir , exist_ok=True)

logger=logging.getLogger("Model_Training")
logger.setLevel("DEBUG")

console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir , "Model_Training.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")


formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path):
    try:
        df=pd.read_csv(file_path)
        logging.info("DataBase loaded Successfully")
        return df
    
    except Exception as e:
        logging.error(f"Error Occured in Model_Training {e}")



def model_training(x_train , y_train , params):
    try:
        logger.debug("Model Random Forest is going to be initialized")

        random_forest=RandomForestClassifier(n_estimators=params["n_estimators"] , random_state=params["random_state"])

        random_forest.fit(x_train , y_train)

        logger.info(f"Model training is Completed")

        return random_forest


    except Exception as e:
        logger.error(f"Error Occured while Model Trainig {e}")


def save_model(model ,file_path):
    try:
        with open(file_path , 'wb') as f:
            pickle.dump(model,f)

        logger.info("Model has been saved")

    except Exception as e:
        logger.error(f"Model has been saved {e}")


def main():
    try:
     params = {"n_estimators" :25 , "random_state" :2}
     train_df=load_data(r"D:\MLOPS\MLOPS-Pipeline-Project\MLOPS-Complete-Pipeline\data\tfidf_processed\train_df_idf.csv")
     X_train =train_df.iloc[: , :-1]
     Y_train=train_df.iloc[: , -1]
     print(Y_train)
     model=model_training(X_train , Y_train ,params=params)
     model_dir="./models"
     os.makedirs(model_dir ,exist_ok=True)

     save_model(model=model ,file_path=os.path.join(model_dir , "Random_Forest_Classifier.pkl") )
     logger.info("Model has been saved")

    except Exception as e:
        logger.error(f"Error occured while model_training {e}")


if __name__ =="__main__":
    main()


    

