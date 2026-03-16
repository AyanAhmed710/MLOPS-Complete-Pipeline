import os
import pandas as pd
from sklearn.metrics import accuracy_score , precision_score ,recall_score ,roc_auc_score
import logging 
import pickle
import json

log_dir="./LOG"
os.makedirs(log_dir , exist_ok=True)

logger=logging.getLogger("Model_Evaluation")
logger.setLevel("DEBUG")

console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir , "Model_Evaluation.log")
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

def load_model(model_path):
    try:
        logger.info("Model is started loading")
        with open(model_path ,"rb") as f:
           model=pickle.load(f)
        logging.info("Model has been loaded")
        return model
      
    except Exception as e:
        logger.error(f"Model is not been able to load")


def evaluate_model(X_test ,Y_test ,model):
    try:
        y_pred=model.predict(X_test)

        accuracy =accuracy_score(Y_test , y_pred)
        precision=precision_score(Y_test , y_pred)
        recall=recall_score(Y_test , y_pred)
        roc_auc=roc_auc_score(Y_test , y_pred)

        metrics_dict = {"accuracy" :accuracy , "pricision" :precision , "recall" :recall ,"auc" :roc_auc}
        logging.info("Metrics has been generated Successfully")

        return metrics_dict
    
       


    except Exception as e:
        logging.error(f"Error occured while evaluating model")


def save_metrics(file_path,metrics):
    try:
        with open(file_path , 'w') as f:
            json.dump(metrics ,f ,indent=4)

        logging.info("Metrics file has been saved Successfully")

    except Exception as e:
        logging.error(f"Error Occured while saving metircs {e}")


def main():
    try:
     df_test =load_data(r"D:\MLOPS\MLOPS-Pipeline-Project\MLOPS-Complete-Pipeline\data\tfidf_processed\test_df_idf.csv")
     model=load_model(r"D:\MLOPS\MLOPS-Pipeline-Project\MLOPS-Complete-Pipeline\models\Random_Forest_Classifier.pkl")
     X_test=df_test.iloc[: , :-1]
     Y_test = df_test.iloc[:,-1]
     metrics_dict=evaluate_model(Y_test=Y_test ,X_test=X_test ,model=model)
     report_dir="./reports"
     os.makedirs(report_dir,exist_ok=True)
     save_metrics(os.path.join(report_dir , "Random_Forest1.json") , metrics_dict)

    except Exception as e:
        logging.error(f"Error Occurs while evaluating the model {e}")



if __name__ =="__main__":
    main()



    