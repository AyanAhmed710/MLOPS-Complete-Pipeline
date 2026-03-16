import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

log_dir="./LOG"
os.makedirs(log_dir , exist_ok=True)

logger=logging.getLogger("data_ingestion")
logger.setLevel("DEBUG")

console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir , "data_ingestion.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")


formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path):
    try:
        with open(params_path , 'r') as file:
           params= yaml.safe_load(file)

        logger.info("Params has been successfully Loaded")

        return params

    except Exception as e:
        logger.error(f"File is not been able to open {e}")



def load_data(data_url : str) -> pd.DataFrame:
    try :
        df = pd.read_csv(data_url)
        logger.debug(f"DataBase has been created successfully %s {data_url}" )
        return df

    except Exception as e:
        logger.error(f"Unexpected error has taken place {e}")


def preprocess_data(df : pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=["Unnamed: 2" ,"Unnamed: 3" , "Unnamed: 4"] ,inplace=True)
        df.rename(columns={"v1" :"target" , "v2" : "text"},inplace=True)
        logger.info("The task as been completed")
        return df
        



    except Exception as e:
        logger.error(f"Unexpected error occurs {e}")


def save_data(train_data : pd.DataFrame,test_data : pd.DataFrame,data_path):
    try:
      data_path = os.path.join(data_path , 'raw')
      os.makedirs(data_path ,exist_ok=True)
      train_data.to_csv(os.path.join(data_path , 'train.csv') , index=False)
      test_data.to_csv(os.path.join(data_path , 'test.csv') , index=False)
      logging.info("The data has been saved Successfully")
    except Exception as e:
        logger.error(f"Unexpected error occurs {e}")


def main():
    try:
     params=load_params(r"D:\MLOPS\MLOPS-Pipeline-Project\MLOPS-Complete-Pipeline\params.yaml")
     test_size=params['data_ingestion']["test_size"]
     random_state=42
     data_path='https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv'
     df=load_data(data_url=data_path)
     processed_df=preprocess_data(df)
     train_df , test_df = train_test_split(processed_df , test_size=test_size ,random_state=random_state)
     data_dir = "./data"
     os.makedirs(data_dir , exist_ok=True)
     save_data(train_data=train_df , test_data=test_df ,data_path=data_dir )
     logging.info("Data Ingetsion Module is Completed")


    except Exception as e:
        logging.error(f"Unexpected Error has taen place {e}")


if __name__ == "__main__":
    main()





