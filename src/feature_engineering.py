import os
import pandas as pd
import nltk
import logging

from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
nltk.download("stopwords")
nltk.download("punkt")

log_dir="./LOG"
os.makedirs(log_dir , exist_ok=True)

logger=logging.getLogger("feautre_Engineering")
logger.setLevel("DEBUG")

console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir , "feature_Engineering.log")
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
    df.fillna('',inplace=True)
    return df
    logger.info("Data has been succesully load and nan values are filled")

   except Exception as e:
      logger.error(f"Failed to load the data {e}")

def apply_tfidf(train_data , test_data , max_features):
    try:
 
        tfidf=TfidfVectorizer(max_features=max_features)
        X_train = train_data["text"].values
        Y_train = train_data["target"].values
        X_test=test_data["text"].values
        Y_test=test_data["target"].values

        

        X_train_transformed=tfidf.fit_transform(X_train)
        X_test_transfomrd=tfidf.transform(X_test)

        train_df=pd.DataFrame(X_train_transformed.toarray())
        train_df["labels"]=Y_train

        print(train_df.head())

        test_df=pd.DataFrame(X_test_transfomrd.toarray())
        test_df["labels"]=Y_test

        print(test_df.head())

        return train_df , test_df




    except Exception as e:
        logger.error(f"Error Occured while applying TFidf {e}")



def save_data(df , file_path):

    try:

     df.to_csv(file_path , index=False)

    except Exception as e :
       logger.error(f"File Cant Saved in feature Enginnering Process {e}")


def main():
   
   try:
   
    train_data=load_data(file_path=r"D:\MLOPS\MLOPS-Pipeline-Project\MLOPS-Complete-Pipeline\data\processed_data\preprocessed_train_data.csv")
    test_data=load_data(file_path=r"D:\MLOPS\MLOPS-Pipeline-Project\MLOPS-Complete-Pipeline\data\processed_data\preprocessed_test_data.csv")

    train_df ,test_df = apply_tfidf(train_data=train_data,test_data=test_data,max_features=50)

    tfidf_dir="./data/tfidf_processed"
    os.makedirs(tfidf_dir,exist_ok=True)

    save_data(train_df , os.path.join(tfidf_dir , "train_df_idf.csv"))
    save_data(test_df , os.path.join(tfidf_dir , "test_df_idf.csv"))

   except Exception as e:
      logger.error(f"Error has occured at feature Engineering Part")



if __name__=="__main__":
   main()




