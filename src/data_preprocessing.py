import os
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import logging
from sklearn.preprocessing import LabelEncoder
nltk.download("stopwords")
nltk.download("punkt")

log_dir="./LOG"
os.makedirs(log_dir , exist_ok=True)

logger=logging.getLogger("data_Preprocessing")
logger.setLevel("DEBUG")

console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir , "data_preprocessing.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")


formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text :str) -> str:
    ps=PorterStemmer()

    text = text.lower()

    text=nltk.word_tokenize(text)

    text = [word for word in text if word.isalnum()]

    text = [word for word in text if word not in stopwords.words("english") and word not in string.punctuation]

    text = [ps.stem(word) for word in text]

    return " ".join(text)


def preprocess_df(df : pd.DataFrame , text_column="text" , target_column="target"):

    try:
        encoder=LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.info("Label Encoding Done")

        df.drop_duplicates(keep="first" ,inplace=True)
        logger.info("Duplicates are removed")

        df.dropna(inplace=True)
        logger.info("Null values have been removed")

        df[text_column] = df[text_column].apply(transform_text)
        logger.info("The text has been transformed")

        return df

    except Exception as e:
        logger.error(f"Unexpected error has taken place {e}")



def main():

    try:

     train_data=pd.read_csv(r"D:\MLOPS\MLOPS-Pipeline-Project\MLOPS-Complete-Pipeline\data\raw\train.csv")
     test_data=pd.read_csv(r"D:\MLOPS\MLOPS-Pipeline-Project\MLOPS-Complete-Pipeline\data\raw\test.csv")
     logger.info("Data Loaded Successfully")


     train_df=preprocess_df(train_data ,"text" , "target")
     test_df=preprocess_df(test_data ,"text" , "target")

     Preprocess_dir = ".\data\processed_data"
     os.makedirs(Preprocess_dir , exist_ok=True)
     
     print(train_df.isna().sum())
     print(test_df.isna().sum())
     
     train_df.to_csv(os.path.join(Preprocess_dir ,"preprocessed_train_data.csv") ,index=False)
     test_df.to_csv(os.path.join(Preprocess_dir ,"preprocessed_test_data.csv") ,index=False)

     logger.info("The Preprocessed data has successfully been saved")

    except Exception as e:
        logger.error(f"An Error Occured during text preprocessing {e}")





if __name__ == "__main__":
    main()




