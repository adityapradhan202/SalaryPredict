import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path
from joblib import dump

def data_split_encode(data:pd.DataFrame, label:str, 
                      test_size:float, seed:int, encoder:OneHotEncoder,
                      encoder_path:str):
    """Splits the data into train test split, without any scaling, one hot encodes the categorical string columns, and saves the encoder which can used while making prediction on new unseen data point."""

    num_data = data.select_dtypes(include="number")
    obj_data = data.select_dtypes(include="object")
    encoded_obj_data = pd.get_dummies(data=obj_data, drop_first=True, dtype=int)
    encoder.fit(obj_data) # fit the encoder the categorical string cols

    encoded_df = pd.concat(objs=[num_data, encoded_obj_data], axis=1)
    X = encoded_df.drop(label, axis=1)
    y = encoded_df[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    try:
        encoder_path = Path(encoder_path)
        dump(value=encoder, filename=encoder_path)
        print(f"Encode is saved to path: {encoder_path}")
    except Exception as e:
        print(f"Some exception occured: {e}")
    
    return X_train, X_test, y_train, y_test


def data_constructor(data:pd.DataFrame, label:str, test_size:float, seed:int,
                     scaler:StandardScaler, encoder:OneHotEncoder, scaler_path:str, encoder_path:str):
    
    """
    A function that scales the numeric columns using standardization and one hot encode the categorical string columns, and also splits the dataset into train test split. It saves the encoder which can used while making prediction on new unseen data point.
    """
    num_data = data.select_dtypes(include="number")
    obj_data = data.select_dtypes(include="object")
    encoded_obj_data = pd.get_dummies(data=obj_data, dtype=int, drop_first=True)
    encoded_df = pd.concat(objs=[num_data, encoded_obj_data], axis=1)
    encoder.fit(encoded_obj_data)

    y = encoded_df[label]
    X = encoded_df.drop(labels=label, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    num_cols = list(num_data.columns)
    num_cols.remove(label)
    
    X_train_num = X_train[num_cols]
    X_train_other = X_train.drop(labels=num_cols, axis=1)
    X_test_num = X_test[num_cols]
    X_test_other = X_test.drop(labels=num_cols, axis=1)

    sc_train_num = scaler.fit_transform(X_train_num)
    sc_test_num = scaler.transform(X_test_num)
    sc_train_num = pd.DataFrame(data=sc_train_num, index=X_train_num.index, columns=X_train_num.columns)
    sc_test_num = pd.DataFrame(data=sc_test_num, index=X_test_num.index, columns=X_test_num.columns)

    X_train_f = pd.concat(objs=[sc_train_num, X_train_other], axis=1)
    X_test_f = pd.concat(objs=[sc_test_num, X_test_other], axis=1)

    encoder_path = Path(encoder_path)
    scaler_path = Path(scaler_path)
    try:
        dump(value=encoder, filename=encoder_path)
        dump(value=scaler, filename=scaler_path)
        print(f"Enocder has been saved to path: {encoder_path}")
        print(f"Scaler has been saved to path: {scaler_path}")
    except Exception as e:
        print(f"Some exception occured: {e}")

    return X_train_f, X_test_f, y_train, y_test

    
if __name__ == "__main__":
    df = pd.read_csv("data/salary_cleaned.csv")
    scaler_obj = StandardScaler()
    encoder_obj = OneHotEncoder(categories="auto", handle_unknown="ignore", drop="first", dtype=int)
    X_train_f, X_test_f, y_train, y_test = data_constructor(
        data=df,
        label="Salary",
        test_size=0.30,
        seed=42,
        scaler=scaler_obj,
        encoder=encoder_obj,
        scaler_path="data/scaler_obj.joblib",
        encoder_path="data/fitted_encoder.joblib"
    )

    print(f"\nX_train_f data:")
    print(X_train_f)
    print(f"\nX_test_f data:")
    print(X_test_f)
    print(f"\ny_train data:")
    print(y_train)
    print(f"\ny_test data")
    print(y_test)

    print("Shapes:")
    print(f"Shape of X_train_f: {X_train_f.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of X_test_f: {X_test_f.shape}")
    print(f"Shape of y_test: {y_test.shape}")


    


    

