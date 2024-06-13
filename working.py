from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import joblib
import pandas as pd

def preprocess_data(dataframe, categorical_columns, numerical_columns, date_columns, loaded_oh_encoder, loaded_num_scaler, loaded_date_scaler):

    # Convert the date column to datetime
    dataframe['date'] = pd.to_datetime(dataframe['date'])
    
    # Encode categorical features
    encoded_categorical_features = loaded_oh_encoder.transform(dataframe[categorical_columns])
    encoded_feature_names = loaded_oh_encoder.get_feature_names_out(categorical_columns)
    dataframe_encoded = pd.DataFrame(encoded_categorical_features, columns=encoded_feature_names, index=dataframe.index)
    dataframe_encoded = dataframe_encoded.drop(columns=['holiday_name_nan'])
    
    # Concatenate the encoded features with the original DataFrame
    dataframe = pd.concat([dataframe, dataframe_encoded], axis=1)
    
    # Drop the original categorical columns
    dataframe = dataframe.drop(columns=categorical_columns)
    
    # Extract date features
    dataframe.index = dataframe.pop('date')
    dataframe['year'] = dataframe.index.year
    dataframe['month'] = dataframe.index.month
    dataframe['day'] = dataframe.index.day
    dataframe['day_of_week'] = dataframe.index.dayofweek
    dataframe['is_weekend'] = dataframe['day_of_week'].isin([5, 6]).astype(int)
    
    # Scale numerical features
    if numerical_columns:
        dataframe[numerical_columns] = loaded_num_scaler.transform(dataframe[numerical_columns])
    
    # Scale date features
    if date_columns:
        dataframe[date_columns] = loaded_date_scaler.transform(dataframe[date_columns])
    
    
    return dataframe