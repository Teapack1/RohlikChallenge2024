from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import joblib
import pandas as pd

def preprocess_data(dataframe, x_scaler, ohsc, cat_col):

    # Convert the date column to datetime
    dataframe['date'] = pd.to_datetime(dataframe['date'])
    
    # Encode categorical features
    encoded_categorical_features = ohsc.transform(dataframe[cat_col])
    encoded_feature_names = ohsc.get_feature_names_out(cat_col)
    dataframe_encoded = pd.DataFrame(encoded_categorical_features, columns=encoded_feature_names, index=dataframe.index)
    if 'holiday_name_nan' in dataframe_encoded.columns:
        dataframe_encoded = dataframe_encoded.drop(columns=['holiday_name_nan'])
    
    
    
    # Concatenate the encoded features with the original DataFrame
    dataframe = pd.concat([dataframe, dataframe_encoded], axis=1)
    
    # Drop the original categorical columns
    dataframe = dataframe.drop(columns=cat_col)
    
    # Extract date features
    dataframe.index = dataframe.pop('date')

    df_trans = x_scaler.transform(dataframe)
    dataframe = pd.DataFrame(df_trans, columns=dataframe.columns)
    
    
    
    return dataframe