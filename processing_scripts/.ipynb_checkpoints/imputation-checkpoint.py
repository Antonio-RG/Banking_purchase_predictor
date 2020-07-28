
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from pathlib import Path

if __name__=="__main__":
    train_dir = '/opt/ml/processing/input/train'
    test_dir = '/opt/ml/processing/input/test'
    seed=0
    train_df = pd.read_csv(os.path.join(train_dir, 'train.csv'), index_col='ID')
    test_df = pd.read_csv(os.path.join(test_dir, 'test.csv'), index_col='ID')

    print('Scaling Data')
    std_scale = preprocessing.StandardScaler().fit(train_df.iloc[:, 1:])
    train_df_scaled = std_scale.transform(train_df.iloc[:, 1:])
    test_df_scaled = std_scale.transform(test_df)
    
    print('Training Data Imputation Model')
    imputer = IterativeImputer(random_state=seed, missing_values=0)
    train_imputed = imputer.fit_transform(train_df_scaled)
    
    # Transforming test data
    test_imputed = imputer.transform(test_df_scaled)
    
    train_imputed_output_path = os.path.join('/opt/ml/processing/train', 'train_imputed.csv')
    test_imputed_output_path = os.path.join('/opt/ml/processing/test', 'test_imputed.csv')
    
    pd.concat([train_df['target'], pd.DataFrame(train_imputed, columns=train_df.columns, index=train_df.index)],  axis=1).to_csv(train_imputed_output_path, index=True, header=True)
    
    pd.DataFrame(test_imputed, columns=test_df.columns, index=test_df.index).to_csv(test_features_output_path, index=True, header=True)