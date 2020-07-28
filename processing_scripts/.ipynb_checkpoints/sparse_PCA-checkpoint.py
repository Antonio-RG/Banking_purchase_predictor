import pandas as pd
from sklearn import preprocessing, decomposition, model_selection
from scipy import stats
import argparse
import os

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_components', type=int, default=200)
    args, _ = parser.parse_known_args()
    
    train_dir = '/opt/ml/processing/input/train'
    test_dir = '/opt/ml/processing/input/test'
    seed=0
    train_df = pd.read_csv(os.path.join(train_dir, 'train.csv'), index_col='ID')

    # Transforming target variable to normal distribution
    train_df['target'] = stats.boxcox(train_df['target'])[0]
    
    # Scaling train data
    print('Scaling Train Data')
    std_scale = preprocessing.StandardScaler().fit(train_df.iloc[:, 1:])
    train_df_scaled = std_scale.transform(train_df.iloc[:, 1:])
    
    # Fitting and transforming train data with sparse PCA
    print('Fitting and transforming Train Data with Sparse PCA algorithm')
    n_components = args.n_components
    
    sparse_sm = decomposition.SparsePCA(n_components=n_components)
    sparse_train = sparse_sm.fit_transform(train_df_scaled)
    
    train_PCA_output_path = os.path.join('/opt/ml/processing/train', 'train_sparse_pca.csv')
    test_PCA_output_path = os.path.join('/opt/ml/processing/test', 'test_sparse_pca.csv')
    
    # Saving transformed train data
    pd.concat([train_df['target'], pd.DataFrame(sparse_train, columns=['c{}'.format(num+1) for num in range(n_components)], index=train_df.index)], axis=1).to_csv(train_PCA_output_path, header=True, index=True)
    
    # Transforming test data using fitted scaler and sparse PCA model
    print('Transforming Test Data')
    
    chunksize = 2000
    header = True
    
    for chunk in pd.read_csv(os.path.join(test_dir, 'test.csv'), index_col='ID', chunksize=chunksize):
        scaled_chunk = std_scale.transform(chunk.values)
        transformed_chunk = pd.DataFrame(sparse_sm.transform(scaled_chunk), columns=['c{}'.format(num+1) for num in range(n_components)], index=chunk.index)
        transformed_chunk.to_csv(test_PCA_output_path, header=header, mode='a')
        header = False