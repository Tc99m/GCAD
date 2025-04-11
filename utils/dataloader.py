import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class SwatDataLoader_AD:
    """Generate data loader from raw data."""

    def __init__(
          self, data, batch_size, seq_len, pred_len, feature_type, target='OT', stride=1
        ):
        self.data = data
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_type = feature_type
        self.target = target
        self.target_slice = slice(0, None)
        self.stride = stride

        self._read_data()

    def _read_data(self):
        """Load raw data and split datasets."""
        df_raw = pd.read_csv(self.data+'/train.csv', index_col=0)
        df_test_raw = pd.read_csv(self.data+'/test.csv', index_col=0)

        df = df_raw.iloc[:,:-1]
        df_test_labels = df_test_raw.iloc[:,-1]
        df_test_value = df_test_raw.iloc[:,:-1]

        # split train/valid/test
        n = len(df)

        train_end = int(n * 0.8)
        val_end = n
        # val_end = n - int(n * 0.2)

        train_df = df[:train_end]
        val_df = df[train_end - self.seq_len : val_end]
        test_df = df_test_value

        # standardize by training set
        self.scaler = StandardScaler()
        self.scaler.fit(train_df.values)

        def scale_df(df, scaler):
            data = scaler.transform(df.values)
            return pd.DataFrame(data, index=df.index, columns=df.columns)

        self.train_df = scale_df(train_df, self.scaler)
        self.val_df = scale_df(val_df, self.scaler)
        self.test_df = scale_df(test_df, self.scaler)

        # self.train_df = train_df
        # self.val_df = val_df
        # self.test_df = test_df   

        self.test_labels = df_test_labels

        self.n_feature = self.train_df.shape[-1]

    def _make_dataset(self, data, shuffle=True, testing=False, test_labels=None, stride=1):
        data = np.array(data, dtype=np.float32)

        data_x = torch.tensor(data, dtype=torch.float32)
        data_y = torch.tensor(data[:, self.target_slice], dtype=torch.float32)

        if testing:
            test_labels = np.array(test_labels, dtype=np.float32)
            test_labels = torch.tensor(test_labels, dtype=torch.float32)

        if testing:
            return DataLoader(
                torch.utils.data.Subset(
                    CustomDataset(data_x, data_y, self.seq_len, self.pred_len, testing=True, test_labels=test_labels),
                    range(0,len(data_x) - self.seq_len - self.pred_len + 1, stride)
                ),
                batch_size=self.batch_size, 
                shuffle=shuffle,
                drop_last=True
            )
        else:
            return DataLoader(
                torch.utils.data.Subset(
                    CustomDataset(data_x, data_y, self.seq_len, self.pred_len),
                    range(len(data_x) - self.seq_len - self.pred_len + 1)
                ),
                batch_size=self.batch_size, 
                shuffle=shuffle,
                drop_last=True
            )

    # def inverse_transform(self, data):
    #     return self.scaler.inverse_transform(data)

    def get_train(self, shuffle=True):
        return self._make_dataset(self.train_df, shuffle=shuffle)

    def get_val(self):
        return self._make_dataset(self.val_df, shuffle=False)

    def get_test(self):
        return self._make_dataset(self.test_df, shuffle=False, testing=True, test_labels=self.test_labels,stride=self.stride)

class smdDataLoader_AD:
    """Generate data loader from raw data."""

    def __init__(
          self, data, batch_size, seq_len, pred_len, feature_type, target='OT'
        ):
        self.data = data
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_type = feature_type
        self.target = target
        self.target_slice = slice(0, None)

        self._read_data()

    def _read_data(self):
        """Load raw data and split datasets."""
        df_raw = pd.read_csv(self.data+'/train.csv', index_col=0)
        df_test_raw = pd.read_csv(self.data+'/test.csv', index_col=0)

        df = df_raw
        df_test_labels = df_test_raw.iloc[:,-1]
        df_test_value = df_test_raw.iloc[:,:-1]

        # split train/valid/test
        n = len(df)

        train_end = int(n * 0.8)
        val_end = n
        # val_end = n - int(n * 0.2)

        train_df = df[:train_end]
        val_df = df[train_end - self.seq_len : val_end]
        test_df = df_test_value

        # standardize by training set
        self.scaler = StandardScaler()
        self.scaler.fit(train_df.values)

        def scale_df(df, scaler):
            data = scaler.transform(df.values)
            return pd.DataFrame(data, index=df.index, columns=df.columns)

        self.train_df = scale_df(train_df, self.scaler)
        self.val_df = scale_df(val_df, self.scaler)
        self.test_df = scale_df(test_df, self.scaler)

        # self.train_df = train_df
        # self.val_df = val_df
        # self.test_df = test_df

        self.test_labels = df_test_labels

        self.n_feature = self.train_df.shape[-1]

    def _make_dataset(self, data, shuffle=True, testing=False, test_labels=None, train_average=False):
        data = np.array(data, dtype=np.float32)

        data_x = torch.tensor(data, dtype=torch.float32)
        data_y = torch.tensor(data[:, self.target_slice], dtype=torch.float32)

        if testing:
            test_labels = np.array(test_labels, dtype=np.float32)
            test_labels = torch.tensor(test_labels, dtype=torch.float32)

        if testing:
            return DataLoader(
                torch.utils.data.Subset(
                    CustomDataset(data_x, data_y, self.seq_len, self.pred_len, testing=True, test_labels=test_labels),
                    range(len(data_x) - self.seq_len - self.pred_len + 1)
                ),
                batch_size=self.batch_size, 
                # batch_size=1,
                shuffle=False,
                drop_last=True
            )

        else:
            return DataLoader(
                torch.utils.data.Subset(
                    CustomDataset(data_x, data_y, self.seq_len, self.pred_len),
                    range(len(data_x) - self.seq_len - self.pred_len + 1)
                ),
                batch_size=self.batch_size, 
                shuffle=shuffle,
                drop_last=True
            )


    def get_train(self, shuffle=True):
        return self._make_dataset(self.train_df, shuffle=shuffle)

    def get_val(self):
        return self._make_dataset(self.val_df, shuffle=False)

    def get_test(self):
        return self._make_dataset(self.test_df, shuffle=False, testing=True, test_labels=self.test_labels)

class CustomDataLoader:
    """Generate data loader from raw data."""

    def __init__(
          self, data, batch_size, seq_len, pred_len, feature_type, target='OT'
        ):
        self.data = data
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_type = feature_type
        self.target = target
        self.target_slice = slice(0, None)

        self._read_data()

    def _read_data(self):
        """Load raw data and split datasets."""
        df_raw = pd.read_csv(self.data)

        # S: univariate-univariate, M: multivariate-multivariate, MS:
        # multivariate-univariate
        df = df_raw.set_index('date')
        if self.feature_type == 'S':
            df = df[[self.target]]
        elif self.feature_type == 'MS':
            target_idx = df.columns.get_loc(self.target)
            self.target_slice = slice(target_idx, target_idx + 1)

        # split train/valid/test
        n = len(df)
        if self.data.stem.startswith('ETTm'):
            train_end = 12 * 30 * 24 * 4
            val_end = train_end + 4 * 30 * 24 * 4
            test_end = val_end + 4 * 30 * 24 * 4
        elif self.data.stem.startswith('ETTh'):
            train_end = 12 * 30 * 24
            val_end = train_end + 4 * 30 * 24
            test_end = val_end + 4 * 30 * 24
        else:
            train_end = int(n * 0.7)
            val_end = n - int(n * 0.2)
            test_end = n
        train_df = df[:train_end]
        val_df = df[train_end - self.seq_len : val_end]
        test_df = df[val_end - self.seq_len : test_end]

        # standardize by training set
        self.scaler = StandardScaler()
        self.scaler.fit(train_df.values)

        def scale_df(df, scaler):
            data = scaler.transform(df.values)
            return pd.DataFrame(data, index=df.index, columns=df.columns)

        self.train_df = scale_df(train_df, self.scaler)
        self.val_df = scale_df(val_df, self.scaler)
        self.test_df = scale_df(test_df, self.scaler)
        self.n_feature = self.train_df.shape[-1]

    def _make_dataset(self, data, shuffle=True):
        data = np.array(data, dtype=np.float32)

        data_x = torch.tensor(data, dtype=torch.float32)
        data_y = torch.tensor(data[:, self.target_slice], dtype=torch.float32)
            
        return DataLoader(
            torch.utils.data.Subset(
                CustomDataset(data_x, data_y, self.seq_len, self.pred_len),
                range(len(data_x) - self.seq_len - self.pred_len + 1)
            ),
            batch_size=self.batch_size, 
            shuffle=shuffle
        )

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def get_train(self, shuffle=True):
        return self._make_dataset(self.train_df, shuffle=shuffle)

    def get_val(self):
        return self._make_dataset(self.val_df, shuffle=False)

    def get_test(self):
        return self._make_dataset(self.test_df, shuffle=False)
    

class CustomDataset(Dataset):
    def __init__(self, data_x, data_y, seq_len, pred_len, testing=False, test_labels=None):
        self.data_x = data_x
        self.data_y = data_y
        self.test_labels  = test_labels
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.testing = testing

    def __len__(self):
        return self.data_x.shape[0]

    def __getitem__(self, idx):
        '''data_x.shape: (batch_size, input_len, num_features)
           data_y.shape: (batch_size, pred_len, num_features)
           test_labels.shape(batch_size, input_len+pred_len)
        '''
        if self.testing:
            return self.data_x[idx : idx + self.seq_len], self.data_y[idx + self.seq_len : idx + self.seq_len + self.pred_len], self.test_labels[idx : idx + self.seq_len + self.pred_len]
        else:
            return self.data_x[idx : idx + self.seq_len], self.data_y[idx + self.seq_len : idx + self.seq_len + self.pred_len]