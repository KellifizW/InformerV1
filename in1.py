import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from models.model import Informer
from exp.exp_informer import Exp_Informer
from utils.tools import StandardScaler, adjust_learning_rate, EarlyStopping
from utils.timefeatures import time_features
import plotly.graph_objs as go
import warnings
import traceback

warnings.filterwarnings('ignore')


# 自定義數據集類，適配 yfinance
class Dataset_Stock(Dataset):
    def __init__(self, ticker, start_date, end_date, flag='train', size=[252, 30, 7],
                 features='S', target='Close', scale=True, timeenc=1, freq='d'):
        self.seq_len, self.label_len, self.pred_len = size
        assert flag in ['train', 'val', 'test', 'pred']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'pred': 3}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.__read_data__()

    def __read_data__(self):
        try:
            df_raw = yf.download(self.ticker, start=self.start_date, end=self.end_date)
            if df_raw.empty:
                raise ValueError(f"無法從 Yahoo Finance 下載 {self.ticker} 的數據，請檢查股票代碼或網絡連接。")
            df_raw = df_raw.reset_index()
            df_raw['date'] = pd.to_datetime(df_raw['Date'])

            # 檢查數據長度
            required_len = self.seq_len + self.pred_len
            if len(df_raw) < required_len:
                raise ValueError(
                    f"數據長度 ({len(df_raw)}) 不足以支持 seq_len={self.seq_len} 和 pred_len={self.pred_len}，請選擇更短的歷史數據或預測範圍。")

            num_train = int(len(df_raw) * 0.7)
            num_val = int(len(df_raw) * 0.15)
            num_test = len(df_raw) - num_train - num_val
            border1s = [0, num_train - self.seq_len, num_train + num_val - self.seq_len, len(df_raw) - self.seq_len]
            border2s = [num_train, num_train + num_val, len(df_raw), len(df_raw)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            if self.features == 'M':
                cols_data = ['Open', 'High', 'Low', 'Close', 'Volume']
                df_data = df_raw[cols_data]
            elif self.features == 'S':
                df_data = df_raw[[self.target]]

            self.scaler = StandardScaler()
            if self.scale:
                train_data = df_data[:border2s[0]].values
                self.scaler.fit(train_data)
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values

            df_stamp = df_raw[['date']][border1:border2]
            if self.set_type == 3:  # pred 模式
                pred_dates = pd.date_range(df_stamp['date'].iloc[-1], periods=self.pred_len + 1, freq=self.freq)[1:]
                df_stamp = pd.DataFrame({'date': list(df_stamp['date']) + list(pred_dates)})
            data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]
            self.data_stamp = data_stamp
            self.dates = df_stamp['date'].values
        except Exception as e:
            st.error(f"數據加載錯誤：{str(e)}")
            raise

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end] if self.set_type != 3 else self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - (self.pred_len if self.set_type != 3 else 0) + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# Streamlit 應用
st.title("Informer 股票價格預測")

# 用戶輸入
ticker = st.text_input("輸入股票代碼 (例如 AAPL)", "AAPL")
historical_years = st.selectbox("觀察過去的歷史數據", ["1 年", "2 年", "3 年"])
seq_len_map = {"1 年": 252, "2 年": 504, "3 年": 756}
seq_len = seq_len_map[historical_years]

label_len_option = st.selectbox("從最後多少天開始初始化預測", ["30 天", "60 天", "90 天"])
label_len_map = {"30 天": 30, "60 天": 60, "90 天": 90}
label_len = label_len_map[label_len_option]

pred_len_option = st.selectbox("預測未來多少天", ["3 天", "5 天", "7 天", "10 天"])
pred_len_map = {"3 天": 3, "5 天": 5, "7 天": 7, "10 天": 10}
pred_len = pred_len_map[pred_len_option]

# 設置參數
args = type('Args', (), {
    'model': 'informer',
    'data': 'custom',
    'features': 'S',
    'target': 'Close',
    'freq': 'd',
    'checkpoints': './checkpoints/',
    'seq_len': seq_len,
    'label_len': label_len,
    'pred_len': pred_len,
    'enc_in': 1,
    'dec_in': 1,
    'c_out': 1,
    'd_model': 512,
    'n_heads': 8,
    'e_layers': 2,
    'd_layers': 1,
    'd_ff': 2048,
    'factor': 5,
    'padding': 0,
    'distil': True,
    'dropout': 0.05,
    'attn': 'prob',
    'embed': 'timeF',
    'activation': 'gelu',
    'output_attention': False,
    'mix': True,
    'num_workers': 0,
    'train_epochs': 6,
    'batch_size': 32,
    'patience': 3,
    'learning_rate': 0.0001,
    'loss': 'mse',
    'lradj': 'type1',
    'use_gpu': torch.cuda.is_available(),
    'gpu': 0,
    'use_multi_gpu': False,
    'inverse': True,
    'detail_freq': 'd'
})()

# 數據下載與展示
start_date = pd.Timestamp.now() - pd.Timedelta(days=int(seq_len_map[historical_years] * 1.2))
end_date = pd.Timestamp.now()
if st.button("獲取數據並展示歷史價格"):
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            st.error(f"無法獲取 {ticker} 的歷史數據，請檢查股票代碼是否正確或網絡是否正常。")
        else:
            st.write(f"{ticker} 的歷史數據（從 {start_date.date()} 到 {end_date.date()}）")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
            fig.update_layout(title=f"{ticker} 歷史收盤價", xaxis_title="日期", yaxis_title="價格")
            st.plotly_chart(fig)
    except Exception as e:
        st.error(f"歷史數據展示錯誤：{str(e)}")
        st.write("詳細錯誤信息：")
        st.write(traceback.format_exc())

# 訓練與保存模型
if st.button("訓練模型並保存"):
    try:
        train_data = Dataset_Stock(ticker, start_date, end_date, flag='train',
                                   size=[args.seq_len, args.label_len, args.pred_len])
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        exp = Exp_Informer(args)
        setting = f"{args.model}_{ticker}_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}"
        st.write(f"開始訓練模型：{setting}")
        exp.train(setting)
        st.success("訓練完成，模型已保存至 checkpoints/")
        st.session_state['trained'] = True
        st.session_state['setting'] = setting
    except Exception as e:
        st.error(f"模型訓練錯誤：{str(e)}")
        st.write("詳細錯誤信息：")
        st.write(traceback.format_exc())

# 上載預訓練模型並預測
uploaded_file = st.file_uploader("上載預訓練模型 (.pth)", type="pth")
if uploaded_file or ('trained' in st.session_state and st.session_state['trained']):
    try:
        pred_data = Dataset_Stock(ticker, start_date, end_date, flag='pred',
                                  size=[args.seq_len, args.label_len, args.pred_len])
        pred_loader = DataLoader(pred_data, batch_size=1, shuffle=False)
        exp = Exp_Informer(args)

        if uploaded_file:
            checkpoint_path = os.path.join(args.checkpoints, "uploaded_model.pth")
            os.makedirs(args.checkpoints, exist_ok=True)
            with open(checkpoint_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            exp.model.load_state_dict(torch.load(checkpoint_path))
            setting = "uploaded_model"
        elif 'setting' in st.session_state:
            setting = st.session_state['setting']
            checkpoint_path = os.path.join(args.checkpoints, setting, 'checkpoint.pth')
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"找不到模型文件：{checkpoint_path}")
            exp.model.load_state_dict(torch.load(checkpoint_path))

        st.write(f"使用模型 {setting} 進行預測")
        exp.predict(setting, load=True)

        pred_file = f'./results/{setting}/real_prediction.npy'
        if not os.path.exists(pred_file):
            raise FileNotFoundError(f"預測結果文件未找到：{pred_file}")
        preds = np.load(pred_file)
        preds = pred_data.inverse_transform(preds[0])

        last_historical = pred_data.data_y[-args.seq_len:]
        last_historical = pred_data.inverse_transform(last_historical)
        pred_dates = pd.date_range(end_date, periods=args.pred_len + 1, freq='D')[1:]
        historical_dates = pred_data.dates[-args.seq_len:]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=historical_dates, y=last_historical[:, 0], mode='lines', name='歷史價格'))
        fig.add_trace(go.Scatter(x=pred_dates, y=preds[:, 0], mode='lines', name='預測價格', line=dict(dash='dash')))
        fig.update_layout(title=f"{ticker} 股票價格預測", xaxis_title="日期", yaxis_title="價格")
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"預測錯誤：{str(e)}")
        st.write("詳細錯誤信息：")
        st.write(traceback.format_exc())
