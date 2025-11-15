import pandas as pd
import numpy as np
import datetime as dt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import yfinance as yf
import os 
import sys 

def run_lstm_prediction(user_ticker_symbol):
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph() 

    ticker = user_ticker_symbol.strip().upper() + ".NS"  
    start_date = "2002-01-01" 
    end_date = dt.date.today()

    num_unrollings = 50 
    test_set_size_ratio = 0.10 

    print(f"Downloading historical data for {ticker} from {start_date} to {end_date}...")
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    except Exception as e:
        print(f"An error occurred during data download: {e}")
        return

    if df.empty:
        print(f"\nInvalid Stock Ticker: '{user_ticker_symbol.strip().upper()}' is not a valid NSE stock symbol recognized by the data source.")
        print("Please use the base symbol without the '.NS' suffix (e.g., 'RELIANCE', 'HDFCBANK', 'TCS').")
        return

    df = df.reset_index().sort_values('Date').reset_index(drop=True)
    df['Date'] = df['Date'].astype(str)

    data_length = len(df)
    min_train_size = num_unrollings * 3 
    min_test_size = num_unrollings + 1 

    split_index = max(min_train_size, int(data_length * (1 - test_set_size_ratio)))

    if data_length < split_index + min_test_size:
        print(f"Insufficient data ({data_length} points). Required total points: {split_index + min_test_size}. Try an earlier start_date or a different stock.")
        return

    print(f"Data for {user_ticker_symbol} loaded. Total points: {data_length}. Training points: {split_index}. Test points: {data_length - split_index}.")
    
    high_prices = df.loc[:,'High'].to_numpy()
    low_prices = df.loc[:,'Low'].to_numpy()
    mid_prices = (high_prices + low_prices) / 2.0
    D = 1 

    train_data = mid_prices[:split_index].reshape(-1, D)
    test_data = mid_prices[split_index:].reshape(-1, D)

    scaler = MinMaxScaler()
    smoothing_window_size = 2500
    total_train_size = len(train_data)

    for di in range(0, total_train_size, smoothing_window_size):
        end_index = min(di + smoothing_window_size, total_train_size)
        scaler.fit(train_data[di:end_index,:])
        train_data[di:end_index,:] = scaler.transform(train_data[di:end_index,:])

    train_data = train_data.reshape(-1)
    test_data = scaler.transform(test_data.copy()).reshape(-1) 

    EMA = 0.0
    gamma = 0.1
    for ti in range(len(train_data)):
        EMA = gamma * train_data[ti] + (1 - gamma) * EMA
        train_data[ti] = EMA

    all_mid_data = np.concatenate([train_data, test_data], axis=0)

    class DataGenerator(object):
        def __init__(self, prices, batch_size, num_unrollings):
            self._prices = prices
            self._prices_length = len(self._prices)
            self._batch_size = batch_size
            self._num_unrollings = num_unrollings
            self._cursor = [offset * (self._prices_length // batch_size) for offset in range(batch_size)]

        def next_batch(self):
            batch_data = np.zeros((self._batch_size, self._num_unrollings, D), dtype=np.float32)
            batch_labels = np.zeros((self._batch_size, D), dtype=np.float32)

            for b in range(self._batch_size):
                if self._cursor[b] + self._num_unrollings + 1 > self._prices_length:
                    self._cursor[b] = np.random.randint(0, self._prices_length - self._num_unrollings - 1)

                batch_data[b, :, 0] = self._prices[self._cursor[b]:self._cursor[b] + self._num_unrollings]
                batch_labels[b, 0] = self._prices[self._cursor[b] + self._unrollings]

                self._cursor[b] = (self._cursor[b] + 1) % (self._prices_length - self._unrollings - 1) 
            return batch_data, batch_labels

    batch_size = 50
    dg = DataGenerator(train_data, batch_size=batch_size, num_unrollings=num_unrollings)

    num_nodes = [100, 100, 100]
    n_layers = len(num_nodes)
    dropout = 0.2
    epochs = 30
    starter_learning_rate = 0.001

    train_inputs, train_outputs = [], []
    for ui in range(num_unrollings):
        train_inputs.append(tf.compat.v1.placeholder(tf.float32, shape=[batch_size, D], name=f'train_inputs_{ui}'))
        train_outputs.append(tf.compat.v1.placeholder(tf.float32, shape=[batch_size, D], name=f'train_outputs_{ui}'))

    lstm_cells = [
        tf.keras.layers.LSTMCell(units=num_nodes[li]) for li in range(n_layers)
    ]
    drop_lstm_cells = [tf.compat.v1.nn.rnn_cell.DropoutWrapper(
        lstm, input_keep_prob=1.0, output_keep_prob=1.0-dropout, state_keep_prob=1.0-dropout
    ) for lstm in lstm_cells]
    drop_multi_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(drop_lstm_cells)

    w = tf.compat.v1.get_variable('w', shape=[num_nodes[-1], D], initializer=tf.compat.v1.glorot_uniform_initializer(seed=42))
    b = tf.compat.v1.get_variable('b', initializer=tf.compat.v1.random_uniform_initializer(-0.1, 0.1))

    initial_state = []
    for li in range(n_layers):
        c = tf.compat.v1.Variable(tf.zeros([batch_size, num_nodes[li]]), trainable=False)
        h = tf.compat.v1.Variable(tf.zeros([batch_size, num_nodes[li]]), trainable=False)
        initial_state.append(tf.compat.v1.nn.rnn_cell.LSTMStateTuple(c, h))

    all_inputs = tf.concat([tf.expand_dims(t, 0) for t in train_inputs], axis=0)
    all_lstm_outputs, state = tf.compat.v1.nn.dynamic_rnn(
        drop_multi_cell, all_inputs, initial_state=tuple(initial_state), time_major=True, dtype=tf.float32
    )

    all_lstm_outputs = tf.reshape(all_lstm_outputs, [batch_size * num_unrollings, num_nodes[-1]])
    all_outputs = tf.compat.v1.nn.xw_plus_b(all_lstm_outputs, w, b)
    split_outputs = tf.split(all_outputs, num_unrollings, axis=0)

    loss = 0.0
    for ui in range(num_unrollings):
        loss += tf.reduce_mean(tf.square(train_outputs[ui] - split_outputs[ui])) 

    global_step = tf.compat.v1.Variable(0, trainable=False)
    learning_rate = tf.compat.v1.train.exponential_decay(
        starter_learning_rate, global_step, 5000, 0.96, staircase=True
    )
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    sample_inputs = tf.compat.v1.placeholder(tf.float32, shape=[1, D])
    sample_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([
        tf.keras.layers.LSTMCell(units=num_nodes[li]) 
        for li in range(n_layers)
    ])
    sample_state = sample_cell.zero_state(1, tf.float32) 

    sample_output, new_sample_state = tf.compat.v1.nn.dynamic_rnn(
        sample_cell, tf.expand_dims(sample_inputs, 0), initial_state=sample_state, time_major=True, dtype=tf.float32
    )
    sample_prediction = tf.compat.v1.nn.xw_plus_b(tf.reshape(sample_output, [1, num_nodes[-1]]), w, b)
    
    sess = tf.compat.v1.InteractiveSession()
    sess.run(tf.compat.v1.global_variables_initializer())

    print("\nStarting training...")
    for epoch in range(epochs):
        epoch_loss = 0
        num_iterations = (len(train_data) - num_unrollings - batch_size) // batch_size
        
        for iteration in range(num_iterations):
            batch_data, batch_labels = dg.next_batch()

            feed_dict = {}
            for ui in range(num_unrollings):
                feed_dict[train_inputs[ui]] = batch_data[:, ui, :].reshape(batch_size, D)
                feed_dict[train_outputs[ui]] = batch_labels[:, 0].reshape(batch_size, D)
            
            _, l = sess.run([optimizer, loss], feed_dict=feed_dict)
            epoch_loss += l
        
        avg_loss = epoch_loss / num_iterations
        print(f"Epoch {epoch+1}/{epochs}, Average Training Loss: {avg_loss:.6f}")


    print("\nStarting prediction...")
    predicted_prices = []
    test_data_size = len(test_data)
    current_inputs_window = all_mid_data[split_index - num_unrollings : split_index].tolist() 

    state_value = sess.run(sample_cell.zero_state(1, tf.float32))

    current_input = np.array(current_inputs_window[-1]).reshape(1, D)
    
    feed_dict = {sample_inputs: current_input, sample_state: state_value}
    pred, state_value = sess.run([sample_prediction, new_sample_state], feed_dict=feed_dict)
    predicted_prices.append(pred[0, 0])
    
    for i in range(1, test_data_size):
        current_input = np.array(all_mid_data[split_index + i - 1]).reshape(1, D)
        feed_dict = {sample_inputs: current_input, sample_state: state_value}
        pred, state_value = sess.run([sample_prediction, new_sample_state], feed_dict=feed_dict)
        predicted_prices.append(pred[0, 0])


    sess.close()
    
    scaler_for_test = MinMaxScaler()
    scaler_for_test.fit(mid_prices[max(0, split_index-smoothing_window_size):split_index].reshape(-1, 1))

    predicted_prices_np = np.array(predicted_prices).reshape(-1, 1)
    predicted_unscaled = scaler_for_test.inverse_transform(predicted_prices_np)

    test_data_unscaled = mid_prices[split_index:].reshape(-1, 1)

    plt.figure(figsize = (18,9))
    plot_range = range(split_index, df.shape[0])
    
    if len(plot_range) > 0:
        plt.plot(plot_range, test_data_unscaled, color='b', label='True Prices')
        plt.plot(plot_range, predicted_unscaled, color='r', label='Predicted Prices')

        tick_indices = np.linspace(split_index, df.shape[0] - 1, min(10, len(plot_range)), dtype=int)
        plt.xticks(tick_indices, df['Date'].loc[tick_indices], rotation=45)

        plt.title(f'{user_ticker_symbol} (NSE) Stock Price Prediction with LSTM')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Mid Price (INR)', fontsize=18)
        plt.legend(fontsize=18)
        plt.grid(True)
        

    final_test_mse = np.mean(np.square(test_data_unscaled - predicted_unscaled))
    print(f"\nFinal Historical Test MSE (Approximation for unscaled prices): {final_test_mse:.6f}")
    
    if predicted_unscaled.size > 0:
        next_day_prediction = predicted_unscaled[0, 0]
        last_data_date = dt.datetime.strptime(df['Date'].iloc[-1], '%Y-%m-%d').date()
        
        next_date = last_data_date + dt.timedelta(days=1)
        
        print(f"\n--- Real-Time Price Prediction (Next Trading Day) ---")
        print(f"Stock: {user_ticker_symbol} (NSE)")
        print(f"Prediction Date: {next_date} (Approximate Next Trading Day)")
        print(f"Predicted Mid Price: **₹{next_day_prediction:.2f}**")
    else:
        print("\nCould not generate a prediction.")

valid_nse_examples = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY']
print("---")
print(f"The code is ready to use the LSTM model to predict the next price for an NSE stock.")
print(f"Examples of valid NSE tickers: {', '.join(valid_nse_examples)}")
print("---")

nse_ticker = input("Enter a valid NSE Stock Ticker (e.g., RELIANCE): ").strip().upper()

if not nse_ticker:
    print("No ticker entered. Exiting.")
else:
    try:
        run_lstm_prediction(nse_ticker)
    except Exception as e:
        print(f"\nAn unexpected error occurred during execution: {e}")