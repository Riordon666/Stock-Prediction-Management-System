from __future__ import annotations

from typing import Optional

try:
    import tensorflow as tf
    Sequential = tf.keras.Sequential
    LSTM = tf.keras.layers.LSTM
    Dense = tf.keras.layers.Dense
    Dropout = tf.keras.layers.Dropout
except Exception:
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout

def build_lstm_regression_model(
    lookback: int,
    feature_count: int = 1,
    units: int = 128,
    layers: int = 3,
    dropout: float = 0.5,
    learning_rate: float = 0.001,
) -> Sequential:
    """
    Baseline multi-layer LSTM.
    """
    if lookback <= 0:
        raise ValueError('lookback must be > 0')
    if layers <= 0:
        raise ValueError('layers must be > 0')

    model = Sequential()
    for i in range(int(layers)):
        return_sequences = i < int(layers) - 1
        if i == 0:
            model.add(
                LSTM(
                    units=int(units),
                    return_sequences=return_sequences,
                    input_shape=(int(lookback), int(feature_count)),
                )
            )
        else:
            model.add(LSTM(units=int(units), return_sequences=return_sequences))
        
        if dropout and float(dropout) > 0:
            model.add(Dropout(float(dropout)))

    # Hidden dense layers
    model.add(Dense(16, activation='leaky_relu'))
    model.add(Dense(1, activation='linear'))

    try:
        opt = tf.keras.optimizers.Adam(learning_rate=float(learning_rate))
    except Exception:
        opt = 'adam'

    model.compile(optimizer=opt, loss='mse')
    return model
