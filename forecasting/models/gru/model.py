from __future__ import annotations

from typing import Optional

try:
    import tensorflow as tf

    Sequential = tf.keras.Sequential
    GRU = tf.keras.layers.GRU
    Dense = tf.keras.layers.Dense
    Dropout = tf.keras.layers.Dropout
except Exception:  # pragma: no cover
    from keras.models import Sequential
    from keras.layers import GRU, Dense, Dropout


def build_gru_regression_model(
    lookback: int,
    units: int = 50,
    layers: int = 3,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
) -> Sequential:
    if lookback <= 0:
        raise ValueError('lookback must be > 0')
    if layers <= 0:
        raise ValueError('layers must be > 0')

    model = Sequential()
    for i in range(int(layers)):
        return_sequences = i < int(layers) - 1
        if i == 0:
            model.add(
                GRU(
                    units=int(units),
                    return_sequences=return_sequences,
                    input_shape=(int(lookback), 1),
                )
            )
        else:
            model.add(GRU(units=int(units), return_sequences=return_sequences))
        if dropout and float(dropout) > 0:
            model.add(Dropout(float(dropout)))

    model.add(Dense(1, activation='linear'))

    try:
        opt = tf.keras.optimizers.Adam(learning_rate=float(learning_rate))
    except Exception:  # pragma: no cover
        opt = 'adam'

    model.compile(optimizer=opt, loss='mse')
    return model
