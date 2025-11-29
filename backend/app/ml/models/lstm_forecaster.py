import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class LSTMForecaster:
    def __init__(self, sequence_length: int = 30, lstm_units: int = 50, dense_units: int = 25, dropout: float = 0.2, epochs: int = 50, batch_size: int = 32, learning_rate: float = 0.001):
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.is_fitted = False
    
    def _build_model(self, n_features: int):
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=(self.sequence_length, n_features)),
            Dropout(self.dropout),
            LSTM(self.lstm_units, return_sequences=False),
            Dropout(self.dropout),
            Dense(self.dense_units, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse', metrics=['mae'])
        return model
    
    def _create_sequences(self, data: np.ndarray, target: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            if target is not None:
                y.append(target[i + self.sequence_length])
        return np.array(X), np.array(y) if target is not None else None
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: int = 0):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.n_features = X.shape[1]
        
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)
        if len(X_seq) == 0:
            raise ValueError(f"Not enough data. Need at least {self.sequence_length + 1} samples.")
        
        self.model = self._build_model(self.n_features)
        
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        
        self.model.fit(X_seq, y_seq, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.1, verbose=verbose)
        self.is_fitted = True
        self.last_sequence = X_scaled[-self.sequence_length:]
        return self
    
    def predict(self, X: np.ndarray = None, horizon: int = 30) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        if X is not None:
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            X_scaled = self.scaler.transform(X)
            X_seq, _ = self._create_sequences(X_scaled)
            if len(X_seq) > 0:
                pred_scaled = self.model.predict(X_seq, verbose=0)
                return self.target_scaler.inverse_transform(pred_scaled).flatten()
            return np.array([])
        
        predictions = []
        current_sequence = self.last_sequence.copy()
        for _ in range(horizon):
            pred_scaled = self.model.predict(current_sequence.reshape(1, self.sequence_length, self.n_features), verbose=0)
            pred_value = self.target_scaler.inverse_transform(pred_scaled)[0, 0]
            predictions.append(pred_value)
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = pred_scaled[0, 0]
        return np.array(predictions)
    
    def get_params(self) -> Dict[str, Any]:
        return {"sequence_length": self.sequence_length, "lstm_units": self.lstm_units, "dense_units": self.dense_units, "dropout": self.dropout, "epochs": self.epochs, "batch_size": self.batch_size, "learning_rate": self.learning_rate}
