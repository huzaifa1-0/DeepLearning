import numpy as np
import pandas as pd
import yfinance as yf
from typing import Tuple, List, Dict, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML & DL Imports
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics, callbacks
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Layer
from tensorflow.keras.models import Model
import tensorflow_probability as tfp

# MLOps & Experiment Tracking
import mlflow
import mlflow.tensorflow
from mlflow.models.signature import infer_signature
import wandb
from tensorboard.plugins.hparams import api as hp

# Data Science Stack
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import scipy.stats as stats
from scipy.signal import savgol_filter

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# System & Logging
import logging
from pathlib import Path
import json
import pickle
import joblib
from abc import ABC, abstractmethod

# ==================== CONFIGURATION MANAGEMENT ====================
@dataclass
class ModelConfig:
    """Configuration for model architecture and training"""
    # Data parameters
    ticker: str = "AAPL"
    start_date: str = "2015-01-01"
    end_date: str = "2023-12-31"
    sequence_length: int = 60
    forecast_horizon: int = 10
    test_size: float = 0.15
    validation_size: float = 0.15
    
    # Feature engineering
    technical_indicators: List[str] = field(default_factory=lambda: [
        'RSI', 'MACD', 'BB_upper', 'BB_lower', 'ATR', 
        'OBV', 'Stochastic', 'ADX', 'CCI'
    ])
    feature_scaler: str = 'robust'  # 'robust', 'standard', 'minmax'
    
    # Model architecture
    rnn_type: str = 'bilstm'  # 'lstm', 'gru', 'bilstm', 'bigru'
    hidden_units: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout_rate: float = 0.3
    recurrent_dropout: float = 0.2
    use_attention: bool = True
    use_residual: bool = True
    bidirectional: bool = True
    
    # Training parameters
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 0.001
    learning_rate_schedule: str = 'cosine_decay'
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    
    # Loss & metrics
    loss_function: str = 'huber'  # 'mse', 'mae', 'huber', 'quantile'
    metrics: List[str] = field(default_factory=lambda: ['mae', 'rmse', 'mape'])
    
    # Advanced features
    use_mcdropout: bool = True  # Monte Carlo Dropout for uncertainty
    use_teacher_forcing: bool = False
    ensemble_size: int = 5
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    
    # MLOps
    experiment_name: str = "financial_forecasting"
    tracking_uri: str = "mlruns"
    use_wandb: bool = False
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 8
    checkpoint_monitor: str = 'val_loss'
    
    def to_dict(self):
        return self.__dict__

# ==================== ADVANCED DATA PIPELINE ====================
class TimeSeriesDataset(tf.data.Dataset):
    """Custom TensorFlow Dataset for time series with caching and prefetching"""
    
    def __new__(cls, 
                features: np.ndarray, 
                targets: np.ndarray, 
                weights: Optional[np.ndarray] = None,
                shuffle: bool = True,
                buffer_size: int = 10000,
                batch_size: int = 32,
                repeat: bool = False):
        
        dataset = tf.data.Dataset.from_tensor_slices((features, targets))
        
        if weights is not None:
            dataset = tf.data.Dataset.from_tensor_slices((features, targets, weights))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
        
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        if repeat:
            dataset = dataset.repeat()
        
        return dataset

class FeatureEngineering:
    """Advanced feature engineering for financial time series"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(prices: pd.Series, 
                      fast: int = 12, 
                      slow: int = 26, 
                      signal: int = 9) -> pd.DataFrame:
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return pd.DataFrame({'MACD': macd, 'Signal': signal_line, 'Histogram': histogram})
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, window: int = 20, num_std: int = 2):
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band, rolling_mean
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr
    
    @staticmethod
    def create_all_features(data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set"""
        df = data.copy()
        
        # Price-based features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Technical indicators
        df['RSI'] = FeatureEngineering.calculate_rsi(df['Close'])
        
        macd_data = FeatureEngineering.calculate_macd(df['Close'])
        df['MACD'] = macd_data['MACD']
        df['MACD_signal'] = macd_data['Signal']
        df['MACD_hist'] = macd_data['Histogram']
        
        df['BB_upper'], df['BB_lower'], df['BB_middle'] = FeatureEngineering.bollinger_bands(df['Close'])
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        
        df['ATR'] = FeatureEngineering.atr(df['High'], df['Low'], df['Close'])
        
        # Volume features
        df['volume_ma'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma']
        
        # Price patterns
        df['high_low_pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        df['close_open_pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
        
        # Lag features
        for lag in [1, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
        
        # Rolling statistics
        windows = [5, 10, 20, 50]
        for window in windows:
            df[f'mean_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'std_{window}'] = df['Close'].rolling(window=window).std()
            df[f'min_{window}'] = df['Close'].rolling(window=window).min()
            df[f'max_{window}'] = df['Close'].rolling(window=window).max()
        
        # Momentum indicators
        df['momentum'] = df['Close'] - df['Close'].shift(5)
        df['roc'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10) * 100
        
        # Seasonality features (day of week, month)
        if 'Date' in df.columns or df.index.name == 'Date':
            dates = pd.to_datetime(df.index if df.index.name == 'Date' else df['Date'])
            df['day_of_week'] = dates.dayofweek
            df['month'] = dates.month
            df['quarter'] = dates.quarter
        
        # Remove NaN values
        df = df.dropna()
        
        return df

# ==================== ADVANCED RNN LAYERS ====================
class TemporalAttention(Layer):
    """Custom attention layer for time series"""
    
    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)
        self.units = units
    
    def call(self, inputs):
        # Self-attention mechanism
        hidden_with_time_axis = tf.expand_dims(inputs, 1)
        score = self.V(tf.nn.tanh(
            self.W1(inputs) + self.W2(hidden_with_time_axis)
        ))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

class ResidualLSTM(Layer):
    """LSTM with residual connections"""
    
    def __init__(self, units: int, return_sequences: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.lstm = LSTM(units, return_sequences=return_sequences, **kwargs)
        self.dense = Dense(units) if not return_sequences else None
        self.units = units
        self.return_sequences = return_sequences
    
    def call(self, inputs):
        lstm_out = self.lstm(inputs)
        if self.return_sequences and inputs.shape[-1] != self.units:
            inputs = self.dense(inputs)
        if self.return_sequences:
            return lstm_out + inputs
        return lstm_out
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "return_sequences": self.return_sequences
        })
        return config

# ==================== ADVANCED MODEL ARCHITECTURE ====================
class UncertaintyModel(Model):
    """Model with built-in uncertainty estimation"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.dropout_rate = config.dropout_rate
        self.mc_samples = 50
        
        # Feature normalization
        self.scaler = LayerNormalization()
        
        # RNN layers
        self.rnn_layers = []
        for i, units in enumerate(config.hidden_units):
            return_sequences = i < len(config.hidden_units) - 1
            
            if config.rnn_type == 'lstm':
                layer = LSTM(units, 
                           return_sequences=return_sequences,
                           dropout=config.dropout_rate,
                           recurrent_dropout=config.recurrent_dropout)
            elif config.rnn_type == 'gru':
                layer = GRU(units,
                          return_sequences=return_sequences,
                          dropout=config.dropout_rate,
                          recurrent_dropout=config.recurrent_dropout)
            elif config.rnn_type == 'bilstm':
                layer = Bidirectional(
                    LSTM(units, 
                        return_sequences=return_sequences,
                        dropout=config.dropout_rate,
                        recurrent_dropout=config.recurrent_dropout)
                )
            else:  # bigru
                layer = Bidirectional(
                    GRU(units,
                       return_sequences=return_sequences,
                       dropout=config.dropout_rate,
                       recurrent_dropout=config.recurrent_dropout)
                )
            
            self.rnn_layers.append(layer)
            self.rnn_layers.append(Dropout(config.dropout_rate))
        
        # Attention mechanism
        if config.use_attention:
            self.attention = TemporalAttention(units=config.hidden_units[-1])
        
        # Output layers for quantile regression
        self.quantile_outputs = []
        for _ in config.quantiles:
            self.quantile_outputs.append(Dense(1))
        
        # Monte Carlo Dropout for uncertainty
        if config.use_mcdropout:
            self.mc_dropout = Dropout(config.dropout_rate)
    
    def call(self, inputs, training=False, mc_dropout=False):
        x = self.scaler(inputs)
        
        # RNN layers
        for layer in self.rnn_layers:
            x = layer(x, training=training)
        
        # Attention
        if self.config.use_attention:
            x, attention_weights = self.attention(x)
        else:
            x = x[:, -1, :] if len(x.shape) == 3 else x
        
        # Monte Carlo Dropout inference
        if mc_dropout and training is False:
            predictions = []
            for _ in range(self.mc_samples):
                x_mc = self.mc_dropout(x, training=True)
                preds = [output(x_mc) for output in self.quantile_outputs]
                predictions.append(preds)
            return tf.stack(predictions, axis=0)
        
        # Standard forward pass
        outputs = [output(x) for output in self.quantile_outputs]
        
        if len(outputs) == 1:
            return outputs[0]
        return outputs
    
    def predict_with_uncertainty(self, inputs, n_samples=100):
        """Monte Carlo dropout for uncertainty estimation"""
        predictions = []
        for _ in range(n_samples):
            pred = self(inputs, training=True)  # Training=True for dropout
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        mean_prediction = np.mean(predictions, axis=0)
        std_prediction = np.std(predictions, axis=0)
        
        return {
            'mean': mean_prediction,
            'std': std_prediction,
            'samples': predictions,
            'confidence_interval': {
                'lower': mean_prediction - 1.96 * std_prediction,
                'upper': mean_prediction + 1.96 * std_prediction
            }
        }

# ==================== ADVANCED TRAINING PIPELINE ====================
class AdvancedTrainer:
    """Professional training pipeline with MLOps integration"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.history = None
        self.setup_logging()
        self.setup_mlflow()
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_mlflow(self):
        """Setup MLFlow for experiment tracking"""
        mlflow.set_tracking_uri(self.config.tracking_uri)
        mlflow.set_experiment(self.config.experiment_name)
        
        if self.config.use_wandb:
            wandb.init(project=self.config.experiment_name)
            wandb.config.update(self.config.to_dict())
    
    def create_callbacks(self) -> List:
        """Create comprehensive callback list"""
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor=self.config.checkpoint_monitor,
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config.reduce_lr_patience,
                min_lr=1e-6,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath='models/best_model.h5',
                monitor=self.config.checkpoint_monitor,
                save_best_only=True,
                verbose=1
            ),
            callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        ]
        
        if self.config.learning_rate_schedule == 'cosine_decay':
            lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=self.config.learning_rate,
                decay_steps=self.config.epochs * 1000
            )
            callbacks_list.append(
                callbacks.LearningRateScheduler(lr_schedule)
            )
        
        return callbacks_list
    
    def build_model(self) -> Model:
        """Build the advanced RNN model"""
        inputs = layers.Input(shape=(self.config.sequence_length, 
                                   len(self.config.technical_indicators) + 4))
        
        self.model = UncertaintyModel(self.config)
        
        # Compile with advanced optimizer
        if self.config.loss_function == 'huber':
            loss = losses.Huber(delta=1.0)
        elif self.config.loss_function == 'quantile':
            def quantile_loss(q):
                def loss(y_true, y_pred):
                    e = y_true - y_pred
                    return tf.keras.backend.mean(tf.maximum(q * e, (q - 1) * e))
                return loss
            
            loss = {f'output_{i}': quantile_loss(q) 
                   for i, q in enumerate(self.config.quantiles)}
        else:
            loss = self.config.loss_function
        
        # AdamW optimizer with weight decay
        optimizer = tfa.optimizers.AdamW(
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            clipvalue=self.config.gradient_clip
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[metrics.MeanAbsoluteError(), 
                    metrics.RootMeanSquaredError()]
        )
        
        return self.model
    
    def train(self, 
              train_data: Tuple,
              val_data: Tuple,
              test_data: Optional[Tuple] = None):
        """Advanced training procedure"""
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(self.config.to_dict())
            
            # Build model
            model = self.build_model()
            self.logger.info(f"Model built with {model.count_params()} parameters")
            
            # Log model architecture
            model_summary = []
            model.summary(print_fn=lambda x: model_summary.append(x))
            mlflow.log_text("\n".join(model_summary), "model_summary.txt")
            
            # Train model
            history = model.fit(
                train_data[0], train_data[1],
                validation_data=val_data,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                callbacks=self.create_callbacks(),
                verbose=1
            )
            
            self.history = history
            
            # Log metrics
            for metric_name, metric_values in history.history.items():
                for epoch, value in enumerate(metric_values):
                    mlflow.log_metric(metric_name, value, step=epoch)
            
            # Evaluate on test set
            if test_data:
                test_results = model.evaluate(
                    test_data[0], test_data[1],
                    verbose=0
                )
                self.logger.info(f"Test results: {test_results}")
                
                # Log test metrics
                for metric_name, value in zip(model.metrics_names, test_results):
                    mlflow.log_metric(f"test_{metric_name}", value)
            
            # Log model
            mlflow.tensorflow.log_model(
                model,
                "model",
                signature=infer_signature(train_data[0][:10], model.predict(train_data[0][:10]))
            )
            
            if self.config.use_wandb:
                wandb.log({"test_loss": test_results[0]})
                wandb.finish()
        
        return model, history

# ==================== ENSEMBLE MODELS ====================
class ModelEnsemble:
    """Ensemble of multiple models for robust predictions"""
    
    def __init__(self, config: ModelConfig, n_models: int = 5):
        self.config = config
        self.n_models = n_models
        self.models = []
        self.weights = None
    
    def create_ensemble(self):
        """Create ensemble of diverse models"""
        for i in range(self.n_models):
            # Create slightly different configurations for diversity
            model_config = ModelConfig(
                **{**self.config.to_dict(), 
                   'dropout_rate': self.config.dropout_rate * (0.8 + 0.4 * i/self.n_models),
                   'hidden_units': [int(u * (0.9 + 0.2 * np.random.rand())) 
                                   for u in self.config.hidden_units]}
            )
            
            trainer = AdvancedTrainer(model_config)
            model = trainer.build_model()
            self.models.append(model)
        
        # Learn ensemble weights (optional)
        self.weights = np.ones(self.n_models) / self.n_models
    
    def predict(self, X: np.ndarray, return_std: bool = True) -> Dict:
        """Make ensemble predictions"""
        predictions = []
        
        for model in self.models:
            pred = model.predict(X, verbose=0)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Weighted average
        weighted_preds = np.average(predictions, axis=0, weights=self.weights)
        
        if return_std:
            std_preds = np.std(predictions, axis=0)
            
            return {
                'mean': weighted_preds,
                'std': std_preds,
                'predictions': predictions,
                'ensemble_variance': np.var(predictions, axis=0)
            }
        
        return weighted_preds

# ==================== ADVANCED EVALUATION ====================
class ModelEvaluator:
    """Comprehensive model evaluation with statistical tests"""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         y_std: Optional[np.ndarray] = None) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        
        metrics_dict = {
            'MAE': np.mean(np.abs(y_true - y_pred)),
            'RMSE': np.sqrt(np.mean((y_true - y_pred) ** 2)),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'R2': 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2),
            'MDA': np.mean(np.sign(y_true[1:] - y_true[:-1]) == 
                          np.sign(y_pred[1:] - y_pred[:-1])),
            'Theil_U': np.sqrt(np.mean((y_true - y_pred) ** 2)) / 
                      (np.sqrt(np.mean(y_true ** 2)) + np.sqrt(np.mean(y_pred ** 2)))
        }
        
        # Uncertainty metrics if available
        if y_std is not None:
            z_scores = (y_true - y_pred) / y_std
            metrics_dict.update({
                'Coverage_95': np.mean(
                    (y_true >= y_pred - 1.96 * y_std) & 
                    (y_true <= y_pred + 1.96 * y_std)
                ),
                'NLL': -np.mean(stats.norm.logpdf(y_true, loc=y_pred, scale=y_std)),
                'Calibration_Error': np.mean(np.abs(
                    np.percentile(z_scores, np.linspace(0, 100, 101)) - 
                    np.linspace(0, 100, 101)
                ))
            })
        
        return metrics_dict
    
    @staticmethod
    def statistical_tests(y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         residuals: np.ndarray) -> Dict:
        """Perform statistical tests on predictions"""
        
        # Jarque-Bera test for normality of residuals
        jb_stat, jb_p = stats.jarque_bera(residuals)
        
        # Ljung-Box test for autocorrelation
        lb_stat, lb_p = stats.acf(residuals, nlags=20, fft=True)
        
        # Diebold-Mariano test for predictive accuracy
        # (Simplified version)
        loss_diff = np.abs(y_true - y_pred)
        dm_stat = np.mean(loss_diff) / (np.std(loss_diff) / np.sqrt(len(loss_diff)))
        
        return {
            'jarque_bera': {'statistic': jb_stat, 'p_value': jb_p},
            'ljung_box': {'statistics': lb_stat, 'p_values': lb_p},
            'diebold_mariano': dm_stat,
            'residual_mean': np.mean(residuals),
            'residual_std': np.std(residuals),
            'shapiro_p': stats.shapiro(residuals)[1]
        }
    
    @staticmethod
    def plot_predictions(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        y_std: Optional[np.ndarray] = None,
                        title: str = "Predictions vs Actual"):
        """Create interactive plotly visualization"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Predictions vs Actual', 
                          'Residuals Distribution',
                          'Cumulative Returns',
                          'Prediction Errors'),
            specs=[[{'secondary_y': True}, {}],
                   [{}, {}]]
        )
        
        # Time series plot
        time_idx = np.arange(len(y_true))
        fig.add_trace(
            go.Scatter(x=time_idx, y=y_true.flatten(),
                      name='Actual', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_idx, y=y_pred.flatten(),
                      name='Predicted', line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        if y_std is not None:
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([time_idx, time_idx[::-1]]),
                    y=np.concatenate([
                        (y_pred + 1.96 * y_std).flatten(),
                        (y_pred - 1.96 * y_std).flatten()[::-1]
                    ]),
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='95% Confidence'
                ),
                row=1, col=1
            )
        
        # Residuals histogram
        residuals = y_true - y_pred
        fig.add_trace(
            go.Histogram(x=residuals.flatten(),
                        nbinsx=50,
                        name='Residuals'),
            row=1, col=2
        )
        
        # Q-Q plot
        qq_x, qq_y = stats.probplot(residuals.flatten(), dist="norm")
        fig.add_trace(
            go.Scatter(x=qq_x[0], y=qq_y[0],
                      mode='markers',
                      name='Q-Q Plot'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=qq_x[0], y=qq_x[0] * qq_x[1][0] + qq_x[1][1],
                      mode='lines',
                      name='Normal Line'),
            row=2, col=1
        )
        
        # Error over time
        fig.add_trace(
            go.Scatter(x=time_idx, y=np.abs(residuals).flatten(),
                      name='Absolute Error',
                      fill='tozeroy'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text=title)
        fig.show()
        
        return fig

# ==================== MAIN PIPELINE ====================
class FinancialForecastingPipeline:
    """End-to-end pipeline for financial time series forecasting"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.data = None
        self.features = None
        self.scaler = None
        self.trainer = AdvancedTrainer(config)
        self.evaluator = ModelEvaluator()
        self.ensemble = None
        
    def fetch_data(self) -> pd.DataFrame:
        """Fetch and preprocess financial data"""
        self.logger.info(f"Fetching data for {self.config.ticker}")
        
        # Fetch from Yahoo Finance
        ticker = yf.Ticker(self.config.ticker)
        self.data = ticker.history(
            start=self.config.start_date,
            end=self.config.end_date,
            interval='1d'
        )
        
        # Reset index and add date features
        self.data = self.data.reset_index()
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        self.logger.info(f"Fetched {len(self.data)} data points")
        return self.data
    
    def engineer_features(self) -> pd.DataFrame:
        """Create advanced features"""
        self.logger.info("Engineering features...")
        
        feature_engineer = FeatureEngineering()
        self.features = feature_engineer.create_all_features(self.data)
        
        # Select final features
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        feature_cols.extend(self.config.technical_indicators)
        
        # Handle missing values
        self.features = self.features[feature_cols].fillna(method='ffill').fillna(method='bfill')
        
        return self.features
    
    def prepare_sequences(self, 
                         data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for RNN training"""
        
        # Scale features
        if self.config.feature_scaler == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - self.config.sequence_length - self.config.forecast_horizon):
            X.append(scaled_data[i:i + self.config.sequence_length])
            y.append(scaled_data[i + self.config.sequence_length: 
                               i + self.config.sequence_length + self.config.forecast_horizon, 3])  # Close price
        
        X = np.array(X)
        y = np.array(y)
        
        # Create multi-output if forecast horizon > 1
        if self.config.forecast_horizon > 1:
            y = y.reshape(-1, self.config.forecast_horizon, 1)
        
        self.logger.info(f"Created sequences: X shape {X.shape}, y shape {y.shape}")
        return X, y
    
    def train_validate_split(self, 
                            X: np.ndarray, 
                            y: np.ndarray) -> Tuple:
        """Time series aware train/validation/test split"""
        
        # Time series split
        n_samples = len(X)
        train_end = int(n_samples * (1 - self.config.test_size - self.config.validation_size))
        val_end = int(n_samples * (1 - self.config.test_size))
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        self.logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def run(self):
        """Execute the complete pipeline"""
        
        # Step 1: Data Acquisition
        self.fetch_data()
        
        # Step 2: Feature Engineering
        self.engineer_features()
        
        # Step 3: Prepare Sequences
        X, y = self.prepare_sequences(self.features)
        
        # Step 4: Split Data
        train_data, val_data, test_data = self.train_validate_split(X, y)
        
        # Step 5: Train Model
        self.logger.info("Starting model training...")
        model, history = self.trainer.train(train_data, val_data, test_data)
        
        # Step 6: Evaluate Model
        self.logger.info("Evaluating model...")
        y_pred = model.predict(test_data[0])
        
        # Calculate metrics
        metrics = self.evaluator.calculate_metrics(
            test_data[1].flatten(), 
            y_pred.flatten()
        )
        
        # Statistical tests
        residuals = test_data[1].flatten() - y_pred.flatten()
        stats_tests = self.evaluator.statistical_tests(
            test_data[1].flatten(),
            y_pred.flatten(),
            residuals
        )
        
        # Step 7: Visualization
        fig = self.evaluator.plot_predictions(
            test_data[1][:100],  # First 100 samples
            y_pred[:100],
            title=f"{self.config.ticker} - Predictions vs Actual"
        )
        
        # Step 8: Create Ensemble (optional)
        if self.config.ensemble_size > 1:
            self.logger.info("Creating model ensemble...")
            self.ensemble = ModelEnsemble(self.config, self.config.ensemble_size)
            self.ensemble.create_ensemble()
            
            # Ensemble predictions
            ensemble_preds = self.ensemble.predict(test_data[0], return_std=True)
            
            # Ensemble metrics
            ensemble_metrics = self.evaluator.calculate_metrics(
                test_data[1].flatten(),
                ensemble_preds['mean'].flatten(),
                ensemble_preds['std'].flatten()
            )
        
        # Step 9: Generate Report
        self.generate_report(metrics, stats_tests)
        
        return model, metrics
    
    def generate_report(self, metrics: Dict, stats_tests: Dict):
        """Generate comprehensive performance report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config.to_dict(),
            'performance_metrics': metrics,
            'statistical_tests': stats_tests,
            'model_info': {
                'parameters': self.trainer.model.count_params(),
                'architecture': str(self.trainer.model.summary())
            }
        }
        
        # Save report
        with open('performance_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Log report
        self.logger.info("Performance Report:")
        self.logger.info(f"MAE: {metrics.get('MAE', 0):.4f}")
        self.logger.info(f"RMSE: {metrics.get('RMSE', 0):.4f}")
        self.logger.info(f"R²: {metrics.get('R2', 0):.4f}")
        self.logger.info(f"MDA: {metrics.get('MDA', 0):.4f}")
        
        return report

# ==================== DEPLOYMENT READY ====================
class ModelDeployer:
    """Production model deployment utilities"""
    
    @staticmethod
    def save_pipeline(pipeline: FinancialForecastingPipeline, 
                     path: str = "models/production"):
        """Save complete pipeline for deployment"""
        
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save model
        pipeline.trainer.model.save(f"{path}/model.h5")
        
        # Save scaler
        joblib.dump(pipeline.scaler, f"{path}/scaler.pkl")
        
        # Save config
        with open(f"{path}/config.json", 'w') as f:
            json.dump(pipeline.config.to_dict(), f, indent=2)
        
        # Create serving signature
        serving_input = tf.keras.layers.Input(
            shape=(pipeline.config.sequence_length, 
                  len(pipeline.config.technical_indicators) + 4),
            dtype=tf.float32,
            name='serving_input'
        )
        
        serving_output = pipeline.trainer.model(serving_input)
        serving_model = tf.keras.Model(inputs=serving_input, outputs=serving_output)
        
        # Save in SavedModel format
        tf.saved_model.save(serving_model, f"{path}/saved_model")
    
    @staticmethod
    def load_pipeline(path: str = "models/production"):
        """Load pipeline for inference"""
        
        # Load model
        model = tf.keras.models.load_model(f"{path}/model.h5")
        
        # Load scaler
        scaler = joblib.load(f"{path}/scaler.pkl")
        
        # Load config
        with open(f"{path}/config.json", 'r') as f:
            config_dict = json.load(f)
        
        return model, scaler, config_dict

# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    # Initialize configuration
    config = ModelConfig(
        ticker="AAPL",
        sequence_length=60,
        forecast_horizon=5,
        hidden_units=[128, 64, 32],
        rnn_type='bilstm',
        use_attention=True,
        use_mcdropout=True,
        ensemble_size=3,
        epochs=50,
        batch_size=32
    )
    
    # Create and run pipeline
    pipeline = FinancialForecastingPipeline(config)
    model, metrics = pipeline.run()
    
    # Deploy model
    ModelDeployer.save_pipeline(pipeline)
    
    print("\n" + "="*60)
    print("PRODUCTION-READY RNN PIPELINE COMPLETED")
    print("="*60)
    print(f"Final MAE: {metrics.get('MAE', 0):.4f}")
    print(f"Final RMSE: {metrics.get('RMSE', 0):.4f}")
    print(f"Final R²: {metrics.get('R2', 0):.4f}")
    print("Model saved in models/production/")
    print("="*60)
