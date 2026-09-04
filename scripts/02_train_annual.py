"""
02_train_annual.py
Train CNN-LSTM model year-by-year with expanding window + ensemble.

Input: data/interim/step4_full_data.parquet
Output: data/predictions/ensemble_preds_YYYY.npz
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import Config

print("=" * 60)
print("STEP 2: ANNUAL MODEL TRAINING")
print("=" * 60)
print(f"   Target years: {Config.TEST_START_YEAR} - {Config.TEST_END_YEAR}")
print(f"   Ensemble size: {Config.N_ENSEMBLE}")
print(f"   Batch size: {Config.BATCH_SIZE}")
print(f"   Learning rate: {Config.LEARNING_RATE}")


def load_data():
    """Load the preprocessed dataset."""
    data_path = os.path.join(Config.DATA_INTERIM_PATH, "step4_full_data.parquet")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            "Please run 01_preprocess.py first."
        )
    df = pd.read_parquet(data_path)
    df['Year'] = df['Trdmnt'].dt.year
    print(f"\nLoaded {len(df):,} rows")
    print(f"Date range: {df['Trdmnt'].min()} to {df['Trdmnt'].max()}")
    return df


def build_model():
    """
    Build the CNN-LSTM model architecture.
    """
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=(12, 1)),
        MaxPooling1D(2),
        Conv1D(32, 3, activation='relu'),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=Config.LEARNING_RATE),
        loss='mse'
    )
    return model


def train_single_year(test_year, df_full):
    """
    Train ensemble models for a single year.
    """
    print(f"\n{'='*40}")
    print(f"Predicting year: {test_year}")
    print(f"{'='*40}")
    
    # Split data (expanding window)
    train_df = df_full[df_full['Year'] < test_year]
    test_df = df_full[df_full['Year'] == test_year]
    
    if len(train_df) == 0 or len(test_df) == 0:
        print(f"   Skipping {test_year}: insufficient data")
        return None
    
    X_train = train_df[Config.FEATURE_COLS].values.astype(np.float32)
    y_train = train_df['rNorm'].values.astype(np.float32)
    X_test = test_df[Config.FEATURE_COLS].values.astype(np.float32)
    y_test = test_df['rNorm'].values.astype(np.float32)
    
    print(f"   Training set: {len(X_train):,} samples")
    print(f"   Test set: {len(X_test):,} samples")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Reshape for CNN input: (samples, timesteps=12, features=1)
    X_train_3d = X_train_scaled.reshape(-1, 12, 1)
    X_test_3d = X_test_scaled.reshape(-1, 12, 1)
    
    # Ensemble predictions
    ensemble_preds = np.zeros((X_test_3d.shape[0], Config.N_ENSEMBLE))
    
    for i in range(Config.N_ENSEMBLE):
        print(f"   Ensemble {i+1}/{Config.N_ENSEMBLE}...", end='', flush=True)
        
        # Set seeds for reproducibility
        tf.random.set_seed(i * 123 + test_year)
        np.random.seed(i * 456 + test_year)
        
        model = build_model()
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        model.fit(
            X_train_3d, y_train,
            batch_size=Config.BATCH_SIZE,
            epochs=Config.EPOCHS_PER_ROUND,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )
        
        pred = model.predict(X_test_3d, verbose=0).ravel()
        ensemble_preds[:, i] = pred
        
        # Clean up
        del model
        clear_session()
        gc.collect()
        
        # Reset GPU memory if available
        if tf.config.list_physical_devices('GPU'):
            try:
                tf.config.experimental.reset_memory_stats('GPU:0')
            except Exception:
                pass
        
        print(" done")
    
    # Average predictions
    y_pred_avg = ensemble_preds.mean(axis=1)
    
    # Save annual predictions
    save_path = os.path.join(
        Config.DATA_PREDICTIONS_PATH,
        f"ensemble_preds_{test_year}.npz"
    )
    np.savez_compressed(
        save_path,
        y_pred=y_pred_avg,
        y_true=y_test,
        months=test_df['Trdmnt'].values,
        stocks=test_df['Stkcd'].values
    )
    print(f"   ✅ Saved: {save_path}")
    
    # Clean up large variables
    del X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled
    del X_train_3d, X_test_3d, ensemble_preds, y_pred_avg, scaler
    del train_df, test_df
    gc.collect()
    clear_session()
    
    return save_path


def main():
    # Load data
    print("\n[2.1] Loading preprocessed data...")
    df_full = load_data()
    
    # Check existing files
    print("\n[2.2] Checking existing prediction files...")
    completed_years = []
    for f in glob.glob(os.path.join(Config.DATA_PREDICTIONS_PATH, "ensemble_preds_*.npz")):
        try:
            year = int(os.path.basename(f).split('_')[-1].split('.')[0])
            completed_years.append(year)
        except Exception:
            pass
    print(f"   Already completed: {completed_years}")
    
    # Train missing years
    years = list(range(Config.TEST_START_YEAR, Config.TEST_END_YEAR + 1))
    print(f"\n[2.3] Training years: {years}")
    
    for year in years:
        if year in completed_years:
            print(f"\n   {year} already exists, skipping.")
            continue
        train_single_year(year, df_full)
    
    # Summary
    all_files = glob.glob(os.path.join(Config.DATA_PREDICTIONS_PATH, "ensemble_preds_*.npz"))
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"   Total prediction files: {len(all_files)}")
    for f in sorted(all_files):
        print(f"     - {os.path.basename(f)}")
    print("   Next: Run 03_merge_predictions.py to merge all years.")


if __name__ == "__main__":
    main()
