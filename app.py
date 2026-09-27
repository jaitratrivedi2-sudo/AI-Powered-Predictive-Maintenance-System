"""
app.py
------
Main application entry point for AI Predictive Maintenance System.

Usage:
    python app.py           # Launch dashboard
    python app.py --train   # Train models
    python app.py --status  # Show system status
    python app.py --sync-db # Sync CSV to Supabase cloud
    python app.py --test-cloud # Test cloud connection
"""

import sys
import os
import argparse
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# PROJECT ROOT SETUP  (must be first before any local imports)
# ══════════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import config
from utils.data_loader import load_sensor_data, load_training_data
from utils.model_loader import load_model, load_scaler
from utils.predictor import predict_maintenance
from sensors.sensor_manager import get_sensor_mode


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def train_pipeline():
    """Complete model training pipeline."""
    print("="*60)
    print("  MODEL TRAINING PIPELINE")
    print("="*60)

    # Step 1: Load training data
    print("\n📊 Step 1: Loading training data...")
    try:
        X, y = load_training_data()
        print(f"✅ Training data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"   Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return False

    # Step 2: Fit and save scaler
    print("\n🔧 Step 2: Fitting feature scaler...")
    try:
        from preprocessing.scaler import fit_and_save_scaler
        scaler = fit_and_save_scaler(X, config.SCALER_PATH)
        X_scaled = scaler.transform(X)
        print(f"✅ Scaler fitted and saved")
        print(f"✅ Features scaled")
    except Exception as e:
        print(f"⚠️  Warning: Scaler error: {e}")
        print("   Continuing without scaling...")
        X_scaled = X

    # Step 3: Train RandomForest classifier
    print("\n🤖 Step 3: Training RandomForest classifier...")
    try:
        from models.train_model import train_model
        model = train_model(X_scaled, y)
        print(f"✅ Model trained and saved to: {config.MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error training model: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 4: Train IsolationForest anomaly detector
    print("\n🔬 Step 4: Training anomaly detector...")
    try:
        from models.anomaly_detector import train_anomaly_model
        anomaly_model = train_anomaly_model(X_scaled)
        print(f"✅ Anomaly detector trained and saved")
    except Exception as e:
        print(f"⚠️  Anomaly detector skipped: {e}")

    # Step 5: Train RUL regression model
    print("\n📅 Step 5: Training RUL (Remaining Useful Life) model...")
    try:
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score
        import joblib

        # Find a CSV that has rul_days column
        rul_df = None
        for path in [config.HISTORICAL_PATH, config.PROCESSED_PATH, config.FALLBACK_PATH]:
            if os.path.exists(path):
                tmp = pd.read_csv(path)
                if "rul_days" in tmp.columns and len(tmp) >= 10:
                    rul_df = tmp
                    print(f"   Using data from : {os.path.basename(path)}")
                    print(f"   Total rows      : {len(rul_df)}")
                    break

        if rul_df is None:
            print("⚠️  No CSV with rul_days found — skipping RUL model")
        else:
            # Pick all available feature columns
            feature_cols = [c for c in [
                "vibration_mean", "vibration_std", "vibration_max", "vibration_rms",
                "sound_mean", "sound_std", "sound_max", "sound_rms",
                "temperature_c", "pressure_bar", "health_score",
                "vibration_reading", "sound_reading"
            ] if c in rul_df.columns]

            if len(feature_cols) < 2:
                print("⚠️  Not enough feature columns — skipping RUL model")
            else:
                X_rul = rul_df[feature_cols].apply(
                    pd.to_numeric, errors="coerce").fillna(0).values
                y_rul = pd.to_numeric(
                    rul_df["rul_days"], errors="coerce").fillna(90).values

                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_rul, y_rul, test_size=0.2, random_state=42)

                rul_model = RandomForestRegressor(
                    n_estimators=100, max_depth=8,
                    random_state=42, n_jobs=-1)
                rul_model.fit(X_tr, y_tr)

                mae = mean_absolute_error(y_te, rul_model.predict(X_te))
                r2  = r2_score(y_te, rul_model.predict(X_te))

                print(f"   Features used   : {feature_cols}")
                print(f"   Training samples: {len(X_tr)}")
                print(f"   MAE (error)     : +/-{mae:.1f} days")
                print(f"   R2 Score        : {r2:.3f}")

                os.makedirs(os.path.dirname(config.RUL_MODEL_PATH), exist_ok=True)
                joblib.dump(rul_model, config.RUL_MODEL_PATH)
                kb = os.path.getsize(config.RUL_MODEL_PATH) // 1024
                print(f"✅ RUL model saved : {config.RUL_MODEL_PATH} ({kb} KB)")

    except Exception as e:
        print(f"⚠️  RUL model training skipped: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "="*60)
    print("  ✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"\nTrained models saved:")
    print(f"  📦 {config.MODEL_PATH}")
    print(f"  📦 {config.SCALER_PATH}")
    if os.path.exists(config.ANOMALY_MODEL_PATH):
        print(f"  📦 {config.ANOMALY_MODEL_PATH}")
    if os.path.exists(config.RUL_MODEL_PATH) and \
       os.path.getsize(config.RUL_MODEL_PATH) > 0:
        kb = os.path.getsize(config.RUL_MODEL_PATH) // 1024
        print(f"  📦 {config.RUL_MODEL_PATH}  ({kb} KB)")
    print("\nLaunch dashboard:")
    print("  streamlit run dashboard/dashboard_app.py")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM STATUS
# ══════════════════════════════════════════════════════════════════════════════

def show_status():
    """Display system status and configuration."""
    print("="*60)
    print("  AI-Powered Predictive Maintenance System — STATUS")
    print("="*60)

    mode = get_sensor_mode()
    df, actual_mode = load_sensor_data()

    print(f"\n  Sensor Mode   : {mode}")
    print(f"  Data Mode     : {actual_mode}")
    print(f"  Actual Source : {actual_mode}")
    print(f"  IoT Sensor    : {'✅ Connected' if mode == 'LIVE' else '❌ Not connected'}")
    print(f"  Data Rows     : {len(df)}")

    if 'machine_id' in df.columns:
        machines = sorted(df['machine_id'].unique())
        print(f"  Machines      : {machines}")

    print("\n" + "="*60)
    print("  CLOUD STATUS")
    print("="*60)
    db_ok = bool(getattr(config, 'CLOUD_DATABASE_ENABLED', False))
    tg_ok = bool(getattr(config, 'TELEGRAM_ENABLED', False))
    print(f"  Database    : Configured={'✅' if db_ok else '❌'}  Enabled={'✅' if db_ok else '❌'}")
    print(f"  Telegram    : Configured={'✅' if tg_ok else '❌'}  Enabled={'✅' if tg_ok else '❌'}")

    print("\n" + "="*60)
    print("  MODEL STATUS")
    print("="*60)

    model  = load_model()
    scaler = load_scaler()
    print(f"  Classifier : {'✅ Loaded' if model  else '❌ Not trained'}")
    print(f"  Scaler     : {'✅ Loaded' if scaler else '❌ Not fitted'}")

    anomaly_ok = (os.path.exists(config.ANOMALY_MODEL_PATH) and
                  os.path.getsize(config.ANOMALY_MODEL_PATH) > 0)
    print(f"  Anomaly Det: {'✅ Loaded' if anomaly_ok else '❌ Not trained'}")

    rul_ok = (os.path.exists(config.RUL_MODEL_PATH) and
              os.path.getsize(config.RUL_MODEL_PATH) > 0)
    print(f"  RUL Model  : {'✅ Loaded' if rul_ok else '❌ Empty — run: python app.py --train'}")

    if not df.empty:
        print("\n" + "="*60)
        print("  CURRENT PREDICTIONS")
        print("="*60)
        try:
            prediction = predict_maintenance(df, model, scaler)
            print(f"  Fleet Status  : {prediction.get('fleet_status', 'UNKNOWN')}")
            print(f"  Avg Health    : {prediction.get('avg_health', 0):.1f}%")
            print(f"  Avg Risk      : {prediction.get('avg_risk', 0):.1f}%")
            print(f"  Critical Count: {prediction.get('critical_count', 0)}")
            if 'machines' in prediction:
                print("\n  Machine Details:")
                for m_id, m_data in prediction['machines'].items():
                    status = m_data.get('health_status', 'UNKNOWN')
                    health = m_data.get('health_score', 0)
                    risk   = m_data.get('failure_risk', 0)
                    rul    = m_data.get('rul_days', 0)
                    print(f"    {m_id}: {status:8s} | Health: {health:5.1f}% | "
                          f"Risk: {risk:5.1f}% | RUL: {rul:3d}d")
        except Exception as e:
            print(f"  ⚠️  Prediction error: {e}")

    print("\n" + "="*60)


# ══════════════════════════════════════════════════════════════════════════════
# CLOUD SYNC AND TEST
# ══════════════════════════════════════════════════════════════════════════════

def sync_database():
    """Sync local CSV data to Supabase cloud database."""
    print("="*60)
    print("  CLOUD DATABASE SYNC")
    print("="*60)
    try:
        from cloud.database import sync_csv_to_cloud

        # Try historical CSV first, then fallback
        csv_path = None
        for path in [config.HISTORICAL_PATH, config.FALLBACK_PATH]:
            if os.path.exists(path):
                csv_path = path
                break

        if csv_path is None:
            print("❌ No CSV file found to sync")
            print(f"   Checked: {config.HISTORICAL_PATH}")
            return

        print(f"📂 Syncing: {csv_path}")
        sync_csv_to_cloud(csv_path)

    except Exception as e:
        print(f"❌ Sync failed: {e}")
        import traceback
        traceback.print_exc()


def test_cloud():
    """Test cloud connection (Supabase + Telegram)."""
    print("="*60)
    print("  CLOUD CONNECTION TEST")
    print("="*60)
    try:
        from cloud.database  import test_connection
        from cloud.messaging import test_telegram
        test_connection()
        test_telegram()
    except Exception as e:
        print(f"❌ Cloud test failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# LAUNCH DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def launch_dashboard():
    """Launch Streamlit dashboard."""
    import subprocess
    dashboard_path = os.path.join(PROJECT_ROOT, "dashboard", "dashboard_app.py")
    if not os.path.exists(dashboard_path):
        print(f"❌ Dashboard not found: {dashboard_path}")
        return
    print("="*60)
    print("  LAUNCHING DASHBOARD")
    print("="*60)
    print(f"  URL: http://localhost:8501")
    print("  Press Ctrl+C to stop")
    print("="*60)
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            dashboard_path,
            "--server.port", str(getattr(config, 'DASHBOARD_PORT', 8501))
        ])
    except KeyboardInterrupt:
        print("\n✅ Dashboard stopped")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI-Powered Predictive Maintenance System"
    )
    parser.add_argument("--train",      action="store_true", help="Train ML models")
    parser.add_argument("--status",     action="store_true", help="Show system status")
    parser.add_argument("--sync-db",    action="store_true", help="Sync CSV to Supabase")
    parser.add_argument("--test-cloud", action="store_true", help="Test cloud connection")
    parser.add_argument("--mode", choices=["LIVE", "HISTORICAL", "FALLBACK"],
                        help="Force data mode")

    args = parser.parse_args()

    if args.train:
        train_pipeline()
    elif args.status:
        show_status()
    elif args.sync_db:
        sync_database()
    elif args.test_cloud:
        test_cloud()
    else:
        launch_dashboard()


if __name__ == "__main__":
    main()




# """
# app.py
# ------
# Main application entry point for AI Predictive Maintenance System.

# Usage:
#     python app.py              → Launch dashboard
#     python app.py --train      → Train ML models
#     python app.py --status     → Show system status
#     python app.py --sync-db    → Upload historical CSV to Supabase (one-time)
#     python app.py --test-cloud → Test cloud database + messaging
# """

# import sys
# import os
# import argparse
# import numpy as np

# # ══════════════════════════════════════════════════════════════════════════════
# # PROJECT ROOT — must be first before ANY project imports
# # ══════════════════════════════════════════════════════════════════════════════
# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, PROJECT_ROOT)

# import config
# from utils.data_loader   import load_sensor_data, load_training_data, get_current_data_mode
# from utils.model_loader  import load_model, load_scaler
# from utils.predictor     import predict_maintenance
# from sensors.sensor_manager import get_sensor_mode


# # ══════════════════════════════════════════════════════════════════════════════
# # SENSOR DATA COLLECTION  (called when IoT sensor is connected)
# # ══════════════════════════════════════════════════════════════════════════════

# def collect_and_upload_sensor_data():
#     """
#     Read from IoT sensor CSV and upload each reading to Supabase.
#     Only runs when:
#       - SENSOR_AVAILABLE = True  (sensor hardware connected)
#       - CLOUD_DATABASE_ENABLED = True  (Supabase configured)
#     """
#     if not config.SENSOR_AVAILABLE:
#         print("ℹ️  Sensor not connected — skipping upload")
#         return None

#     if not getattr(config, 'CLOUD_DATABASE_ENABLED', False):
#         print("ℹ️  Cloud database disabled — skipping upload")
#         return None

#     try:
#         # Load the live sensor CSV data
#         df, mode = load_sensor_data()

#         if df is None or df.empty:
#             print("⚠️  No sensor data to upload")
#             return None

#         # Upload latest readings to cloud
#         try:
#             from cloud.database import upload_sensor_reading

#             latest = df.iloc[-1]
#             upload_sensor_reading(
#                 machine_id    = str(latest.get("machine_id", "M001")),
#                 vibration_mean= float(latest.get("vibration_mean", 0)),
#                 sound_mean    = float(latest.get("sound_mean", 0)),
#                 temperature_c = float(latest.get("temperature_c", 0)),
#                 pressure_bar  = float(latest.get("pressure_bar", 0)),
#                 vibration_std = float(latest.get("vibration_std", 0)),
#                 sound_std     = float(latest.get("sound_std", 0)),
#             )
#             print(f"✅ Sensor reading uploaded to Supabase")
#         except Exception as e:
#             print(f"⚠️  Upload error: {e}")

#         return df

#     except Exception as e:
#         print(f"⚠️  collect_and_upload_sensor_data error: {e}")
#         return None


# def run_diagnosis_and_upload(df, model, scaler):
#     """
#     Run ML prediction on sensor data and upload results to Supabase.
#     Also sends alerts for CRITICAL/WARNING machines.

#     Args:
#         df     : sensor DataFrame
#         model  : trained ML model
#         scaler : fitted scaler

#     Returns:
#         prediction dict
#     """
#     prediction = predict_maintenance(df, model, scaler)

#     # Only upload to cloud if database is enabled
#     if not getattr(config, 'CLOUD_DATABASE_ENABLED', False):
#         return prediction

#     try:
#         from cloud.database  import upload_prediction
#         from cloud.messaging import send_free_alert

#         for machine_id, machine_data in prediction.get('machines', {}).items():
#             # Upload prediction to Supabase
#             upload_prediction(
#                 machine_id    = machine_id,
#                 health_score  = machine_data.get('health_score', 0),
#                 failure_risk  = machine_data.get('failure_risk', 0),
#                 rul_days      = machine_data.get('rul_days', 90),
#                 health_status = machine_data.get('health_status', 'GOOD'),
#             )

#             # Send alert for critical/warning machines
#             status = machine_data.get('health_status', 'GOOD')
#             if status in ['CRITICAL', 'WARNING']:
#                 auto_critical = getattr(config, 'AUTO_ALERT_ON_CRITICAL', False)
#                 auto_warning  = getattr(config, 'AUTO_ALERT_ON_WARNING', False)

#                 should_alert = (
#                     (status == 'CRITICAL' and auto_critical) or
#                     (status == 'WARNING'  and auto_warning)
#                 )

#                 if should_alert:
#                     send_free_alert(machine_data, methods=['telegram'])

#     except Exception as e:
#         print(f"⚠️  Diagnosis upload error: {e}")

#     return prediction


# # ══════════════════════════════════════════════════════════════════════════════
# # TRAINING PIPELINE
# # ══════════════════════════════════════════════════════════════════════════════

# def train_pipeline():
#     """Complete model training pipeline."""
#     print("=" * 60)
#     print("  MODEL TRAINING PIPELINE")
#     print("=" * 60)

#     # Step 1: Load training data
#     print("\n📊 Step 1: Loading training data...")
#     try:
#         X, y = load_training_data()
#         print(f"✅ Training data: {X.shape[0]} samples, {X.shape[1]} features")
#         print(f"   Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
#     except Exception as e:
#         print(f"❌ Error loading training data: {e}")
#         return False

#     # Step 2: Fit and save scaler
#     print("\n🔧 Step 2: Fitting feature scaler...")
#     try:
#         from preprocessing.scaler import fit_and_save_scaler
#         scaler = fit_and_save_scaler(X, config.SCALER_PATH)
#         X_scaled = scaler.transform(X)
#         print(f"✅ Scaler fitted and saved")
#     except Exception as e:
#         print(f"⚠️  Scaler error: {e} — continuing without scaling")
#         X_scaled = X

#     # Step 3: Train main model
#     print("\n🤖 Step 3: Training RandomForest classifier...")
#     try:
#         from models.train_model import train_model
#         model = train_model(X_scaled, y)
#         print(f"✅ Model saved to: {config.MODEL_PATH}")
#     except Exception as e:
#         print(f"❌ Error training model: {e}")
#         import traceback
#         traceback.print_exc()
#         return False

#     # Step 4: Train anomaly detector (optional)
#     print("\n🔬 Step 4: Training anomaly detector...")
#     try:
#         from models.anomaly_detector import train_anomaly_model
#         train_anomaly_model(X_scaled)
#         print(f"✅ Anomaly detector trained and saved")
#     except Exception as e:
#         print(f"⚠️  Anomaly detector skipped: {e}")

#     print("\n" + "=" * 60)
#     print("  ✅ TRAINING COMPLETE!")
#     print("=" * 60)
#     print(f"\nTrained models saved:")
#     print(f"  📦 {config.MODEL_PATH}")
#     print(f"  📦 {config.SCALER_PATH}")
#     print("\nLaunch dashboard:")
#     print("  streamlit run dashboard/dashboard_app.py")
#     return True


# # ══════════════════════════════════════════════════════════════════════════════
# # SYSTEM STATUS
# # ══════════════════════════════════════════════════════════════════════════════

# def show_status():
#     """Display full system status."""
#     print("=" * 60)
#     print("  AI-Powered Predictive Maintenance System — STATUS")
#     print("=" * 60)

#     # Data mode
#     mode     = get_sensor_mode()
#     data_mode = get_current_data_mode()
#     df, actual_mode = load_sensor_data()

#     print(f"\n  Sensor Mode   : {mode}")
#     print(f"  Data Mode     : {data_mode}")
#     print(f"  Actual Source : {actual_mode}")
#     print(f"  IoT Sensor    : {'✅ Connected' if mode == 'LIVE' else '❌ Not connected'}")
#     print(f"  Data Rows     : {len(df)}")

#     if 'machine_id' in df.columns:
#         print(f"  Machines      : {sorted(df['machine_id'].unique().tolist())}")

#     # Cloud status
#     print("\n" + "=" * 60)
#     print("  CLOUD STATUS")
#     print("=" * 60)
#     try:
#         from cloud import status as cloud_status
#         svc = cloud_status()
#         for svc_name, info in svc.items():
#             cfg = "✅" if info['configured'] else "❌"
#             ena = "✅" if info['enabled']    else "❌"
#             print(f"  {svc_name.capitalize():12s}: Configured={cfg}  Enabled={ena}")
#     except Exception as e:
#         print(f"  ⚠️  Cloud status error: {e}")

#     # Model status
#     print("\n" + "=" * 60)
#     print("  MODEL STATUS")
#     print("=" * 60)

#     model  = load_model()
#     scaler = load_scaler()

#     print(f"  Classifier : {'✅ Loaded' if model  else '❌ Not trained (run --train)'}")
#     print(f"  Scaler     : {'✅ Loaded' if scaler else '❌ Not fitted (run --train)'}")
#     print(f"  Anomaly Det: {'✅ Loaded' if os.path.exists(config.ANOMALY_MODEL_PATH) else '❌ Not trained'}")

#     # Predictions
#     if not df.empty:
#         print("\n" + "=" * 60)
#         print("  CURRENT PREDICTIONS")
#         print("=" * 60)
#         try:
#             prediction = predict_maintenance(df, model, scaler)
#             print(f"  Fleet Status  : {prediction.get('fleet_status', 'UNKNOWN')}")
#             print(f"  Avg Health    : {prediction.get('avg_health', 0):.1f}%")
#             print(f"  Avg Risk      : {prediction.get('avg_risk', 0):.1f}%")
#             print(f"  Critical Count: {prediction.get('critical_count', 0)}")

#             if 'machines' in prediction:
#                 print("\n  Machine Details:")
#                 for m_id, m in list(prediction['machines'].items())[:5]:
#                     print(
#                         f"    {m_id}: {m.get('health_status','?'):8s} | "
#                         f"Health: {m.get('health_score',0):5.1f}% | "
#                         f"Risk: {m.get('failure_risk',0):5.1f}% | "
#                         f"RUL: {m.get('rul_days',0):3d}d"
#                     )
#         except Exception as e:
#             print(f"  ⚠️  Prediction error: {e}")

#     print("\n" + "=" * 60)


# # ══════════════════════════════════════════════════════════════════════════════
# # ONE-TIME DATABASE SYNC  (upload historical CSV to Supabase)
# # ══════════════════════════════════════════════════════════════════════════════

# def sync_historical_to_database():
#     """
#     One-time sync: upload your historical_maintenance.csv to Supabase.
#     Run this ONCE to populate the cloud database with your existing data.
#     After this, the sensor will keep adding new readings automatically.
#     """
#     print("=" * 60)
#     print("  SYNCING HISTORICAL DATA TO SUPABASE")
#     print("=" * 60)

#     if not os.path.exists(config.HISTORICAL_PATH):
#         print(f"❌ Historical file not found: {config.HISTORICAL_PATH}")
#         return False

#     try:
#         from cloud.database import sync_csv_to_cloud
#         count = sync_csv_to_cloud(config.HISTORICAL_PATH)
#         print(f"\n✅ Synced {count} records to Supabase")
#         return True
#     except Exception as e:
#         print(f"❌ Sync failed: {e}")
#         return False


# # ══════════════════════════════════════════════════════════════════════════════
# # LAUNCH DASHBOARD
# # ══════════════════════════════════════════════════════════════════════════════

# def launch_dashboard():
#     """Launch Streamlit dashboard."""
#     import subprocess

#     dashboard_path = os.path.join(PROJECT_ROOT, "dashboard", "dashboard_app.py")

#     if not os.path.exists(dashboard_path):
#         print(f"❌ Dashboard not found at: {dashboard_path}")
#         return

#     print("=" * 60)
#     print("  LAUNCHING DASHBOARD")
#     print("=" * 60)
#     print(f"  URL: http://localhost:{config.DASHBOARD_PORT}")
#     print(f"  Data mode: {get_current_data_mode()}")
#     print("\n  Press Ctrl+C to stop")
#     print("=" * 60)

#     try:
#         subprocess.run([
#             sys.executable, "-m", "streamlit", "run",
#             dashboard_path,
#             "--server.port", str(config.DASHBOARD_PORT)
#         ])
#     except KeyboardInterrupt:
#         print("\n✅ Dashboard stopped")


# # ══════════════════════════════════════════════════════════════════════════════
# # MAIN
# # ══════════════════════════════════════════════════════════════════════════════

# def main():
#     parser = argparse.ArgumentParser(
#         description="AI-Powered Predictive Maintenance System"
#     )
#     parser.add_argument("--train",      action="store_true", help="Train ML models")
#     parser.add_argument("--status",     action="store_true", help="Show system status")
#     parser.add_argument("--sync-db",    action="store_true", help="Upload historical CSV to Supabase (one-time)")
#     parser.add_argument("--test-cloud", action="store_true", help="Test cloud database & messaging")
#     parser.add_argument(
#         "--mode",
#         choices=["LIVE", "HISTORICAL", "FALLBACK", "DUMMY"],
#         help="Force data mode (for testing)"
#     )

#     args = parser.parse_args()

#     if args.train:
#         train_pipeline()

#     elif args.status:
#         show_status()

#     elif args.sync_db:
#         sync_historical_to_database()

#     elif args.test_cloud:
#         try:
#             from cloud import test_all
#             success = test_all()
#             sys.exit(0 if success else 1)
#         except Exception as e:
#             print(f"❌ Cloud test error: {e}")
#             sys.exit(1)

#     else:
#         launch_dashboard()


# if __name__ == "__main__":
#     main()



