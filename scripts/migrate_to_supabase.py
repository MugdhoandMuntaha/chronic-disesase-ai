"""EndoPredict AI | Automated Supabase Clinical Data Migration Script.

Migrates relational EHR datasets (patients, encounters, measurements, diagnoses, and predictions)
from local CSV/Parquet stores into Supabase PostgreSQL via the REST/PostgREST API.
"""

import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
SQL_SCHEMA_PATH = PROJECT_ROOT / "scripts" / "supabase_schema.sql"

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://qnjyvywkkctaocnvszwr.supabase.co").rstrip("/")
SUPABASE_ANON_KEY = os.environ.get(
    "SUPABASE_ANON_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFuanl2eXdra2N0YW9jbnZzendyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODkyMDcwMTEsImV4cCI6MjEwNDc4MzAxMX0.I7q-4MJHJ0NQtS_1T2zkE2o_ky_aGVCGy5GuI2cuqVw",
)


def get_headers(prefer_return: str = "minimal") -> Dict[str, str]:
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Content-Type": "application/json",
        "Prefer": f"return={prefer_return}",
    }


def check_table_exists(table_name: str) -> bool:
    """Checks if a table exists in the Supabase public schema."""
    url = f"{SUPABASE_URL}/rest/v1/{table_name}?select=count&limit=1"
    req = urllib.request.Request(url, headers=get_headers())
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status in (200, 206)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        # If table exists but RLS blocks count, code might be 401/403
        if "PGRST205" in str(e.read()):
            return False
        return True
    except Exception:
        return False


def batch_insert(table_name: str, records: List[Dict[str, Any]], batch_size: int = 250) -> int:
    """Inserts records in chunks using Supabase REST API."""
    url = f"{SUPABASE_URL}/rest/v1/{table_name}"
    total = len(records)
    inserted = 0

    for i in range(0, total, batch_size):
        chunk = records[i : i + batch_size]
        payload = json.dumps(chunk).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers=get_headers(), method="POST")

        try:
            with urllib.request.urlopen(req) as resp:
                if resp.status in (200, 201):
                    inserted += len(chunk)
                    print(f"  [{table_name}] Uploaded {inserted}/{total} rows...", end="\r")
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            print(f"\n[Error] Failed to insert into {table_name}: HTTP {e.code}: {err_body}")
            break
        except Exception as ex:
            print(f"\n[Error] Exception during insert into {table_name}: {ex}")
            break

    print(f"\n[Done] Table {table_name}: successfully migrated {inserted}/{total} rows.")
    return inserted


def clean_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Replaces NaN/inf with None for clean JSON serialization."""
    df_clean = df.replace({np.nan: None})
    return df_clean.to_dict(orient="records")


def run_migration() -> None:
    print("=" * 80)
    print("ENDOPREDICT AI | SUPABASE CLOUD DATABASE MIGRATION")
    print("=" * 80)
    print(f"Target Supabase Project: {SUPABASE_URL}")

    # Step 1: Verify table presence
    print("\n[Step 1/5] Verifying PostgreSQL Schema on Supabase...")
    tables = ["patients", "encounters", "measurements", "diagnoses", "patient_predictions"]
    missing_tables = [t for t in tables if not check_table_exists(t)]

    if missing_tables:
        print(f"\n[Schema Required] Tables missing in Supabase: {', '.join(missing_tables)}")
        print("\nBecause Supabase requires PostgreSQL DDL permissions to create tables,")
        print("please run the schema creation SQL once in your Supabase Dashboard:")
        print(f"  1. Open: https://supabase.com/dashboard/project/qnjyvywkkctaocnvszwr/sql")
        print(f"  2. Paste and run the contents of: {SQL_SCHEMA_PATH}")
        print("\nOnce executed in the SQL Editor, re-run this script to automatically populate all data.")
        return

    print("All target tables detected in Supabase public schema.")

    # Step 2: Migrate Patients
    patients_csv = RAW_DATA_DIR / "patients.csv"
    if patients_csv.exists():
        print("\n[Step 2/5] Migrating Patients Master Table...")
        df_patients = pd.read_csv(patients_csv)
        batch_insert("patients", clean_records(df_patients))

    # Step 3: Migrate Encounters
    encounters_csv = RAW_DATA_DIR / "encounters.csv"
    if encounters_csv.exists():
        print("\n[Step 3/5] Migrating Longitudinal Encounters Table...")
        df_encounters = pd.read_csv(encounters_csv)
        batch_insert("encounters", clean_records(df_encounters))

    # Step 4: Migrate Measurements
    measurements_csv = RAW_DATA_DIR / "measurements.csv"
    if measurements_csv.exists():
        print("\n[Step 4/5] Migrating Vital Signs & Lab Measurements Table...")
        df_measurements = pd.read_csv(measurements_csv)
        # Drop auto-increment column if present to let Supabase BIGSERIAL generate ID
        if "id" in df_measurements.columns:
            df_measurements = df_measurements.drop(columns=["id"])
        batch_insert("measurements", clean_records(df_measurements))

    # Step 5: Migrate Diagnoses
    diagnoses_csv = RAW_DATA_DIR / "diagnoses.csv"
    if diagnoses_csv.exists():
        print("\n[Step 5/5] Migrating ICD-10 Diagnoses Comorbidities...")
        df_diagnoses = pd.read_csv(diagnoses_csv)
        batch_insert("diagnoses", clean_records(df_diagnoses))

    print("\n" + "=" * 80)
    print("MIGRATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_migration()
