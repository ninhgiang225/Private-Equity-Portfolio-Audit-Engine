"""src/utils/bigquery.py

Thin helpers for reading/writing DataFrames to BigQuery.
Only needed in Week 4 — everything before that uses CSV.

Prerequisites:
  1. Set GOOGLE_APPLICATION_CREDENTIALS in .env
  2. Set BIGQUERY_PROJECT_ID and BIGQUERY_DATASET in .env
  3. pip install google-cloud-bigquery db-dtypes
"""
from __future__ import annotations
import pandas as pd
from src.utils.config import BQ_PROJECT, BQ_DATASET


def write_dataframe(
    df: pd.DataFrame,
    table: str,
    if_exists: str = "replace",
) -> None:
    """
    Write a DataFrame to BigQuery.

    Args:
        df:        The DataFrame to write.
        table:     Table name (without project/dataset prefix).
        if_exists: 'replace' overwrites; 'append' adds rows.
    """
    from google.cloud import bigquery  # lazy import — only needed Week 4

    client    = bigquery.Client(project=BQ_PROJECT)
    table_ref = f"{BQ_PROJECT}.{BQ_DATASET}.{table}"

    job_config = bigquery.LoadJobConfig(
        write_disposition=(
            bigquery.WriteDisposition.WRITE_TRUNCATE
            if if_exists == "replace"
            else bigquery.WriteDisposition.WRITE_APPEND
        ),
        autodetect=True,
    )

    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()  # wait for completion
    print(f"[bigquery] Wrote {len(df)} rows → {table_ref}")


def read_table(table: str, limit: int | None = None) -> pd.DataFrame:
    """
    Read a BigQuery table into a DataFrame.

    Args:
        table: Table name (without project/dataset prefix).
        limit: Optional row limit for quick checks.
    """
    from google.cloud import bigquery

    client = bigquery.Client(project=BQ_PROJECT)
    query  = f"SELECT * FROM `{BQ_PROJECT}.{BQ_DATASET}.{table}`"
    if limit:
        query += f" LIMIT {limit}"

    return client.query(query).to_dataframe()
