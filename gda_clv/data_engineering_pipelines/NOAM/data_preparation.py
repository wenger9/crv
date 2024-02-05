'''
NOTE
-------
Author | rjaswal@estee.com
Date   | 2023-12-29 
-------
Objective:
- Prepare and preprocess data for CLV model training.
- Perform necessary data cleaning, transformation (and in the future, feature engineering / target labeling)

Key Functionality:
- Load raw data from specified sources.
- Clean and preprocess data (handling missing values, outliers, etc.).
- Generate and select features relevant for CLV modeling.
- Split data into training and testing sets.
- Save processed data for model training.
'''


import pandas as pd
from pyspark.sql import SparkSession

# Initialize Spark Session
spark = SparkSession.builder.appName("CLV Data Preparation").getOrCreate()

def create_view_noam_txn_data_bronze():
    """
    Create or replace view clv.vw_noam_txn_data_bronze to filter and preprocess data.
    This view filters transactions based on various criteria like region, valid indicators, and more.
    """
    spark.sql("""
        create or replace view clv.vw_noam_txn_data_bronze as 
        select elc_master_id, consumer_id, brand_id, sale_dt, txn_id, usd_net_order_total, currency_cd, channel_cd
        from delta.`abfss://ds-gda@saameusdevdsfdata01.dfs.core.windows.net/data_foundation/bronze/full/t_consumer_transaction_header_noam`
        where cdp_region = 'noam' and elc_master_id is not null 
        and consumer_id IS NOT NULL and valid_ind = 1 and consumer_valid_flag = 'Y' and mac_pro_flag = 'N' and usd_net_order_total > 0 
        and currency_cd in ('USD', 'CAD') and visit_ind = 1
        and CDC_CODE <> "D"
    """)

def create_training_data_table():
    """
    Create a monthly partitioned table to store training data.
    This table includes various fields like snapshot_dt, CustomerID, frequency_cal, and more.
    """
    spark.sql("""
        create table if not exists clv.master_train_noam (
            snapshot_dt date,
            CustomerID string,
            frequency_cal float,
            recency_cal float,
            T_cal float,
            monetary_value_cal double,
            frequency_holdout double,
            monetary_value_holdout double,
            duration_holdout float
        )
        partitioned by (snapshot_dt);
    """)

def generate_training_data(period, duration):
    """
    Generate training data for the given period and duration.
    This function calculates various metrics like frequency, recency, monetary value for both calibration and holdout periods.
    """
    period_end = spark.sql(f"select date_sub(add_months('{period}', 1),1) d ").toPandas().d.astype(str).values[0]
    qry_train = f"""
    WITH 
    orders as (
        SELECT elc_master_id customerid, sale_dt as invoicedate, usd_net_order_total as salesamount 
        FROM clv.vw_noam_txn_data_bronze
        WHERE sale_dt BETWEEN '2013-06-01' AND '@period_end'
    ),
    CustomerHistory  AS (
        SELECT m.*, @duration as duration_holdout
        FROM (
            SELECT x.customerid, z.first_at, x.transaction_at, y.current_dt, x.salesamount
            FROM (
                SELECT customerid, TO_DATE(invoicedate) as transaction_at, SUM(SalesAmount) as salesamount 
                FROM orders 
                GROUP BY customerid, TO_DATE(invoicedate)
            ) x
            CROSS JOIN (SELECT MAX(TO_DATE(invoicedate)) as current_dt FROM orders) y
            INNER JOIN (SELECT customerid, MIN(TO_DATE(invoicedate)) as first_at FROM orders GROUP BY customerid) z
            ON x.customerid=z.customerid
            WHERE x.customerid is not null
        ) m
    )
    SELECT
        to_date(date_trunc('MONTH', '@period_end')) as snapshot_dt,
        a.customerid as CustomerID,
        a.frequency as frequency_cal,
        a.recency as recency_cal,
        a.T as T_cal,
        COALESCE(a.monetary_value,0.0) as monetary_value_cal,
        COALESCE(b.frequency_holdout, 0.0) as frequency_holdout,
        COALESCE(b.monetary_value_holdout, 0.0) as monetary_value_holdout,
        a.duration_holdout
    FROM (
        SELECT
            p.customerid,
            CAST(p.duration_holdout as float) as duration_holdout,
            CAST(DATEDIFF(MAX(p.transaction_at), p.first_at) as float) as recency,
            CAST(COUNT(DISTINCT p.transaction_at) - 1 as float) as frequency,
            CAST(DATEDIFF(DATE_SUB(p.current_dt, p.duration_holdout), p.first_at) as float) as T,
            CASE
            WHEN COUNT(DISTINCT p.transaction_at)=1 THEN 0
            ELSE
                SUM(
                CASE WHEN p.first_at=p.transaction_at THEN 0
                ELSE p.salesamount
                END
                ) / (COUNT(DISTINCT p.transaction_at)-1)
            END as monetary_value    
        FROM CustomerHistory p
        WHERE p.transaction_at < DATE_SUB(p.current_dt, p.duration_holdout)  
        GROUP BY p.customerid, p.duration_holdout, p.current_dt, p.first_at
    ) a
    LEFT OUTER JOIN (
        SELECT
            p.customerid,
            CAST(COUNT(DISTINCT p.transaction_at) as float) as frequency_holdout,
            AVG(p.salesamount) as monetary_value_holdout
        FROM CustomerHistory p
        WHERE 
            p.transaction_at >= DATE_SUB(p.current_dt, p.duration_holdout) AND
            p.transaction_at <= p.current_dt
        GROUP BY p.customerid
    ) b
    ON a.customerid=b.customerid
    ORDER BY CustomerID
    """

    new_qry = qry_train.replace("@period_end", period_end).replace("@duration", str(duration))
    spark.sql(new_qry).write.mode("overwrite").option("replaceWhere", f"snapshot_dt = '{period}'").saveAsTable('clv.master_train_noam')

def create_full_dataset_table():
    """
    Create a table to store the full dataset.
    This table includes snapshot_dt, CustomerID, frequency, recency, T, and monetary_value.
    """
    spark.sql("""
        create table if not exists clv.master_full_noam (
            snapshot_dt date,
            CustomerID string,
            frequency float,
            recency float,
            T float,
            monetary_value double
        )
        partitioned by (snapshot_dt);
    """)

def generate_full_dataset(period):
    """
    Generate the full dataset for the given period.
    This function calculates frequency, recency, T, and monetary_value for each customer.
    """
    period_end = spark.sql(f"select date_sub(add_months('{period}', 1),1) d ").toPandas().d.astype(str).values[0]
    qry_full = f"""
    WITH 
    orders as (
        SELECT elc_master_id AS customerid, sale_dt AS invoicedate, usd_net_order_total  AS salesamount 
        FROM clv.vw_noam_txn_data_bronze
        WHERE sale_dt BETWEEN '2013-06-01' AND '@period_end'
    ),
    x AS (
        SELECT customerid, TO_DATE(invoicedate) AS transaction_at, SUM(SalesAmount) AS salesamount
        FROM orders
        GROUP BY customerid, TO_DATE(invoicedate)
    ),
    y AS (SELECT MAX(TO_DATE(invoicedate)) AS current_dt FROM orders),
    z AS (SELECT customerid, MIN(TO_DATE(invoicedate)) AS first_at FROM orders GROUP BY customerid),
    a AS (
        SELECT x.customerid, z.first_at, x.transaction_at, y.current_dt,  x.salesamount                  
        FROM x
        CROSS JOIN y
        INNER JOIN z ON x.customerid=z.customerid
        WHERE x.customerid IS NOT NULL
    ) 
    SELECT
        to_date(date_trunc('MONTH', '@period_end')) as snapshot_dt,
        a.customerid AS CustomerID,
        CAST(COUNT(DISTINCT a.transaction_at) - 1 AS float) AS frequency,
        CAST(DATEDIFF(MAX(a.transaction_at), a.first_at) AS float) AS recency,
        CAST(DATEDIFF(a.current_dt, a.first_at) AS float) AS T,
        CASE
            WHEN COUNT(DISTINCT a.transaction_at)=1 THEN 0
            ELSE SUM(CASE WHEN a.first_at=a.transaction_at THEN 0
                    ELSE a.salesamount END) / (COUNT(DISTINCT a.transaction_at)-1)
        END as monetary_value    
    FROM a
    GROUP BY snapshot_dt, a.customerid, a.current_dt, a.first_at
    ORDER BY CustomerID
    """
    new_qry = qry_full.replace("@period_end", period_end)
    spark.sql(new_qry).write.mode("overwrite").option("replaceWhere", f"snapshot_dt = '{period}'").saveAsTable('clv.master_full_noam')

def main():
    """
    Main function to execute the data preparation steps.
    """
    create_view_noam_txn_data_bronze()
    create_training_data_table()
    generate_training_data('2023-06-01', 365)
    create_full_dataset_table()
    generate_full_dataset('2023-06-01')

if __name__ == "__main__":
    main()
