import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.font_manager as fm
import datetime
from typing import Dict, Tuple, Set

BASE_DIR = Path(r'D:/DATATHON/datathon-2026-round-1')
CHART_DIR = Path(r'D:/DATATHON/outputs/charts')
SUMMARY_DIR = Path(r'D:/DATATHON/outputs/summary_table')


try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    import seaborn as sns
except ModuleNotFoundError:
    plt = None
    FuncFormatter = None
    sns = None


def money_formatter(value: float, _: int) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


def load_data(base_dir: Path) -> dict[str, pd.DataFrame]:
    csv_files = [
        "orders",
        "order_items",
        "products",
        "payments",
        "inventory",
        "customers",
    ]
    data = {
        name: pd.read_csv(base_dir / f"{name}.csv", low_memory=False) for name in csv_files
    }

    data["orders"]["order_date"] = pd.to_datetime(data["orders"]["order_date"], errors="coerce")
    data["inventory"]["snapshot_date"] = pd.to_datetime(
        data["inventory"]["snapshot_date"], errors="coerce"
    )
    return data


def build_sales_fact(
    orders: pd.DataFrame, order_items: pd.DataFrame, products: pd.DataFrame
) -> pd.DataFrame:
    product_lookup = products[["product_id", "price", "cogs", "category", "segment"]].copy()

    sales_fact = (
        order_items.merge(product_lookup, on="product_id", how="left", validate="m:1")
        .merge(
            orders[
                [
                    "order_id",
                    "order_date",
                    "customer_id",
                    "order_status",
                    "payment_method",
                    "order_source",
                ]
            ],
            on="order_id",
            how="left",
            validate="m:1",
        )
        .assign(
            price=lambda df: df["price"].fillna(df["unit_price"]),
            discount_amount=lambda df: df["discount_amount"].fillna(0.0),
        )
    )

    sales_fact["gross_revenue"] = sales_fact["quantity"] * sales_fact["price"]
    sales_fact["net_revenue"] = (
        sales_fact["quantity"] * sales_fact["unit_price"] - sales_fact["discount_amount"]
    )
    sales_fact["cogs_value"] = sales_fact["quantity"] * sales_fact["cogs"]
    sales_fact["gross_margin"] = sales_fact["net_revenue"] - sales_fact["cogs_value"]
    sales_fact["discount_rate"] = np.where(
        sales_fact["quantity"] * sales_fact["price"] > 0,
        sales_fact["discount_amount"] / (sales_fact["quantity"] * sales_fact["price"]),
        0.0,
    )
    sales_fact["order_month"] = sales_fact["order_date"].dt.to_period("M")
    return sales_fact


def build_monthly_pnl(sales_fact: pd.DataFrame) -> pd.DataFrame:
    return (
        sales_fact.groupby("order_month", as_index=False)
        .agg(
            Gross_Revenue=("gross_revenue", "sum"),
            Net_Revenue=("net_revenue", "sum"),
            COGS=("cogs_value", "sum"),
            Gross_Margin=("gross_margin", "sum"),
            Order_Count=("order_id", "nunique"),
            Item_Quantity=("quantity", "sum"),
            Avg_Discount_Rate=("discount_rate", "mean"),
        )
        .rename(columns={"order_month": "month_period"})
    )


def build_cash_in_schedule(orders: pd.DataFrame, payments: pd.DataFrame) -> pd.DataFrame:
    payments_clean = (
        payments.groupby("order_id", as_index=False)
        .agg(
            payment_value=("payment_value", "sum"),
            installments=("installments", "max"),
        )
        .assign(installments=lambda df: df["installments"].fillna(1).clip(lower=1).astype(int))
    )

    order_payment = orders[["order_id", "order_date"]].merge(
        payments_clean, on="order_id", how="inner", validate="1:1"
    )
    order_payment = order_payment.dropna(subset=["order_date", "payment_value"]).copy()
    order_payment["cash_per_installment"] = (
        order_payment["payment_value"] / order_payment["installments"]
    )

    repeated = order_payment.loc[
        order_payment.index.repeat(order_payment["installments"])
    ].reset_index(drop=True)
    repeated["installment_no"] = repeated.groupby("order_id").cumcount() + 1
    repeated["month_offset"] = repeated["installment_no"] - 1
    repeated["cash_in_date"] = (
        repeated["order_date"].dt.to_period("M") + repeated["month_offset"]
    ).dt.to_timestamp()
    repeated["month_period"] = repeated["cash_in_date"].dt.to_period("M")
    repeated["Cash_In_Actual"] = repeated["cash_per_installment"]
    return repeated[
        [
            "order_id",
            "order_date",
            "payment_value",
            "installments",
            "installment_no",
            "cash_in_date",
            "month_period",
            "Cash_In_Actual",
        ]
    ]


def build_inventory_cash_out(
    inventory: pd.DataFrame, products: pd.DataFrame
) -> pd.DataFrame:
    inventory_lookup = products[["product_id", "cogs"]].copy()
    inventory_cash = (
        inventory.merge(inventory_lookup, on="product_id", how="left", validate="m:1").assign(
            units_received=lambda df: df["units_received"].fillna(0)
        )
    )
    inventory_cash["Cash_Out_Inventory"] = (
        inventory_cash["units_received"] * inventory_cash["cogs"]
    )
    inventory_cash["month_period"] = inventory_cash["snapshot_date"].dt.to_period("M")
    return inventory_cash


def build_monthly_cash_out(inventory_cash: pd.DataFrame) -> pd.DataFrame:
    return (
        inventory_cash.groupby("month_period", as_index=False)
        .agg(
            Cash_Out_Inventory=("Cash_Out_Inventory", "sum"),
            Units_Received=("units_received", "sum"),
        )
    )


def build_financial_summary(
    monthly_pnl: pd.DataFrame,
    cash_in_schedule: pd.DataFrame,
    monthly_cash_out: pd.DataFrame,
) -> pd.DataFrame:
    monthly_cash_in = cash_in_schedule.groupby("month_period", as_index=False).agg(
        Cash_In_Actual=("Cash_In_Actual", "sum")
    )

    financial_summary_df = (
        monthly_pnl.merge(monthly_cash_in, on="month_period", how="outer")
        .merge(monthly_cash_out, on="month_period", how="outer")
        .sort_values("month_period")
        .fillna(0)
    )
    financial_summary_df["Net_Cash_Flow"] = (
        financial_summary_df["Cash_In_Actual"] - financial_summary_df["Cash_Out_Inventory"]
    )
    financial_summary_df["month_start"] = financial_summary_df["month_period"].dt.to_timestamp()
    financial_summary_df["year"] = financial_summary_df["month_period"].dt.year
    financial_summary_df["month"] = financial_summary_df["month_period"].dt.month
    financial_summary_df["month_label"] = financial_summary_df["month_period"].astype(str)

    ordered_columns = [
        "year",
        "month",
        "month_label",
        "month_start",
        "Gross_Revenue",
        "Net_Revenue",
        "COGS",
        "Gross_Margin",
        "Cash_In_Actual",
        "Cash_Out_Inventory",
        "Net_Cash_Flow",
        "Order_Count",
        "Item_Quantity",
        "Units_Received",
        "Avg_Discount_Rate",
    ]
    return financial_summary_df[ordered_columns].copy()


def build_tableau_long_extract(financial_summary_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "Gross_Revenue",
        "Net_Revenue",
        "COGS",
        "Gross_Margin",
        "Cash_In_Actual",
        "Cash_Out_Inventory",
        "Net_Cash_Flow",
    ]
    long_extract = financial_summary_df.melt(
        id_vars=["year", "month", "month_label", "month_start"],
        value_vars=metrics,
        var_name="metric",
        value_name="amount",
    )
    long_extract["statement_type"] = long_extract["metric"].map(
        {
            "Gross_Revenue": "P&L Proxy",
            "Net_Revenue": "P&L Proxy",
            "COGS": "P&L Proxy",
            "Gross_Margin": "P&L Proxy",
            "Cash_In_Actual": "Cash Flow Proxy",
            "Cash_Out_Inventory": "Cash Flow Proxy",
            "Net_Cash_Flow": "Cash Flow Proxy",
        }
    )
    return long_extract


def build_annual_summary(financial_summary_df: pd.DataFrame) -> pd.DataFrame:
    annual_summary = (
        financial_summary_df.groupby("year", as_index=False)
        .agg(
            Gross_Revenue=("Gross_Revenue", "sum"),
            Net_Revenue=("Net_Revenue", "sum"),
            COGS=("COGS", "sum"),
            Gross_Margin=("Gross_Margin", "sum"),
            Cash_In_Actual=("Cash_In_Actual", "sum"),
            Cash_Out_Inventory=("Cash_Out_Inventory", "sum"),
            Net_Cash_Flow=("Net_Cash_Flow", "sum"),
            Order_Count=("Order_Count", "sum"),
        )
        .sort_values("year")
    )
    annual_summary = annual_summary[
        (annual_summary["Net_Revenue"] != 0) | (annual_summary["Cash_In_Actual"] != 0)
    ].copy()
    annual_summary["Margin_Rate"] = np.where(
        annual_summary["Net_Revenue"] != 0,
        annual_summary["Gross_Margin"] / annual_summary["Net_Revenue"],
        0.0,
    )
    return annual_summary


def build_profit_segment_summary(
    sales_fact: pd.DataFrame,
    customers: pd.DataFrame,
    products: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    customer_lookup = customers[["customer_id", "gender", "age_group"]].copy()
    enriched_sales = sales_fact.merge(
        customer_lookup, on="customer_id", how="left", validate="m:1"
    ).copy()
    enriched_sales["year"] = enriched_sales["order_date"].dt.year

    price_thresholds = products["price"].quantile([0.4, 0.7]).to_dict()
    medium_start = price_thresholds[0.4]
    high_start = price_thresholds[0.7]

    enriched_sales["price_bucket"] = pd.cut(
        enriched_sales["price"],
        bins=[-np.inf, medium_start, high_start, np.inf],
        labels=["Low", "Medium", "High"],
        right=False,
    ).astype("object")
    enriched_sales["price_bucket"] = enriched_sales["price_bucket"].fillna("Low")

    gender_clean = (
        enriched_sales["gender"].fillna("Unknown").astype(str).str.strip().str.lower()
    )
    male_mask = gender_clean.eq("male")
    female_mask = gender_clean.eq("female")
    nb_mask = gender_clean.eq("non-binary")

    male_view = enriched_sales.loc[male_mask | nb_mask].copy()
    male_view["gender_report"] = "Male"
    male_view["gender_weight"] = np.where(nb_mask.loc[male_view.index], 0.5, 1.0)

    female_view = enriched_sales.loc[female_mask | nb_mask].copy()
    female_view["gender_report"] = "Female"
    female_view["gender_weight"] = np.where(nb_mask.loc[female_view.index], 0.5, 1.0)

    gender_sales = pd.concat([male_view, female_view], ignore_index=True)
    gender_sales["weighted_revenue"] = gender_sales["net_revenue"] * gender_sales["gender_weight"]
    gender_sales["weighted_cost"] = gender_sales["cogs_value"] * gender_sales["gender_weight"]
    gender_sales["weighted_profit"] = (
        gender_sales["weighted_revenue"] - gender_sales["weighted_cost"]
    )

    detail_summary = (
        gender_sales.groupby(
            ["year", "gender_report", "price_bucket", "age_group"], as_index=False
        )
        .agg(
            revenue=("weighted_revenue", "sum"),
            cost=("weighted_cost", "sum"),
            profit=("weighted_profit", "sum"),
        )
    )

    bucket_summary = (
        gender_sales.groupby(["year", "gender_report", "price_bucket"], as_index=False)
        .agg(
            revenue=("weighted_revenue", "sum"),
            cost=("weighted_cost", "sum"),
            profit=("weighted_profit", "sum"),
        )
        .assign(age_group="All Ages")
    )

    combined = pd.concat([bucket_summary, detail_summary], ignore_index=True)
    combined["prev_profit"] = combined.groupby(
        ["gender_report", "price_bucket", "age_group"]
    )["profit"].shift(1)

    total_profit = combined.groupby(["year", "gender_report"], as_index=False).agg(
        total_profit=("profit", lambda s: s.loc[combined.loc[s.index, "age_group"].eq("All Ages")].sum())
    )
    combined = combined.merge(total_profit, on=["year", "gender_report"], how="left")
    combined["profit_change"] = combined["profit"] - combined["prev_profit"]
    combined["profit_share"] = np.where(
        combined["total_profit"] != 0, combined["profit"] / combined["total_profit"], 0.0
    )
    combined["pct_change_vs_last_year"] = np.where(
        combined["prev_profit"].notna() & (combined["prev_profit"] != 0),
        combined["profit_change"] / combined["prev_profit"],
        np.nan,
    )
    combined["pct_contribute_to_year_change"] = np.where(
        combined["total_profit"] != 0,
        combined["profit_change"] / combined["total_profit"],
        np.nan,
    )
    combined["row_type"] = np.where(combined["age_group"].eq("All Ages"), "bucket", "detail")

    bucket_order = {"Low": 0, "Medium": 1, "High": 2}
    age_order = {"All Ages": -1, "18-24": 0, "25-34": 1, "35-44": 2, "45-54": 3, "55+": 4}
    combined["bucket_order"] = combined["price_bucket"].map(bucket_order)
    combined["age_order"] = combined["age_group"].map(age_order).fillna(99)
    combined = combined.sort_values(
        ["gender_report", "year", "bucket_order", "age_order"]
    ).reset_index(drop=True)

    threshold_df = pd.DataFrame(
        {
            "metric": ["medium_start_p40", "high_start_p70"],
            "value": [medium_start, high_start],
        }
    )
    return combined, threshold_df


def build_high_customer_pareto(
    sales_fact: pd.DataFrame,
    customers: pd.DataFrame,
    products: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    customer_lookup = customers[["customer_id", "gender", "age_group"]].copy()
    enriched_sales = sales_fact.merge(
        customer_lookup, on="customer_id", how="left", validate="m:1"
    ).copy()

    high_start = products["price"].quantile(0.7)
    high_sales = enriched_sales.loc[enriched_sales["price"] >= high_start].copy()

    high_customer_profit = (
        high_sales.groupby(["customer_id", "gender", "age_group"], as_index=False)
        .agg(
            high_revenue=("net_revenue", "sum"),
            high_cost=("cogs_value", "sum"),
        )
    )
    high_customer_profit["high_profit"] = (
        high_customer_profit["high_revenue"] - high_customer_profit["high_cost"]
    )
    high_customer_profit = high_customer_profit.sort_values(
        "high_profit", ascending=False
    ).reset_index(drop=True)
    high_customer_profit = high_customer_profit.loc[
        high_customer_profit["high_profit"] > 0
    ].reset_index(drop=True)
    high_customer_profit["customer_rank"] = np.arange(1, len(high_customer_profit) + 1)
    high_customer_profit["customer_share"] = (
        high_customer_profit["customer_rank"] / len(high_customer_profit)
        if len(high_customer_profit) > 0
        else 0
    )
    high_customer_profit["cum_profit"] = high_customer_profit["high_profit"].cumsum()
    total_profit = high_customer_profit["high_profit"].sum()
    high_customer_profit["cum_profit_share"] = np.where(
        total_profit != 0,
        high_customer_profit["cum_profit"] / total_profit,
        0.0,
    )

    checkpoints = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    checkpoint_rows = []
    for share in checkpoints:
        idx = max(int(np.ceil(len(high_customer_profit) * share)) - 1, 0)
        if len(high_customer_profit) == 0:
            profit_share = 0.0
            customer_count = 0
        else:
            profit_share = float(high_customer_profit.iloc[idx]["cum_profit_share"])
            customer_count = int(high_customer_profit.iloc[idx]["customer_rank"])
        checkpoint_rows.append(
            {
                "customer_share_checkpoint": share,
                "customer_count": customer_count,
                "cum_profit_share": profit_share,
            }
        )

    checkpoint_df = pd.DataFrame(checkpoint_rows)
    return high_customer_profit, checkpoint_df


def build_high_customer_order_mix(
    sales_fact: pd.DataFrame,
    products: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    medium_start = products["price"].quantile(0.4)
    high_start = products["price"].quantile(0.7)

    enriched_sales = sales_fact.copy()
    enriched_sales["price_bucket"] = pd.cut(
        enriched_sales["price"],
        bins=[-np.inf, medium_start, high_start, np.inf],
        labels=["Low", "Medium", "High"],
        right=False,
    ).astype("object")
    enriched_sales["price_bucket"] = enriched_sales["price_bucket"].fillna("Low")

    customer_high_profit = (
        enriched_sales.loc[enriched_sales["price_bucket"].eq("High")]
        .groupby("customer_id", as_index=False)
        .agg(high_profit=("gross_margin", "sum"))
    )
    high_customer_ids = customer_high_profit["customer_id"].drop_duplicates()

    high_customer_orders = enriched_sales.loc[
        enriched_sales["customer_id"].isin(high_customer_ids)
    ].copy()

    order_mix_bucket = (
        high_customer_orders.groupby("price_bucket", as_index=False)
        .agg(
            order_count=("order_id", "nunique"),
            line_count=("order_id", "size"),
            customer_count=("customer_id", "nunique"),
            quantity=("quantity", "sum"),
            revenue=("net_revenue", "sum"),
            profit=("gross_margin", "sum"),
        )
    )
    bucket_order = pd.Categorical(
        order_mix_bucket["price_bucket"], categories=["Low", "Medium", "High"], ordered=True
    )
    order_mix_bucket = (
        order_mix_bucket.assign(price_bucket=bucket_order)
        .sort_values("price_bucket")
        .reset_index(drop=True)
    )

    order_mix_price = (
        high_customer_orders.groupby(["product_id", "price"], as_index=False)
        .agg(
            order_count=("order_id", "nunique"),
            line_count=("order_id", "size"),
            customer_count=("customer_id", "nunique"),
            quantity=("quantity", "sum"),
            revenue=("net_revenue", "sum"),
            profit=("gross_margin", "sum"),
        )
        .sort_values(["price", "product_id", "order_count"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    return order_mix_bucket, order_mix_price


def build_high_value_customer_summary(
    sales_fact: pd.DataFrame,
    products: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    medium_start = products["price"].quantile(0.4)
    high_start = products["price"].quantile(0.7)

    enriched_sales = sales_fact.copy()
    enriched_sales["year"] = enriched_sales["order_date"].dt.year
    enriched_sales["price_bucket"] = pd.cut(
        enriched_sales["price"],
        bins=[-np.inf, medium_start, high_start, np.inf],
        labels=["Low", "Medium", "High"],
        right=False,
    ).astype("object")
    enriched_sales["price_bucket"] = enriched_sales["price_bucket"].fillna("Low")

    customer_year_base = (
        enriched_sales.groupby(["year", "customer_id"], as_index=False)
        .agg(
            order_count=("order_id", "nunique"),
            total_revenue=("net_revenue", "sum"),
        )
    )
    customer_year_high = (
        enriched_sales.loc[enriched_sales["price_bucket"].eq("High")]
        .groupby(["year", "customer_id"], as_index=False)
        .agg(
            high_order_count=("order_id", "nunique"),
            high_revenue=("net_revenue", "sum"),
            high_profit=("gross_margin", "sum"),
        )
    )
    customer_year_buckets = (
        enriched_sales.groupby(["year", "customer_id"])["price_bucket"]
        .agg(lambda s: tuple(sorted(set(s))))
        .reset_index(name="bucket_combo")
    )

    high_customer_year = (
        customer_year_high.merge(
            customer_year_base, on=["year", "customer_id"], how="left", validate="1:1"
        )
        .merge(
            customer_year_buckets, on=["year", "customer_id"], how="left", validate="1:1"
        )
        .sort_values(["customer_id", "year"])
        .reset_index(drop=True)
    )

    def map_group(combo: tuple[str, ...]) -> str | None:
        combo_set = set(combo)
        if "High" not in combo_set:
            return None
        if combo_set == {"High"}:
            return "High only"
        if combo_set == {"High", "Medium"}:
            return "High overlap medium"
        if combo_set == {"High", "Low"}:
            return "High overlap low"
        if combo_set == {"High", "Low", "Medium"}:
            return "High + medium + low"
        return None

    high_customer_year["customer_group"] = high_customer_year["bucket_combo"].map(map_group)
    high_customer_year = high_customer_year.dropna(subset=["customer_group"]).copy()
    high_customer_year["prev_high_year"] = high_customer_year.groupby("customer_id")["year"].shift(1)
    high_customer_year["customer_status"] = np.where(
        high_customer_year["prev_high_year"].eq(high_customer_year["year"] - 1),
        "Old customer",
        "New customer",
    )
    high_customer_year["frequency"] = np.where(
        high_customer_year["order_count"] != 0,
        365.0 / high_customer_year["order_count"],
        0.0,
    )
    high_customer_year["aov"] = np.where(
        high_customer_year["high_order_count"] != 0,
        high_customer_year["high_revenue"] / high_customer_year["high_order_count"],
        0.0,
    )

    prior_active = high_customer_year[
        ["customer_id", "year", "customer_group", "high_revenue", "high_profit", "high_order_count", "order_count"]
    ].copy()
    dropped = prior_active.copy()
    dropped["year"] = dropped["year"] + 1
    active_pairs = set(zip(high_customer_year["customer_id"], high_customer_year["year"]))
    dropped = dropped.loc[
        ~dropped.apply(lambda row: (row["customer_id"], row["year"]) in active_pairs, axis=1)
    ].copy()
    dropped["customer_status"] = "Dropped customer from last year"
    dropped["high_revenue"] = 0.0
    dropped["high_profit"] = 0.0
    dropped["high_order_count"] = 0.0
    dropped["order_count"] = 0.0
    dropped["frequency"] = 0.0
    dropped["aov"] = 0.0

    combined = pd.concat(
        [
            high_customer_year[
                [
                    "year",
                    "customer_id",
                    "customer_group",
                    "customer_status",
                    "high_revenue",
                    "high_profit",
                    "high_order_count",
                    "order_count",
                    "frequency",
                    "aov",
                ]
            ],
            dropped[
                [
                    "year",
                    "customer_id",
                    "customer_group",
                    "customer_status",
                    "high_revenue",
                    "high_profit",
                    "high_order_count",
                    "order_count",
                    "frequency",
                    "aov",
                ]
            ],
        ],
        ignore_index=True,
    )
    combined = combined.sort_values(["year", "customer_group", "customer_status", "customer_id"]).reset_index(drop=True)

    summary = (
        combined.groupby(["year", "customer_group", "customer_status"], as_index=False)
        .agg(
            total_customer=("customer_id", "nunique"),
            frequency=("frequency", "mean"),
            aov=("aov", "mean"),
            high_revenue=("high_revenue", "sum"),
            high_profit=("high_profit", "sum"),
        )
    )

    summary = summary.sort_values(["customer_group", "customer_status", "year"])

    summary["prev_total_customer"] = summary.groupby(["customer_group", "customer_status"])["total_customer"].shift(1)
    summary["pct_change_total_customer"] = np.where(
        summary["prev_total_customer"].notna() & (summary["prev_total_customer"] != 0),
        (summary["total_customer"] - summary["prev_total_customer"]) / summary["prev_total_customer"],
        np.nan,
    )

    summary["prev_frequency"] = summary.groupby(["customer_group", "customer_status"])["frequency"].shift(1)
    summary["pct_change_frequency"] = np.where(
        summary["prev_frequency"].notna() & (summary["prev_frequency"] != 0),
        (summary["frequency"] - summary["prev_frequency"]) / summary["prev_frequency"],
        np.nan,
    )

    summary["prev_aov"] = summary.groupby(["customer_group", "customer_status"])["aov"].shift(1)
    summary["pct_change_aov"] = np.where(
        summary["prev_aov"].notna() & (summary["prev_aov"] != 0),
        (summary["aov"] - summary["prev_aov"]) / summary["prev_aov"],
        np.nan,
    )

    year_total = summary.groupby("year", as_index=False).agg(
        total_high_revenue=("high_revenue", "sum")
    )
    summary = summary.merge(year_total, on="year", how="left")
    summary["revenue_share"] = np.where(
        summary["total_high_revenue"] != 0,
        summary["high_revenue"] / summary["total_high_revenue"],
        0.0,
    )
    summary["prev_high_revenue"] = summary.groupby(
        ["customer_group", "customer_status"]
    )["high_revenue"].shift(1)
    summary["high_revenue_change"] = summary["high_revenue"] - summary["prev_high_revenue"]
    summary["pct_change_vs_last_year"] = np.where(
        summary["prev_high_revenue"].notna() & (summary["prev_high_revenue"] != 0),
        summary["high_revenue_change"] / summary["prev_high_revenue"],
        np.nan,
    )
    year_change = year_total.sort_values("year").copy()
    year_change["year_total_change"] = year_change["total_high_revenue"].diff()
    summary = summary.merge(year_change[["year", "year_total_change"]], on="year", how="left")
    summary["pct_contribute_to_high_change"] = np.where(
        summary["year_total_change"].notna() & (summary["year_total_change"] != 0),
        summary["high_revenue_change"] / summary["year_total_change"],
        np.nan,
    )

    group_order = {
        "High only": 0,
        "High overlap medium": 1,
        "High overlap low": 2,
        "High + medium + low": 3,
    }
    status_order = {
        "Old customer": 0,
        "New customer": 1,
        "Dropped customer from last year": 2,
    }
    summary["group_order"] = summary["customer_group"].map(group_order)
    summary["status_order"] = summary["customer_status"].map(status_order)
    summary = summary.sort_values(["year", "group_order", "status_order"]).reset_index(drop=True)
    return summary, combined


def build_gender_summary_tables(summary_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ordered_years = sorted(summary_df["year"].dropna().unique())
    metric_order = [
        ("profit", "Profit"),
        ("profit_share", "Share"),
        ("pct_change_vs_last_year", "%Change"),
        ("pct_contribute_to_year_change", "%Contribute"),
    ]
    tables: dict[str, pd.DataFrame] = {}

    for gender in ["Male", "Female"]:
        gender_df = summary_df.loc[summary_df["gender_report"].eq(gender)].copy()
        display_df = gender_df[
            [
                "year",
                "price_bucket",
                "age_group",
                "profit",
                "profit_share",
                "pct_change_vs_last_year",
                "pct_contribute_to_year_change",
                "bucket_order",
                "age_order",
            ]
        ].copy()
        display_df["Segment"] = np.where(
            display_df["age_group"].eq("All Ages"),
            display_df["price_bucket"],
            "   " + display_df["age_group"].astype(str),
        )

        row_order = (
            display_df[["price_bucket", "age_group", "Segment", "bucket_order", "age_order"]]
            .drop_duplicates()
            .sort_values(["bucket_order", "age_order"])
            .drop(columns=["bucket_order", "age_order"])
        )

        wide_parts = []
        for metric_key, metric_label in metric_order:
            metric_wide = (
                display_df.pivot_table(
                    index=["price_bucket", "age_group", "Segment"],
                    columns="year",
                    values=metric_key,
                    aggfunc="sum",
                )
                .reindex(
                    pd.MultiIndex.from_frame(row_order[["price_bucket", "age_group", "Segment"]])
                )
                .reindex(columns=ordered_years)
            )
            metric_wide.columns = [f"{year}_{metric_label}" for year in metric_wide.columns]
            wide_parts.append(metric_wide)

        gender_wide = pd.concat(wide_parts, axis=1).reset_index()
        gender_wide = gender_wide[row_order.columns.tolist() + sorted(
            [col for col in gender_wide.columns if col not in row_order.columns],
            key=lambda col: (int(col.split("_")[0]), [m[1] for m in metric_order].index(col.split("_", 1)[1])),
        )]
        gender_wide = gender_wide.drop(columns=["price_bucket", "age_group"])
        tables[gender] = gender_wide

    return tables


def build_high_value_group_summary_table(combined: pd.DataFrame) -> pd.DataFrame:
    group_summary = (
        combined.groupby(["year", "customer_group"], as_index=False)
        .agg(high_profit=("high_profit", "sum"))
    )

    year_total = group_summary.groupby("year", as_index=False).agg(
        total_high_profit=("high_profit", "sum")
    )
    group_summary = group_summary.merge(year_total, on="year", how="left")

    group_summary["profit_share"] = np.where(
        group_summary["total_high_profit"] != 0,
        group_summary["high_profit"] / group_summary["total_high_profit"],
        0.0,
    )

    group_summary["prev_high_profit"] = group_summary.groupby("customer_group")["high_profit"].shift(1)
    group_summary["high_profit_change"] = group_summary["high_profit"] - group_summary["prev_high_profit"]

    group_summary["pct_change_vs_last_year"] = np.where(
        group_summary["prev_high_profit"].notna() & (group_summary["prev_high_profit"] != 0),
        group_summary["high_profit_change"] / group_summary["prev_high_profit"],
        np.nan,
    )

    group_summary["pct_contribute_to_year_change"] = np.where(
        group_summary["total_high_profit"].notna() & (group_summary["total_high_profit"] != 0),
        group_summary["high_profit_change"] / group_summary["total_high_profit"],
        np.nan,
    )

    group_order = {
        "High only": 0,
        "High overlap medium": 1,
        "High overlap low": 2,
        "High + medium + low": 3,
    }
    group_summary["group_order"] = group_summary["customer_group"].map(group_order)
    group_summary = group_summary.sort_values(["group_order", "year"]).reset_index(drop=True)

    ordered_years = sorted(group_summary["year"].dropna().unique())
    metric_order = [
        ("high_profit", "Profit"),
        ("profit_share", "Share"),
        ("pct_change_vs_last_year", "%Change"),
        ("pct_contribute_to_year_change", "%Contribute"),
    ]

    display_df = group_summary[
        [
            "year",
            "customer_group",
            "group_order",
            "high_profit",
            "profit_share",
            "pct_change_vs_last_year",
            "pct_contribute_to_year_change",
        ]
    ].copy()
    display_df["Segment"] = display_df["customer_group"]

    row_order = (
        display_df[["customer_group", "Segment", "group_order"]]
        .drop_duplicates()
        .sort_values("group_order")
        .drop(columns=["group_order"])
    )

    wide_parts = []
    for metric_key, metric_label in metric_order:
        metric_wide = (
            display_df.pivot_table(
                index=["customer_group", "Segment"],
                columns="year",
                values=metric_key,
                aggfunc="sum",
            )
            .reindex(pd.MultiIndex.from_frame(row_order[["customer_group", "Segment"]]))
            .reindex(columns=ordered_years)
        )
        metric_wide.columns = [f"{year}_{metric_label}" for year in metric_wide.columns]
        wide_parts.append(metric_wide)

    final_wide = pd.concat(wide_parts, axis=1).reset_index()
    final_wide = final_wide[
        row_order.columns.tolist()
        + sorted(
            [col for col in final_wide.columns if col not in row_order.columns],
            key=lambda col: (
                int(col.split("_")[0]),
                [m[1] for m in metric_order].index(col.split("_", 1)[1]),
            ),
        )
    ]
    final_wide = final_wide.drop(columns=["customer_group"])
    return final_wide


def build_high_value_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    ordered_years = sorted(summary_df["year"].dropna().unique())
    metric_order = [
        ("total_customer", "Total customer"),
        ("pct_change_total_customer", "%change from last year "),
        ("frequency", "Frequency"),
        ("pct_change_frequency", "%change from last year  "),
        ("aov", "AOV"),
        ("pct_change_aov", "%change from last year   "),
    ]
    status_order = [
        "Old customer",
        "New customer",
        "Dropped customer from last year",
    ]
    group_order = [
        "High only",
        "High overlap medium",
        "High overlap low",
        "High + medium + low",
    ]

    all_rows = []
    for group in group_order:
        for metric_key, metric_label in metric_order:
            row = {"Customer group": group, "Metric": metric_label.strip()}
            for year in ordered_years:
                for status in status_order:
                    value = summary_df.loc[
                        (summary_df["year"].eq(year))
                        & (summary_df["customer_group"].eq(group))
                        & (summary_df["customer_status"].eq(status)),
                        metric_key,
                    ]
                    row[f"{year}_{status}"] = value.iloc[0] if not value.empty else np.nan
            all_rows.append(row)

    return pd.DataFrame(all_rows)


def export_gender_summary_tables(
    summary_tables: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    summary_dir = output_dir / "summary_tables"
    summary_dir.mkdir(parents=True, exist_ok=True)

    for gender, table_df in summary_tables.items():
        slug = gender.lower()
        table_df.to_csv(summary_dir / f"summary_{slug}_all_years.csv", index=False)

    try:
        with pd.ExcelWriter(summary_dir / "summary_tables_all_years.xlsx") as writer:
            for gender, table_df in summary_tables.items():
                table_df.to_excel(writer, sheet_name=gender, index=False)
    except ModuleNotFoundError:
        print("Skipping Excel export because openpyxl is not installed.")


def export_high_value_summary_table(
    summary_table: pd.DataFrame,
    summary_detail: pd.DataFrame,
    customer_detail: pd.DataFrame,
    output_dir: Path,
) -> None:
    summary_dir = output_dir / "summary_tables"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_table.to_csv(summary_dir / "high_value_customer_summary_all_years.csv", index=False)
    summary_detail.to_csv(output_dir / "high_value_customer_summary_detail.csv", index=False)
    customer_detail.to_csv(output_dir / "high_value_customer_year_status.csv", index=False)


def render_segment_summary_tables(summary_tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    if plt is None:
        return

    table_dir = output_dir / "summary_tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    format_money = lambda x: f"{x:,.0f}"
    format_pct = lambda x: "" if pd.isna(x) else f"{x:.1%}"

    for gender, table_df in summary_tables.items():
        display_df = table_df.copy()
        for column in display_df.columns:
            if column.endswith("_Profit"):
                display_df[column] = display_df[column].map(format_money)
            elif any(
                column.endswith(metric_suffix)
                for metric_suffix in ["_Share", "_%Change", "_%Contribute"]
            ):
                display_df[column] = display_df[column].map(format_pct)

        col_labels = ["Segment"] + [
            col.replace("_", "\n", 1) for col in display_df.columns if col != "Segment"
        ]
        cell_values = display_df.values

        fig_height = max(4.8, 0.42 * len(display_df) + 1.8)
        fig_width = max(18, 0.95 * len(display_df.columns))
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
        ax.axis("off")
        ax.set_title(
            f"Summary Profit Table - {gender}",
            loc="left",
            fontsize=15,
            fontweight="bold",
        )
        table = ax.table(
            cellText=cell_values,
            colLabels=col_labels,
            loc="upper left",
            cellLoc="center",
            colLoc="center",
            bbox=[0, 0, 1, 0.95],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)

        for (row, col), cell in table.get_celld().items():
            cell.set_linewidth(0.35)
            if row == 0:
                cell.set_facecolor("#DCE6F2")
                cell.set_text_props(weight="bold")
            elif display_df.iloc[row - 1, 0] in {"Low", "Medium", "High"}:
                cell.set_facecolor("#EEF3F8")
                if col == 0:
                    cell.set_text_props(weight="bold")
            elif col == 0:
                cell.set_text_props(ha="left")

        file_name = f"summary_{gender.lower()}_all_years.png"
        fig.savefig(table_dir / file_name, dpi=220, bbox_inches="tight")
        plt.close(fig)


def render_high_value_group_summary_table(table_df: pd.DataFrame, output_dir: Path) -> None:
    if plt is None:
        return

    table_dir = output_dir / "summary_tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    format_money = lambda x: f"{x:,.0f}"
    format_pct = lambda x: "" if pd.isna(x) else f"{x:.1%}"

    display_df = table_df.copy()
    for column in display_df.columns:
        if column.endswith("_Profit"):
            display_df[column] = display_df[column].map(format_money)
        elif any(
            column.endswith(metric_suffix)
            for metric_suffix in ["_Share", "_%Change", "_%Contribute"]
        ):
            display_df[column] = display_df[column].map(format_pct)

    col_labels = ["Customer Group"] + [
        col.replace("_", "\n", 1) for col in display_df.columns if col != "Segment"
    ]
    cell_values = display_df.values

    fig_height = max(4.8, 0.42 * len(display_df) + 1.8)
    fig_width = max(18, 0.95 * len(display_df.columns))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    ax.axis("off")
    ax.set_title(
        "Summary Profit Table - Customer Groups",
        loc="left",
        fontsize=15,
        fontweight="bold",
    )
    table = ax.table(
        cellText=cell_values,
        colLabels=col_labels,
        loc="upper left",
        cellLoc="center",
        colLoc="center",
        bbox=[0, 0, 1, 0.95],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.35)
        if row == 0:
            cell.set_facecolor("#DCE6F2")
            cell.set_text_props(weight="bold")
        elif display_df.iloc[row - 1, 0] in {
            "High only",
            "High overlap medium",
            "High overlap low",
            "High + medium + low",
        }:
            cell.set_facecolor("#EEF3F8")
            if col == 0:
                cell.set_text_props(weight="bold", ha="left")
        elif col == 0:
            cell.set_text_props(ha="left")

    file_name = "summary_high_value_groups_all_years.png"
    fig.savefig(table_dir / file_name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_high_value_summary_table(summary_table: pd.DataFrame, output_dir: Path) -> None:
    if plt is None:
        return

    table_dir = output_dir / "summary_tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    display_df = summary_table.copy().astype(object)
    for column in display_df.columns:
        if column in {"Customer group", "Metric"}:
            continue
        if "Total customer" in column:
            continue
    for idx, row in display_df.iterrows():
        metric = row["Metric"]
        for column in display_df.columns[2:]:
            value = display_df.at[idx, column]
            if pd.isna(value):
                display_df.at[idx, column] = ""
            elif metric == "Total customer":
                display_df.at[idx, column] = f"{value:,.0f}"
            elif metric in {"Frequency", "AOV"}:
                display_df.at[idx, column] = f"{value:,.2f}" if metric == "Frequency" else f"{value:,.0f}"
            else:
                display_df.at[idx, column] = f"{value:.1%}"

    col_labels = [col.replace("_", "\n", 1) for col in display_df.columns]
    fig_height = max(8.5, 0.34 * len(display_df) + 2.2)
    fig_width = max(22, 0.42 * len(display_df.columns) + 6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    ax.axis("off")
    ax.set_title(
        "High Value Customer Summary - All Years",
        loc="left",
        fontsize=15,
        fontweight="bold",
    )
    table = ax.table(
        cellText=display_df.values,
        colLabels=col_labels,
        loc="upper left",
        cellLoc="center",
        colLoc="center",
        bbox=[0, 0, 1, 0.96],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.0)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.35)
        if row == 0:
            cell.set_facecolor("#DCE6F2")
            cell.set_text_props(weight="bold")
        elif col == 0 and display_df.iloc[row - 1, 0] != display_df.iloc[max(row - 2, 0), 0]:
            cell.set_facecolor("#EEF3F8")
            cell.set_text_props(weight="bold", ha="left")
        elif col in {0, 1}:
            cell.set_text_props(ha="left")

    fig.savefig(table_dir / "high_value_customer_summary_all_years.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_gender_summary_dashboards(summary_df: pd.DataFrame, output_dir: Path) -> None:
    if plt is None or sns is None:
        return

    viz_dir = output_dir / "summary_tables"
    viz_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    bucket_rows = summary_df.loc[summary_df["row_type"].eq("bucket")].copy()
    detail_rows = summary_df.loc[summary_df["row_type"].eq("detail")].copy()

    for gender in ["Male", "Female"]:
        bucket_df = bucket_rows.loc[bucket_rows["gender_report"].eq(gender)].copy()
        detail_df = detail_rows.loc[detail_rows["gender_report"].eq(gender)].copy()
        years = sorted(bucket_df["year"].dropna().unique())

        fig = plt.figure(figsize=(18, 12), constrained_layout=True)
        grid = fig.add_gridspec(2, 2, height_ratios=[0.48, 0.52])
        ax_profit = fig.add_subplot(grid[0, 0])
        ax_share = fig.add_subplot(grid[0, 1])
        ax_change = fig.add_subplot(grid[1, 0])
        ax_heat = fig.add_subplot(grid[1, 1])

        profit_pivot = bucket_df.pivot(index="year", columns="price_bucket", values="profit").reindex(years)
        profit_pivot = profit_pivot[["Low", "Medium", "High"]]
        profit_pivot.plot(
            kind="bar",
            stacked=True,
            color=["#9ecae1", "#4c78a8", "#1d3557"],
            ax=ax_profit,
            width=0.75,
        )
        ax_profit.set_title("Profit by Price Bucket", loc="left", fontsize=14, fontweight="bold")
        ax_profit.set_xlabel("Year")
        ax_profit.set_ylabel("Profit")
        ax_profit.yaxis.set_major_formatter(FuncFormatter(money_formatter))
        ax_profit.legend(title="Bucket", frameon=False)

        share_pivot = bucket_df.pivot(index="year", columns="price_bucket", values="profit_share").reindex(years)
        share_pivot = share_pivot[["Low", "Medium", "High"]]
        share_pivot.plot(
            kind="area",
            stacked=True,
            color=["#9ecae1", "#4c78a8", "#1d3557"],
            alpha=0.90,
            ax=ax_share,
        )
        ax_share.set_title("Profit Share Mix", loc="left", fontsize=14, fontweight="bold")
        ax_share.set_xlabel("Year")
        ax_share.set_ylabel("Share")
        ax_share.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax_share.legend(title="Bucket", frameon=False, loc="upper right")

        change_plot_df = bucket_df.copy()
        change_plot_df["pct_change_plot"] = change_plot_df["pct_change_vs_last_year"].replace([np.inf, -np.inf], np.nan)
        sns.lineplot(
            data=change_plot_df,
            x="year",
            y="pct_change_plot",
            hue="price_bucket",
            style="price_bucket",
            markers=True,
            dashes=False,
            palette={"Low": "#9ecae1", "Medium": "#4c78a8", "High": "#1d3557"},
            ax=ax_change,
        )
        ax_change.axhline(0, color="#8c8c8c", linewidth=0.9, linestyle="--")
        ax_change.set_title("YoY Profit Change by Bucket", loc="left", fontsize=14, fontweight="bold")
        ax_change.set_xlabel("Year")
        ax_change.set_ylabel("% Change")
        ax_change.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax_change.legend(title="Bucket", frameon=False)

        heat_df = detail_df.copy()
        heat_df["segment_label"] = heat_df["price_bucket"] + " | " + heat_df["age_group"]
        heat_order = (
            heat_df[["price_bucket", "age_group", "segment_label", "bucket_order", "age_order"]]
            .drop_duplicates()
            .sort_values(["bucket_order", "age_order"])["segment_label"]
            .tolist()
        )
        heat_pivot = heat_df.pivot(
            index="segment_label",
            columns="year",
            values="pct_contribute_to_year_change",
        ).reindex(heat_order)
        sns.heatmap(
            heat_pivot,
            cmap="RdYlBu_r",
            center=0,
            linewidths=0.25,
            linecolor="white",
            cbar_kws={"format": FuncFormatter(lambda y, _: f"{y:.0%}")},
            ax=ax_heat,
        )
        ax_heat.set_title("% Contribute to Overall Year Change", loc="left", fontsize=14, fontweight="bold")
        ax_heat.set_xlabel("Year")
        ax_heat.set_ylabel("")

        fig.suptitle(f"Summary Dashboard - {gender}", x=0.02, ha="left", fontsize=18, fontweight="bold")
        fig.savefig(viz_dir / f"summary_{gender.lower()}_dashboard.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_high_value_group_dashboards(combined: pd.DataFrame, output_dir: Path) -> None:
    if plt is None or sns is None:
        return

    viz_dir = output_dir / "summary_tables"
    viz_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    group_summary = (
        combined.groupby(["year", "customer_group"], as_index=False)
        .agg(high_profit=("high_profit", "sum"))
    )
    year_total = group_summary.groupby("year", as_index=False).agg(
        total_high_profit=("high_profit", "sum")
    )
    group_summary = group_summary.merge(year_total, on="year", how="left")
    group_summary["profit_share"] = np.where(
        group_summary["total_high_profit"] != 0,
        group_summary["high_profit"] / group_summary["total_high_profit"],
        0.0,
    )
    group_summary["prev_high_profit"] = group_summary.groupby("customer_group")["high_profit"].shift(1)
    group_summary["high_profit_change"] = group_summary["high_profit"] - group_summary["prev_high_profit"]
    group_summary["pct_change_vs_last_year"] = np.where(
        group_summary["prev_high_profit"].notna() & (group_summary["prev_high_profit"] != 0),
        group_summary["high_profit_change"] / group_summary["prev_high_profit"],
        np.nan,
    )
    group_summary["pct_contribute_to_year_change"] = np.where(
        group_summary["total_high_profit"] != 0,
        group_summary["high_profit_change"] / group_summary["total_high_profit"],
        np.nan,
    )

    group_order = {
        "High only": 0,
        "High overlap medium": 1,
        "High overlap low": 2,
        "High + medium + low": 3,
    }
    group_summary["group_order"] = group_summary["customer_group"].map(group_order)
    group_summary = group_summary.sort_values(["group_order", "year"]).reset_index(drop=True)

    years = sorted(group_summary["year"].dropna().unique())
    group_labels = ["High only", "High overlap medium", "High overlap low", "High + medium + low"]

    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.48, 0.52])
    ax_profit = fig.add_subplot(grid[0, 0])
    ax_share = fig.add_subplot(grid[0, 1])
    ax_change = fig.add_subplot(grid[1, 0])
    ax_heat = fig.add_subplot(grid[1, 1])

    colors = ["#1d3557", "#457b9d", "#a8dadc", "#e63946"]

    profit_pivot = group_summary.pivot(index="year", columns="customer_group", values="high_profit").reindex(years)
    profit_pivot = profit_pivot[group_labels]
    profit_pivot.plot(
        kind="bar",
        stacked=True,
        color=colors,
        ax=ax_profit,
        width=0.75,
    )
    ax_profit.set_title("Profit by Customer Group", loc="left", fontsize=14, fontweight="bold")
    ax_profit.set_xlabel("Year")
    ax_profit.set_ylabel("Profit")
    ax_profit.yaxis.set_major_formatter(FuncFormatter(money_formatter))
    ax_profit.legend(title="Customer Group", frameon=False)

    share_pivot = group_summary.pivot(index="year", columns="customer_group", values="profit_share").reindex(years)
    share_pivot = share_pivot[group_labels]
    share_pivot.plot(
        kind="area",
        stacked=True,
        color=colors,
        alpha=0.90,
        ax=ax_share,
    )
    ax_share.set_title("Profit Share Mix", loc="left", fontsize=14, fontweight="bold")
    ax_share.set_xlabel("Year")
    ax_share.set_ylabel("Share")
    ax_share.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax_share.legend(title="Customer Group", frameon=False, loc="upper right")

    change_plot_df = group_summary.copy()
    change_plot_df["pct_change_plot"] = change_plot_df["pct_change_vs_last_year"].replace([np.inf, -np.inf], np.nan)
    sns.lineplot(
        data=change_plot_df,
        x="year",
        y="pct_change_plot",
        hue="customer_group",
        style="customer_group",
        markers=True,
        dashes=False,
        palette=dict(zip(group_labels, colors)),
        ax=ax_change,
    )
    ax_change.axhline(0, color="#8c8c8c", linewidth=0.9, linestyle="--")
    ax_change.set_title("YoY Profit Change by Group", loc="left", fontsize=14, fontweight="bold")
    ax_change.set_xlabel("Year")
    ax_change.set_ylabel("% Change")
    ax_change.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax_change.legend(title="Customer Group", frameon=False)

    heat_pivot = group_summary.pivot(
        index="customer_group",
        columns="year",
        values="pct_contribute_to_year_change",
    ).reindex(group_labels)
    sns.heatmap(
        heat_pivot,
        cmap="RdYlBu_r",
        center=0,
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"format": FuncFormatter(lambda y, _: f"{y:.0%}")},
        ax=ax_heat,
    )
    ax_heat.set_title("% Contribute to Overall Year Total", loc="left", fontsize=14, fontweight="bold")
    ax_heat.set_xlabel("Year")
    ax_heat.set_ylabel("")

    fig.suptitle("Summary Dashboard - High Value Customer Groups", x=0.02, ha="left", fontsize=18, fontweight="bold")
    fig.savefig(viz_dir / "summary_high_value_groups_dashboard.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_high_value_summary_dashboards(summary_df: pd.DataFrame, output_dir: Path) -> None:
    if plt is None or sns is None:
        return

    viz_dir = output_dir / "summary_tables"
    viz_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plot_df = summary_df.melt(
        id_vars=["year", "customer_group", "customer_status"],
        value_vars=["total_customer", "frequency", "aov"],
        var_name="metric", value_name="value"
    )

    g = sns.relplot(
        data=plot_df, x="year", y="value", 
        hue="customer_group", col="customer_status", row="metric",
        kind="line", markers=True, dashes=False,
        height=3.5, aspect=1.3, facet_kws={'sharey': 'row'}
    )
    g.fig.suptitle("High Value Customer Summary Dashboard", y=1.02, fontsize=16, fontweight="bold")
    
    for (row_val, col_val), ax in g.axes_dict.items():
        if row_val == "total_customer":
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))
        elif row_val == "frequency":
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.2f}"))
        elif row_val == "aov":
            ax.yaxis.set_major_formatter(FuncFormatter(money_formatter))

    g.savefig(viz_dir / "high_value_customer_summary_dashboard.png", dpi=220, bbox_inches="tight")
    plt.close(g.fig)


def plot_high_customer_pareto(
    high_customer_profit: pd.DataFrame,
    checkpoint_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    if plt is None or sns is None:
        return

    viz_dir = output_dir / "summary_tables"
    viz_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
    ax_curve, ax_bar = axes

    if not high_customer_profit.empty:
        curve_x = np.concatenate([[0], high_customer_profit["customer_share"].to_numpy()])
        curve_y = np.concatenate([[0], high_customer_profit["cum_profit_share"].to_numpy()])
        ax_curve.plot(
            curve_x,
            curve_y,
            color="#1d3557",
            linewidth=2.6,
            label="High Group Cumulative Profit",
        )
        ax_curve.plot([0, 1], [0, 1], color="#9aa0a6", linestyle="--", linewidth=1.2, label="Equal Distribution")
        for _, row in checkpoint_df.iterrows():
            ax_curve.scatter(
                row["customer_share_checkpoint"],
                row["cum_profit_share"],
                color="#e63946",
                s=38,
                zorder=4,
            )
            ax_curve.text(
                row["customer_share_checkpoint"],
                row["cum_profit_share"] + 0.03,
                f"{int(row['customer_share_checkpoint'] * 100)}% cust\n{row['cum_profit_share']:.0%} profit",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax_curve.set_title("Pareto / Lorenz Curve for High Customers", loc="left", fontsize=14, fontweight="bold")
    ax_curve.set_xlabel("% of High Customers")
    ax_curve.set_ylabel("% of High Group Profit")
    ax_curve.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax_curve.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax_curve.legend(frameon=False, loc="upper left")

    top_cutoff = min(30, len(high_customer_profit))
    top_customers = high_customer_profit.head(top_cutoff).copy()
    top_customers["customer_label"] = top_customers["customer_id"].astype(str)
    sns.barplot(
        data=top_customers,
        x="customer_label",
        y="high_profit",
        color="#4c78a8",
        ax=ax_bar,
    )
    ax_bar.set_title("Top High Customers by Profit", loc="left", fontsize=14, fontweight="bold")
    ax_bar.set_xlabel("Customer ID")
    ax_bar.set_ylabel("Profit")
    ax_bar.yaxis.set_major_formatter(FuncFormatter(money_formatter))
    ax_bar.tick_params(axis="x", rotation=90)

    fig.savefig(viz_dir / "high_customer_pareto_dashboard.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_high_customer_order_mix(
    high_customer_order_mix_price: pd.DataFrame,
    output_dir: Path,
) -> None:
    if plt is None or sns is None:
        return

    viz_dir = output_dir / "summary_tables"
    viz_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plot_df = high_customer_order_mix_price.copy().sort_values(
        ["price", "product_id"], ascending=[True, True]
    ).reset_index(drop=True)
    plot_df["x_pos"] = np.arange(len(plot_df))
    plot_df["product_label"] = plot_df["product_id"].astype(str)

    fig, ax1 = plt.subplots(figsize=(22, 8), constrained_layout=True)
    ax2 = ax1.twinx()

    ax1.bar(
        plot_df["x_pos"],
        plot_df["order_count"],
        color="#4c78a8",
        alpha=0.82,
        width=0.85,
        label="Order Count",
    )
    ax2.plot(
        plot_df["x_pos"],
        plot_df["price"],
        color="#e76f51",
        linewidth=1.8,
        alpha=0.9,
        label="Price",
    )

    tick_step = max(1, len(plot_df) // 35)
    tick_positions = plot_df["x_pos"].iloc[::tick_step]
    tick_labels = plot_df["product_label"].iloc[::tick_step]
    ax1.set_xticks(tick_positions, tick_labels, rotation=90)
    ax1.set_title(
        "Order Count by Product Price for High Customers",
        loc="left",
        fontsize=16,
        fontweight="bold",
    )
    ax1.set_xlabel("Product ID sorted by ascending price")
    ax1.set_ylabel("Order Count")
    ax2.set_ylabel("Price")
    ax2.yaxis.set_major_formatter(FuncFormatter(money_formatter))

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", frameon=False)

    fig.savefig(viz_dir / "high_customer_order_mix_by_price.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_high_value_customer_change(summary_df: pd.DataFrame, output_dir: Path) -> None:
    if plt is None or sns is None:
        return

    viz_dir = output_dir / "summary_tables"
    viz_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plot_df = summary_df.copy()
    plot_df["group_status"] = plot_df["customer_group"] + " | " + plot_df["customer_status"]

    fig, axes = plt.subplots(2, 1, figsize=(18, 12), constrained_layout=True)
    ax_change, ax_share = axes

    change_pivot = plot_df.pivot(
        index="year",
        columns="group_status",
        values="high_revenue_change",
    ).fillna(0)
    change_pivot.plot(kind="bar", stacked=True, ax=ax_change, width=0.82, colormap="tab20")
    ax_change.axhline(0, color="#7f7f7f", linewidth=0.9)
    ax_change.set_title("High Value Revenue Change Drivers", loc="left", fontsize=15, fontweight="bold")
    ax_change.set_xlabel("Year")
    ax_change.set_ylabel("Revenue change")
    ax_change.yaxis.set_major_formatter(FuncFormatter(money_formatter))
    ax_change.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")

    share_pivot = plot_df.pivot(
        index="year",
        columns="group_status",
        values="revenue_share",
    ).fillna(0)
    share_pivot.plot(kind="area", stacked=True, ax=ax_share, alpha=0.9, colormap="tab20")
    ax_share.set_title("High Value Revenue Mix by Group and Status", loc="left", fontsize=15, fontweight="bold")
    ax_share.set_xlabel("Year")
    ax_share.set_ylabel("Revenue share")
    ax_share.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax_share.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")

    fig.savefig(viz_dir / "high_value_customer_change_dashboard.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_financial_summary(
    financial_summary_df: pd.DataFrame,
    cash_in_schedule: pd.DataFrame,
    output_dir: Path,
) -> None:
    if plt is None or FuncFormatter is None or sns is None:
        print("Skipping chart export because matplotlib/seaborn is not installed.")
        return

    sns.set_theme(style="whitegrid")
    monthly = financial_summary_df.copy().reset_index(drop=True)
    pnl_monthly = monthly[(monthly["Net_Revenue"] != 0) | (monthly["COGS"] != 0)].copy()
    pnl_monthly = pnl_monthly.reset_index(drop=True)
    pnl_x = np.arange(len(pnl_monthly))
    pnl_year_ticks = pnl_monthly.groupby("year").head(1).index.to_numpy()
    pnl_year_labels = pnl_monthly.loc[pnl_year_ticks, "year"].astype(str).tolist()

    cash_monthly = monthly.copy()
    cash_x = np.arange(len(cash_monthly))
    cash_year_ticks = cash_monthly.groupby("year").head(1).index.to_numpy()
    cash_year_labels = cash_monthly.loc[cash_year_ticks, "year"].astype(str).tolist()

    annual_summary = build_annual_summary(financial_summary_df)
    annual_pnl = annual_summary[annual_summary["Net_Revenue"] != 0].copy().reset_index(drop=True)
    annual_cash = annual_summary.copy().reset_index(drop=True)

    def year_span_positions(frame: pd.DataFrame) -> pd.DataFrame:
        span = (
            frame.reset_index()
            .groupby("year", as_index=False)
            .agg(start_idx=("index", "min"), end_idx=("index", "max"))
        )
        span["center"] = (span["start_idx"] + span["end_idx"]) / 2
        span["width"] = (span["end_idx"] - span["start_idx"] + 1) * 0.9
        return span

    pnl_span = year_span_positions(pnl_monthly)
    cash_span = year_span_positions(cash_monthly)
    annual_pnl = annual_pnl.merge(pnl_span[["year", "center", "width"]], on="year", how="left")
    annual_cash = annual_cash.merge(cash_span[["year", "center", "width"]], on="year", how="left")

    fig, axes = plt.subplots(2, 1, figsize=(19, 11), constrained_layout=True)
    ax_pnl, ax_cash = axes

    ax_pnl.bar(
        annual_pnl["center"],
        annual_pnl["Gross_Margin"],
        width=annual_pnl["width"],
        color="#4c78a8",
        alpha=0.28,
        label="Annual Profit",
        zorder=1,
    )
    ax_pnl.plot(
        pnl_x,
        pnl_monthly["Net_Revenue"],
        color="#1f77b4",
        linewidth=2.3,
        label="Monthly Revenue",
        zorder=3,
    )
    ax_pnl.plot(
        pnl_x,
        pnl_monthly["COGS"],
        color="#ff7f0e",
        linewidth=2.1,
        label="Monthly Cost",
        zorder=3,
    )
    ax_pnl.plot(
        pnl_x,
        pnl_monthly["Gross_Margin"],
        color="#2ca02c",
        linewidth=2.4,
        label="Monthly Profit",
        zorder=4,
    )
    for tick in pnl_year_ticks[1:]:
        ax_pnl.axvline(tick - 0.5, color="#d0d7de", linewidth=0.8, alpha=0.8, zorder=0)
    ax_pnl.axhline(0, color="#8c8c8c", linewidth=0.8, linestyle="--")
    ax_pnl.set_title("P&L Proxy", loc="left", fontsize=17, fontweight="bold")
    ax_pnl.set_ylabel("Value")
    ax_pnl.set_xticks(pnl_year_ticks, pnl_year_labels)
    ax_pnl.yaxis.set_major_formatter(FuncFormatter(money_formatter))
    ax_pnl.legend(loc="upper left", frameon=False, ncol=4)

    ax_cash.bar(
        annual_cash["center"],
        annual_cash["Net_Cash_Flow"],
        width=annual_cash["width"],
        color="#2a9d8f",
        alpha=0.28,
        label="Annual Net Cash Flow",
        zorder=1,
    )
    ax_cash.plot(
        cash_x,
        cash_monthly["Cash_In_Actual"],
        color="#2a9d8f",
        linewidth=2.3,
        label="Monthly Cash In",
        zorder=3,
    )
    ax_cash.plot(
        cash_x,
        cash_monthly["Cash_Out_Inventory"],
        color="#e76f51",
        linewidth=2.1,
        label="Monthly Cash Out",
        zorder=3,
    )
    ax_cash.plot(
        cash_x,
        cash_monthly["Net_Cash_Flow"],
        color="#264653",
        linewidth=2.4,
        label="Monthly Net Cash Flow",
        zorder=4,
    )
    for tick in cash_year_ticks[1:]:
        ax_cash.axvline(tick - 0.5, color="#d0d7de", linewidth=0.8, alpha=0.8, zorder=0)
    ax_cash.axhline(0, color="#8c8c8c", linewidth=0.8, linestyle="--")
    ax_cash.set_title("Cash Flow Proxy", loc="left", fontsize=17, fontweight="bold")
    ax_cash.set_ylabel("Value")
    ax_cash.set_xlabel("Year of Order Date")
    ax_cash.set_xticks(cash_year_ticks, cash_year_labels)
    ax_cash.yaxis.set_major_formatter(FuncFormatter(money_formatter))
    ax_cash.legend(loc="upper left", frameon=False, ncol=4)

    chart_path = output_dir / "financial_summary_step1_dashboard.png"
    fig.savefig(chart_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def export_outputs(
    output_dir: Path,
    financial_summary_df: pd.DataFrame,
    sales_fact: pd.DataFrame,
    cash_in_schedule: pd.DataFrame,
    inventory_cash: pd.DataFrame,
    profit_segment_summary: pd.DataFrame,
    price_thresholds: pd.DataFrame,
    high_customer_profit: pd.DataFrame,
    high_customer_checkpoints: pd.DataFrame,
    high_customer_order_mix_bucket: pd.DataFrame,
    high_customer_order_mix_price: pd.DataFrame,
    high_value_summary_detail: pd.DataFrame,
    high_value_customer_year: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    financial_summary_df.to_csv(output_dir / "financial_summary_monthly.csv", index=False)
    build_annual_summary(financial_summary_df).to_csv(
        output_dir / "financial_summary_annual.csv", index=False
    )
    build_tableau_long_extract(financial_summary_df).to_csv(
        output_dir / "financial_summary_tableau_long.csv", index=False
    )
    profit_segment_summary.to_csv(output_dir / "profit_segment_summary.csv", index=False)
    price_thresholds.to_csv(output_dir / "price_bucket_thresholds.csv", index=False)
    high_customer_profit.to_csv(output_dir / "high_customer_profit_distribution.csv", index=False)
    high_customer_checkpoints.to_csv(output_dir / "high_customer_pareto_checkpoints.csv", index=False)
    high_customer_order_mix_bucket.to_csv(
        output_dir / "high_customer_order_mix_by_bucket.csv", index=False
    )
    high_customer_order_mix_price.to_csv(
        output_dir / "high_customer_order_mix_by_price.csv", index=False
    )
    high_value_summary_detail.to_csv(
        output_dir / "high_value_customer_summary_detail.csv", index=False
    )
    high_value_customer_year.to_csv(
        output_dir / "high_value_customer_year_status.csv", index=False
    )
    
    # Export the new table group table
    high_value_group_table = build_high_value_group_summary_table(high_value_customer_year)
    high_value_group_table.to_csv(
        output_dir / "summary_tables" / "summary_high_value_groups_all_years.csv", index=False
    )

    sales_fact[
        [
            "order_id",
            "order_date",
            "order_month",
            "product_id",
            "category",
            "segment",
            "quantity",
            "price",
            "unit_price",
            "discount_amount",
            "gross_revenue",
            "net_revenue",
            "cogs_value",
            "gross_margin",
            "discount_rate",
        ]
    ].to_csv(output_dir / "sales_fact_step1.csv", index=False)

    cash_in_schedule.to_csv(output_dir / "cash_in_schedule_monthly.csv", index=False)

    inventory_cash[
        [
            "snapshot_date",
            "month_period",
            "product_id",
            "category",
            "segment",
            "units_received",
            "cogs",
            "Cash_Out_Inventory",
            "stockout_days",
            "overstock_flag",
        ]
    ].to_csv(output_dir / "inventory_cash_out_detail.csv", index=False)


def main() -> None:
    data = load_data(BASE_DIR)
    sales_fact = build_sales_fact(data["orders"], data["order_items"], data["products"])
    profit_segment_summary, price_thresholds = build_profit_segment_summary(
        sales_fact, data["customers"], data["products"]
    )
    high_customer_profit, high_customer_checkpoints = build_high_customer_pareto(
        sales_fact, data["customers"], data["products"]
    )
    high_customer_order_mix_bucket, high_customer_order_mix_price = build_high_customer_order_mix(
        sales_fact, data["products"]
    )
    high_value_summary_detail, high_value_customer_year = build_high_value_customer_summary(
        sales_fact, data["products"]
    )
    monthly_pnl = build_monthly_pnl(sales_fact)
    cash_in_schedule = build_cash_in_schedule(data["orders"], data["payments"])
    inventory_cash = build_inventory_cash_out(data["inventory"], data["products"])
    monthly_cash_out = build_monthly_cash_out(inventory_cash)
    financial_summary_df = build_financial_summary(
        monthly_pnl=monthly_pnl,
        cash_in_schedule=cash_in_schedule,
        monthly_cash_out=monthly_cash_out,
    )
    summary_tables = build_gender_summary_tables(profit_segment_summary)
    high_value_summary_table = build_high_value_summary_table(high_value_summary_detail)
    high_value_group_table = build_high_value_group_summary_table(high_value_customer_year)

    export_outputs(
        output_dir=OUTPUT_DIR,
        financial_summary_df=financial_summary_df,
        sales_fact=sales_fact,
        cash_in_schedule=cash_in_schedule,
        inventory_cash=inventory_cash,
        profit_segment_summary=profit_segment_summary,
        price_thresholds=price_thresholds,
        high_customer_profit=high_customer_profit,
        high_customer_checkpoints=high_customer_checkpoints,
        high_customer_order_mix_bucket=high_customer_order_mix_bucket,
        high_customer_order_mix_price=high_customer_order_mix_price,
        high_value_summary_detail=high_value_summary_detail,
        high_value_customer_year=high_value_customer_year,
    )
    export_gender_summary_tables(summary_tables, OUTPUT_DIR)
    export_high_value_summary_table(
        high_value_summary_table,
        high_value_summary_detail,
        high_value_customer_year,
        OUTPUT_DIR,
    )
    render_segment_summary_tables(summary_tables, OUTPUT_DIR)
    render_high_value_summary_table(high_value_summary_table, OUTPUT_DIR)
    render_high_value_group_summary_table(high_value_group_table, OUTPUT_DIR)
    plot_gender_summary_dashboards(profit_segment_summary, OUTPUT_DIR)
    plot_high_value_group_dashboards(high_value_customer_year, OUTPUT_DIR)
    plot_high_value_summary_dashboards(high_value_summary_detail, OUTPUT_DIR)
    plot_high_customer_pareto(high_customer_profit, high_customer_checkpoints, OUTPUT_DIR)
    plot_high_customer_order_mix(high_customer_order_mix_price, OUTPUT_DIR)
    plot_high_value_customer_change(high_value_summary_detail, OUTPUT_DIR)
    plot_financial_summary(financial_summary_df, cash_in_schedule, OUTPUT_DIR)

    pd.set_option("display.float_format", lambda value: f"{value:,.2f}")
    print("financial_summary_df preview:")
    print(financial_summary_df.head(12).to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_DIR}")



# ========================================# FROM analysis_traffic.py# ========================================


BASE_DIR = Path("d:/DATATHON/datathon-2026-round-1")
OUTPUT_DIR = Path("d:/DATATHON/outputs/summary_table")

print("Loading data...")
data = load_data(BASE_DIR)
products = data["products"]
orders = data["orders"].copy()

print("Calculating New High-Price Customer Orders...")
sales_fact = build_sales_fact(orders, data["order_items"], products)
_, combined = build_high_value_customer_summary(sales_fact, products)

# Filter to New Customers
new_customers = combined[combined["customer_status"] == "New customer"].copy()
# We want their orders in the year they are new
new_customers["order_year"] = new_customers["year"]

# Add year to orders
orders["year"] = pd.to_datetime(orders["order_date"], errors="coerce").dt.year

# Join with orders to get their orders
new_cust_orders = orders.merge(
    new_customers[["customer_id", "order_year"]],
    left_on=["customer_id", "year"],
    right_on=["customer_id", "order_year"],
    how="inner"
)

# Aggregate orders by source and year
order_summary = new_cust_orders.groupby(["order_source", "year"]).agg(
    total_orders=("order_id", "count")
).reset_index()

print("Processing traffic data...")
traffic = pd.read_csv(BASE_DIR / "web_traffic.csv")
traffic["date"] = pd.to_datetime(traffic["date"], errors="coerce")
traffic["year"] = traffic["date"].dt.year

# Weighted average for bounce rate: sum(bounce_rate * sessions) / sum(sessions)
traffic["bounce_sessions"] = traffic["bounce_rate"] * traffic["sessions"]

traffic_summary = traffic.groupby(["traffic_source", "year"]).agg(
    total_sessions=("sessions", "sum"),
    total_bounce_sessions=("bounce_sessions", "sum")
).reset_index()

traffic_summary["avg_bounce_rate"] = traffic_summary["total_bounce_sessions"] / traffic_summary["total_sessions"]

print("Merging to calculate conversion rate...")
summary_df = traffic_summary.merge(
    order_summary,
    left_on=["traffic_source", "year"],
    right_on=["order_source", "year"],
    how="left"
)
summary_df["total_orders"] = summary_df["total_orders"].fillna(0)
summary_df["conversion_rate"] = summary_df["total_orders"] / summary_df["total_sessions"]

# Select final columns
final_df = summary_df[["year", "traffic_source", "total_sessions", "total_orders", "conversion_rate", "avg_bounce_rate"]]
final_df = final_df.sort_values(["traffic_source", "year"]).reset_index(drop=True)

final_df.to_csv(OUTPUT_DIR / "traffic_conversion_summary.csv", index=False)
print("Saved to traffic_conversion_summary.csv")
print(final_df.head(10))

# ========================================# FROM analysis_customer_factors.py# ========================================


BASE_DIR = Path("d:/DATATHON/datathon-2026-round-1")
OUTPUT_DIR = Path("d:/DATATHON/outputs/summary_table")

print("Loading data using ..")
data = load_data(BASE_DIR)
orders = data["orders"].copy()
order_items = data["order_items"].copy()
products = data["products"].copy()

# Need to load the extra files not loaded by Workflow_1
print("Loading extra files...")
returns = pd.read_csv(BASE_DIR / "returns.csv", low_memory=False)
reviews = pd.read_csv(BASE_DIR / "reviews.csv", low_memory=False)
shipments = pd.read_csv(BASE_DIR / "shipments.csv", low_memory=False)

orders["year"] = orders["order_date"].dt.year
shipments["ship_date"] = pd.to_datetime(shipments["ship_date"], errors="coerce")

print("Processing order level metrics...")
# 1. Return rate
returned_orders = set(returns["order_id"])
orders["is_returned"] = orders["order_id"].isin(returned_orders)

# 2. Promo rate
# Assuming a promo exists if promo_id is not null or not empty
if "promo_id" in order_items.columns:
    promo_orders = order_items.dropna(subset=["promo_id"])
    promo_order_ids = set(promo_orders["order_id"])
else:
    promo_order_ids = set()
orders["has_promo"] = orders["order_id"].isin(promo_order_ids)

# 3. Shipping time
orders = orders.merge(shipments[["order_id", "ship_date"]], on="order_id", how="left")
orders["shipping_time"] = (orders["ship_date"] - orders["order_date"]).dt.days

# 4. Reviews
order_reviews = reviews.groupby("order_id")["rating"].mean().reset_index()
orders = orders.merge(order_reviews, on="order_id", how="left")

print("Aggregating to customer-year level...")
cust_year_metrics = orders.groupby(["customer_id", "year"]).agg(
    total_orders=("order_id", "count"),
    returned_orders=("is_returned", "sum"),
    promo_orders=("has_promo", "sum"),
    avg_shipping_time=("shipping_time", "mean"),
    avg_rating=("rating", "mean")
).reset_index()

cust_year_metrics["return_rate"] = cust_year_metrics["returned_orders"] / cust_year_metrics["total_orders"]
cust_year_metrics["promo_rate"] = cust_year_metrics["promo_orders"] / cust_year_metrics["total_orders"]

print("Loading customer status from ..")
sales_fact = build_sales_fact(data["orders"], data["order_items"], products)
_, combined = build_high_value_customer_summary(sales_fact, products)

print("Merging metrics with status...")
combined_metrics = combined[["year", "customer_id", "customer_status"]].copy()

# For New and Old, metric_year = year
# For Dropped, metric_year = year - 1
combined_metrics["metric_year"] = np.where(
    combined_metrics["customer_status"] == "Dropped customer from last year",
    combined_metrics["year"] - 1,
    combined_metrics["year"]
)

final_df = combined_metrics.merge(
    cust_year_metrics, 
    left_on=["customer_id", "metric_year"], 
    right_on=["customer_id", "year"], 
    how="inner"
)

summary_table = final_df.groupby("customer_status").agg(
    Return_Rate=("return_rate", "mean"),
    Avg_Review=("avg_rating", "mean"),
    Avg_Shipping_Time=("avg_shipping_time", "mean"),
    Promo_Rate=("promo_rate", "mean")
).reset_index()

summary_table.to_csv(OUTPUT_DIR / "customer_status_factors.csv", index=False)
print("\n--- RESULT ---")
print(summary_table.to_string())


# Custom color palette defined in chart_rules.md
COLORS = {
    "light_green": "#85ff5f",
    "black": "#000000",
    "dark_green": "#5cd628ff",
    "gray": "#878787",
    "light_yellow": "#ffef91",
    "orange": "#ffa02e",
    "dark_yellow": "#ffc81e",
    "blue": "#73a5ca"
}

def money_formatter(value: float, _: int) -> str:
    """Format money values to standard K or M representations."""
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"

def redraw_pnl_proxy_chart():
    """
    DRAW CHART: P&L PROXY
    - Displays Annual Gross Margin as a grouped bar.
    - Displays Monthly Gross Revenue, Monthly COGS, and Monthly Profit as lines.
    - Obeys visualization requirements from `chart_rules.md`:
      + Colors used from palette.
      + Min size 1000x1000px (16x12 inches at 100 DPI = 1600x1200 pixels).
      + No grid lines.
      + Font: Arial, 30px (Title 36, Axis Label 28, Numbers 32).
      + Number fonts > Axis Label fonts.
      + X-axis labels centered below columns.
    """
    print("Loading data for P&L Proxy chart...")
    data = load_data(BASE_DIR)
    
    # Calculate P&L proxy metrics
    sales_fact = build_sales_fact(data["orders"], data["order_items"], data["products"])
    monthly_pnl = build_monthly_pnl(sales_fact)
    cash_in_schedule = build_cash_in_schedule(data["orders"], data["payments"])
    inventory_cash = build_inventory_cash_out(data["inventory"], data["products"])
    monthly_cash_out = build_monthly_cash_out(inventory_cash)
    
    financial_summary_df = build_financial_summary(
        monthly_pnl=monthly_pnl,
        cash_in_schedule=cash_in_schedule,
        monthly_cash_out=monthly_cash_out,
    )
    
    # Filter monthly data
    monthly = financial_summary_df.copy().reset_index(drop=True)
    pnl_monthly = monthly[(monthly["Net_Revenue"] != 0) | (monthly["COGS"] != 0)].copy()
    pnl_monthly = pnl_monthly.reset_index(drop=True)
    
    # Filter annual data
    annual_summary = build_annual_summary(financial_summary_df)
    annual_pnl = annual_summary[annual_summary["Net_Revenue"] != 0].copy().reset_index(drop=True)
    
    # Export summary table for P&L Proxy
    table_name = "pnl_proxy_summary"
    summary_export = pnl_monthly[["year", "month_label", "Gross_Revenue", "Net_Revenue", "COGS", "Gross_Margin"]].copy()
    summary_export.to_csv(SUMMARY_DIR / f"{table_name}.csv", index=False)
    
    # ==========================
    # CHART PLOTTING SETUP
    # ==========================
    plt.rcParams['font.family'] = 'Arial'
    
    # Size 24x12 inches at 100 DPI = 2400x1200 pixels (wider to ensure columns > text size)
    fig, ax = plt.subplots(figsize=(24, 12), dpi=100)
    ax.grid(False) # Remove gridlines
    
    # Setup data coordinates for plotting based on absolute month index
    # This ensures equal distance between years even if some months are missing
    min_year = pnl_monthly["year"].min()
    pnl_x = (pnl_monthly["year"] - min_year) * 12 + (pnl_monthly["month"] - 1)
    
    def get_annual_positions(annual_df: pd.DataFrame) -> pd.DataFrame:
        span = annual_df.copy()
        # Center of the year is between month index 5 (June) and 6 (July), which is 5.5
        span["center"] = (span["year"] - min_year) * 12 + 5.5
        # Fixed width for all columns
        span["width"] = 10.5
        return span

    annual_pnl = get_annual_positions(annual_pnl)
    
    
    # 1. Bar Chart: Annual Profit
    # Độ rộng các column bằng nhau (width is fixed to 10.5)
    bars = ax.bar(
        annual_pnl["center"],
        annual_pnl["Gross_Margin"],
        width=annual_pnl["width"],
        color=COLORS["dark_green"], # Màu xanh lá theo yêu cầu
        alpha=0.8,
        label="Annual Profit",
        zorder=1,
    )
    
    # 2. Line Chart: Monthly Revenue
    ax.plot(
        pnl_x,
        pnl_monthly["Net_Revenue"],
        color=COLORS["blue"],
        linewidth=4,
        label="Monthly Revenue",
        zorder=3,
        marker='o'
    )
    
    # 3. Line Chart: Monthly Cost
    ax.plot(
        pnl_x,
        pnl_monthly["COGS"],
        color=COLORS["orange"],
        linewidth=4,
        label="Monthly Cost",
        zorder=3,
        marker='o'
    )
    
    # 4. Line Chart: Monthly Profit
    ax.plot(
        pnl_x,
        pnl_monthly["Gross_Margin"],
        color=COLORS["dark_yellow"],
        linewidth=4,
        label="Monthly Profit",
        zorder=4,
        marker='o'
    )
    
    # X-axis base line
    ax.axhline(0, color=COLORS["black"], linewidth=1.5, linestyle="--")
    
    # Text rules: Cỡ chữ 30px, Số > Text
    ax.set_title("P&L Proxy Performance", loc="center", fontsize=36, fontweight="bold", color=COLORS["black"], pad=20)
    ax.set_ylabel("Monetary Value", fontsize=28, color=COLORS["black"])
    
    # X Axis Setup
    # "Label của mỗi column cần nằm ở giữa column đó" -> Using annual_pnl["center"]
    ax.set_xticks(annual_pnl["center"])
    ax.set_xticklabels(annual_pnl["year"].astype(str), fontsize=32, color=COLORS["black"])
    
    # Y Axis Setup
    ax.tick_params(axis='y', labelsize=32, labelcolor=COLORS["black"])
    ax.yaxis.set_major_formatter(FuncFormatter(money_formatter))
    
    # Legend without overlap
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), frameon=False, ncol=4, fontsize=24)
    
    # Bar annotations
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                money_formatter(height, 0),
                ha='center', va='bottom', fontsize=26, color=COLORS["black"], fontweight='bold')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(COLORS["black"])
    ax.spines['left'].set_color(COLORS["black"])
    
    # Save the chart and close
    fig.savefig(CHART_DIR / f"{table_name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Exported chart and summary table for {table_name}")

def redraw_cash_flow_proxy_chart():
    """
    DRAW CHART: CASH FLOW PROXY
    - Displays Annual Net Cash Flow as a grouped bar.
    - Displays Monthly Cash In, Monthly Cash Out, and Monthly Net Cash Flow as lines.
    - Obeys visualization requirements from `chart_rules.md`.
    """
    print("Loading data for Cash Flow Proxy chart...")
    data = load_data(BASE_DIR)
    
    # Calculate Cash Flow proxy metrics
    sales_fact = build_sales_fact(data["orders"], data["order_items"], data["products"])
    monthly_pnl = build_monthly_pnl(sales_fact)
    cash_in_schedule = build_cash_in_schedule(data["orders"], data["payments"])
    inventory_cash = build_inventory_cash_out(data["inventory"], data["products"])
    monthly_cash_out = build_monthly_cash_out(inventory_cash)
    
    financial_summary_df = build_financial_summary(
        monthly_pnl=monthly_pnl,
        cash_in_schedule=cash_in_schedule,
        monthly_cash_out=monthly_cash_out,
    )
    
    # Filter monthly data
    cash_monthly = financial_summary_df.copy().reset_index(drop=True)
    
    # Filter annual data
    annual_summary = build_annual_summary(financial_summary_df)
    annual_cash = annual_summary.copy().reset_index(drop=True)
    
    # Export summary table for Cash Flow Proxy
    table_name = "cash_flow_proxy_summary"
    summary_export = cash_monthly[["year", "month_label", "Cash_In_Actual", "Cash_Out_Inventory", "Net_Cash_Flow"]].copy()
    summary_export.to_csv(SUMMARY_DIR / f"{table_name}.csv", index=False)
    
    # ==========================
    # CHART PLOTTING SETUP
    # ==========================
    plt.rcParams['font.family'] = 'Arial'
    
    # Size 24x12 inches at 100 DPI = 2400x1200 pixels
    fig, ax = plt.subplots(figsize=(24, 12), dpi=100)
    ax.grid(False) # Remove gridlines
    
    # Setup data coordinates for plotting based on absolute month index
    min_year = cash_monthly["year"].min()
    cash_x = (cash_monthly["year"] - min_year) * 12 + (cash_monthly["month"] - 1)
    
    def get_annual_positions(annual_df: pd.DataFrame) -> pd.DataFrame:
        span = annual_df.copy()
        span["center"] = (span["year"] - min_year) * 12 + 5.5
        span["width"] = 10.5
        return span

    annual_cash = get_annual_positions(annual_cash)
    
    # 1. Bar Chart: Annual Net Cash Flow
    bars = ax.bar(
        annual_cash["center"],
        annual_cash["Net_Cash_Flow"],
        width=annual_cash["width"],
        color=COLORS["dark_green"],
        alpha=0.8,
        label="Annual Net Cash Flow",
        zorder=1,
    )
    
    # 2. Line Chart: Monthly Cash In
    ax.plot(
        cash_x,
        cash_monthly["Cash_In_Actual"],
        color=COLORS["blue"],
        linewidth=4,
        label="Monthly Cash In",
        zorder=3,
        marker='o'
    )
    
    # 3. Line Chart: Monthly Cash Out
    ax.plot(
        cash_x,
        cash_monthly["Cash_Out_Inventory"],
        color=COLORS["orange"],
        linewidth=4,
        label="Monthly Cash Out",
        zorder=3,
        marker='o'
    )
    
    # 4. Line Chart: Monthly Net Cash Flow
    ax.plot(
        cash_x,
        cash_monthly["Net_Cash_Flow"],
        color=COLORS["dark_yellow"],
        linewidth=4,
        label="Monthly Net Cash Flow",
        zorder=4,
        marker='o'
    )
    
    # X-axis base line
    ax.axhline(0, color=COLORS["black"], linewidth=1.5, linestyle="--")
    
    # Text rules
    ax.set_title("Cash Flow Proxy Performance", loc="center", fontsize=36, fontweight="bold", color=COLORS["black"], pad=20)
    ax.set_ylabel("Monetary Value", fontsize=28, color=COLORS["black"])
    
    # X Axis Setup
    ax.set_xticks(annual_cash["center"])
    ax.set_xticklabels(annual_cash["year"].astype(str), fontsize=32, color=COLORS["black"])
    
    # Y Axis Setup
    ax.tick_params(axis='y', labelsize=32, labelcolor=COLORS["black"])
    ax.yaxis.set_major_formatter(FuncFormatter(money_formatter))
    
    # Legend
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), frameon=False, ncol=4, fontsize=24)
    
    # Bar annotations
    for bar in bars:
        height = bar.get_height()
        # Offset for negative heights
        y_offset = height if height >= 0 else height - abs(height)*0.05
        va = 'bottom' if height >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width() / 2, y_offset,
                money_formatter(height, 0),
                ha='center', va=va, fontsize=26, color=COLORS["black"], fontweight='bold')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(COLORS["black"])
    ax.spines['left'].set_color(COLORS["black"])
    
    # Save the chart
    fig.savefig(CHART_DIR / f"{table_name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Exported chart and summary table for {table_name}")

def redraw_price_bucket_charts():
    """
    DRAW 4 CHARTS: PRICE BUCKET CONTRIBUTION
    1. Profit by Price Bucket (Stacked Bar Chart with % share text)
    2. Profit Share Mix (Stacked Area Chart)
    3. YoY Profit Change by Bucket (Line Chart)
    4. % Contribute to Overall Year Change (Heatmap)
    - Aggregated for all customers (no gender split).
    - No age split.
    """
    print("Loading data for Price Bucket charts...")
    data = load_data(BASE_DIR)
    sales_fact = build_sales_fact(data["orders"], data["order_items"], data["products"])
    
    enriched_sales = sales_fact.copy()
    enriched_sales["year"] = enriched_sales["order_date"].dt.year
    
    products = data["products"]
    medium_start = products["price"].quantile(0.4)
    high_start = products["price"].quantile(0.7)
    
    enriched_sales["price_bucket"] = pd.cut(
        enriched_sales["price"],
        bins=[-np.inf, medium_start, high_start, np.inf],
        labels=["Low", "Medium", "High"],
        right=False,
    ).fillna("Low")
    
    # Aggregate by year and price_bucket
    bucket_summary = enriched_sales.groupby(["year", "price_bucket"], as_index=False).agg(
        profit=("gross_margin", "sum")
    )
    
    # Filter out empty years
    years_with_data = bucket_summary.groupby("year")["profit"].sum()
    valid_years = years_with_data[years_with_data > 0].index
    bucket_summary = bucket_summary[bucket_summary["year"].isin(valid_years)].copy()
    
    # Calculate total profit per year
    total_profit = bucket_summary.groupby("year", as_index=False).agg(total_profit=("profit", "sum"))
    bucket_summary = bucket_summary.merge(total_profit, on="year", how="left")
    
    bucket_summary = bucket_summary.sort_values(["price_bucket", "year"])
    bucket_summary["prev_profit"] = bucket_summary.groupby("price_bucket")["profit"].shift(1)
    bucket_summary["profit_change"] = bucket_summary["profit"] - bucket_summary["prev_profit"]
    
    bucket_summary["profit_share"] = np.where(
        bucket_summary["total_profit"] != 0, 
        bucket_summary["profit"] / bucket_summary["total_profit"], 
        0.0
    )
    
    bucket_summary["pct_change_vs_last_year"] = np.where(
        bucket_summary["prev_profit"].notna() & (bucket_summary["prev_profit"] != 0),
        bucket_summary["profit_change"] / bucket_summary["prev_profit"],
        np.nan
    )
    
    bucket_summary["pct_contribute_to_year_change"] = np.where(
        bucket_summary["total_profit"] != 0,
        bucket_summary["profit_change"] / bucket_summary["total_profit"],
        np.nan
    )
    
    # Export summary table
    table_name = "price_bucket_summary"
    bucket_summary.to_csv(SUMMARY_DIR / f"{table_name}.csv", index=False)
    
    # Common Plotting Setup
    plt.rcParams['font.family'] = 'Arial'
    years = sorted(bucket_summary["year"].unique())
    x_pos = np.arange(len(years))
    
    bucket_labels = ["Low", "Medium", "High"]
    bucket_colors = [COLORS["blue"], COLORS["orange"], COLORS["dark_green"]]
    
    # Prepare pivot tables
    profit_pivot = bucket_summary.pivot(index="year", columns="price_bucket", values="profit").reindex(years)[bucket_labels]
    share_pivot = bucket_summary.pivot(index="year", columns="price_bucket", values="profit_share").reindex(years)[bucket_labels]
    change_pivot = bucket_summary.pivot(index="year", columns="price_bucket", values="pct_change_vs_last_year").reindex(years)[bucket_labels]
    contrib_pivot = bucket_summary.pivot(index="price_bucket", columns="year", values="pct_contribute_to_year_change").reindex(bucket_labels)[years]
    
    # Calculate overall change for the baseline
    total_profit_sorted = total_profit.sort_values("year").copy()
    total_profit_sorted["prev_total"] = total_profit_sorted["total_profit"].shift(1)
    total_profit_sorted["pct_change_overall"] = np.where(
        total_profit_sorted["prev_total"].notna() & (total_profit_sorted["prev_total"] != 0),
        (total_profit_sorted["total_profit"] - total_profit_sorted["prev_total"]) / total_profit_sorted["prev_total"],
        np.nan
    )
    overall_change = total_profit_sorted.set_index("year")["pct_change_overall"].reindex(years).replace([np.inf, -np.inf], np.nan).values
    
    # ==========================================
    # CHART 1: Profit by Price Bucket (Stacked Bar)
    # ==========================================
    fig, ax = plt.subplots(figsize=(24, 12), dpi=100)
    ax.grid(False)
    
    bottoms = np.zeros(len(years))
    for i, bucket in enumerate(bucket_labels):
        values = profit_pivot[bucket].fillna(0).values
        bars = ax.bar(x_pos, values, bottom=bottoms, width=0.7, color=bucket_colors[i], label=bucket)
        
        # Add percentage text in the middle of each stacked bar
        shares = share_pivot[bucket].fillna(0).values
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                y_center = bottoms[j] + height / 2
                share_pct = shares[j]
                if share_pct > 0.05: # Only show text if share > 5% to avoid clutter
                    ax.text(bar.get_x() + bar.get_width() / 2, y_center,
                            f"{share_pct:.1%}",
                            ha='center', va='center', fontsize=24, color=COLORS["black"], fontweight='bold')
        bottoms += values
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(COLORS["black"])
    ax.spines['left'].set_color(COLORS["black"])
    ax.set_title("Profit by Price Bucket", loc="center", fontsize=36, fontweight="bold", color=COLORS["black"], pad=20)
    ax.set_ylabel("Profit", fontsize=28, color=COLORS["black"])
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(y) for y in years], fontsize=32, color=COLORS["black"])
    ax.tick_params(axis='y', labelsize=32, labelcolor=COLORS["black"])
    ax.yaxis.set_major_formatter(FuncFormatter(money_formatter))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), frameon=False, ncol=3, fontsize=24)
    fig.savefig(CHART_DIR / "price_bucket_profit.png", bbox_inches="tight")
    plt.close(fig)
    
    # ==========================================
    # CHART 2: Profit Share Mix (Stacked Area)
    # ==========================================
    fig, ax = plt.subplots(figsize=(24, 12), dpi=100)
    ax.grid(False)
    
    ax.stackplot(x_pos, share_pivot["Low"], share_pivot["Medium"], share_pivot["High"],
                 labels=bucket_labels, colors=bucket_colors, alpha=0.9)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(COLORS["black"])
    ax.spines['left'].set_color(COLORS["black"])
    ax.set_title("Profit Share Mix by Price Bucket", loc="center", fontsize=36, fontweight="bold", color=COLORS["black"], pad=20)
    ax.set_ylabel("Share", fontsize=28, color=COLORS["black"])
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(y) for y in years], fontsize=32, color=COLORS["black"])
    ax.tick_params(axis='y', labelsize=32, labelcolor=COLORS["black"])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), frameon=False, ncol=3, fontsize=24)
    fig.savefig(CHART_DIR / "price_bucket_share.png", bbox_inches="tight")
    plt.close(fig)
    
    # ==========================================
    # CHART 3: YoY Profit Change by Bucket (Line)
    # ==========================================
    fig, ax = plt.subplots(figsize=(24, 12), dpi=100)
    ax.grid(False)
    
    for i, bucket in enumerate(bucket_labels):
        values = change_pivot[bucket].replace([np.inf, -np.inf], np.nan).values
        ax.plot(x_pos, values, color=bucket_colors[i], linewidth=4, marker='o', markersize=10, label=bucket)
        
    ax.plot(x_pos, overall_change, color=COLORS["black"], linewidth=4, linestyle="--", marker='o', markersize=10, label="Overall Change")
        
    ax.axhline(0, color=COLORS["gray"], linewidth=2, linestyle="--")
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(COLORS["black"])
    ax.spines['left'].set_color(COLORS["black"])
    ax.set_title("YoY Profit Change by Bucket", loc="center", fontsize=36, fontweight="bold", color=COLORS["black"], pad=20)
    ax.set_ylabel("% Change", fontsize=28, color=COLORS["black"])
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(y) for y in years], fontsize=32, color=COLORS["black"])
    ax.tick_params(axis='y', labelsize=32, labelcolor=COLORS["black"])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), frameon=False, ncol=4, fontsize=24)
    fig.savefig(CHART_DIR / "price_bucket_yoy_change.png", bbox_inches="tight")
    plt.close(fig)
    
    # ==========================================
    # CHART 4: % Contribute to Overall Year Change (Heatmap)
    # ==========================================
    fig, ax = plt.subplots(figsize=(24, 12), dpi=100)
    ax.grid(False)
    
    import seaborn as sns
    sns.heatmap(
        contrib_pivot,
        cmap="RdYlBu_r",
        center=0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"format": FuncFormatter(lambda y, _: f"{y:.0%}")},
        ax=ax,
        annot=True,
        fmt=".1%",
        annot_kws={"size": 24, "weight": "bold"}
    )
    
    # Heatmap specific customization
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=26)
    
    ax.set_title("% Contribute to Overall Year Change", loc="center", fontsize=36, fontweight="bold", color=COLORS["black"], pad=20)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks(x_pos + 0.5)
    ax.set_xticklabels([str(y) for y in years], fontsize=32, color=COLORS["black"])
    ax.set_yticklabels(bucket_labels, fontsize=32, color=COLORS["black"], rotation=0)
    
    fig.savefig(CHART_DIR / "price_bucket_contribute.png", bbox_inches="tight")
    plt.close(fig)
    print("Exported 4 charts for Price Bucket Contribution.")

def redraw_product_price_histogram():
    """
    DRAW CHART: PRODUCT PRICE HISTOGRAM
    - Displays distribution of product prices.
    - Obeys visualization requirements from `chart_rules.md`.
    """
    print("Loading data for Product Price Histogram...")
    data = load_data(BASE_DIR)
    products = data["products"]
    
    # Calculate bins/counts for summary
    counts, bins = np.histogram(products["price"].dropna(), bins=30)
    summary_df = pd.DataFrame({
        "bin_start": bins[:-1],
        "bin_end": bins[1:],
        "product_count": counts
    })
    table_name = "product_price_histogram"
    summary_df.to_csv(SUMMARY_DIR / f"{table_name}.csv", index=False)
    
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize=(24, 12), dpi=100)
    ax.grid(False)
    
    ax.hist(products["price"].dropna(), bins=30, color=COLORS["blue"], edgecolor='white', alpha=0.8)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(COLORS["black"])
    ax.spines['left'].set_color(COLORS["black"])
    
    ax.set_title("Distribution of Product Prices", loc="center", fontsize=36, fontweight="bold", color=COLORS["black"], pad=20)
    ax.set_xlabel("Price", fontsize=28, color=COLORS["black"])
    ax.set_ylabel("Number of Products", fontsize=28, color=COLORS["black"])
    
    ax.tick_params(axis='both', labelsize=32, labelcolor=COLORS["black"])
    ax.xaxis.set_major_formatter(FuncFormatter(money_formatter))
    
    fig.savefig(CHART_DIR / f"{table_name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Exported chart and summary table for {table_name}")

def redraw_high_customer_scatter():
    """
    DRAW CHART: ORDER COUNT BY PRODUCT PRICE FOR HIGH-PRICE-PRODUCT CONSUMERS
    - Scatter chart: X = price, Y = order_count
    - Obeys visualization requirements from `chart_rules.md`.
    """
    print("Loading data for High Customer Scatter...")
    data = load_data(BASE_DIR)
    sales_fact = build_sales_fact(data["orders"], data["order_items"], data["products"])
    
    _, order_mix_price = build_high_customer_order_mix(sales_fact, data["products"])
    
    # Filter to only focus on high price products
    high_start = data["products"]["price"].quantile(0.7)
    order_mix_price = order_mix_price[order_mix_price["price"] >= high_start].copy()
    
    table_name = "order_count_by_product_price"
    order_mix_price.to_csv(SUMMARY_DIR / f"{table_name}.csv", index=False)
    
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize=(24, 12), dpi=100)
    ax.grid(False)
    
    ax.scatter(order_mix_price["price"], order_mix_price["order_count"], 
               color=COLORS["orange"], s=100, alpha=0.7, edgecolor='white')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(COLORS["black"])
    ax.spines['left'].set_color(COLORS["black"])
    
    # Update title as requested
    ax.set_title("Order Count by Product Price for High-Price-Product Consumers", loc="center", fontsize=36, fontweight="bold", color=COLORS["black"], pad=20)
    ax.set_xlabel("Price", fontsize=28, color=COLORS["black"])
    ax.set_ylabel("Order Count", fontsize=28, color=COLORS["black"])
    
    ax.tick_params(axis='both', labelsize=32, labelcolor=COLORS["black"])
    ax.xaxis.set_major_formatter(FuncFormatter(money_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))
    
    fig.savefig(CHART_DIR / f"{table_name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Exported chart and summary table for {table_name}")

def redraw_high_value_group_charts():
    """
    DRAW 4 CHARTS: HIGH VALUE CUSTOMER GROUPS
    1. Profit by High Value Group (Stacked Bar Chart with % share text)
    2. Profit Share Mix (Stacked Area Chart)
    3. YoY Profit Change by Group (Line Chart)
    4. % Contribute to Overall Year Total (Heatmap)
    """
    print("Loading data for High Value Group charts...")
    data = load_data(BASE_DIR)
    sales_fact = build_sales_fact(data["orders"], data["order_items"], data["products"])
    
    _, combined = build_high_value_customer_summary(sales_fact, data["products"])
    
    group_summary = combined.groupby(["year", "customer_group"], as_index=False).agg(
        high_profit=("high_profit", "sum")
    )
    
    # Filter out empty years
    years_with_data = group_summary.groupby("year")["high_profit"].sum()
    valid_years = years_with_data[years_with_data > 0].index
    group_summary = group_summary[group_summary["year"].isin(valid_years)].copy()
    
    year_total = group_summary.groupby("year", as_index=False).agg(
        total_high_profit=("high_profit", "sum")
    )
    group_summary = group_summary.merge(year_total, on="year", how="left")
    
    group_order = {
        "High only": 0,
        "High overlap medium": 1,
        "High overlap low": 2,
        "High + medium + low": 3,
    }
    group_summary["group_order"] = group_summary["customer_group"].map(group_order)
    group_summary = group_summary.sort_values(["customer_group", "year"])
    
    group_summary["profit_share"] = np.where(
        group_summary["total_high_profit"] != 0,
        group_summary["high_profit"] / group_summary["total_high_profit"],
        0.0,
    )
    group_summary["prev_high_profit"] = group_summary.groupby("customer_group")["high_profit"].shift(1)
    group_summary["high_profit_change"] = group_summary["high_profit"] - group_summary["prev_high_profit"]
    group_summary["pct_change_vs_last_year"] = np.where(
        group_summary["prev_high_profit"].notna() & (group_summary["prev_high_profit"] != 0),
        group_summary["high_profit_change"] / group_summary["prev_high_profit"],
        np.nan,
    )
    group_summary["pct_contribute_to_year_change"] = np.where(
        group_summary["total_high_profit"] != 0,
        group_summary["high_profit_change"] / group_summary["total_high_profit"],
        np.nan,
    )
    
    # Baseline for YoY change
    year_total_sorted = year_total.sort_values("year").copy()
    year_total_sorted["prev_total"] = year_total_sorted["total_high_profit"].shift(1)
    year_total_sorted["pct_change_overall"] = np.where(
        year_total_sorted["prev_total"].notna() & (year_total_sorted["prev_total"] != 0),
        (year_total_sorted["total_high_profit"] - year_total_sorted["prev_total"]) / year_total_sorted["prev_total"],
        np.nan
    )
    
    group_summary = group_summary.sort_values(["group_order", "year"]).reset_index(drop=True)
    table_name = "high_value_group_summary"
    group_summary.to_csv(SUMMARY_DIR / f"{table_name}.csv", index=False)
    
    plt.rcParams['font.family'] = 'Arial'
    years = sorted(group_summary["year"].unique())
    x_pos = np.arange(len(years))
    
    group_labels = ["High only", "High overlap medium", "High overlap low", "High + medium + low"]
    group_colors = [COLORS["dark_green"], COLORS["blue"], COLORS["orange"], COLORS["dark_yellow"]]
    
    profit_pivot = group_summary.pivot(index="year", columns="customer_group", values="high_profit").reindex(years)[group_labels]
    share_pivot = group_summary.pivot(index="year", columns="customer_group", values="profit_share").reindex(years)[group_labels]
    change_pivot = group_summary.pivot(index="year", columns="customer_group", values="pct_change_vs_last_year").reindex(years)[group_labels]
    contrib_pivot = group_summary.pivot(index="customer_group", columns="year", values="pct_contribute_to_year_change").reindex(group_labels)[years]
    overall_change = year_total_sorted.set_index("year")["pct_change_overall"].reindex(years).replace([np.inf, -np.inf], np.nan).values
    
    # 1. Bar Chart
    fig, ax = plt.subplots(figsize=(24, 12), dpi=100)
    ax.grid(False)
    bottoms = np.zeros(len(years))
    for i, grp in enumerate(group_labels):
        values = profit_pivot[grp].fillna(0).values
        bars = ax.bar(x_pos, values, bottom=bottoms, width=0.7, color=group_colors[i], label=grp)
        shares = share_pivot[grp].fillna(0).values
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                y_center = bottoms[j] + height / 2
                if shares[j] > 0.05:
                    ax.text(bar.get_x() + bar.get_width() / 2, y_center, f"{shares[j]:.1%}", ha='center', va='center', fontsize=24, color=COLORS["black"], fontweight='bold')
        bottoms += values
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(COLORS["black"])
    ax.spines['left'].set_color(COLORS["black"])
    ax.set_title("Profit by High Value Customer Group", loc="center", fontsize=36, fontweight="bold", color=COLORS["black"], pad=20)
    ax.set_ylabel("Profit", fontsize=28, color=COLORS["black"])
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(y) for y in years], fontsize=32, color=COLORS["black"])
    ax.tick_params(axis='y', labelsize=32, labelcolor=COLORS["black"])
    ax.yaxis.set_major_formatter(FuncFormatter(money_formatter))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), frameon=False, ncol=4, fontsize=24)
    fig.savefig(CHART_DIR / "high_value_group_profit.png", bbox_inches="tight")
    plt.close(fig)
    
    # 2. Area Chart
    fig, ax = plt.subplots(figsize=(24, 12), dpi=100)
    ax.grid(False)
    ax.stackplot(x_pos, share_pivot["High only"], share_pivot["High overlap medium"], share_pivot["High overlap low"], share_pivot["High + medium + low"],
                 labels=group_labels, colors=group_colors, alpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(COLORS["black"])
    ax.spines['left'].set_color(COLORS["black"])
    ax.set_title("Profit Share Mix by High Value Group", loc="center", fontsize=36, fontweight="bold", color=COLORS["black"], pad=20)
    ax.set_ylabel("Share", fontsize=28, color=COLORS["black"])
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(y) for y in years], fontsize=32, color=COLORS["black"])
    ax.tick_params(axis='y', labelsize=32, labelcolor=COLORS["black"])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), frameon=False, ncol=4, fontsize=24)
    fig.savefig(CHART_DIR / "high_value_group_share.png", bbox_inches="tight")
    plt.close(fig)
    
    # 3. Line Chart
    fig, ax = plt.subplots(figsize=(24, 12), dpi=100)
    ax.grid(False)
    for i, grp in enumerate(group_labels):
        values = change_pivot[grp].replace([np.inf, -np.inf], np.nan).values
        ax.plot(x_pos, values, color=group_colors[i], linewidth=4, marker='o', markersize=10, label=grp)
    ax.plot(x_pos, overall_change, color=COLORS["black"], linewidth=4, linestyle="--", marker='o', markersize=10, label="Overall Change")
    ax.axhline(0, color=COLORS["gray"], linewidth=2, linestyle="--")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(COLORS["black"])
    ax.spines['left'].set_color(COLORS["black"])
    ax.set_title("YoY Profit Change by High Value Group", loc="center", fontsize=36, fontweight="bold", color=COLORS["black"], pad=20)
    ax.set_ylabel("% Change", fontsize=28, color=COLORS["black"])
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(y) for y in years], fontsize=32, color=COLORS["black"])
    ax.tick_params(axis='y', labelsize=32, labelcolor=COLORS["black"])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), frameon=False, ncol=5, fontsize=20)
    fig.savefig(CHART_DIR / "high_value_group_yoy_change.png", bbox_inches="tight")
    plt.close(fig)
    
    # 4. Heatmap
    fig, ax = plt.subplots(figsize=(24, 12), dpi=100)
    ax.grid(False)
    import seaborn as sns
    sns.heatmap(contrib_pivot, cmap="RdYlBu_r", center=0, linewidths=0.5, linecolor="white",
                cbar_kws={"format": FuncFormatter(lambda y, _: f"{y:.0%}")}, ax=ax,
                annot=True, fmt=".1%", annot_kws={"size": 24, "weight": "bold"})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=26)
    ax.set_title("% Contribute to Overall Year Total (High Value)", loc="center", fontsize=36, fontweight="bold", color=COLORS["black"], pad=20)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks(x_pos + 0.5)
    ax.set_xticklabels([str(y) for y in years], fontsize=32, color=COLORS["black"])
    ax.set_yticklabels(group_labels, fontsize=32, color=COLORS["black"], rotation=0)
    fig.savefig(CHART_DIR / "high_value_group_contribute.png", bbox_inches="tight")
    plt.close(fig)
    print("Exported 4 charts for High Value Customer Groups.")

def redraw_high_value_kpi_dashboard():
    """
    DRAW CHART: HIGH VALUE KPI DASHBOARD (AOV, Frequency, Total Customer)
    - 4 rows (customer_group) x 3 columns (metrics) in 1 image.
    - Broken down by customer_status (Old, New, Dropped).
    """
    print("Loading data for High Value KPI Dashboard...")
    data = load_data(BASE_DIR)
    sales_fact = build_sales_fact(data["orders"], data["order_items"], data["products"])
    
    summary, _ = build_high_value_customer_summary(sales_fact, data["products"])
    
    table_name = "high_value_kpi_summary"
    summary.to_csv(SUMMARY_DIR / f"{table_name}.csv", index=False)
    
    plt.rcParams['font.family'] = 'Arial'
    years = sorted(summary["year"].dropna().unique())
    x_pos = np.arange(len(years))
    
    group_order = ["High only", "High overlap medium", "High overlap low", "High + medium + low"]
    status_order = ["New customer", "Old customer", "Dropped customer from last year"]
    status_colors = {
        "New customer": COLORS["light_green"],
        "Old customer": COLORS["blue"],
        "Dropped customer from last year": COLORS["gray"]
    }
    
    metrics = [
        ("total_customer", "Total Customers", lambda y, _: f"{y:,.0f}"),
        ("frequency", "Frequency (Days/Order)", lambda y, _: f"{y:,.1f}"),
        ("aov", "AOV", money_formatter)
    ]
    
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(36, 32), dpi=100)
    fig.suptitle("High Value Customer KPIs by Group and Status", fontsize=48, fontweight="bold", color=COLORS["black"], y=0.95)
    
    for row, grp in enumerate(group_order):
        grp_data = summary[summary["customer_group"] == grp]
        for col, (metric, title, formatter) in enumerate(metrics):
            ax = axes[row, col]
            ax.grid(False)
            
            for status in status_order:
                status_data = grp_data[grp_data["customer_status"] == status].set_index("year")
                values = status_data[metric].reindex(years).replace([np.inf, -np.inf], np.nan).values
                
                # Hide 0 values for frequency and AOV for dropped customers to avoid skewing the chart
                if status == "Dropped customer from last year" and metric in ["frequency", "aov"]:
                    values = np.where(values == 0, np.nan, values)
                    
                ax.plot(x_pos, values, color=status_colors[status], linewidth=4, marker='o', markersize=10, label=status)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color(COLORS["black"])
            ax.spines['left'].set_color(COLORS["black"])
            
            if row == 0:
                ax.set_title(title, loc="center", fontsize=36, fontweight="bold", color=COLORS["black"], pad=20)
            
            if col == 0:
                ax.set_ylabel(grp, fontsize=28, fontweight="bold", color=COLORS["black"], rotation=90, labelpad=20)
                
            ax.set_xticks(x_pos)
            ax.set_xticklabels([str(y) for y in years], fontsize=20, color=COLORS["black"], rotation=45, ha='right')
            ax.tick_params(axis='y', labelsize=24, labelcolor=COLORS["black"])
            ax.yaxis.set_major_formatter(FuncFormatter(formatter))
            
            # Add legend only to the top left subplot to avoid clutter
            if row == 0 and col == 0:
                ax.legend(loc="upper left", frameon=False, fontsize=20)
                
    # Adjust spacing
    plt.subplots_adjust(top=0.90, bottom=0.05, hspace=0.3, wspace=0.2)
    
    fig.savefig(CHART_DIR / f"{table_name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Exported chart and summary table for {table_name}")

def redraw_traffic_conversion_chart():
    """
    DRAW CHART: TRAFFIC CONVERSION RATE AND BOUNCE RATE
    - 2 subplots vertically: Conversion Rate and Bounce Rate over years by Channel.
    """
    print("Loading data for Traffic Conversion Chart...")
    table_name = "traffic_conversion_summary"
    data_path = SUMMARY_DIR / f"{table_name}.csv"
    
    if not data_path.exists():
        print(f"File not found: {data_path}. Please run analysis_traffic.py first.")
        return
        
    df = pd.read_csv(data_path)
    
    plt.rcParams['font.family'] = 'Arial'
    years = sorted(df["year"].unique())
    x_pos = np.arange(len(years))
    
    channels = sorted(df["traffic_source"].unique())
    # Define colors for up to 6 channels
    palette = [COLORS["dark_green"], COLORS["blue"], COLORS["orange"], COLORS["dark_yellow"], COLORS["light_green"], COLORS["gray"]]
    channel_colors = {ch: palette[i % len(palette)] for i, ch in enumerate(channels)}
    
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(24, 20), dpi=100)
    
    metrics = [
        ("conversion_rate", "Conversion Rate by Traffic Source (New High-Price Customers)"),
        ("avg_bounce_rate", "Average Bounce Rate by Traffic Source")
    ]
    
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx]
        ax.grid(False)
        
        pivot = df.pivot(index="year", columns="traffic_source", values=metric).reindex(years)[channels]
        
        for ch in channels:
            values = pivot[ch].replace([np.inf, -np.inf], np.nan).values
            ax.plot(x_pos, values, color=channel_colors[ch], linewidth=4, marker='o', markersize=10, label=ch)
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(COLORS["black"])
        ax.spines['left'].set_color(COLORS["black"])
        
        ax.set_title(title, loc="center", fontsize=36, fontweight="bold", color=COLORS["black"], pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(y) for y in years], fontsize=32, color=COLORS["black"])
        ax.tick_params(axis='y', labelsize=32, labelcolor=COLORS["black"])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1%}"))
        
        if idx == 0:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), frameon=False, ncol=6, fontsize=24)
            
    plt.subplots_adjust(hspace=0.4)
    
    fig.savefig(CHART_DIR / f"{table_name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Exported chart for {table_name}")

def redraw_bounce_rate_chart():
    """
    DRAW CHART: AVERAGE BOUNCE RATE ONLY
    - Single line chart showing Bounce Rate over years by Channel.
    """
    print("Loading data for Bounce Rate Chart...")
    table_name = "bounce_rate_summary"
    data_path = SUMMARY_DIR / "traffic_conversion_summary.csv"
    
    if not data_path.exists():
        print(f"File not found: {data_path}. Please run analysis_traffic.py first.")
        return
        
    df = pd.read_csv(data_path)
    
    plt.rcParams['font.family'] = 'Arial'
    years = sorted(df["year"].unique())
    x_pos = np.arange(len(years))
    
    channels = sorted(df["traffic_source"].unique())
    # Define colors for up to 6 channels
    palette = [COLORS["dark_green"], COLORS["blue"], COLORS["orange"], COLORS["dark_yellow"], COLORS["light_green"], COLORS["gray"]]
    channel_colors = {ch: palette[i % len(palette)] for i, ch in enumerate(channels)}
    
    fig, ax = plt.subplots(figsize=(24, 10), dpi=100)
    ax.grid(False)
    
    metric = "avg_bounce_rate"
    title = "Average Bounce Rate by Traffic Source"
    
    pivot = df.pivot(index="year", columns="traffic_source", values=metric).reindex(years)[channels]
    
    for ch in channels:
        values = pivot[ch].replace([np.inf, -np.inf], np.nan).values
        ax.plot(x_pos, values, color=channel_colors[ch], linewidth=4, marker='o', markersize=10, label=ch)
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(COLORS["black"])
    ax.spines['left'].set_color(COLORS["black"])
    
    ax.set_title(title, loc="center", fontsize=36, fontweight="bold", color=COLORS["black"], pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(y) for y in years], fontsize=32, color=COLORS["black"])
    ax.tick_params(axis='y', labelsize=32, labelcolor=COLORS["black"])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1%}"))
    
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), frameon=False, ncol=6, fontsize=24)
            
    fig.savefig(CHART_DIR / f"{table_name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Exported chart for {table_name}")


def redraw_high_cust_category_chart():
    """
    DRAW CHART: CATEGORY DISTRIBUTION FOR HIGH-PRICE CUSTOMERS
    - Bar chart showing revenue share by category for customers who purchased high-price products.
    """
    print("Loading data for High-Price Customer Category Chart...")
    table_name = "high_cust_category_dist"
    data_path = SUMMARY_DIR / f"{table_name}.csv"
    
    if not data_path.exists():
        print(f"File not found: {data_path}. Please run analysis_high_cust_categories.py first.")
        return
        
    df = pd.read_csv(data_path)
    df = df.sort_values("total_revenue", ascending=False)
    
    plt.rcParams['font.family'] = 'Arial'
    
    fig, ax = plt.subplots(figsize=(24, 12), dpi=100)
    ax.grid(False)
    
    # Use green theme
    colors = [COLORS["dark_green"], COLORS["blue"], COLORS["orange"], COLORS["dark_yellow"]]
    bars = ax.bar(df["category"], df["total_revenue"], color=colors[:len(df)], width=0.6)
    
    # Add percentage labels on top of bars
    total_rev = df["total_revenue"].sum()
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (total_rev * 0.01),
                f'{(height/total_rev):.1%}', ha='center', va='bottom', 
                fontsize=32, fontweight='bold', color=COLORS["black"])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(COLORS["black"])
    ax.spines['left'].set_color(COLORS["black"])
    
    ax.set_title("Revenue Distribution by Category for High-Price Purchasers", loc="center", fontsize=36, fontweight="bold", color=COLORS["black"], pad=20)
    ax.set_ylabel("Total Revenue", fontsize=28, color=COLORS["black"])
    ax.tick_params(axis='x', labelsize=32, labelcolor=COLORS["black"])
    ax.tick_params(axis='y', labelsize=32, labelcolor=COLORS["black"])
    ax.yaxis.set_major_formatter(FuncFormatter(money_formatter))
    
    fig.savefig(CHART_DIR / f"{table_name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Exported chart for {table_name}")

def redraw_high_cust_promo_source_chart():
    """
    DRAW CHART: ACQUISITION SOURCE FOR HIGH-PRICE PROMO USERS
    - Bar chart showing the distribution of order sources for high-price customers who used promos.
    """
    print("Loading data for High-Price Promo Source Chart...")
    table_name = "high_cust_promo_source_dist"
    data_path = SUMMARY_DIR / f"{table_name}.csv"
    
    if not data_path.exists():
        print(f"File not found: {data_path}. Please run analysis_high_cust_promo_source.py first.")
        return
        
    df = pd.read_csv(data_path)
    df = df.sort_values("order_count", ascending=False)
    
    plt.rcParams['font.family'] = 'Arial'
    
    fig, ax = plt.subplots(figsize=(24, 12), dpi=100)
    ax.grid(False)
    
    # Use green-centric theme
    colors = [COLORS["dark_green"], COLORS["blue"], COLORS["orange"], COLORS["dark_yellow"], COLORS["light_green"], COLORS["gray"]]
    bars = ax.bar(df["order_source"], df["order_count"], color=colors[:len(df)], width=0.6)
    
    # Add percentage labels
    total_orders = df["order_count"].sum()
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (total_orders * 0.01),
                f'{(height/total_orders):.1%}', ha='center', va='bottom', 
                fontsize=32, fontweight='bold', color=COLORS["black"])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(COLORS["black"])
    ax.spines['left'].set_color(COLORS["black"])
    
    ax.set_title("Order Source for High-Price Customers using Promos", loc="center", fontsize=36, fontweight="bold", color=COLORS["black"], pad=20)
    ax.set_ylabel("Order Count", fontsize=28, color=COLORS["black"])
    ax.tick_params(axis='x', labelsize=28, labelcolor=COLORS["black"])
    ax.tick_params(axis='y', labelsize=28, labelcolor=COLORS["black"])
    
    fig.savefig(CHART_DIR / f"{table_name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Exported chart for {table_name}")


if __name__ == '__main__':     
    redraw_pnl_proxy_chart()    
    redraw_cash_flow_proxy_chart()    
    redraw_price_bucket_charts()    
    redraw_product_price_histogram()    
    redraw_high_customer_scatter()    
    redraw_high_value_group_charts()    
    redraw_high_value_kpi_dashboard()    
    redraw_traffic_conversion_chart()    
    redraw_bounce_rate_chart()    
    redraw_high_cust_category_chart()    
    redraw_high_cust_promo_source_chart()    
    print('ALL DONE!')