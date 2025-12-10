import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# 🔐 PASSWORD PROTECTION
def check_password():
    """Simple password check using Streamlit secrets."""
    def password_entered():
        """Checks whether the password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
            st.session_state["authenticated"] = True
            del st.session_state["password"]  # don't store the password
        else:
            st.session_state["authenticated"] = False

    # First run: show input
    if "authenticated" not in st.session_state:
        st.text_input("Password", type="password", key="password", on_change=password_entered)
        st.stop()

    # If not authenticated, show input again
    if not st.session_state["authenticated"]:
        st.text_input("Password", type="password", key="password", on_change=password_entered)
        st.error("Incorrect password")
        st.stop()

    # If we get here, the user is authenticated
    return True

# GOOGLE SHEET LINKING
import gspread
from google.oauth2.service_account import Credentials
import json

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

def load_google_sheet_to_df():
    sa = st.secrets["gcp_service_account"]
    sheet_url = st.secrets["google"]["sheet_url"]

    creds = Credentials.from_service_account_info(sa).with_scopes(SCOPES)
    gc = gspread.authorize(creds)

    sh = gc.open_by_url(sheet_url)
    ws = sh.sheet1  # or sh.worksheet("Final Consolidated Raw Data")
    data = ws.get_all_records()
    return pd.DataFrame(data)

RATE_METRICS = [
    "CTR",
    "CVR",
    "purchase_rate",
    "add_to_cart_rate",
    "view_content_rate",
    "page_view_rate",
    "online_order_rate",
    "reservation_rate",
]

RATE_METRICS_SET = set(RATE_METRICS)

CURRENCY_METRICS_SET = {
    "CPC",
    "CPA",
    "CPM",
    "spend",
    "cost_per_online_order",
    "cost_per_reservation",
    "cost_per_store_visit",
}

pd.options.display.float_format = '{:,.2f}'.format

def format_currency_columns(df):
    currency_cols = [
        'spend','Spend',
        'total_spend','Total Spend',
        'cpc','CPC',
        'cpa','CPA',
        'cpm','CPM',
        'online_order_revenue',
        'reservation_revenue',
        'store_sales',
        'total_revenue_est'
    ]
    for col in df.columns:
        if col.lower() in [c.lower() for c in currency_cols]:
            df[col] = df[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else x)
    return df

def classify_objective(objective: str) -> str:
    """
    Classify objective into Awareness, Conversion, or Other based on keywords.
    """
    if not isinstance(objective, str):
        return "Other"
    obj = objective.lower()
    awareness_keywords = ["awareness", "reach", "video view", "brand", "impression"]
    conversion_keywords = ["conversion", "purchase", "sale", "lead", "catalog", "app install"]

    if any(k in obj for k in awareness_keywords):
        return "Awareness"
    if any(k in obj for k in conversion_keywords):
        return "Conversion"
    return "Other"


import numpy as np

def classify_journey_role(
    row,
    ctr_median=None,
    cpc_median=None,
    cvr_median=None,
    intent_median=None,
    purchase_rate_median=None,
    cpa_median=None,
    # tunable multipliers
    cvr_boost=1.2,
    intent_boost=1.1,
    ctr_boost=1.1,
    cpc_discount=0.9, 
    cpa_discount=0.8,
    purchase_boost=1.2,
):
    """
    Classify creative into Engagement, Intent, or Conversion.

    Priority:
      1) Conversion  – strong CVR / purchase_rate and/or efficient CPA
      2) Intent      – strong micro-conversion activity vs peers
      3) Engagement  – strong CTR vs peers

    Falls back sensibly when medians or metrics are missing.
    """

    ctr = row.get("CTR", 0.0)
    cvr = row.get("CVR", 0.0)  # may not exist in your data
    purchase_rate = row.get("purchase_rate", 0.0)
    cpa = row.get("cost_per_purchase", None) or row.get("CPA", None)

    # ---- Intent metrics (mid-funnel) ----
    intent_metrics = []
    for metric in ["add_to_cart_rate", "view_content_rate", "page_view_rate"]:
        val = row.get(metric, 0.0)
        if val and val > 0:
            intent_metrics.append(val)
    avg_intent = float(np.mean(intent_metrics)) if intent_metrics else 0.0
    has_intent_metrics = avg_intent > 0

    # ---------------------------------------------------------
    # 1) Conversion: strong closers
    #    - high CVR vs median (if present)
    #    - OR high purchase_rate vs non-zero median
    #    - OR efficient CPA vs median
    # ---------------------------------------------------------
    strong_conversion = False

    # CVR-based signal (if CVR exists in your data)
    if cvr_median and cvr_median > 0:
        if cvr >= cvr_boost * cvr_median:
            strong_conversion = True

    # purchase_rate-based signal (works for your current dataset)
    if purchase_rate_median and purchase_rate_median > 0:
        if purchase_rate >= purchase_boost * purchase_rate_median:
            strong_conversion = True
    elif purchase_rate > 0:
        # if median is 0, ANY non-zero purchase_rate is meaningful
        strong_conversion = True

    # CPA-based signal (lower is better)
    if cpa is not None and cpa_median and cpa_median > 0:
        if cpa <= cpa_discount * cpa_median:
            strong_conversion = True

    if strong_conversion:
        return "Conversion"

    # ---------------------------------------------------------
    # 2) Intent: strong mid-funnel signals
    # ---------------------------------------------------------
    strong_intent = False
    if has_intent_metrics and intent_median and intent_median > 0:
        if avg_intent >= intent_boost * intent_median:
            strong_intent = True

    if strong_intent:
        return "Intent"

    # ---------------------------------------------------------
    # 3) Engagement: strong cost-efficiency (CPC) + optional CTR
    # ---------------------------------------------------------
    strong_engagement = False

    # CPC-based signal (lower = better). e.g. <= 90% of median CPC
    if cpc_median and cpc_median > 0 and "CPC" in row:
        if row["CPC"] <= cpc_discount * cpc_median:
            strong_engagement = True

    # Optional backup: if CTR is also clearly strong, treat as Engagement
    if (not strong_engagement) and ctr_median and ctr_median > 0:
        if ctr >= ctr_boost * ctr_median:
            strong_engagement = True

    if strong_engagement:
        return "Engagement"

    # ---------------------------------------------------------
    # 4) Fallbacks when medians are missing or everything is meh
    # ---------------------------------------------------------
    if purchase_rate > 0:
        return "Conversion"
    if has_intent_metrics:
        return "Intent"
    if ctr > 0:
        return "Engagement"

    return "Engagement"


def load_and_prepare_data(df_raw: pd.DataFrame):
    """
    Load CSV file and prepare data with derived metrics.
    Supports the Lazy Dog schema and maps it into the app's internal columns.
    Returns processed dataframe or None if validation fails.
    """
    try:
        df = df_raw.copy()

        # --- 1) Normalize column names from Lazy Dog template to app internals ---
        rename_map = {
            "Date": "date",
            "Week Start": "week_start", 
            "Period": "period",              
            "Fiscal Year": "fiscal_year",   
            "Channel": "platform",
            "Campaign": "campaign_name",
            "Content Topic": "topic",
            "Ad name": "creative_name",
            "Creative Size": "format",
            "Impressions": "impressions",
            "Clicks": "clicks",
            "Spend": "spend",
        
            # conversions & events
            "Online Order": "online_orders",
            "Online Order Revenue": "online_order_revenue",
            "Store Traffic": "store_visits",
            "Store Visit Revenue": "store_sales",
            "Reservations": "reservations",
            "Content Views": "view_content",
            "Add To Carts": "add_to_carts",
            "Page Views": "page_views",
        }
        df = df.rename(columns=rename_map)

        # --- 2) Validate required columns (after renaming) ---
        required_cols = [
            "date",
            "platform",
            "campaign_name",
            "creative_name",
            "impressions",
            "clicks",
            "spend",
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            st.info(
                "Required columns (after renaming): "
                "date, platform, campaign_name, creative_name, impressions, clicks, spend"
            )
            return None

        # --- 3) Parse dates ---
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isna().all():
            st.error("❌ Could not parse any dates in the 'date' column")
            return None
        
        if "week_start" in df.columns:
            df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")

        # --- 4) Clean numeric columns ---
        numeric_cols = ["impressions", "clicks", "spend"]

        # legacy optional numeric columns
        optional_numeric = [
            "conversions",
            "revenue",
            "purchases",
            "add_to_carts",
            "view_content",
            "page_views",
        ]

        # new Lazy Dog numeric columns
        optional_numeric.extend(
            [
                "online_orders",
                "online_order_revenue",
                "store_visits",
                "store_sales",
                "reservations",
                "video_views",
                "video_completions",
            ]
        )

        def clean_numeric(series):
            # Convert to string, remove $ and commas, strip spaces
            return (
                series.astype(str)
                .str.replace(r"[\$,]", "", regex=True)
                .str.strip()
            )

        for col in numeric_cols:
            df[col] = clean_numeric(df[col])
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        for col in optional_numeric:
            if col in df.columns:
                df[col] = clean_numeric(df[col])
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # --- 5) Base funnel metrics ---
        df["CTR"] = np.where(df["impressions"] > 0, df["clicks"] / df["impressions"], 0)
        df["CPM"] = np.where(
            df["impressions"] > 0, df["spend"] / df["impressions"] * 1000, 0
        )
        df["CPC"] = np.where(df["clicks"] > 0, df["spend"] / df["clicks"], 0)

        # Legacy conversions (if present)
        if "conversions" in df.columns:
            df["CVR"] = np.where(df["clicks"] > 0, df["conversions"] / df["clicks"], 0)
            df["CPA"] = np.where(
                df["conversions"] > 0, df["spend"] / df["conversions"], 0
            )

        if "purchases" in df.columns:
            df["purchase_rate"] = np.where(
                df["clicks"] > 0, df["purchases"] / df["clicks"], 0
            )
            df["cost_per_purchase"] = np.where(
                df["purchases"] > 0, df["spend"] / df["purchases"], 0
            )

        # --- 6) New Lazy Dog conversion metrics ---

        # Online Orders (all platforms)
        if "online_orders" in df.columns:
            df["online_order_rate"] = np.where(
                df["clicks"] > 0, df["online_orders"] / df["clicks"], 0
            )
            df["cost_per_online_order"] = np.where(
                df["online_orders"] > 0, df["spend"] / df["online_orders"], 0
            )

        if "online_order_revenue" in df.columns:
            df["online_order_roas"] = np.where(
                df["spend"] > 0, df["online_order_revenue"] / df["spend"], 0
            )

        # Reservations (Meta + Google)
        if "reservations" in df.columns:
            df["reservation_rate"] = np.where(
                df["clicks"] > 0, df["reservations"] / df["clicks"], 0
            )
            df["cost_per_reservation"] = np.where(
                df["reservations"] > 0, df["spend"] / df["reservations"], 0
            )

        if "reservation_revenue" in df.columns:
            df["reservation_roas"] = np.where(
                df["spend"] > 0, df["reservation_revenue"] / df["spend"], 0
            )

        # Store Visits (Google)
        if "store_visits" in df.columns:
            df["store_visit_rate"] = np.where(
                df["clicks"] > 0, df["store_visits"] / df["clicks"], 0
            )
            df["cost_per_store_visit"] = np.where(
                df["store_visits"] > 0, df["spend"] / df["store_visits"], 0
            )

        if "store_sales" in df.columns:
            df["store_sales_roas"] = np.where(
                df["spend"] > 0, df["store_sales"] / df["spend"], 0
            )

        # Micro-conversions (content / page engagement)
        if "add_to_carts" in df.columns:
            df["add_to_cart_rate"] = np.where(
                df["clicks"] > 0, df["add_to_carts"] / df["clicks"], 0
            )
            df["cost_per_add_to_cart"] = np.where(
                df["add_to_carts"] > 0, df["spend"] / df["add_to_carts"], 0
            )

        if "view_content" in df.columns:
            df["view_content_rate"] = np.where(
                df["clicks"] > 0, df["view_content"] / df["clicks"], 0
            )
            df["cost_per_view_content"] = np.where(
                df["view_content"] > 0, df["spend"] / df["view_content"], 0
            )

        if "page_views" in df.columns:
            df["page_view_rate"] = np.where(
                df["clicks"] > 0, df["page_views"] / df["clicks"], 0
            )
            df["cost_per_page_view"] = np.where(
                df["page_views"] > 0, df["spend"] / df["page_views"], 0
            )

        # Legacy revenue
        if "revenue" in df.columns:
            df["ROAS"] = np.where(df["spend"] > 0, df["revenue"] / df["spend"], 0)

        # --- 7) Sort, age, cumulative impressions, objective classification ---
        df = df.sort_values(["creative_name", "date"])

        creative_first_dates = df.groupby("creative_name")["date"].transform("min")
        df["age_in_days"] = (df["date"] - creative_first_dates).dt.days

        df["cumulative_impressions"] = df.groupby("creative_name")["impressions"].cumsum()

        if "objective" in df.columns:
            df["objective_type"] = df["objective"].apply(classify_objective)

        return df

    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return None


def apply_global_filters(df, filters):
    """
    Apply global filters to the dataframe.
    """
    filtered_df = df.copy()

    if filters['date_range']:
        start_date, end_date = filters['date_range']
        filtered_df = filtered_df[
            (filtered_df['date'] >= pd.to_datetime(start_date)) & 
            (filtered_df['date'] <= pd.to_datetime(end_date))
        ]

    if filters['platforms']:
        filtered_df = filtered_df[filtered_df['platform'].isin(filters['platforms'])]

    if filters['campaigns']:
        filtered_df = filtered_df[filtered_df['campaign_name'].isin(filters['campaigns'])]

    if filters.get('objectives') is not None:
        all_objectives_in_data = set(df['objective'].dropna().unique())
        if set(filters['objectives']) != all_objectives_in_data:
            filtered_df = filtered_df[filtered_df['objective'].isin(filters['objectives'])]

    if 'objective_type' in filtered_df.columns and filters.get('objective_type') not in (None, 'All'):
        filtered_df = filtered_df[filtered_df['objective_type'] == filters['objective_type']]

    if filters.get('topics') is not None and 'topic' in df.columns:
        all_topics_in_data = set(df['topic'].dropna().unique())
        if set(filters['topics']) != all_topics_in_data:
            filtered_df = filtered_df[filtered_df['topic'].isin(filters['topics'])]

    # --- NEW: format filter ---
    if filters.get('formats') is not None and 'format' in filtered_df.columns:
        all_formats_in_data = set(df['format'].dropna().unique())
        if set(filters['formats']) != all_formats_in_data:
            filtered_df = filtered_df[filtered_df['format'].isin(filters['formats'])]

    # --- NEW: placement filter ---
    if filters.get('placements') is not None and 'placement' in filtered_df.columns:
        all_place_in_data = set(df['placement'].dropna().unique())
        if set(filters['placements']) != all_place_in_data:
            filtered_df = filtered_df[filtered_df['placement'].isin(filters['placements'])]

    agg_dict = {'impressions': 'sum'}
    if 'conversions' in filtered_df.columns:
        agg_dict['conversions'] = 'sum'

    creative_totals = filtered_df.groupby('creative_name').agg(agg_dict).reset_index()

    valid_creatives = creative_totals[
        creative_totals['impressions'] >= filters['min_impressions']
    ]['creative_name']

    if 'conversions' in filtered_df.columns and filters['min_conversions'] > 0:
        valid_creatives_conv = creative_totals[
            creative_totals['conversions'] >= filters['min_conversions']
        ]['creative_name']
        valid_creatives = set(valid_creatives) & set(valid_creatives_conv)

    filtered_df = filtered_df[filtered_df['creative_name'].isin(valid_creatives)]

    return filtered_df


@st.cache_data
def compute_aggregated_creative_metrics(df):
    """
    Aggregate metrics at the creative level.
    """
    agg_dict = {
        'impressions': 'sum',
        'clicks': 'sum',
        'spend': 'sum',
        'platform': 'first',
        'campaign_name': 'first',
        'age_in_days': 'max',
        'date': 'nunique'
    }

    if 'conversions' in df.columns:
        agg_dict['conversions'] = 'sum'

    if 'add_to_carts' in df.columns:
        agg_dict['add_to_carts'] = 'sum'

    if 'view_content' in df.columns:
        agg_dict['view_content'] = 'sum'

    if 'page_views' in df.columns:
        agg_dict['page_views'] = 'sum'

    if 'format' in df.columns:
        agg_dict['format'] = 'first'

    if 'objective' in df.columns:
        agg_dict['objective'] = 'first'

    if 'objective_type' in df.columns:
        agg_dict['objective_type'] = 'first'

    if 'topic' in df.columns:
        agg_dict['topic'] = 'first'

    if "online_orders" in df.columns:
        agg_dict["online_orders"] = "sum"
        
    if "online_order_revenue" in df.columns:
        agg_dict["online_order_revenue"] = "sum"

    if "store_visits" in df.columns:
        agg_dict["store_visits"] = "sum"
        
    if "store_sales" in df.columns:
        agg_dict["store_sales"] = "sum"

    if "reservations" in df.columns:
        agg_dict["reservations"] = "sum"


    creative_metrics = df.groupby('creative_name').agg(agg_dict).reset_index()

    creative_metrics.rename(columns={
        'age_in_days': 'age_in_days_max',
        'date': 'total_days_active'
    }, inplace=True)

    creative_metrics['CTR'] = np.where(
        creative_metrics['impressions'] > 0,
        creative_metrics['clicks'] / creative_metrics['impressions'],
        0
    )
    creative_metrics['CPC'] = np.where(
        creative_metrics['clicks'] > 0,
        creative_metrics['spend'] / creative_metrics['clicks'],
        0
    )
    creative_metrics['CPM'] = np.where(
        creative_metrics['impressions'] > 0,
        creative_metrics['spend'] / creative_metrics['impressions'] * 1000,
        0
    )

    if 'conversions' in creative_metrics.columns:
        creative_metrics['CVR'] = np.where(
            creative_metrics['clicks'] > 0,
            creative_metrics['conversions'] / creative_metrics['clicks'],
            0
        )
        creative_metrics['CPA'] = np.where(
            creative_metrics['conversions'] > 0,
            creative_metrics['spend'] / creative_metrics['conversions'],
            0
        )

    if 'revenue' in creative_metrics.columns:
        creative_metrics['ROAS'] = np.where(
            creative_metrics['spend'] > 0,
            creative_metrics['revenue'] / creative_metrics['spend'],
            0
        )

    if 'add_to_carts' in creative_metrics.columns:
        creative_metrics['add_to_cart_rate'] = np.where(
            creative_metrics['clicks'] > 0,
            creative_metrics['add_to_carts'] / creative_metrics['clicks'],
            0
        )
        creative_metrics['cost_per_add_to_cart'] = np.where(
            creative_metrics['add_to_carts'] > 0,
            creative_metrics['spend'] / creative_metrics['add_to_carts'],
            0
        )

    if 'view_content' in creative_metrics.columns:
        creative_metrics['view_content_rate'] = np.where(
            creative_metrics['clicks'] > 0,
            creative_metrics['view_content'] / creative_metrics['clicks'],
            0
        )
        creative_metrics['cost_per_view_content'] = np.where(
            creative_metrics['view_content'] > 0,
            creative_metrics['spend'] / creative_metrics['view_content'],
            0
        )

    if 'page_views' in creative_metrics.columns:
        creative_metrics['page_view_rate'] = np.where(
            creative_metrics['clicks'] > 0,
            creative_metrics['page_views'] / creative_metrics['clicks'],
            0
        )
        creative_metrics['cost_per_page_view'] = np.where(
            creative_metrics['page_views'] > 0,
            creative_metrics['spend'] / creative_metrics['page_views'],
            0
        )

        # --- Online Orders (creative-level) ---
    if "online_orders" in creative_metrics.columns:
        creative_metrics["online_order_rate"] = np.where(
            creative_metrics["clicks"] > 0,
            creative_metrics["online_orders"] / creative_metrics["clicks"],
            0,
        )
        creative_metrics["cost_per_online_order"] = np.where(
            creative_metrics["online_orders"] > 0,
            creative_metrics["spend"] / creative_metrics["online_orders"],
            0,
        )

    if "online_order_revenue" in creative_metrics.columns:
        creative_metrics["online_order_roas"] = np.where(
            creative_metrics["spend"] > 0,
            creative_metrics["online_order_revenue"] / creative_metrics["spend"],
            0,
        )

        # --- Reservations (creative-level) ---
    if "reservations" in creative_metrics.columns:
    
        creative_metrics["reservation_rate"] = np.where(
            creative_metrics["clicks"] > 0,
            creative_metrics["reservations"] / creative_metrics["clicks"],
            0,
        )
    
        creative_metrics["cost_per_reservation"] = np.where(
            creative_metrics["reservations"] > 0,
            creative_metrics["spend"] / creative_metrics["reservations"],
            0,
        )
    
        # ---- Reservation Revenue (explicit or estimated) ----
    if "reservation_revenue" not in creative_metrics.columns:

        AVG_RES_CHECK = 62  # <== default static avg
        creative_metrics["reservation_revenue"] = (
            creative_metrics["reservations"] * AVG_RES_CHECK
        )

    # ---- ROAS based on reservation revenue ----
    creative_metrics["reservation_roas"] = np.where(
        creative_metrics["spend"] > 0,
        creative_metrics["reservation_revenue"] / creative_metrics["spend"],
        0,
    )


        # --- Store Visits (creative-level) ---
    if "store_visits" in creative_metrics.columns:
        creative_metrics["store_visit_rate"] = np.where(
            creative_metrics["clicks"] > 0,
            creative_metrics["store_visits"] / creative_metrics["clicks"],
            0,
        )
        creative_metrics["cost_per_store_visit"] = np.where(
            creative_metrics["store_visits"] > 0,
            creative_metrics["spend"] / creative_metrics["store_visits"],
            0,
        )

    if "store_sales" in creative_metrics.columns:
        creative_metrics["store_sales_roas"] = np.where(
            creative_metrics["spend"] > 0,
            creative_metrics["store_sales"] / creative_metrics["spend"],
            0,
        )


    # ---- Journey role thresholds ----
    # CTR median (only for creatives with impressions)
    ctr_median = creative_metrics.loc[
        creative_metrics['impressions'] > 0, 'CTR'
    ].median() if len(creative_metrics) > 0 else 0

    # CVR median (only > 0 so we don't get dragged down by zeros)
    cvr_median = None
    if 'CVR' in creative_metrics.columns:
        cvr_vals = creative_metrics.loc[creative_metrics['CVR'] > 0, 'CVR']
        cvr_median = cvr_vals.median() if len(cvr_vals) > 0 else None

    # CPC median (only > 0)
    cpc_median = None
    if 'CPC' in creative_metrics.columns:
        cpc_vals = creative_metrics.loc[creative_metrics['CPC'] > 0, 'CPC']
        cpc_median = cpc_vals.median() if len(cpc_vals) > 0 else None

    # Intent median based on micro-conversion rates
    intent_scores = []
    intent_cols = [
        col for col in ['add_to_cart_rate', 'view_content_rate', 'page_view_rate']
        if col in creative_metrics.columns
    ]

    for _, row in creative_metrics.iterrows():
        vals = [row[col] for col in intent_cols if row.get(col, 0) > 0]
        if vals:
            intent_scores.append(np.mean(vals))

    intent_median = np.median(intent_scores) if intent_scores else None

    # Apply journey role classification
    creative_metrics['journey_role'] = creative_metrics.apply(
        lambda row: classify_journey_role(
            row=row,
            ctr_median=ctr_median,
            cvr_median=cvr_median,
            intent_median=intent_median,
            cpc_median=cpc_median,
        ),
        axis=1
    )

    return creative_metrics

def compute_platform_metrics_lazy_dog(df: pd.DataFrame, avg_res_check: float = 62.0):
    """
    Aggregate metrics at the platform level for Lazy Dog.
    Includes online orders, reservations, store visits and their revenues.
    """

    agg_dict = {
        "impressions": "sum",
        "clicks": "sum",
        "spend": "sum",
    }

    for col in [
        "online_orders",
        "online_order_revenue",
        "reservations",
        "reservation_revenue",
        "store_visits",
        "store_sales",
    ]:
        if col in df.columns:
            agg_dict[col] = "sum"

    plat = df.groupby("platform").agg(agg_dict).reset_index()

    # Base funnel
    plat["CTR"] = np.where(plat["impressions"] > 0, plat["clicks"] / plat["impressions"], 0)
    plat["CPC"] = np.where(plat["clicks"] > 0, plat["spend"] / plat["clicks"], 0)
    plat["CPM"] = np.where(plat["impressions"] > 0, plat["spend"] / plat["impressions"] * 1000, 0)

    # Online orders
    if "online_orders" in plat.columns:
        plat["online_order_rate"] = np.where(
            plat["clicks"] > 0, plat["online_orders"] / plat["clicks"], 0
        )
        plat["cost_per_online_order"] = np.where(
            plat["online_orders"] > 0, plat["spend"] / plat["online_orders"], 0
        )
    else:
        plat["online_order_rate"] = 0.0
        plat["cost_per_online_order"] = 0.0

    if "online_order_revenue" in plat.columns:
        plat["online_order_roas"] = np.where(
            plat["spend"] > 0, plat["online_order_revenue"] / plat["spend"], 0
        )
    else:
        plat["online_order_roas"] = 0.0

    # Reservations
    if "reservations" in plat.columns:
        plat["reservation_rate"] = np.where(
            plat["clicks"] > 0, plat["reservations"] / plat["clicks"], 0
        )
        plat["cost_per_reservation"] = np.where(
            plat["reservations"] > 0, plat["spend"] / plat["reservations"], 0
        )

        # If no reservation_revenue aggregated, estimate using avg check
        if "reservation_revenue" not in plat.columns:
            plat["reservation_revenue"] = plat["reservations"] * avg_res_check
    else:
        plat["reservation_rate"] = 0.0
        plat["cost_per_reservation"] = 0.0
        if "reservation_revenue" not in plat.columns:
            plat["reservation_revenue"] = 0.0

    plat["reservation_roas"] = np.where(
        plat["spend"] > 0, plat["reservation_revenue"] / plat["spend"], 0
    )

    # Store visits
    if "store_visits" in plat.columns:
        plat["store_visit_rate"] = np.where(
            plat["clicks"] > 0, plat["store_visits"] / plat["clicks"], 0
        )
        plat["cost_per_store_visit"] = np.where(
            plat["store_visits"] > 0, plat["spend"] / plat["store_visits"], 0
        )
    else:
        plat["store_visit_rate"] = 0.0
        plat["cost_per_store_visit"] = 0.0

    if "store_sales" in plat.columns:
        plat["store_sales_roas"] = np.where(
            plat["spend"] > 0, plat["store_sales"] / plat["spend"], 0
        )
    else:
        plat["store_sales_roas"] = 0.0

    # Total estimated ROAS across all three revenue streams
    plat["total_revenue_est"] = (
        plat.get("online_order_revenue", 0)
        + plat.get("reservation_revenue", 0)
        + plat.get("store_sales", 0)
    )
    plat["total_roas"] = np.where(
        plat["spend"] > 0, plat["total_revenue_est"] / plat["spend"], 0
    )

    return plat

def build_leaderboard(creative_metrics):
    """
    Build leaderboard with journey-aware performance scores
    tailored for Lazy Dog (restaurant outcomes).

    Scoring logic by journey_role:
    - Engagement: emphasize CTR + low CPC
    - Intent: emphasize micro-conversion rates (ATC, views) + online orders / reservations
    - Conversion: emphasize store visits, reservations, online orders + ROAS
    """

    leaderboard = creative_metrics.copy()

    # What metrics are available?
    has_conversions = 'CVR' in leaderboard.columns  # legacy, optional
    has_intent_metrics = any(
        col in leaderboard.columns
        for col in ['add_to_cart_rate', 'view_content_rate', 'page_view_rate']
    )

    has_online_orders = 'online_order_rate' in leaderboard.columns
    has_reservations = 'reservation_rate' in leaderboard.columns
    has_store_visits = 'store_visit_rate' in leaderboard.columns

    has_online_roas = 'online_order_roas' in leaderboard.columns
    has_reservation_roas = 'reservation_roas' in leaderboard.columns
    has_store_sales_roas = 'store_sales_roas' in leaderboard.columns

    # --- Base percentiles (higher = better) ---
    leaderboard['CTR_percentile'] = leaderboard['CTR'].rank(pct=True)
    leaderboard['CPC_percentile'] = leaderboard['CPC'].rank(pct=True)  # cost -> will invert later

    if has_conversions:
        leaderboard['CVR_percentile'] = leaderboard['CVR'].rank(pct=True)

    # Intent: average of add_to_cart / view_content / page_view rates
    if has_intent_metrics:
        intent_cols = [
            col for col in ['add_to_cart_rate', 'view_content_rate', 'page_view_rate']
            if col in leaderboard.columns
        ]
        leaderboard['intent_avg'] = leaderboard[intent_cols].mean(axis=1)
        leaderboard['intent_percentile'] = leaderboard['intent_avg'].rank(pct=True)

    # Restaurant-specific conversion metrics
    if has_online_orders:
        leaderboard['online_order_rate_percentile'] = leaderboard['online_order_rate'].rank(pct=True)

    if has_reservations:
        leaderboard['reservation_rate_percentile'] = leaderboard['reservation_rate'].rank(pct=True)

    if has_store_visits:
        leaderboard['store_visit_rate_percentile'] = leaderboard['store_visit_rate'].rank(pct=True)

    if has_online_roas:
        leaderboard['online_order_roas_percentile'] = leaderboard['online_order_roas'].rank(pct=True)

    if has_reservation_roas:
        leaderboard['reservation_roas_percentile'] = leaderboard['reservation_roas'].rank(pct=True)

    if has_store_sales_roas:
        leaderboard['store_sales_roas_percentile'] = leaderboard['store_sales_roas'].rank(pct=True)

    def get_p(row, col, default=0.5):
        val = row.get(col, np.nan)
        return default if pd.isna(val) else val

    def compute_journey_score(row):
        journey_role = row.get('journey_role', 'Engagement')

        ctr_p = get_p(row, 'CTR_percentile')
        cpc_p = get_p(row, 'CPC_percentile')          # lower CPC is better -> we use (1 - cpc_p)
        cvr_p = get_p(row, 'CVR_percentile') if has_conversions else 0.5
        intent_p = get_p(row, 'intent_percentile') if has_intent_metrics else 0.5

        online_rate_p = get_p(row, 'online_order_rate_percentile') if has_online_orders else 0.5
        res_rate_p = get_p(row, 'reservation_rate_percentile') if has_reservations else 0.5
        store_visit_p = get_p(row, 'store_visit_rate_percentile') if has_store_visits else 0.5

        online_roas_p = get_p(row, 'online_order_roas_percentile') if has_online_roas else 0.5
        res_roas_p = get_p(row, 'reservation_roas_percentile') if has_reservation_roas else 0.5
        store_roas_p = get_p(row, 'store_sales_roas_percentile') if has_store_sales_roas else 0.5

        # Weight lower CPC as (1 - percentile)
        cpc_score = 1 - cpc_p

        # --- Engagement: focus on cheap attention ---
        if journey_role == "Engagement":
            # CTR + low CPC, light touch of online orders if present
            return (
                0.5 * ctr_p +
                0.4 * cpc_score +
                0.1 * online_rate_p
            )

        # --- Intent: mid-funnel signals + soft conversions ---
        elif journey_role == "Intent":
            # micro-conversions + online orders + reservations
            return (
                0.25 * ctr_p +
                0.15 * cpc_score +
                0.30 * intent_p +
                0.15 * online_rate_p +
                0.15 * res_rate_p
            )

        # --- Conversion: restaurant outcomes + ROAS ---
        elif journey_role == "Conversion":
            # prioritize store visits / reservations / orders + ROAS
            return (
                0.10 * ctr_p +
                0.10 * cpc_score +
                0.20 * online_rate_p +
                0.20 * res_rate_p +
                0.20 * store_visit_p +
                0.10 * max(online_roas_p, res_roas_p, store_roas_p) +
                0.10 * cvr_p  # keep CVR if present
            )

        # --- Fallback: generic blend ---
        else:
            return (
                0.30 * ctr_p +
                0.20 * cpc_score +
                0.15 * intent_p +
                0.15 * online_rate_p +
                0.10 * res_rate_p +
                0.10 * store_visit_p
            )

    leaderboard['score'] = leaderboard.apply(compute_journey_score, axis=1)
    leaderboard = leaderboard.sort_values('score', ascending=False)

    return leaderboard

def compute_topic_summary(creative_metrics, has_conversions):
    if "topic" not in creative_metrics.columns:
        return pd.DataFrame()
    # simple aggregate to get you going
    agg = {
        "impressions": "sum",
        "clicks": "sum",
        "spend": "sum",
        "creative_name": "nunique",
    }
    if "ROAS" in creative_metrics.columns:
        agg["ROAS"] = "mean"
    topic_df = creative_metrics.groupby("topic").agg(agg).reset_index()
    topic_df.rename(columns={"creative_name": "num_creatives"}, inplace=True)
    topic_df["CTR"] = np.where(
        topic_df["impressions"] > 0,
        topic_df["clicks"] / topic_df["impressions"],
        0,
    )
    topic_df["CPC"] = np.where(
        topic_df["clicks"] > 0,
        topic_df["spend"] / topic_df["clicks"],
        0,
    )
    # optionally total_revenue_est, ROAS, etc.
    return topic_df


def compute_fatigue_metrics_for_creative(df, creative_name):
    """
    Compute fatigue metrics for a specific creative.
    """
    topic_data = df[df['creative_name'] == creative_name].copy()
    topic_data = topic_data.sort_values('date')

    return topic_data


def fit_simple_adjusted_model(df, outcome_metric):
    """
    Fit a simple linear regression model to adjust for context.
    Returns model results with adjusted scores per creative.
    """
    creative_agg = compute_aggregated_creative_metrics(df)

    if outcome_metric not in creative_agg.columns:
        return None, "Selected metric not available in the data"

    model_df = creative_agg[creative_agg[outcome_metric] > 0].copy()

    if len(model_df) < 10:
        return None, f"Insufficient data: only {len(model_df)} creatives with {outcome_metric} > 0. Need at least 10."

    model_df['log_impressions'] = np.log1p(model_df['impressions'])
    model_df['log_spend'] = np.log1p(model_df['spend'])

    feature_cols = ['log_impressions', 'log_spend']

    if 'placement' in df.columns:
        placement_dummies = pd.get_dummies(model_df['placement'], prefix='placement')
        model_df = pd.concat([model_df, placement_dummies], axis=1)
        feature_cols.extend(placement_dummies.columns.tolist())
    else:
        platform_dummies = pd.get_dummies(model_df['platform'], prefix='platform')
        model_df = pd.concat([model_df, platform_dummies], axis=1)
        feature_cols.extend(platform_dummies.columns.tolist())

    if 'format' in model_df.columns:
        format_dummies = pd.get_dummies(model_df['format'], prefix='format')
        model_df = pd.concat([model_df, format_dummies], axis=1)
        feature_cols.extend(format_dummies.columns.tolist())

    X = model_df[feature_cols].fillna(0)
    y = model_df[outcome_metric]

    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)
    model_df['predicted'] = predictions
    model_df['residual'] = y - predictions
    model_df['adjusted_score'] = model_df['residual']

    results = model_df[['creative_name', 'platform', 'campaign_name', outcome_metric, 
                       'predicted', 'adjusted_score', 'impressions', 'spend']].copy()
    results = results.sort_values('adjusted_score', ascending=False)

    return results, None


def show_welcome_screen():
    """
    Display welcome screen with instructions.
    """
    st.title("📊 Creative Performance Analysis")
    st.markdown("---")

    st.markdown("""
    ### Welcome to Creative Performance Analysis

    This app helps you analyze ad creative performance across platforms using a **3-layer journey framework**:
    
    - 📢 **Engagement Layer** (CTR, CPC) — Top-of-funnel attention drivers
    - 🛒 **Intent Layer** (micro-conversions) — Mid-funnel purchase intent builders
    - 💰 **Conversion Layer** (CVR, CPA, ROAS) — Bottom-of-funnel closers
    
    Each creative is automatically classified by its funnel role, helping you understand which assets drive awareness vs. which drive sales.

    #### Key Features
    - 📈 Journey-aware performance leaderboards
    - 🎯 Creative fatigue detection
    - 📊 Portfolio spend distribution by funnel layer
    - 🏷️ Topic insights with layer breakdowns

    #### Getting Started

    Upload a CSV file with your creative performance data. The file should contain:
    """)

    st.markdown("""
    **Required columns:**
    - `date` - Date of the performance data
    - `platform` - Advertising platform (e.g., Meta, Google, TikTok)
    - `campaign_name` - Campaign name
    - `creative_name` - Creative name
    - `impressions` - Number of impressions
    - `clicks` - Number of clicks
    - `spend` - Ad spend

    **Optional columns:**
    - `purchases` - Number of purchases
    - `add_to_carts` - Number of add-to-cart events
    - `view_content` - Number of content view events
    - `page_views` - Number of page views
    - `conversions` - Number of conversions (generic)
    - `revenue` - Revenue generated
    - `placement` - Ad placement (Feed, Stories, etc.)
    - `format` - Creative format (Image, Video, etc.)
    """)

    st.info("💡 **Tip:** Use the sidebar to upload your CSV file and start analyzing!")


def main():
    # SET FAVICON
    st.set_page_config(
    page_title="Lazy Dog Creative Performance",
    page_icon="lazy-dog-restaurant-favicon.png", 
    layout="wide"
)
    
    # SET LOGO & TITLE
    logo_path = "lazy_dog_logo.png"  # file is in repo root
        
    with st.sidebar:
        st.markdown("""
            <style>
            section[data-testid="stSidebar"] .css-1d391kg {
                padding-top: 0rem !important;
            }
            .sidebar-logo {
                margin-top: 0px !important;
                margin-bottom: 0px !important;
            }
            </style>
        """, unsafe_allow_html=True)
    
        st.markdown(
            f"<img class='sidebar-logo' src='{logo_path}' width='240'>",
            unsafe_allow_html=True
        )
    
        st.markdown("""
            <h2 style='margin-top: -10px; margin-bottom:10px; font-size:26px; font-weight:700;'>
                Media Performance Tracker
            </h2>
        """, unsafe_allow_html=True)
    
        st.markdown("<hr style='margin-top:0px;'>", unsafe_allow_html=True)
    
    
        with st.sidebar.expander("🔗 View Source Data (Google Sheet)"):
            st.markdown("""
            View or download the live Lazy Dog creative tracking sheet directly in Google Sheets.  
            """)
        
            st.markdown(
                f"[👉 Open Google Sheet](https://docs.google.com/spreadsheets/d/1JcSaWPiavp2_XLV8OVPlxvg8fJGTmNQTZotDeJLXous/edit?gid=1029811642#gid=1029811642)"
            )
        
            st.sidebar.markdown("---")

    # --- LOAD GOOGLE SHEET USING SERVICE ACCOUNT ---
    try:
        raw_df = load_google_sheet_to_df()
    except Exception as e:
        st.error(f"❌ Error loading Google Sheet via service account: {e}")
        st.stop()
    
    df = load_and_prepare_data(raw_df)


    if df is None:
        st.error("⚠️ Could not prepare data from Google Sheet")
        st.stop()

    st.sidebar.success(f"📡 Loaded live data ({len(df):,} rows)")
    st.sidebar.markdown("---")


    st.sidebar.subheader("🔍 Filters")

    # make sure we're using only valid dates
    date_series = df["date"].dropna()

    # --- DATE FILTERING ---
    min_date = df["date"].min()
    max_date = df["date"].max()
    
    # Sort once
    df_sorted = df.sort_values("date")
    
    # Determine "current" fiscal year from the most recent row
    current_fy = None
    df_fy = None
    period_order = []
    current_period = None
    
    if "fiscal_year" in df.columns:
        latest_row = df_sorted.iloc[-1]
        current_fy = latest_row["fiscal_year"]
    
        # restrict to this fiscal year only
        df_fy = df_sorted[df_sorted["fiscal_year"] == current_fy]
    
        if "period" in df_fy.columns and not df_fy.empty:
            # order periods within this fiscal year by their first date
            period_order = (
                df_fy.groupby("period")["date"]
                .min()
                .sort_values()
                .index
                .tolist()
            )
            current_period = df_fy.iloc[-1]["period"]
    
    # For week-based filters (can span years; that’s usually fine)
    week_starts = None
    if "week_start" in df.columns:
        week_starts = sorted(df["week_start"].dropna().unique())
    
    quick_choice = st.sidebar.radio(
        "Quick Date Filter",
        options=[
            "Custom Range",
            "This Week",
            "Last Week",
            "This Period",
            "Last Period",
            "This Fiscal Year",
        ],
        index=0,
        help="Use quick filters based on Week Start / Period / Fiscal Year, or choose a custom range.",
    )
    
    start_date = min_date
    end_date = max_date
    
    # ---- Custom Range ----
    if quick_choice == "Custom Range":
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = (min_date, max_date)
    
    # ---- This Week ----
    elif quick_choice == "This Week" and week_starts is not None and len(week_starts) >= 1:
        this_week_start = week_starts[-1]
        start_date = this_week_start
        end_date = this_week_start + pd.Timedelta(days=6)
        st.sidebar.info(f"Using week starting {start_date.date()}")
    
    # ---- Last Week ----
    elif quick_choice == "Last Week" and week_starts is not None and len(week_starts) >= 2:
        last_week_start = week_starts[-2]
        start_date = last_week_start
        end_date = last_week_start + pd.Timedelta(days=6)
        st.sidebar.info(f"Using week starting {start_date.date()}")
    
    # ---- This Period (within current fiscal year) ----
    elif quick_choice == "This Period" and df_fy is not None and current_period is not None:
        period_mask = (df["fiscal_year"] == current_fy) & (df["period"] == current_period)
        period_dates = df.loc[period_mask, "date"]
        if not period_dates.empty:
            start_date = period_dates.min()
            end_date = period_dates.max()
            st.sidebar.info(
                f"Using Period {current_period} in {current_fy} "
                f"({start_date.date()} – {end_date.date()})"
            )
    
    # ---- Last Period (within current fiscal year) ----
    elif (
        quick_choice == "Last Period"
        and df_fy is not None
        and current_period is not None
        and period_order
        and current_period in period_order
    ):
        idx = period_order.index(current_period)
        if idx > 0:  # there *is* a previous period in this FY
            last_period = period_order[idx - 1]
            last_period_mask = (df["fiscal_year"] == current_fy) & (df["period"] == last_period)
            last_period_dates = df.loc[last_period_mask, "date"]
            if not last_period_dates.empty:
                start_date = last_period_dates.min()
                end_date = last_period_dates.max()
                st.sidebar.info(
                    f"Using Last Period {last_period} in {current_fy} "
                    f"({start_date.date()} – {end_date.date()})"
                )
    
    # ---- This Fiscal Year ----
    elif quick_choice == "This Fiscal Year" and df_fy is not None and not df_fy.empty:
        fy_dates = df_fy["date"]
        start_date = fy_dates.min()
        end_date = fy_dates.max()
        st.sidebar.info(
            f"Using Fiscal Year {current_fy} "
            f"({start_date.date()} – {end_date.date()})"
        )
    
    # Fallback guards
    if pd.isna(start_date):
        start_date = min_date
    if pd.isna(end_date):
        end_date = max_date
    
    date_range_filter = (start_date, end_date)

    all_platforms = sorted(df['platform'].unique().tolist())
    selected_platforms = st.sidebar.multiselect(
        "Platform",
        options=all_platforms,
        default=all_platforms
    )

    all_campaigns = sorted(df['campaign_name'].unique().tolist())
    selected_campaigns = st.sidebar.multiselect(
        "Campaign",
        options=all_campaigns,
        default=all_campaigns
    )

    selected_objectives = None
    selected_objective_type = "All"

    if 'objective' in df.columns:
        all_objectives = sorted([o for o in df['objective'].dropna().unique().tolist()])
        selected_objectives = st.sidebar.multiselect(
            "Objective",
            options=all_objectives,
            default=all_objectives
        )

    if 'objective_type' in df.columns:
        objective_type_options = ["All", "Awareness", "Conversion", "Other"]
        selected_objective_type = st.sidebar.selectbox(
            "Objective Type (Awareness vs Conversion)",
            options=objective_type_options,
            index=0
        )

    selected_formats = None
    if 'format' in df.columns:
        all_formats = sorted([f for f in df['format'].dropna().unique().tolist()])
        if all_formats:
            selected_formats = st.sidebar.multiselect(
                "Format",
                options=all_formats,
                default=all_formats
            )

    selected_placements = None
    if 'placement' in df.columns:
        all_placements = sorted([p for p in df['placement'].dropna().unique().tolist()])
        if all_placements:
            selected_placements = st.sidebar.multiselect(
                "Placement",
                options=all_placements,
                default=all_placements
            )

    selected_topics = None
    if 'topic' in df.columns:
        all_topics = sorted([t for t in df['topic'].dropna().unique().tolist()])
        if len(all_topics) > 0:
            selected_topics = st.sidebar.multiselect(
                "Topic",
                options=all_topics,
                default=all_topics
            )
        else:
            st.sidebar.info("ℹ️ No topics found in data. Add a 'topic' column to enable topic filtering.")

    min_impressions = st.sidebar.slider(
        "Min Impressions per Creative",
        min_value=0,
        max_value=50000,
        value=5000,
        step=1000
    )

    has_conversions = 'conversions' in df.columns
    has_any_conversion_metrics = (
        has_conversions or
        'purchases' in df.columns or
        'add_to_carts' in df.columns or
        'view_content' in df.columns or
        'page_views' in df.columns or
        'revenue' in df.columns
    )

    if has_conversions:
        min_conversions = st.sidebar.slider(
            "Min Conversions per Creative",
            min_value=0,
            max_value=100,
            value=10,
            step=5
        )
    else:
        min_conversions = 0

    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Metrics")

    available_kpis = ['CTR', 'CPC', 'CPM']
    if has_conversions:
        available_kpis.extend(['CVR', 'CPA'])
    if 'purchases' in df.columns:
        available_kpis.extend(['purchase_rate', 'cost_per_purchase'])
    if 'add_to_carts' in df.columns:
        available_kpis.extend(['add_to_cart_rate', 'cost_per_add_to_cart'])
    if 'view_content' in df.columns:
        available_kpis.extend(['view_content_rate', 'cost_per_view_content'])
    if 'page_views' in df.columns:
        available_kpis.extend(['page_view_rate', 'cost_per_page_view'])
    if 'revenue' in df.columns:
        available_kpis.append('ROAS')

    selected_kpi = st.sidebar.selectbox(
        "Primary KPI",
        options=available_kpis,
        index=0
    )

    if has_any_conversion_metrics:
        show_conversion_metrics = st.sidebar.checkbox(
            "Show Conversion Metrics (Directional)",
            value=True
        )
    else:
        show_conversion_metrics = False

    filters = {
        'date_range': date_range_filter,
        'platforms': selected_platforms if selected_platforms else all_platforms,
        'campaigns': selected_campaigns if selected_campaigns else all_campaigns,
        'objectives': selected_objectives if selected_objectives else None,
        'objective_type': selected_objective_type,
        'topics': selected_topics if selected_topics else None,
        'formats': selected_formats if selected_formats else None,       # NEW
        'placements': selected_placements if selected_placements else None,  # NEW
        'min_impressions': min_impressions,
        'min_conversions': min_conversions
    }

    filtered_df = apply_global_filters(df, filters)

    if len(filtered_df) == 0:
        st.warning("⚠️ No data matches the current filters. Please adjust your filter settings.")
        st.stop()

    st.sidebar.info(f"📊 {len(filtered_df):,} rows after filtering")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📌 Executive Summary",
        "🟦 Platform Comparison",
        "🏆 Creative Leaderboard",
        "📉 Creative Detail & Fatigue",
        "🏷️ Topic Insights"
    ])

    # ---------- UPDATED EXEC SUMMARY TAB ----------
    with tab1:
        st.header("📌 Executive Summary")

        st.caption(
            "High-level view for marketing leaders: which platforms and creative themes "
            "are actually driving online orders, reservations, and store visits—and where to shift budget next."
        )

        # If you already have avg_res_check from the sidebar, use that instead of hardcoding:
        avg_res_check = 62.0

        # --- 1) Portfolio KPIs (restaurant outcomes) ---
        total_spend = filtered_df["spend"].sum()
        total_impressions = filtered_df["impressions"].sum()
        total_clicks = filtered_df["clicks"].sum()

        total_online_orders = filtered_df["online_orders"].sum() if "online_orders" in filtered_df.columns else 0
        total_reservations = filtered_df["reservations"].sum() if "reservations" in filtered_df.columns else 0
        total_store_visits = filtered_df["store_visits"].sum() if "store_visits" in filtered_df.columns else 0

        total_online_order_revenue = filtered_df["online_order_revenue"].sum() if "online_order_revenue" in filtered_df.columns else 0

        # Reservation revenue: use explicit column if present; otherwise estimate via avg check
        if "reservation_revenue" in filtered_df.columns:
            total_reservation_revenue = filtered_df["reservation_revenue"].sum()
        else:
            total_reservation_revenue = total_reservations * avg_res_check

        total_store_sales = filtered_df["store_sales"].sum() if "store_sales" in filtered_df.columns else 0

        total_estimated_revenue = (
            total_online_order_revenue
            + total_reservation_revenue
            + total_store_sales
        )

        overall_ctr = total_clicks / total_impressions if total_impressions > 0 else 0
        overall_cpc = total_spend / total_clicks if total_clicks > 0 else 0
        overall_cpm = total_spend / total_impressions * 1000 if total_impressions > 0 else 0
        blended_roas = total_estimated_revenue / total_spend if total_spend > 0 else 0

        avg_cost_per_store_visit = (
            total_spend / total_store_visits if total_store_visits > 0 else 0
        )

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Spend", f"${total_spend:,.0f}")
        with col2:
            st.metric("Impressions", f"{total_impressions:,.0f}")
        with col3:
            st.metric("Clicks", f"{total_clicks:,.0f}")
        with col4:
            if total_online_orders > 0:
                st.metric("Online Orders", f"{int(total_online_orders):,}")
            else:
                st.metric("Online Orders", "N/A")
        with col5:
            if total_reservations > 0:
                st.metric("Reservations", f"{int(total_reservations):,}")
            else:
                st.metric("Reservations", "N/A")

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("CTR", f"{overall_ctr:.2%}")
        with col2:
            st.metric("CPC", f"${overall_cpc:.2f}")
        with col3:
            st.metric("CPM", f"${overall_cpm:.2f}")
        with col4:
            if total_store_visits > 0:
                st.metric("Store Visits", f"{int(total_store_visits):,}")
            else:
                st.metric("Store Visits", "N/A")
        with col5:
            if total_spend > 0:
                st.metric("Blended ROAS (All Revenue)", f"{blended_roas:.2f}x")
            else:
                st.metric("Blended ROAS (All Revenue)", "N/A")

        st.markdown("---")

        # --- 2) Prep aggregates once ---
        creative_metrics = compute_aggregated_creative_metrics(filtered_df)
        # Use Lazy Dog–specific platform metrics
        platform_metrics = compute_platform_metrics_lazy_dog(
            filtered_df, avg_res_check=avg_res_check
        )
        topic_summary = compute_topic_summary(creative_metrics, has_conversions)

        # --- 3) Light platform summary (details live in Platform tab) ---
        st.subheader("🔍 Platform snapshot")

        if len(platform_metrics) == 0:
            st.info("No platform data available after filters.")
        else:
            total_platform_spend = platform_metrics["spend"].sum()
            cols = st.columns(min(4, len(platform_metrics)))

            for idx, row in platform_metrics.iterrows():
                col = cols[idx % len(cols)]
                with col:
                    spend_share = (
                        row["spend"] / total_platform_spend
                        if total_platform_spend > 0
                        else 0
                    )

                    st.markdown(f"#### {row['platform']}")
                    st.metric("Share of Spend", f"{spend_share:.1%}")
                    st.metric("CPC", f"${row['CPC']:.2f}")
                    # online orders
                    if "online_orders" in row and row["online_orders"] > 0:
                        st.metric(
                            "Online Orders (Cost)",
                            f"{int(row['online_orders'])} | ${row['cost_per_online_order']:.2f}",
                        )
                    # reservations
                    if "reservations" in row and row["reservations"] > 0:
                        st.metric(
                            "Reservations (Cost)",
                            f"{int(row['reservations'])} | ${row['cost_per_reservation']:.2f}",
                        )
                    # store visits
                    if "store_visits" in row and row["store_visits"] > 0:
                        st.metric(
                            "Store Visits (Cost)",
                            f"{int(row['store_visits'])} | ${row['cost_per_store_visit']:.2f}",
                        )
                    st.metric("Total ROAS", f"{row['total_roas']:.2f}x")

        st.markdown("---")

        # --- 4) Top creative themes (topics) ---
        st.subheader("🏷️ Top creative themes")

        if topic_summary.empty:
            st.info(
                "No 'topic' data found. Add a `topic` column (e.g., 'Product Demo', 'UGC Testimonial', "
                "'Brand Story') to unlock theme-level insights."
            )
        else:
            # Prefer revenue-based sort if available, else fall back to CTR
            sort_metric = None
            for candidate in ["total_revenue_est", "ROAS", "CTR"]:
                if candidate in topic_summary.columns:
                    sort_metric = candidate
                    break
            if sort_metric is None:
                sort_metric = "CTR"

            top_topics = topic_summary.sort_values(sort_metric, ascending=False).head(5)

            display_topics = top_topics.copy()
            if "CTR" in display_topics.columns:
                display_topics["CTR"] = display_topics["CTR"] * 100
            if "CVR" in display_topics.columns:
                display_topics["CVR"] = display_topics["CVR"] * 100

            cols_for_display = [
                c
                for c in [
                    "topic",
                    "num_creatives",
                    "impressions",
                    "spend",
                    "CTR",
                    "CPC",
                    "ROAS",
                ]
                if c in display_topics.columns
            ]

            topic_col_config = {
                "topic": st.column_config.TextColumn("Theme"),
                "num_creatives": st.column_config.NumberColumn(
                    "# Creatives", format="%d"
                ),
                "impressions": st.column_config.NumberColumn(
                    "Impressions", format="%,d"
                ),
                "spend": st.column_config.NumberColumn(
                    "Spend", format="$ %,.0f"
                ),
            }
            if "CTR" in display_topics.columns:
                topic_col_config["CTR"] = st.column_config.NumberColumn(
                    "CTR", format="%.2f %%"
                )
            if "CPC" in display_topics.columns:
                topic_col_config["CPC"] = st.column_config.NumberColumn(
                    "CPC", format="$ %.2f"
                )
            if "ROAS" in display_topics.columns:
                topic_col_config["ROAS"] = st.column_config.NumberColumn(
                    "ROAS", format="%.2f x"
                )

            st.dataframe(
                display_topics[cols_for_display],
                width="stretch",
                column_config=topic_col_config,
            )

        st.markdown("---")

        # --- 5) Simple auto-insights / recommendations ---
        st.subheader("🎯 Key insights & recommendations")

        insights = []

        # Platform insight: best store-visit / reservation / online-order performance
        if len(platform_metrics) > 0:
            # Best store visit platform
            if (platform_metrics["store_visits"] > 0).any():
                best_store = platform_metrics[platform_metrics["store_visits"] > 0].nsmallest(
                    1, "cost_per_store_visit"
                ).iloc[0]
                insights.append(
                    f"• **{best_store['platform']}** is your most efficient in-restaurant driver "
                    f"(Store Visits at ≈ ${best_store['cost_per_store_visit']:.2f} each)."
                )

            # Best reservation platform
            if (platform_metrics["reservations"] > 0).any():
                best_res = platform_metrics[platform_metrics["reservations"] > 0].nsmallest(
                    1, "cost_per_reservation"
                ).iloc[0]
                insights.append(
                    f"• **{best_res['platform']}** is strongest on reservations "
                    f"(≈ ${best_res['cost_per_reservation']:.2f} per reservation)."
                )

            # Best blended ROAS
            if (platform_metrics["total_roas"] > 0).any():
                best_roas = platform_metrics[platform_metrics["total_roas"] > 0].nlargest(
                    1, "total_roas"
                ).iloc[0]
                insights.append(
                    f"• **{best_roas['platform']}** delivers the best blended ROAS "
                    f"across orders, reservations, and visits (≈ {best_roas['total_roas']:.2f}x)."
                )

        # Topic insight: best theme by ROAS or CTR
        if not topic_summary.empty and "topic" in topic_summary.columns:
            metric_for_theme = "ROAS" if "ROAS" in topic_summary.columns else "CTR"
            top_theme = topic_summary.sort_values(metric_for_theme, ascending=False).iloc[0]
            theme_label = top_theme["topic"]
            if metric_for_theme == "ROAS":
                insights.append(
                    f"• Creative theme **'{theme_label}'** shows the strongest ROAS; consider scaling similar concepts."
                )
            else:
                insights.append(
                    f"• Creative theme **'{theme_label}'** leads on CTR; use it to fuel awareness and retargeting pools."
                )

        # Journey role coverage: high-level note only
        if "journey_role" in creative_metrics.columns:
            jr = creative_metrics.groupby("journey_role")["spend"].sum().reset_index()
            total_jr_spend = jr["spend"].sum()
            if total_jr_spend > 0:
                jr["share"] = jr["spend"] / total_jr_spend
                for role, label in [
                    ("Engagement", "top-of-funnel"),
                    ("Intent", "mid-funnel"),
                    ("Conversion", "bottom-of-funnel"),
                ]:
                    if role in jr["journey_role"].values:
                        row = jr[jr["journey_role"] == role].iloc[0]
                        insights.append(
                            f"• About **{row['share']:.0%}** of spend is currently in **{label}** creatives ({role})."
                        )

        if not insights:
            st.write(
                "No strong auto-insights yet. Try expanding the date range or ensuring conversion data exists."
            )
        else:
            for line in insights:
                st.markdown(line)

        st.caption(
            "Use the other tabs for deeper analysis: platform comparison, leaderboard, creative fatigue, and detailed topic insights."
        )
    # ---------- END EXEC SUMMARY ----------

    # ---------- PLATFORM COMPARISON TAB ----------
    with tab2:
        st.header("🟦 Platform Comparison")

        st.caption(
            "Compare how each platform performs on traffic cost, online orders, reservations, and store visits."
        )

        avg_res_check = 62.0  # or use the same sidebar input here

        platform_metrics = compute_platform_metrics_lazy_dog(
            filtered_df, avg_res_check=avg_res_check
        )

        if len(platform_metrics) == 0:
            st.info("No platform data available with the current filters.")
        else:
            total_spend_pf = platform_metrics["spend"].sum()

            cols = st.columns(min(4, len(platform_metrics)))

            for idx, row in platform_metrics.iterrows():
                col = cols[idx % len(cols)]
                with col:
                    spend_share = (
                        row["spend"] / total_spend_pf if total_spend_pf > 0 else 0
                    )

                    st.markdown(f"### {row['platform']}")
                    st.metric("Share of Spend", f"{spend_share:.1%}")
                    st.metric("CPC", f"${row['CPC']:.2f}")

                    # Online orders
                    if "online_orders" in row and row["online_orders"] > 0:
                        st.metric(
                            "Online Orders",
                            f"{int(row['online_orders'])} | ${row['cost_per_online_order']:.2f} / order",
                        )
                        if "online_order_roas" in row:
                            st.metric(
                                "Online Order ROAS",
                                f"{row['online_order_roas']:.2f}x",
                            )

                    # Reservations
                    if "reservations" in row and row["reservations"] > 0:
                        st.metric(
                            "Reservations",
                            f"{int(row['reservations'])} | ${row['cost_per_reservation']:.2f} / res",
                        )
                        st.metric(
                            "Reservation ROAS",
                            f"{row['reservation_roas']:.2f}x",
                        )

                    # Store visits
                    if "store_visits" in row and row["store_visits"] > 0:
                        st.metric(
                            "Store Visits",
                            f"{int(row['store_visits'])} | ${row['cost_per_store_visit']:.2f} / visit",
                        )
                        st.metric(
                            "Store Sales ROAS",
                            f"{row['store_sales_roas']:.2f}x",
                        )

                    st.metric("Blended ROAS (All Revenue)", f"{row['total_roas']:.2f}x")

            st.markdown("---")
            st.subheader("Spend vs CPC (color = Blended ROAS)")

            pf_chart = platform_metrics.copy()
            pf_chart["spend_share"] = np.where(
                total_spend_pf > 0, pf_chart["spend"] / total_spend_pf, 0
            )

            fig_pf = px.scatter(
                pf_chart,
                x="spend_share",
                y="CPC",
                size="spend",
                color="total_roas",
                hover_data=[
                    "impressions",
                    "clicks",
                    "online_orders",
                    "reservations",
                    "store_visits",
                ],
                labels={
                    "spend_share": "Share of Spend",
                    "CPC": "CPC ($)",
                    "total_roas": "Blended ROAS",
                },
                title="Platform Efficiency: CPC vs Spend (color = Blended ROAS)",
                color_continuous_scale="RdYlGn",
            )
            fig_pf.update_xaxes(tickformat=".1%")
            st.plotly_chart(fig_pf, width="stretch")
    # ---------- END PLATFORM COMPARISON ----------

    # ---------- CREATIVE LEADERBOARD TAB ----------
    with tab3:
        st.header("🏆 Creative Leaderboard")

        creative_metrics = compute_aggregated_creative_metrics(filtered_df)
        leaderboard = build_leaderboard(creative_metrics)

        st.info("💡 Scoring is journey-aware: **Engagement** creatives are scored primarily on CTR/CPC, **Intent** creatives on micro-conversion rates, and **Conversion** creatives on CVR/CPA.")
        
        journey_role_filter = st.selectbox(
            "Filter by Journey Role",
            options=["All", "Engagement", "Intent", "Conversion"],
            index=0,
            help="Filter creatives by their funnel position"
        )
        
        if journey_role_filter != "All":
            leaderboard = leaderboard[leaderboard['journey_role'] == journey_role_filter]

        st.subheader(f"Top Performing Creatives ({len(leaderboard)} total)")

        display_cols = ['creative_name', 'journey_role', 'platform', 'campaign_name']

        if 'topic' in leaderboard.columns:
            display_cols.append('topic')

        if 'objective' in leaderboard.columns:
            display_cols.append('objective')

        if 'format' in leaderboard.columns:
            display_cols.append('format')

        display_cols.extend(['impressions', 'clicks', 'spend', 'CTR', 'CPC', 'CPM'])

        if has_conversions and show_conversion_metrics:
            display_cols.extend(['conversions', 'CVR', 'CPA'])

        if 'revenue' in leaderboard.columns and show_conversion_metrics:
            display_cols.extend(['ROAS'])

        if 'purchases' in leaderboard.columns and show_conversion_metrics:
            display_cols.extend(['purchases', 'purchase_rate', 'cost_per_purchase'])

        if 'add_to_carts' in leaderboard.columns and show_conversion_metrics:
            display_cols.extend(['add_to_carts', 'add_to_cart_rate', 'cost_per_add_to_cart'])

        if 'view_content' in leaderboard.columns and show_conversion_metrics:
            display_cols.extend(['view_content', 'view_content_rate', 'cost_per_view_content'])

        if 'page_views' in leaderboard.columns and show_conversion_metrics:
            display_cols.extend(['page_views', 'page_view_rate', 'cost_per_page_view'])

        display_cols.extend(['age_in_days_max', 'total_days_active', 'score'])

        display_df = leaderboard[display_cols].copy()

        display_df['CTR'] = display_df['CTR'] * 100
        if 'CVR' in display_df.columns:
            display_df['CVR'] = display_df['CVR'] * 100
        if 'purchase_rate' in display_df.columns:
            display_df['purchase_rate'] = display_df['purchase_rate'] * 100
        if 'add_to_cart_rate' in display_df.columns:
            display_df['add_to_cart_rate'] = display_df['add_to_cart_rate'] * 100
        if 'view_content_rate' in display_df.columns:
            display_df['view_content_rate'] = display_df['view_content_rate'] * 100
        if 'page_view_rate' in display_df.columns:
            display_df['page_view_rate'] = display_df['page_view_rate'] * 100

        column_config = {
            'journey_role': st.column_config.TextColumn('Journey Role'),
            'CTR': st.column_config.NumberColumn('CTR', format="%.3f %%"),
            'CPC': st.column_config.NumberColumn('CPC', format="$ %.2f"),
            'CPM': st.column_config.NumberColumn('CPM', format="$ %.2f"),
            'score': st.column_config.NumberColumn('Score', format="%.3f"),
            'impressions': st.column_config.NumberColumn('Impressions', format="%,d"),
            'clicks': st.column_config.NumberColumn('Clicks', format="%,d"),
            'spend': st.column_config.NumberColumn('Spend', format="$ %,.2f"),
        }

        if 'CVR' in display_df.columns:
            column_config['CVR'] = st.column_config.NumberColumn('CVR', format="%.3f %%")
        if 'CPA' in display_df.columns:
            column_config['CPA'] = st.column_config.NumberColumn('CPA', format="$ %.2f")
        if 'ROAS' in display_df.columns:
            column_config['ROAS'] = st.column_config.NumberColumn('ROAS', format="%.2f x")

        if 'conversions' in display_df.columns:
            column_config['conversions'] = st.column_config.NumberColumn('Conversions', format="%,d")

        if 'purchases' in display_df.columns:
            column_config['purchases'] = st.column_config.NumberColumn('Purchases', format="%,d")
        if 'purchase_rate' in display_df.columns:
            column_config['purchase_rate'] = st.column_config.NumberColumn('Purchase Rate', format="%.3f %%")
        if 'cost_per_purchase' in display_df.columns:
            column_config['cost_per_purchase'] = st.column_config.NumberColumn('Cost/Purchase', format="$ %.2f")

        if 'add_to_carts' in display_df.columns:
            column_config['add_to_carts'] = st.column_config.NumberColumn('Add to Carts', format="%,d")
        if 'add_to_cart_rate' in display_df.columns:
            column_config['add_to_cart_rate'] = st.column_config.NumberColumn('Add to Cart Rate', format="%.3f %%")
        if 'cost_per_add_to_cart' in display_df.columns:
            column_config['cost_per_add_to_cart'] = st.column_config.NumberColumn('Cost/Add to Cart', format="$ %.2f")

        if 'view_content' in display_df.columns:
            column_config['view_content'] = st.column_config.NumberColumn('View Content', format="%,d")
        if 'view_content_rate' in display_df.columns:
            column_config['view_content_rate'] = st.column_config.NumberColumn('View Content Rate', format="%.3f %%")
        if 'cost_per_view_content' in display_df.columns:
            column_config['cost_per_view_content'] = st.column_config.NumberColumn('Cost/View Content', format="$ %.2f")

        if 'page_views' in display_df.columns:
            column_config['page_views'] = st.column_config.NumberColumn('Page Views', format="%,d")
        if 'page_view_rate' in display_df.columns:
            column_config['page_view_rate'] = st.column_config.NumberColumn('Page View Rate', format="%.3f %%")
        if 'cost_per_page_view' in display_df.columns:
            column_config['cost_per_page_view'] = st.column_config.NumberColumn('Cost/Page View', format="$ %.2f")

        # --- NEW: pick a creative to sync with Detail tab ---
        if "selected_topic" not in st.session_state:
            first_row = leaderboard.iloc[0]
            st.session_state["selected_topic"] = first_row.get("topic", first_row["creative_name"])
        
        selected_from_leaderboard = st.selectbox(
            "Select a creative to analyze in the Topic Detail & Fatigue tab",
            options=leaderboard["creative_name"].tolist(),
            key="leaderboard_creative_select",
        )
        
        # map creative -> topic (fallback to creative name if no topic column)
        if "topic" in leaderboard.columns:
            topic_for_creative = (
                leaderboard.loc[leaderboard["creative_name"] == selected_from_leaderboard, "topic"]
                .iloc[0]
            )
        else:
            topic_for_creative = selected_from_leaderboard
        
        st.session_state["selected_topic"] = topic_for_creative

        st.dataframe(
            format_currency_columns(display_df.copy()),
            width="stretch",
            height=400,
            column_config=column_config
        )
        
        csv_leaderboard = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Leaderboard CSV",
            data=csv_leaderboard,
            file_name="creative_leaderboard.csv",
            mime="text/csv"
        )

        st.markdown("---")
        st.subheader("Creative Performance Scatter Plot")

        y_axis_metric = 'CVR' if has_conversions and 'CVR' in leaderboard.columns else 'CTR'

        fig = px.scatter(
            leaderboard,
            x='CPC',
            y=y_axis_metric,
            size='spend',
            color='score',
            hover_data=['creative_name', 'campaign_name', 'platform', 'impressions', 'clicks'],
            title=f"Creative Performance: {y_axis_metric} vs CPC (size = spend)",
            labels={'CPC': 'Cost Per Click ($)', y_axis_metric: y_axis_metric},
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, width="stretch")

        # ---------- END PLATFORM COMPARISON ----------

        # ---------- CREATIVE DETAIL & FATIGUE ----------
    with tab4:
        st.header("📉 Creative Detail & Fatigue Analysis")

        topic_list = sorted(
            t for t in filtered_df['topic'].unique().tolist()
            if t and str(t).strip() != ""
        )
        if len(topic_list) == 0:
            st.warning("No topics available with current filters.")
            st.stop()

        # ensure we have a valid selected_topic in session_state
        if "selected_topic" not in st.session_state or st.session_state["selected_topic"] not in topic_list:
            st.session_state["selected_topic"] = topic_list[0]
        
        selected_topic = st.selectbox(
            "Select Topic to Analyze",
            options=topic_list,
            index=topic_list.index(st.session_state["selected_topic"]),
            key="topic_detail_select",   # new key just for this widget
        )
        
        # keep global selected_topic in sync
        st.session_state["selected_topic"] = selected_topic


        # ---- Topic-level data + summary ----
        AVG_RES_CHECK = 62.0  # keep in sync with rest of app
        
        # all rows for this topic
        topic_data = filtered_df[filtered_df["topic"] == selected_topic].copy()
        topic_data = topic_data.sort_values("date")
        
        if topic_data.empty:
            st.warning("No data for this topic with current filters.")
            st.stop()
        
        # aggregate to topic level (totals)
        agg_dict = {
            "impressions": "sum",
            "clicks": "sum",
            "spend": "sum",
        }
        
        for col in [
            "online_orders",
            "online_order_revenue",
            "reservations",
            "reservation_revenue",
            "store_visits",
            "store_sales",
        ]:
            if col in topic_data.columns:
                agg_dict[col] = "sum"
        
        topic_summary = (
            topic_data.groupby("topic")
            .agg(agg_dict)
            .reset_index()
            .iloc[0]
        )
        
        # extra display fields
        topic_summary["platforms"] = ", ".join(
            sorted(topic_data["platform"].dropna().unique())
        )
        topic_summary["days_active"] = topic_data["date"].nunique()
        
        # ---- derived funnel metrics ----
        impr = topic_summary["impressions"]
        clicks = topic_summary["clicks"]
        spend = topic_summary["spend"]
        
        topic_summary["CTR"] = clicks / impr if impr > 0 else 0
        topic_summary["CPC"] = spend / clicks if clicks > 0 else 0
        topic_summary["CPM"] = spend / impr * 1000 if impr > 0 else 0
        
        online_orders = topic_summary.get("online_orders", 0)
        reservations = topic_summary.get("reservations", 0)
        
        topic_summary["online_order_rate"] = (
            online_orders / clicks if clicks > 0 and online_orders > 0 else 0
        )
        topic_summary["reservation_rate"] = (
            reservations / clicks if clicks > 0 and reservations > 0 else 0
        )
        
        # revenues + blended ROAS
        online_rev = topic_summary.get("online_order_revenue", 0.0)
        
        if "reservation_revenue" in topic_summary.index:
            res_rev = topic_summary["reservation_revenue"]
        else:
            res_rev = reservations * AVG_RES_CHECK
        
        store_sales = topic_summary.get("store_sales", 0.0)
        
        total_rev = online_rev + res_rev + store_sales
        topic_summary["total_revenue_est"] = total_rev
        topic_summary["total_roas"] = total_rev / spend if spend > 0 else 0
        
        st.markdown("---")
        st.subheader("Topic Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Platform(s)", topic_summary["platforms"])
        with col2:
            st.metric("Total Spend", f"${topic_summary['spend']:,.2f}")
        with col3:
            st.metric("Impressions", f"{topic_summary['impressions']:,.0f}")
        with col4:
            st.metric("Days Active", f"{topic_summary['days_active']:.0f}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("CTR", f"{topic_summary['CTR']:.3%}")
        with col2:
            st.metric("CPC", f"${topic_summary['CPC']:.2f}")
        with col3:
            if topic_summary.get("online_order_rate", 0) > 0:
                st.metric("CVR (Online Order)", f"{topic_summary['online_order_rate']:.3%}")
            else:
                st.metric("CVR (Online Order)", "N/A")
        with col4:
            if topic_summary.get("reservation_rate", 0) > 0:
                st.metric("CVR (Reservation)", f"{topic_summary['reservation_rate']:.3%}")
            else:
                st.metric("CVR (Reservation)", "N/A")
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Blended ROAS (All Revenue)",
                f"{topic_summary['total_roas']:.2f}x" if topic_summary["spend"] > 0 else "N/A",
            )
        with col2:
            st.metric(
                "Total Estimated Revenue",
                f"${topic_summary['total_revenue_est']:,.0f}",
            )


        # ---- Daily aggregation for fatigue charts (smooth lines) ----
        daily_agg_dict = {
            "impressions": "sum",
            "clicks": "sum",
            "spend": "sum",
        }
        
        for col in ["online_orders", "reservations"]:
            if col in topic_data.columns:
                daily_agg_dict[col] = "sum"
        
        topic_daily = (
            topic_data.groupby("date")
            .agg(daily_agg_dict)
            .reset_index()
            .sort_values("date")
        )
        
        # per-day derived metrics
        topic_daily["CTR"] = np.where(
            topic_daily["impressions"] > 0,
            topic_daily["clicks"] / topic_daily["impressions"],
            0,
        )
        topic_daily["CPC"] = np.where(
            topic_daily["clicks"] > 0,
            topic_daily["spend"] / topic_daily["clicks"],
            0,
        )
        
        if "online_orders" in topic_daily.columns:
            topic_daily["online_order_rate"] = np.where(
                topic_daily["clicks"] > 0,
                topic_daily["online_orders"] / topic_daily["clicks"],
                0,
            )
        
        if "reservations" in topic_daily.columns:
            topic_daily["reservation_rate"] = np.where(
                topic_daily["clicks"] > 0,
                topic_daily["reservations"] / topic_daily["clicks"],
                0,
            )
        
        topic_daily["cumulative_impressions"] = topic_daily["impressions"].cumsum()
        topic_daily["age_in_days"] = (
            topic_daily["date"] - topic_daily["date"].min()
        ).dt.days
        
        st.markdown("---")
        st.subheader("Fatigue Analysis")

        fatigue_kpi_options = ["CTR", "CPC"]
        if "online_order_rate" in topic_daily.columns:
            fatigue_kpi_options.append("online_order_rate")
        if "reservation_rate" in topic_daily.columns:
            fatigue_kpi_options.append("reservation_rate")

        fatigue_kpi = st.selectbox(
            "Select KPI for Fatigue Analysis",
            options=fatigue_kpi_options,
            index=0
        )

        secondary_kpi = st.selectbox(
            "Optional secondary KPI (overlay)",
            options=["None"] + fatigue_kpi_options,
            index=0
        )

        if len(topic_daily) >= 3:
            age_days = topic_daily['age_in_days'].values
            kpi_values = topic_daily[fatigue_kpi].values
        
            valid_indices = ~np.isnan(kpi_values) & ~np.isinf(kpi_values)
            age_days_clean = age_days[valid_indices]
            kpi_values_clean = kpi_values[valid_indices]
        
            if len(age_days_clean) >= 3:
                coeffs = np.polyfit(age_days_clean, kpi_values_clean, 1)
                slope = coeffs[0]
                trend_line = coeffs[0] * age_days_clean + coeffs[1]
        
                fig = go.Figure()
        
                # primary KPI (aggregated per day)
                fig.add_trace(go.Scatter(
                    x=topic_daily['date'],
                    y=topic_daily[fatigue_kpi],
                    mode='lines+markers',
                    name=f'Actual {fatigue_kpi}',
                    line=dict(width=2),
                    marker=dict(size=6),
                    yaxis="y1"
                ))
        
                # trend line for primary
                fig.add_trace(go.Scatter(
                    x=topic_daily['date'].values[valid_indices],
                    y=trend_line,
                    mode='lines',
                    name='Trend Line',
                    line=dict(width=2, dash='dash'),
                    yaxis="y1"
                ))
        
                # optional secondary KPI
                if secondary_kpi != "None":
                    fig.add_trace(go.Scatter(
                        x=topic_daily['date'],
                        y=topic_daily[secondary_kpi],
                        mode='lines+markers',
                        name=f'{secondary_kpi}',
                        line=dict(width=2, dash='dot'),
                        marker=dict(size=5),
                        yaxis="y2"
                    ))
        
                    fig.update_layout(
                        yaxis=dict(title=fatigue_kpi),
                        yaxis2=dict(
                            title=secondary_kpi,
                            overlaying='y',
                            side='right'
                        )
                    )
                else:
                    fig.update_layout(
                        yaxis=dict(title=fatigue_kpi)
                    )
        
                fig.update_layout(
                    title=f"{fatigue_kpi} Over Time for {selected_topic}",
                    xaxis_title="Date",
                    hovermode='x unified'
                )
        
                # ---- PRIMARY AXIS formatting ----
                if fatigue_kpi in RATE_METRICS_SET:
                    fig.update_layout(yaxis=dict(tickformat=".2%"))
                elif fatigue_kpi in CURRENCY_METRICS_SET:
                    fig.update_layout(yaxis=dict(tickprefix="$"))
                else:
                    fig.update_layout(yaxis=dict(tickformat=None))
        
                # ---- SECONDARY AXIS formatting ----
                if secondary_kpi != "None":
                    if secondary_kpi in RATE_METRICS_SET:
                        fig.update_layout(yaxis2=dict(tickformat=".2%"))
                    elif secondary_kpi in CURRENCY_METRICS_SET:
                        fig.update_layout(yaxis2=dict(tickprefix="$"))
                    else:
                        fig.update_layout(yaxis2=dict(tickformat=None))

                st.plotly_chart(fig, width="stretch")
        
                rate_metrics = ['CTR', 'online_order_rate', 'reservation_rate', 'purchase_rate',
                                'add_to_cart_rate', 'view_content_rate', 'page_view_rate']
                fatigue_threshold = -0.0001 if fatigue_kpi in rate_metrics else 0.01
                min_days_for_fatigue = 7
                min_impressions_for_fatigue = 10000
        
                total_impressions = topic_summary['impressions']
                total_days = topic_summary['days_active']   # <– changed from total_days_active
        
                is_fatiguing = (
                    slope < fatigue_threshold and
                    total_days >= min_days_for_fatigue and
                    total_impressions >= min_impressions_for_fatigue
                )
        
                if is_fatiguing:
                    st.error(f"🔴 **Likely Fatigue Detected** - {fatigue_kpi} is declining over time (slope: {slope:.6f})")
                else:
                    st.success(f"🟢 **No Clear Fatigue Signal** - {fatigue_kpi} is stable or improving (slope: {slope:.6f})")
            else:
                st.warning("Not enough valid data points to compute trend.")
        else:
            st.warning("Not enough data points for fatigue analysis (minimum 3 days required).")

        st.markdown("---")
        st.subheader(f"{fatigue_kpi} vs Cumulative Impressions")

        if len(topic_daily) >= 3:
            fig = go.Figure()
        
            fig.add_trace(go.Scatter(
                x=topic_daily['cumulative_impressions'],
                y=topic_daily[fatigue_kpi],
                mode='lines+markers',
                name=f'{fatigue_kpi}',
                line=dict(width=2),
                marker=dict(size=6)
            ))
        
            cum_impr = topic_daily['cumulative_impressions'].values
            kpi_vals = topic_daily[fatigue_kpi].values
        
            valid_idx = ~np.isnan(kpi_vals) & ~np.isinf(kpi_vals)
            if np.sum(valid_idx) >= 3:
                coeffs_cum = np.polyfit(cum_impr[valid_idx], kpi_vals[valid_idx], 1)
                trend_cum = coeffs_cum[0] * cum_impr[valid_idx] + coeffs_cum[1]
        
                fig.add_trace(go.Scatter(
                    x=cum_impr[valid_idx],
                    y=trend_cum,
                    mode='lines',
                    name='Trend Line',
                    line=dict(width=2, dash='dash')
                ))
        
            fig.update_layout(
                title=f"{fatigue_kpi} vs Cumulative Impressions (Topic: {selected_topic})",
                xaxis_title="Cumulative Impressions",
                yaxis_title=fatigue_kpi,
                hovermode='x unified'
            )
            if fatigue_kpi in RATE_METRICS:
                fig.update_yaxes(tickformat=".2%")
        
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("Not enough data points for cumulative impression analysis.")

    # ---------- END CREATIVE FATIGUE & DETAIL  ----------

    # ---------- TOPIC INSIGHTS TAB ----------
    with tab5:
        st.header("🏷️ Topic Insights")

        creative_metrics = compute_aggregated_creative_metrics(filtered_df)

        if 'topic' not in creative_metrics.columns or creative_metrics['topic'].isna().all():
            st.warning("⚠️ No topic data available. Add a 'topic' column to your CSV to enable topic-based analysis.")
            st.info("💡 **Tip:** Topics help you group creatives by theme or content type (e.g., 'Product Demo', 'UGC Content', 'Brand Messaging').")
            st.stop()

        st.info("💡 Analyze creative performance by topic to identify which content themes drive the best results.")

        st.markdown("---")
        st.subheader("CTR vs CPC Performance by Topic")
        
        # only rows with a topic
        topic_level = creative_metrics[creative_metrics["topic"].notna()].copy()
        
        if len(topic_level) == 0:
            st.warning("No data available with topics after filtering.")
        else:
            # aggregate creatives up to topic level
            topic_summary = (
                topic_level
                .groupby("topic")
                .agg(
                    impressions=("impressions", "sum"),
                    clicks=("clicks", "sum"),
                    spend=("spend", "sum"),
                    num_creatives=("creative_name", "nunique"),
                )
                .reset_index()
            )
        
            # compute topic-level CTR & CPC
            topic_summary["CTR"] = np.where(
                topic_summary["impressions"] > 0,
                topic_summary["clicks"] / topic_summary["impressions"],
                0,
            )
            topic_summary["CPC"] = np.where(
                topic_summary["clicks"] > 0,
                topic_summary["spend"] / topic_summary["clicks"],
                0,
            )
        
            fig = px.scatter(
                topic_summary,
                x="CPC",
                y="CTR",
                size="spend",
                color="topic",
                hover_data=[
                    "impressions",
                    "clicks",
                    "spend",
                    "num_creatives",
                ],
                title="CTR vs CPC Performance by Topic (bubble size = spend)",
                labels={
                    "CPC": "Cost Per Click ($)",
                    "CTR": "Click-Through Rate",
                    "topic": "Topic",
                    "num_creatives": "# of Creatives",
                },
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
        
            fig.update_yaxes(tickformat=".2%")
            fig.update_layout(hovermode="closest", height=500)
            st.plotly_chart(fig, width="stretch")


        st.markdown("---")
        st.subheader("📊 Spend by Topic & Journey Role")
        st.caption("See which topics are skewed toward top-of-funnel (Engagement), mid-funnel (Intent), or bottom-funnel (Conversion) creatives.")
        
        topic_journey_data = creative_metrics[creative_metrics['topic'].notna()].copy()
        
        if len(topic_journey_data) > 0:
            topic_journey_spend = topic_journey_data.groupby(['topic', 'journey_role']).agg({
                'spend': 'sum',
                'creative_name': 'count'
            }).reset_index()
            topic_journey_spend.rename(columns={'creative_name': 'num_creatives'}, inplace=True)
            
            colors = {'Engagement': '#4CAF50', 'Intent': '#FF9800', 'Conversion': '#2196F3'}
            fig_stacked = px.bar(
                topic_journey_spend,
                x='topic',
                y='spend',
                color='journey_role',
                title="Spend Distribution by Topic and Journey Role",
                labels={'topic': 'Topic', 'spend': 'Spend ($)', 'journey_role': 'Journey Role'},
                color_discrete_map=colors,
                barmode='stack'
            )
            fig_stacked.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_stacked, width="stretch")
            
            st.markdown("---")
            st.subheader("🎯 Topic Performance by Layer")
            
            layer_tabs = st.tabs(["📢 Engagement", "🛒 Intent", "💰 Conversion"])
            
            with layer_tabs[0]:
                eng_topics = topic_journey_data[topic_journey_data['journey_role'] == 'Engagement']
                if len(eng_topics) > 0:
                    eng_by_topic = eng_topics.groupby('topic').agg({
                        'spend': 'sum',
                        'impressions': 'sum',
                        'clicks': 'sum',
                        'CTR': 'mean',
                        'CPC': 'mean',
                        'creative_name': 'count'
                    }).reset_index()
                    eng_by_topic.rename(columns={'creative_name': 'num_creatives'}, inplace=True)
                    eng_by_topic = eng_by_topic.sort_values('CTR', ascending=False)
                    eng_by_topic['CTR'] = eng_by_topic['CTR'] * 100
                    st.dataframe(eng_by_topic, width="stretch", column_config={
                        'CTR': st.column_config.NumberColumn('Avg CTR', format="%.3f %%"),
                        'CPC': st.column_config.NumberColumn('Avg CPC', format="$ %.2f"),
                        'spend': st.column_config.NumberColumn('Spend', format="$ %,.0f"),
                    })
                else:
                    st.info("No engagement creatives with topics.")
            
            with layer_tabs[1]:
                int_topics = topic_journey_data[topic_journey_data['journey_role'] == 'Intent']
                if len(int_topics) > 0:
                    int_agg = {'spend': 'sum', 'impressions': 'sum', 'clicks': 'sum', 'creative_name': 'count'}
                    for col in ['add_to_cart_rate', 'view_content_rate', 'page_view_rate']:
                        if col in int_topics.columns:
                            int_agg[col] = 'mean'
                    int_by_topic = int_topics.groupby('topic').agg(int_agg).reset_index()
                    int_by_topic.rename(columns={'creative_name': 'num_creatives'}, inplace=True)
                    col_config = {
                        'spend': st.column_config.NumberColumn('Spend', format="$ %,.0f"),
                    }
                    for col in ['add_to_cart_rate', 'view_content_rate', 'page_view_rate']:
                        if col in int_by_topic.columns:
                            int_by_topic[col] = int_by_topic[col] * 100
                            col_config[col] = st.column_config.NumberColumn(col.replace('_', ' ').title(), format="%.3f %%")
                    st.dataframe(int_by_topic, width="stretch", column_config=col_config)
                else:
                    st.info("No intent creatives with topics.")
            
            with layer_tabs[2]:
                conv_topics = topic_journey_data[topic_journey_data['journey_role'] == 'Conversion']
                if len(conv_topics) > 0:
                    conv_agg = {'spend': 'sum', 'impressions': 'sum', 'clicks': 'sum', 'creative_name': 'count'}
                    for col in ['CVR', 'CPA', 'ROAS']:
                        if col in conv_topics.columns:
                            conv_agg[col] = 'mean'
                    if 'conversions' in conv_topics.columns:
                        conv_agg['conversions'] = 'sum'
                    conv_by_topic = conv_topics.groupby('topic').agg(conv_agg).reset_index()
                    conv_by_topic.rename(columns={'creative_name': 'num_creatives'}, inplace=True)
                    conv_by_topic = conv_by_topic.sort_values('CVR' if 'CVR' in conv_by_topic.columns else 'spend', ascending=False)
                    col_config = {
                        'spend': st.column_config.NumberColumn('Spend', format="$ %,.0f"),
                    }
                    if 'CVR' in conv_by_topic.columns:
                        conv_by_topic['CVR'] = conv_by_topic['CVR'] * 100
                        col_config['CVR'] = st.column_config.NumberColumn('Avg CVR', format="%.3f %%")
                    if 'CPA' in conv_by_topic.columns:
                        col_config['CPA'] = st.column_config.NumberColumn('Avg CPA', format="$ %.2f")
                    if 'ROAS' in conv_by_topic.columns:
                        col_config['ROAS'] = st.column_config.NumberColumn('Avg ROAS', format="%.2f x")
                    if 'conversions' in conv_by_topic.columns:
                        col_config['conversions'] = st.column_config.NumberColumn('Conversions', format="%,d")
                    st.dataframe(conv_by_topic, width="stretch", column_config=col_config)
                else:
                    st.info("No conversion creatives with topics.")
        else:
            st.info("No topic data available for journey role analysis.")

        st.markdown("---")
        st.subheader("Topic Performance Summary")

        topic_agg_dict = {
            'impressions': 'sum',
            'clicks': 'sum',
            'spend': 'sum',
            'creative_name': 'nunique'
        }

        if has_conversions:
            topic_agg_dict['conversions'] = 'sum'

        topic_metrics = filtered_df[filtered_df['topic'].notna()].groupby('topic').agg(topic_agg_dict).reset_index()
        topic_metrics.rename(columns={'creative_name': 'num_creatives'}, inplace=True)

        topic_metrics['CTR'] = np.where(
            topic_metrics['impressions'] > 0,
            topic_metrics['clicks'] / topic_metrics['impressions'],
            0
        )
        topic_metrics['CPC'] = np.where(
            topic_metrics['clicks'] > 0,
            topic_metrics['spend'] / topic_metrics['clicks'],
            0
        )

        if has_conversions:
            topic_metrics['CVR'] = np.where(
                topic_metrics['clicks'] > 0,
                topic_metrics['conversions'] / topic_metrics['clicks'],
                0
            )
            topic_metrics['CPA'] = np.where(
                topic_metrics['conversions'] > 0,
                topic_metrics['spend'] / topic_metrics['conversions'],
                0
            )

        topic_metrics = topic_metrics.sort_values('CTR', ascending=False)

        # --- NEW: Quadrant chart: CTR vs Spend share by topic ---
        st.markdown("---")
        st.subheader("Spend vs CTR by Topic (Quadrant View)")

        # Work with raw CTR (0-1) for chart; keep a copy to avoid fighting with % scaling
        topic_quadrant = topic_metrics.copy()

        total_spend_topics = topic_quadrant['spend'].sum()
        topic_quadrant['spend_share'] = np.where(
            total_spend_topics > 0,
            topic_quadrant['spend'] / total_spend_topics,
            0
        )

        avg_ctr_raw = topic_quadrant['CTR'].mean()
        avg_spend_share = topic_quadrant['spend_share'].mean()

        fig_q = px.scatter(
            topic_quadrant,
            x='spend_share',
            y='CTR',
            size='spend',
            text='topic',
            labels={
                'spend_share': 'Share of Spend',
                'CTR': 'CTR'
            },
            title="Topic Efficiency: CTR vs Share of Spend (size = spend)"
        )
        fig_q.update_traces(textposition="top center")

        # Add quadrant lines
        fig_q.add_vline(x=avg_spend_share, line_dash="dash", line_color="grey")
        fig_q.add_hline(y=avg_ctr_raw, line_dash="dash", line_color="grey")

        fig_q.update_yaxes(tickformat=".2%")
        fig_q.update_xaxes(tickformat=".1%")
        st.plotly_chart(fig_q, width="stretch")


        topic_metrics['CTR'] = topic_metrics['CTR'] * 100
        if 'CVR' in topic_metrics.columns:
            topic_metrics['CVR'] = topic_metrics['CVR'] * 100

        topic_column_config = {
            'topic': st.column_config.TextColumn('Topic'),
            'num_creatives': st.column_config.NumberColumn('# Creatives', format="%d"),
            'impressions': st.column_config.NumberColumn('Impressions', format="%,d"),
            'clicks': st.column_config.NumberColumn('Clicks', format="%,d"),
            'spend': st.column_config.NumberColumn('Spend', format="$ %,.2f"),
            'CTR': st.column_config.NumberColumn('CTR', format="%.3f %%"),
            'CPC': st.column_config.NumberColumn('CPC', format="$ %.2f"),
        }

        if 'CVR' in topic_metrics.columns:
            topic_column_config['CVR'] = st.column_config.NumberColumn('CVR', format="%.3f %%")
            topic_column_config['conversions'] = st.column_config.NumberColumn('Conversions', format="%,d")
            topic_column_config['CPA'] = st.column_config.NumberColumn('CPA', format="$ %.2f")

        st.dataframe(
            format_currency_columns(display_df.copy()),
            width="stretch",
            height=400,
            column_config=column_config
        )

        csv_topics = topic_metrics.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Topic Performance CSV",
            data=csv_topics,
            file_name="topic_performance.csv",
            mime="text/csv"
        )

        st.markdown("---")
        st.subheader("📊 Key Topic Insights")

        if len(topic_metrics) >= 1:
            top_topic = topic_metrics.iloc[0]
            bottom_topic = topic_metrics.iloc[-1]

            col1, col2 = st.columns(2)

            with col1:
                st.success(f"**🏆 Best Performing Topic**")
                st.write(f"**{top_topic['topic']}**")
                st.write(f"- CTR: **{top_topic['CTR']:.3f}%**")
                st.write(f"- CPC: **${top_topic['CPC']:.2f}**")
                st.write(f"- {int(top_topic['num_creatives'])} creatives")
                st.write(f"- ${top_topic['spend']:,.0f} total spend")

            with col2:
                st.error(f"**⚠️ Lowest Performing Topic**")
                st.write(f"**{bottom_topic['topic']}**")
                st.write(f"- CTR: **{bottom_topic['CTR']:.3f}%**")
                st.write(f"- CPC: **${bottom_topic['CPC']:.2f}**")
                st.write(f"- {int(bottom_topic['num_creatives'])} creatives")
                st.write(f"- ${bottom_topic['spend']:,.0f} total spend")

            st.markdown("---")

            insights = []

            ctr_range = topic_metrics['CTR'].max() - topic_metrics['CTR'].min()
            if ctr_range > 2.0:
                insights.append(f"• **High CTR variance** across topics ({ctr_range:.2f}% spread) - some topics significantly outperform others")

            high_spend_topics = topic_metrics.nlargest(3, 'spend')
            high_ctr_topics = topic_metrics.nlargest(3, 'CTR')

            overlap = set(high_spend_topics['topic']) & set(high_ctr_topics['topic'])
            if len(overlap) > 0:
                insights.append(f"• **Efficient spend allocation** - High-spend topics ({', '.join(overlap)}) also have high CTR")
            else:
                insights.append(f"• **Opportunity for reallocation** - Your highest-spend topics aren't your best performers")

            avg_ctr = topic_metrics['CTR'].mean()
            above_avg_count = len(topic_metrics[topic_metrics['CTR'] > avg_ctr])
            insights.append(f"• {above_avg_count}/{len(topic_metrics)} topics perform above average CTR ({avg_ctr:.2f}%)")

            if len(insights) > 0:
                st.markdown("**Summary:**")
                for insight in insights:
                    st.markdown(insight)


if __name__ == "__main__":
    main()
