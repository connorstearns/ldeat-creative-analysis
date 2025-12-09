import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Sheets Test", page_icon="🧪")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

def load_google_sheet_to_df():
    sa = st.secrets["gcp_service_account"]
    sheet_url = st.secrets["google"]["sheet_url"]

    # --- secrets shape debug ---
    st.write("Secrets debug:", {
        "type": str(type(sa)),
        "keys": list(sa.keys()),
        "client_email": sa.get("client_email", "MISSING"),
        "has_private_key": "private_key" in sa,
        "has_token_uri": "token_uri" in sa,
    })

    try:
        creds = Credentials.from_service_account_info(sa).with_scopes(SCOPES)
        gc = gspread.authorize(creds)

        st.write("Trying to open sheet:", sheet_url)
        sh = gc.open_by_url(sheet_url)
        ws = sh.sheet1
        data = ws.get_all_records()
        df = pd.DataFrame(data)
        return df

    except Exception as e:
        st.error(f"Error loading Google Sheet: {repr(e)}")

        # show any response body if present
        resp = getattr(e, "response", None)
        if resp is not None and hasattr(resp, "text"):
            st.code(resp.text, language="json")

        st.write("Service account email:", sa.get("client_email", "MISSING"))
        raise

st.title("🧪 Google Sheets Test")

try:
    df = load_google_sheet_to_df()
    st.success(f"Loaded {len(df)} rows from Google Sheet")
    st.dataframe(df.head())
except Exception:
    st.stop()
