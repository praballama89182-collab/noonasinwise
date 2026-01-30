import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="FINAL NOON SKU WISE", page_icon="🕛", layout="wide")

# Brand Configuration (Noon Brand Codes)
BRAND_MAP = {
    'maison_de_lavenir': 'Maison de l’Avenir',
    'creation_lamis': 'Creation Lamis',
    'jean_paul_dupont': 'Jean Paul Dupont',
    'paris_collection': 'Paris Collection',
    'dorall_collection': 'Dorall Collection',
    'cp_trendies': 'CP Trendies'
}

def clean_numeric(val):
    """Deep clean of currency, commas, and formatting to return pure numbers."""
    if isinstance(val, str):
        cleaned = val.replace('AED', '').replace('$', '').replace('\xa0', '').replace(',', '').strip()
        try: return pd.to_numeric(cleaned)
        except: return 0.0
    return val if isinstance(val, (int, float)) else 0.0

def find_robust_col(df, keywords, exclude=['acos', 'roas', 'cpc', 'ctr', 'target']):
    """Dynamically identifies columns based on keywords."""
    for col in df.columns:
        col_clean = str(col).strip().lower()
        if any(kw.lower() in col_clean for kw in keywords):
            if not any(ex.lower() in col_clean for ex in exclude):
                return col
    return None

def load_flexible_df(file):
    """Loads any spreadsheet format and cleans headers."""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.txt'):
            df = pd.read_csv(file, sep='\t')
        else:
            df = pd.read_excel(file)
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception as e:
        st.error(f"Error loading {file.name}: {e}")
        return None

st.title("🕛 FINAL NOON SKU WISE AUDIT")
st.info("Universal Mapping: Scanning all tabs for SKU/Partner SKU commonality.")

st.sidebar.header("Step 1: Upload Noon Data")
sales_file = st.sidebar.file_uploader("1. Noon Sales Export", type=["csv", "xlsx", "xls", "txt"])
ad_files = st.sidebar.file_uploader("2. Ad Report Tabs (Multiple)", type=["csv", "xlsx"], accept_multiple_files=True)
inv_file = st.sidebar.file_uploader("3. Noon Inventory Report", type=["csv", "xlsx", "xls", "txt"])

if sales_file and ad_files:
    # --- 1. PROCESS SALES ---
    sales_raw = load_flexible_df(sales_file)
    sl_sku_col = find_robust_col(sales_raw, ['sku'], exclude=['partner'])
    sl_psku_col = find_robust_col(sales_raw, ['partner_sku', 'partner sku'])
    sl_rev_col = find_robust_col(sales_raw, ['gmv_lcy', 'revenue', 'price'])
    sl_brand_col = find_robust_col(sales_raw, ['brand_code', 'brand'])

    sales_raw[sl_rev_col] = sales_raw[sl_rev_col].apply(clean_numeric)
    # Filter to ensure we have valid SKUs
    sales_raw = sales_raw[sales_raw[sl_sku_col].notna()].copy()
    sales_raw[sl_sku_col] = sales_raw[sl_sku_col].astype(str).str.strip()

    # Anchor SKU Table
    sl_summary = sales_raw.groupby([sl_sku_col, sl_psku_col, sl_brand_col], as_index=False)[sl_rev_col].sum()
    sl_summary.columns = ['Noon SKU', 'Partner SKU', 'Brand_Key', 'Total Sales']
    sl_summary['Brand'] = sl_summary['Brand_Key'].map(BRAND_MAP).fillna(sl_summary['Brand_Key'])

    # --- 2. PROCESS ADS (SCANNING ALL TABS) ---
    ad_sku_data_list = []
    
    for f in ad_files:
        df_tmp = load_flexible_df(f)
        sku_col = find_robust_col(df_tmp, ['sku'], exclude=['partner'])
        camp_col = find_robust_col(df_tmp, ['campaign'])
        
        if sku_col:
            # Filter 'header' rows found in Noon exports
            df_tmp = df_tmp[df_tmp[sku_col].astype(str).str.lower() != 'header'].copy()
            df_tmp[sku_col] = df_tmp[sku_col].astype(str).str.strip()
            
            # Map metrics
            rev = find_robust_col(df_tmp, ['revenue', 'sales'])
            spnd = find_robust_col(df_tmp, ['spends', 'spend'])
            
            if rev and spnd:
                df_tmp[rev] = df_tmp[rev].apply(clean_numeric)
                df_tmp[spnd] = df_tmp[spnd].apply(clean_numeric)
                df_tmp['source_tab'] = f.name
                ad_sku_data_list.append(df_tmp[[camp_col, sku_col, rev, spnd]])

    if ad_sku_data_list:
        ads_combined = pd.concat(ad_sku_data_list, ignore_index=True)
        # Final Grouping for Ads (Campaign + SKU)
        ads_camp_grouped = ads_combined.groupby(['Campaign Name', 'Sku'], as_index=False).agg({'Revenue': 'sum', 'Spends': 'sum'})
        
        # Per-SKU Ad Totals for Organic Calculation
        ads_sku_totals = ads_combined.groupby('Sku', as_index=False).agg({'Revenue': 'sum', 'Spends': 'sum'})
        ads_sku_totals.columns = ['Ad_SKU_Match', 'SKU_AD_SALES', 'SKU_AD_SPEND']
    else:
        st.error("No valid 'Sku' columns found in Ad reports. Ensure you upload '(Product) Sku' or '(Brand) Sku' tabs.")
        st.stop()

    # --- 3. PROCESS INVENTORY ---
    if inv_file:
        inv_raw = load_flexible_df(inv_file)
        iv_sku_col = find_robust_col(inv_raw, ['sku'], exclude=['partner'])
        iv_qty_col = find_robust_col(inv_raw, ['qty', 'available'])
        inv_raw[iv_sku_col] = inv_raw[iv_sku_col].astype(str).str.strip()
        inv_grouped = inv_raw.groupby(iv_sku_col, as_index=False)[iv_qty_col].sum()
        inv_grouped.columns = ['Inv_SKU_Match', 'Stock']
    else:
        inv_grouped = pd.DataFrame(columns=['Inv_SKU_Match', 'Stock'])

    # --- 4. DATA INTEGRATION (THE 3-WAY BRIDGE) ---
    merged = pd.merge(sl_summary, ads_camp_grouped, left_on='Noon SKU', right_on='Sku', how='left')
    merged = pd.merge(merged, ads_sku_totals, left_on='Noon SKU', right_on='Ad_SKU_Match', how='left').fillna(0)
    merged = pd.merge(merged, inv_grouped, left_on='Noon SKU', right_on='Inv_SKU_Match', how='left').fillna(0)

    # --- 5. CALCULATIONS ---
    merged['Campaign'] = merged['Campaign Name'].apply(lambda x: x if x != 0 and str(x) != "" else "Organic")
    merged['Organic Sales'] = merged['Total Sales'] - merged['SKU_AD_SALES']
    merged['DRR'] = merged['Total Sales'] / 30
    
    # Efficiency
    merged['ROAS'] = (merged['Revenue'] / merged['Spends']).replace([np.inf, -np.inf], 0).fillna(0)
    merged['TACOS'] = (merged['SKU_AD_SPEND'] / merged['Total Sales']).replace([np.inf, -np.inf], 0).fillna(0)

    # --- 6. DASHBOARD RENDERING ---
    def render_dash(sales_df, ads_df):
        t_sales = sales_df[sl_rev_col].sum()
        a_sales = ads_df['Revenue'].sum()
        t_spend = ads_df['Spends'].sum()
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Sales", f"{t_sales:,.2f}")
        c2.metric("Ad Sales", f"{a_sales:,.2f}")
        c3.metric("Organic Sales", f"{(t_sales - a_sales):,.2f}")
        c4.metric("Ad Spend", f"{t_spend:,.2f}")
        
        e1, e2, e3 = st.columns(3)
        e1.metric("ROAS", f"{(a_sales/t_spend if t_spend > 0 else 0):.2f}")
        e2.metric("TACOS", f"{(t_spend/t_sales if t_sales > 0 else 0):.1%}")
        e3.metric("Ad Contribution", f"{(a_sales/t_sales if t_sales > 0 else 0):.1%}")

    tabs = st.tabs(["🌍 Portfolio Overview"] + sorted(list(BRAND_MAP.values())))
    cols = ['Campaign', 'Partner SKU', 'Noon SKU', 'Stock', 'Total Sales', 'DRR', 'Revenue', 'Spends', 'Organic Sales', 'ROAS', 'TACOS']

    with tabs[0]:
        st.subheader("Global Portfolio Performance")
        render_dash(sales_raw, ads_combined)
        st.divider()
        st.dataframe(merged[cols].sort_values(by='Total Sales', ascending=False), hide_index=True, use_container_width=True)

    for i, (key, b_name) in enumerate(sorted(BRAND_MAP.items())):
        with tabs[i+1]:
            b_sales = sales_raw[sales_raw[sl_brand_col] == key]
            b_table = merged[merged['Brand_Key'] == key]
            b_ads = ads_combined[ads_combined['Sku'].isin(b_table['Noon SKU'])]
            
            if not b_table.empty:
                st.subheader(f"{b_name} Performance")
                render_dash(b_sales, b_ads)
                st.divider()
                st.dataframe(b_table[cols].sort_values(by='Total Sales', ascending=False), hide_index=True, use_container_width=True)

    # MASTER EXPORT
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        merged[cols].to_excel(writer, sheet_name='MASTER_AUDIT', index=False)
    st.sidebar.download_button("📥 Download Full Audit", data=output.getvalue(), file_name="Noon_Final_Audit.xlsx", use_container_width=True)

else:
    st.info("Upload your Sales export and Ad Sku reports to start the master audit.")
