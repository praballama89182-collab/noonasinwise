import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIOimport streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="FINAL NOON SKU WISE", page_icon="🕛", layout="wide")

# Master Noon Brand Mapping
BRAND_MAP = {
    'maison_de_lavenir': 'Maison de l’Avenir',
    'creation_lamis': 'Creation Lamis',
    'jean_paul_dupont': 'Jean Paul Dupont',
    'paris_collection': 'Paris Collection',
    'dorall_collection': 'Dorall Collection',
    'cp_trendies': 'CP Trendies'
}

def clean_numeric(val):
    """Safely converts strings with currency/commas to floats."""
    if isinstance(val, str):
        cleaned = val.replace('AED', '').replace('$', '').replace('\xa0', '').replace(',', '').strip()
        try: return pd.to_numeric(cleaned)
        except: return 0.0
    return val if isinstance(val, (int, float)) else 0.0

def load_flexible_df(file):
    """Loads any spreadsheet type and sanitizes headers."""
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith('.txt'):
        df = pd.read_csv(file, sep='\t')
    else:
        df = pd.read_excel(file)
    df.columns = [str(c).strip() for c in df.columns]
    return df

st.title("🕛 FINAL NOON SKU WISE")
st.info("Consolidated Noon Audit: Mapping Ad Sku, Sales Export, and Inventory")

st.sidebar.header("Upload Noon Reports")
sales_file = st.sidebar.file_uploader("1. Noon Sales Export", type=["csv", "xlsx", "xls", "txt"])
ad_files = st.sidebar.file_uploader("2. Noon Ad SKU Reports (Multiple Allowed)", type=["csv", "xlsx"], accept_multiple_files=True)
inv_file = st.sidebar.file_uploader("3. Noon Inventory Report", type=["csv", "xlsx", "xls", "txt"])

if sales_file and ad_files:
    # 1. Process Sales (Primary Data)
    sales_raw = load_flexible_df(sales_file)
    # Cast SKU to string to prevent TypeErrors during merge
    sales_raw['sku'] = sales_raw['sku'].astype(str).str.strip()
    sales_raw['gmv_lcy'] = sales_raw['gmv_lcy'].apply(clean_numeric)
    
    sales_base = sales_raw.groupby(['sku', 'partner_sku', 'brand_code'], as_index=False)['gmv_lcy'].sum()
    sales_base.rename(columns={'sku': 'Noon SKU', 'partner_sku': 'Partner SKU', 'brand_code': 'Brand_Key', 'gmv_lcy': 'Total Sales'}, inplace=True)
    sales_base['Brand'] = sales_base['Brand_Key'].map(BRAND_MAP).fillna(sales_base['Brand_Key'])

    # 2. Process Combined Ads (Bridge Logic)
    ad_list = []
    for f in ad_files:
        df_tmp = load_flexible_df(f)
        if 'Sku' in df_tmp.columns:
            # Filter out 'header' or summary rows often found in Noon exports
            df_tmp = df_tmp[df_tmp['Sku'] != 'header'].copy()
            df_tmp['Sku'] = df_tmp['Sku'].astype(str).str.strip()
            ad_list.append(df_tmp)
    
    if ad_list:
        ads_combined = pd.concat(ad_list, ignore_index=True)
        ad_metrics = ['Spends', 'Revenue', 'Clicks', 'Views', 'Orders']
        for c in ad_metrics:
            if c in ads_combined.columns: ads_combined[c] = ads_combined[c].apply(clean_numeric)

        # SKU Totals for Organic Calc
        ads_sku_totals = ads_combined.groupby('Sku', as_index=False).agg({'Revenue': 'sum', 'Spends': 'sum'})
        ads_sku_totals.columns = ['Ad_SKU_Match', 'SKU_AD_SALES', 'SKU_AD_SPEND']
        
        # Campaign + SKU level for the table
        ads_camp_grouped = ads_combined.groupby(['Campaign Name', 'Sku'], as_index=False).agg({
            'Spends': 'sum', 'Revenue': 'sum', 'Clicks': 'sum', 'Views': 'sum', 'Orders': 'sum'
        })
    else:
        st.error("Uploaded Ad files are missing the 'Sku' column.")
        st.stop()

    # 3. Process Inventory
    if inv_file:
        inv_raw = load_flexible_df(inv_file)
        inv_raw['sku'] = inv_raw['sku'].astype(str).str.strip()
        inv_raw['qty'] = inv_raw['qty'].apply(clean_numeric)
        inv_grouped = inv_raw.groupby('sku', as_index=False)['qty'].sum()
        inv_grouped.columns = ['Inv_SKU_Match', 'Stock']
    else:
        inv_grouped = pd.DataFrame(columns=['Inv_SKU_Match', 'Stock'])

    # 4. Final Triple-Merge
    merged = pd.merge(sales_base, ads_camp_grouped, left_on='Noon SKU', right_on='Sku', how='left')
    merged = pd.merge(merged, ads_sku_totals, left_on='Noon SKU', right_on='Ad_SKU_Match', how='left').fillna(0)
    merged = pd.merge(merged, inv_grouped, left_on='Noon SKU', right_on='Inv_SKU_Match', how='left').fillna(0)

    # 5. KPI Logic
    merged['Campaign'] = merged['Campaign Name'].replace(0, "Organic")
    merged['Organic Sales'] = merged['Total Sales'] - merged['SKU_AD_SALES']
    merged['DRR'] = merged['Total Sales'] / 30
    merged['ROAS'] = (merged['Revenue'] / merged['Spends']).replace([np.inf, -np.inf], 0).fillna(0)
    merged['ACOS'] = (merged['Spends'] / merged['Revenue']).replace([np.inf, -np.inf], 0).fillna(0)
    merged['TACOS'] = (merged['SKU_AD_SPEND'] / merged['Total Sales']).replace([np.inf, -np.inf], 0).fillna(0)

    # UI Rendering
    tabs = st.tabs(["🌍 Portfolio Overview"] + sorted(list(BRAND_MAP.values())))
    table_cols = ['Campaign', 'Partner SKU', 'Noon SKU', 'Stock', 'Total Sales', 'DRR', 'Revenue', 'Spends', 'Organic Sales', 'ROAS', 'ACOS', 'TACOS']

    def render_dashboard(sales_df, ads_df):
        t_sales = sales_df['gmv_lcy'].sum()
        a_sales = ads_df['Revenue'].sum()
        t_spend = ads_df['Spends'].sum()
        st.markdown("#### 💰 Portfolio KPIs (Numeric)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Sales", f"{t_sales:,.2f}")
        c2.metric("Ad Sales", f"{a_sales:,.2f}")
        c3.metric("Organic Sales", f"{(t_sales-a_sales):,.2f}")
        c4.metric("Ad Spend", f"{t_spend:,.2f}")
        e1, e2, e3 = st.columns(3)
        e1.metric("ROAS", f"{(a_sales/t_spend if t_spend > 0 else 0):.2f}")
        e2.metric("TACOS", f"{(t_spend/t_sales if t_sales > 0 else 0):.1%}")
        e3.metric("Ad Contrib.", f"{(a_sales/t_sales):.1%}")

    with tabs[0]:
        render_dashboard(sales_raw, ads_combined)
        st.divider()
        st.dataframe(merged[table_cols].sort_values(by='Total Sales', ascending=False), hide_index=True, use_container_width=True)

    for i, (key, b_name) in enumerate(sorted(BRAND_MAP.items())):
        with tabs[i+1]:
            b_table = merged[merged['Brand_Key'] == key]
            if not b_table.empty:
                b_sales = sales_raw[sales_raw['brand_code'] == key]
                b_ads = ads_combined[ads_combined['Sku'].isin(b_table['Noon SKU'])]
                render_dashboard(b_sales, b_ads)
                st.divider()
                st.dataframe(b_table[table_cols].sort_values(by='Total Sales', ascending=False), hide_index=True, use_container_width=True)

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        merged[table_cols].to_excel(writer, sheet_name='MASTER_AUDIT', index=False)
    st.sidebar.download_button("📥 Download Master Report", data=output.getvalue(), file_name="Noon_Master_Audit.xlsx", use_container_width=True)
else:
    st.info("Upload Sales, Ad SKU, and Inventory files to begin.")

st.set_page_config(page_title="FINAL NOON SKU WISE", page_icon="🕛", layout="wide")

# Noon Brand Logic
BRAND_MAP = {
    'maison_de_lavenir': 'Maison de l’Avenir',
    'creation_lamis': 'Creation Lamis',
    'jean_paul_dupont': 'Jean Paul Dupont',
    'paris_collection': 'Paris Collection',
    'dorall_collection': 'Dorall Collection',
    'cp_trendies': 'CP Trendies'
}

def clean_numeric(val):
    """Deep clean currency and commas to return pure numbers."""
    if isinstance(val, str):
        cleaned = val.replace('AED', '').replace('$', '').replace('\xa0', '').replace(',', '').strip()
        try: return pd.to_numeric(cleaned)
        except: return 0.0
    return val if isinstance(val, (int, float)) else 0.0

def find_robust_col(df, keywords):
    """Finds a column name safely across different report formats."""
    cols = df.columns.tolist()
    # Priority 1: Exact Match
    for col in cols:
        if any(kw.lower() == str(col).lower().strip() for kw in keywords):
            return col
    # Priority 2: Keyword Inclusion
    for col in cols:
        if any(kw.lower() in str(col).lower().strip() for kw in keywords):
            return col
    return None

st.title("🕛 FINAL NOON SKU WISE")
st.info("Universal Noon Audit: Optimized for Multi-Tab Tracking and Stock Integration")

st.sidebar.header("Upload Noon Reports")
# Accepting all common spreadsheet formats
sales_file = st.sidebar.file_uploader("1. Noon Sales Export", type=["csv", "xlsx", "xls", "txt"])
ad_file = st.sidebar.file_uploader("2. Noon Ad SKU Report", type=["csv", "xlsx", "xls", "txt"])
inv_file = st.sidebar.file_uploader("3. Noon Inventory Report", type=["csv", "xlsx", "xls", "txt"])

if sales_file and ad_file:
    def load_df(file):
        if file.name.endswith('.csv'): return pd.read_csv(file)
        elif file.name.endswith('.txt'): return pd.read_csv(file, sep='\t')
        else: return pd.read_excel(file)

    sales_raw = load_df(sales_file)
    ads_raw = load_df(ad_file)
    inv_raw = load_df(inv_file) if inv_file else None

    # Standardize column cleaning
    sales_raw.columns = [str(c).strip() for c in sales_raw.columns]
    ads_raw.columns = [str(c).strip() for c in ads_raw.columns]

    # 1. Identify Key Columns
    sl_sku_col = find_robust_col(sales_raw, ['sku'])
    sl_psku_col = find_robust_col(sales_raw, ['partner_sku', 'partner sku'])
    sl_brand_col = find_robust_col(sales_raw, ['brand_code', 'brand'])
    sl_sales_col = find_robust_col(sales_raw, ['gmv_lcy', 'revenue', 'price'])

    ad_sku_col = find_robust_col(ads_raw, ['sku', 'product sku'])
    ad_camp_col = find_robust_col(ads_raw, ['campaign'])
    ad_spend_col = find_robust_col(ads_raw, ['spends', 'spend'])
    ad_sales_col = find_robust_col(ads_raw, ['revenue', 'sales'])
    ad_clicks_col = find_robust_col(ads_raw, ['clicks'])
    ad_views_col = find_robust_col(ads_raw, ['views', 'impressions'])
    ad_orders_col = find_robust_col(ads_raw, ['orders'])

    # 2. Cleanup & Normalize Data
    for c in [ad_spend_col, ad_sales_col, ad_clicks_col, ad_views_col, ad_orders_col]:
        if c: ads_raw[c] = ads_raw[c].apply(clean_numeric)
    sales_raw[sl_sales_col] = sales_raw[sl_sales_col].apply(clean_numeric)

    # 3. Aggregate Sales
    sales_grouped = sales_raw.groupby([sl_sku_col, sl_psku_col, sl_brand_col], as_index=False)[sl_sales_col].sum()
    sales_grouped.rename(columns={sl_sku_col: 'Noon SKU', sl_psku_col: 'Partner SKU', sl_brand_col: 'Brand_Key', sl_sales_col: 'Total Sales'}, inplace=True)
    sales_grouped['Brand'] = sales_grouped['Brand_Key'].map(BRAND_MAP).fillna(sales_grouped['Brand_Key'])

    # 4. Aggregate Ads
    ad_metrics = {ad_spend_col: 'sum', ad_sales_col: 'sum', ad_clicks_col: 'sum', ad_views_col: 'sum', ad_orders_col: 'sum'}
    ads_camp_grouped = ads_raw.groupby([ad_camp_col, ad_sku_col], as_index=False).agg(ad_metrics)
    
    ads_sku_totals = ads_raw.groupby(ad_sku_col, as_index=False).agg({ad_sales_col: 'sum', ad_spend_col: 'sum'})
    ads_sku_totals.columns = ['Ad_SKU_Match', 'SKU_AD_SALES', 'SKU_AD_SPEND']

    # 5. Inventory
    if inv_raw is not None:
        inv_raw.columns = [str(c).strip() for c in inv_raw.columns]
        iv_sku_col = find_robust_col(inv_raw, ['sku'])
        iv_qty_col = find_robust_col(inv_raw, ['qty', 'available'])
        inv_grouped = inv_raw.groupby(iv_sku_col, as_index=False)[iv_qty_col].sum()
        inv_grouped.columns = ['Inv_SKU_Match', 'Stock']
    else:
        inv_grouped = pd.DataFrame(columns=['Inv_SKU_Match', 'Stock'])

    # 6. Master Join
    merged = pd.merge(sales_grouped, ads_camp_grouped, left_on='Noon SKU', right_on=ad_sku_col, how='left')
    merged = pd.merge(merged, ads_sku_totals, left_on='Noon SKU', right_on='Ad_SKU_Match', how='left').fillna(0)
    merged = pd.merge(merged, inv_grouped, left_on='Noon SKU', right_on='Inv_SKU_Match', how='left').fillna(0)

    # Clean Metrics Logic
    merged['Campaign'] = merged[ad_camp_col].apply(lambda x: x if x != 0 and str(x).strip() != "" else "Organic")
    merged['Organic Sales'] = merged['Total Sales'] - merged['SKU_AD_SALES']
    merged['DRR'] = merged['Total Sales'] / 30
    
    # 7. Dashboard Rendering
    def show_metrics(raw_sales, raw_ads):
        t_sales = raw_sales[sl_sales_col].sum()
        a_sales = raw_ads[ad_sales_col].sum()
        t_spend = raw_ads[ad_spend_col].sum()
        st.markdown("#### 💰 Portfolio Health Dashboard")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Sales", f"{t_sales:,.2f}")
        c2.metric("Ad Sales", f"{a_sales:,.2f}")
        c3.metric("Organic Sales", f"{(t_sales - a_sales):,.2f}")
        c4.metric("Ad Spend", f"{t_spend:,.2f}")
        
        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric("ROAS", f"{(a_sales/t_spend if t_spend > 0 else 0):.2f}")
        e2.metric("ACOS", f"{(t_spend/a_sales if a_sales > 0 else 0):.1%}")
        e3.metric("TACOS", f"{(t_spend/t_sales if t_sales > 0 else 0):.1%}")
        e4.metric("CTR", f"{(raw_ads[ad_clicks_col].sum()/raw_ads[ad_views_col].sum() if raw_ads[ad_views_col].sum() > 0 else 0):.2%}")
        e5.metric("CVR", f"{(raw_ads[ad_orders_col].sum()/raw_ads[ad_clicks_col].sum() if raw_ads[ad_clicks_col].sum() > 0 else 0):.2%}")

    tabs = st.tabs(["🌍 Portfolio Overview"] + sorted(list(BRAND_MAP.values())))
    table_cols = ['Campaign', 'Partner SKU', 'Noon SKU', 'Stock', 'Total Sales', 'DRR', 'Ad Sales (Campaign)', 'Ad Spend', 'Organic Sales', 'ROAS', 'ACOS', 'TACOS']

    with tabs[0]:
        st.subheader("Global Noon Audit")
        show_metrics(sales_raw, ads_raw)
        st.divider()
        st.dataframe(merged[table_cols].sort_values(by='Total Sales', ascending=False), hide_index=True, use_container_width=True)

    for i, (b_key, b_name) in enumerate(sorted(BRAND_MAP.items())):
        with tabs[i+1]:
            b_sales = sales_raw[sales_raw[sl_brand_col] == b_key]
            b_skus = b_sales[sl_sku_col].unique()
            b_ads = ads_raw[ads_raw[ad_sku_col].isin(b_skus)]
            b_table = merged[merged['Brand_Key'] == b_key]
            if not b_table.empty:
                st.subheader(f"{b_name} Performance")
                show_metrics(b_sales, b_ads)
                st.divider()
                st.dataframe(b_table[table_cols].sort_values(by='Total Sales', ascending=False), hide_index=True, use_container_width=True)

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        merged[table_cols].to_excel(writer, sheet_name='MASTER_AUDIT', index=False)
    st.sidebar.download_button("📥 Download Master Report", data=output.getvalue(), file_name="Noon_Master_Audit.xlsx", use_container_width=True)
else:
    st.info("Upload Noon Sales, Ads, and Inventory to begin.")

