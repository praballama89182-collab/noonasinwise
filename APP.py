import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="FINAL NOON SKU WISE", page_icon="🕛", layout="wide")

# Master Brand Mapping
BRAND_MAP = {
    'maison_de_lavenir': 'Maison de l’Avenir',
    'creation_lamis': 'Creation Lamis',
    'jean_paul_dupont': 'Jean Paul Dupont',
    'paris_collection': 'Paris Collection',
    'dorall_collection': 'Dorall Collection',
    'cp_trendies': 'CP Trendies'
}

def clean_numeric(val):
    """Pure numeric cleaning - No currency prefixes."""
    if isinstance(val, str):
        cleaned = val.replace('AED', '').replace('$', '').replace('\xa0', '').replace(',', '').strip()
        try: return pd.to_numeric(cleaned)
        except: return 0.0
    return val if isinstance(val, (int, float)) else 0.0

def find_robust_col(df, keywords):
    for col in df.columns:
        if any(kw.lower() in str(col).lower().strip() for kw in keywords):
            return col
    return None

st.title("🕛 FINAL NOON SKU WISE")
st.info("Comprehensive Noon Audit: SKU-Level Velocity, Organic Isolation, and Stock Tracking")

st.sidebar.header("Upload Noon Reports")
sales_file = st.sidebar.file_uploader("1. Noon Sales Export", type=["csv", "xlsx", "xls"])
ad_sku_files = st.sidebar.file_uploader("2. Noon Ad SKU Reports (Upload Multiple)", type=["csv"], accept_multiple_files=True)
inv_file = st.sidebar.file_uploader("3. Noon Inventory Report", type=["csv", "xlsx", "xls"])

if sales_file and ad_sku_files:
    def load_df(file):
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        df.columns = [str(c).strip() for c in df.columns]
        return df

    # 1. Load Sales & Base identifiers
    sales_raw = load_df(sales_file)
    sl_sku_col = find_robust_col(sales_raw, ['sku'])
    sl_psku_col = find_robust_col(sales_raw, ['partner_sku', 'partner sku'])
    sl_brand_col = find_robust_col(sales_raw, ['brand_code', 'brand'])
    sl_sales_col = find_robust_col(sales_raw, ['gmv_lcy', 'revenue', 'price'])

    sales_raw[sl_sales_col] = sales_raw[sl_sales_col].apply(clean_numeric)
    
    # Aggregate Base Sales
    sales_base = sales_raw.groupby([sl_sku_col, sl_psku_col, sl_brand_col], as_index=False)[sl_sales_col].sum()
    sales_base.rename(columns={sl_sku_col: 'Noon SKU', sl_psku_col: 'Partner SKU', sl_brand_col: 'Brand_Key', sl_sales_col: 'Total Sales'}, inplace=True)
    sales_base['Brand'] = sales_base['Brand_Key'].map(BRAND_MAP).fillna(sales_base['Brand_Key'])

    # 2. Combine Multiple Ad Tabs (Product Sku + Brand Sku)
    ad_list = []
    for f in ad_sku_files:
        df_temp = load_df(f)
        if 'Sku' in df_temp.columns:
            ad_list.append(df_temp)
    
    if ad_list:
        ads_raw = pd.concat(ad_list, ignore_index=True)
        ad_sku_col = 'Sku'
        ad_camp_col = 'Campaign Name'
        
        # Numeric Clean
        ad_cols = ['Spends', 'Revenue', 'Clicks', 'Views', 'Orders']
        for c in ad_cols:
            if c in ads_raw.columns: ads_raw[c] = ads_raw[c].apply(clean_numeric)

        # Aggregate Ads for Table
        ads_camp_grouped = ads_raw.groupby([ad_camp_col, ad_sku_col], as_index=False).agg({
            'Spends': 'sum', 'Revenue': 'sum', 'Clicks': 'sum', 'Views': 'sum', 'Orders': 'sum'
        })
        
        # Aggregate Ads for Organic Calculation (SKU Total)
        ads_sku_totals = ads_raw.groupby(ad_sku_col, as_index=False).agg({'Revenue': 'sum', 'Spends': 'sum'})
        ads_sku_totals.columns = ['Ad_SKU_Match', 'SKU_AD_SALES', 'SKU_AD_SPEND']
    else:
        st.error("None of the uploaded Ad files contained a 'Sku' column.")
        st.stop()

    # 3. Inventory
    if inv_file:
        inv_raw = load_df(inv_file)
        iv_sku_col = find_robust_col(inv_raw, ['sku'])
        iv_qty_col = find_robust_col(inv_raw, ['qty', 'available'])
        inv_grouped = inv_raw.groupby(iv_sku_col, as_index=False)[iv_qty_col].sum()
        inv_grouped.columns = ['Inv_SKU_Match', 'Stock']
    else:
        inv_grouped = pd.DataFrame(columns=['Inv_SKU_Match', 'Stock'])

    # 4. Master Join
    merged = pd.merge(sales_base, ads_camp_grouped, left_on='Noon SKU', right_on=ad_sku_col, how='left')
    merged = pd.merge(merged, ads_sku_totals, left_on='Noon SKU', right_on='Ad_SKU_Match', how='left').fillna(0)
    merged = pd.merge(merged, inv_grouped, left_on='Noon SKU', right_on='Inv_SKU_Match', how='left').fillna(0)

    # Clean Up UI
    merged['Campaign'] = merged[ad_camp_col].apply(lambda x: x if x != 0 and str(x).strip() != "" else "Organic")
    merged['Organic Sales'] = merged['Total Sales'] - merged['SKU_AD_SALES']
    merged['DRR'] = merged['Total Sales'] / 30
    
    # KPI Ratios
    merged['ROAS'] = (merged['Revenue'] / merged['Spends']).replace([np.inf, -np.inf], 0).fillna(0)
    merged['ACOS'] = (merged['Spends'] / merged['Revenue']).replace([np.inf, -np.inf], 0).fillna(0)
    merged['TACOS'] = (merged['SKU_AD_SPEND'] / merged['Total Sales']).replace([np.inf, -np.inf], 0).fillna(0)

    # Dashboard Rendering
    def show_metrics(sales_df, ads_df):
        t_sales = sales_df[sl_sales_col].sum()
        a_sales = ads_df['Revenue'].sum()
        t_spend = ads_df['Spends'].sum()
        
        st.markdown("#### 💰 Portfolio Overview (Pure Numbers)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Sales", f"{t_sales:,.2f}")
        c2.metric("Ad Sales", f"{a_sales:,.2f}")
        c3.metric("Organic Sales", f"{(t_sales - a_sales):,.2f}")
        c4.metric("Ad Spend", f"{t_spend:,.2f}")
        
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("ROAS", f"{(a_sales/t_spend if t_spend > 0 else 0):.2f}")
        e2.metric("ACOS", f"{(t_spend/a_sales if a_sales > 0 else 0):.1%}")
        e3.metric("TACOS", f"{(t_spend/t_sales if t_sales > 0 else 0):.1%}")
        e4.metric("DRR (Portfolio)", f"{(t_sales/30):,.2f}")

    tabs = st.tabs(["🌍 Portfolio Overview"] + sorted(list(BRAND_MAP.values())))
    table_cols = ['Campaign', 'Partner SKU', 'Noon SKU', 'Stock', 'Total Sales', 'DRR', 'Revenue', 'Spends', 'Organic Sales', 'ROAS', 'ACOS', 'TACOS']

    with tabs[0]:
        st.subheader("Global Noon Dashboard")
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

    # Export Logic
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        merged[table_cols].to_excel(writer, sheet_name='MASTER_AUDIT', index=False)
    st.sidebar.download_button("📥 Download Master Report", data=output.getvalue(), file_name="Noon_Master_Audit.xlsx", use_container_width=True)
else:
    st.info("Upload Sales, Ad SKU reports (Product and Brand), and Inventory to begin.")
