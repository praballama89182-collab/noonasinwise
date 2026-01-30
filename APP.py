import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="FINAL NOON SKU WISE", page_icon="🕛", layout="wide")

# Noon Brand Code Mapping
BRAND_MAP = {
    'maison_de_lavenir': 'Maison de l’Avenir',
    'creation_lamis': 'Creation Lamis',
    'jean_paul_dupont': 'Jean Paul Dupont',
    'paris_collection': 'Paris Collection',
    'dorall_collection': 'Dorall Collection',
    'cp_trendies': 'CP Trendies'
}

def clean_numeric(val):
    """Deep clean of currency, commas, and non-breaking spaces."""
    if isinstance(val, str):
        cleaned = val.replace('AED', '').replace('$', '').replace('\xa0', '').replace(',', '').strip()
        try: return pd.to_numeric(cleaned)
        except: return 0.0
    return val if isinstance(val, (int, float)) else 0.0

st.title("🕛 FINAL NOON SKU WISE")
st.info("Noon Strategic Audit: Campaign-First View, Organic Isolation, and Stock-to-Velocity Tracking")

st.sidebar.header("Upload Noon Reports")
sales_file = st.sidebar.file_uploader("1. Noon Sales Export (CSV)", type=["csv"])
ad_file = st.sidebar.file_uploader("2. Noon Ad Sku Report (CSV)", type=["csv"])
inv_file = st.sidebar.file_uploader("3. Noon Inventory Report (CSV)", type=["csv"])

if sales_file and ad_file:
    # 1. Load Data
    sales_df_raw = pd.read_csv(sales_file)
    ad_df_raw = pd.read_csv(ad_file)
    inv_df_raw = pd.read_csv(inv_file) if inv_file else None

    # Normalize headers
    sales_df_raw.columns = [str(c).strip() for c in sales_df_raw.columns]
    ad_df_raw.columns = [str(c).strip() for c in ad_df_raw.columns]
    if inv_df_raw is not None: inv_df_raw.columns = [str(c).strip() for c in inv_df_raw.columns]

    # 2. Process Sales Data (Source of Truth for SKUs)
    sales_df_raw['gmv_lcy'] = sales_df_raw['gmv_lcy'].apply(clean_numeric)
    
    # Aggregate sales by Noon SKU to get total platform revenue
    sales_summary = sales_df_raw.groupby(['sku', 'partner_sku', 'brand_code']).agg({
        'gmv_lcy': 'sum'
    }).reset_index().rename(columns={'gmv_lcy': 'Total Sales', 'brand_code': 'Brand_Key'})
    
    sales_summary['Brand'] = sales_summary['Brand_Key'].map(BRAND_MAP).fillna(sales_summary['Brand_Key'])

    # 3. Process Ad Data (Mapped by internal 'sku')
    for c in ['Spends', 'Revenue', 'Clicks', 'Views', 'Orders']:
        ad_df_raw[c] = ad_df_raw[c].apply(clean_numeric)

    # Aggregate by Campaign + Sku (Handling multi-campaign SKUs)
    ad_camp_summary = ad_df_raw.groupby(['Campaign Name', 'Sku']).agg({
        'Spends': 'sum', 'Revenue': 'sum', 'Clicks': 'sum', 'Views': 'sum', 'Orders': 'sum'
    }).reset_index()

    # Per-SKU Ad Totals for Organic isolation
    ad_sku_total = ad_df_raw.groupby('Sku').agg({
        'Revenue': 'sum', 'Spends': 'sum'
    }).rename(columns={'Revenue': 'SKU_AD_SALES', 'Spends': 'SKU_AD_SPEND'}).reset_index()

    # 4. Process Inventory Data
    if inv_df_raw is not None:
        inv_summary = inv_df_raw.groupby('sku')['qty'].sum().reset_index().rename(columns={'qty': 'Stock', 'sku': 'inv_sku'})
    else:
        inv_summary = pd.DataFrame(columns=['inv_sku', 'Stock'])

    # 5. Final Merge (Campaign Level)
    merged_df = pd.merge(sales_summary, ad_camp_summary, left_on='sku', right_on='Sku', how='left')
    merged_df = pd.merge(merged_df, ad_sku_total, left_on='sku', right_on='Sku', how='left').fillna(0)
    merged_df = pd.merge(merged_df, inv_summary, left_on='sku', right_on='inv_sku', how='left').fillna(0)

    merged_df['Campaign Name'] = merged_df['Campaign Name'].apply(lambda x: x if x != 0 and str(x).strip() != "" else "Organic / No Ads")

    # 6. KPI Calculations
    merged_df['Organic Sales'] = merged_df['Total Sales'] - merged_df['SKU_AD_SALES']
    merged_df['DRR'] = merged_df['Total Sales'] / 30
    merged_df['Ad Contribution %'] = (merged_df['SKU_AD_SALES'] / merged_df['Total Sales']).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Efficiency Calculations
    merged_df['ROAS'] = (merged_df['Revenue'] / merged_df['Spends']).replace([np.inf, -np.inf], 0).fillna(0)
    merged_df['ACOS'] = (merged_df['Spends'] / merged_df['Revenue']).replace([np.inf, -np.inf], 0).fillna(0)
    merged_df['TACOS'] = (merged_df['SKU_AD_SPEND'] / merged_df['Total Sales']).replace([np.inf, -np.inf], 0).fillna(0)
    merged_df['CTR'] = (merged_df['Clicks'] / merged_df['Views']).replace([np.inf, -np.inf], 0).fillna(0)
    merged_df['CVR'] = (merged_df['Orders'] / merged_df['Clicks']).replace([np.inf, -np.inf], 0).fillna(0)

    # Renaming for Table View
    table_df = merged_df.rename(columns={
        'Campaign Name': 'Campaign', 'partner_sku': 'Partner SKU', 'sku': 'Noon SKU',
        'Revenue': 'Ad Sales (Campaign)', 'Spends': 'Ad Spend'
    })

    tabs = st.tabs(["🌍 Portfolio Overview"] + sorted(list(BRAND_MAP.values())))

    def display_metrics_dashboard(raw_ad, raw_sales):
        """Unified 12-metric dashboard for overall and brand headers."""
        t_sales = raw_sales['gmv_lcy'].sum()
        a_sales = raw_ad['Revenue'].sum()
        o_sales = t_sales - a_sales
        t_spend = raw_ad['Spends'].sum()
        t_clicks = raw_ad['Clicks'].sum()
        t_views = raw_ad['Views'].sum()
        t_orders = raw_ad['Orders'].sum()

        st.markdown("#### 💰 Volume & Velocity Dashboard")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Sales", f"{t_sales:,.2f}")
        c2.metric("Ad Sales", f"{a_sales:,.2f}")
        c3.metric("Organic Sales", f"{o_sales:,.2f}")
        c4.metric("Ad Spend", f"{t_spend:,.2f}")
        c5.metric("Ad Contrib.", f"{(a_sales/t_sales):.1%}" if t_sales > 0 else "0%")

        st.markdown("#### ⚡ Efficiency & ROI Metrics")
        e1, e2, e3, e4, e5, e6 = st.columns(6)
        e1.metric("ROAS", f"{(a_sales/t_spend if t_spend > 0 else 0):.2f}")
        e2.metric("ACOS", f"{(t_spend/a_sales if a_sales > 0 else 0):.1%}")
        e3.metric("TACOS", f"{(t_spend/t_sales if t_sales > 0 else 0):.1%}")
        e4.metric("CTR", f"{(t_clicks/t_views if t_views > 0 else 0):.2%}")
        e5.metric("CVR", f"{(t_orders/t_clicks if t_clicks > 0 else 0):.2%}")
        e6.metric("DRR (Portfolio)", f"{(t_sales/30):,.2f}")

    # Columns to show in table
    cols = ['Campaign', 'Partner SKU', 'Noon SKU', 'Stock', 'Total Sales', 'DRR', 'Ad Sales (Campaign)', 'Ad Spend', 'Organic Sales', 'Ad Contribution %', 'ROAS', 'ACOS', 'TACOS', 'CTR', 'CVR']

    with tabs[0]:
        st.subheader("Global Noon Portfolio Overview")
        display_metrics_dashboard(ad_df_raw, sales_df_raw)
        st.divider()
        st.dataframe(table_df[cols].sort_values(by='Total Sales', ascending=False), hide_index=True, use_container_width=True)

    for i, (key, b_name) in enumerate(sorted(BRAND_MAP.items())):
        with tabs[i+1]:
            b_data = table_df[table_df['Brand_Key'] == key]
            raw_ad_b = ad_df_raw[ad_df_raw['Campaign Name'].str.contains(key[:3].upper(), na=False)]
            raw_sales_b = sales_df_raw[sales_df_raw['brand_code'] == key]
            
            if not b_data.empty:
                st.subheader(f"{b_name} Performance")
                display_metrics_dashboard(raw_ad_b, raw_sales_b)
                st.divider()
                st.dataframe(b_data[cols].sort_values(by='Total Sales', ascending=False), hide_index=True, use_container_width=True)
            else:
                st.warning(f"No active data for {b_name} in the current reports.")

    # Export
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        table_df[cols].to_excel(writer, sheet_name='NOON_SKU_AUDIT', index=False)
        for b_key, b_name in BRAND_MAP.items():
            b_sheet = table_df[table_df['Brand_Key'] == b_key]
            if not b_sheet.empty:
                b_sheet[cols].to_excel(writer, sheet_name=b_name[:31], index=False)
    st.sidebar.download_button("📥 Download Noon Master Report", data=output.getvalue(), file_name="Noon_SKU_Performance_Audit.xlsx", use_container_width=True)

else:
    st.info("Upload your Noon Sales, Ad SKU, and Inventory reports to generate the audit.")
