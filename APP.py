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

def load_flexible_df(file):
    """Loads CSV or Excel based on file extension."""
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    df.columns = [str(c).strip() for c in df.columns]
    return df

st.title("🕛 FINAL NOON SKU WISE")
st.info("Noon Framework: Campaign-First View, SKU-Level Velocity, and Stock Integration")

st.sidebar.header("Upload Noon Reports")
sales_file = st.sidebar.file_uploader("1. Noon Sales Export", type=["csv", "xlsx", "xls"])
ad_file = st.sidebar.file_uploader("2. Noon Ad SKU Report", type=["csv", "xlsx", "xls"])
inv_file = st.sidebar.file_uploader("3. Noon Inventory Report", type=["csv", "xlsx", "xls"])

if sales_file and ad_file:
    # 1. Load Data
    sales_df_raw = load_flexible_df(sales_file)
    ad_df_raw = load_flexible_df(ad_file)
    inv_df_raw = load_flexible_df(inv_file) if inv_file else None

    # 2. Process Sales Data (Source of Truth)
    # Detect noon-specific sales column
    sales_val_col = 'gmv_lcy' if 'gmv_lcy' in sales_df_raw.columns else 'Net Sale (Price)'
    sales_df_raw[sales_val_col] = sales_df_raw[sales_val_col].apply(clean_numeric)
    
    # Aggregate sales by Noon SKU + Partner SKU
    sales_summary = sales_df_raw.groupby(['sku', 'partner_sku', 'brand_code']).agg({
        sales_val_col: 'sum'
    }).reset_index().rename(columns={sales_val_col: 'Total Sales', 'brand_code': 'Brand_Key'})
    
    sales_summary['Brand'] = sales_summary['Brand_Key'].map(BRAND_MAP).fillna(sales_summary['Brand_Key'])

    # 3. Process Ad Data
    # Normalizing column names as Noon reports can vary
    ad_df_raw = ad_df_raw.rename(columns={
        'Spends': 'Spend', 'Revenue': 'Ad_Sales', 'Views': 'Impressions'
    })
    
    ad_metrics = ['Spend', 'Ad_Sales', 'Clicks', 'Impressions', 'Orders']
    for c in ad_metrics:
        if c in ad_df_raw.columns:
            ad_df_raw[c] = ad_df_raw[c].apply(clean_numeric)

    # Aggregate by Campaign + Sku
    ad_camp_summary = ad_df_raw.groupby(['Campaign Name', 'Sku']).agg({
        'Spend': 'sum', 'Ad_Sales': 'sum', 'Clicks': 'sum', 'Impressions': 'sum', 'Orders': 'sum'
    }).reset_index()

    # Per-SKU Ad Totals for Organic isolation
    ad_sku_total = ad_df_raw.groupby('Sku').agg({
        'Ad_Sales': 'sum', 'Spend': 'sum'
    }).rename(columns={'Ad_Sales': 'SKU_AD_TOTAL_SALES', 'Spend': 'SKU_AD_TOTAL_SPEND'}).reset_index()

    # 4. Process Inventory
    if inv_df_raw is not None:
        inv_qty_col = 'qty' if 'qty' in inv_df_raw.columns else 'Quantity Available'
        inv_summary = inv_df_raw.groupby('sku')[inv_qty_col].sum().reset_index().rename(columns={inv_qty_col: 'Stock', 'sku': 'inv_sku'})
    else:
        inv_summary = pd.DataFrame(columns=['inv_sku', 'Stock'])

    # 5. Final Merge
    merged_df = pd.merge(sales_summary, ad_camp_summary, left_on='sku', right_on='Sku', how='left')
    merged_df = pd.merge(merged_df, ad_sku_total, left_on='sku', right_on='Sku', how='left').fillna(0)
    merged_df = pd.merge(merged_df, inv_summary, left_on='sku', right_on='inv_sku', how='left').fillna(0)

    merged_df['Campaign Name'] = merged_df['Campaign Name'].apply(lambda x: x if x != 0 and str(x).strip() != "" else "Organic / None")

    # 6. KPI Logic
    merged_df['Organic Sales'] = merged_df['Total Sales'] - merged_df['SKU_AD_TOTAL_SALES']
    merged_df['DRR'] = merged_df['Total Sales'] / 30
    merged_df['Ad Contribution %'] = (merged_df['SKU_AD_TOTAL_SALES'] / merged_df['Total Sales']).replace([np.inf, -np.inf], 0).fillna(0)
    
    merged_df['ROAS'] = (merged_df['Ad_Sales'] / merged_df['Spend']).replace([np.inf, -np.inf], 0).fillna(0)
    merged_df['ACOS'] = (merged_df['Spend'] / merged_df['Ad_Sales']).replace([np.inf, -np.inf], 0).fillna(0)
    merged_df['TACOS'] = (merged_df['SKU_AD_TOTAL_SPEND'] / merged_df['Total Sales']).replace([np.inf, -np.inf], 0).fillna(0)
    merged_df['CTR'] = (merged_df['Clicks'] / merged_df['Impressions']).replace([np.inf, -np.inf], 0).fillna(0)
    merged_df['CVR'] = (merged_df['Orders'] / merged_df['Clicks']).replace([np.inf, -np.inf], 0).fillna(0)

    # Formatting Table
    table_df = merged_df.rename(columns={
        'Campaign Name': 'Campaign', 'partner_sku': 'Partner SKU', 'sku': 'Noon SKU',
        'Ad_Sales': 'Ad Sales (Campaign)', 'Spend': 'Ad Spend (Campaign)'
    })

    tabs = st.tabs(["🌍 Portfolio Overview"] + sorted(list(BRAND_MAP.values())))

    def display_metrics_dashboard(raw_ad, raw_sales):
        t_sales = raw_sales[sales_val_col].sum()
        a_sales = raw_ad['Ad_Sales'].sum() if 'Ad_Sales' in raw_ad.columns else raw_ad['Revenue'].sum()
        o_sales = t_sales - a_sales
        t_spend = raw_ad['Spend'].sum() if 'Spend' in raw_ad.columns else raw_ad['Spends'].sum()
        
        st.markdown("#### 💰 Sales & Volume Metrics")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Sales", f"{t_sales:,.2f}")
        c2.metric("Ad Sales", f"{a_sales:,.2f}")
        c3.metric("Organic Sales", f"{o_sales:,.2f}")
        c4.metric("Ad Spend", f"{t_spend:,.2f}")
        c5.metric("Ad Contrib.", f"{(a_sales/t_sales):.1%}" if t_sales > 0 else "0%")

        st.markdown("#### ⚡ Efficiency & ROI")
        e1, e2, e3, e4, e5, e6 = st.columns(6)
        e1.metric("ROAS", f"{(a_sales/t_spend if t_spend > 0 else 0):.2f}")
        e2.metric("ACOS", f"{(t_spend/a_sales if a_sales > 0 else 0):.1%}")
        e3.metric("TACOS", f"{(t_spend/t_sales if t_sales > 0 else 0):.1%}")
        e4.metric("CTR", f"{(raw_ad['Clicks'].sum()/raw_ad['Impressions'].sum() if raw_ad['Impressions'].sum() > 0 else 0):.2%}")
        e5.metric("CVR", f"{(raw_ad['Orders'].sum()/raw_ad['Clicks'].sum() if raw_ad['Clicks'].sum() > 0 else 0):.2%}")
        e6.metric("DRR (Portfolio)", f"{(t_sales/30):,.2f}")

    cols_to_show = ['Campaign', 'Partner SKU', 'Noon SKU', 'Stock', 'Total Sales', 'DRR', 'Ad Sales (Campaign)', 'Ad Spend (Campaign)', 'Organic Sales', 'Ad Contribution %', 'ROAS', 'ACOS', 'TACOS', 'CTR', 'CVR']

    with tabs[0]:
        st.subheader("Global Noon Portfolio Overview")
        # For dashboard accuracy, rename headers back for the display function
        display_metrics_dashboard(ad_df_raw.rename(columns={'Spends':'Spend','Revenue':'Ad_Sales','Views':'Impressions'}), sales_df_raw)
        st.divider()
        st.dataframe(table_df[cols_to_show].sort_values(by='Total Sales', ascending=False), hide_index=True, use_container_width=True)

    for i, (key, b_name) in enumerate(sorted(BRAND_MAP.items())):
        with tabs[i+1]:
            b_data = table_df[table_df['Brand_Key'] == key]
            raw_ad_b = ad_df_raw[ad_df_raw['Campaign Name'].str.contains(key[:3].upper(), na=False)].rename(columns={'Spends':'Spend','Revenue':'Ad_Sales','Views':'Impressions'})
            raw_sales_b = sales_df_raw[sales_df_raw['brand_code'] == key]
            
            if not b_data.empty:
                st.subheader(f"{b_name} Brand Overview")
                display_metrics_dashboard(raw_ad_b, raw_sales_b)
                st.divider()
                st.dataframe(b_data[cols_to_show].sort_values(by='Total Sales', ascending=False), hide_index=True, use_container_width=True)
            else:
                st.warning(f"No active data for {b_name}.")

    # Multi-Sheet Export
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        table_df[cols_to_show].to_excel(writer, sheet_name='OVERVIEW', index=False)
        for b_key, b_name in BRAND_MAP.items():
            b_sheet = table_df[table_df['Brand_Key'] == b_key]
            if not b_sheet.empty:
                b_sheet[cols_to_show].to_excel(writer, sheet_name=b_name[:31], index=False)
    st.sidebar.download_button("📥 Download Master Report", data=output.getvalue(), file_name="Noon_Master_SKU_Audit.xlsx", use_container_width=True)
else:
    st.info("Upload Noon Sales, Ad SKU, and Inventory reports to begin.")
