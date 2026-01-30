import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="FINAL NOON SKU WISE", page_icon="🕛", layout="wide")

# Noon Brand Code Mapping (Matched to Noon Seller Lab Exports)
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

def find_robust_col(df, keywords, exclude=['acos', 'roas', 'cpc', 'ctr', 'rate', 'target']):
    """Dynamically finds the best matching column for a metric."""
    for col in df.columns:
        col_clean = str(col).strip().lower()
        if any(kw.lower() in col_clean for kw in keywords):
            if not any(ex.lower() in col_clean for ex in exclude):
                return col
    return None

st.title("🕛 FINAL NOON SKU WISE")
st.info("Verified Framework: Campaign-First SKU View, Organic Isolation, and Stock Tracking")

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

    # Detect Critical Columns
    ad_sku_col = find_robust_col(ad_df_raw, ['sku'])
    ad_camp_col = find_robust_col(ad_df_raw, ['campaign'])
    biz_sku_col = find_robust_col(sales_df_raw, ['sku'])
    biz_psku_col = find_robust_col(sales_df_raw, ['partner_sku', 'partner sku'])
    biz_sales_col = find_robust_col(sales_df_raw, ['gmv_lcy', 'revenue', 'price'])
    biz_brand_col = find_robust_col(sales_df_raw, ['brand_code', 'brand'])

    # Verification Step: Stop if Sku column is missing from Ad Report
    if not ad_sku_col:
        st.error("❌ 'Sku' column not found in Ad Report. Please ensure you are uploading the '(Product) Sku' report from the Noon Ad Center.")
        st.stop()

    # 2. Clean Data
    sales_df_raw[biz_sales_col] = sales_df_raw[biz_sales_col].apply(clean_numeric)
    
    # Process Ad metrics
    ad_metrics = {
        'Spend': find_robust_col(ad_df_raw, ['spends', 'spend']),
        'AdSales': find_robust_col(ad_df_raw, ['revenue', 'sales']),
        'Clicks': find_robust_col(ad_df_raw, ['clicks']),
        'Views': find_robust_col(ad_df_raw, ['views', 'impressions']),
        'Orders': find_robust_col(ad_df_raw, ['orders'])
    }
    for k, col in ad_metrics.items():
        if col: ad_df_raw[col] = ad_df_raw[col].apply(clean_numeric)

    # 3. Aggregate Sales by SKU
    sales_summary = sales_df_raw.groupby([biz_sku_col, biz_psku_col, biz_brand_col]).agg({
        biz_sales_col: 'sum'
    }).reset_index().rename(columns={biz_sales_col: 'Total Sales', biz_brand_col: 'Brand_Key'})
    
    sales_summary['Brand'] = sales_summary['Brand_Key'].map(BRAND_MAP).fillna(sales_summary['Brand_Key'])

    # 4. Aggregate Ad Data (Campaign + Sku)
    ad_camp_summary = ad_df_raw.groupby([ad_camp_col, ad_sku_col]).agg({
        ad_metrics['Spend']: 'sum', ad_metrics['AdSales']: 'sum', 
        ad_metrics['Clicks']: 'sum', ad_metrics['Views']: 'sum', ad_metrics['Orders']: 'sum'
    }).reset_index()

    # Per-SKU Ad Totals for Organic isolation
    ad_sku_total = ad_df_raw.groupby(ad_sku_col).agg({
        ad_metrics['AdSales']: 'sum', ad_metrics['Spend']: 'sum'
    }).rename(columns={ad_metrics['AdSales']: 'SKU_AD_TOTAL_SALES', ad_metrics['Spend']: 'SKU_AD_TOTAL_SPEND'}).reset_index()

    # 5. Process Inventory
    if inv_df_raw is not None:
        inv_sku_col = find_robust_col(inv_df_raw, ['sku'])
        inv_qty_col = find_robust_col(inv_df_raw, ['qty', 'quantity'])
        inv_summary = inv_df_raw.groupby(inv_sku_col)[inv_qty_col].sum().reset_index().rename(columns={inv_qty_col: 'Stock', inv_sku_col: 'inv_sku'})
    else:
        inv_summary = pd.DataFrame(columns=['inv_sku', 'Stock'])

    # 6. Final Merge
    merged_df = pd.merge(sales_summary, ad_camp_summary, left_on=biz_sku_col, right_on=ad_sku_col, how='left')
    merged_df = pd.merge(merged_df, ad_sku_total, left_on=biz_sku_col, right_on=ad_sku_col, how='left').fillna(0)
    merged_df = pd.merge(merged_df, inv_summary, left_on=biz_sku_col, right_on='inv_sku', how='left').fillna(0)

    merged_df[ad_camp_col] = merged_df[ad_camp_col].apply(lambda x: x if x != 0 and str(x).strip() != "" else "Organic / No Ads")

    # 7. KPI Calculations
    merged_df['Organic Sales'] = merged_df['Total Sales'] - merged_df['SKU_AD_TOTAL_SALES']
    merged_df['DRR'] = merged_df['Total Sales'] / 30
    merged_df['Ad Contribution %'] = (merged_df['SKU_AD_TOTAL_SALES'] / merged_df['Total Sales']).replace([np.inf, -np.inf], 0).fillna(0)
    
    merged_df['ROAS'] = (merged_df[ad_metrics['AdSales']] / merged_df[ad_metrics['Spend']]).replace([np.inf, -np.inf], 0).fillna(0)
    merged_df['ACOS'] = (merged_df[ad_metrics['Spend']] / merged_df[ad_metrics['AdSales']]).replace([np.inf, -np.inf], 0).fillna(0)
    merged_df['TACOS'] = (merged_df['SKU_AD_TOTAL_SPEND'] / merged_df['Total Sales']).replace([np.inf, -np.inf], 0).fillna(0)
    merged_df['CTR'] = (merged_df[ad_metrics['Clicks']] / merged_df[ad_metrics['Views']]).replace([np.inf, -np.inf], 0).fillna(0)
    merged_df['CVR'] = (merged_df[ad_metrics['Orders']] / merged_df[ad_metrics['Clicks']]).replace([np.inf, -np.inf], 0).fillna(0)

    # Clean Table for UI
    table_df = merged_df.rename(columns={
        ad_camp_col: 'Campaign', biz_psku_col: 'Partner SKU', biz_sku_col: 'Noon SKU',
        ad_metrics['AdSales']: 'Ad Sales (Campaign)', ad_metrics['Spend']: 'Ad Spend'
    })

    tabs = st.tabs(["🌍 Portfolio Overview"] + sorted([BRAND_MAP[k] for k in BRAND_MAP if k in sales_summary['Brand_Key'].unique()]))

    def display_metrics_dashboard(raw_ad, raw_sales):
        t_sales = raw_sales[biz_sales_col].sum()
        a_sales = raw_ad[ad_metrics['AdSales']].sum()
        o_sales = t_sales - a_sales
        t_spend = raw_ad[ad_metrics['Spend']].sum()
        
        st.markdown("#### 💰 Sales & Efficiency Dashboard")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Sales", f"{t_sales:,.2f}")
        c2.metric("Ad Sales", f"{a_sales:,.2f}")
        c3.metric("Organic Sales", f"{o_sales:,.2f}")
        c4.metric("Ad Spend", f"{t_spend:,.2f}")
        c5.metric("Ad Contrib.", f"{(a_sales/t_sales):.1%}" if t_sales > 0 else "0%")

        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric("ROAS", f"{(a_sales/t_spend if t_spend > 0 else 0):.2f}")
        e2.metric("ACOS", f"{(t_spend/a_sales if a_sales > 0 else 0):.1%}")
        e3.metric("TACOS", f"{(t_spend/t_sales if t_sales > 0 else 0):.1%}")
        e4.metric("CTR", f"{(raw_ad[ad_metrics['Clicks']].sum()/raw_ad[ad_metrics['Views']].sum() if raw_ad[ad_metrics['Views']].sum() > 0 else 0):.2%}")
        e5.metric("CVR", f"{(raw_ad[ad_metrics['Orders']].sum()/raw_ad[ad_metrics['Clicks']].sum() if raw_ad[ad_metrics['Clicks']].sum() > 0 else 0):.2%}")

    cols_to_show = ['Campaign', 'Partner SKU', 'Noon SKU', 'Stock', 'Total Sales', 'DRR', 'Ad Sales (Campaign)', 'Ad Spend', 'Organic Sales', 'Ad Contribution %', 'ROAS', 'ACOS', 'TACOS', 'CTR', 'CVR']

    with tabs[0]:
        st.subheader("Global Noon Portfolio Overview")
        display_metrics_dashboard(ad_df_raw, sales_df_raw)
        st.divider()
        st.dataframe(table_df[cols_to_show].sort_values(by='Total Sales', ascending=False), hide_index=True, use_container_width=True)

    for i, (key, b_name) in enumerate(sorted(BRAND_MAP.items())):
        if key in sales_summary['Brand_Key'].unique():
            with tabs[i+1]:
                b_data = table_df[table_df['Brand_Key'] == key]
                # Filter raw reports by internal Noon sku to ensure 100% accuracy in dash
                raw_skus = sales_summary[sales_summary['Brand_Key'] == key][biz_sku_col].unique()
                raw_ad_b = ad_df_raw[ad_df_raw[ad_sku_col].isin(raw_skus)]
                raw_sales_b = sales_df_raw[sales_df_raw[biz_sku_col].isin(raw_skus)]
                
                st.subheader(f"{b_name} Overview")
                display_metrics_dashboard(raw_ad_b, raw_sales_b)
                st.divider()
                st.dataframe(b_data[cols_to_show].sort_values(by='Total Sales', ascending=False), hide_index=True, use_container_width=True)

    # Multi-Sheet Export
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        table_df[cols_to_show].to_excel(writer, sheet_name='OVERVIEW', index=False)
    st.sidebar.download_button("📥 Download Master Report", data=output.getvalue(), file_name="Noon_Master_SKU_Audit.xlsx", use_container_width=True)
else:
    st.info("Upload Noon Reports to begin.")
