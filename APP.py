import streamlit as st
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
    if isinstance(val, str):
        cleaned = val.replace('AED', '').replace('$', '').replace('\xa0', '').replace(',', '').strip()
        try: return pd.to_numeric(cleaned)
        except: return 0.0
    return val if isinstance(val, (int, float)) else 0.0

def find_robust_col(df, keywords):
    """Deep keyword search across headers."""
    for col in df.columns:
        if any(kw.lower() in str(col).lower().strip() for kw in keywords):
            return col
    return None

st.title("🕛 FINAL NOON SKU WISE")
st.info("Verified Framework: SKU-Level Multi-Tab Mapping, Stock Integration, and Velocity tracking")

st.sidebar.header("Upload Noon Reports")
sales_file = st.sidebar.file_uploader("1. Noon Sales Export", type=["csv", "xlsx", "xls"])
ad_file = st.sidebar.file_uploader("2. Noon Ad (Product) SKU Report", type=["csv", "xlsx", "xls"])
inv_file = st.sidebar.file_uploader("3. Noon Inventory Report", type=["csv", "xlsx", "xls"])

if sales_file and ad_file:
    def load_df(file):
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        df.columns = [str(c).strip() for c in df.columns]
        return df

    sales_raw = load_df(sales_file)
    ads_raw = load_df(ad_file)
    inv_raw = load_df(inv_file) if inv_file else None

    # 1. Identify Key Columns
    ad_sku_col = find_robust_col(ads_raw, ['sku'])
    ad_camp_col = find_robust_col(ads_raw, ['campaign'])
    sl_sku_col = find_robust_col(sales_raw, ['sku'])
    sl_psku_col = find_robust_col(sales_raw, ['partner_sku', 'partner sku'])
    sl_sales_col = find_robust_col(sales_raw, ['gmv_lcy', 'revenue', 'price'])
    sl_brand_col = find_robust_col(sales_raw, ['brand_code', 'brand'])

    # 2. Cleanup Metrics
    ad_metrics_map = {
        'Spend': find_robust_col(ads_raw, ['spends', 'spend']),
        'AdSales': find_robust_col(ads_raw, ['revenue', 'sales']),
        'Clicks': find_robust_col(ads_raw, ['clicks']),
        'Views': find_robust_col(ads_raw, ['views', 'impressions']),
        'Orders': find_robust_col(ads_raw, ['orders'])
    }
    for k, col in ad_metrics_map.items():
        if col: ads_raw[col] = ads_raw[col].apply(clean_numeric)
    sales_raw[sl_sales_col] = sales_raw[sl_sales_col].apply(clean_numeric)

    # 3. Process Sales (SKU Master)
    sales_grouped = sales_raw.groupby([sl_sku_col, sl_psku_col, sl_brand_col])[sl_sales_col].sum().reset_index()
    sales_grouped.columns = ['Noon SKU', 'Partner SKU', 'Brand_Key', 'Total Sales']
    sales_grouped['Brand'] = sales_grouped['Brand_Key'].map(BRAND_MAP).fillna(sales_grouped['Brand_Key'])

    # 4. Process Ads (Campaign Level)
    ads_agg_cols = {col: 'sum' for col in ad_metrics_map.values() if col}
    ads_campaign_grouped = ads_raw.groupby([ad_camp_col, ad_sku_col]).agg(ads_agg_cols).reset_index()
    
    # SKU-level Ad Totals for Organic subtraction
    ads_sku_totals = ads_raw.groupby(ad_sku_col).agg({ad_metrics_map['AdSales']: 'sum', ad_metrics_map['Spend']: 'sum'}).reset_index()
    ads_sku_totals.columns = ['Ad_SKU_Match', 'SKU_AD_SALES', 'SKU_AD_SPEND']

    # 5. Process Inventory
    if inv_raw is not None:
        iv_sku_col = find_robust_col(inv_raw, ['sku'])
        iv_qty_col = find_robust_col(inv_raw, ['qty', 'available'])
        inv_grouped = inv_raw.groupby(iv_sku_col)[iv_qty_col].sum().reset_index()
        inv_grouped.columns = ['Inv_SKU_Match', 'Stock']
    else:
        inv_grouped = pd.DataFrame(columns=['Inv_SKU_Match', 'Stock'])

    # 6. Master Merge
    merged = pd.merge(sales_grouped, ads_campaign_grouped, left_on='Noon SKU', right_on=ad_sku_col, how='left')
    merged = pd.merge(merged, ads_sku_totals, left_on='Noon SKU', right_on='Ad_SKU_Match', how='left').fillna(0)
    merged = pd.merge(merged, inv_grouped, left_on='Noon SKU', right_on='Inv_SKU_Match', how='left').fillna(0)

    # Metrics Logic
    merged['Campaign'] = merged[ad_camp_col].apply(lambda x: x if x != 0 and str(x).strip() != "" else "Organic")
    merged['Organic Sales'] = merged['Total Sales'] - merged['SKU_AD_SALES']
    merged['DRR'] = merged['Total Sales'] / 30
    
    # Efficiency Calculations
    merged['ROAS'] = (merged[ad_metrics_map['AdSales']] / merged[ad_metrics_map['Spend']]).replace([np.inf, -np.inf], 0).fillna(0)
    merged['ACOS'] = (merged[ad_metrics_map['Spend']] / merged[ad_metrics_map['AdSales']]).replace([np.inf, -np.inf], 0).fillna(0)
    merged['TACOS'] = (merged['SKU_AD_TOTAL_SPEND'] / merged['Total Sales']).replace([np.inf, -np.inf], 0).fillna(0)
    merged['CTR'] = (merged[ad_metrics_map['Clicks']] / merged[ad_metrics_map['Views']]).replace([np.inf, -np.inf], 0).fillna(0)
    merged['CVR'] = (merged[ad_metrics_map['Orders']] / merged[ad_metrics_map['Clicks']]).replace([np.inf, -np.inf], 0).fillna(0)

    # 7. Multi-Tab Dashboard
    def show_metrics(raw_sales, raw_ads):
        t_sales = raw_sales[sl_sales_col].sum()
        a_sales = raw_ads[ad_metrics_map['AdSales']].sum()
        t_spend = raw_ads[ad_metrics_map['Spend'].sum()] if isinstance(ad_metrics_map['Spend'], str) else raw_ads[ad_metrics_map['Spend']].sum()
        
        st.markdown("#### 💰 Portfolio Overview (Pure Numbers)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Sales", f"{t_sales:,.2f}")
        c2.metric("Ad Sales", f"{a_sales:,.2f}")
        c3.metric("Organic Sales", f"{(t_sales - a_sales):,.2f}")
        c4.metric("Ad Spend", f"{t_spend:,.2f}")

        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric("ROAS", f"{(a_sales/t_spend if t_spend > 0 else 0):.2f}")
        e2.metric("ACOS", f"{(t_spend/a_sales if a_sales > 0 else 0):.1%}")
        e3.metric("TACOS", f"{(t_spend/t_sales if t_sales > 0 else 0):.1%}")
        e4.metric("CTR", f"{(raw_ads[ad_metrics_map['Clicks']].sum()/raw_ads[ad_metrics_map['Views']].sum() if raw_ads[ad_metrics_map['Views']].sum() > 0 else 0):.2%}")
        e5.metric("CVR", f"{(raw_ads[ad_metrics_map['Orders']].sum()/raw_ads[ad_metrics_map['Clicks']].sum() if raw_ads[ad_metrics_map['Clicks']].sum() > 0 else 0):.2%}")

    tabs = st.tabs(["🌍 Portfolio Overview"] + sorted(list(BRAND_MAP.values())))
    table_cols = ['Campaign', 'Partner SKU', 'Noon SKU', 'Stock', 'Total Sales', 'DRR', 'Ad Sales (Campaign)', 'Ad Spend', 'Organic Sales', 'ROAS', 'ACOS', 'TACOS', 'CTR', 'CVR']

    with tabs[0]:
        st.subheader("Global Performance")
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
                st.subheader(f"{b_name} Dashboard")
                show_metrics(b_sales, b_ads)
                st.divider()
                st.dataframe(b_table[table_cols].sort_values(by='Total Sales', ascending=False), hide_index=True, use_container_width=True)

    # Unified Export
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        merged[table_cols].to_excel(writer, sheet_name='MASTER_AUDIT', index=False)
    st.sidebar.download_button("📥 Download Master Report", data=output.getvalue(), file_name="Noon_Master_Audit.xlsx", use_container_width=True)
else:
    st.info("Upload Sales, Ad SKU, and Inventory reports to begin the audit.")
