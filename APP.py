import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

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
    """Strips AED, currency symbols, and commas to return pure floats."""
    if isinstance(val, str):
        cleaned = val.replace('AED', '').replace('$', '').replace('\xa0', '').replace(',', '').strip()
        try: return pd.to_numeric(cleaned)
        except: return 0.0
    return val if isinstance(val, (int, float)) else 0.0

def find_robust_col(df, keywords, exclude=['acos', 'roas', 'cpc', 'ctr', 'rate', 'target']):
    """Finds columns dynamically across Noon's various report formats."""
    for col in df.columns:
        col_clean = str(col).strip().lower()
        if any(kw.lower() in col_clean for kw in keywords):
            if not any(ex.lower() in col_clean for ex in exclude):
                return col
    return None

st.title("🕛 FINAL NOON SKU WISE")
st.info("Noon Audit Framework: Campaign-First View, Organic Isolation, and Stock Tracking")

st.sidebar.header("Upload Noon Reports")
sales_file = st.sidebar.file_uploader("1. Noon Sales Export (CSV/Excel)", type=["csv", "xlsx", "xls"])
ad_file = st.sidebar.file_uploader("2. Noon Ad SKU Report (CSV/Excel)", type=["csv", "xlsx", "xls"])
inv_file = st.sidebar.file_uploader("3. Noon Inventory Report (CSV/Excel)", type=["csv", "xlsx", "xls"])

if sales_file and ad_file:
    def load_df(file):
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        df.columns = [str(c).strip() for c in df.columns]
        return df

    sales_raw = load_df(sales_file)
    ads_raw = load_df(ad_file)
    inv_raw = load_df(inv_file) if inv_file else None

    # 1. Identify Columns
    # Ads Report
    ad_sku_col = find_robust_col(ads_raw, ['sku'])
    ad_camp_col = find_robust_col(ads_raw, ['campaign'])
    ad_spend_col = find_robust_col(ads_raw, ['spends', 'spend'])
    ad_sales_col = find_robust_col(ads_raw, ['revenue', 'sales'])
    ad_clicks_col = find_robust_col(ads_raw, ['clicks'])
    ad_views_col = find_robust_col(ads_raw, ['views', 'impressions'])
    ad_orders_col = find_robust_col(ads_raw, ['orders'])

    # Sales Report
    sl_sku_col = find_robust_col(sales_raw, ['sku'])
    sl_psku_col = find_robust_col(sales_raw, ['partner_sku', 'partner sku'])
    sl_sales_col = find_robust_col(sales_raw, ['gmv_lcy', 'revenue', 'price'])
    sl_brand_col = find_robust_col(sales_raw, ['brand_code', 'brand'])

    # Inventory Report
    iv_sku_col = find_robust_col(inv_raw, ['sku']) if inv_raw is not None else None
    iv_qty_col = find_robust_col(inv_raw, ['qty', 'available']) if inv_raw is not None else None

    # Safety Check
    if not ad_sku_col or not sl_sku_col:
        st.error("❌ Could not detect 'SKU' columns. Please ensure you are uploading the '(Product) SKU' report for ads.")
        st.stop()

    # 2. Numeric Cleaning
    for c in [ad_spend_col, ad_sales_col, ad_clicks_col, ad_views_col, ad_orders_col]:
        if c: ads_raw[c] = ads_raw[c].apply(clean_numeric)
    sales_raw[sl_sales_col] = sales_raw[sl_sales_col].apply(clean_numeric)
    if inv_raw is not None: inv_raw[iv_qty_col] = inv_raw[iv_qty_col].apply(clean_numeric)

    # 3. Process Sales (Anchor)
    sales_grouped = sales_raw.groupby([sl_sku_col, sl_psku_col, sl_brand_col]).agg({sl_sales_col: 'sum'}).reset_index()
    sales_grouped.columns = ['Noon SKU', 'Partner SKU', 'Brand_Key', 'Total Sales']
    sales_grouped['Brand'] = sales_grouped['Brand_Key'].map(BRAND_MAP).fillna(sales_grouped['Brand_Key'])

    # 4. Process Ads
    # Campaign + SKU level
    ads_campaign_grouped = ads_raw.groupby([ad_camp_col, ad_sku_col]).agg({
        ad_spend_col: 'sum', ad_sales_col: 'sum', ad_clicks_col: 'sum', ad_views_col: 'sum', ad_orders_col: 'sum'
    }).reset_index()
    
    # SKU level totals (for organic subtraction)
    ads_sku_totals = ads_raw.groupby(ad_sku_col).agg({ad_sales_col: 'sum', ad_spend_col: 'sum'}).reset_index()
    ads_sku_totals.columns = ['Ad_SKU', 'SKU_AD_SALES', 'SKU_AD_SPEND']

    # 5. Process Inventory
    if inv_raw is not None:
        inv_grouped = inv_raw.groupby(iv_sku_col)[iv_qty_col].sum().reset_index()
        inv_grouped.columns = ['Inv_SKU', 'Stock']
    else:
        inv_grouped = pd.DataFrame(columns=['Inv_SKU', 'Stock'])

    # 6. Final Merge
    merged = pd.merge(sales_grouped, ads_campaign_grouped, left_on='Noon SKU', right_on=ad_sku_col, how='left')
    merged = pd.merge(merged, ads_sku_totals, left_on='Noon SKU', right_on='Ad_SKU', how='left').fillna(0)
    merged = pd.merge(merged, inv_grouped, left_on='Noon SKU', right_on='Inv_SKU', how='left').fillna(0)

    # Clean Up
    merged['Campaign'] = merged[ad_camp_col].apply(lambda x: x if x != 0 and str(x).strip() != "" else "Organic")

    # Metrics Calculations
    merged['Organic Sales'] = merged['Total Sales'] - merged['SKU_AD_SALES']
    merged['DRR'] = merged['Total Sales'] / 30
    merged['Ad Contribution %'] = (merged['SKU_AD_SALES'] / merged['Total Sales']).replace([np.inf, -np.inf], 0).fillna(0)
    
    merged['ROAS'] = (merged[ad_sales_col] / merged[ad_spend_col]).replace([np.inf, -np.inf], 0).fillna(0)
    merged['ACOS'] = (merged[ad_spend_col] / merged[ad_sales_col]).replace([np.inf, -np.inf], 0).fillna(0)
    merged['TACOS'] = (merged['SKU_AD_SPEND'] / merged['Total Sales']).replace([np.inf, -np.inf], 0).fillna(0)
    merged['CTR'] = (merged[ad_clicks_col] / merged[ad_views_col]).replace([np.inf, -np.inf], 0).fillna(0)
    merged['CVR'] = (merged[ad_orders_col] / merged[ad_clicks_col]).replace([np.inf, -np.inf], 0).fillna(0)

    # 7. Dashboard Function
    def show_header_metrics(raw_sales, raw_ads):
        t_sales = raw_sales[sl_sales_col].sum()
        a_sales = raw_ads[ad_sales_col].sum()
        o_sales = t_sales - a_sales
        t_spend = raw_ads[ad_spend_col].sum()
        t_clicks = raw_ads[ad_clicks_col].sum()
        t_views = raw_ads[ad_views_col].sum()
        t_orders = raw_ads[ad_orders_col].sum()

        st.markdown("#### 💰 Sales & Volume (Pure Numbers)")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Sales", f"{t_sales:,.2f}")
        c2.metric("Ad Sales", f"{a_sales:,.2f}")
        c3.metric("Organic Sales", f"{o_sales:,.2f}")
        c4.metric("Ad Spend", f"{t_spend:,.2f}")
        c5.metric("Ad Contrib.", f"{(a_sales/t_sales):.1%}" if t_sales > 0 else "0%")

        st.markdown("#### ⚡ Efficiency & Contribution")
        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric("ROAS", f"{(a_sales/t_spend if t_spend > 0 else 0):.2f}")
        e2.metric("ACOS", f"{(t_spend/a_sales if a_sales > 0 else 0):.1%}")
        e3.metric("TACOS", f"{(t_spend/t_sales if t_sales > 0 else 0):.1%}")
        e4.metric("CTR", f"{(t_clicks/t_views if t_views > 0 else 0):.2%}")
        e5.metric("CVR", f"{(t_orders/t_clicks if t_clicks > 0 else 0):.2%}")

    tabs = st.tabs(["🌍 Portfolio Overview"] + sorted(list(BRAND_MAP.values())))

    table_cols = ['Campaign', 'Partner SKU', 'Noon SKU', 'Stock', 'Total Sales', 'DRR', 'Ad Sales (Campaign)', 'Ad Spend', 'Organic Sales', 'Ad Contribution %', 'ROAS', 'ACOS', 'TACOS', 'CTR', 'CVR']

    with tabs[0]:
        st.subheader("Noon Global Portfolio Overview")
        show_header_metrics(sales_raw, ads_raw)
        st.divider()
        st.dataframe(merged[table_cols].sort_values(by='Total Sales', ascending=False), hide_index=True, use_container_width=True)

    for i, (b_key, b_name) in enumerate(sorted(BRAND_MAP.items())):
        with tabs[i+1]:
            brand_sales = sales_raw[sales_raw[sl_brand_col] == b_key]
            brand_skus = brand_sales[sl_sku_col].unique()
            brand_ads = ads_raw[ads_raw[ad_sku_col].isin(brand_skus)]
            brand_table = merged[merged['Brand_Key'] == b_key]

            if not brand_table.empty:
                st.subheader(f"{b_name} Overview")
                show_header_metrics(brand_sales, brand_ads)
                st.divider()
                st.dataframe(brand_table[table_cols].sort_values(by='Total Sales', ascending=False), hide_index=True, use_container_width=True)
            else:
                st.warning(f"No active data for {b_name}.")

    # Multi-Sheet Export
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        merged[table_cols].to_excel(writer, sheet_name='OVERVIEW', index=False)
    st.sidebar.download_button("📥 Download Master Audit", data=output.getvalue(), file_name="Noon_Master_Audit.xlsx", use_container_width=True)
else:
    st.info("Upload Sales, Ad SKU, and Inventory reports to begin the audit.")
