import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="FINAL NOON SKU WISE", page_icon="🕛", layout="wide")

# Brand Mapping for Noon
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
    for col in df.columns:
        if any(kw.lower() in str(col).lower().strip() for kw in keywords):
            return col
    return None

st.title("🕛 FINAL NOON SKU WISE")
st.info("Mapping Ad SKU performance with Sales GMV and Inventory Stock levels.")

st.sidebar.header("Upload Noon Reports")
sales_file = st.sidebar.file_uploader("1. Noon Sales Export", type=["csv", "xlsx"])
# Allow multiple ad files (for different tabs like Product SKU and Brand SKU)
ad_files = st.sidebar.file_uploader("2. Noon Ad SKU Report Tabs (Upload Multiple)", type=["csv"], accept_multiple_files=True)
inv_file = st.sidebar.file_uploader("3. Noon Inventory Report", type=["csv", "xlsx"])

if sales_file and ad_files:
    def load_df(file):
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        df.columns = [str(c).strip() for c in df.columns]
        return df

    # 1. Load Sales Master
    sales_raw = load_df(sales_file)
    sl_sku = find_robust_col(sales_raw, ['sku'])
    sl_psku = find_robust_col(sales_raw, ['partner_sku', 'partner sku'])
    sl_brand = find_robust_col(sales_raw, ['brand_code', 'brand'])
    sl_rev = find_robust_col(sales_raw, ['gmv_lcy', 'revenue'])

    sales_raw[sl_rev] = sales_raw[sl_rev].apply(clean_numeric)
    sales_raw[sl_sku] = sales_raw[sl_sku].astype(str).str.strip()

    # 2. Combine and Filter Ad SKU Reports
    ad_list = []
    for f in ad_files:
        df_tmp = load_df(f)
        ad_sku_col = find_robust_col(df_tmp, ['sku'])
        if ad_sku_col:
            # Filter 'header' rows and strip whitespace
            df_tmp = df_tmp[df_tmp[ad_sku_col].astype(str).str.lower() != 'header'].copy()
            df_tmp[ad_sku_col] = df_tmp[ad_sku_col].astype(str).str.strip()
            # Clean ad metrics
            for c in ['Spends', 'Revenue', 'Clicks', 'Views', 'Orders']:
                col = find_robust_col(df_tmp, [c])
                if col: df_tmp[col] = df_tmp[col].apply(clean_numeric)
            ad_list.append(df_tmp)
    
    if ad_list:
        ads_combined = pd.concat(ad_list, ignore_index=True)
    else:
        st.error("No valid 'Sku' column found in any uploaded Ad reports.")
        st.stop()

    # 3. Process Inventory
    if inv_file:
        inv_raw = load_df(inv_file)
        inv_sku_col = find_robust_col(inv_raw, ['sku'])
        inv_qty_col = find_robust_col(inv_raw, ['qty', 'available'])
        inv_raw[inv_sku_col] = inv_raw[inv_sku_col].astype(str).str.strip()
        inv_grouped = inv_raw.groupby(inv_sku_col, as_index=False)[inv_qty_col].sum().rename(columns={inv_qty_col: 'Stock', inv_sku_col: 'inv_sku'})
    else:
        inv_grouped = pd.DataFrame(columns=['inv_sku', 'Stock'])

    # 4. Aggregation & Bridge
    # Sales Base
    sales_summary = sales_raw.groupby([sl_sku, sl_psku, sl_brand], as_index=False)[sl_rev].sum()
    sales_summary.columns = ['Noon SKU', 'Partner SKU', 'Brand_Key', 'Total Sales']
    
    # Ad Base
    ad_sku_bridge = find_robust_col(ads_combined, ['sku'])
    ad_camp_col = find_robust_col(ads_combined, ['campaign'])
    ad_rev_col = find_robust_col(ads_combined, ['revenue'])
    ad_spnd_col = find_robust_col(ads_combined, ['spends', 'spend'])
    
    ads_sku_totals = ads_combined.groupby(ad_sku_bridge, as_index=False).agg({ad_rev_col: 'sum', ad_spnd_col: 'sum'})
    ads_sku_totals.columns = ['ad_sku', 'SKU_AD_SALES', 'SKU_AD_SPEND']
    
    ads_camp_detail = ads_combined.groupby([ad_camp_col, ad_sku_bridge], as_index=False).agg({ad_rev_col: 'sum', ad_spnd_col: 'sum'})
    ads_camp_detail.columns = ['Campaign', 'camp_sku', 'Ad Sales (Campaign)', 'Ad Spend (Campaign)']

    # 5. Final Merge
    merged = pd.merge(sales_summary, ads_camp_detail, left_on='Noon SKU', right_on='camp_sku', how='left')
    merged = pd.merge(merged, ads_sku_totals, left_on='Noon SKU', right_on='ad_sku', how='left').fillna(0)
    merged = pd.merge(merged, inv_grouped, left_on='Noon SKU', right_on='inv_sku', how='left').fillna(0)

    # 6. Final Calculations
    merged['Brand'] = merged['Brand_Key'].map(BRAND_MAP).fillna(merged['Brand_Key'])
    merged['Campaign'] = merged['Campaign'].replace(0, "Organic / SEO")
    merged['Organic Sales'] = merged['Total Sales'] - merged['SKU_AD_SALES']
    merged['DRR'] = merged['Total Sales'] / 30
    merged['ROAS'] = (merged['Ad Sales (Campaign)'] / merged['Ad Spend (Campaign)']).replace([np.inf, -np.inf], 0).fillna(0)
    merged['TACOS'] = (merged['SKU_AD_SPEND'] / merged['Total Sales']).replace([np.inf, -np.inf], 0).fillna(0)

    # 7. UI Dashboard
    def render_dashboard(sales_df, ads_df):
        t_sales = sales_df[sl_rev].sum()
        a_sales = ads_df[find_robust_col(ads_df, ['revenue'])].sum()
        t_spend = ads_df[find_robust_col(ads_df, ['spends', 'spend'])].sum()
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Sales", f"{t_sales:,.2f}")
        c2.metric("Ad Sales", f"{a_sales:,.2f}")
        c3.metric("Organic Sales", f"{(t_sales - a_sales):,.2f}")
        c4.metric("Ad Spend", f"{t_spend:,.2f}")
        e1, e2, e3 = st.columns(3)
        e1.metric("ROAS", f"{(a_sales/t_spend if t_spend > 0 else 0):.2f}")
        e2.metric("TACOS", f"{(t_spend/t_sales if t_sales > 0 else 0):.1%}")
        e3.metric("Ad Contrib.", f"{(a_sales/t_sales if t_sales > 0 else 0):.1%}")

    tabs = st.tabs(["🌍 Overview"] + sorted(list(BRAND_MAP.values())))
    cols = ['Campaign', 'Partner SKU', 'Noon SKU', 'Stock', 'Total Sales', 'DRR', 'Ad Sales (Campaign)', 'Ad Spend (Campaign)', 'Organic Sales', 'ROAS', 'TACOS']

    with tabs[0]:
        render_dashboard(sales_raw, ads_combined)
        st.divider()
        st.dataframe(merged[cols].sort_values(by='Total Sales', ascending=False), hide_index=True, use_container_width=True)

    for i, (key, b_name) in enumerate(sorted(BRAND_MAP.items())):
        with tabs[i+1]:
            b_table = merged[merged['Brand_Key'] == key]
            if not b_table.empty:
                b_sales = sales_raw[sales_raw[sl_brand] == key]
                b_ads = ads_combined[ads_combined[ad_sku_bridge].isin(b_table['Noon SKU'])]
                render_dashboard(b_sales, b_ads)
                st.divider()
                st.dataframe(b_table[cols].sort_values(by='Total Sales', ascending=False), hide_index=True, use_container_width=True)

    # Multi-Sheet Excel Export
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        merged[cols].to_excel(writer, sheet_name='NOON_MASTER_AUDIT', index=False)
    st.sidebar.download_button("📥 Download Master Report", data=output.getvalue(), file_name="Noon_Master_Audit.xlsx", use_container_width=True)
else:
    st.info("Upload Sales, Ad Tabs, and Inventory to begin.")
