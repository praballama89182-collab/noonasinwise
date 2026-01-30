import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="FINAL NOON SKU WISE", page_icon="🕛", layout="wide")

# Noon Brand Mapping (Matches your Sales brand_code)
BRAND_MAP = {
    'maison_de_lavenir': 'Maison de l’Avenir',
    'creation_lamis': 'Creation Lamis',
    'jean_paul_dupont': 'Jean Paul Dupont',
    'paris_collection': 'Paris Collection',
    'dorall_collection': 'Dorall Collection',
    'cp_trendies': 'CP Trendies'
}

def clean_numeric(val):
    """Deep numeric clean for Pure Numbers (Stripping AED/commas)."""
    if isinstance(val, str):
        cleaned = val.replace('AED', '').replace('$', '').replace('\xa0', '').replace(',', '').strip()
        try: return pd.to_numeric(cleaned)
        except: return 0.0
    return val if isinstance(val, (int, float)) else 0.0

def find_col(df, keywords):
    """Finds column name regardless of casing/spaces."""
    for col in df.columns:
        if any(kw.lower() in str(col).lower().strip() for kw in keywords):
            return col
    return None

st.title("🕛 FINAL NOON SKU WISE")
st.info("Consolidated Noon Audit: Mapping Ad SKU, Sales Export, and Inventory (Pure Numbers)")

st.sidebar.header("Upload Noon Reports")
sales_file = st.sidebar.file_uploader("1. Noon Sales Export", type=["csv", "xlsx"])
ad_files = st.sidebar.file_uploader("2. Noon Ad SKU Reports (Multiple Allowed)", type=["csv", "xlsx"], accept_multiple_files=True)
inv_file = st.sidebar.file_uploader("3. Noon Inventory Report", type=["csv", "xlsx"])

if sales_file and ad_files:
    # 1. Load Sales (The Anchor)
    sales_raw = pd.read_csv(sales_file) if sales_file.name.endswith('.csv') else pd.read_excel(sales_file)
    sales_raw.columns = [str(c).strip() for c in sales_raw.columns]
    
    sl_sku = find_col(sales_raw, ['sku'])
    sl_psku = find_col(sales_raw, ['partner_sku', 'partner sku'])
    sl_brand = find_col(sales_raw, ['brand_code', 'brand'])
    sl_rev = find_col(sales_raw, ['gmv_lcy', 'revenue'])

    sales_raw[sl_rev] = sales_raw[sl_rev].apply(clean_numeric)
    
    # 2. Process Ad SKU Data (Flexible Casing & Header Filter)
    ad_list = []
    for f in ad_files:
        df_tmp = pd.read_csv(f) if f.name.endswith('.csv') else pd.read_excel(f)
        df_tmp.columns = [str(c).strip() for c in df_tmp.columns]
        
        # Noon uses 'Sku' (Capitalized) in Ad reports. Checking robustly:
        actual_ad_sku_col = find_col(df_tmp, ['sku'])
        
        if actual_ad_sku_col:
            # Filter 'header' rows found in Noon Brand reports
            df_tmp = df_tmp[df_tmp[actual_ad_sku_col].astype(str).str.lower() != 'header'].copy()
            df_tmp['sku_bridge'] = df_tmp[actual_ad_sku_col].astype(str).str.strip()
            ad_list.append(df_tmp)
    
    if ad_list:
        ads_combined = pd.concat(ad_list, ignore_index=True)
        # Identify metric columns robustly
        ad_rev_col = find_col(ads_combined, ['revenue'])
        ad_spnd_col = find_col(ads_combined, ['spends', 'spend'])
        ad_camp_col = find_col(ads_combined, ['campaign'])
        
        for c in [ad_rev_col, ad_spnd_col, 'Clicks', 'Views', 'Orders']:
            if c in ads_combined.columns: ads_combined[c] = ads_combined[c].apply(clean_numeric)
    else:
        st.error("Could not find a 'Sku' column in any uploaded Ad reports.")
        st.stop()

    # 3. Process Inventory
    if inv_file:
        inv_raw = pd.read_csv(inv_file) if inv_file.name.endswith('.csv') else pd.read_excel(inv_file)
        inv_raw.columns = [str(c).strip() for c in inv_raw.columns]
        inv_grouped = inv_raw.groupby('sku', as_index=False)['qty'].sum().rename(columns={'qty': 'Stock', 'sku': 'inv_sku'})
    else:
        inv_grouped = pd.DataFrame(columns=['inv_sku', 'Stock'])

    # 4. Consolidate and Bridge
    # Sales grouped by SKU
    sales_summary = sales_raw.groupby([sl_sku, sl_psku, sl_brand], as_index=False)[sl_rev].sum()
    sales_summary.columns = ['Noon SKU', 'Partner SKU', 'Brand_Key', 'Total Sales']
    
    # Ad totals per SKU
    ads_sku_totals = ads_combined.groupby('sku_bridge', as_index=False).agg({ad_rev_col: 'sum', ad_spnd_col: 'sum'})
    ads_sku_totals.columns = ['ad_sku', 'Ad Sales (Total SKU)', 'Ad Spend (Total SKU)']
    
    # Ad Campaign-level detail
    ads_camp_grouped = ads_combined.groupby([ad_camp_col, 'sku_bridge'], as_index=False).agg({ad_rev_col: 'sum', ad_spnd_col: 'sum'})
    ads_camp_grouped.columns = ['Campaign', 'camp_sku', 'Ad Sales (Campaign)', 'Ad Spend (Campaign)']

    # Final Merge
    merged = pd.merge(sales_summary, ads_camp_grouped, left_on='Noon SKU', right_on='camp_sku', how='left')
    merged = pd.merge(merged, ads_sku_totals, left_on='Noon SKU', right_on='ad_sku', how='left').fillna(0)
    merged = pd.merge(merged, inv_grouped, left_on='Noon SKU', right_on='inv_sku', how='left').fillna(0)

    # Final Calculations
    merged['Brand'] = merged['Brand_Key'].map(BRAND_MAP).fillna(merged['Brand_Key'])
    merged['Campaign'] = merged['Campaign'].replace(0, "Organic / SEO")
    merged['Organic Sales'] = merged['Total Sales'] - merged['Ad Sales (Total SKU)']
    merged['DRR'] = merged['Total Sales'] / 30
    merged['ROAS'] = (merged['Ad Sales (Campaign)'] / merged['Ad Spend (Campaign)']).replace([np.inf, -np.inf], 0).fillna(0)
    merged['TACOS'] = (merged['Ad Spend (Total SKU)'] / merged['Total Sales']).replace([np.inf, -np.inf], 0).fillna(0)

    # 5. Dashboard Engine
    def render_dash(sales_df, ads_df):
        t_sales = sales_df[sl_rev].sum()
        a_sales = ads_df[ad_rev_col].sum()
        t_spend = ads_df[ad_spnd_col].sum()
        st.markdown("#### 💰 Portfolio Overview (Numeric)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Sales", f"{t_sales:,.2f}")
        c2.metric("Ad Sales", f"{a_sales:,.2f}")
        c3.metric("Organic Sales", f"{(t_sales - a_sales):,.2f}")
        c4.metric("Ad Spend", f"{t_spend:,.2f}")
        e1, e2 = st.columns(2)
        e1.metric("ROAS", f"{(a_sales/t_spend if t_spend > 0 else 0):.2f}")
        e2.metric("TACOS", f"{(t_spend/t_sales if t_sales > 0 else 0):.1%}")

    tabs = st.tabs(["🌍 Portfolio Overview"] + sorted(list(BRAND_MAP.values())))
    cols = ['Campaign', 'Partner SKU', 'Noon SKU', 'Stock', 'Total Sales', 'DRR', 'Ad Sales (Campaign)', 'Ad Spend (Campaign)', 'Organic Sales', 'ROAS', 'TACOS']

    with tabs[0]:
        render_dash(sales_raw, ads_combined)
        st.divider()
        st.dataframe(merged[cols].sort_values(by='Total Sales', ascending=False), hide_index=True, use_container_width=True)

    for i, (key, b_name) in enumerate(sorted(BRAND_MAP.items())):
        with tabs[i+1]:
            b_table = merged[merged['Brand_Key'] == key]
            if not b_table.empty:
                b_sales = sales_raw[sales_raw[sl_brand] == key]
                b_ads = ads_combined[ads_combined['sku_bridge'].isin(b_table['Noon SKU'])]
                render_dash(b_sales, b_ads)
                st.divider()
                st.dataframe(b_table[cols].sort_values(by='Total Sales', ascending=False), hide_index=True, use_container_width=True)

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        merged[cols].to_excel(writer, sheet_name='MASTER_AUDIT', index=False)
    st.sidebar.download_button("📥 Download Master Report", data=output.getvalue(), file_name="Noon_SKU_Audit.xlsx", use_container_width=True)
else:
    st.info("Upload Noon Sales, Ads, and Inventory reports to begin.")
