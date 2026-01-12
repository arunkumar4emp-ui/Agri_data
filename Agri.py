import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
from pathlib import Path
import numpy as np

# Page config
st.set_page_config(
    page_title="India Agri Dashboard",
    page_icon="🌾",
    layout="wide"
)

@st.cache_data
def load_data():
    
    """Load cleaned agricultural data."""
    db_path = Path("IndiaAgri.db")
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        query = "SELECT * FROM agri_data LIMIT 10000"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    else:
        st.error("Database not found! Run Agri.py first to create IndiaAgri.db")
        st.stop()

@st.cache_data
def get_crop_columns(df):
    """Extract crop-related columns."""
    crops = {}
    for col in df.columns:
        if '_production_tons' in col:
            crop = col.replace('_production_tons', '')
            crops[crop] = col
        if '_area_ha' in col:
            crop = col.replace('_area_ha', '')
            if crop not in crops:
                crops[crop] = col.replace('_area_ha', '_production_tons')
    return list(crops.keys())

def main():
    st.title("🌾 India Agriculture Dashboard")
    st.markdown("Interactive analysis of crop production across states and districts")
    
    df = load_data()
    st.success(f"Loaded {len(df):,} records from {df['year'].min()}-{df['year'].max()}")
    
    crops = get_crop_columns(df)
    
    # Sidebar filters
    st.sidebar.header("Filters")
    selected_crop = st.sidebar.selectbox("Select Crop", crops, index=0)
    selected_year = st.sidebar.slider(
        "Year",
        int(df['year'].min()),
        int(df['year'].max()),
        int(df['year'].max())
    )
    selected_state = st.sidebar.selectbox(
        "State",
        ["All"] + sorted(df['state_name'].unique().tolist())
    )
    
    # Filter data
    filtered_df = df[df['year'] == selected_year].copy()
    if selected_state != "All":
        filtered_df = filtered_df[filtered_df['state_name'] == selected_state]

    # ---- METRICS ----
    col1, col2, col3, col4 = st.columns(4)
    prod_col = f'{selected_crop}_production_tons'
    area_col = f'{selected_crop}_area_ha'

    if prod_col in filtered_df.columns:
        col1.metric("Total Production", f"{filtered_df[prod_col].sum():,.0f} tons")
    if area_col in filtered_df.columns:
        col2.metric("Total Area", f"{filtered_df[area_col].sum():,.0f} Ha")

    col3.metric("States Covered", filtered_df['state_name'].nunique())
    col4.metric("Districts", filtered_df['dist_name'].nunique())
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10 States by Production")
        if prod_col in filtered_df.columns:
            top_states = filtered_df.groupby('state_name')[prod_col].sum().nlargest(10)
            fig = px.bar(
                x=top_states.values,
                y=top_states.index,
                orientation='h',
                title=f"{selected_crop.title()} Production",
                labels={'x': 'Production (tons)', 'y': 'State'}
            )
            st.plotly_chart(fig, width='stretch')  # replaced use_container_width=True
    
    with col2:
        st.subheader("Production vs Area")
        if prod_col in filtered_df.columns and area_col in filtered_df.columns:
            fig = px.scatter(
                filtered_df,
                x=area_col,
                y=prod_col,
                size=prod_col,
                color='state_name',
                hover_name='dist_name',
                title=f"{selected_crop.title()} Yield Analysis"
            )
            st.plotly_chart(fig, width='stretch')  # replaced use_container_width=True
    
    # Trend analysis
    st.subheader("National Production Trend")
    if prod_col in df.columns:
        trend_df = df.groupby('year')[prod_col].sum().reset_index()
        fig = px.line(
            trend_df,
            x='year',
            y=prod_col,
            title=f"India {selected_crop.title()} Production Trend",
            markers=True
        )
        st.plotly_chart(fig, width='stretch')  # replaced use_container_width=True
    
    # Raw data viewer
    with st.expander("View Raw Data Sample"):
        st.dataframe(filtered_df.head(1000), width='stretch')

if __name__ == "__main__":
    main()