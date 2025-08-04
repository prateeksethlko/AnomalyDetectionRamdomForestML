import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Integrated Anomaly Detection", layout="wide")
st.header("Integrated Model & Rule-Based Anomaly Detection")
st.markdown(
    "This tool supports anomaly detection using Isolation Forest (model-based) "
    "and rule-based comparison to a master dataset. Upload your files, set thresholds, "
    "and visualize detected anomalies below."
)

# Compact right panel for uploads/settings and expander for minimal space
row1_left, row1_right = st.columns([3, 1])

with row1_right:
    with st.expander("Upload Datasets & Settings", expanded=True):
        master_file = st.file_uploader("Master/Training Dataset (optional)", type=["csv"], key="master")
        target_file = st.file_uploader("Dataset to Analyze", type=["csv"], key="target")
        st.caption("Once uploaded, select columns at left.")
        contamination = st.slider(
            "Anomaly Proportion (%)", 1, 20, 2,
            help="Estimated percent of data expected as anomalies"
        )
        multiplier = st.number_input(
            "Rule Threshold (× mean)", min_value=1.0, value=7.0,
            help="Flag value > multiplier × mean (from master dataset)"
        )

# After file uploads, enable column selection (left) and show results
master_df = None
df = None
numeric_cols = []
selected_model_cols = []
rulebased_columns = []

if target_file is not None:
    try:
        df = pd.read_csv(target_file)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    except Exception as e:
        st.error(f"Error reading analysis dataset: {e}")

    if master_file is not None:
        try:
            master_df = pd.read_csv(master_file)
        except Exception as e:
            st.error(f"Error loading training file: {e}")

    with row1_left:
        st.subheader("Column Selection")
        selected_model_cols = st.multiselect(
            "Model-Based Columns:",
            options=numeric_cols,
            default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols,
            help="Columns used in Isolation Forest anomaly detection"
        )
        rulebased_columns = st.multiselect(
            "Rule-Based Columns (value > multiplier × mean):",
            options=numeric_cols,
            default=numeric_cols[:1] if numeric_cols else [],
            help="Compare to historical means from master dataset"
        )

    st.success(f"Uploaded analysis dataset: {target_file.name}")
    with st.expander("Preview Analysis Data"):
        st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")

    if st.button("Run Anomaly Detection"):
        # Model-Based Anomaly Detection (Isolation Forest)
        if selected_model_cols:
            X = df[selected_model_cols].copy().dropna()
            if master_df is not None and all(c in master_df.columns for c in selected_model_cols):
                train_X = master_df[selected_model_cols].copy().dropna()
                clf = IsolationForest(contamination=contamination / 100, random_state=42)
                clf.fit(train_X)
                preds = clf.predict(X)
            else:
                clf = IsolationForest(contamination=contamination / 100, random_state=42)
                preds = clf.fit_predict(X)
            anomalies_isoforest = preds == -1
        else:
            anomalies_isoforest = np.zeros(len(df), dtype=bool)

        # Rule-Based Detection
        anomalies_rulebased = np.zeros(len(df), dtype=bool)
        for col in rulebased_columns:
            if master_df is not None and col in master_df.columns and col in df.columns:
                mean_val = master_df[col].mean()
                anomalies_rulebased |= df[col] > multiplier * mean_val

        # Combine anomaly types
        combined_anomalies = anomalies_isoforest | anomalies_rulebased
        anomalydf = df.iloc[np.where(combined_anomalies)[0]].copy()

        st.divider()
        st.metric("Anomalies Detected", len(anomalydf))
        st.markdown("### Detected Anomalies")
        st.dataframe(anomalydf, use_container_width=True)

        st.markdown("### Anomaly Visualizations")

        # --- Plotting function with distinct colors ---
        def plot_anomalies(df, column, anomalydf, model_mask, rule_mask):
            fig = px.scatter(df, x=df.index, y=column, opacity=0.6,
                             title=f"Anomaly Detection for {column}",
                             labels={"y": column, "x": "Index"})
            avg_value = df[column].mean()
            fig.add_trace(go.Scatter(
                x=df.index, y=[avg_value]*len(df),
                mode='lines', name='Mean',
                line=dict(color='green', width=6, dash='dash')
            ))

            # Model-based only: RED
            model_only_idx = anomalydf.index[model_mask[anomalydf.index] & ~rule_mask[anomalydf.index]]
            if len(model_only_idx) > 0:
                fig.add_trace(go.Scatter(
                    x=model_only_idx,
                    y=df.loc[model_only_idx, column],
                    mode='markers',
                    name='Model-Based Anomaly',
                    marker=dict(color='red', size=6, symbol='circle')
                ))
            # Rule-based only: ORANGE
            rule_only_idx = anomalydf.index[rule_mask[anomalydf.index] & ~model_mask[anomalydf.index]]
            if len(rule_only_idx) > 0:
                fig.add_trace(go.Scatter(
                    x=rule_only_idx,
                    y=df.loc[rule_only_idx, column],
                    mode='markers',
                    name='Rule-Based Anomaly',
                    marker=dict(color='orange', size=6, symbol='circle')
                ))
            # Both: PURPLE
            both_idx = anomalydf.index[model_mask[anomalydf.index] & rule_mask[anomalydf.index]]
            if len(both_idx) > 0:
                fig.add_trace(go.Scatter(
                    x=both_idx,
                    y=df.loc[both_idx, column],
                    mode='markers',
                    name='Both (Model & Rule)',
                    marker=dict(color='purple', size=6, symbol='circle')
                ))
            fig.update_layout(showlegend=True)
            return fig

        col_layout = st.columns(2 if len(selected_model_cols) >= 2 else 1)
        for idx, col in enumerate(selected_model_cols):
            if col in df.columns:
                fig = plot_anomalies(
                    df, col, anomalydf,
                    model_mask=anomalies_isoforest,
                    rule_mask=anomalies_rulebased
                )
                with col_layout[idx % 2]:
                    st.plotly_chart(fig, use_container_width=True)

        if not anomalydf.empty:
            st.download_button(
                "Download Anomalies as CSV",
                data=anomalydf.to_csv(index=False),
                file_name="anomalies.csv",
                mime="text/csv"
            )
        else:
            st.info("No anomalies detected with current settings.")

else:
    st.warning("Upload your analysis dataset to configure columns and run detection.")

# --- Add custom UI/visual polish ---
st.markdown(
    """
    <style>
    .stExpander, .stForm {background: #fafafa !important; border-radius: 12px; }
    .stFileUploader, .stSlider, .stNumberInput {margin-bottom: 0.75rem;}
    .stMetric {margin-bottom: 1.5rem;}
    .stDataFrame {margin-top: 1.1rem;}
    .block-container {padding-top: 2rem;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("---")
st.caption(
    "✔ Upload & settings compact on top-right • Column selection on top-left • "
    "Anomalies: model-based (red), rule-based (orange), both (purple)"
)
