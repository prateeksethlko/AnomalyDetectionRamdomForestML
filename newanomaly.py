import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os
import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Directory to store uploaded CSV files
UPLOAD_FOLDER = "uploaded_csvs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def load_historical_data(upload_folder):
    """Loads a single CSV file."""
    try:
        files = os.listdir(upload_folder)
        if files:
            latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(upload_folder, x)))
            filepath = os.path.join(upload_folder, latest_file)
            return pd.read_csv(filepath)
        else:
            return None
    except FileNotFoundError:
        st.warning("No historical data file found.")
        return None
    except pd.errors.EmptyDataError:
        st.warning("The historical data file is empty.")
        return None
    except Exception as e:
        st.warning(f"Error loading historical data: {e}")
        return None

def detect_anomalies_isolation_forest(df, historical_df, rule_based_columns, contamination, anomaly_threshold, multiplier, use_rule_based):
    """Detects anomalies using Isolation Forest and a dynamic rule-based condition."""
    if historical_df is None or historical_df.empty:
        st.warning("No historical data available for anomaly detection.")
        return pd.DataFrame(), None

    try:
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        historical_numerical_cols = historical_df.select_dtypes(include=np.number).columns.tolist()

        common_numerical_cols = list(set(numerical_cols).intersection(set(historical_numerical_cols)))

        if not common_numerical_cols:
            st.warning("No common numerical columns to perform anomaly detection.")
            return pd.DataFrame(), None

        # Handle missing values by filling with the mean of each column
        df[common_numerical_cols] = df[common_numerical_cols].fillna(df[common_numerical_cols].mean())
        historical_df[common_numerical_cols] = historical_df[common_numerical_cols].fillna(historical_df[common_numerical_cols].mean())

        scaler = StandardScaler()
        scaler.fit(historical_df[common_numerical_cols])
        df_scaled = scaler.transform(df[common_numerical_cols])
        historical_df_scaled = scaler.transform(historical_df[common_numerical_cols])

        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(historical_df_scaled)
        anomaly_scores = model.decision_function(df_scaled)
        anomalies_isolation_forest = anomaly_scores < anomaly_threshold

        anomalies = anomalies_isolation_forest # default Anamolies is just Isolation forest

        if use_rule_based:
            # Dynamic rule-based anomaly detection
            anomalies_rule_based = np.zeros(len(df), dtype=bool)  # Initialize with all False

            for col in rule_based_columns:
                if col in historical_df.columns and col in df.columns:
                    average_value = historical_df[col].mean()
                    anomalies_rule_based = anomalies_rule_based | (df[col] > multiplier * average_value) #Applying the rule for all selected columns

            # Combine anomalies from both methods
            anomalies = anomalies_isolation_forest | anomalies_rule_based

        anomaly_df = df.iloc[np.where(anomalies)[0]].copy() # Preserve original values
        return anomaly_df, anomaly_scores

    except Exception as e:
        st.error(f"Error during anomaly detection: {e}")
        return pd.DataFrame(), None

def plot_anomalies(df, column, anomaly_df):
    """Plots a column with highlighted anomalies and the average line."""
    fig = px.scatter(df, x=df.index, y=column, title=f"Anomaly Detection for {column}")

    # Add average line
    avg_value = df[column].mean()
    fig.add_trace(go.Scatter(x=df.index, y=[avg_value] * len(df), mode='lines', name='Average', line=dict(color='green', width=2)))

    if not anomaly_df.empty:
        fig.add_trace(
            px.scatter(anomaly_df, x=anomaly_df.index, y=column, color_discrete_sequence=['red']).data[0]
        )

    fig.update_layout(showlegend=False)
    return fig # Return figure instead of displaying it

def main():
    st.set_page_config(layout="wide")
    st.title("CSV Anomaly Detection")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            historical_df = load_historical_data(UPLOAD_FOLDER)

            col1, col2 = st.columns([2, 2])

            with col1:
                st.write("Historic Data:")
                st.dataframe(historical_df)

            with col2:
                st.write("Uploaded Data:")
                st.dataframe(df)

            # Column selection
            numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
            selected_columns = st.multiselect("Select columns for anomaly detection (Isolation Forest)", numerical_cols, default=numerical_cols)
            rule_based_columns = st.multiselect("Select columns for rule-based anomaly detection", numerical_cols)

            # Parameter sliders
            contamination = st.slider("Contamination", min_value=0.0, max_value=0.5, value=0.2, step=0.01)
            anomaly_threshold = st.slider("Anomaly Threshold", min_value=-1.0, max_value=0.0, value=-0.14, step=0.01)
            multiplier = st.slider("Rule Multiplier", min_value=1.0, max_value=20.0, value=10.0, step=0.5)

            use_rule_based = st.checkbox("Use Rule-Based Anomaly Detection", value=True)

            combined_anomaly_df = pd.DataFrame()
            for _ in range(1):
                anomaly_df, anomaly_scores = detect_anomalies_isolation_forest(df.copy(), historical_df.copy(), rule_based_columns, contamination, anomaly_threshold, multiplier, use_rule_based)
                combined_anomaly_df = pd.concat([combined_anomaly_df, anomaly_df], ignore_index=True)

            if not anomaly_df.empty:
                st.write("Anomalies Detected:")
                st.dataframe(anomaly_df)
                st.write(f"Number of Anomalies Detected: {len(anomaly_df)}") #Added so that its clear the number of anomalies detected

                # Plotting in 2 columns
                num_cols = len(selected_columns)
                cols = st.columns(2 if num_cols >= 2 else num_cols)  # Create 2 columns if enough columns, otherwise create 1
                col_index = 0

                for i, col in enumerate(selected_columns):
                    if col in df.columns:
                        fig = plot_anomalies(df, col, anomaly_df)
                        with cols[col_index % len(cols)]:
                            st.plotly_chart(fig, use_container_width=True)  # Display the chart
                        col_index += 1

            else:
                st.info("No anomalies detected.")

        except Exception as e:
            st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
