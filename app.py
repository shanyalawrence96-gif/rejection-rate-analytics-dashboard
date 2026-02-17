import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Rejection Rate Analytics", layout="wide")

st.title("📊 Rejection Rate Analytics Dashboard")
st.write("This dashboard helps small industries analyze rejection rates and identify major causes.")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/production_quality_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

# -----------------------------
# Feature Engineering
# -----------------------------
df["Rejection_Percent"] = (df["Rejected_Qty"] / df["Produced_Qty"]) * 100

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("🔍 Filters")

product_filter = st.sidebar.multiselect(
    "Select Product",
    options=df["Product"].unique(),
    default=df["Product"].unique()
)

machine_filter = st.sidebar.multiselect(
    "Select Machine",
    options=df["Machine_ID"].unique(),
    default=df["Machine_ID"].unique()
)

shift_filter = st.sidebar.multiselect(
    "Select Shift",
    options=df["Shift"].unique(),
    default=df["Shift"].unique()
)

filtered_df = df[
    (df["Product"].isin(product_filter)) &
    (df["Machine_ID"].isin(machine_filter)) &
    (df["Shift"].isin(shift_filter))
]

# -----------------------------
# KPI Metrics
# -----------------------------
st.subheader("📌 Key Metrics")

total_produced = filtered_df["Produced_Qty"].sum()
total_rejected = filtered_df["Rejected_Qty"].sum()

if total_produced == 0:
    overall_rejection_percent = 0
else:
    overall_rejection_percent = (total_rejected / total_produced) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Total Produced", f"{total_produced}")
col2.metric("Total Rejected", f"{total_rejected}")
col3.metric("Overall Rejection %", f"{overall_rejection_percent:.2f}%")

st.divider()

# -----------------------------
# Data Preview
# -----------------------------
st.subheader("📄 Data Preview")
st.dataframe(filtered_df, use_container_width=True)
st.divider()

# ==============================
# STEP 11: UI POLISHING WITH TABS
# ==============================
tabs = st.tabs([
    "🏠 Overview",
    "📊 Visualizations",
    "🧠 Pareto + Root Cause",
    "📉 SPC Chart",
    "🤖 ML Prediction",
    "📥 Reports"
])

# ==============================
# TAB 0: OVERVIEW
# ==============================
with tabs[0]:
    st.subheader("🏠 Overview")

    if len(filtered_df) == 0:
        st.warning("⚠️ No data available under current filters.")
    else:
        st.markdown("### ✅ Quick Summary")
        st.write("Use the tabs above to explore rejection rate trends, root causes, SPC control charts, and ML prediction.")

        st.markdown("### 🔥 Highest Rejection Areas")
        top_defect = filtered_df.groupby("Defect_Type")["Rejected_Qty"].sum().sort_values(ascending=False).head(1)
        top_machine = filtered_df.groupby("Machine_ID")["Rejected_Qty"].sum().sort_values(ascending=False).head(1)
        top_shift = filtered_df.groupby("Shift")["Rejected_Qty"].sum().sort_values(ascending=False).head(1)

        c1, c2, c3 = st.columns(3)

        c1.metric("Top Defect", top_defect.index[0], int(top_defect.values[0]))
        c2.metric("Top Machine", top_machine.index[0], int(top_machine.values[0]))
        c3.metric("Top Shift", top_shift.index[0], int(top_shift.values[0]))

# ==============================
# TAB 1: STEP 6 VISUALIZATIONS
# ==============================
with tabs[1]:
    st.subheader("📊 Step 6: Visualizations")

    if len(filtered_df) == 0:
        st.warning("⚠️ No data available under current filters.")
    else:
        # 1) Rejection % Trend Over Time
        st.markdown("### 📈 Rejection % Trend Over Time")

        trend_df = filtered_df.groupby("Date")[["Produced_Qty", "Rejected_Qty"]].sum().reset_index()
        trend_df["Rejection_Percent"] = (trend_df["Rejected_Qty"] / trend_df["Produced_Qty"]) * 100

        st.line_chart(trend_df.set_index("Date")["Rejection_Percent"])

        # 2) Rejection % by Machine
        st.markdown("### 🏭 Rejection % by Machine")

        machine_df = filtered_df.groupby("Machine_ID")[["Produced_Qty", "Rejected_Qty"]].sum().reset_index()
        machine_df["Rejection_Percent"] = (machine_df["Rejected_Qty"] / machine_df["Produced_Qty"]) * 100

        st.bar_chart(machine_df.set_index("Machine_ID")["Rejection_Percent"])

        # 3) Rejection % by Shift
        st.markdown("### ⏱️ Rejection % by Shift")

        shift_df = filtered_df.groupby("Shift")[["Produced_Qty", "Rejected_Qty"]].sum().reset_index()
        shift_df["Rejection_Percent"] = (shift_df["Rejected_Qty"] / shift_df["Produced_Qty"]) * 100

        st.bar_chart(shift_df.set_index("Shift")["Rejection_Percent"])

        # 4) Top Defect Types
        st.markdown("### 🚨 Top Defect Types")

        defect_df = filtered_df.groupby("Defect_Type")["Rejected_Qty"].sum().reset_index()
        defect_df = defect_df.sort_values(by="Rejected_Qty", ascending=False)

        st.bar_chart(defect_df.set_index("Defect_Type")["Rejected_Qty"])

# ==============================
# TAB 2: STEP 7 PARETO + ROOT CAUSE
# ==============================
with tabs[2]:
    st.subheader("🧠 Step 7: Pareto + Root Cause Analysis")

    if len(filtered_df) == 0:
        st.warning("⚠️ No data available under current filters.")
    else:
        st.markdown("### 📌 Pareto Chart: Defect Types (80/20 Rule)")

        pareto_df = filtered_df.groupby("Defect_Type")["Rejected_Qty"].sum().reset_index()
        pareto_df = pareto_df.sort_values(by="Rejected_Qty", ascending=False)

        pareto_df["Cumulative_Rejections"] = pareto_df["Rejected_Qty"].cumsum()
        pareto_df["Cumulative_Percent"] = (pareto_df["Cumulative_Rejections"] / pareto_df["Rejected_Qty"].sum()) * 100

        st.dataframe(pareto_df, use_container_width=True)

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.bar(pareto_df["Defect_Type"], pareto_df["Rejected_Qty"])
        ax1.set_xlabel("Defect Type")
        ax1.set_ylabel("Rejected Qty")
        plt.xticks(rotation=45)

        ax2 = ax1.twinx()
        ax2.plot(pareto_df["Defect_Type"], pareto_df["Cumulative_Percent"], marker="o")
        ax2.set_ylabel("Cumulative %")
        ax2.axhline(80, linestyle="--")

        st.pyplot(fig)

        st.markdown("### 🔍 Root Cause Summary")

        colA, colB = st.columns(2)

        with colA:
            st.markdown("#### 🏭 Top Machines")
            machine_root = filtered_df.groupby("Machine_ID")["Rejected_Qty"].sum().reset_index()
            machine_root = machine_root.sort_values(by="Rejected_Qty", ascending=False)
            st.dataframe(machine_root.head(5), use_container_width=True)

        with colB:
            st.markdown("#### ⏱️ Top Shifts")
            shift_root = filtered_df.groupby("Shift")["Rejected_Qty"].sum().reset_index()
            shift_root = shift_root.sort_values(by="Rejected_Qty", ascending=False)
            st.dataframe(shift_root.head(5), use_container_width=True)

        st.markdown("#### 📦 Top Products")
        product_root = filtered_df.groupby("Product")["Rejected_Qty"].sum().reset_index()
        product_root = product_root.sort_values(by="Rejected_Qty", ascending=False)
        st.dataframe(product_root.head(5), use_container_width=True)

        st.markdown("#### 🚨 Worst Combination")
        combo_df = filtered_df.groupby(["Machine_ID", "Shift", "Product"])["Rejected_Qty"].sum().reset_index()
        combo_df = combo_df.sort_values(by="Rejected_Qty", ascending=False)
        st.dataframe(combo_df.head(10), use_container_width=True)

# ==============================
# TAB 3: STEP 8 SPC CHART
# ==============================
with tabs[3]:
    st.subheader("📉 Step 8: SPC Control Chart (p-chart) + Alerts")

    if len(filtered_df) == 0:
        st.warning("⚠️ No data available under current filters.")
    else:
        spc_df = filtered_df.groupby("Date")[["Produced_Qty", "Rejected_Qty"]].sum().reset_index()
        spc_df = spc_df[spc_df["Produced_Qty"] > 0]

        if len(spc_df) < 2:
            st.warning("⚠️ Not enough data to calculate SPC chart.")
        else:
            spc_df["p"] = spc_df["Rejected_Qty"] / spc_df["Produced_Qty"]
            p_bar = spc_df["p"].mean()

            spc_df["sigma_p"] = np.sqrt((p_bar * (1 - p_bar)) / spc_df["Produced_Qty"])

            spc_df["UCL"] = p_bar + (3 * spc_df["sigma_p"])
            spc_df["LCL"] = p_bar - (3 * spc_df["sigma_p"])
            spc_df["LCL"] = spc_df["LCL"].clip(lower=0)

            spc_df["p_percent"] = spc_df["p"] * 100
            spc_df["UCL_percent"] = spc_df["UCL"] * 100
            spc_df["LCL_percent"] = spc_df["LCL"] * 100
            CL_percent = p_bar * 100

            spc_df["Out_of_Control"] = spc_df["p"] > spc_df["UCL"]

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(spc_df["Date"], spc_df["p_percent"], marker="o", label="Daily Rejection %")
            ax.plot(spc_df["Date"], spc_df["UCL_percent"], linestyle="--", label="UCL")
            ax.plot(spc_df["Date"], spc_df["LCL_percent"], linestyle="--", label="LCL")
            ax.axhline(CL_percent, linestyle="-.", label="Center Line (CL)")

            ax.set_title("SPC p-Chart: Rejection % with Control Limits")
            ax.set_xlabel("Date")
            ax.set_ylabel("Rejection %")
            plt.xticks(rotation=45)

            st.pyplot(fig)

            out_of_control_days = spc_df[spc_df["Out_of_Control"] == True]

            if len(out_of_control_days) > 0:
                st.error("🚨 ALERT: Process may be OUT OF CONTROL! Rejection % crossed UCL on these dates:")
                st.dataframe(
                    out_of_control_days[["Date", "Produced_Qty", "Rejected_Qty", "p_percent", "UCL_percent"]],
                    use_container_width=True
                )
            else:
                st.success("✅ No out-of-control points detected. Process is stable.")

# ==============================
# TAB 4: STEP 9 ML PREDICTION
# ==============================
with tabs[4]:
    st.subheader("🤖 Step 9: ML Prediction Model (Rejection %)")

    st.write("This model predicts rejection percentage based on production parameters.")

    ml_df = df.copy()
    ml_df["Rejection_Percent"] = (ml_df["Rejected_Qty"] / ml_df["Produced_Qty"]) * 100
    ml_df = ml_df.replace([np.inf, -np.inf], np.nan).dropna()

    if len(ml_df) < 10:
        st.warning("⚠️ Not enough data to train ML model.")
    else:
        X = ml_df[["Product", "Machine_ID", "Shift", "Defect_Type", "Produced_Qty"]]
        y = ml_df["Rejection_Percent"]

        categorical_cols = ["Product", "Machine_ID", "Shift", "Defect_Type"]
        numeric_cols = ["Produced_Qty"]

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
                ("num", "passthrough", numeric_cols)
            ]
        )

        model = RandomForestRegressor(n_estimators=200, random_state=42)

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        col1, col2 = st.columns(2)
        col1.metric("📌 Model MAE", f"{mae:.2f}")
        col2.metric("📌 Model R² Score", f"{r2:.2f}")

        st.divider()

        st.markdown("### 🧾 Predict Rejection % (Custom Input)")

        with st.form("prediction_form"):
            p_product = st.selectbox("Select Product", df["Product"].unique())
            p_machine = st.selectbox("Select Machine", df["Machine_ID"].unique())
            p_shift = st.selectbox("Select Shift", df["Shift"].unique())
            p_defect = st.selectbox("Select Defect Type", df["Defect_Type"].unique())
            p_produced = st.number_input("Produced Quantity", min_value=1, value=500)

            submitted = st.form_submit_button("Predict")

        if submitted:
            input_data = pd.DataFrame([{
                "Product": p_product,
                "Machine_ID": p_machine,
                "Shift": p_shift,
                "Defect_Type": p_defect,
                "Produced_Qty": p_produced
            }])

            prediction = pipeline.predict(input_data)[0]
            st.success(f"✅ Predicted Rejection %: **{prediction:.2f}%**")

# ==============================
# TAB 5: STEP 10 REPORTS
# ==============================
with tabs[5]:
    st.subheader("📥 Step 10: Download Report + Auto Insights")

    if len(filtered_df) == 0:
        st.warning("⚠️ No data available under current filters.")
    else:
        st.markdown("### ✅ Download Filtered Data (CSV)")

        csv = filtered_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Filtered Data as CSV",
            data=csv,
            file_name="filtered_rejection_report.csv",
            mime="text/csv"
        )

        st.divider()

        st.markdown("### 📊 Download Full Summary Report (Excel)")

        kpi_table = pd.DataFrame({
            "Metric": ["Total Produced", "Total Rejected", "Overall Rejection %"],
            "Value": [total_produced, total_rejected, round(overall_rejection_percent, 2)]
        })

        top_defects = filtered_df.groupby("Defect_Type")["Rejected_Qty"].sum().reset_index()
        top_defects = top_defects.sort_values(by="Rejected_Qty", ascending=False)

        machine_root = filtered_df.groupby("Machine_ID")["Rejected_Qty"].sum().reset_index()
        machine_root = machine_root.sort_values(by="Rejected_Qty", ascending=False)

        shift_root = filtered_df.groupby("Shift")["Rejected_Qty"].sum().reset_index()
        shift_root = shift_root.sort_values(by="Rejected_Qty", ascending=False)

        product_root = filtered_df.groupby("Product")["Rejected_Qty"].sum().reset_index()
        product_root = product_root.sort_values(by="Rejected_Qty", ascending=False)

        combo_df = filtered_df.groupby(["Machine_ID", "Shift", "Product"])["Rejected_Qty"].sum().reset_index()
        combo_df = combo_df.sort_values(by="Rejected_Qty", ascending=False)

        output = io.BytesIO()

        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            kpi_table.to_excel(writer, sheet_name="KPI Summary", index=False)
            top_defects.to_excel(writer, sheet_name="Top Defects", index=False)
            machine_root.to_excel(writer, sheet_name="Top Machines", index=False)
            shift_root.to_excel(writer, sheet_name="Top Shifts", index=False)
            product_root.to_excel(writer, sheet_name="Top Products", index=False)
            combo_df.head(20).to_excel(writer, sheet_name="Worst Combos", index=False)

        output.seek(0)

        st.download_button(
            label="⬇️ Download Full Summary Report (Excel)",
            data=output,
            file_name="rejection_summary_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.divider()

        st.markdown("### 🧠 Auto Insights (Smart Suggestions)")

        top_defect = top_defects.iloc[0]["Defect_Type"]
        top_defect_qty = int(top_defects.iloc[0]["Rejected_Qty"])

        top_machine = machine_root.iloc[0]["Machine_ID"]
        top_machine_qty = int(machine_root.iloc[0]["Rejected_Qty"])

        top_shift = shift_root.iloc[0]["Shift"]
        top_shift_qty = int(shift_root.iloc[0]["Rejected_Qty"])

        worst_combo = combo_df.iloc[0]
        worst_machine = worst_combo["Machine_ID"]
        worst_shift = worst_combo["Shift"]
        worst_product = worst_combo["Product"]
        worst_qty = int(worst_combo["Rejected_Qty"])

        st.success("✅ Insights Generated Successfully!")

        st.markdown(f"""
        #### 🔥 Key Findings:
        - **Top Defect Type:** `{top_defect}` → **{top_defect_qty} rejections**
        - **Worst Machine:** `{top_machine}` → **{top_machine_qty} rejections**
        - **Worst Shift:** `{top_shift}` → **{top_shift_qty} rejections**
        - **Worst Combination:**  
          Machine `{worst_machine}` + Shift `{worst_shift}` + Product `{worst_product}`  
          → **{worst_qty} rejections**
        """)

        st.markdown("#### ✅ Recommended Actions:")
        st.write("1) Inspect the top defect type and apply preventive QC checks.")
        st.write("2) Perform maintenance/calibration on the worst machine.")
        st.write("3) Review shift operator handling and training.")
        st.write("4) Monitor the worst combination daily using SPC charts.")