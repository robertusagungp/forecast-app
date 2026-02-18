import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="Forecasting App", layout="wide")
st.title("Forecasting Web App")
st.write("Upload data → choose frequency → train → forecast per segment (non-negative)")

# ================================
# FREQUENCY MAP
# ================================
FREQ_MAP = {
    "Daily": "D",
    "Weekly": "W",
    "Monthly": "MS",  # Month Start (lebih stabil daripada "M")
    "Yearly": "YS"    # Year Start
}

# ================================
# LOAD DATA
# ================================
@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    df.columns = df.columns.str.strip()
    return df

# ================================
# PLOT
# ================================
def plot_forecast(actual_df, forecast_df, segment):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=actual_df["ds"],
        y=actual_df["y"],
        name="Actual",
        mode="lines+markers"
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df["ds"],
        y=forecast_df["yhat"],
        name="Forecast",
        mode="lines"
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df["ds"],
        y=forecast_df["yhat_upper"],
        name="Upper Bound",
        line=dict(dash="dash")
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df["ds"],
        y=forecast_df["yhat_lower"],
        name="Lower Bound",
        line=dict(dash="dash")
    ))

    fig.update_layout(
        title=f"Forecast - {segment}",
        height=520,
        xaxis_title="Date",
        yaxis_title="Value"
    )
    return fig

# ================================
# TRAIN MODELS (CACHED)
# ================================
@st.cache_resource
def train_models(
    df,
    date_col,
    segment_col,
    value_col,
    forecast_period,
    freq,
    cap_multiplier,
    changepoint_prior_scale,
    seasonality_prior_scale,
    clamp_non_negative=True
):
    results = {}
    forecasts = {}

    df = df.copy()
    df.columns = df.columns.str.strip()

    # Clean numeric column globally
    df[value_col] = (
        df[value_col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    # Clean date column globally
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Drop invalid rows
    df = df.dropna(subset=[date_col, segment_col, value_col])

    # Basic sanity: remove negatives in training (kalau input user ternyata ada minus)
    df = df[df[value_col] >= 0]

    segments = df[segment_col].dropna().unique()

    for seg in segments:
        seg_df = df[df[segment_col] == seg].copy()

        seg_df = seg_df.rename(columns={date_col: "ds", value_col: "y"})
        seg_df = seg_df.sort_values("ds")

        # Minimal points check
        if len(seg_df) < 6:
            # skip segment yang terlalu sedikit datanya
            continue

        # ---- NON-NEGATIVE FORECAST SETUP ----
        # logistic growth requires floor & cap
        seg_df["floor"] = 0

        y_max = float(seg_df["y"].max())
        # cap harus > y, kalau y_max kecil bisa jadi terlalu ketat
        cap = max(1.0, y_max * cap_multiplier)
        seg_df["cap"] = cap

        # ---- Prophet model ----
        model = Prophet(
            growth="logistic",
            daily_seasonality=(freq == "D"),
            weekly_seasonality=(freq in ["D", "W"]),
            yearly_seasonality=True,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale
        )

        model.fit(seg_df[["ds", "y", "cap", "floor"]])

        future = model.make_future_dataframe(periods=forecast_period, freq=freq)
        future["floor"] = 0
        future["cap"] = cap

        forecast = model.predict(future)

        # ---- Clamp (hard guarantee) ----
        if clamp_non_negative:
            for c in ["yhat", "yhat_lower", "yhat_upper"]:
                forecast[c] = forecast[c].clip(lower=0)

        results[seg] = seg_df
        forecasts[seg] = forecast

    return results, forecasts

# ================================
# UI
# ================================
uploaded_file = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx"])

if uploaded_file:
    df = load_data(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        date_col = st.selectbox("Date Column", df.columns)
    with c2:
        segment_col = st.selectbox("Segment Column", df.columns)
    with c3:
        value_col = st.selectbox("Value Column", df.columns)

    c4, c5, c6 = st.columns(3)

    with c4:
        freq_label = st.selectbox("Data Frequency", ["Daily", "Weekly", "Monthly", "Yearly"])
        freq = FREQ_MAP[freq_label]

    with c5:
        forecast_period = st.number_input(
            f"Forecast Length ({freq_label})",
            min_value=1,
            max_value=2000,
            value=12
        )

    with c6:
        cap_multiplier = st.number_input(
            "Cap Multiplier (upper ceiling)",
            min_value=1.05,
            max_value=10.0,
            value=1.5,
            step=0.05
        )

    with st.expander("Advanced Model Tuning (optional)", expanded=False):
        changepoint_prior_scale = st.slider(
            "Changepoint Prior Scale (trend flexibility)",
            min_value=0.001,
            max_value=0.5,
            value=0.05,
            step=0.001
        )
        seasonality_prior_scale = st.slider(
            "Seasonality Prior Scale",
            min_value=0.1,
            max_value=30.0,
            value=10.0,
            step=0.1
        )
        clamp_non_negative = st.checkbox("Hard clamp forecast ≥ 0", value=True)

    st.caption(
        f"Forecast will generate **{forecast_period}** future **{freq_label.lower()}** periods."
    )

    # Basic warning (data length)
    tmp = df.copy()
    tmp.columns = tmp.columns.str.strip()
    if date_col in tmp.columns:
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        if tmp[date_col].notna().any():
            date_min = tmp[date_col].min()
            date_max = tmp[date_col].max()
            st.info(f"Training data range: **{date_min.date()} → {date_max.date()}**")

    if st.button("Train Model"):
        with st.spinner("Training model... (cached)"):
            results, forecasts = train_models(
                df=df,
                date_col=date_col,
                segment_col=segment_col,
                value_col=value_col,
                forecast_period=forecast_period,
                freq=freq,
                cap_multiplier=cap_multiplier,
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
                clamp_non_negative=clamp_non_negative
            )

        st.session_state["results"] = results
        st.session_state["forecasts"] = forecasts

        if not results:
            st.warning(
                "Tidak ada segment yang bisa ditrain (biasanya karena data per segment terlalu sedikit atau banyak nilai invalid). "
                "Coba pastikan tiap segment punya minimal ~6 titik data."
            )
        else:
            st.success("Training completed (results cached). Silakan pilih segment.")

    # Show results (no retrain)
    if "results" in st.session_state and st.session_state["results"]:
        results = st.session_state["results"]
        forecasts = st.session_state["forecasts"]

        segments = sorted(list(results.keys()))
        selected_segment = st.selectbox("Select Segment", segments)

        actual_df = results[selected_segment][["ds", "y"]].copy()
        forecast_df = forecasts[selected_segment].copy()

        fig = plot_forecast(actual_df, forecast_df, selected_segment)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Forecast Table")
        st.dataframe(
            forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]],
            use_container_width=True
        )
else:
    st.info("Upload file CSV/Excel untuk mulai.")
