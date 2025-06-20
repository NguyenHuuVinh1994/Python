import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

# Set page configuration
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

# Title and description
st.title("📈 Sales Forecasting Dashboard")
st.markdown("Tải lên file Excel (xlsx, xls) bất kỳ nhưng phải có cột Date (YYYY--MM-DD), Store ID và Quantity")

# File upload
uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx", "xls"])
if uploaded_file is not None:
    try:
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_names = excel_file.sheet_names
        if not sheet_names:
            st.error("Không thấy sheets có thể sử dụng")
            st.stop()
        selected_sheets = st.selectbox("Chọn sheet", sheet_names, index=0)

        # Load data
        sales = pd.read_excel(uploaded_file, sheet_name=selected_sheets)
        st.success(f"Data loaded successfully from sheet {selected_sheets} !")
        st.write(f"**Number of rows in original data**: {sales.shape[0]}")
        st.write(f"**Unique Store IDs**: {sales['Store ID'].nunique()}")
        
        #Check columns
        required_columns = ['Date', 'Store ID', 'Quantity']
        if not all(col in sales.columns for col in required_columns):
            st.error(f"Không có dữ liệu: {', '.join(required_columns)}")
            st.stop()

        # Data preprocessing
        daily = sales.groupby(['Date', 'Store ID'])['Quantity'].sum().reset_index()
        daily['Date'] = pd.to_datetime(daily['Date'])
        daily['Original_Store_ID'] = daily['Store ID']
        start_date = '2023-01-01'
        daily = daily[daily['Date'] >= start_date].copy()
        
        st.write(f"**Data filtered from {start_date}**: {daily.shape[0]} rows")
        st.write(f"**Unique Store IDs after filtering**: {daily['Store ID'].nunique()}")

        if daily.empty:
            st.error("Error: Filtered DataFrame is empty. Check the start date or data.")
            st.stop()

        # Feature engineering
        daily = daily.sort_values(by=['Original_Store_ID', 'Date'])
        daily['dayofweek'] = daily['Date'].dt.dayofweek
        daily['weekofyear'] = daily['Date'].dt.isocalendar().week
        daily['month'] = daily['Date'].dt.month
        daily['year'] = daily['Date'].dt.year
        daily['dayofyear'] = daily['Date'].dt.dayofyear
        daily['is_weekend'] = (daily['dayofweek'] >= 5).astype(int)
        daily['qty_lag_1'] = daily.groupby('Original_Store_ID')['Quantity'].shift(1)
        daily['qty_lag_7'] = daily.groupby('Original_Store_ID')['Quantity'].shift(7)
        daily['qty_lag_365'] = daily.groupby('Original_Store_ID')['Quantity'].shift(365)
        daily['rolling_mean_7'] = daily.groupby('Original_Store_ID')['Quantity'].rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)
        daily['rolling_std_7'] = daily.groupby('Original_Store_ID')['Quantity'].rolling(window=7, min_periods=1).std().reset_index(level=0, drop=True)
        daily['qty_lag_7'] = daily.groupby('Original_Store_ID')['qty_lag_7'].transform(lambda x: x.fillna(x.mean()))
        daily['qty_lag_365'] = daily.groupby('Original_Store_ID')['qty_lag_365'].transform(lambda x: x.fillna(x.mean()))

        # Exponential Smoothing Model
        st.subheader("📊 Exponential Smoothing Forecast (Total Sales)")
        total_daily_sales_series = daily.groupby('Date')['Quantity'].sum()
        last_historical_date = daily['Date'].max()
        es_future_dates = pd.date_range(start=last_historical_date + timedelta(days=1), periods=7, freq='D')
        
        with st.spinner("Fitting Exponential Smoothing model..."):
            es_model = ExponentialSmoothing(
                total_daily_sales_series,
                seasonal_periods=7,
                trend='mul',
                seasonal='mul',
                damped_trend=True
            )
            es_fit = es_model.fit()
            es_forecast = es_fit.fittedvalues
            es_forecast_future = es_fit.forecast(len(es_future_dates))
            es_forecast_df = pd.DataFrame({'Date': es_future_dates, 'Unit_Sold': es_forecast_future}).reset_index(drop=True)
            es_forecast_df['Unit_Sold'] = es_forecast_df['Unit_Sold'].clip(lower=0)
            total_es_forecast = es_forecast_df['Unit_Sold'].sum()

        st.write("**7-Day Forecast (Total Sales)**")
        st.dataframe(es_forecast_df.style.format({"Unit_Sold": "{:.2f}"}))
        st.write(f"**Total Forecast for Next 7 Days (ES)**: {total_es_forecast:.2f}")

        # Plot ES forecast
        fig_es = go.Figure()
        fig_es.add_trace(go.Scatter(
            x=total_daily_sales_series.index[-30:],
            y=total_daily_sales_series.values[-30:],
            mode='lines',
            name='Historical Sales',
            line=dict(color='blue')
        ))
        fig_es.add_trace(go.Scatter(
            x=es_forecast_df['Date'],
            y=es_forecast_df['Unit_Sold'],
            mode='lines',
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        fig_es.update_layout(
            title="Exponential Smoothing: Historical Sales and 7-Day Forecast",
            xaxis_title="Date",
            yaxis_title="Total Quantity Sold",
            template="plotly_white"
        )
        st.plotly_chart(fig_es, use_container_width=True)

        # LightGBM Model
        st.subheader("📊 LightGBM Forecast (Per Store)")
        daily_onehot = pd.get_dummies(daily, columns=['Store ID'], prefix='Store ID')
        target = 'Quantity'
        X = daily_onehot.drop(columns=['Date', target, 'Original_Store_ID'])
        y = daily_onehot[target]
        test_start_date = last_historical_date - timedelta(days=6)
        train_mask = (daily_onehot['Date'] < test_start_date)
        test_mask = (daily_onehot['Date'] >= test_start_date)
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        results_test_info = daily_onehot[test_mask][['Date', 'Original_Store_ID']].copy()

        if X_train.empty or X_test.empty:
            st.error("Error: Train or Test set is empty. Check date ranges.")
            st.stop()

        with st.spinner("Training LightGBM model..."):
            lgbm = lgb.LGBMRegressor(random_state=42, n_estimators=100)
            lgbm.fit(X_train, y_train)
            y_pred_lgbm = lgbm.predict(X_test)
            y_pred_lgbm[y_pred_lgbm < 0] = 0
            mae = mean_absolute_error(y_test, y_pred_lgbm)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_lgbm))

        st.write("**LightGBM Model Performance on Test Set**")
        st.write(f"**MAE**: {mae:.2f}")
        st.write(f"**RMSE**: {rmse:.2f}")

        # Plot Actual vs Predicted for a sample store
        sample_store_id = results_test_info['Original_Store_ID'].unique()[0] if results_test_info['Original_Store_ID'].nunique() > 0 else None
        if sample_store_id:
            results_test = results_test_info.copy()
            results_test['Actual'] = y_test.values
            results_test['Predicted'] = y_pred_lgbm
            results_store = results_test[results_test['Original_Store_ID'] == sample_store_id]
            fig_lgbm_test = px.line(
                results_store,
                x='Date',
                y=['Actual', 'Predicted'],
                title=f'LightGBM: Actual vs Predicted for Store {sample_store_id}',
                labels={'value': 'Quantity', 'variable': 'Type'}
            )
            st.plotly_chart(fig_lgbm_test, use_container_width=True)

        # Iterative Forecasting for LightGBM
        store_ids = daily_onehot['Original_Store_ID'].unique()
        future_data_list = [{'Date': date, 'Original_Store_ID': store_id} for store_id in store_ids for date in es_future_dates]
        df_future = pd.DataFrame(future_data_list)
        feature_cols = X_train.columns.tolist()
        ohe_cols = [col for col in feature_cols if col.startswith('Store ID_')]
        non_ohe_features = [col for col in feature_cols if not col.startswith('Store ID_')]
        df_future_processed = pd.get_dummies(df_future, columns=['Original_Store_ID'], prefix='Store ID')
        for col in ohe_cols:
            if col not in df_future_processed.columns:
                df_future_processed[col] = 0
        for col in non_ohe_features:
            if col not in df_future_processed.columns:
                df_future_processed[col] = 0
        df_future_processed['Quantity'] = np.nan
        df_future_processed['Original_Store_ID'] = df_future['Original_Store_ID'].values
        combined_cols = ['Date'] + feature_cols + ['Quantity', 'Original_Store_ID']
        daily_processed = daily_onehot[combined_cols].copy()
        df_future_processed = df_future_processed[combined_cols].copy()
        combined_data = pd.concat([daily_processed, df_future_processed], ignore_index=True)
        combined_data = combined_data.sort_values(by=['Original_Store_ID', 'Date']).reset_index(drop=True)

        final_forecast_results = []
        with st.spinner("Generating 7-day LightGBM forecast..."):
            for i in range(7):
                forecast_date = es_future_dates[i]
                combined_data['qty_lag_1'] = combined_data.groupby('Original_Store_ID')['Quantity'].shift(1).fillna(0)
                combined_data['qty_lag_7'] = combined_data.groupby('Original_Store_ID')['Quantity'].shift(7).fillna(0)
                combined_data['qty_lag_365'] = combined_data.groupby('Original_Store_ID')['Quantity'].shift(365).fillna(0)
                combined_data['rolling_mean_7'] = combined_data.groupby('Original_Store_ID')['Quantity'].rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)
                combined_data['rolling_std_7'] = combined_data.groupby('Original_Store_ID')['Quantity'].rolling(window=7, min_periods=1).std().reset_index(level=0, drop=True)
                combined_data[['rolling_mean_7', 'rolling_std_7']] = combined_data[['rolling_mean_7', 'rolling_std_7']].fillna(0)
                data_to_predict = combined_data[combined_data['Date'] == forecast_date].copy()
                X_predict = data_to_predict[feature_cols]
                predicted_quantity = lgbm.predict(X_predict)
                predicted_quantity[predicted_quantity < 0] = 0
                indices_to_update = data_to_predict.index
                combined_data.loc[indices_to_update, 'Quantity'] = predicted_quantity
                day_results_df = data_to_predict[['Date', 'Original_Store_ID']].copy()
                day_results_df['Predicted_Quantity'] = predicted_quantity
                final_forecast_results.append(day_results_df)

        final_forecast_df = pd.concat(final_forecast_results, ignore_index=True)
        weekly_forecast = final_forecast_df.groupby('Original_Store_ID')['Predicted_Quantity'].sum().reset_index()
        total_weekly_forecast = final_forecast_df['Predicted_Quantity'].sum()

        st.write("**7-Day Forecast by Store (LightGBM)**")
        st.dataframe(weekly_forecast.style.format({"Predicted_Quantity": "{:.2f}"}))
        st.write(f"**Total Forecast for Next 7 Days (LightGBM)**: {total_weekly_forecast:.2f}")

        # Interactive store selection for visualization
        selected_stores = st.multiselect(
            "Select Store ID for Forecast Visualization",
            store_ids,
            default=[store_ids[0]],
            key="store_selector")
        if selected_stores: #Chỉ hiển thị store được chọn
            display_store = selected_stores[0] #Kiểm tra xem có store nào được chọn không, nếu không lấy store đầu tiên trong danh sách
            st.write(f"**Showing 7-Day Forecast for Store {display_store} (LightGBM)**")
            results_store = final_forecast_df[final_forecast_df['Original_Store_ID'] == display_store]
            fig_lgbm_forecast = px.line(
                results_store,
                x='Date',
                y='Predicted_Quantity',
                title=f'7-Day Forecast for Store {display_store} (LightGBM)',
                labels={'Predicted_Quantity': 'Quantity'})
            st.plotly_chart(fig_lgbm_forecast, use_container_width=True)
        else:
            st.warning("Chọn một store để hiển thị")

        # Comparison
        st.subheader("🔍 Comparison of Forecasts")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Exponential Smoothing (Total Sales)**")
            st.write(f"Total 7-Day Forecast: {total_es_forecast:.2f}")
            st.dataframe(es_forecast_df.style.format({"Unit_Sold": "{:.2f}"}))
        with col2:
            st.write("**LightGBM (Sum of Stores)**")
            st.write(f"Total 7-Day Forecast: {total_weekly_forecast:.2f}")
            st.dataframe(weekly_forecast.style.format({"Predicted_Quantity": "{:.2f}"}))

        if total_es_forecast and total_weekly_forecast:
            percentage_diff = ((total_weekly_forecast - total_es_forecast) / total_es_forecast) * 100 if total_es_forecast != 0 else float('inf')
            st.write(f"**Difference (LightGBM - ES)**: {total_weekly_forecast - total_es_forecast:.2f} ({percentage_diff:.2f}% of ES)")

    except Exception as e:
        st.error(f"Error processing data: {e}")
        import traceback
        st.code(traceback.format_exc())
else:
    st.info("Please upload an Excel file to proceed.")