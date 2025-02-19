import streamlit as st
import numpy as np  
import torch  
import torch.nn as nn  
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt

class GlobalLSTM(nn.Module):
    def __init__(self, num_stores, num_depts, emb_dim_store=4, emb_dim_dept=8, num_numeric_features=15, hidden_size=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.store_emb = nn.Embedding(num_stores, emb_dim_store)
        self.dept_emb = nn.Embedding(num_depts, emb_dim_dept)
        self.input_dim = emb_dim_store + emb_dim_dept + num_numeric_features
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, X_cat, X_num):
        store_id = X_cat[:, :, 0].long()
        dept_id = X_cat[:, :, 1].long()
        store_emb_out = self.store_emb(store_id)
        dept_emb_out = self.dept_emb(dept_id)
        concat_input = torch.cat([store_emb_out, dept_emb_out, X_num], dim=-1)
        lstm_out, _ = self.lstm(concat_input)
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)
        return out.squeeze(-1)

@st.cache_data()
def load_model():
    num_stores = 45
    num_depts = 81
    model = GlobalLSTM(num_stores=num_stores, num_depts=num_depts, emb_dim_store=4, emb_dim_dept=8, num_numeric_features=15, hidden_size=64, num_layers=1, dropout=0.2)
    model.load_state_dict(torch.load("models\demand_prediction\modelo_demanda.pth", map_location="cpu"), strict=False)
    model.device = torch.device("cpu")
    return model

def preprocess_data(df):
    df.columns = df.columns.str.strip()
    df = df[df["Weekly_Sales"] > 0].reset_index(drop=True)
    df.sort_values(by=["Store", "Dept", "Date"], inplace=True)
    df.fillna(method="ffill", inplace=True)
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["IsHoliday"] = df["IsHoliday"].astype(int)
    store_encoder = LabelEncoder()
    dept_encoder = LabelEncoder()
    df["Store_id"] = store_encoder.fit_transform(df["Store"])
    df["Dept_id"] = dept_encoder.fit_transform(df["Dept"])
    num_cols = ["Size", "Temperature", "Fuel_Price", "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5", "CPI", "Unemployment", "Year", "Month", "Week", "DayOfYear", "IsHoliday"]
    max_dia = df["Date"].max()
    test_start_date = max_dia - pd.Timedelta(weeks=12)
    train_df = df[df["Date"] < test_start_date]
    test_df = df[df["Date"] >= test_start_date]
    scaler = MinMaxScaler()
    train_df[num_cols[:-1]] = scaler.fit_transform(train_df[num_cols[:-1]])
    test_df[num_cols[:-1]] = scaler.transform(test_df[num_cols[:-1]])
    return df, train_df, test_df, num_cols

def predict_demand(preprocessed_data, model, store_id, dept_id):
    df, train_df, test_df, num_cols = preprocessed_data
    cat_cols = ["Store_id", "Dept_id"]
    forecast_horizon = 4
    preds = []
    sub_df = df[(df["Store_id"] == store_id) & (df["Dept_id"] == dept_id)].sort_values("Date")
    if len(sub_df) < 8:
        raise ValueError(f"No hay suficientes datos para la tienda {store_id} y depto {dept_id}.")
    last_seq = sub_df.iloc[-8:].copy()
    X_cat = last_seq[cat_cols].values[np.newaxis, :, :]
    X_num = last_seq[num_cols].values[np.newaxis, :, :]
    X_cat_t = torch.tensor(X_cat, dtype=torch.float32).to(model.device)
    X_num_t = torch.tensor(X_num, dtype=torch.float32).to(model.device)
    model.eval()
    with torch.no_grad():
        for week_ahead in range(forecast_horizon):
            pred = model(X_cat_t, X_num_t).cpu().item()
            preds.append({"Store_id": store_id, "Dept_id": dept_id, "Week_Ahead": week_ahead + 1, "Predicted_Weekly_Sales": pred})
            new_cat = X_cat_t.cpu().numpy()[0, -1, :]
            last_num = X_num_t.cpu().numpy()[0, -1, :]
            new_num = np.append(last_num[1:], pred)
            new_X_cat = np.vstack([X_cat_t.cpu().numpy()[0, 1:, :], new_cat])
            new_X_num = np.vstack([X_num_t.cpu().numpy()[0, 1:, :], new_num])
            X_cat_t = torch.tensor(new_X_cat, dtype=torch.float32).unsqueeze(0).to(model.device)
            X_num_t = torch.tensor(new_X_num, dtype=torch.float32).unsqueeze(0).to(model.device)
    return pd.DataFrame(preds)

def plot_sales_with_predictions(df, future_pred_df, store_id, dept_id):
    sub_df = df[(df["Store_id"] == store_id) & (df["Dept_id"] == dept_id)].sort_values("Date")
    sub_df["Date"] = pd.to_datetime(sub_df["Date"])
    last_historical_date = sub_df["Date"].iloc[-1]
    last_historical_value = sub_df["Weekly_Sales"].iloc[-1]
    pred_dates = [last_historical_date] + [last_historical_date + pd.Timedelta(weeks=i) for i in range(1, len(future_pred_df) + 1)]
    pred_values = [last_historical_value] + future_pred_df["Predicted_Weekly_Sales"].tolist()
    predictions_df = pd.DataFrame({"Date": pred_dates, "Predicted_Weekly_Sales": pred_values})
    plt.figure(figsize=(12, 6))
    plt.plot(sub_df["Date"], sub_df["Weekly_Sales"], label="Ventas Históricas", color="blue", marker="o")
    plt.plot(predictions_df["Date"], predictions_df["Predicted_Weekly_Sales"], label="Predicciones", color="red", linestyle="--", marker="o")
    plt.title(f"Comportamiento de Ventas con Predicciones - Tienda {store_id}, Depto {dept_id}")
    plt.xlabel("Fecha")
    plt.ylabel("Ventas Semanales")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

def display_demand_prediction():
    st.subheader("Predicción de demanda")

    st.write("Este modelo de predicción de demanda utiliza un modelo LSTM entrenado con datos históricos de ventas para predecir las ventas futuras de un producto en una tienda específica.")
    st.write("El modelo toma como entrada un archivo CSV con los datos de ventas históricos y predice las ventas para los próximos 30 días.")
    st.image("models\demand_prediction\demand_example.png", use_container_width=True)
    with st.expander("➡️Aquí tienes un video guía para utilizar este módulo"):
        st.video(os.path.join(os.getcwd(), "videos", "Demanda.mp4"))

    st.divider()

    st.write("Sube tu archivo CSV con los datos de ventas para predecir la demanda de tus productos a 30 días.")
    input_csv_file = st.file_uploader("Escoge un archivo CSV", type="csv")

    if input_csv_file is not None:
        try:
            data = pd.read_csv(input_csv_file, parse_dates=["Date"])
            st.write("Previsualización de los datos:")
            st.write(data.head())
            
            store_id = st.text_input("Ingrese el ID de la tienda (código numérico):")
            dept_id = st.text_input("Ingrese el ID del departamento (código numérico):")

            int_store_id = int(store_id)
            int_dept_id = int(dept_id)

            model = load_model()
            preprocessed_data = preprocess_data(data)
            predictions = predict_demand(preprocessed_data, model, int_store_id, int_dept_id)
            
            st.subheader("Predicciones para los próximos 30 días:")
            st.write(predictions)            
            plot_sales_with_predictions(preprocessed_data[0], predictions, int_store_id, int_dept_id)

        except pd.errors.EmptyDataError:
            st.error("El archivo CSV no contiene datos. Por favor, sube un archivo válido.")
        except Exception as e:
            st.error(f"Se produjo un error al procesar el archivo: {e}")