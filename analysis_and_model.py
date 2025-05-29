import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
def analysis_and_model_page():
    st.title("Анализ данных и модель")
    # Загрузка данных
    uploaded_file = st.file_uploader("Загрузите датасет (CSV)", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        data = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
        data['Type'] = data['Type'].map({'L': 0, 'M': 1, 'H': 2})
        #пред обработка данных
        scaler = StandardScaler()
        numerical_features = ['Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        data[numerical_features] = scaler.fit_transform(data[numerical_features])
        X = data.drop(columns=['Machine failure'])
        y = data['Machine failure']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train.columns = X_train.columns.astype(str)
        X_train.columns = X_train.columns.str.replace(r"[\[\]<>]", "", regex=True)
        X_test.columns = X_test.columns.astype(str)
        X_test.columns = X_test.columns.str.replace(r"[\[\]<>]", "", regex=True)
        
        model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] # Вероятности для ROC-AUC
        # Метрики
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        # Визуализация результатов
        st.header("Результаты обучения модели")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"ROC-AUC: {roc_auc:.2f}")
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)  
        st.subheader("Classification Report")
        st.text(class_report)

        # Интерфейс для предсказания
        st.header("Предсказание по новым данным")
        with st.form("prediction_form"):
            st.write("Введите значения признаков для предсказания:")
            productID = st.selectbox("productID", ["L", "M", "H"])
            air_temp = st.number_input("air temperature [K]")
            process_temp = st.number_input("process temperature [K]")
            rotational_speed = st.number_input("rotational speed [rpm]")
            torque = st.number_input("torque [Nm]")
            tool_wear = st.number_input("tool wear [min]")
            submit_button = st.form_submit_button("Предсказать")
        if submit_button:
            # Преобразование введенных данных
            input_dict = {
                'Type': [productID],
                'Air temperature [K]': [air_temp],
                'Process temperature [K]': [process_temp],
                'Rotational speed [rpm]': [rotational_speed],
                'Torque [Nm]': [torque],
                'Tool wear [min]': [tool_wear],
            }

            input_df = pd.DataFrame(input_dict)
            input_df['Type'] = input_df['Type'].map({'L': 0, 'M': 1, 'H': 2})
            input_df[numerical_features] = scaler.transform(input_df[numerical_features])
            input_df.columns = X_train.columns.str.replace(r"[\[\]<>]", "", regex=True)
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0][1]

            st.markdown("### Результат предсказания:")
            st.write(f"**Предсказание:" + ("отказ**" if prediction else "работает**"))
            st.write(f"**Вероятность отказа:** {prediction_proba:.2f}")
