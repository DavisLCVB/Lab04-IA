import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, classification_report, confusion_matrix)

import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="🏥 Predictor de Medicamentos",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DrugPredictionSystem:
    """
    Sistema completo de predicción de medicamentos usando modelos de caja blanca.
    Diseñado para principiantes en Machine Learning.
    """
    
    def __init__(self):
        self.models = {}
        self.best_models = {}
        self.scalers = {}
        self.encoders = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.target_names = None
        
    def load_and_preprocess_data(self, df):
        """
        Carga y preprocesa los datos del dataset de medicamentos.
        
        Args:
            df: DataFrame con los datos del dataset
            
        Returns:
            X_processed, y_processed: Datos preprocesados
        """
        st.subheader("📊 Análisis Exploratorio de Datos")
        
        # Mostrar información básica del dataset
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Número de Pacientes", len(df))
        with col2:
            st.metric("Características", len(df.columns) - 1)
        with col3:
            st.metric("Tipos de Medicamentos", df['Drug'].nunique())
        
        # Mostrar distribución de medicamentos
        fig_dist = px.pie(df, names='Drug', title="Distribución de Tipos de Medicamentos")
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Preparar los datos
        X = df.drop('Drug', axis=1)
        y = df['Drug']
        
        # Codificar variables categóricas
        le_sex = LabelEncoder()
        le_bp = LabelEncoder()
        le_chol = LabelEncoder()
        le_drug = LabelEncoder()
        
        X_processed = X.copy()
        X_processed['Sex'] = le_sex.fit_transform(X['Sex'])
        X_processed['BP'] = le_bp.fit_transform(X['BP'])
        X_processed['Cholesterol'] = le_chol.fit_transform(X['Cholesterol'])
        
        y_processed = le_drug.fit_transform(y)
        
        # Guardar encoders para uso posterior
        self.encoders = {
            'sex': le_sex,
            'bp': le_bp,
            'cholesterol': le_chol,
            'drug': le_drug
        }
        
        self.feature_names = X.columns.tolist()
        self.target_names = le_drug.classes_
        
        # Matriz de correlación
        st.subheader("🔗 Matriz de Correlación")
        corr_matrix = X_processed.corr()
        fig_corr = px.imshow(corr_matrix, 
                            text_auto=True, 
                            aspect="auto",
                            title="Correlación entre Características")
        st.plotly_chart(fig_corr, use_container_width=True)
        
        return X_processed, y_processed
    
    def split_and_scale_data(self, X, y):
        """
        Divide y escala los datos para entrenamiento y prueba.
        """
        # División 80/20
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Escalado de características
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        
        self.scalers['standard'] = scaler
        
        st.success(f"✅ Datos divididos: {len(self.X_train)} entrenamiento, {len(self.X_test)} prueba")
        
    def define_models_and_grids(self):
        """
        Define los modelos de caja blanca y sus grids de hiperparámetros.
        """
        self.models = {
            'Decision Tree': {
                'model': DecisionTreeClassifier(random_state=42),
                'param_grid': {
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'criterion': ['gini', 'entropy']
                },
                'use_scaled': False
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'param_grid': {
                    'C': [0.1, 1.0, 10.0, 100.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                },
                'use_scaled': True
            },
            'Naive Bayes': {
                'model': GaussianNB(),
                'param_grid': {
                    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
                },
                'use_scaled': True
            }
        }
        
    def train_models_with_grid_search(self):
        """
        Entrena todos los modelos usando Grid Search y validación cruzada.
        """
        st.subheader("🎯 Entrenamiento de Modelos con Grid Search")
        
        results = {}
        progress_bar = st.progress(0)
        
        for i, (name, config) in enumerate(self.models.items()):
            st.write(f"**Entrenando {name}...**")
            
            # Seleccionar datos (escalados o no)
            X_train_data = self.X_train_scaled if config['use_scaled'] else self.X_train
            X_test_data = self.X_test_scaled if config['use_scaled'] else self.X_test
            
            # Grid Search con validación cruzada K=5
            grid_search = GridSearchCV(
                config['model'],
                config['param_grid'],
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(X_train_data, self.y_train)
            
            # Mejor modelo
            best_model = grid_search.best_estimator_
            self.best_models[name] = best_model
            
            # Predicciones
            y_pred_train = best_model.predict(X_train_data)
            y_pred_test = best_model.predict(X_test_data)
            
            # Métricas
            train_accuracy = accuracy_score(self.y_train, y_pred_train)
            test_accuracy = accuracy_score(self.y_test, y_pred_test)
            
            # Validación cruzada
            cv_scores = cross_val_score(best_model, X_train_data, self.y_train, cv=5)
            
            results[name] = {
                'best_params': grid_search.best_params_,
                'cv_score_mean': cv_scores.mean(),
                'cv_score_std': cv_scores.std(),
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'y_pred_test': y_pred_test
            }
            
            # Mostrar resultados del modelo
            with st.expander(f"📋 Resultados de {name}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Mejores Hiperparámetros:**")
                    for param, value in grid_search.best_params_.items():
                        st.write(f"- {param}: {value}")
                with col2:
                    st.metric("Precisión CV", f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                    st.metric("Precisión Prueba", f"{test_accuracy:.3f}")
            
            progress_bar.progress((i + 1) / len(self.models))
        
        return results
    
    def generate_detailed_metrics(self, results):
        """
        Genera métricas detalladas para todos los modelos.
        """
        st.subheader("📈 Métricas Detalladas de los Modelos")
        
        # Tabla comparativa
        comparison_data = []
        for name, result in results.items():
            comparison_data.append({
                'Modelo': name,
                'Precisión CV': f"{result['cv_score_mean']:.3f} ± {result['cv_score_std']:.3f}",
                'Precisión Entrenamiento': f"{result['train_accuracy']:.3f}",
                'Precisión Prueba': f"{result['test_accuracy']:.3f}",
                'Overfitting': f"{result['train_accuracy'] - result['test_accuracy']:.3f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Métricas detalladas por modelo
        for name, result in results.items():
            with st.expander(f"📊 Métricas Detalladas - {name}"):
                y_pred = result['y_pred_test']
                
                # Métricas por clase
                precision = precision_score(self.y_test, y_pred, average=None)
                recall = recall_score(self.y_test, y_pred, average=None)
                f1 = f1_score(self.y_test, y_pred, average=None)
                
                # Crear DataFrame de métricas
                metrics_df = pd.DataFrame({
                    'Medicamento': self.target_names,
                    'Precisión': precision,
                    'Recall': recall,
                    'F1-Score': f1
                })
                
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(metrics_df, use_container_width=True)
                
                with col2:
                    # Matriz de confusión
                    cm = confusion_matrix(self.y_test, y_pred)
                    fig_cm = px.imshow(cm, 
                                      text_auto=True,
                                      aspect="auto",
                                      title=f"Matriz de Confusión - {name}",
                                      labels=dict(x="Predicho", y="Real"),
                                      x=self.target_names,
                                      y=self.target_names)
                    st.plotly_chart(fig_cm, use_container_width=True)
        
        # Gráfico comparativo de rendimiento
        st.subheader("📊 Comparación de Rendimiento")
        
        models_names = list(results.keys())
        cv_scores = [results[name]['cv_score_mean'] for name in models_names]
        test_scores = [results[name]['test_accuracy'] for name in models_names]
        
        fig_comparison = go.Figure(data=[
            go.Bar(name='Validación Cruzada', x=models_names, y=cv_scores),
            go.Bar(name='Prueba', x=models_names, y=test_scores)
        ])
        fig_comparison.update_layout(
            title="Comparación de Precisión de Modelos",
            yaxis_title="Precisión",
            barmode='group'
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    def visualize_model_interpretability(self):
        """
        Visualiza la interpretabilidad de los modelos de caja blanca.
        """
        st.subheader("🔍 Interpretabilidad de los Modelos")
        
        # Decision Tree Visualization
        if 'Decision Tree' in self.best_models:
            st.write("**🌳 Árbol de Decisión**")
            with st.expander("Ver Estructura del Árbol"):
                fig, ax = plt.subplots(figsize=(20, 10))
                plot_tree(self.best_models['Decision Tree'], 
                         feature_names=self.feature_names,
                         class_names=self.target_names,
                         filled=True, 
                         rounded=True,
                         fontsize=10)
                st.pyplot(fig)
                
                # Importancia de características
                importances = self.best_models['Decision Tree'].feature_importances_
                feat_imp_df = pd.DataFrame({
                    'Característica': self.feature_names,
                    'Importancia': importances
                }).sort_values('Importancia', ascending=False)
                
                fig_imp = px.bar(feat_imp_df, 
                               x='Importancia', 
                               y='Característica',
                               orientation='h',
                               title="Importancia de Características - Decision Tree")
                st.plotly_chart(fig_imp, use_container_width=True)
        
        # Logistic Regression Coefficients
        if 'Logistic Regression' in self.best_models:
            st.write("**📈 Regresión Logística - Coeficientes**")
            with st.expander("Ver Coeficientes del Modelo"):
                lr_model = self.best_models['Logistic Regression']
                
                # Para clasificación multiclase
                if hasattr(lr_model, 'coef_'):
                    coef_df_list = []
                    for i, class_name in enumerate(self.target_names):
                        for j, feature in enumerate(self.feature_names):
                            coef_df_list.append({
                                'Medicamento': class_name,
                                'Característica': feature,
                                'Coeficiente': lr_model.coef_[i][j]
                            })
                    
                    coef_df = pd.DataFrame(coef_df_list)
                    
                    # Heatmap de coeficientes
                    pivot_coef = coef_df.pivot(index='Característica', 
                                             columns='Medicamento', 
                                             values='Coeficiente')
                    
                    fig_coef = px.imshow(pivot_coef, 
                                       text_auto=True,
                                       aspect="auto",
                                       title="Coeficientes de Regresión Logística")
                    st.plotly_chart(fig_coef, use_container_width=True)
    
    def create_prediction_interface(self):
        """
        Crea la interfaz para hacer predicciones con datos de nuevos pacientes.
        """
        st.subheader("🩺 Predicción para Nuevo Paciente")
        
        with st.form("patient_prediction"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Edad", min_value=0, max_value=120, value=30)
                sex = st.selectbox("Sexo", ["M", "F"])
                bp = st.selectbox("Presión Arterial", ["HIGH", "NORMAL", "LOW"])
            
            with col2:
                cholesterol = st.selectbox("Colesterol", ["HIGH", "NORMAL"])
                na_k_ratio = st.number_input("Relación Na/K", min_value=0.0, max_value=50.0, value=15.0, step=0.1)
            
            submit_button = st.form_submit_button("🔮 Predecir Medicamento")
            
            if submit_button and hasattr(self, 'best_models'):
                # Preparar datos del paciente
                patient_data = pd.DataFrame({
                    'Age': [age],
                    'Sex': [sex],
                    'BP': [bp],
                    'Cholesterol': [cholesterol],
                    'Na_to_K': [na_k_ratio]
                })
                
                # Codificar datos
                patient_encoded = patient_data.copy()
                patient_encoded['Sex'] = self.encoders['sex'].transform([sex])[0]
                patient_encoded['BP'] = self.encoders['bp'].transform([bp])[0]
                patient_encoded['Cholesterol'] = self.encoders['cholesterol'].transform([cholesterol])[0]
                
                # Hacer predicciones con todos los modelos
                st.write("### 🎯 Predicciones de los Modelos")
                
                predictions = {}
                confidences = {}
                
                for name, model in self.best_models.items():
                    # Seleccionar datos apropiados
                    if name == 'Decision Tree':
                        patient_input = patient_encoded.values
                    else:
                        patient_input = self.scalers['standard'].transform(patient_encoded.values)
                    
                    # Predicción
                    pred_encoded = model.predict(patient_input)[0]
                    pred_drug = self.encoders['drug'].inverse_transform([pred_encoded])[0]
                    predictions[name] = pred_drug
                    
                    # Confianza (probabilidades)
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(patient_input)[0]
                        confidences[name] = max(proba)
                
                # Mostrar resultados
                for name, pred in predictions.items():
                    confidence = confidences.get(name, 0)
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{name}:** {pred}")
                    with col2:
                        if confidence > 0:
                            st.write(f"Confianza: {confidence:.2%}")
                
                # Consenso de modelos
                pred_counts = pd.Series(list(predictions.values())).value_counts()
                most_common = pred_counts.index[0]
                
                st.success(f"### 🏆 Recomendación Final: **{most_common}**")
                
                if len(pred_counts) > 1:
                    st.info(f"📊 Consenso: {pred_counts[most_common]}/{len(predictions)} modelos concuerdan")
                
                # Explicación de la recomendación
                st.write("### 💡 Interpretación de la Recomendación")
                
                if 'Decision Tree' in self.best_models:
                    # Mostrar el camino en el árbol de decisión
                    dt_model = self.best_models['Decision Tree']
                    leaf_id = dt_model.decision_path(patient_encoded.values).indices[-1]
                    
                    st.write("**Factores clave según el Árbol de Decisión:**")
                    feature_importance = dt_model.feature_importances_
                    top_features = sorted(zip(self.feature_names, feature_importance), 
                                        key=lambda x: x[1], reverse=True)[:3]
                    
                    for feature, importance in top_features:
                        if feature == 'Age':
                            value = age
                        elif feature == 'Sex':
                            value = sex
                        elif feature == 'BP':
                            value = bp
                        elif feature == 'Cholesterol':
                            value = cholesterol
                        else:  # Na_to_K
                            value = na_k_ratio
                        
                        st.write(f"- **{feature}**: {value} (importancia: {importance:.3f})")

def main():
    """
    Función principal de la aplicación Streamlit.
    """
    st.title("🏥 Sistema de Predicción de Medicamentos")
    st.markdown("""
    ### 🎯 Aprendizaje de Machine Learning para Principiantes
    
    Esta aplicación demuestra cómo usar **modelos de caja blanca** para predecir 
    medicamentos apropiados basándose en características de pacientes.
    
    **Modelos implementados:**
    - 🌳 **Decision Tree**: Fácil de interpretar, muestra reglas de decisión
    - 📈 **Logistic Regression**: Modelo lineal con coeficientes interpretables  
    - 🎯 **Naive Bayes**: Modelo probabilístico basado en teorema de Bayes
    """)
    
    # Inicializar el sistema
    if 'drug_system' not in st.session_state:
        st.session_state.drug_system = DrugPredictionSystem()
    
    system = st.session_state.drug_system
    
    # Sidebar para navegación
    st.sidebar.title("🧭 Navegación")
    option = st.sidebar.selectbox(
        "Selecciona una sección:",
        ["📁 Cargar Datos", "🔬 Entrenar Modelos", "🩺 Hacer Predicciones", "📚 Documentación"]
    )
    
    if option == "📁 Cargar Datos":
        st.header("📁 Carga y Exploración de Datos")
        
        # Opción para cargar archivo
        uploaded_file = st.file_uploader(
            "Sube tu archivo CSV del dataset de medicamentos",
            type=['csv'],
            help="El archivo debe contener las columnas: Age, Sex, BP, Cholesterol, Na_to_K, Drug"
        )
        
        # Opción para generar datos de ejemplo
        if st.button("🎲 Generar Datos de Ejemplo"):
            # Crear dataset de ejemplo
            np.random.seed(42)
            n_samples = 200
            
            example_data = {
                'Age': np.random.randint(20, 70, n_samples),
                'Sex': np.random.choice(['M', 'F'], n_samples),
                'BP': np.random.choice(['HIGH', 'NORMAL', 'LOW'], n_samples),
                'Cholesterol': np.random.choice(['HIGH', 'NORMAL'], n_samples),
                'Na_to_K': np.random.uniform(6, 38, n_samples),
                'Drug': np.random.choice(['DrugY', 'drugA', 'drugB', 'drugC', 'drugX'], n_samples)
            }
            
            st.session_state.df = pd.DataFrame(example_data)
            uploaded_file = "example"
        
        if uploaded_file is not None:
            if uploaded_file != "example":
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
            else:
                df = st.session_state.df
            
            st.success("✅ ¡Datos cargados exitosamente!")
            
            # Mostrar preview de los datos
            st.subheader("👀 Vista Previa de los Datos")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Información del dataset
            st.subheader("ℹ️ Información del Dataset")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Filas", len(df))
            with col2:
                st.metric("Columnas", len(df.columns))
            with col3:
                st.metric("Valores Nulos", df.isnull().sum().sum())
            with col4:
                st.metric("Medicamentos Únicos", df['Drug'].nunique())
            
            # Procesar datos
            if st.button("🔄 Procesar Datos"):
                with st.spinner("Procesando datos..."):
                    X, y = system.load_and_preprocess_data(df)
                    system.split_and_scale_data(X, y)
                    st.session_state.data_processed = True
                    st.success("✅ ¡Datos procesados y listos para entrenamiento!")
    
    elif option == "🔬 Entrenar Modelos":
        st.header("🔬 Entrenamiento de Modelos de Machine Learning")
        
        if not hasattr(st.session_state, 'data_processed'):
            st.warning("⚠️ Primero debes cargar y procesar los datos en la sección 'Cargar Datos'")
            return
        
        if st.button("🚀 Entrenar Todos los Modelos"):
            with st.spinner("Entrenando modelos... Esto puede tomar unos minutos."):
                # Definir modelos y grids
                system.define_models_and_grids()
                
                # Entrenar con grid search
                results = system.train_models_with_grid_search()
                st.session_state.training_results = results
                
                # Generar métricas detalladas
                system.generate_detailed_metrics(results)
                
                # Visualizar interpretabilidad
                system.visualize_model_interpretability()
                
                st.session_state.models_trained = True
                st.balloons()
                st.success("🎉 ¡Todos los modelos han sido entrenados exitosamente!")
        
        # Mostrar resultados si ya están entrenados
        if hasattr(st.session_state, 'training_results'):
            system.generate_detailed_metrics(st.session_state.training_results)
            system.visualize_model_interpretability()
    
    elif option == "🩺 Hacer Predicciones":
        st.header("🩺 Predicciones para Nuevos Pacientes")
        
        if not hasattr(st.session_state, 'models_trained'):
            st.warning("⚠️ Primero debes entrenar los modelos en la sección 'Entrenar Modelos'")
            return
        
        system.create_prediction_interface()
    
    elif option == "📚 Documentación":
        st.header("📚 Documentación y Guía de Aprendizaje")
        
        st.markdown("""
        ## 🎯 Objetivo del Proyecto
        
        Este sistema está diseñado para enseñar conceptos fundamentales de Machine Learning
        usando un problema real de predicción de medicamentos.
        
        ## 🔍 Modelos de Caja Blanca Implementados
        
        ### 🌳 Decision Tree (Árbol de Decisión)
        - **¿Cómo funciona?** Crea reglas de decisión simples basadas en las características
        - **Interpretabilidad:** Muy alta - puedes ver exactamente qué decisiones toma
        - **Ventajas:** Fácil de entender, no requiere escalado de datos
        - **Desventajas:** Puede sobreajustarse fácilmente
        
        ### 📈 Logistic Regression (Regresión Logística)
        - **¿Cómo funciona?** Encuentra la mejor línea para separar las clases
        - **Interpretabilidad:** Alta - los coeficientes muestran la importancia de cada característica
        - **Ventajas:** Rápido, estable, proporciona probabilidades
        - **Desventajas:** Asume relaciones lineales
        
        ### 🎯 Naive Bayes
        - **¿Cómo funciona?** Usa probabilidades condicionales (Teorema de Bayes)
        - **Interpretabilidad:** Moderada - muestra probabilidades por característica
        - **Ventajas:** Funciona bien con pocos datos, rápido
        - **Desventajas:** Asume independencia entre características
        
        ## 📊 Métricas de Evaluación
        
        - **Accuracy (Precisión):** % de predicciones correctas
        - **Precision:** De las predicciones positivas, % que son correctas
        - **Recall:** De los casos positivos reales, % que fueron detectados
        - **F1-Score:** Media armónica entre precision y recall
        
        ## 🔧 Técnicas Implementadas
        
        ### ✂️ División de Datos (80/20)
        - 80% para entrenamiento
        - 20% para prueba final
        
        ### 🔄 Validación Cruzada (K=5)
        - Divide los datos de entrenamiento en 5 partes
        - Entrena 5 veces, cada vez dejando una parte para validación
        - Proporciona una estimación más robusta del rendimiento
        
        ### 🎛️ Grid Search
        - Prueba automáticamente diferentes combinaciones de hiperparámetros
        - Encuentra la mejor configuración para cada modelo
        - Evita el proceso manual de prueba y error
        
        ## 💡 Consejos para Principiantes
        
        1. **Comienza simple:** Los modelos de caja blanca son perfectos para aprender
        2. **Entiende tus datos:** Siempre explora antes de modelar
        3. **No te obsesiones con la precisión:** La interpretabilidad también es importante
        4. **Valida correctamente:** Usa validación cruzada para resultados confiables
        5. **Compara modelos:** Diferentes modelos pueden ser mejores para diferentes problemas
        
        ## 🚀 Próximos Pasos
        
        1. Experimenta con diferentes conjuntos de datos
        2. Prueba técnicas de ingeniería de características
        3. Aprende sobre ensemble methods (Random Forest, etc.)
        4. Explora modelos de caja negra (Neural Networks, SVM)
        """)

if __name__ == "__main__":
    main()