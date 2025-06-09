# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from core_analysis import MethaneEmissionsAnalyzer

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Clusters de Emisiones de Metano",
    page_icon="🌍",
    layout="wide"
)

# Título y descripción
st.title("🌍 Análisis de Clusters de Emisiones de Metano")
st.markdown("""
Esta herramienta utiliza algoritmos de machine learning para identificar patrones en las emisiones globales de metano,
un gas que contribuye significativamente al calentamiento global. Los resultados pueden ayudar a diseñar políticas
ambientales más efectivas.
""")

# Carga de datos
@st.cache_resource
def load_data_and_analyzer():
    """Carga los datos y inicializa el analizador."""
    # En una implementación real, esto cargaría el dataset real
    # Aquí creamos datos de ejemplo para demostración
    data_path = "data/Methane_final.csv"
    return MethaneEmissionsAnalyzer(data_path)

analyzer = load_data_and_analyzer()

# Sidebar para configuración
st.sidebar.header("Configuración del Modelo")
st.sidebar.markdown("Seleccione las características y parámetros para el análisis de clustering.")

# Selección de características
available_features = analyzer.df.select_dtypes(include=[np.number, 'object']).columns.tolist()
default_features = ['emissions', 'log_emissions', 'type', 'segment']
selected_features = st.sidebar.multiselect(
    "Características para clustering:",
    options=available_features,
    default=default_features
)

# Número de clusters (automático o manual)
cluster_mode = st.sidebar.radio(
    "Determinación de clusters:",
    options=['Automático (método del codo y silueta)', 'Manual'],
    index=0
)

if cluster_mode == 'Manual':
    n_clusters = st.sidebar.slider("Número de clusters:", 2, 10, 4)
else:
    n_clusters = None

# Botón para ejecutar análisis
run_analysis = st.sidebar.button("Ejecutar Análisis")

# Sección principal
if run_analysis and len(selected_features) > 0:
    st.header("Resultados del Análisis de Clustering")
    
    # Preprocesamiento
    with st.spinner("Preprocesando datos..."):
        scaled_data = analyzer.preprocess_for_clustering(selected_features)
    
    # Determinación de clusters óptimos
    if cluster_mode == 'Automático (método del codo y silueta)':
        with st.spinner("Determinando número óptimo de clusters..."):
            optimal_k, k_values, wcss, silhouette_scores = analyzer.find_optimal_clusters(scaled_data)
            n_clusters = optimal_k
        
        # Mostrar resultados de determinación de clusters
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Método del Codo")
            fig_elbow = px.line(
                x=k_values,
                y=wcss,
                labels={'x': 'Número de Clusters (k)', 'y': 'Suma de Cuadrados Intra-Cluster (WCSS)'},
                title='Método del Codo para Determinar k Óptimo'
            )
            fig_elbow.add_vline(x=n_clusters, line_dash="dash", line_color="red")
            st.plotly_chart(fig_elbow, use_container_width=True)
        
        with col2:
            st.subheader("Método de Silueta")
            fig_silhouette = px.line(
                x=range(2, len(silhouette_scores)+2),
                y=silhouette_scores,
                labels={'x': 'Número de Clusters (k)', 'y': 'Puntaje de Silueta'},
                title='Análisis de Silueta para Determinar k Óptimo'
            )
            fig_silhouette.add_vline(x=n_clusters, line_dash="dash", line_color="red")
            st.plotly_chart(fig_silhouette, use_container_width=True)
        
        st.success(f"Número óptimo de clusters determinado: {n_clusters}")
    
    # Ejecutar clustering
    with st.spinner(f"Ejecutando K-means con {n_clusters} clusters..."):
        analyzer.perform_clustering(scaled_data, n_clusters)
    
    # Mostrar resultados
    st.subheader("Distribución de Clusters")
    
    # Visualización seleccionada
    viz_option = st.selectbox(
        "Tipo de visualización:",
        options=['Mapa Global', 'Scatter Plot', 'Gráfico de Barras'],
        index=0
    )
    
    if viz_option == 'Mapa Global':
        fig = analyzer.visualize_clusters('map')
    elif viz_option == 'Scatter Plot':
        fig = analyzer.visualize_clusters('scatter')
    else:
        fig = analyzer.visualize_clusters('bar')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Estadísticas e interpretación de clusters
    st.subheader("Análisis Detallado por Cluster")
    
    cluster_stats, interpretations = analyzer.analyze_clusters()
    
    # Mostrar estadísticas
    st.dataframe(
        cluster_stats.style
        .background_gradient(cmap='YlOrRd', subset=['total_emissions', 'mean_emissions'])
        .format({
            'mean_emissions': '{:,.0f}',
            'median_emissions': '{:,.0f}',
            'total_emissions': '{:,.0f}'
        }),
        use_container_width=True
    )
    
    # Mostrar interpretaciones
    st.subheader("Interpretación de Clusters y Recomendaciones")
    
    selected_cluster = st.selectbox(
        "Seleccione un cluster para ver su interpretación:",
        options=sorted(interpretations.keys())
    )
    
    st.markdown("### Características del Cluster")
    st.info(interpretations[selected_cluster])
    
    # Descargar resultados
    st.subheader("Exportar Resultados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="Descargar Datos con Clusters",
            data=analyzer.df.to_csv(index=False).encode('utf-8'),
            file_name="methane_emissions_with_clusters.csv",
            mime="text/csv"
        )
    
    with col2:
        st.download_button(
            label="Descargar Estadísticas por Cluster",
            data=cluster_stats.to_csv(index=False).encode('utf-8'),
            file_name="cluster_statistics.csv",
            mime="text/csv"
        )
    
elif run_analysis and len(selected_features) == 0:
    st.error("Por favor seleccione al menos una característica para el clustering.")
else:
    # Estado inicial - mostrar datos de ejemplo
    st.header("Exploración de Datos")
    st.markdown("Revise los datos antes de ejecutar el análisis.")
    
    st.dataframe(analyzer.df.head(), use_container_width=True)
    
    # Mostrar estadísticas descriptivas
    st.subheader("Estadísticas Descriptivas")
    st.dataframe(analyzer.df.describe(), use_container_width=True)
    
    # Mostrar distribución de emisiones
    fig_dist = px.histogram(
        analyzer.df,
        x='emissions',
        nbins=50,
        title='Distribución de Emisiones de Metano',
        labels={'emissions': 'Emisiones (kt)'}
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# Información adicional
st.sidebar.markdown("---")
st.sidebar.info("""
**Nota sobre el metano:**
- 30% del calentamiento global desde la Revolución Industrial
- Potencia de calentamiento 28-36 veces mayor que el CO₂ (100 años)
- Vida atmosférica: ~12 años
""")

# Pie de página
st.markdown("---")
st.markdown("""
**Uso recomendado:**
1. Seleccione características relevantes en el panel izquierdo
2. Elija método para determinar número de clusters
3. Haga clic en "Ejecutar Análisis"
4. Explore resultados e interpretaciones
""")