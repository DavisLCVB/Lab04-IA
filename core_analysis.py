# core_analysis.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import matplotlib.pyplot as plt
from kneed import KneeLocator
import json

class MethaneEmissionsAnalyzer:
    """
    Clase para analizar emisiones de metano mediante clustering.
    Proporciona métodos para procesamiento, modelado y visualización de datos.
    """
    
    def __init__(self, data_path):
        """
        Inicializa el analizador con la ruta al dataset.
        
        Args:
            data_path (str): Ruta al archivo CSV con los datos de emisiones
        """
        self.df = self.load_data(data_path)
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.cluster_labels = None
        self.optimal_k = None
        
    def load_data(self, data_path):
        """
        Carga y preprocesa básicamente los datos de emisiones.
        
        Args:
            data_path (str): Ruta al archivo de datos
            
        Returns:
            pd.DataFrame: DataFrame con los datos procesados
        """
        df = pd.read_csv(data_path)
        
        # Limpieza básica y transformaciones
        df['emissions'] = pd.to_numeric(df['emissions'], errors='coerce')
        df = df.dropna(subset=['emissions'])
        df['log_emissions'] = np.log1p(df['emissions'])
        
        # Agregar datos regionales si no están presentes
        if 'region' not in df.columns:
            df = self._add_region_data(df)
            
        return df
    
    def _add_region_data(self, df):
        """
        Añade información regional usando un dataset geoespacial integrado.
        """
        # Esto es un placeholder - en implementación real usaríamos un dataset geográfico
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        region_mapping = dict(zip(world['name'], world['continent']))
        
        df['region'] = df['country'].map(region_mapping)
        df['region'] = df['region'].fillna('Unknown')
        
        return df
    
    def preprocess_for_clustering(self, features):
        """
        Prepara los datos para clustering seleccionando y escalando features.
        
        Args:
            features (list): Lista de columnas a usar para clustering
            
        Returns:
            np.array: Datos escalados listos para clustering
        """
        # Seleccionar y codificar variables categóricas
        df_numeric = self.df.select_dtypes(include=[np.number])
        
        if 'type' in features and 'type' in self.df.columns:
            type_dummies = pd.get_dummies(self.df['type'], prefix='type')
            df_numeric = pd.concat([df_numeric, type_dummies], axis=1)
            
        if 'segment' in features and 'segment' in self.df.columns:
            segment_dummies = pd.get_dummies(self.df['segment'], prefix='segment')
            df_numeric = pd.concat([df_numeric, segment_dummies], axis=1)
            
        # Seleccionar solo las features solicitadas que existen
        available_features = [f for f in features if f in df_numeric.columns]
        clustering_data = df_numeric[available_features]
        
        # Escalar datos
        scaled_data = self.scaler.fit_transform(clustering_data)
        
        return scaled_data
    
    def find_optimal_clusters(self, data, max_k=10):
        """
        Determina el número óptimo de clusters usando el método del codo y silueta.
        
        Args:
            data (np.array): Datos escalados para clustering
            max_k (int): Número máximo de clusters a evaluar
            
        Returns:
            int: Número óptimo de clusters
        """
        wcss = []
        silhouette_scores = []
        k_values = range(2, max_k+1)
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            wcss.append(kmeans.inertia_)
            
            if k > 1:  # Silhouette necesita al menos 2 clusters
                score = silhouette_score(data, kmeans.labels_)
                silhouette_scores.append(score)
        
        # Método del codo
        knee_locator = KneeLocator(range(2, max_k+1), wcss, curve='convex', direction='decreasing')
        optimal_k_elbow = knee_locator.elbow
        
        # Método de silueta
        optimal_k_silhouette = np.argmax(silhouette_scores) + 2  # +2 porque empezamos en k=2
        
        # Tomamos el promedio de ambos métodos, redondeado
        self.optimal_k = int(round((optimal_k_elbow + optimal_k_silhouette)/2))
        
        return self.optimal_k, k_values, wcss, silhouette_scores
    
    def perform_clustering(self, data, n_clusters):
        """
        Ejecuta el algoritmo K-means con el número de clusters especificado.
        
        Args:
            data (np.array): Datos escalados para clustering
            n_clusters (int): Número de clusters a crear
        """
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans_model.fit_predict(data)
        
        # Añadir etiquetas al dataframe original
        self.df['cluster'] = self.cluster_labels
        
    def analyze_clusters(self):
        """
        Analiza las características de cada cluster y genera insights.
        
        Returns:
            pd.DataFrame: Estadísticas resumidas por cluster
            dict: Interpretación de cada cluster
        """
        if self.cluster_labels is None:
            raise ValueError("Primero debe ejecutarse el clustering")
            
        # Calcular estadísticas por cluster
        cluster_stats = self.df.groupby('cluster').agg({
            'emissions': ['mean', 'median', 'sum', 'count'],
            'log_emissions': ['mean', 'median'],
            'country': 'nunique',
            'region': lambda x: x.mode()[0]
        })
        
        # Renombrar columnas para claridad
        cluster_stats.columns = [
            'mean_emissions', 'median_emissions', 'total_emissions', 'n_entries',
            'mean_log_emissions', 'median_log_emissions', 'n_countries', 'most_common_region'
        ]
        
        # Resetear índice para mejor visualización
        cluster_stats = cluster_stats.reset_index()
        
        # Generar interpretaciones
        interpretations = self._generate_cluster_interpretations(cluster_stats)
        
        return cluster_stats, interpretations
    
    def _generate_cluster_interpretations(self, cluster_stats):
        """
        Genera interpretaciones textuales de los clusters basadas en sus estadísticas.
        
        Args:
            cluster_stats (pd.DataFrame): Estadísticas por cluster
            
        Returns:
            dict: Interpretaciones por cluster
        """
        interpretations = {}
        
        # Ordenar clusters por emisiones totales
        sorted_clusters = cluster_stats.sort_values('total_emissions', ascending=False)
        
        for _, row in sorted_clusters.iterrows():
            cluster_num = row['cluster']
            
            # Determinar características clave
            size_rank = (sorted_clusters['total_emissions'] > row['total_emissions']).sum() + 1
            region = row['most_common_region']
            n_countries = row['n_countries']
            avg_emissions = row['mean_emissions']
            
            # Generar interpretación
            interpretation = (
                f"Cluster {cluster_num} es el {size_rank}° en contribución total de emisiones. "
                f"Principalmente compuesto por países de {region} ({n_countries} países). "
                f"Las emisiones promedio son {avg_emissions:,.0f} kt, "
            )
            
            # Añadir detalles específicos basados en percentiles
            if row['total_emissions'] > cluster_stats['total_emissions'].quantile(0.75):
                interpretation += "representando una de las mayores fuentes de emisiones de metano. "
                policy_implication = "Prioridad alta para intervenciones políticas."
            elif row['total_emissions'] > cluster_stats['total_emissions'].quantile(0.5):
                interpretation += "con contribución significativa a las emisiones globales. "
                policy_implication = "Considerar estrategias de mitigación específicas."
            else:
                interpretation += "con contribución menor al panorama global. "
                policy_implication = "Enfoque en prevención de crecimiento de emisiones."
            
            interpretation += f"\n\nImplicación política: {policy_implication}"
            
            interpretations[cluster_num] = interpretation
            
        return interpretations
    
    def visualize_clusters(self, plot_type='map'):
        """
        Genera visualizaciones de los clusters.
        
        Args:
            plot_type (str): Tipo de visualización ('map', 'scatter', 'bar')
            
        Returns:
            plotly.graph_objects.Figure: Figura generada
        """
        if plot_type == 'map':
            return self._create_cluster_map()
        elif plot_type == 'scatter':
            return self._create_scatter_plot()
        elif plot_type == 'bar':
            return self._create_bar_chart()
        else:
            raise ValueError("Tipo de gráfico no soportado")
    
    def _create_cluster_map(self):
        """Versión simplificada sin GeoPandas"""
        country_data = self.df.groupby('country').agg({
            'emissions': 'sum',
            'cluster': lambda x: x.mode()[0],
            'region': 'first'
        }).reset_index()
        
        fig = px.choropleth(
            country_data,
            locations='country',
            locationmode='country names',
            color='cluster',
            hover_name='country',
            hover_data=['emissions', 'region'],
            projection='natural earth',
            title='Distribución Global de Clusters de Emisiones de Metano'
        )
        
        fig.update_layout(height=600)
        return fig
    
    def _create_scatter_plot(self):
        """
        Crea un scatter plot interactivo de emisiones vs. otra variable con colores por cluster.
        """
        # Seleccionar variables numéricas disponibles
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        x_axis = 'emissions' if 'emissions' in numeric_cols else numeric_cols[0]
        y_axis = 'log_emissions' if 'log_emissions' in numeric_cols else numeric_cols[1]
        
        fig = px.scatter(
            self.df,
            x=x_axis,
            y=y_axis,
            color='cluster',
            hover_name='country',
            hover_data=['region', 'type', 'segment'],
            title='Relación entre Variables por Cluster',
            labels={
                'emissions': 'Emisiones de Metano (kt)',
                'log_emissions': 'Log Emisiones de Metano',
                'cluster': 'Cluster'
            }
        )
        
        fig.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
        fig.update_layout(height=600)
        
        return fig
    
    def _create_bar_chart(self):
        """
        Crea un gráfico de barras mostrando emisiones promedio por cluster y región/segmento.
        """
        # Agrupar datos por cluster y región o segmento (lo que esté disponible)
        group_by = 'region' if 'region' in self.df.columns else 'segment' if 'segment' in self.df.columns else 'type'
        
        cluster_group = self.df.groupby(['cluster', group_by])['emissions'].mean().reset_index()
        
        fig = px.bar(
            cluster_group,
            x='cluster',
            y='emissions',
            color=group_by,
            barmode='group',
            title=f'Emisiones Promedio por Cluster y {group_by.capitalize()}',
            labels={
                'emissions': 'Emisiones Promedio (kt)',
                'cluster': 'Cluster',
                group_by: group_by.capitalize()
            }
        )
        
        fig.update_layout(height=500, xaxis={'type': 'category'})
        
        return fig