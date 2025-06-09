import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class DiabetesClassificationSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Clasificación de Diabetes - ML")
        self.root.geometry("1000x700")
        
        # Colores Catppuccin Mocha
        self.colors = {
            'base': '#1e1e2e',
            'mantle': '#181825', 
            'crust': '#11111b',
            'text': '#cdd6f4',
            'subtext0': '#a6adc8',
            'subtext1': '#bac2de',
            'surface0': '#313244',
            'surface1': '#45475a',
            'surface2': '#585b70',
            'blue': '#89b4fa',
            'lavender': '#b4befe',
            'green': '#a6e3a1',
            'yellow': '#f9e2af',
            'peach': '#fab387',
            'red': '#f38ba8',
            'mauve': '#cba6f7',
            'pink': '#f5c2e7',
            'teal': '#94e2d5'
        }
        
        # Configurar tema oscuro
        self.setup_theme()
        
        # Variables de control
        self.data = None
        self.data_clean = None
        self.missing_info = {}
        self.preprocessing_done = False
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.imputer = None
        self.models_results = {}
        
        # Configuración de modelos de caja blanca con hiperparámetros
        self.models_config = {
            'Decision Tree': {
                'model': DecisionTreeClassifier(random_state=42),
                'params': {
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'criterion': ['gini', 'entropy']
                }
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1.0, 10.0, 100.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            }
        }
        
        self.setup_ui()
    
    def setup_theme(self):
        """Configurar tema Catppuccin Mocha"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configurar colores del tema
        style.configure('TFrame', background=self.colors['base'])
        style.configure('TLabel', background=self.colors['base'], foreground=self.colors['text'], 
                       font=('SF Pro Display', 10))
        style.configure('Title.TLabel', background=self.colors['base'], foreground=self.colors['blue'], 
                       font=('SF Pro Display', 20, 'bold'))
        style.configure('Heading.TLabel', background=self.colors['base'], foreground=self.colors['lavender'], 
                       font=('SF Pro Display', 12, 'bold'))
        style.configure('Subheading.TLabel', background=self.colors['base'], foreground=self.colors['subtext1'], 
                       font=('SF Pro Display', 10))
        
        # Botones modernos
        style.configure('Primary.TButton', 
                       background=self.colors['blue'],
                       foreground=self.colors['base'],
                       font=('SF Pro Display', 10, 'bold'),
                       relief='flat',
                       borderwidth=0,
                       focuscolor='none')
        style.map('Primary.TButton',
                 background=[('active', self.colors['lavender']),
                           ('pressed', self.colors['surface2'])])
        
        style.configure('Secondary.TButton', 
                       background=self.colors['surface1'],
                       foreground=self.colors['text'],
                       font=('SF Pro Display', 10),
                       relief='flat',
                       borderwidth=0,
                       focuscolor='none')
        style.map('Secondary.TButton',
                 background=[('active', self.colors['surface2']),
                           ('pressed', self.colors['surface0'])])
        
        # Checkbuttons y Radiobuttons
        style.configure('TCheckbutton', background=self.colors['base'], foreground=self.colors['text'],
                       font=('SF Pro Display', 10), focuscolor='none')
        style.map('TCheckbutton', background=[('active', self.colors['base'])])
        
        style.configure('TRadiobutton', background=self.colors['base'], foreground=self.colors['text'],
                       font=('SF Pro Display', 10), focuscolor='none')
        style.map('TRadiobutton', background=[('active', self.colors['base'])])
        
        # Notebook
        style.configure('TNotebook', background=self.colors['base'], borderwidth=0)
        style.configure('TNotebook.Tab', background=self.colors['surface1'], foreground=self.colors['text'],
                       padding=[12, 8], font=('SF Pro Display', 10))
        style.map('TNotebook.Tab', 
                 background=[('selected', self.colors['blue']),
                           ('active', self.colors['surface2'])],
                 foreground=[('selected', self.colors['base'])])
        
        # Configurar ventana principal
        self.root.configure(bg=self.colors['base'])
    
    def create_card_frame(self, parent, **kwargs):
        """Crear un frame con estilo de tarjeta moderna"""
        card = tk.Frame(parent, bg=self.colors['surface0'], relief='flat', bd=0, **kwargs)
        return card
    
    def update_progress_bar(self, value):
        """Actualizar barra de progreso personalizada"""
        try:
            if hasattr(self, 'progress_fill') and self.progress_fill.master.winfo_exists():
                parent_width = self.progress_fill.master.winfo_width()
                if parent_width > 1:  # Asegurar que el widget esté renderizado
                    new_width = int((value / 100) * parent_width)
                    self.progress_fill.place_configure(width=new_width)
                    self.root.update_idletasks()
        except Exception:
            # Si hay error con la barra de progreso, simplemente ignorar
            pass
    
    def setup_ui(self):
        # Frame principal con padding
        main_frame = tk.Frame(self.root, bg=self.colors['base'])
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Header con título
        header_frame = self.create_card_frame(main_frame)
        header_frame.pack(fill='x', pady=(0, 20))
        
        title_label = tk.Label(header_frame, text="🧬 Sistema de Clasificación de Diabetes", 
                              bg=self.colors['surface0'], fg=self.colors['blue'],
                              font=('SF Pro Display', 24, 'bold'))
        title_label.pack(pady=20)
        
        subtitle_label = tk.Label(header_frame, text="Análisis Predictivo con Modelos de Machine Learning Interpretables", 
                                 bg=self.colors['surface0'], fg=self.colors['subtext1'],
                                 font=('SF Pro Display', 12))
        subtitle_label.pack(pady=(0, 20))
        
        # Container principal con scroll
        canvas = tk.Canvas(main_frame, bg=self.colors['base'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['base'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Sección 1: Carga de datos
        data_card = self.create_card_frame(scrollable_frame)
        data_card.pack(fill='x', pady=(0, 15))
        
        data_header = tk.Frame(data_card, bg=self.colors['surface0'])
        data_header.pack(fill='x', padx=20, pady=15)
        
        tk.Label(data_header, text="📊 Carga de Dataset", 
                bg=self.colors['surface0'], fg=self.colors['green'],
                font=('SF Pro Display', 14, 'bold')).pack(anchor='w')
        
        tk.Label(data_header, text="Seleccione su archivo CSV con datos clínicos de diabetes", 
                bg=self.colors['surface0'], fg=self.colors['subtext1'],
                font=('SF Pro Display', 10)).pack(anchor='w', pady=(5, 0))
        
        data_controls = tk.Frame(data_card, bg=self.colors['surface0'])
        data_controls.pack(fill='x', padx=20, pady=(0, 20))
        
        button_frame = tk.Frame(data_controls, bg=self.colors['surface0'])
        button_frame.pack(fill='x', pady=10)
        
        self.load_button = tk.Button(button_frame, text="📁 Seleccionar Archivo", 
                                    command=self.load_data,
                                    bg=self.colors['blue'], fg=self.colors['base'],
                                    font=('SF Pro Display', 11, 'bold'),
                                    relief='flat', bd=0, padx=20, pady=12,
                                    cursor='hand2')
        self.load_button.pack(side='left')
        
        self.data_info_label = tk.Label(button_frame, text="No se ha cargado ningún dataset",
                                       bg=self.colors['surface0'], fg=self.colors['subtext0'],
                                       font=('SF Pro Display', 10))
        self.data_info_label.pack(side='left', padx=(20, 0))
        
        # Sección 2: Configuración
        config_card = self.create_card_frame(scrollable_frame)
        config_card.pack(fill='x', pady=(0, 15))
        
        config_header = tk.Frame(config_card, bg=self.colors['surface0'])
        config_header.pack(fill='x', padx=20, pady=15)
        
        tk.Label(config_header, text="⚙️ Configuración del Modelo", 
                bg=self.colors['surface0'], fg=self.colors['yellow'],
                font=('SF Pro Display', 14, 'bold')).pack(anchor='w')
        
        config_content = tk.Frame(config_card, bg=self.colors['surface0'])
        config_content.pack(fill='x', padx=20, pady=(0, 20))
        
        # Tipo de clasificación
        classification_frame = tk.Frame(config_content, bg=self.colors['surface0'])
        classification_frame.pack(fill='x', pady=10)
        
        tk.Label(classification_frame, text="Tipo de Clasificación:", 
                bg=self.colors['surface0'], fg=self.colors['text'],
                font=('SF Pro Display', 11, 'bold')).pack(anchor='w')
        
        self.binary_var = tk.BooleanVar(value=True)
        self.binary_check = tk.Checkbutton(classification_frame, 
                                          text="Clasificación Binaria (Diabetes/No Diabetes)", 
                                          variable=self.binary_var,
                                          bg=self.colors['surface0'], fg=self.colors['text'],
                                          selectcolor=self.colors['surface1'],
                                          activebackground=self.colors['surface0'],
                                          font=('SF Pro Display', 10))
        self.binary_check.pack(anchor='w', pady=5)
        
        # Estrategia de imputación
        imputation_frame = tk.Frame(config_content, bg=self.colors['surface0'])
        imputation_frame.pack(fill='x', pady=15)
        
        tk.Label(imputation_frame, text="Estrategia para Valores Faltantes:", 
                bg=self.colors['surface0'], fg=self.colors['text'],
                font=('SF Pro Display', 11, 'bold')).pack(anchor='w')
        
        self.imputation_var = tk.StringVar(value="mean")
        
        radio_container = tk.Frame(imputation_frame, bg=self.colors['surface0'])
        radio_container.pack(fill='x', pady=5)
        
        radio_options = [
            ("Media/Moda", "mean"),
            ("Mediana/Moda", "median"), 
            ("KNN (k=5)", "knn"),
            ("Eliminar filas", "drop")
        ]
        
        for i, (text, value) in enumerate(radio_options):
            radio = tk.Radiobutton(radio_container, text=text, variable=self.imputation_var, 
                                  value=value, bg=self.colors['surface0'], fg=self.colors['text'],
                                  selectcolor=self.colors['blue'], activebackground=self.colors['surface0'],
                                  font=('SF Pro Display', 10))
            radio.pack(side='left', padx=(0, 20))
        
        # Botones de acción
        action_frame = tk.Frame(config_card, bg=self.colors['surface0'])
        action_frame.pack(fill='x', padx=20, pady=(0, 20))
        
        button_container = tk.Frame(action_frame, bg=self.colors['surface0'])
        button_container.pack(fill='x')
        
        self.preprocess_button = tk.Button(button_container, text="🔄 Preprocesar Datos", 
                                          command=self.start_preprocessing, state='disabled',
                                          bg=self.colors['teal'], fg=self.colors['base'],
                                          font=('SF Pro Display', 11, 'bold'),
                                          relief='flat', bd=0, padx=20, pady=12,
                                          cursor='hand2')
        self.preprocess_button.pack(side='left', padx=(0, 15))
        
        self.preprocess_info_label = tk.Label(button_container, text="",
                                             bg=self.colors['surface0'], fg=self.colors['subtext0'],
                                             font=('SF Pro Display', 10))
        self.preprocess_info_label.pack(side='left')
        
        self.train_button = tk.Button(button_container, text="🚀 Entrenar Modelos", 
                                     command=self.start_training, state='disabled',
                                     bg=self.colors['green'], fg=self.colors['base'],
                                     font=('SF Pro Display', 11, 'bold'),
                                     relief='flat', bd=0, padx=20, pady=12,
                                     cursor='hand2')
        self.train_button.pack(side='right')
        
        # Sección 3: Progreso
        progress_card = self.create_card_frame(scrollable_frame)
        progress_card.pack(fill='x', pady=(0, 15))
        
        progress_header = tk.Frame(progress_card, bg=self.colors['surface0'])
        progress_header.pack(fill='x', padx=20, pady=15)
        
        tk.Label(progress_header, text="📈 Progreso del Entrenamiento", 
                bg=self.colors['surface0'], fg=self.colors['peach'],
                font=('SF Pro Display', 14, 'bold')).pack(anchor='w')
        
        progress_content = tk.Frame(progress_card, bg=self.colors['surface0'])
        progress_content.pack(fill='x', padx=20, pady=(0, 20))
        
        self.progress_var = tk.StringVar(value="Esperando inicio de entrenamiento...")
        self.progress_label = tk.Label(progress_content, textvariable=self.progress_var,
                                      bg=self.colors['surface0'], fg=self.colors['text'],
                                      font=('SF Pro Display', 10))
        self.progress_label.pack(anchor='w', pady=(0, 10))
        
        # Progress bar personalizada
        progress_bg = tk.Frame(progress_content, bg=self.colors['surface1'], height=8)
        progress_bg.pack(fill='x', pady=5)
        
        self.progress_fill = tk.Frame(progress_bg, bg=self.colors['blue'], height=8)
        self.progress_fill.place(x=0, y=0, width=0, height=8)
        
        # Sección 4: Resultados
        results_card = self.create_card_frame(scrollable_frame)
        results_card.pack(fill='both', expand=True, pady=(0, 0))
        
        results_header = tk.Frame(results_card, bg=self.colors['surface0'])
        results_header.pack(fill='x', padx=20, pady=15)
        
        tk.Label(results_header, text="📊 Resultados de Evaluación", 
                bg=self.colors['surface0'], fg=self.colors['mauve'],
                font=('SF Pro Display', 14, 'bold')).pack(anchor='w')
        
        # Notebook para resultados
        notebook_frame = tk.Frame(results_card, bg=self.colors['surface0'])
        notebook_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))
        
        self.results_notebook = ttk.Notebook(notebook_frame)
        self.results_notebook.pack(fill='both', expand=True)
        
        # Configurar el canvas
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Configurar scroll con rueda del mouse
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind("<MouseWheel>", on_mousewheel)
    
    def load_data(self):
        """Función para cargar y analizar el dataset de diabetes incluyendo valores faltantes"""
        file_path = filedialog.askopenfilename(
            title="Seleccionar Dataset CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                
                # Verificación de columnas requeridas
                required_columns = ['Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 
                                  'TG', 'HDL', 'LDL', 'VLDL', 'BMI', 'CLASS']
                
                if not all(col in self.data.columns for col in required_columns):
                    messagebox.showerror("Error", "El dataset no contiene todas las columnas requeridas")
                    return
                
                # Análisis de valores faltantes
                self.missing_info = self.analyze_missing_values()
                
                # Información del dataset
                total_missing = sum(self.missing_info.values())
                missing_percentage = (total_missing / (len(self.data) * len(required_columns))) * 100
                
                info_text = f"Dataset: {len(self.data)} registros, {len(self.data.columns)} columnas\nValores faltantes: {total_missing} ({missing_percentage:.1f}%)"
                self.data_info_label.config(text=info_text, fg=self.colors['green'])
                self.preprocess_button.config(state='normal', bg=self.colors['teal'])
                self.preprocessing_done = False
                self.train_button.config(state='disabled', bg=self.colors['surface1'])
                
                # Mensaje detallado sobre valores faltantes
                missing_msg = "Dataset cargado correctamente\n\n"
                if total_missing > 0:
                    missing_msg += "VALORES FALTANTES DETECTADOS:\n"
                    for col, count in self.missing_info.items():
                        if count > 0:
                            percentage = (count / len(self.data)) * 100
                            missing_msg += f"• {col}: {count} valores ({percentage:.1f}%)\n"
                    missing_msg += f"\nTotal: {total_missing} valores faltantes"
                else:
                    missing_msg += "No se detectaron valores faltantes en el dataset."
                
                messagebox.showinfo("Análisis del Dataset", missing_msg)
                
            except Exception as e:
                messagebox.showerror("Error", f"Error al cargar el archivo: {str(e)}")
    
    def analyze_missing_values(self):
        """Análisis detallado de valores faltantes por columna"""
        missing_info = {}
        feature_columns = ['Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 
                          'TG', 'HDL', 'LDL', 'VLDL', 'BMI', 'CLASS']
        
        for col in feature_columns:
            if col in self.data.columns:
                missing_info[col] = self.data[col].isnull().sum()
        
        return missing_info
    
    def validate_dataset(self, df):
        """Validación exhaustiva del dataset antes del preprocesamiento"""
        validation_report = []
        
        # Verificar estructura básica
        if df.empty:
            validation_report.append("ERROR: Dataset vacío")
            return validation_report
        
        required_columns = ['Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 
                           'TG', 'HDL', 'LDL', 'VLDL', 'BMI', 'CLASS']
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_report.append(f"ERROR: Columnas faltantes: {missing_cols}")
        
        # Verificar tipos de datos y valores problemáticos
        for col in required_columns:
            if col not in df.columns:
                continue
                
            total_values = len(df)
            non_null_values = df[col].notna().sum()
            null_percentage = (total_values - non_null_values) / total_values * 100
            
            validation_report.append(f"{col}: {non_null_values}/{total_values} valores válidos ({null_percentage:.1f}% NaN)")
            
            # Verificaciones específicas por columna
            if col == 'Gender':
                # Limpiar espacios para análisis
                gender_clean = df[col].astype(str).str.strip()
                unique_genders = gender_clean.dropna().unique()
                validation_report.append(f"  Géneros únicos: {list(unique_genders)}")
                
                # Detectar espacios problemáticos
                has_spaces = df[col].astype(str).str.contains(' ', na=False).any()
                if has_spaces:
                    validation_report.append(f"  ADVERTENCIA: Espacios detectados en Gender")
                
            elif col == 'CLASS':
                # Limpiar espacios para análisis
                class_clean = df[col].astype(str).str.strip().str.upper()
                unique_classes = class_clean.dropna().unique()
                validation_report.append(f"  Clases únicas: {list(unique_classes)}")
                
                # Detectar espacios problemáticos
                has_spaces = df[col].astype(str).str.contains(' ', na=False).any()
                if has_spaces:
                    validation_report.append(f"  ADVERTENCIA: Espacios detectados en CLASS")
                
                # Mostrar problemas específicos
                original_classes = df[col].dropna().unique()
                for orig_class in original_classes:
                    clean_class = str(orig_class).strip().upper()
                    if str(orig_class) != clean_class:
                        validation_report.append(f"  PROBLEMA: '{orig_class}' se limpiará a '{clean_class}'")
                
                class_counts = df[col].value_counts()
                validation_report.append(f"  Distribución: {dict(class_counts)}")
                
            elif col in ['AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']:
                # Verificar valores numéricos
                numeric_vals = pd.to_numeric(df[col], errors='coerce')
                non_numeric = df[col].notna() & numeric_vals.isna()
                if non_numeric.any():
                    validation_report.append(f"  ADVERTENCIA: {non_numeric.sum()} valores no numéricos en {col}")
                    # Mostrar ejemplos de valores problemáticos
                    problem_values = df[col][non_numeric].unique()[:3]  # Primeros 3 ejemplos
                    validation_report.append(f"  Ejemplos problemáticos: {list(problem_values)}")
                
                if numeric_vals.notna().any():
                    min_val = numeric_vals.min()
                    max_val = numeric_vals.max()
                    validation_report.append(f"  Rango: {min_val:.2f} - {max_val:.2f}")
                    
                    # Detectar valores atípicos extremos
                    if col == 'AGE' and (min_val < 0 or max_val > 150):
                        validation_report.append(f"  ADVERTENCIA: Edades atípicas detectadas")
                    elif col == 'BMI' and (min_val < 10 or max_val > 70):
                        validation_report.append(f"  ADVERTENCIA: BMI atípico detectado")
        
        return validation_report
    
    def start_preprocessing(self):
        """Inicio del proceso de preprocesamiento con validación previa"""
        if self.data is None:
            messagebox.showerror("Error", "Debe cargar un dataset primero")
            return
        
        # Validación del dataset antes del preprocesamiento
        validation_report = self.validate_dataset(self.data)
        
        # Mostrar reporte de validación
        validation_text = "VALIDACIÓN DEL DATASET:\n\n" + "\n".join(validation_report)
        
        # Verificar si hay errores críticos
        has_errors = any("ERROR:" in line for line in validation_report)
        if has_errors:
            messagebox.showerror("Errores en Dataset", validation_text)
            return
        
        # Mostrar advertencias si las hay
        has_warnings = any("ADVERTENCIA:" in line for line in validation_report)
        if has_warnings:
            response = messagebox.askyesno(
                "Advertencias Detectadas", 
                validation_text + "\n\n¿Desea continuar con el preprocesamiento?"
            )
            if not response:
                return
        
        # Ejecutar preprocesamiento en hilo separado
        preprocessing_thread = threading.Thread(target=self.run_preprocessing)
        preprocessing_thread.daemon = True
        preprocessing_thread.start()
    
    def handle_missing_values(self, df):
        """Manejo robusto de valores faltantes según la estrategia seleccionada"""
        strategy = self.imputation_var.get()
        
        # Verificar columnas requeridas
        required_columns = ['Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI', 'CLASS']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise Exception(f"Columnas faltantes en el dataset: {missing_cols}")
        
        # Trabajar solo con columnas requeridas
        df_work = df[required_columns].copy()
        
        # Verificar que CLASS no tenga valores faltantes
        if df_work['CLASS'].isnull().any():
            nan_count = df_work['CLASS'].isnull().sum()
            raise Exception(f"La variable objetivo CLASS contiene {nan_count} valores faltantes. "
                          "No se puede proceder sin etiquetas válidas.")
        
        if strategy == "drop":
            # Eliminar filas con valores faltantes (excepto CLASS que ya verificamos)
            df_clean = df_work.dropna()
            if len(df_clean) < len(df_work) * 0.5:  # Si se pierden más del 50% de datos
                response = messagebox.askyesno(
                    "Advertencia", 
                    f"Eliminar filas con NaN resultará en pérdida de {len(df_work) - len(df_clean)} registros "
                    f"({((len(df_work) - len(df_clean)) / len(df_work)) * 100:.1f}% del dataset).\n"
                    "¿Desea continuar?"
                )
                if not response:
                    raise Exception("Operación cancelada por el usuario")
            return df_clean
        
        else:
            # Imputación de valores faltantes
            df_clean = df_work.copy()
            
            # Separar columnas por tipo
            numeric_columns = ['AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']
            categorical_columns = ['Gender']
            
            # Verificar que las columnas numéricas sean realmente numéricas
            for col in numeric_columns:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            if strategy == "knn":
                # Validar que tengamos suficientes datos para KNN
                total_features = len(numeric_columns) + len(categorical_columns)
                if df_clean.dropna().shape[0] < 5:  # Necesitamos al menos 5 registros completos para KNN
                    raise Exception("Datos insuficientes para imputación KNN. Use otra estrategia o elimine filas.")
                
                # Imputación KNN
                self.imputer = KNNImputer(n_neighbors=min(5, len(df_clean.dropna())))
                
                # Manejar Gender para KNN
                le_temp = LabelEncoder()
                gender_backup = df_clean['Gender'].copy()
                
                # Imputar Gender con moda si hay NaN
                if df_clean['Gender'].isnull().any():
                    gender_mode = df_clean['Gender'].mode()
                    mode_value = gender_mode[0] if len(gender_mode) > 0 else 'M'  # Default a 'M'
                    df_clean['Gender'].fillna(mode_value, inplace=True)
                
                # Codificar Gender
                df_clean['Gender_encoded'] = le_temp.fit_transform(df_clean['Gender'])
                
                # Aplicar KNN a todas las características
                feature_cols = numeric_columns + ['Gender_encoded']
                available_cols = [col for col in feature_cols if col in df_clean.columns]
                
                try:
                    imputed_data = self.imputer.fit_transform(df_clean[available_cols])
                    df_clean[available_cols] = imputed_data
                    
                    # Revertir codificación de Gender
                    df_clean['Gender'] = le_temp.inverse_transform(df_clean['Gender_encoded'].round().astype(int))
                    df_clean.drop('Gender_encoded', axis=1, inplace=True)
                    
                except Exception as e:
                    raise Exception(f"Error en imputación KNN: {str(e)}. Intente con otra estrategia.")
                
            else:
                # Imputación simple (media/mediana para numéricas, moda para categóricas)
                strategy_num = 'mean' if strategy == 'mean' else 'median'
                
                # Imputar variables numéricas
                numeric_cols_present = [col for col in numeric_columns if col in df_clean.columns]
                if numeric_cols_present and df_clean[numeric_cols_present].isnull().any().any():
                    try:
                        imputer_num = SimpleImputer(strategy=strategy_num)
                        df_clean[numeric_cols_present] = imputer_num.fit_transform(df_clean[numeric_cols_present])
                    except Exception as e:
                        # Si falla la imputación numérica, usar mediana como respaldo
                        imputer_backup = SimpleImputer(strategy='median')
                        df_clean[numeric_cols_present] = imputer_backup.fit_transform(df_clean[numeric_cols_present])
                
                # Imputar variables categóricas con moda
                for col in categorical_columns:
                    if col in df_clean.columns and df_clean[col].isnull().any():
                        mode_values = df_clean[col].mode()
                        mode_value = mode_values[0] if len(mode_values) > 0 else 'M'  # Default
                        df_clean[col].fillna(mode_value, inplace=True)
            
            # Verificación final: asegurar que no queden NaN
            remaining_nan = df_clean.isnull().sum().sum()
            if remaining_nan > 0:
                nan_cols = df_clean.columns[df_clean.isnull().any()].tolist()
                raise Exception(f"Aún quedan {remaining_nan} valores NaN en columnas: {nan_cols}. "
                              "Intente con estrategia 'Eliminar filas'.")
            
            return df_clean
    
    def preprocess_data(self):
        """Preprocesamiento robusto de datos con validaciones exhaustivas"""
        try:
            # Manejo de valores faltantes
            self.progress_var.set("🔄 Manejando valores faltantes...")
            self.update_progress_bar(20)
            self.root.update()
            
            self.data_clean = self.handle_missing_values(self.data)
            
            # Verificar que aún tengamos suficientes datos
            if len(self.data_clean) < 50:
                raise Exception(f"Dataset insuficiente después del manejo de valores faltantes: {len(self.data_clean)} registros")
            
            self.progress_var.set("🧹 Limpiando y codificando variables...")
            self.update_progress_bar(40)
            self.root.update()
            
            # Copia de datos para procesamiento
            df = self.data_clean.copy()
            
            # Limpieza de espacios en blanco en columnas de texto
            text_columns = ['Gender', 'CLASS']
            for col in text_columns:
                if col in df.columns:
                    # Convertir a string y limpiar espacios
                    df[col] = df[col].astype(str).str.strip()
                    # Reemplazar 'nan' string con NaN real
                    df[col] = df[col].replace('nan', np.nan)
            
            # Verificar tipos de datos
            expected_types = {
                'AGE': 'numeric', 'Urea': 'numeric', 'Cr': 'numeric', 'HbA1c': 'numeric',
                'Chol': 'numeric', 'TG': 'numeric', 'HDL': 'numeric', 'LDL': 'numeric',
                'VLDL': 'numeric', 'BMI': 'numeric'
            }
            
            for col, expected_type in expected_types.items():
                if col in df.columns:
                    if expected_type == 'numeric':
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        # Verificar si la conversión introdujo NaN
                        if df[col].isnull().any():
                            raise Exception(f"Error en conversión numérica de columna {col}. Verifique los datos.")
            
            # Codificación de variable categórica Gender
            if 'Gender' not in df.columns:
                raise Exception("Columna 'Gender' no encontrada")
            
            # Limpieza adicional para Gender
            valid_genders = df['Gender'].dropna().unique()
            if len(valid_genders) == 0:
                raise Exception("No hay valores válidos en la columna Gender")
            
            # Normalizar valores de Gender comunes
            gender_mapping = {
                'M': 'M', 'm': 'M', 'Male': 'M', 'MALE': 'M', 'male': 'M',
                'F': 'F', 'f': 'F', 'Female': 'F', 'FEMALE': 'F', 'female': 'F'
            }
            
            df['Gender'] = df['Gender'].map(gender_mapping).fillna(df['Gender'])
            
            # Verificar que solo tengamos M/F
            final_genders = df['Gender'].dropna().unique()
            invalid_genders = set(final_genders) - {'M', 'F'}
            if invalid_genders:
                raise Exception(f"Valores inválidos en Gender: {invalid_genders}. Solo se permiten M/F")
            
            le_gender = LabelEncoder()
            try:
                df['Gender'] = le_gender.fit_transform(df['Gender'])
            except Exception as e:
                raise Exception(f"Error en codificación de Gender: {str(e)}")
            
            self.progress_var.set("🎯 Preparando variables objetivo...")
            self.update_progress_bar(60)
            self.root.update()
            
            # Verificar variable objetivo CLASS
            if 'CLASS' not in df.columns:
                raise Exception("Columna 'CLASS' no encontrada")
            
            # Limpieza y normalización de CLASS
            df['CLASS'] = df['CLASS'].str.upper()  # Convertir a mayúsculas
            
            valid_classes = df['CLASS'].dropna().unique()
            if len(valid_classes) == 0:
                raise Exception("No hay valores válidos en la columna CLASS")
            
            # Preparación de variables objetivo según configuración
            if self.binary_var.get():
                # Clasificación binaria: N=0 (No diabetes), P,Y=1 (Diabetes)
                class_mapping = {'N': 0, 'P': 1, 'Y': 1}
                unknown_classes = set(valid_classes) - set(class_mapping.keys())
                if unknown_classes:
                    # Mostrar clases problemáticas con sus representaciones
                    problem_classes = []
                    for cls in unknown_classes:
                        problem_classes.append(f"'{cls}' (longitud: {len(cls)}, ASCII: {[ord(c) for c in cls]})")
                    
                    raise Exception(f"Clases desconocidas en CLASS: {unknown_classes}. "
                                  f"Esperadas: {list(class_mapping.keys())}. "
                                  f"Detalles: {problem_classes}")
                
                df['CLASS'] = df['CLASS'].map(class_mapping)
                self.class_names = ['No Diabetes', 'Diabetes']
            else:
                # Clasificación multiclase
                # Verificar que tengamos clases válidas
                expected_multiclass = {'N', 'P', 'Y'}
                unknown_classes = set(valid_classes) - expected_multiclass
                if unknown_classes:
                    raise Exception(f"Clases desconocidas en CLASS: {unknown_classes}. "
                                  f"Esperadas: {list(expected_multiclass)}")
                
                le_class = LabelEncoder()
                df['CLASS'] = le_class.fit_transform(df['CLASS'])
                self.class_names = le_class.classes_
            
            # Verificar que no hay NaN después de la codificación
            if df['CLASS'].isnull().any():
                raise Exception("Valores faltantes en CLASS después de la codificación")
            
            self.progress_var.set("📊 Dividiendo conjunto de datos...")
            self.update_progress_bar(80)
            self.root.update()
            
            # Separación de características y variable objetivo
            feature_columns = ['Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 
                              'TG', 'HDL', 'LDL', 'VLDL', 'BMI']
            
            # Verificar que todas las columnas de características existan
            missing_features = [col for col in feature_columns if col not in df.columns]
            if missing_features:
                raise Exception(f"Características faltantes: {missing_features}")
            
            X = df[feature_columns]
            y = df['CLASS']
            
            # Verificar que no hay NaN en las características finales
            if X.isnull().any().any():
                nan_features = X.columns[X.isnull().any()].tolist()
                raise Exception(f"Valores NaN en características finales: {nan_features}")
            
            # Verificar distribución de clases para estratificación
            unique_classes, class_counts = np.unique(y, return_counts=True)
            min_class_count = min(class_counts)
            if min_class_count < 2:
                raise Exception(f"Clase con muy pocas muestras para división estratificada: {min_class_count}")
            
            # División 80/20 para entrenamiento y prueba
            try:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            except Exception as e:
                raise Exception(f"Error en división de datos: {str(e)}")
            
            # Normalización de características numéricas
            try:
                self.X_train_scaled = self.scaler.fit_transform(self.X_train)
                self.X_test_scaled = self.scaler.transform(self.X_test)
            except Exception as e:
                raise Exception(f"Error en normalización: {str(e)}")
            
            # Verificaciones finales
            if np.isnan(self.X_train_scaled).any():
                raise Exception("NaN detectados en datos de entrenamiento normalizados")
            if np.isnan(self.X_test_scaled).any():
                raise Exception("NaN detectados en datos de prueba normalizados")
            
        except Exception as e:
            # Re-lanzar con contexto adicional
            raise Exception(f"Error en preprocesamiento: {str(e)}")
    
    def run_preprocessing(self):
        """Proceso de preprocesamiento de datos con manejo robusto de errores"""
        try:
            self.progress_var.set("Iniciando preprocesamiento...")
            self.update_progress_bar(10)
            self.root.update()
            
            # Ejecutar preprocesamiento
            self.preprocess_data()
            
            # Actualizar información de preprocesamiento
            self.update_progress_bar(100)
            self.progress_var.set("✅ Preprocesamiento completado")
            
            # Mostrar resultados del preprocesamiento
            original_size = len(self.data)
            final_size = len(self.data_clean)
            loss_percentage = ((original_size - final_size) / original_size) * 100 if original_size != final_size else 0
            
            preprocess_info = f"✅ Procesado: {final_size} registros"
            if original_size != final_size:
                preprocess_info += f" (perdidos: {original_size - final_size})"
            
            self.preprocess_info_label.config(text=preprocess_info, fg=self.colors['green'])
            
            # Habilitar botón de entrenamiento
            self.preprocessing_done = True
            self.train_button.config(state='normal', bg=self.colors['green'])
            self.preprocess_button.config(state='disabled', bg=self.colors['surface1'])
            
            # Mostrar resumen detallado
            summary_msg = f"PREPROCESAMIENTO COMPLETADO EXITOSAMENTE\n\n"
            summary_msg += f"Estrategia utilizada: {self._get_strategy_name()}\n"
            summary_msg += f"Registros originales: {original_size}\n"
            summary_msg += f"Registros finales: {final_size}\n"
            
            if original_size != final_size:
                summary_msg += f"Registros eliminados: {original_size - final_size} ({loss_percentage:.1f}%)\n"
            
            summary_msg += f"\nConjuntos creados:\n"
            summary_msg += f"• Entrenamiento: {len(self.y_train)} muestras\n"
            summary_msg += f"• Prueba: {len(self.y_test)} muestras\n"
            summary_msg += f"\nDistribución de clases en entrenamiento:\n"
            
            # Mostrar distribución de clases
            unique, counts = np.unique(self.y_train, return_counts=True)
            for i, (class_val, count) in enumerate(zip(unique, counts)):
                class_name = self.class_names[class_val] if hasattr(self, 'class_names') else f"Clase {class_val}"
                percentage = (count / len(self.y_train)) * 100
                summary_msg += f"• {class_name}: {count} ({percentage:.1f}%)\n"
            
            summary_msg += f"\nCalidad de datos verificada:\n"
            summary_msg += f"• Sin valores NaN restantes\n"
            summary_msg += f"• Variables codificadas correctamente\n"
            summary_msg += f"• Datos normalizados listos\n"
            summary_msg += f"\nEl sistema está listo para el entrenamiento."
            
            messagebox.showinfo("Preprocesamiento Completado", summary_msg)
            
        except Exception as e:
            error_msg = str(e)
            
            # Proporcionar sugerencias específicas basadas en el error
            suggestions = "\n\nSUGERENCIAS:\n"
            if "NaN" in error_msg or "contains NaN" in error_msg:
                suggestions += "• Intente con estrategia 'Eliminar filas'\n"
                suggestions += "• Verifique que todas las columnas numéricas contengan valores válidos\n"
                suggestions += "• Revise si hay celdas vacías o texto en columnas numéricas"
            elif "desconocidas en CLASS" in error_msg or "Clases desconocidas" in error_msg:
                suggestions += "• PROBLEMA DETECTADO: Espacios extra en valores CLASS\n"
                suggestions += "• El sistema ahora limpia automáticamente los espacios\n"
                suggestions += "• Verifique que CLASS contenga solo: N, P, Y (sin espacios)\n"
                suggestions += "• Revise el archivo CSV original para caracteres invisibles"
            elif "Gender" in error_msg:
                suggestions += "• Verifique que la columna Gender contenga valores válidos (M/F)\n"
                suggestions += "• Asegúrese de que no haya espacios extra o caracteres especiales\n"
                suggestions += "• El sistema acepta: M, F, Male, Female (en cualquier caso)"
            elif "CLASS" in error_msg:
                suggestions += "• Verifique que la columna CLASS contenga valores válidos (N/P/Y)\n"
                suggestions += "• Asegúrese de que cada registro tenga una etiqueta de clase\n"
                suggestions += "• Revise si hay espacios extra antes o después de los valores"
            elif "insuficiente" in error_msg:
                suggestions += "• Use una estrategia de imputación en lugar de eliminar filas\n"
                suggestions += "• Verifique que el dataset tenga suficientes registros válidos"
            else:
                suggestions += "• Verifique la calidad de los datos de entrada\n"
                suggestions += "• Intente con una estrategia diferente de manejo de valores faltantes\n"
                suggestions += "• Revise que no haya espacios extra en columnas de texto"
            
            full_error_msg = f"Error durante el preprocesamiento:\n\n{error_msg}{suggestions}"
            messagebox.showerror("Error en Preprocesamiento", full_error_msg)
            
            self.progress_var.set("❌ Error en preprocesamiento")
            self.preprocess_button.config(state='normal', bg=self.colors['teal'])
            self.preprocessing_done = False
    
    def start_training(self):
        """Inicio del proceso de entrenamiento en hilo separado"""
        if self.data is None:
            messagebox.showerror("Error", "Debe cargar un dataset primero")
            return
        
        if not self.preprocessing_done:
            messagebox.showerror("Error", "Debe preprocesar los datos antes del entrenamiento")
            return
        
        # Ejecutar entrenamiento en hilo separado para mantener UI responsiva
        training_thread = threading.Thread(target=self.run_training)
        training_thread.daemon = True
        training_thread.start()
    
    def train_model(self, model_name, model_config):
        """Entrenamiento de modelo individual con Grid Search y validación cruzada"""
        self.progress_var.set(f"🤖 Entrenando {model_name}...")
        self.root.update()
        
        # Grid Search con validación cruzada k=5
        grid_search = GridSearchCV(
            model_config['model'],
            model_config['params'],
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        # Entrenamiento con datos normalizados para Logistic Regression
        if model_name == 'Logistic Regression':
            grid_search.fit(self.X_train_scaled, self.y_train)
            y_pred = grid_search.predict(self.X_test_scaled)
        else:
            grid_search.fit(self.X_train, self.y_train)
            y_pred = grid_search.predict(self.X_test)
        
        # Cálculo de métricas de evaluación
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
        
        # Validación cruzada para robustez del modelo
        if model_name == 'Logistic Regression':
            cv_scores = cross_val_score(grid_search.best_estimator_, 
                                      self.X_train_scaled, self.y_train, cv=5)
        else:
            cv_scores = cross_val_score(grid_search.best_estimator_, 
                                      self.X_train, self.y_train, cv=5)
        
        return {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred
        }
    
    def run_training(self):
        """Proceso principal de entrenamiento y evaluación"""
        try:
            # Verificar que el preprocesamiento esté completo
            if not self.preprocessing_done:
                messagebox.showerror("Error", "Los datos no han sido preprocesados")
                return
            
            self.progress_var.set("🚀 Iniciando entrenamiento de modelos...")
            self.update_progress_bar(0)
            self.root.update()
            
            # Entrenamiento de modelos
            total_models = len(self.models_config)
            for i, (model_name, model_config) in enumerate(self.models_config.items()):
                
                # Actualización de progreso
                progress = (i * 80 / total_models)
                self.update_progress_bar(progress)
                
                # Entrenamiento del modelo
                results = self.train_model(model_name, model_config)
                self.models_results[model_name] = results
                
                self.root.update()
            
            # Visualización de resultados
            self.progress_var.set("📊 Generando resultados...")
            self.update_progress_bar(90)
            self.root.update()
            
            self.display_results()
            
            self.progress_var.set("✅ Entrenamiento completado exitosamente")
            self.update_progress_bar(100)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error durante el entrenamiento: {str(e)}")
            self.progress_var.set("❌ Error en entrenamiento")
    
    def display_results(self):
        """Visualización de resultados de evaluación en interface"""
        # Limpiar pestañas anteriores
        for tab in self.results_notebook.tabs():
            self.results_notebook.forget(tab)
        
        # Crear pestaña para cada modelo
        for model_name, results in self.models_results.items():
            
            # Frame para resultados del modelo
            tab_frame = tk.Frame(self.results_notebook, bg=self.colors['base'])
            self.results_notebook.add(tab_frame, text=model_name)
            
            # Scroll para el contenido
            canvas = tk.Canvas(tab_frame, bg=self.colors['base'], highlightthickness=0)
            scrollbar = ttk.Scrollbar(tab_frame, orient="vertical", command=canvas.yview)
            scrollable_content = tk.Frame(canvas, bg=self.colors['base'])
            
            scrollable_content.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_content, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Contenido del modelo
            model_card = self.create_card_frame(scrollable_content)
            model_card.pack(fill='both', expand=True, padx=10, pady=10)
            
            # Header del modelo
            header = tk.Frame(model_card, bg=self.colors['surface0'])
            header.pack(fill='x', padx=20, pady=15)
            
            model_icons = {
                'Decision Tree': '🌳',
                'Logistic Regression': '📈', 
                'Random Forest': '🌲'
            }
            
            icon = model_icons.get(model_name, '🤖')
            tk.Label(header, text=f"{icon} {model_name}", 
                    bg=self.colors['surface0'], fg=self.colors['blue'],
                    font=('SF Pro Display', 16, 'bold')).pack(anchor='w')
            
            # Métricas principales
            metrics_frame = tk.Frame(model_card, bg=self.colors['surface0'])
            metrics_frame.pack(fill='x', padx=20, pady=10)
            
            metrics = [
                ('Accuracy', results['accuracy'], self.colors['green']),
                ('Precision', results['precision'], self.colors['blue']),
                ('Recall', results['recall'], self.colors['yellow']),
                ('F1-Score', results['f1_score'], self.colors['mauve'])
            ]
            
            metrics_grid = tk.Frame(metrics_frame, bg=self.colors['surface0'])
            metrics_grid.pack(fill='x')
            
            for i, (metric, value, color) in enumerate(metrics):
                metric_card = tk.Frame(metrics_grid, bg=self.colors['surface1'], relief='flat', bd=1)
                metric_card.grid(row=0, column=i, padx=5, pady=5, sticky='ew')
                
                tk.Label(metric_card, text=metric, 
                        bg=self.colors['surface1'], fg=self.colors['subtext1'],
                        font=('SF Pro Display', 10)).pack(pady=(10, 5))
                
                tk.Label(metric_card, text=f"{value:.4f}", 
                        bg=self.colors['surface1'], fg=color,
                        font=('SF Pro Display', 14, 'bold')).pack(pady=(0, 10))
                
                metrics_grid.columnconfigure(i, weight=1)
            
            # Información detallada
            details_text = f"""
HIPERPARÁMETROS OPTIMIZADOS:
{self._format_params_modern(results['best_params'])}

VALIDACIÓN CRUZADA (k=5):
• Media: {results['cv_mean']:.4f}
• Desviación Estándar: {results['cv_std']:.4f}

CONFIGURACIÓN DEL EXPERIMENTO:
• Conjunto de entrenamiento: {len(self.y_train)} muestras
• Conjunto de prueba: {len(self.y_test)} muestras
• Estrategia de validación: Validación cruzada estratificada
• Optimización: Grid Search exhaustivo
            """
            
            # Widget de texto personalizado
            text_frame = tk.Frame(model_card, bg=self.colors['surface0'])
            text_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))
            
            text_widget = tk.Text(text_frame, wrap=tk.WORD, 
                                 bg=self.colors['surface1'], fg=self.colors['text'],
                                 font=('SF Pro Text', 10), relief='flat', bd=0,
                                 padx=15, pady=15, insertbackground=self.colors['text'])
            text_widget.pack(fill='both', expand=True)
            text_widget.insert(tk.END, details_text)
            text_widget.config(state=tk.DISABLED)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # Configurar scroll
            def on_mousewheel(event):
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            canvas.bind("<MouseWheel>", on_mousewheel)
        
        # Pestaña de comparación general
        comparison_frame = tk.Frame(self.results_notebook, bg=self.colors['base'])
        self.results_notebook.add(comparison_frame, text="📊 Comparación")
        
        comparison_card = self.create_card_frame(comparison_frame)
        comparison_card.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Header de comparación
        comp_header = tk.Frame(comparison_card, bg=self.colors['surface0'])
        comp_header.pack(fill='x', padx=20, pady=15)
        
        tk.Label(comp_header, text="📊 Comparación de Modelos", 
                bg=self.colors['surface0'], fg=self.colors['blue'],
                font=('SF Pro Display', 16, 'bold')).pack(anchor='w')
        
        # Contenido de comparación
        comparison_text = self._generate_comparison_modern()
        
        comp_text_frame = tk.Frame(comparison_card, bg=self.colors['surface0'])
        comp_text_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))
        
        comp_widget = tk.Text(comp_text_frame, wrap=tk.WORD,
                             bg=self.colors['surface1'], fg=self.colors['text'],
                             font=('SF Pro Text', 10), relief='flat', bd=0,
                             padx=15, pady=15, insertbackground=self.colors['text'])
        comp_widget.pack(fill='both', expand=True)
        comp_widget.insert(tk.END, comparison_text)
        comp_widget.config(state=tk.DISABLED)
    
    def _format_params_modern(self, params_dict):
        """Formateo moderno de hiperparámetros"""
        formatted = []
        for k, v in params_dict.items():
            formatted.append(f"  • {k}: {v}")
        return '\n'.join(formatted)
    
    def _generate_comparison_modern(self):
        """Generación moderna de tabla comparativa de modelos"""
        comparison = "RANKING DE MODELOS\n"
        comparison += "=" * 50 + "\n\n"
        
        # Crear tabla de comparación
        models_data = []
        for model_name, results in self.models_results.items():
            models_data.append((model_name, results['accuracy'], results))
        
        # Ordenar por accuracy
        models_data.sort(key=lambda x: x[1], reverse=True)
        
        # Mostrar ranking
        for i, (model_name, accuracy, results) in enumerate(models_data, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            comparison += f"{medal} {model_name}\n"
            comparison += f"   Accuracy: {results['accuracy']:.4f} | "
            comparison += f"Precision: {results['precision']:.4f} | "
            comparison += f"Recall: {results['recall']:.4f} | "
            comparison += f"F1: {results['f1_score']:.4f}\n\n"
        
        # Información sobre el experimento
        comparison += "CONFIGURACIÓN DEL EXPERIMENTO\n"
        comparison += "=" * 50 + "\n"
        
        # Información sobre manejo de valores faltantes
        total_missing = sum(self.missing_info.values())
        original_size = len(self.data)
        final_size = len(self.data_clean)
        
        comparison += f"📊 Datos procesados:\n"
        comparison += f"   • Registros originales: {original_size}\n"
        comparison += f"   • Registros finales: {final_size}\n"
        comparison += f"   • Valores faltantes procesados: {total_missing}\n"
        comparison += f"   • Estrategia de imputación: {self._get_strategy_name()}\n"
        if original_size != final_size:
            loss_percentage = ((original_size - final_size) / original_size) * 100
            comparison += f"   • Pérdida de datos: {loss_percentage:.1f}%\n"
        
        comparison += f"\n🎯 Configuración del modelo:\n"
        comparison += f"   • Tipo: {'Binaria' if self.binary_var.get() else 'Multiclase'}\n"
        comparison += f"   • Clases: {', '.join(map(str, self.class_names))}\n"
        comparison += f"   • División: 80% entrenamiento, 20% prueba\n"
        comparison += f"   • Validación: Cruzada estratificada (k=5)\n"
        comparison += f"   • Optimización: Grid Search exhaustivo\n"
        
        comparison += f"\n📈 Resultados del conjunto de prueba:\n"
        comparison += f"   • Muestras evaluadas: {len(self.y_test)}\n"
        
        # Distribución de clases en el conjunto de prueba
        unique, counts = np.unique(self.y_test, return_counts=True)
        for class_val, count in zip(unique, counts):
            class_name = self.class_names[class_val] if hasattr(self, 'class_names') else f"Clase {class_val}"
            percentage = (count / len(self.y_test)) * 100
            comparison += f"   • {class_name}: {count} ({percentage:.1f}%)\n"
        
        return comparison
    
    def _get_strategy_name(self):
        """Obtener nombre descriptivo de la estrategia de imputación"""
        strategy_names = {
            'mean': 'Media/Moda',
            'median': 'Mediana/Moda', 
            'knn': 'KNN (k=5)',
            'drop': 'Eliminación de filas'
        }
        return strategy_names.get(self.imputation_var.get(), 'Desconocida')

def main():
    """Función principal para ejecutar el sistema de clasificación"""
    root = tk.Tk()
    app = DiabetesClassificationSystem(root)
    root.mainloop()

if __name__ == "__main__":
    main()