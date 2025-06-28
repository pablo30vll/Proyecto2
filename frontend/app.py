import gradio as gr
import requests
import json
import pandas as pd
import os
from datetime import datetime

# Configuración de la API
API_BASE_URL = os.getenv("API_BASE_URL", "http://backend:8000")

def check_api_health():
    """Verifica que la API esté funcionando"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_current_week():
    """Obtiene el número de semana actual del año"""
    return datetime.now().isocalendar()[1]

def predict_single_purchase(
    customer_type,
    y_coord,
    x_coord,
    deliveries_per_week,
    brand,
    sub_category,
    segment,
    package,
    size,
    week_num
):
    """Realiza predicción individual"""
    
    # Verificar conexión con API
    if not check_api_health():
        return "❌ Error: No se puede conectar con la API"
    
    # Preparar datos en el formato que espera la API (customer y product separados)
    # JSON plano (los 10 features al nivel raíz)
    request_data = {
        "customer_type": customer_type,
        "Y": float(y_coord),
        "X": float(x_coord),
        "num_deliver_per_week": int(deliveries_per_week),
        "brand": brand,
        "sub_category": sub_category,
        "segment": segment,
        "package": package,
        "size": float(size),
        "week_num": int(week_num)
    }

    
    try:
        # Hacer petición a la API
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=request_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Formatear resultado
            probability = result['probability']
            prediction = result['prediction']
            
            # Generar mensaje de resultado
            if prediction:
                emoji = "✅"
                message = "SÍ COMPRARÁ"
                interpretation = "Este perfil de cliente-producto tiene una alta probabilidad de generar una compra."
                confidence_level = "Alta" if probability > 0.7 else "Media"
            else:
                emoji = "❌"
                message = "NO COMPRARÁ"
                interpretation = "Este perfil de cliente-producto tiene baja probabilidad de generar una compra."
                confidence_level = "Alta" if probability < 0.3 else "Media"
            
            result_text = f"""
{emoji} **PREDICCIÓN: {message}**

📊 **Probabilidad de compra:** {probability:.1%}
🎯 **Nivel de confianza:** {confidence_level}
📅 **Semana predicha:** {week_num}

---

### 👤 **Perfil del Cliente**
- **Tipo:** {customer_type} 
- **Ubicación:** Lat {y_coord:.3f}, Lon {x_coord:.3f}
- **Entregas semanales:** {deliveries_per_week}

### 🥤 **Perfil del Producto**
- **Marca:** {brand}
- **Categoría:** {sub_category}
- **Segmento:** {segment}
- **Empaque:** {package}
- **Tamaño:** {size}L

---

### 💡 **Interpretación**
{interpretation}

### 📈 **Recomendaciones:**
"""
            
            # Agregar recomendaciones basadas en la probabilidad
            if prediction:
                if probability > 0.8:
                    result_text += "• **Acción inmediata:** Contactar al cliente esta semana\n"
                    result_text += "• **Inventario:** Asegurar stock disponible\n" 
                    result_text += "• **Marketing:** Cliente ideal para promociones premium"
                elif probability > 0.6:
                    result_text += "• **Seguimiento:** Agendar visita comercial\n"
                    result_text += "• **Oferta:** Considerar descuentos o promociones\n"
                    result_text += "• **Timing:** Contactar en primeros días de la semana"
                else:
                    result_text += "• **Oportunidad moderada:** Incluir en campañas generales\n"
                    result_text += "• **Seguimiento:** Monitorear comportamiento\n"
                    result_text += "• **Estrategia:** Evaluar productos alternativos"
            else:
                if probability < 0.2:
                    result_text += "• **Baja prioridad:** Enfocar recursos en otros clientes\n"
                    result_text += "• **Análisis:** Revisar historial y preferencias\n"
                    result_text += "• **Alternativa:** Considerar otros productos del catálogo"
                else:
                    result_text += "• **Potencial latente:** Cliente para estrategias a largo plazo\n"
                    result_text += "• **Investigación:** Analizar barreras de compra\n"
                    result_text += "• **Timing:** Reevaluar en semanas siguientes"
            
            return result_text
            
        else:
            error_detail = response.json().get('detail', 'Error desconocido')
            return f"❌ Error en la predicción: {error_detail}"
            
    except requests.exceptions.Timeout:
        return "⏰ Error: Tiempo de espera agotado. Verifica la conexión."
    except requests.exceptions.ConnectionError:
        return "🔌 Error: No se puede conectar con el servidor. Verifica que el backend esté corriendo."
    except Exception as e:
        return f"❌ Error inesperado: {str(e)}"

def predict_bulk_from_file(file):
    if file is None:
        return "❌ Por favor, sube un archivo CSV con las 10 features del modelo"
    if not check_api_health():
        return "❌ Error: No se puede conectar con la API"
    try:
        # Leer archivo CSV
        df = pd.read_csv(file.name)
        required_cols = [
            'customer_type', 'Y', 'X', 'num_deliver_per_week',
            'brand', 'sub_category', 'segment', 'package', 'size', 'week_num'
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return f"❌ Faltan columnas requeridas: {missing_cols}\n\n💡 El CSV debe tener exactamente estas 10 columnas:\n{', '.join(required_cols)}"
        df_sample = df.head(20)

        # Formato correcto: {"data": [ {...}, {...}, ... ]}
        json_data = {"data": df_sample.to_dict(orient="records")}
        response = requests.post(
            f"{API_BASE_URL}/predict/bulk",
            json=json_data,
            timeout=20
        )
        if response.status_code == 200:
            results = response.json()
            predictions = results.get("predictions", [])
            positive_preds = [p for p in predictions if p.get("prediction", False)]
            
            summary = f"""
        # 📈 **RESULTADOS DE PREDICCIÓN MASIVA**
        - **Total de filas:** {len(df_sample)}
        - **Predicciones positivas:** {len(positive_preds)}
        - **Probabilidad máxima:** {max([p['probability'] for p in positive_preds], default=0):.1%}

        ## 🟢 **Solo Predicciones Positivas (probabilidad > 50%)**
        """

            if positive_preds:
                # Ordena por probabilidad descendente
                positive_preds.sort(key=lambda x: x.get("probability", 0), reverse=True)
                for i, pred in enumerate(positive_preds):
                    prob = pred["probability"]
                    week = pred.get("week", "")
                    # Si tu backend devuelve más campos, agrégalos aquí:
                    summary += (
                        f"- Fila {i+1:2d} | Prob: {prob:.1%} | Semana: {week}\n"
                    )
            else:
                summary += "\nNinguna fila con probabilidad > 50%.\n"
            return summary
        else:
            error_detail = response.json().get('detail', 'Error desconocido')
            return f"❌ Error en la predicción masiva: {error_detail}"
    except Exception as e:
        return f"❌ Error procesando archivo: {str(e)}"

def create_sample_csv():
    """Crea un archivo CSV de ejemplo con datos reales del dataset SodAI"""
    sample_data = {
        'customer_type': ['ABARROTES', 'MAYORISTA', 'CANAL FRIO', 'RESTAURANT', 'SUPERMERCADO'],
        'Y': [-46.556, -46.600, -46.520, -46.480, -46.590],  # Valores reales del dataset
        'X': [-107.895, -107.850, -107.920, -107.880, -107.870],  # Valores reales del dataset
        'num_deliver_per_week': [3, 4, 2, 3, 5],  # Valores reales del dataset (1-6)
        'brand': ['Brand 7', 'Brand 3', 'Brand 35', 'Brand 31', 'Brand 1'],  # Marcas más comunes
        'sub_category': ['GASEOSAS', 'JUGOS', 'GASEOSAS', 'AGUAS SABORIZADAS', 'GASEOSAS'],
        'segment': ['PREMIUM', 'MEDIUM', 'HIGH', 'LOW', 'PREMIUM'],
        'package': ['BOTELLA', 'LATA', 'BOTELLA', 'TETRA', 'BOTELLA'],
        'size': [0.5, 0.66, 1.0, 0.5, 2.0],  # Valores reales del dataset
        'week_num': [26, 26, 26, 26, 26]  # Semana actual
    }
    
    df = pd.DataFrame(sample_data)
    return df

# Crear interfaz Gradio con datos reales de SodAI
with gr.Blocks(title="SodAI Drinks - Predictor de Compras 🥤", theme=gr.themes.Soft()) as app:
    
    gr.Markdown("""
    # 🥤 SodAI Drinks - Predictor Inteligente de Compras
    
    ### 🎯 Predicciones basadas en **254,051 transacciones** reales de **1,569 clientes** y **971 productos**
    
    ## 📊 Features del Modelo (Datos Reales del Dataset)
    
    ### 🔢 Variables Numéricas (4):
    - **Y (Latitud):** -109.0 a -46.2 (Mediana: -46.56)
    - **X (Longitud):** -108.6 a -46.4 (Mediana: -107.90)  
    - **Entregas/Semana:** 1 a 6 entregas (Mediana: 3)
    - **Tamaño:** 0.125L a 20L (Mediana: 0.5L)
    
    ### 🏷️ Variables Categóricas (6):
    - **Tipos de Cliente:** 7 categorías (73.9% ABARROTES, 14.2% MAYORISTA)
    - **Marcas:** 61 marcas diferentes en el catálogo
    - **Subcategorías:** GASEOSAS (69.3%), JUGOS (23.6%), AGUAS SABORIZADAS (7.1%)
    - **Segmentos:** PREMIUM, MEDIUM, HIGH, LOW
    - **Empaques:** BOTELLA (61.5%), LATA (30%), TETRA (4.4%), KEG (4.1%)
    - **Semanas:** 1-52 (Datos completos de 2024)
    
    ---
    """)
    
    # Indicador de estado de la API
    with gr.Row():
        with gr.Column():
            api_status = gr.HTML()
        with gr.Column():
            model_info = gr.HTML()
        
        def update_status():
            if check_api_health():
                api_html = "🟢 <b>Estado de la API:</b> Conectado y funcionando correctamente"
            else:
                api_html = "🔴 <b>Estado de la API:</b> Desconectado - Verifica que el backend esté corriendo en puerto 8000"
            
            try:
                response = requests.get(f"{API_BASE_URL}/model/info", timeout=5)
                if response.status_code == 200:
                    model_data = response.json()
                    if model_data.get("model_loaded"):
                        model_type = model_data.get("model_type", "Unknown")
                        model_html = f"🤖 <b>Modelo:</b> {model_type} cargado (10 Features SodAI)"
                    else:
                        model_html = "❌ <b>Modelo:</b> No cargado"
                else:
                    model_html = "❓ <b>Modelo:</b> No se pudo verificar el estado"
            except:
                model_html = "❓ <b>Modelo:</b> No se pudo verificar el estado"
            
            return api_html, model_html
        
        api_status.value, model_info.value = update_status()
    
    with gr.Tab("🎯 Predicción Individual"):
        
        gr.Markdown("### 🔧 Configura el perfil de cliente y producto para obtener una predicción personalizada")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 👤 **Perfil del Cliente**")
                # Usando datos reales del análisis
                customer_type = gr.Dropdown(
                    choices=['ABARROTES', 'MAYORISTA', 'CANAL FRIO', 'RESTAURANT', 'SUPERMERCADO', 'MINIMARKET', 'TIENDA DE CONVENIENCIA'],
                    label="Tipo de Cliente",
                    value="ABARROTES",
                    info="73.9% de clientes son ABARROTES"
                )
                
                gr.Markdown("### 📍 **Ubicación Geográfica** (Coordenadas Reales)")
                with gr.Row():
                    y_coord = gr.Number(
                        label="Y (Latitud)", 
                        value=-46.556, 
                        info="Rango: -109.0 a -46.2 (Mediana dataset: -46.56)"
                    )
                    x_coord = gr.Number(
                        label="X (Longitud)", 
                        value=-107.895, 
                        info="Rango: -108.6 a -46.4 (Mediana dataset: -107.90)"
                    )
                
                deliveries_per_week = gr.Number(
                    label="Entregas por Semana", 
                    value=3, 
                    info="Rango: 1-6 entregas (Mediana: 3)",
                    minimum=1, 
                    maximum=6
                )
            
            with gr.Column():
                gr.Markdown("### 🥤 **Perfil del Producto**")
                
                # Campo de texto para brand debido a las 61 opciones
                brand = gr.Textbox(
                    label="Marca del Producto", 
                    value="Brand 7", 
                    info="61 marcas disponibles. Principales: Brand 7, Brand 3, Brand 35, Brand 31, Brand 1"
                )
                
                sub_category = gr.Dropdown(
                    choices=['GASEOSAS', 'JUGOS', 'AGUAS SABORIZADAS'],
                    label="Subcategoría",
                    value="GASEOSAS",
                    info="GASEOSAS: 69.3% | JUGOS: 23.6% | AGUAS: 7.1%"
                )
                
                segment = gr.Dropdown(
                    choices=['PREMIUM', 'MEDIUM', 'HIGH', 'LOW'],
                    label="Segmento de Mercado",
                    value="PREMIUM",
                    info="Distribución equilibrada en el dataset"
                )
                
                package = gr.Dropdown(
                    choices=['BOTELLA', 'LATA', 'TETRA', 'KEG'],
                    label="Tipo de Empaque",
                    value="BOTELLA",
                    info="BOTELLA: 61.5% | LATA: 30% | TETRA: 4.4% | KEG: 4.1%"
                )
                
                size = gr.Number(
                    label="Tamaño (Litros)", 
                    value=0.5, 
                    info="Rango: 0.125L a 20L (Mediana: 0.5L)",
                    minimum=0.125,
                    maximum=20.0,
                    step=0.125
                )
                
                gr.Markdown("### 📅 **Temporalidad**")
                week_num = gr.Number(
                    label="Semana del Año", 
                    value=get_current_week(), 
                    info=f"Semana actual: {get_current_week()} | Rango válido: 1-52",
                    minimum=1,
                    maximum=52
                )
        
        with gr.Row():
            predict_btn = gr.Button("🎯 Predecir Compra", variant="primary", size="lg")
            clear_btn = gr.Button("🔄 Resetear a Valores por Defecto", variant="secondary")
        
        result_output = gr.Textbox(
            label="📊 Resultado de la Predicción", 
            lines=25, 
            max_lines=30,
            placeholder="Los resultados de la predicción aparecerán aquí..."
        )
        
        # Configurar botones
        predict_btn.click(
            fn=predict_single_purchase,
            inputs=[
                customer_type, y_coord, x_coord, deliveries_per_week,
                brand, sub_category, segment, package, size, week_num
            ],
            outputs=[result_output]
        )
        
        def clear_inputs():
            current_week = get_current_week()
            return ["ABARROTES", -46.556, -107.895, 3, "Brand 7", "GASEOSAS", 
                   "PREMIUM", "BOTELLA", 0.5, current_week, ""]
        
        clear_btn.click(
            fn=clear_inputs,
            outputs=[customer_type, y_coord, x_coord, deliveries_per_week, 
                    brand, sub_category, segment, package, size, week_num, result_output]
        )
    
    with gr.Tab("📊 Predicción Masiva"):
        
        gr.Markdown("""
        ### 📁 Análisis Masivo de Oportunidades de Venta
        
        Sube un archivo CSV con múltiples perfiles de cliente-producto para obtener un análisis completo de oportunidades.
        
        **🔧 Formato requerido (10 columnas exactas):**
        ```
        customer_type, Y, X, num_deliver_per_week, brand, sub_category, segment, package, size, week_num
        ```
        
        **⚠️ Limitación de demo:** Máximo 20 filas procesadas por solicitud
        """)
        
        with gr.Row():
            with gr.Column():
                file_input = gr.File(
                    label="📎 Subir Archivo CSV", 
                    file_types=[".csv"],
                    height=100
                )
                predict_bulk_btn = gr.Button("📊 Analizar Oportunidades", variant="primary", size="lg")
                
                gr.Markdown("""
                **💡 Tips para mejores resultados:**
                - Usa datos geográficos reales (Y: -109 a -46, X: -108 a -46)
                - Incluye variedad de tipos de cliente
                - Considera diferentes tamaños de producto
                - Usa semanas relevantes para tu análisis
                """)
            
            with gr.Column():
                gr.Markdown("### 📥 **Ejemplo con Datos Reales de SodAI**")
                sample_df = create_sample_csv()
                gr.Dataframe(
                    value=sample_df, 
                    label="Estructura correcta con valores del dataset real",
                    height=300
                )
        
        bulk_result = gr.Textbox(
            label="📈 Resultados del Análisis", 
            lines=20,
            placeholder="Los resultados del análisis masivo aparecerán aquí..."
        )
        
        predict_bulk_btn.click(
            fn=predict_bulk_from_file,
            inputs=[file_input],
            outputs=[bulk_result]
        )
    
    with gr.Tab("📊 Información del Dataset"):
        current_week = get_current_week()
        gr.Markdown(f"""
        # 📊 **Información Completa del Dataset SodAI**
        
        ## 🗄️ **Datos del Dataset**
        
        ### 📈 Transacciones
        - **Total de registros:** 254,051 transacciones
        - **Período:** Todo el año 2024 (366 días únicos)
        - **Clientes únicos:** 1,490 clientes activos
        - **Productos únicos:** 114 productos diferentes
        - **Órdenes únicas:** 64,600 órdenes procesadas
        
        ### 👥 Clientes (1,569 registros)
        - **Tipos de cliente:** 7 categorías
          - ABARROTES: 73.9% (1,160 clientes)
          - MAYORISTA: 14.2% (223 clientes) 
          - CANAL FRIO: 5.4% (85 clientes)
          - RESTAURANT: 3.8% (60 clientes)
          - SUPERMERCADO: 1.6% (25 clientes)
          - MINIMARKET: 0.6% (9 clientes)
          - TIENDA DE CONVENIENCIA: 0.4% (7 clientes)
        
        - **Ubicación geográfica:**
          - Latitud (Y): -109.0 a -46.2
          - Longitud (X): -108.6 a -46.4
          - Cobertura: Amplia distribución territorial
        
        - **Logística:**
          - Entregas por semana: 1 a 6 (mediana: 3)
          - Visitas por semana: 1 (uniforme)
        
        ### 🥤 Productos (971 registros)
        - **Marcas:** 61 marcas diferentes
          - Top 5: Brand 7 (10.7%), Brand 3 (10.5%), Brand 35 (8.1%), Brand 31 (6.6%), Brand 1 (5.4%)
        
        - **Categorías de producto:**
          - BEBIDAS CARBONATADAS (categoría principal)
          - Subcategorías: GASEOSAS (69.3%), JUGOS (23.6%), AGUAS SABORIZADAS (7.1%)
        
        - **Segmentación:**
          - PREMIUM: 31.9% (310 productos)
          - MEDIUM: 27.5% (267 productos)
          - HIGH: 23.8% (231 productos)
          - LOW: 16.8% (163 productos)
        
        - **Empaques:**
          - BOTELLA: 61.5% (597 productos)
          - LATA: 30.0% (291 productos)
          - TETRA: 4.4% (43 productos)
          - KEG: 4.1% (40 productos)
        
        - **Tamaños:**
          - Rango: 0.125L a 20L
          - Mediana: 0.5L
          - 13 tamaños únicos disponibles
        
        ---
        
        ## 🤖 **Modelo de Machine Learning**
        
         ### 🎯 Objetivo
        Predecir la probabilidad de que un cliente específico compre un producto específico en una semana determinada.
        
        ### 🔧 Arquitectura
        - **Algoritmo:** Random Forest Classifier
        - **Features:** 10 variables (4 numéricas + 6 categóricas)
        - **Entrenamiento:** Basado en 254,051 transacciones reales
        - **Validación:** Datos de todo el año 2024
        
        ### 📊 Features del Modelo
        
        #### Variables Numéricas (4):
        1. **Y (Latitud):** Coordenada geográfica del cliente
           - Rango: -109.003 a -46.162
           - Mediana: -46.556
           - Outliers: 16.8% (variación geográfica natural)
        
        2. **X (Longitud):** Coordenada geográfica del cliente  
           - Rango: -108.621 a -46.443
           - Mediana: -107.895
           - Outliers: 24.9% (distribución territorial amplia)
        
        3. **num_deliver_per_week:** Frecuencia de entregas
           - Rango: 1 a 6 entregas por semana
           - Mediana: 3 entregas
           - Outliers: 21.0% (clientes con patrones especiales)
        
        4. **size:** Tamaño del producto en litros
           - Rango: 0.125L a 20L
           - Mediana: 0.5L
           - Outliers: 6.0% (productos especiales)
        
        #### Variables Categóricas (6):
        1. **customer_type:** Segmentación de clientes
        2. **brand:** Marca del producto (61 opciones)
        3. **sub_category:** Tipo de bebida (3 opciones)
        4. **segment:** Segmento de mercado (4 opciones)
        5. **package:** Tipo de empaque (4 opciones)  
        6. **week_num:** Semana del año (1-52, para estacionalidad)
        
        ### 📈 Interpretación de Resultados
        - **Probabilidad > 70%:** Oportunidad de alta prioridad (contacto inmediato)
        - **Probabilidad 50-70%:** Oportunidad media (seguimiento comercial)
        - **Probabilidad 30-50%:** Potencial moderado (campañas generales)
        - **Probabilidad < 30%:** Baja prioridad (estrategias a largo plazo)
        
        ### 🎯 Casos de Uso Empresariales
        
        #### 📞 Ventas y Marketing
        - **Priorización de clientes:** Enfoque en oportunidades de alta probabilidad
        - **Campañas dirigidas:** Segmentación inteligente por perfil cliente-producto
        - **Timing óptimo:** Predicciones semanales para contacto oportuno
        - **Cross-selling:** Identificar productos complementarios por cliente
        
        #### 📦 Operaciones y Logística
        - **Planificación de inventario:** Stock basado en demanda predicha
        - **Optimización de rutas:** Entregas priorizadas por probabilidad de venta
        - **Gestión de almacén:** Distribución regional según predicciones
        - **Forecasting:** Proyecciones de demanda por zona geográfica
        
        #### 📊 Análisis Estratégico
        - **Segmentación de mercado:** Identificar patrones de comportamiento
        - **Análisis de producto:** Performance por marca y categoría
        - **Estacionalidad:** Tendencias por semana del año
        - **Penetración de mercado:** Oportunidades en nuevos territorios
        
        ### 🔄 Patrones Temporales Identificados
        
        #### Actividad por Semana (2024)
        - **Semana 1:** 6,109 transacciones (pico año nuevo)
        - **Semana 6:** 5,617 transacciones (alta actividad)
        - **Semana 3:** 5,514 transacciones (estable)
        - **Semana actual ({current_week}):** Predicciones en tiempo real
        
        #### Insights Estacionales
        - **Enero:** Arranque fuerte del año comercial
        - **Primer trimestre:** Mayor volumen de transacciones
        - **Variación semanal:** Fluctuaciones naturales de demanda
        
        ---
        
        ## 🚀 **Guía de Uso Práctica**
        
        ### 🎯 Para Equipos de Ventas
        1. **Predicción Individual:** Evalúa clientes específicos antes de visitas
        2. **Análisis masivo:** Procesa listas de clientes para priorización semanal  
        3. **Seguimiento:** Usa las recomendaciones generadas para acciones concretas
        
        ### 📈 Para Gerencia Comercial
        1. **Dashboard semanal:** Análisis masivo de la cartera de clientes
        2. **KPIs predictivos:** Métricas de conversión esperada por territorio
        3. **Estrategia de territorio:** Expansión basada en datos geográficos
        
        ### 🔧 Para Equipos Técnicos
        1. **API Integration:** Endpoints RESTful para integración con CRM
        2. **Batch Processing:** Procesamiento masivo para análisis periódicos
        3. **Real-time:** Predicciones en tiempo real para sistemas de ventas
        
        ### 📊 Métricas de Rendimiento
        - **Precisión del modelo:** Validado con datos históricos 2024
        - **Cobertura:** 100% de tipos de cliente y productos del catálogo
        - **Actualización:** Modelo entrenado con datos más recientes
        - **Escalabilidad:** Procesamiento eficiente de volúmenes empresariales
        
        ---
        
        ## 🔗 **Integración y API**
        
        ### Endpoints Disponibles
        - `GET /health` - Estado del servicio
        - `POST /predict` - Predicción individual  
        - `POST /predict/bulk` - Predicción masiva
        - `GET /model/info` - Información del modelo cargado
        
        ### Formato de Datos
        ```json
        {{
          "customer_type": "ABARROTES",
          "Y": -46.556,
          "X": -107.895,
          "num_deliver_per_week": 3,
          "brand": "Brand 7",
          "sub_category": "GASEOSAS", 
          "segment": "PREMIUM",
          "package": "BOTELLA",
          "size": 0.5,
          "week_num": 26
        }}
        ```
        
        ### Respuesta del Modelo
        ```json
        {{
          "prediction": true,
          "probability": 0.85,
          "week": 26,
          "confidence": "high"
        }}
        ```
        
        ---
        
        *🤖 Sistema desarrollado con FastAPI + Gradio | Datos reales de SodAI 2024*
        """)

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )