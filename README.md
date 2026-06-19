# 🎵 Spotify Music Success Predictor

Análisis completo de datos musicales de Spotify para identificar los factores que determinan el éxito comercial de una canción, combinando clustering de géneros musicales y modelado predictivo con regresión logística.

---

## 📋 Descripción del proyecto

Este proyecto aplica técnicas de ciencia de datos sobre un dataset de canciones de Spotify para responder a una pregunta de negocio concreta:

> **¿Qué características sonoras determinan que una canción sea un éxito comercial?**

El pipeline completo cubre desde la ingesta y limpieza de datos hasta la exportación de resultados para un dashboard en Power BI, pasando por análisis exploratorio, clustering no supervisado y regresión logística con umbral de clasificación ajustable por criterio de negocio.

---

## 🗂️ Estructura del proyecto

```
├── 01_ingesta_spotify.ipynb          # Carga y primera inspección del dataset
├── 02_limpieza_datos_spotify.ipynb   # Limpieza, deduplicación y optimización de tipos
├── 03_exploratorio.ipynb             # EDA, análisis bivariante, correlaciones y feature engineering
├── 04_clustering.ipynb               # Segmentación KMeans en 7 clusters musicales
├── 05_modelo_predictivo.ipynb        # Regresión logística para predicción de éxito
├── spotify_dataset.csv               # Dataset original
├── spotify_dataset_limpio.parquet    # Tras limpieza
├── spotify_dataset_explorado.parquet # Tras EDA y feature engineering
├── spotify_dataset_7_clus.parquet    # Tras clustering
└── spotify_powerbi.csv               # Export final para dashboard
```

---

## 🔄 Pipeline

```
CSV original
    │
    ▼
01 · Ingesta ──────────► Primera inspección (shape, nulos, tipos)
    │
    ▼
02 · Limpieza ─────────► Eliminación de columnas irrelevantes, nulos, duplicados
                         Optimización de tipos (int8, float32, category)
                         Guardado en Parquet
    │
    ▼
03 · EDA ──────────────► Análisis univariante de audio features
                         Análisis bivariante vs exito_musical
                         Matriz de correlaciones y colinealidad
                         Detección de outliers (método IQR)
                         Feature engineering: n_artistas, energia_bailable
                         Definición de variable objetivo: popularity ≥ 70
    │
    ▼
04 · Clustering ───────► StandardScaler sobre audio features
                         Método del codo + Silhouette → k=7
                         KMeans con 7 clusters
                         Caracterización y naming de clusters
                         Visualización PCA 3D interactiva (Plotly)
    │
    ▼
05 · Modelo ───────────► One-Hot Encoding de cluster_nombre y n_artistas
                         Split 70/30 estratificado sin data leakage
                         Regresión logística con class_weight='balanced'
                         Umbral de clasificación ajustable por decisión de negocio
                         Evaluación: matriz de confusión, AUC-ROC, coeficientes
                         Export CSV para Power BI
```

---

## 📊 Resultados principales

### Variable objetivo

Se define **éxito musical** como `popularity ≥ 70`, lo que supone aproximadamente el **5% del catálogo**. El dataset presenta un desbalanceo severo (clase 0: ~95%, clase 1: ~5%) que se gestiona con `class_weight='balanced'` en el modelo.

### Clustering: 7 segmentos musicales

| Cluster | Nombre | Géneros predominantes | Tasa de éxito |
|---------|--------|-----------------------|---------------|
| 4 | 💃 Pop y Dance | Reggaeton, Latino, Reggae | 7.3% |
| 2 | 🎸 Rock y Metal | Metalcore, Heavy-metal, Grunge | 5.5% |
| 6 | 🪕 Acústicas y Folk | Tango, Honky-tonk, Jazz | 4.3% |
| 3 | 🎤 Conciertos y Directos | Pagode, Sertanejo, Samba | 2.2% |
| 1 | ⚡ Instrumental Eléctrico | Minimal-techno, Detroit-techno | 1.1% |
| 0 | 🎼 Ambient | New-age, Sleep, Classical | 0.7% |
| 5 | 🗣️ Spoken Word | Comedy, Show-tunes | 0.7% |

### Modelo predictivo

**Variables predictoras** (25 en total):
- 12 numéricas: audio features de Spotify (`danceability`, `energy`, `valence`, `acousticness`, `instrumentalness`, `speechiness`, `liveness`, `loudness`, `tempo`, `duration_ms`) + `explicit` + `energia_bailable`
- 6 dummies de `cluster_nombre` (One-Hot Encoding, drop_first=True)
- Variables dummies de `n_artistas` (One-Hot Encoding, drop_first=True)

**Decisión de codificación:** `cluster_nombre` sustituye a `track_genre` (114 géneros → 7 grupos coherentes), evitando la maldición de la dimensionalidad y el orden arbitrario del Label Encoding.

**Evaluación:** Matriz de confusión + Curva ROC + AUC + coeficientes estandarizados

---

## 🎯 Decisión de negocio: umbral de clasificación

El modelo devuelve una **probabilidad** de éxito para cada canción (0 a 1). El umbral es ajustable según el objetivo de negocio:

```python
UMBRAL = 0.5  # ← ajustar según criterio de negocio
y_pred = (y_prob >= UMBRAL).astype(int)
```

| Umbral | Efecto | Cuándo usarlo |
|--------|--------|---------------|
| **Alto (ej. 0.8)** | Menos falsos positivos, más falsos negativos | Cuando la inversión por canción es alta y no queremos malgastarla |
| **Bajo (ej. 0.3)** | Más recall, más falsos positivos | Cuando el coste de perder un éxito real es mayor que el de invertir en un fracaso |

---

## 🛠️ Tecnologías

- **Python 3.11**
- `pandas` · `numpy` — manipulación de datos
- `scikit-learn` — escalado, KMeans, PCA, regresión logística, métricas
- `matplotlib` · `seaborn` — visualización estática
- `plotly` — visualización 3D interactiva
- **Power BI** — dashboard final

---

## ⚙️ Instalación y uso

```bash
# Clonar el repositorio
git clone https://github.com/tu_usuario/spotify-success-predictor.git
cd spotify-success-predictor

# Instalar dependencias
pip install pandas numpy scikit-learn matplotlib seaborn plotly

# Ejecutar los notebooks en orden
jupyter notebook
```

Ejecutar los notebooks en orden numérico: `01` → `02` → `03` → `04` → `05`. Cada notebook lee el archivo Parquet generado por el anterior.

---

## 👥 Autores

Proyecto desarrollado como trabajo de curso de ciencia de datos.
