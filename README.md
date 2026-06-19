# 🎵 Predictor de exito musical en Spotify

Análisis completo de datos musicales de Spotify para identificar los factores que determinan el éxito comercial de una canción, combinando clustering de géneros musicales y modelado predictivo con regresión logística.

---

## Descripción del proyecto

Este proyecto aplica técnicas de ciencia de datos sobre un dataset de canciones de Spotify para responder a una pregunta de negocio concreta:

> **¿Qué características sonoras determinan que una canción sea un éxito comercial?**

El pipeline completo cubre la ingesta y limpieza de datos, el análisis exploratorio y clustering no supervisado así como la regresión logística con umbral de clasificación ajustable por criterio de negocio.

## 📊 Resultados principales

### Variable objetivo

Se define **éxito musical** como `popularity ≥ 70`, lo que supone aproximadamente el **5% del catálogo**. El dataset presenta un desbalanceo severo (clase 0: ~95%, clase 1: ~5%) que se gestiona en el modelo de regresión logística.

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

**Decisión de codificación:** `cluster_nombre` sustituye a `track_genre` (114 géneros → 7 grupos coherentes), evitando la maldición de la dimensionalidad.

**Evaluación:** Matriz de confusión + Curva ROC + AUC + coeficientes estandarizados

---

## 🎯 Decisión de negocio: umbral de clasificación

El modelo devuelve una **probabilidad** de éxito para cada canción (0 a 1). El umbral es ajustable según el objetivo de negocio:

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
---
