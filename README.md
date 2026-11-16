# Clasificación de reportes de daños con transformers

Modelo de lenguaje que clasifica automáticamente los reportes de daños de clientes de una empresa de servicios eléctricos en dos acciones: **ejecutar** (enviar una cuadrilla) o **cancelar** (el reporte no requiere visita). El objetivo es ayudar a priorizar las órdenes de trabajo y reducir visitas innecesarias.

> Los datos pertenecen a la empresa y **no se incluyen** en este repositorio. Los notebooks se publican sin salidas para no exponer información de clientes.

## Qué hice

- **EDA y etiquetado** (`notebooks/EDA.ipynb`): limpié los comentarios de cancelación y resolución, eliminé duplicados priorizando órdenes finalizadas y construí la etiqueta de salida con reglas de negocio.
- **Modelo** (`notebooks/Modelo001.ipynb`): ajusté **DistilBERT** (`distilbert-base-uncased`) para clasificación de texto con Hugging Face Transformers y seguimiento de experimentos en Weights & Biases.
- **Aplicación** (`src/main.py`): interfaz en **Gradio** para clasificar un texto o un archivo CSV completo, con validación de entradas.
- **Calidad y despliegue**: pruebas unitarias con `unittest`, imagen **Docker** y **CI con GitHub Actions** que ejecuta las pruebas y construye la imagen en cada push.

## Resultados y aprendizaje

| Evaluación | Exactitud | F1 |
|---|---|---|
| Validación interna | 0.86 | 0.86 |
| Conjunto de prueba nuevo | 0.52 | 0.47 |

La caída se explica porque la clase minoritaria (*cancelar*, 8 % de los datos) se sobremuestreó **antes** de dividir en entrenamiento y validación: los duplicados quedaron en ambos conjuntos e inflaron la validación. La siguiente versión divide primero y balancea solo el entrenamiento (o usa pesos de clase), y evalúa con F1 por clase.

## Estructura

```
├── notebooks/        EDA y entrenamiento (sin salidas)
├── src/main.py       app de Gradio y clase del modelo
├── test/test.py      pruebas unitarias
├── Dockerfile
└── .github/workflows/ci.yml
```

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
python src/main.py          # abre la interfaz de Gradio
python -m unittest discover -s . -p "test*.py"
```

**Herramientas:** Python, PyTorch, Hugging Face Transformers, Gradio, pandas, scikit-learn, Docker, GitHub Actions.

## Autor

Diego Parra · [LinkedIn](https://www.linkedin.com/in/diegoparragar)
