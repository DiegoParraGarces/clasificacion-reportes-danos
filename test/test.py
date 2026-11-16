import unittest
from src.main import HuggingFaceModel, process_file
import tempfile
import pandas as pd

class TestHuggingFaceModel(unittest.TestCase):
    def setUp(self):
        """Configura una instancia del modelo para usar en las pruebas."""
        self.model = HuggingFaceModel()

    def test_predict_valid_text(self):
        """Prueba que el modelo predice correctamente etiquetas válidas."""
        text = "un arbol corto las cuerdas"
        prediction = self.model.predict(text)
        self.assertIsInstance(prediction, str)  # La salida debe ser un string
        self.assertIn(prediction, ["ejecutar", "cancelar"])  # Validar etiquetas

    def test_predict_invalid_text(self):
        """Prueba que el modelo maneja entradas inválidas correctamente."""
        invalid_texts = ["", "12", "!!"]
        for text in invalid_texts:
            prediction = self.model.predict(text)
            self.assertEqual(prediction, "Entrada inválida. Por favor, ingrese un texto coherente.")

    def test_predict_with_special_characters(self):
        """Prueba que el modelo maneja entradas con caracteres especiales."""
        text = "El daño fue causado por un *pájaro* 🐦."
        prediction = self.model.predict(text)
        self.assertEqual(prediction, "Entrada inválida. Por favor, ingrese un texto coherente.")

class TestProcessFile(unittest.TestCase):
    def setUp(self):
        """Crea un archivo temporal para pruebas de procesamiento."""
        self.test_data = pd.DataFrame({"daño": ["un árbol cortó las cuerdas", "no hay energía"]})
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        self.test_data.to_csv(self.temp_file.name, index=False, encoding="utf-8")

    def tearDown(self):
        """Cierra y elimina el archivo temporal después de las pruebas."""
        self.temp_file.close()

    def test_process_file_success(self):
        """Prueba que el archivo es procesado correctamente."""
        message, output_file = process_file(self.temp_file)
        self.assertEqual(message, "Archivo procesado exitosamente.")
        self.assertIsNotNone(output_file)

        # Validar el contenido del archivo generado
        result_data = pd.read_csv(output_file)
        self.assertIn("Predicción", result_data.columns)
        self.assertEqual(len(result_data), len(self.test_data))

    def test_process_file_missing_column(self):
        """Prueba que el archivo con columnas incorrectas genera un error."""
        invalid_data = pd.DataFrame({"texto": ["un árbol cortó las cuerdas"]})
        invalid_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        invalid_data.to_csv(invalid_file.name, index=False, encoding="utf-8")

        message, output_file = process_file(invalid_file)
        self.assertEqual(message, "Error: El archivo no tiene una columna llamada 'daño'.")
        self.assertIsNone(output_file)

        invalid_file.close()


if __name__ == "__main__":
    unittest.main()
