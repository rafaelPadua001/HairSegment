import os
import unittest
import numpy as np
import pandas as pd
import cv2
from main import (load_annotations, create_hair_mask, 
                   preprocess_image_and_mask, load_data, 
                   check_directories)

class TestHairSegmentation(unittest.TestCase):

    def setUp(self):
        # Setup paths for tests
        self.test_csv_path = "test_annotations.csv"
        self.test_image_dir = "test_images"
        self.test_mask_dir = "test_masks"
        self.test_output_dir = "test_outputs"

        # Criar diretórios de teste
        os.makedirs(self.test_image_dir, exist_ok=True)
        os.makedirs(self.test_mask_dir, exist_ok=True)
        os.makedirs(self.test_output_dir, exist_ok=True)

        # Criar um CSV de teste
        data = {
            'filename': ['test_image.jpg'],
            'xmin': [10],
            'ymin': [10],
            'xmax': [50],
            'ymax': [50]
        }
        df = pd.DataFrame(data)
        df.to_csv(self.test_csv_path, index=False)

        # Criar uma imagem de teste
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(self.test_image_dir, 'test_image.jpg'), test_image)

    def tearDown(self):
        # Remover arquivos de teste antes de excluir diretórios
        if os.path.exists(self.test_image_dir):
            for filename in os.listdir(self.test_image_dir):
                os.remove(os.path.join(self.test_image_dir, filename))
            os.rmdir(self.test_image_dir)

        if os.path.exists(self.test_mask_dir):
            for filename in os.listdir(self.test_mask_dir):
                os.remove(os.path.join(self.test_mask_dir, filename))
            os.rmdir(self.test_mask_dir)

        if os.path.exists(self.test_output_dir):
            for filename in os.listdir(self.test_output_dir):
                os.remove(os.path.join(self.test_output_dir, filename))
            os.rmdir(self.test_output_dir)

        # Remove o CSV de teste
        if os.path.exists(self.test_csv_path):
            os.remove(self.test_csv_path)

    def test_load_annotations(self):
        annotations = load_annotations(self.test_csv_path)
        self.assertIsNotNone(annotations)
        self.assertIn('filename', annotations.columns)
        self.assertIn('xmin', annotations.columns)
        self.assertIn('ymin', annotations.columns)
        self.assertIn('xmax', annotations.columns)
        self.assertIn('ymax', annotations.columns)
        self.assertEqual(len(annotations), 1)  # Deve haver uma entrada no CSV

    def test_create_hair_mask(self):
        image_shape = (100, 100, 3)
        coordinates = [(10, 10, 50, 50)]
        mask = create_hair_mask(image_shape, coordinates)
        self.assertEqual(mask.shape, (100, 100))
        self.assertEqual(np.unique(mask).tolist(), [0, 255])

    def test_preprocess_image_and_mask(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        coordinates = [(10, 10, 50, 50)]
        image_resized, mask_resized = preprocess_image_and_mask(image, coordinates)
        self.assertEqual(image_resized.shape, (128, 128, 3))
        self.assertEqual(mask_resized.shape, (128, 128))

    def test_load_data(self):
        X, y = load_data()
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertEqual(X.shape[0], 1)  # Deveria carregar uma imagem apenas

    def test_check_directories(self):
        result = check_directories()
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()
