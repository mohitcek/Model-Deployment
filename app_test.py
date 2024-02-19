"""Unit test file for app.py"""
from app import model_prediction
import unittest

class TestApp(unittest.TestCase):
    """Unit tests defined for app.py"""

    def test_model_prediction(self):
        """Test return backwards simple string"""
        new_prediction = model_prediction('figures/test_image.png')
        correct_prediction = 0
        self.assertEqual(correct_prediction, new_prediction)

if __name__ == "__main__":
    unittest.main()
