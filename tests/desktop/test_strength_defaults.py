import unittest
from PySide6.QtWidgets import QApplication
from puregpu3d.desktop.window import MainWindow
from puregpu3d.runtime.protocol import WorkerCommand


class TestStrengthDefaults(unittest.TestCase):
    def test_default_and_limits(self):
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        self.assertEqual(window.controller.disparity_strength, 0.001)
        self.assertEqual(window.strength_spin.value(), 0.001)
        self.assertEqual(window.strength_spin.maximum(), 0.01)
        self.assertEqual(window.strength_spin.singleStep(), 0.001)
        window.strength_slider.setValue(window.strength_slider.maximum())
        self.assertEqual(window.controller.disparity_strength, 0.01)
        window.controller.set_disparity_strength(0.03)
        self.assertEqual(window.controller.disparity_strength, 0.01)
        self.assertEqual(WorkerCommand(job_id='test', input_path='in', output_path='out').disparity_strength, 0.001)
        window.close()
