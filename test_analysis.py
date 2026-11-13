"""Independent checks using mathematical inputs only; no laboratory data."""
import math
from pathlib import Path
import unittest
import numpy as np
from streamlit.testing.v1 import AppTest
from ic_analysis import integrate, detect_peaks, parse_chromatogram, fit, predict, invert, quantify


class AnalysisTests(unittest.TestCase):
    def test_gaussian_integral(self):
        t = np.linspace(0, 12, 2401)
        xy = np.column_stack((t, 0.2*t + 3*np.exp(-0.5*((t-6)/0.25)**2)))
        peak = integrate(xy, 4, 8)
        self.assertAlmostEqual(peak['area'], 3*0.25*math.sqrt(2*math.pi), places=6)
        self.assertAlmostEqual(detect_peaks(xy)[0]['rt'], 6, delta=0.02)

    def test_invalid_time(self):
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            parse_chromatogram('\n'.join(f'{i//2},1' for i in range(40)))

    def test_piecewise_continuity_and_inverse(self):
        # Arbitrary mathematical response, not an instrument calibration.
        points = [{'concentration': c, 'area': 1000*c} for c in [0.0002, 0.0006, 0.001, 0.003, 0.008]]
        model = fit(points)
        self.assertEqual(invert(model, 0)['concentration'], 0)
        self.assertAlmostEqual(predict(model, 0.001), model['joinArea'])
        for c in [0.00001, 0.0008, 0.001, 0.005, 0.01]:
            self.assertAlmostEqual(invert(model, predict(model, c))['concentration'], c)
        self.assertTrue(invert(model, predict(model, 0.00001))['belowMeasuredRange'])
        with self.assertRaisesRegex(ValueError, 'method differs'):
            quantify({'method': 'other', 'volume': 2}, {'area': 1}, model, {'method': 'reference', 'volume': 2})

    def test_empty_app_starts_without_calibration_files(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parent/'streamlit_app.py')).run()
        self.assertFalse(app.exception)
        self.assertIn('No calibration data', app.info[0].value)
        self.assertEqual(len(app.get('file_uploader')), 2)

    def test_app_with_in_memory_standards(self):
        from streamlit_app import analyze
        t = np.linspace(0, 12, 2401)
        standards = {}
        for i, c in enumerate([0.0002, 0.0006, 0.001, 0.003, 0.008]):
            xy = np.column_stack((t, c*1000*np.exp(-0.5*((t-6)/0.25)**2)))
            standards[str(i)] = analyze({'name': f'NaI-{c}M.txt', 'concentration': c, 'xy': xy, 'method': 'mathematical-test', 'volume': 2.})
        app = AppTest.from_file(str(Path(__file__).resolve().parent/'streamlit_app.py'))
        app.session_state['standards'] = standards
        app.run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        app.selectbox[0].select('Piecewise (0.001 M boundary)').run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)


if __name__ == '__main__':
    unittest.main()
