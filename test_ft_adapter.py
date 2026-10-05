import unittest
import numpy as np
from data import WindowTable
from metrics import inverse_frequency_sample_weights
from models import ft_transformer as ft


class FTAdapterTests(unittest.TestCase):
    def test_refit_serialization_and_scaler(self):
        rng = np.random.default_rng(17)
        X = rng.normal(size=(32, 135)).astype('float32')
        y = np.tile(np.arange(4), 8)
        table = WindowTable(X, y, np.repeat(['a','b'], 16), np.full(32,'FL'), np.arange(32), np.arange(32))
        weights, _ = inverse_frequency_sample_weights(y)
        state, info = ft.fit(table, None, value=.0001, seed=17,
                             settings={'amp': False, 'batch_size': 16}, sample_weight=weights, epochs=2)
        self.assertEqual(info['epochs_run'], 2)
        self.assertIsNone(info['best_epoch'])
        np.testing.assert_allclose(state['scaler'].mean, X.mean(0), atol=1e-6)
        before = ft.predict_proba(state, X)
        restored = ft.deserialize(ft.serialize(state))
        np.testing.assert_allclose(before, ft.predict_proba(restored, X), atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(before.sum(1), 1, atol=1e-6)
        with self.assertRaisesRegex(ValueError, 'epoch'):
            ft.fit(table, None, value=.0001, seed=17, settings={}, sample_weight=weights)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            ft.fit(table, table, value=.0001, seed=17, settings={}, sample_weight=weights)


if __name__ == '__main__':
    unittest.main()
