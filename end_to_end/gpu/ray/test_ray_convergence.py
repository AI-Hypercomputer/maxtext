import unittest
import sys
import json

class RayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):    
        if not hasattr(cls, 'metrics_file') or not hasattr(cls, 'loss_threshold'):
            raise ValueError("Filename or loss threshold not provided")
    
    def _get_last_n_data(self, metrics_file, target, n=10):
        last_n_data = []
        with open(metrics_file, 'r', encoding='utf8') as file:
            lines = file.readlines()
            for line in lines[::-1]:
                metrics = json.loads(line)
                if target in metrics:
                    last_n_data.append(metrics[target])
                    if len(last_n_data) >= n:
                        break
        return last_n_data

    def test_final_loss(self):
        use_last_n_data = 10
        last_n_data = self._get_last_n_data(self.metrics_file, 'learning/loss', use_last_n_data)
        avg_last_n_data = sum(last_n_data) / len(last_n_data)
        self.assertLess(avg_last_n_data, self.loss_threshold)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise ValueError("Filename and loss threshold must be provided as command-line arguments")
    
    metrics_file = sys.argv[1]
    loss_threshold = float(sys.argv[2])
    sys.argv = sys.argv[:1]

    RayTest.metrics_file = metrics_file
    RayTest.loss_threshold = loss_threshold

    unittest.main()