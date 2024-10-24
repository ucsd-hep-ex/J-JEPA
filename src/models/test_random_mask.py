import unittest
import torch
from train_model import create_random_masks

class TestCreateRandomMasks(unittest.TestCase):
    def test_create_random_masks(self):
        batch_size = 5
        num_subjets = 20
        device = 'cpu'
        context_scale = 0.7

        context_masks, target_masks = create_random_masks(batch_size, num_subjets, device, context_scale)

        # Check the shape of the masks
        self.assertEqual(context_masks.shape, (batch_size, num_subjets))
        self.assertEqual(target_masks.shape, (batch_size, num_subjets))

        # Check that the masks contain only boolean values
        self.assertTrue(context_masks.dtype == torch.bool)
        self.assertTrue(target_masks.dtype == torch.bool)

        # Check that context and target masks are mutually exclusive
        for i in range(batch_size):
            overlap = torch.all((context_masks[i] & target_masks[i]) == 0)
            if not overlap:
                print(f"Overlap found in batch {i}")
                print(f"Context mask: {context_masks[i]}")
                print(f"Target mask: {target_masks[i]}")
            self.assertTrue(overlap)

if __name__ == '__main__':
    unittest.main()
