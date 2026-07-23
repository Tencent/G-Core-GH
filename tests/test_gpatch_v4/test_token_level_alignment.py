import unittest

import torch

from gpatch_v4.utils.ppo_utils import align_token_level_tensors_to_logprobs


class TokenLevelAlignmentTest(unittest.TestCase):
    def _align(self, tensor, logprobs_length, sequence_length, truncate_head):
        return align_token_level_tensors_to_logprobs(
            [torch.tensor(tensor, dtype=torch.float32)],
            [torch.zeros(logprobs_length)],
            [torch.tensor(sequence_length)],
            truncate_head,
        )[0]

    def test_full_token_tensor_truncates_head(self):
        aligned = self._align([10, 20, 30, 40], 3, 4, truncate_head=True)

        self.assertTrue(torch.equal(aligned, torch.tensor([20.0, 30.0, 40.0])))

    def test_full_token_tensor_truncates_tail(self):
        aligned = self._align([10, 20, 30, 40], 3, 4, truncate_head=False)

        self.assertTrue(torch.equal(aligned, torch.tensor([10.0, 20.0, 30.0])))

    def test_full_token_tensor_is_shifted_when_padded_logprobs_has_same_length(self):
        aligned = self._align([10, 20, 30, 40], 4, 4, truncate_head=True)

        self.assertTrue(torch.equal(aligned, torch.tensor([20.0, 30.0, 40.0, 0.0])))

    def test_pre_aligned_tensor_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "full, unpadded token axis"):
            self._align([20, 30, 40], 5, 4, truncate_head=True)

    def test_padded_full_token_tensor_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "full, unpadded token axis"):
            self._align([0, 1, 2, 3, 4, 5], 5, 4, truncate_head=True)

    def test_unpadded_full_token_tensor_is_padded_to_logprobs_length(self):
        aligned = self._align([0, 1, 2, 3], 5, 4, truncate_head=True)

        self.assertTrue(torch.equal(aligned, torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0])))


if __name__ == "__main__":
    unittest.main()
