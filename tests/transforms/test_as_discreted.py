# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import unittest

import torch
from parameterized import parameterized

from monai.transforms import AsDiscreted
from monai.transforms.post import array as post_array
from tests.test_utils import TEST_NDARRAYS, assert_allclose

TEST_CASES = []
for p in TEST_NDARRAYS:
    TEST_CASES.append(
        [
            {"keys": ["pred", "label"], "argmax": [True, False], "to_onehot": 2, "threshold": 0.5},
            {"pred": p([[[0.0, 1.0]], [[2.0, 3.0]]]), "label": p([[[0, 1]]])},
            {"pred": p([[[0.0, 0.0]], [[1.0, 1.0]]]), "label": p([[[1.0, 0.0]], [[0.0, 1.0]]])},
            (2, 1, 2),
        ]
    )

    TEST_CASES.append(
        [
            {"keys": ["pred", "label"], "argmax": False, "to_onehot": None, "threshold": [0.6, None]},
            {"pred": p([[[0.0, 1.0], [2.0, 3.0]]]), "label": p([[[0, 1], [1, 1]]])},
            {"pred": p([[[0.0, 1.0], [1.0, 1.0]]]), "label": p([[[0.0, 1.0], [1.0, 1.0]]])},
            (1, 2, 2),
        ]
    )

    TEST_CASES.append(
        [
            {"keys": ["pred"], "argmax": True, "to_onehot": 2, "threshold": 0.5, "dim": 0, "keepdim": True},
            {"pred": p([[[0.0, 1.0]], [[2.0, 3.0]]])},
            {"pred": p([[[0.0, 0.0]], [[1.0, 1.0]]])},
            (2, 1, 2),
        ]
    )

    TEST_CASES.append(
        [
            {"keys": "pred", "rounding": "torchrounding"},
            {"pred": p([[[0.123, 1.345], [2.567, 3.789]]])},
            {"pred": p([[[0.0, 1.0], [3.0, 4.0]]])},
            (1, 2, 2),
        ]
    )

    # test threshold = 0.0
    TEST_CASES.append(
        [
            {"keys": ["pred", "label"], "argmax": False, "to_onehot": None, "threshold": [0.0, None]},
            {"pred": p([[[0.0, -1.0], [-2.0, 3.0]]]), "label": p([[[0, 1], [1, 1]]])},
            {"pred": p([[[1.0, 0.0], [0.0, 1.0]]]), "label": p([[[0.0, 1.0], [1.0, 1.0]]])},
            (1, 2, 2),
        ]
    )

PRED_PROBS = [[[0.01, 0.99]], [[0.99, 0.01]]]
LABEL_PROBS = [[[0.99, 0.01]], [[0.01, 0.99]]]
PRED_LABELS = [[1.0, 0.0]]
LABEL_LABELS = [[0.0, 1.0]]
RANKSEG_TEST_CASES = []
for p in TEST_NDARRAYS:
    RANKSEG_TEST_CASES.append(
        [
            {"keys": ["pred", "label"], "rankseg": True},
            {"pred": p(PRED_PROBS), "label": p(LABEL_PROBS)},
            {"pred": p(PRED_LABELS), "label": p(LABEL_LABELS)},
            {"pred": (1, 2), "label": (1, 2)},
        ]
    )
    RANKSEG_TEST_CASES.append(
        [
            {"keys": ["pred", "label"], "rankseg": [True, False]},
            {"pred": p(PRED_PROBS), "label": p(LABEL_PROBS)},
            {"pred": p(PRED_LABELS), "label": p(LABEL_PROBS)},
            {"pred": (1, 2), "label": (2, 1, 2)},
        ]
    )


class TestAsDiscreted(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_value_shape(self, input_param, test_input, output, expected_shape):
        result = AsDiscreted(**input_param)(test_input)
        assert_allclose(result["pred"], output["pred"], rtol=1e-3, type_test="tensor")
        self.assertTupleEqual(result["pred"].shape, expected_shape)
        if "label" in result:
            assert_allclose(result["label"], output["label"], rtol=1e-3, type_test="tensor")
            self.assertTupleEqual(result["label"].shape, expected_shape)

    @parameterized.expand(RANKSEG_TEST_CASES)
    def test_rankseg_value_shape(self, input_param, test_input, output, expected_shape):
        if not post_array.has_rankseg:
            self.skipTest("Requires rankseg.")
        result = AsDiscreted(**input_param)(test_input)

        assert_allclose(result["pred"], output["pred"], type_test="tensor")
        assert_allclose(result["label"], output["label"], type_test="tensor")
        self.assertTupleEqual(result["pred"].shape, expected_shape["pred"])
        self.assertTupleEqual(result["label"].shape, expected_shape["label"])

    def test_rankseg_argmax_conflict(self):
        with self.assertRaises(ValueError):
            AsDiscreted(keys="pred", argmax=True, rankseg=True)({"pred": torch.as_tensor(PRED_PROBS)})


if __name__ == "__main__":
    unittest.main()
