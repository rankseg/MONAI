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
from unittest import mock

import torch
from parameterized import parameterized

from monai.transforms import AsDiscrete
from monai.transforms.post import array as post_array
from monai.utils import OptionalImportError
from tests.test_utils import TEST_NDARRAYS, assert_allclose

TEST_CASES = []
for p in TEST_NDARRAYS:
    TEST_CASES.append(
        [
            {"argmax": True, "to_onehot": None, "threshold": 0.5},
            p([[[0.0, 1.0]], [[2.0, 3.0]]]),
            p([[[1.0, 1.0]]]),
            (1, 1, 2),
        ]
    )

    TEST_CASES.append(
        [
            {"argmax": True, "to_onehot": 2, "threshold": 0.5, "dim": 0},
            p([[[0.0, 1.0]], [[2.0, 3.0]]]),
            p([[[0.0, 0.0]], [[1.0, 1.0]]]),
            (2, 1, 2),
        ]
    )

    TEST_CASES.append(
        [
            {"argmax": False, "to_onehot": None, "threshold": 0.6},
            p([[[0.0, 1.0], [2.0, 3.0]]]),
            p([[[0.0, 1.0], [1.0, 1.0]]]),
            (1, 2, 2),
        ]
    )

    # test threshold = 0.0
    TEST_CASES.append(
        [
            {"argmax": False, "to_onehot": None, "threshold": 0.0},
            p([[[0.0, -1.0], [-2.0, 3.0]]]),
            p([[[1.0, 0.0], [0.0, 1.0]]]),
            (1, 2, 2),
        ]
    )

    TEST_CASES.append([{"argmax": False, "to_onehot": 3}, p(1), p([0.0, 1.0, 0.0]), (3,)])

    TEST_CASES.append(
        [{"rounding": "torchrounding"}, p([[[0.123, 1.345], [2.567, 3.789]]]), p([[[0.0, 1.0], [3.0, 4.0]]]), (1, 2, 2)]
    )

PROBS = [[[0.01, 0.99]], [[0.99, 0.01]]]
LABELS = [[1.0, 0.0]]
RANKSEG_TEST_CASES = []
for p in TEST_NDARRAYS:
    RANKSEG_TEST_CASES.append([{"rankseg": True}, p(PROBS), p(LABELS), (1, 2)])


class TestAsDiscrete(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_value_shape(self, input_param, img, out, expected_shape):
        result = AsDiscrete(**input_param)(img)
        assert_allclose(result, out, rtol=1e-3, type_test="tensor")
        self.assertTupleEqual(result.shape, expected_shape)

    def test_additional(self):
        for p in TEST_NDARRAYS:
            out = AsDiscrete(argmax=True, dim=1, keepdim=False)(p([[[0.0, 1.0]], [[2.0, 3.0]]]))
            assert_allclose(out, p([[0.0, 0.0], [0.0, 0.0]]), type_test=False)

    @parameterized.expand(TEST_CASES)
    def test_rankseg_false_keeps_default_behavior(self, input_param, img, out, expected_shape):
        result = AsDiscrete(rankseg=False, **input_param)(img)
        assert_allclose(result, out, rtol=1e-3, type_test="tensor")
        self.assertTupleEqual(result.shape, expected_shape)

    @parameterized.expand(RANKSEG_TEST_CASES)
    def test_rankseg_constructor_flag(self, input_param, img, out, expected_shape):
        if not post_array.has_rankseg:
            self.skipTest("Requires rankseg.")
        result = AsDiscrete(**input_param)(img)
        assert_allclose(result, out, type_test="tensor")
        self.assertTupleEqual(result.shape, expected_shape)

    @parameterized.expand(RANKSEG_TEST_CASES)
    def test_rankseg_call_flag(self, input_param, img, out, expected_shape):
        if not post_array.has_rankseg:
            self.skipTest("Requires rankseg.")
        result = AsDiscrete()(img, rankseg=input_param["rankseg"])
        assert_allclose(result, out, type_test="tensor")
        self.assertTupleEqual(result.shape, expected_shape)

    def test_rankseg_call_override_disables_decoder(self):
        transform = AsDiscrete(rankseg=True)
        result = transform(torch.as_tensor(PROBS), rankseg=False)
        assert_allclose(result, torch.as_tensor(PROBS), type_test="tensor")

    def test_rankseg_missing_dependency(self):
        with mock.patch.object(post_array, "has_rankseg", False):
            with self.assertRaises(OptionalImportError):
                AsDiscrete(rankseg=True)(torch.as_tensor(PROBS))

    def test_rankseg_argmax_conflict(self):
        img = torch.as_tensor(PROBS)
        with self.assertRaises(ValueError):
            AsDiscrete(argmax=True, rankseg=True)
        with self.assertRaises(ValueError):
            AsDiscrete(argmax=True)(img, rankseg=True)
        with self.assertRaises(ValueError):
            AsDiscrete(rankseg=True)(img, argmax=True)


if __name__ == "__main__":
    unittest.main()
