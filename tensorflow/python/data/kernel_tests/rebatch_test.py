# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for `tf.data.Dataset.rebatch()`."""

from absl.testing import parameterized

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import combinations
from tensorflow.python.ops.ragged import ragged_tensor


def _flat_shapes(dataset):
  return [
      ts.as_list()
      for ts in nest.flatten(dataset_ops.get_legacy_output_shapes(dataset))
  ]


class RebatchTest(test_base.DatasetTestBase, parameterized.TestCase):

  ##############################################################################
  # The following tests exercise our static computation of output_shapes.
  ##############################################################################

  @combinations.generate(test_base.default_test_combinations())
  def testShapeInferenceNotAllBatchSizesEqual(self):
    dataset = dataset_ops.Dataset.range(8).batch(4, drop_remainder=True)
    rebatched_dataset = dataset_ops.rebatch(dataset, batch_sizes=[2, 1, 1])
    expected_shapes = [[None]]
    self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(drop_remainder=[True, False])))
  def testShapeInferenceInputBatchDimDivisible(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(8).batch(4, drop_remainder=True)
    rebatched_dataset = dataset_ops.rebatch(
        dataset, batch_sizes=[2, 2], drop_remainder=drop_remainder)
    expected_shapes = [[2]]
    self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testShapeInferenceInputBatchDimUnknown(self):
    dataset = dataset_ops.Dataset.range(8).batch(4, drop_remainder=False)
    rebatched_dataset = dataset_ops.rebatch(
        dataset, batch_sizes=[2, 2], drop_remainder=False)
    expected_shapes = [[None]]
    self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testShapeInferenceInputBatchDimUnknownWithDropRemainder(self):
    dataset = dataset_ops.Dataset.range(8).batch(4, drop_remainder=False)
    rebatched_dataset = dataset_ops.rebatch(
        dataset, batch_sizes=[2, 2], drop_remainder=True)
    expected_shapes = [[2]]
    self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testShapeInferenceInputBatchDimIndivisible(self):
    dataset = dataset_ops.Dataset.range(10).batch(5, drop_remainder=True)
    rebatched_dataset = dataset_ops.rebatch(
        dataset, batch_sizes=[2, 2], drop_remainder=False)
    expected_shapes = [[None]]
    self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testShapeInferenceInputBatchDimIndivisibleWithDropRemainder(self):
    dataset = dataset_ops.Dataset.range(10).batch(5, drop_remainder=True)
    rebatched_dataset = dataset_ops.rebatch(
        dataset, batch_sizes=[2, 2], drop_remainder=True)
    expected_shapes = [[2]]
    self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))

  ##############################################################################
  # The following tests check _RebatchDataset's output.
  ##############################################################################
  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(drop_remainder=[True, False])))
  def testBasic(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(8).batch(4, drop_remainder=True)
    rebatched_dataset = dataset_ops.rebatch(
        dataset, batch_sizes=[2, 2], drop_remainder=drop_remainder)

    expected_shapes = [[2]]
    self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))

    expected_output = [[0, 1], [2, 3], [4, 5], [6, 7]]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testPartialBatch(self):
    dataset = dataset_ops.Dataset.range(5).batch(4, drop_remainder=False)
    rebatched_dataset = dataset_ops.rebatch(
        dataset, batch_sizes=[2, 2], drop_remainder=False)

    expected_shapes = [[None]]
    self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))
    expected_output = [[0, 1], [2, 3], [4]]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testPartialBatchWithDropRemainder(self):
    dataset = dataset_ops.Dataset.range(5).batch(4, drop_remainder=False)
    rebatched_dataset = dataset_ops.rebatch(
        dataset, batch_sizes=[2, 2], drop_remainder=True)

    expected_shapes = [[2]]
    self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))
    expected_output = [[0, 1], [2, 3]]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(drop_remainder=[True, False])))
  def testBatchSizeGreaterThanOriginal(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(12).batch(4, drop_remainder=False)
    rebatched_dataset = dataset_ops.rebatch(
        dataset, batch_sizes=[6], drop_remainder=drop_remainder)

    expected_output = [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(drop_remainder=[True, False])))
  def testEmptySplits(self, drop_remainder):
    # It's possible for splits to be empty if the batch size is smaller than
    # the number of replicas. Here, we use an example with batch_size == 4
    # and num_replicas == 5.
    dataset = dataset_ops.Dataset.range(8).batch(4, drop_remainder=True)
    rebatched_dataset = dataset_ops.rebatch(
        dataset, batch_sizes=[1, 1, 1, 1, 0], drop_remainder=drop_remainder)

    expected_shapes = [[None]]
    self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))

    expected_output = [[0], [1], [2], [3], [], [4], [5], [6], [7], []]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(drop_remainder=[True, False])))
  def testEmptyFirstSplits(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(8).batch(4, drop_remainder=True)
    rebatched_dataset = dataset_ops.rebatch(
        dataset, batch_sizes=[0, 1], drop_remainder=drop_remainder)

    expected_shapes = [[None]]
    self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))

    # We have an extra element at the end because if the desired batch size is
    # zero, then we never read any inputs from the input_dataset at all, so we
    # will keep producting empty outputs until we reach a non zero desired batch
    # size split.
    expected_output = [[], [0], [], [1], [], [2], [], [3], [], [4], [], [5], [],
                       [6], [], [7], []]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(drop_remainder=[True, False])))
  def testEmptyLastSplits(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(8).batch(4, drop_remainder=True)
    rebatched_dataset = dataset_ops.rebatch(
        dataset, batch_sizes=[1, 0], drop_remainder=drop_remainder)

    expected_shapes = [[None]]
    self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))

    expected_output = [[0], [], [1], [], [2], [], [3], [], [4], [], [5], [],
                       [6], [], [7], []]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(drop_remainder=[True, False])))
  def testScalarBatchSizeInput(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(8).batch(4, drop_remainder=True)
    rebatched_dataset = dataset_ops.rebatch(
        dataset, batch_sizes=2, drop_remainder=drop_remainder)

    expected_shapes = [[2]]
    self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))

    expected_output = [[0, 1], [2, 3], [4, 5], [6, 7]]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testMultipleBatches(self):
    dataset = dataset_ops.Dataset.range(16).batch(
        2, drop_remainder=True).batch(
            4, drop_remainder=True)
    self.assertEqual([[4, 2]], _flat_shapes(dataset))

    rebatched_dataset = dataset_ops.rebatch(dataset, [2, 2])
    self.assertEqual([[2, 2]], _flat_shapes(rebatched_dataset))
    # Each element is a list of 2 elements where each element is a list of 2.
    expected_output = [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]],
                       [[12, 13], [14, 15]]]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testNestedDictionaryOutput(self):

    def map_fn(x):
      return {"a": x, "b": {"c": x + 1}}

    dataset = dataset_ops.Dataset.range(8).map(map_fn).batch(
        4, drop_remainder=True)
    rebatched_dataset = dataset_ops.rebatch(dataset, [2, 2])
    self.assertEqual([[2], [2]], _flat_shapes(rebatched_dataset))

    expected_output = [{
        "a": [0, 1],
        "b": {
            "c": [1, 2]
        }
    }, {
        "a": [2, 3],
        "b": {
            "c": [3, 4]
        }
    }, {
        "a": [4, 5],
        "b": {
            "c": [5, 6]
        }
    }, {
        "a": [6, 7],
        "b": {
            "c": [7, 8]
        }
    }]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(drop_remainder=[True, False])))
  def testRaggedDataset(self, drop_remainder):
    # Set up a dataset that produces ragged tensors with a static batch size.
    dataset = dataset_ops.Dataset.from_tensor_slices(
        ragged_tensor.RaggedTensor.from_row_lengths(
            list(range(10)), [1, 2, 3, 4]))
    # The map changes the internal representation of the ragged tensor.
    # This test will fail if we don't normalize the tensor representation.
    dataset = dataset.batch(4, drop_remainder=True).map(lambda x: x)

    rebatched_dataset = dataset_ops.rebatch(dataset, batch_sizes=[2, 2])

    expected_output = [
        ragged_tensor.RaggedTensor.from_row_lengths(list(range(3)), [1, 2]),
        ragged_tensor.RaggedTensor.from_row_lengths(list(range(3, 10)), [3, 4]),
    ]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testNoneDataset(self):
    # Some datasets, e.g. datasets with None tensors, have components without
    # output shapes. Test that this doesn't break rebatching shape inference
    # logic.
    dataset = dataset_ops.Dataset.range(4)
    dataset = dataset.map(lambda x: (x, None))
    dataset = dataset.batch(4, drop_remainder=True)
    _ = dataset_ops.rebatch(dataset, batch_sizes=[2, 2])
