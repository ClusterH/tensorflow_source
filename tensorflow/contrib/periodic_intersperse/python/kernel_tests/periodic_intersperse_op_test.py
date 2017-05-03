import numpy as np
import tensorflow as tf


class PeriodicIntersperseTest(tf.test.TestCase):
    def testPeriodicIntersperse(self):
        from tensorflow.contrib.periodic_intersperse import periodic_intersperse as pi

        # basic 2-D tensor
        input_tensor = np.arange(12).reshape((3, 4))
        desired_shape = np.array([6, -1])
        output_tensor = input_tensor.reshape((6, 2))
        with self.test_session():
            result = pi(input_tensor, desired_shape)
            self.assertAllEqual(result.eval(), output_tensor)

        # basic 2-D tensor (truncated)
        input_tensor = np.arange(12).reshape((3, 4))
        desired_shape = np.array([5, -1])
        output_tensor = input_tensor.reshape((6, 2))[:-1]
        with self.test_session():
            result = pi(input_tensor, desired_shape)
            self.assertAllEqual(result.eval(), output_tensor)


if __name__ == "__main__":
    tf.test.main()
