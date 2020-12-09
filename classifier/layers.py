from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import math_ops

class Rescaling(Layer):
    """Multiply inputs by `scale` and adds `offset`.
  For instance:
  1. To rescale an input in the `[0, 255]` range
  to be in the `[0, 1]` range, you would pass `scale=1./255`.
  2. To rescale an input in the `[0, 255]` range to be in the `[-1, 1]` range,
  you would pass `scale=1./127.5, offset=-1`.
  The rescaling is applied both during training and inference.
  Input shape:
    Arbitrary.
  Output shape:
    Same as input.
  Arguments:
    scale: Float, the scale to apply to the inputs.
    offset: Float, the offset to apply to the inputs.
    name: A string, the name of the layer.
  """

    def __init__(self, scale, offset=0., name=None, **kwargs):
        self.scale = scale
        self.offset = offset
        super(Rescaling, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        dtype = self._compute_dtype
        scale = math_ops.cast(self.scale, dtype)
        offset = math_ops.cast(self.offset, dtype)
        return math_ops.cast(inputs, dtype) * scale + offset

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'scale': self.scale,
            'offset': self.offset,
        }
        base_config = super(Rescaling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
