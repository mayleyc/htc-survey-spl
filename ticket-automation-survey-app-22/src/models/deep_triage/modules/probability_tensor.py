from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Wrapper, TimeDistributed, InputSpec


def make_safe(x):
    return K.clip(x, K.epsilon(), 1.0 - K.epsilon())


class ProbabilityTensor(Wrapper):
    """ function for turning 3d tensor to 2d probability matrix, which is the set of a_i's """

    def __init__(self, dense_function=None, *args, **kwargs):
        self.supports_masking = True
        self.input_spec = [InputSpec(ndim=3)]
        # layer = TimeDistributed(dense_function) or TimeDistributed(Dense(1, name='ptensor_func'))
        layer = TimeDistributed(Dense(1, name="ptensor_func"))
        super().__init__(layer, *args, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.input_spec = [InputSpec(shape=input_shape)]
        if K.backend() == "tensorflow":
            if not input_shape[1]:
                raise Exception("When using TensorFlow, you should define "
                                "explicitly the number of timesteps of "
                                "your sequences.\n"
                                "If your first layer is an Embedding, "
                                "make sure to pass it an \"input_length\" "
                                "argument. Otherwise, make sure "
                                "the first layer has "
                                "an \"input_shape\" or \"batch_input_shape\" "
                                "argument, including the time axis.")

        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(ProbabilityTensor, self).build()

    def get_output_shape_for(self, input_shape):
        # b,n,f -> b,n
        #       s.t. \sum_n n = 1
        if isinstance(input_shape, (list, tuple)) and not isinstance(input_shape[0], int):
            input_shape = input_shape[0]

        return input_shape[0], input_shape[1]

    def squash_mask(self, mask):
        if K.ndim(mask) == 2:
            return mask
        elif K.ndim(mask) == 3:
            return K.any(mask, axis=-1)

    def compute_mask(self, x, mask=None):
        if mask is None:
            return None
        return self.squash_mask(mask)

    def call(self, x, mask=None):
        energy = K.squeeze(self.layer(x), 2)
        p_matrix = K.softmax(energy)
        if mask is not None:
            mask = self.squash_mask(mask)
            p_matrix = make_safe(p_matrix * mask)
            p_matrix = (p_matrix / K.sum(p_matrix, axis=-1, keepdims=True)) * mask
        return p_matrix

    def get_config(self):
        config = {}
        base_config = super(ProbabilityTensor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
