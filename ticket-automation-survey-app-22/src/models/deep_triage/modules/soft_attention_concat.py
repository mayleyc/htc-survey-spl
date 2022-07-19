from tensorflow.keras import backend as K

from src.models.deep_triage.modules.probability_tensor import ProbabilityTensor


# The Soft Attention layer is implemented as follows:
class SoftAttentionConcat(ProbabilityTensor):
    """This will create the context vector and then concatenate it with the last output of the LSTM"""

    def get_output_shape_for(self, input_shape):
        # b,n,f -> b,f where f is weighted features summed across n
        return input_shape[0], 2 * input_shape[2]

    def compute_mask(self, x, mask=None):
        if mask is None or mask.ndim == 2:
            return None
        else:
            raise Exception("Unexpected situation")

    def call(self, x, mask=None):
        # b,n,f -> b,f via b,n broadcast
        p_vectors = K.expand_dims(super(SoftAttentionConcat, self).call(x, mask), 2)
        expanded_p = K.repeat_elements(p_vectors, K.int_shape(x)[2], axis=2)
        context = K.sum(expanded_p * x, axis=1)
        last_out = x[:, -1, :]
        return K.concatenate([context, last_out])
