import tensorflow as tf
from abc import abstractmethod

_layer_names = {}


class Layer(object):

    def __init__(self, name=None, *args, **kwargs):

        global _layer_names

        if name is None:
            prefix = self.__class__.__name__
            prefix = prefix.lower()

            if _layer_names.get(prefix) is not None:
                _layer_names[prefix] += 1
                name = prefix + '_' + str(_layer_names.get(prefix))
            else:
                _layer_names[prefix] = 0
                name = prefix
            while True:
                if _layer_names.get(name) is None:
                    break
                _layer_names[prefix] += 1
                name = prefix + '_' + str(_layer_names[prefix])
        else:
            if _layer_names.get(name) is not None:
                raise ValueError('The name:has been used by other layers, please rename it')
            else:
                _layer_names[name] = 0
        self._name = name
        self._all_weights = None
        self._trainable_weights = None
        self._is_training = True
        self._built = False

        self._non_trainable_weights = []
        self._trainable_weights = []

    def _get_weights(self, name, shape, initializer, regularizer, trainable=True):
        with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
            weights = tf.get_variable(name=name,
                                      shape=shape,
                                      initializer=initializer,
                                      regularizer=regularizer)
        if trainable is True:
            self._trainable_weights.append(weights)

        return weights


    @staticmethod
    def _compute_shape(tensors):
        if isinstance(tensors, list):
            shape = [i.get_shape().as_list() for i in tensors]
        else:
            shape = tensors.get_shape().as_list()
        return shape

    @abstractmethod
    def build(self, inputs_shape):
        pass

    @abstractmethod
    def forward(self, inputs):
        pass

    def __call__(self, inputs, *args, **kwargs):
        input_tensors = tf.convert_to_tensor(inputs)
        input_shape = self._compute_shape(input_tensors)
        self.build(input_shape)
        outputs = self.forward(input_tensors)

        return outputs


