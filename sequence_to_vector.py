# std lib imports
from typing import Dict

# external libs
import tensorflow as tf
from tensorflow.keras import layers, models

class SequenceToVector(models.Model):
    """
    It is an abstract class defining SequenceToVector enocoder
    abstraction. To build you own SequenceToVector encoder, subclass
    this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    """
    def __init__(self,
                 input_dim: int) -> 'SequenceToVector':
        super(SequenceToVector, self).__init__()
        self._input_dim = input_dim

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Main Classifier.

        Parameters
        ----------
        vector_sequence : ``tf.Tensor``
            Sequence of embedded vectors of shape (batch_size, max_tokens_num, embedding_dim)
        sequence_mask : ``tf.Tensor``
            Boolean tensor of shape (batch_size, max_tokens_num). Entries with 1 indicate that
            token is a real token, and 0 indicate that it's a padding token.
        training : ``bool``
            Whether this call is in training mode or prediction mode.
            This flag is useful while applying dropout because dropout should
            only be applied during training.

        Returns
        -------
        An output dictionary consisting of:
        combined_vector : tf.Tensor
            A tensor of shape ``(batch_size, embedding_dim)`` representing vector
            compressed from sequence of vectors.
        layer_representations : tf.Tensor
            A tensor of shape ``(batch_size, num_layers, embedding_dim)``.
            For each layer, you typically have (batch_size, embedding_dim) combined
            vectors. This is a stack of them.
        """
        # ...
        # return {"combined_vector": combined_vector,
        #         "layer_representations": layer_representations}
        raise NotImplementedError


class DanSequenceToVector(SequenceToVector):
    """
    It is a class defining Deep Averaging Network based Sequence to Vector
    encoder. You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this DAN encoder.
    dropout : `float`
        Token dropout probability as described in the paper.
    """
    def __init__(self, input_dim: int, num_layers: int, dropout: float = 0.3):
        super(DanSequenceToVector, self).__init__(input_dim)
        
        self._input_dim = input_dim
        self._num_layers = num_layers
        self._dropout = dropout
        self.dan_layers = []
        for i in range(num_layers):
            var = tf.keras.layers.Dense(input_dim,activation="relu")
            self.dan_layers.append(var)

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        
        #combined_vector : tf.Tensor
        #    A tensor of shape ``(batch_size, embedding_dim)`` representing vector
        #    compressed from sequence of vectors.
        #layer_representations : tf.Tensor
        #    A tensor of shape ``(batch_size, num_layers, embedding_dim)``.
        #    For each layer, you typically have (batch_size, embedding_dim) combined
        #    vectors. This is a stack of them.
        # TODO(students): start
        # ...
        # TODO(students): end
       # print(vector_sequence)     
        new_sequence_mask = tf.cast(sequence_mask,tf.float32)
        if training:
            random_distribution = tf.random.uniform(tf.shape(sequence_mask))
            mask = random_distribution > self._dropout
            mask = tf.cast(mask, tf.float32)#converting bool to float
            new_sequence_mask = sequence_mask*mask
        dim_list = vector_sequence.get_shape().as_list()#[64,209,50]
        tiled_sequence = tf.tile(new_sequence_mask, (1,dim_list[2]))
        r = tf.transpose(tf.reshape(tiled_sequence, [-1,dim_list[2],dim_list[1]]), [0, 2, 1])
        w_3d = vector_sequence*r
        w_2d = tf.reduce_sum(w_3d,axis=1)
        w_final = w_2d/tf.reshape(tf.reduce_sum(new_sequence_mask,axis=1),[dim_list[0],1])
        list_layers = []
        list_layers.append(self.dan_layers[0](w_final))
        for i in range(1,self._num_layers):
            list_layers.append(self.dan_layers[i](list_layers[-1]))
        layers_stacked = tf.stack(list_layers)
        layer_representations = tf.transpose(layers_stacked, [1, 0, 2])
        combined_vector = list_layers[-1]
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}

class GruSequenceToVector(SequenceToVector):
    """
    It is a class defining GRU based Sequence To Vector encoder.
    You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this GRU-based encoder. Note that each layer
        is a GRU encoder unlike DAN where they were feedforward based.
    """
    def __init__(self, input_dim: int, num_layers: int):
        super(GruSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        # ...
        # TODO(students): end
        self._input_dim = input_dim
        self._num_layers = num_layers
        self.gru_layers = []
        for i in range(num_layers):
            var = tf.keras.layers.GRU(input_dim,return_sequences=True,return_state=True)
            self.gru_layers.append(var)

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        # TODO(students): start
        # ...
        # TODO(students): end        
        sequence_layers = []
        state_layers = []
        sequence, state = self.gru_layers[0](vector_sequence,mask=sequence_mask)
        sequence_layers.append(sequence)
        state_layers.append(state)
        for i in range(1,self._num_layers):
            sequence, state = self.gru_layers[i](sequence_layers[-1],mask=sequence_mask)
            sequence_layers.append(sequence)
            state_layers.append(state)
        layers_stacked = tf.stack(state_layers)
        layer_representations = tf.transpose(layers_stacked, [1, 0, 2])
        combined_vector = state_layers[-1]
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}
