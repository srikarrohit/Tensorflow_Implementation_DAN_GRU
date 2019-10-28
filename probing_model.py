# external libs
import tensorflow as tf
from tensorflow.keras import models, layers

# project imports
from util import load_pretrained_model

class ProbingClassifier(models.Model):
    def __init__(self,
                 pretrained_model_path: str,
                 layer_num: int,
                 classes_num: int) -> 'ProbingClassifier':
        """
        It loads a pretrained main model. On the given input,
        it takes the representations it generates on certain layer
        and learns a linear classifier on top of these frozen
        features.

        Parameters
        ----------
        pretrained_model_path : ``str``
            Serialization directory of the main model which you
            want to probe at one of the layers.
        layer_num : ``int``
            Layer number of the pretrained model on which to learn
            a linear classifier probe.
        classes_num : ``int``
            Number of classes that the ProbingClassifier chooses from.
        """
        super(ProbingClassifier, self).__init__()
        self._pretrained_model = load_pretrained_model(pretrained_model_path)
        self._pretrained_model.trainable = False
        self._layer_num = layer_num
        self._var = tf.keras.layers.Dense(classes_num,activation="softmax")
        #self._sequence = tf.keras.layers.GRU(classes_num,activation='softmax',return_sequences=True,return_state=True)
        # TODO(students): start
        # ...
        # TODO(students): end

    def call(self, inputs: tf.Tensor, training: bool =False) -> tf.Tensor:
        """
        Forward pass of Probing Classifier.

        Parameters
        ----------
        inputs : ``str``
            Tensorized version of the batched input text. It is of shape:
            (batch_size, max_tokens_num) and entries are indices of tokens
            in to the vocabulary. 0 means that it's a padding token. max_tokens_num
            is maximum number of tokens in any text sequence in this batch.
        training : ``bool``
            Whether this call is in training mode or prediction mode.
            This flag is useful while applying dropout because dropout should
            only be applied during training.
        """
        # TODO(students): start
        # ...
        # TODO(students): end
        #print(tf.shape(self._pretrained_model))
        #out= self._pretrained_model(inputs)
        #embedding = out["layer_representations"]
        #logits = self._var(embedding[:,self._layer_num-1,:])
        
        #embedded_tokens = tf.nn.embedding_lookup(self._pretrained_model._embeddings, inputs)#for gru working
        #tokens_mask = tf.cast(inputs!=0, tf.float32)
        #outputs = self._pretrained_model._seq2vec_layer(embedded_tokens, tokens_mask, training)
        #embedding = outputs["layer_representations"]
        #logits = self._var(embedding[self._layer_num-1])
        out= self._pretrained_model(inputs)#for dan working
        embedding = out["layer_representations"]
        logits = self._var(embedding[:,self._layer_num-1,:])
        return {"logits": logits}
