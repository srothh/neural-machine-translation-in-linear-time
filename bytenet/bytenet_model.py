import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F


# TODO: Everything, just a basic placeholder CNN encoder/decoder for now
class BytenetEncoder(nn.Module):
    def __init__(self, num_layers, num_channels, kernel_size, dilation_rate):
        super(BytenetEncoder, self).__init__()
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Conv1d(num_channels, kernel_size, dilation_rate ** i))

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
        return x


class BytenetDecoder(nn.Module):
    def __init__(self, num_layers, num_channels, kernel_size, dilation_rate):
        super(BytenetDecoder, self).__init__()
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Conv1d(num_channels, kernel_size, dilation_rate ** i))

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
        return x


class DynamicUnfolding(nn.Module):
    """
    Initialize the dynamic unfolding with given parameters.

    :param a: Scaling factor for source sequence length.
    :param b: Offset for target sequence length.
    """

    def __init__(self, a=1.20, b=0):
        super(DynamicUnfolding, self).__init__()
        self.a = a
        self.b = b

    def calculate_target_length(self, source):
        """
        Calculates the target length for dynamic unfolding

        :param source:
        :return: rounded target length as int
        """
        source_length = len(source)
        target_length = self.a * source_length + self.b

        return int(round(target_length))

    def unfold(self, encoder_output, decoder, source):
        """
        Creates the unfolding process using the encoder output as
        input for the decoder. the decoder unfold step-
        by-step over the encoder representation until the decoder itself
        outputs an end-of-sequence symbol

        :param encoder_output: The output of the encoder model used for unfolding
        :param decoder: The decoder model
        :param source: The source, used for calculating the target length
        """
        batch_size = encoder_output.size(1)
        hidden_state = None
        end_of_sequence_symbol = None
        target_length = self.calculate_target_length(source)
        while end_of_sequence_symbol is not None:
            pass

class InputEmbeddingTensor:
    """
    Class which enables the embedding of tokens.

    :param vocab_size: The size of the vocabulary as int.
    :param embed_size: The size of the embedding units as int.
    """
    def __init__(self, vocab_size, embed_size):
        super(InputEmbeddingTensor, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.lookup_table_non_zero = nn.Embedding(vocab_size - 1, embed_size)
        init.xavier_uniform_(self.lookup_table_non_zero.weight)

    def embed(self, inputs):
        """
        In this method the first n tokens are embedded via look-up table.
        The n tokens serve as targets for the predictions.

        :param inputs: The train input values from batch
        :return: A embedded tensor of size n × 2d where d is the number of inner
                channels in the network
        """
        lookup_table_zero = torch.zeros(1, self.embed_size, device=inputs.device)

        lookup_table = torch.cat((lookup_table_zero, self.lookup_table_non_zero.weight), 0)

        return F.embedding(inputs, lookup_table)


