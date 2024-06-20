import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.data_loader import WMTLoader


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
        Calculates the target length for dynamic unfolding with the
        formula: |t| = a|s| + b
        Saves unnecessary computing and is more memory efficient

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
        The unfolding works as for the example "Hi all" as follows:
        Decoder input: start_token => Decoder output: Hi
        Decoder input: Hi          => Decoder output: all
        Decoder input: Hi all      => Decoder output: end_of_sequence

        :param encoder_output: The output of the encoder model as vector representing a sequence used for unfolding
        :param decoder: The decoder model
        :param source: The source, used for calculating the target length
        """
        batch_size = encoder_output.size(1)
        hidden_state = None
        end_of_sequence_symbol = 0
        output_sequence = []
        target_length = self.calculate_target_length(source)

        # The first element used as the start token
        decoder_input = torch.zeros((batch_size, 1, encoder_output.size(2)), device=encoder_output.device)

        # Loop until reach the computed target length
        for _ in range(target_length):
            decoder_output = decoder(decoder_input)

            # Get the last token
            last_token = decoder_output[:, -1, :]

            # From the output sequence, append the last token into it
            output_sequence.append(last_token)

            # Concatenates the given sequence of seq tensors in the given dimension of
            # the decoder input and the last token. The decoder input is initialized with the concatenated values
            decoder_input = torch.cat((decoder_input, last_token.unsqueeze(1)), dim=1)

            # Check if the last token is the end of sequence token, in this case the 0 is used
            # TODO: eventually this should be changed, used 0 as default
            if torch.argmax(last_token, dim=-1).item() == end_of_sequence_symbol:
                break

        return output_sequence


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
        # This is the actual lookup table.
        # A lookup table is an array of data that maps input values to output values
        self.lookup_table_non_zero = nn.Embedding(vocab_size - 1, embed_size)
        init.xavier_uniform_(self.lookup_table_non_zero.weight)

    def embed(self, inputs):
        """
        In this method the first n tokens are embedded via look-up table.
        The n tokens serve as targets for the predictions.

        :param inputs: The train input values from batch, more exact: the tokens
        :return: A embedded tensor of size n × 2d where d is the number of inner
                channels in the network
        """
        lookup_table_zero = torch.zeros(1, self.embed_size, device=inputs.device)
        # Here the both look up tables are combined. The rows with the zeros and the rows
        # with values from the actual lookup table are combined therefore
        lookup_table = torch.cat((lookup_table_zero, self.lookup_table_non_zero.weight), 0)
        # Next the input ids are embedded into the lookup table, which means that each id has it own
        # embedding-vector, f.e:
        # id: 5 => [1,5,4]; id:7 => [3,2,9]
        # The input ids are the tokens
        # If a token sequence of 5;7 is used, the resulting matrix is:
        # [1,5,4],[3,2,9]
        return F.embedding(inputs, lookup_table)

if __name__ == '__main__':
    cache_dir = 'D:/wmt19_cache'
    # wmt_loader = WMTLoader(split="train", cache_dir=cache_dir)
    # index = 0
    # source, target = wmt_loader[index]
    # print("Source:", source)
    # print("Target:", target)

