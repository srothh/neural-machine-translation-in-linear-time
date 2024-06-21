import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.data_loader import WMTLoader


# TODO hope this works lol
class Masked1DConvolution(nn.Conv1d):
    """
    Implementation of a masked 1D convolution layer.

    Params are the same as in nn.Conv1d from PyTorch.
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, groups=1, bias=True, device=None,
                 dtype=None):
        super(Masked1DConvolution, self).__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups,
                                                  bias, padding_mode='zeros')
        self.mask = torch.ones(self.weight.size())
        # How many 0 need to be added
        # by adding this many it is ensured that future values are not considered
        # for the computation
        # See: https://arxiv.org/pdf/1609.03499v2, https://arxiv.org/pdf/1610.10099
        # For less confusing Terminology : Masked Convolution == Causal Convolution
        receptive_field = (kernel_size - 1) * dilation
        # padding = (left, right)
        # No right padding, everything on the left
        self.padding = (receptive_field, 0)

    def forward(self, x):
        # Add padding to the input
        x = F.pad(x, self.padding)
        return super(Masked1DConvolution, self).forward(x)


class ResidualBlockReLu(nn.Module):
    """
    Implementation of residual Layer for Bytenet machine translation task.

    :param in_features: The number of input features.
    :param dilation: The initial dilation rate for the convolution layers.
    """

    def __init__(self, in_features, dilation, k, decoder = False):
        super(ResidualBlockReLu, self).__init__()
        self.layer_norm1 = nn.LayerNorm(2 * in_features)
        self.reLu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(2 * in_features, in_features, 1)
        self.layer_norm2 = nn.LayerNorm(in_features)
        self.reLu2 = nn.ReLU()
        # Masked kernel size is k
        # Dilation only used for masked convolution
        if decoder:
            self.conv2 = Masked1DConvolution(in_features, in_features, k, dilation=dilation)
        else:
            self.conv2 = nn.Conv1d(in_features, in_features, k, dilation=dilation)
        self.layer_norm3 = nn.LayerNorm(in_features)
        self.reLu3 = nn.ReLU()
        self.conv3 = nn.Conv1d(in_features, in_features * 2, 1)

    def forward(self, x):
        residual = x
        x = self.layer_norm1(x)
        x = self.reLu1(x)
        x = self.conv1(x)
        x = self.layer_norm2(x)
        x = self.reLu2(x)
        x = self.conv2(x)
        x = self.layer_norm3(x)
        x = self.reLu3(x)
        x = self.conv3(x)
        x += residual
        return x


# TODO: Everything, just a basic placeholder CNN encoder/decoder for now
class BytenetEncoder(nn.Module):
    def __init__(self, num_layers, in_features, kernel_size, max_dilation_rate, masked_kernel_size):
        super(BytenetEncoder, self).__init__()
        self.num_layers = num_layers
        self.num_channels = in_features
        self.kernel_size = kernel_size
        self.layers = nn.Sequential()
        dilation_rate = 1
        # From the Paper:
        # Model has a series of residual blocks of increased dilation rate
        # With unmasked convolutions for the encoder
        for i in range(num_layers):
            # Dilation Rate doubles each layer (starting out at 1)
            dilation_rate = dilation_rate * 2
            # Dilation rate does not exceed a given maximum
            # Example from the paper: 16
            self.layers.append(ResidualBlockReLu(in_features, dilation_rate if dilation_rate <= max_dilation_rate else max_dilation_rate, masked_kernel_size))
        # "the network applies one more convolution"
        # Note: The output of the residual layers is 2*input_features, however the output of the final convolutions is not specified in the paper
        # Experimentation needed if it should be 2*input_features or input_features
        self.layers.append(nn.Conv1d(in_features * 2, in_features, kernel_size))
        # "and ReLU"
        # Not sure if these last 2 layers should be in encoder or just decoder
        self.layers.append(nn.ReLU())
        # "followed by a convolution"
        self.layers.append(nn.Conv1d(in_features, in_features, kernel_size))
        # "and a final softmax layer" (probably not for encoder, however paper does not specify)
        # self.layers.append(nn.Softmax(dim=1))


    def forward(self, x):
        x = self.layers(x)
        return x


class BytenetDecoder(nn.Module):
    def __init__(self, num_layers, in_features, kernel_size, max_dilation_rate, masked_kernel_size):
        super(BytenetDecoder, self).__init__()
        self.num_layers = num_layers
        self.num_channels = in_features
        self.kernel_size = kernel_size
        self.layers = nn.Sequential()
        dilation_rate = 1
        # From the Paper:
        # Model has a series of residual blocks of increased dilation rate
        # With masekd convolution for decoder
        for i in range(num_layers):
            # Dilation Rate doubles each layer (starting out at 1)
            dilation_rate = dilation_rate * 2
            # Dilation rate does not exceed a given maximum
            # Example from the paper: 16
            self.layers.append(ResidualBlockReLu(in_features,
                                                 dilation_rate if dilation_rate <= max_dilation_rate else max_dilation_rate,
                                                 masked_kernel_size, decoder=True))
        # "the network applies one more convolution"
        # Note: The output of the residual layers is 2*input_features, however the output of the final convolutions is not specified in the paper
        # Experimentation needed if it should be 2*input_features or input_features
        self.layers.append(nn.Conv1d(in_features * 2, in_features, kernel_size))
        # "and ReLU"
        self.layers.append(nn.ReLU())
        # "followed by a convolution"
        self.layers.append(nn.Conv1d(in_features, in_features, kernel_size))
        # "and a final softmax layer"
        self.layers.append(nn.Softmax(dim=1))

    def forward(self, x):
        x = self.layers(x)
        return x


# Todo: might needs to be overworked
class EncoderDecoderStacking(nn.Module):
    """
    Stacks the encoder and decoder for the ByteNet model.
    This means passing the output of the encoder as input to the decoder.

    :param encoder_layers: The number of layers in the encoder as int.
    :param decoder_layers: The number of layers in the decoder as int.
    :param num_channels: The number of channels in the network as int.
    :param kernel_size: The size of the kernel as int.
    :param dilation_rate: The dilation rate as int.

    :return x: The output of the decoder.
    """
    def __init__(self, encoder_layers, decoder_layers, num_channels, kernel_size, dilation_rate):
        super(EncoderDecoderStacking, self).__init__()
        self.encoder = BytenetEncoder(encoder_layers, num_channels, kernel_size, dilation_rate)
        self.decoder = BytenetDecoder(decoder_layers, num_channels, kernel_size, dilation_rate)
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
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
