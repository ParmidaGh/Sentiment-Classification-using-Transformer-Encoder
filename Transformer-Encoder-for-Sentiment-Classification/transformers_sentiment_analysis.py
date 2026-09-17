"""
Implements a Transformer in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
# This code was provided as a ready-to-use template.
# Some parts were intentionally left incomplete and required implementation.

from torch.nn import functional as F
import math
import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F

def hello_transformers():
    print("Hello from transformers_sentiment_analysis.py!")

def scaled_dot_product_two_loop_single(
    query: Tensor, key: Tensor, value: Tensor
) -> Tensor:
    """
    The function performs a fundamental block for attention mechanism, the scaled
    dot product. We map the input query, key, and value to the output. Follow the
    description in TODO for implementation.

    args:
        query: a Tensor of shape (K, M) where K is the sequence length and M is
            the sequence embeding dimension

        key: a Tensor of shape (K, M) where K is the sequence length and M is the 
            sequence embeding dimension

        value: a Tensor of shape (K, M) where K is the sequence length and M is 
            the sequence embeding dimension
             

    Returns
        out: a tensor of shape (K, M) which is the output of self-attention from
        the function
    """
    # make a placeholder for the output
    out = None
    ###############################################################################
    # TODO: Implement this function using exactly two for loops. For each of the  #
    # K queries, compute its dot product with each of the K keys. The scalar      #
    # output of the dot product will the be scaled by dividing it with the sqrt(M)#
    # Once we get all the K scaled weights corresponding to a query, we apply a   #
    # softmax function on them and use the value matrix to compute the weighted   #
    # sum of values using the matrix-vector product. This single vector computed  #
    # using weighted sum becomes an output to the Kth query vector                #
    ###############################################################################

    K, M = query.size()  # sequence len and embed dim

    for i in range(K):  # through each query vector
        weights = torch.zeros(K)

        for j in range(K):  # through each key vector
            # dot product (query and key), scale by dividing sqrt(M)
            dot_product = torch.dot(query[i], key[j]) / (M ** 0.5)

            # scaled weights using softmax
            weights[j] = dot_product
        weights = torch.softmax(weights, dim=0)
        # weighted sum of values for query
        weighted_sum = torch.matmul(weights, value)

        # construct outputs
        if out is None:
            out = weighted_sum.unsqueeze(0)
        else:
            out = torch.cat((out, weighted_sum.unsqueeze(0)), dim=0)

    ##############################################################################
    #               END OF COMPLETED CODE                                             #
    ##############################################################################
    return out

def scaled_dot_product_two_loop_batch(
    query: Tensor, key: Tensor, value: Tensor
) -> Tensor:

    """
    The function performs a fundamental block for attention mechanism, the scaled
    dot product. We map the input query, key, and value to the output. Follow the
    description in TODO for implementation.

    args:
        query: a Tensor of shape (N,K, M) where N is the batch size, K is the 
            sequence length and  M is the sequence embeding dimension

        key: a Tensor of shape (N, K, M) where N is the batch size, K is the 
            sequence length and M is the sequence embeding dimension
             

        value: a Tensor of shape (N, K, M) where N is the batch size, K is the 
            sequence length and M is the sequence embeding dimension
             

    Returns:
        out: a tensor of shape (N, K, M) that contains the weighted sum of values
           

    """
    # make a placeholder for the output
    out = None
    N, K, M = query.shape
    ###############################################################################
    # TODO: This function is extedning self_attention_two_loop_single for a batch #
    # of N. Implement this function using exactly two for loops. For each N       #
    # we have a query, key and value. The final output is the weighted sum of     #
    # values of these N queries and keys. The weight here is computed using scaled#
    # dot product  between each of the K queries and key. The scaling value here  #
    # is sqrt(M). For each of the N sequences, compute the softmaxed weights and  #
    # use them to compute weighted average of value matrix.                       #
    # Hint: look at torch.bmm                                                     #
    ###############################################################################
    
    for n in range(N):  # through each sequence in batch
        weighted_sum = None
        
        for i in range(K):
            weights = torch.zeros(K)
            
            for j in range(K):
                dot_product = torch.dot(query[n, i], key[n, j]) / (M ** 0.5)
                
                weights[j] = dot_product
            weights = torch.softmax(weights, dim=0)           
            weighted_value = torch.matmul(weights, value[n])
            
            if weighted_sum is None:
                weighted_sum = weighted_value.unsqueeze(0)
            else:
                weighted_sum = torch.cat((weighted_sum, weighted_value.unsqueeze(0)), dim=0)
        
        # assigne computed weighted sum to output
        if out is None:
            out = weighted_sum.unsqueeze(0)
        else:
            out = torch.cat((out, weighted_sum.unsqueeze(0)), dim=0)

    ##############################################################################
    #               END OF COMPLETED CODE                                             #
    ##############################################################################
    return out

def scaled_dot_product_no_loop_batch(
    query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None
) -> Tensor:
    """

    The function performs a fundamental block for attention mechanism, the scaled
    dot product. We map the input query, key, and value to the output. It uses
    Matrix-matrix multiplication to find the scaled weights and then matrix-matrix
    multiplication to find the final output.

    args:
        query: a Tensor of shape (N,K, M) where N is the batch size, K is the 
            sequence length and M is the sequence embeding dimension

        key:  a Tensor of shape (N, K, M) where N is the batch size, K is the 
            sequence length and M is the sequence embeding dimension
             

        value: a Tensor of shape (N, K, M) where N is the batch size, K is the 
            sequence length and M is the sequence embeding dimension

             
        mask: a Bool Tensor of shape (N, K, K) that is used to mask the weights
            used for computing weighted sum of values
              

    return:
        y: a tensor of shape (N, K, M) that contains the weighted sum of values
           
        weights_softmax: a tensor of shape (N, K, K) that contains the softmaxed
            weight matrix.

    """

    _, _, M = query.shape
    y = None
    weights_softmax = None
    ###############################################################################
    # TODO: This function performs same function as self_attention_two_loop_batch #
    # Implement this function using no loops.                                     #
    # For the mask part, you can ignore it for now and revisit it in the later part.
    # Given the shape of the mask is (N, K, K), and it is boolean with True values#
    # indicating  the weights that have to be masked and False values indicating  #
    # the weghts that dont need to be masked at that position. These masked-scaled#
    # weights can then be softmaxed to compute the final weighted sum of values   #
    # Hint: look at torch.bmm and torch.masked_fill                               #
    ###############################################################################

    N,K,M=query.shape
    scaled=query.bmm(key.swapaxes(1,2))/(M**0.5)
    if mask is not None:
        ##########################################################################
        # TODO: Apply the mask to the weight matrix by assigning -1e9 to the     #
        # positions where the mask value is True, otherwise keep it as it is.    #
        ##########################################################################

        scaled[mask==True]=-1e9
    
        # scaled = scaled.masked_fill(mask, -1e9)

    weights_softmax = torch.softmax(scaled, dim=-1)
    y = weights_softmax.bmm(value)

    ##############################################################################
    #               END OF COMPLETED CODE                                             #
    ##############################################################################
    return y, weights_softmax

class SelfAttention(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_v: int):
        super().__init__()

        """
        This class encapsulates the implementation of self-attention layer. We map 
        the input query, key, and value using MLP layers and then use 
        scaled_dot_product_no_loop_batch to the final output.
        
        args:
            dim_in: an int value for input sequence embedding dimension
            dim_q: an int value for output dimension of query and ley vector
            dim_v: an int value for output dimension for value vectors

        """
        self.q = None  # initialize for query
        self.k = None  # initialize for key
        self.v = None  # initialize for value
        self.weights_softmax = None
        ##########################################################################
        # TODO: This function initializes three functions to transform the 3 input
        # sequences to key, query and value vectors. More precisely, initialize  #
        # three nn.Linear layers that can transform the input with dimension     #
        # dim_in to query with dimension dim_q, key with dimension dim_q, and    #
        # values with dim_v. For each Linear layer, use the following strategy to#
        # initialize the weights:                                                #
        # If a Linear layer has input dimension D_in and output dimension D_out  #
        # then initialize the weights sampled from a uniform distribution bounded#
        # by [-c, c]                                                             #
        # where c = sqrt(6/(D_in + D_out))                                       #
        # Please use the same names for query, key and value transformations     #
        # as given above. self.q, self.k, and self.v respectively.               #
        ##########################################################################

        import numpy as np
        c = np.sqrt(6 / (dim_in + dim_q))

        # linear layers for query, key and value transformations
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_q)
        self.v = nn.Linear(dim_in, dim_v)
        # weights for linear layers
        self.q.weight.data.uniform_(-c, c)
        self.k.weight.data.uniform_(-c, c)
        self.v.weight.data.uniform_(-c, c)

        ##########################################################################
        #               END OF COMPLETED CODE                                         #
        ##########################################################################

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None
    ) -> Tensor:

        """
        An implementation of the forward pass of the self-attention layer.

        args:
            query: Tensor of shape (N, K, M)
            key: Tensor of shape (N, K, M)
            value: Tensor of shape (N, K, M)
            mask: Tensor of shape (K, M)
        return:
            y: Tensor of shape (N, K, dim_v)
        """
        self.weights_softmax = (
            None  # weight matrix after applying self_attention_no_loop_batch
        )
        y = None
        ##########################################################################
        # TODO: Use the functions initialized in the init fucntion to find the   #
        # output tensors. Precisely, pass the inputs query, key and value to the #
        #  three functions iniitalized above. Then, pass these three transformed #
        # query,  key and value tensors to the self_attention_no_loop_batch to   #
        # get the final output. For now, dont worry about the mask and just      #
        # pass it as a variable in self_attention_no_loop_batch. Assign the value#
        # of output weight matrix from self_attention_no_loop_batch to the       #
        # variable self.weights_softmax                                          #
        ##########################################################################

        # find the output tensors
        query_transformed = self.q(query)
        key_transformed = self.k(key)
        value_transformed = self.v(value)
        # pass to scaled_dot_product_no_loop_batch
        y, self.weights_softmax = scaled_dot_product_no_loop_batch(
            query_transformed, key_transformed, value_transformed, mask
        )

        ##########################################################################
        #               END OF COMPLETED CODE                                         #
        ##########################################################################
        return y

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_out: int):
        super().__init__()

        """
        
        A naive implementation of the MultiheadAttention layer for Transformer model.
        We use multiple SelfAttention layers parallely on the input and then concat
        them to into a single tensor. This Tensor is then passed through an MLP to 
        generate the final output. The input shape will look like (N, K, M) where  
        N is the batch size, K is the batch size and M is the sequence embedding  
        dimension.
        args:
            num_heads: int value specifying the number of heads
            dim_in: int value specifying the input dimension of the query, key
                and value. This will be the input dimension to each of the
                SingleHeadAttention blocks
            dim_out: int value specifying the output dimension of the complete 
                MultiHeadAttention block

        NOTE: Here, when we say dimension, we mean the dimesnion of the embeddings.
              In Transformers the input is a tensor of shape (N, K, M), here N is
              the batch size , K is the sequence length and M is the size of the
              input embeddings. As the sequence length(K) and number of batches(N)
              don't change usually, we mostly transform
              the dimension(M) dimension.

        """

        ##########################################################################
        # TODO: Initialize two things here:                                      #
        # 1.) Use nn.ModuleList to initialze a list of SingleHeadAttention layer #
        # modules.The length of this list should be equal to num_heads with each #
        # SingleHeadAttention layer having input dimension as dim_in, and query  #
        # , key, and value dimension as emb_out.                                 #
        # 2.) Use nn.Linear to map the output of nn.Modulelist block back to     #
        # dim_in                                                                 #
        ##########################################################################

        # list of SelfAttention layer modules
        self.heads = nn.ModuleList([
            SelfAttention(dim_in, dim_out, dim_out) for _ in range(num_heads)
        ])
        # linear layer for map output to dim_in
        self.linear = nn.Linear(num_heads*dim_out, dim_in)

        ##########################################################################
        #               END OF COMPLETED CODE                                         #
        ##########################################################################

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None
    ) -> Tensor:

        """
        An implementation of the forward pass of the MultiHeadAttention layer.

        args:
            query: Tensor of shape (N, K, M)
            key: Tensor of shape (N, K, M)
            value: Tensor of shape (N, K, M)

        returns:
            y: Tensor of shape (N, K, M)
        """
        y = None
        ##########################################################################
        # TODO: You need to perform a forward pass through the MultiHeadAttention#
        # block using the variables defined in the initializing function. The    #
        # nn.ModuleList behaves as a list and you could use a for loop or list   #
        # comprehension to extract different elements of it. Each of the elements#
        # inside nn.ModuleList is a SingleHeadAttention that  will take the same #
        # query, key and value tensors and you will get a list of tensors as     #
        # output. Concatenate this list if tensors and pass them through the     #
        # nn.Linear mapping function defined in the initialization step.         #
        ##########################################################################

        # forward pass through list of SelfAttention
        outputs = [head(query, key, value, mask) for head in self.heads]
        # concat list of output tensors
        concat = torch.cat(outputs, dim=-1)
        # pass through linear mapping function
        y = self.linear(concat)

        ##########################################################################
        #               END OF COMPLETED CODE                                         #
        ##########################################################################
        return y



class LayerNormalization(nn.Module):
    def __init__(self, emb_dim: int, epsilon: float = 1e-10):
        super().__init__()
        """
        The class implements the Layer Normalization for Linear layers in 
        Transformers.  Unlike BathcNorm ,it estimates the normalization statistics 
        for each element present in the batch and hence does not depend on the  
        complete batch.
        The input shape will look something like (N, K, M) where N is the batch 
        size, K is the sequence length and M is the sequence length embedding. We 
        compute the  mean with shape (N, K) and standard deviation with shape (N, K) 
        and use them to normalize each sequence.
        
        args:
            emb_dim: int representing embedding dimension
            epsilon: float value

        """

        self.epsilon = epsilon

        ##########################################################################
        # TODO: Initialize the scale and shift parameters for LayerNorm.         #
        # Initialize the scale parameters to all ones and shift parameter to all #
        # zeros. As we have seen in the lecture, the shape of scale and shift    #
        # parameters remains the same as in Batchnorm, initialize these parameters
        # with appropriate dimensions                                            #
        ##########################################################################

        # scale and shift parameters
        self.gamma = nn.Parameter(torch.ones(emb_dim))
        self.beta = nn.Parameter(torch.zeros(emb_dim))

        ##########################################################################
        #               END OF COMPLETED CODE                                         #
        ##########################################################################

    def forward(self, x: Tensor):
        """
        An implementation of the forward pass of the Layer Normalization.

        args:
            x: a Tensor of shape (N, K, M) where N is the batch size, K is the
               sequence length and M is the embedding dimension
               
        returns:
            y: a Tensor of shape (N, K, M) after applying layer normalization
               
        """
        y = None
        ##########################################################################
        # TODO: Implement the forward pass of the LayerNormalization layer.      #
        # Compute the mean and standard deviation of input and use these to      #
        # normalize the input. Further, use self.gamma and self.beta to scale    #
        # these and shift this normalized input                                  #
        ##########################################################################

        # mean and standard deviation of samples
        mean = x.mean(dim=1, keepdim=True) # keep dimensions for broadcast
        std = x.std(dim=1, unbiased=False, keepdim=True)               
        x_normalized = (x - mean) / (std + self.epsilon) # normalize input
        # apply scale and shift parameters
        y = self.gamma * x_normalized + self.beta

        ##########################################################################
        #               END OF COMPLETED CODE                                         #
        ##########################################################################
        return y


class FeedForwardBlock(nn.Module):
    def __init__(self, inp_dim: int, hidden_dim_feedforward: int):
        super().__init__()

        """
        An implementation of the FeedForward block in the Transformers. We pass  
        the input through stacked 2 MLPs and 1 ReLU layer. The forward pass has  
        following architecture:
        
        linear - relu -linear
        
        The input will have a shape of (N, K, M) where N is the batch size, K is 
        the sequence length and M is the embedding dimension. 
        
        args:
            inp_dim: int representing embedding dimension of the input tensor
                     
            hidden_dim_feedforward: int representing the hidden dimension for
                the feedforward block
        """

        ##########################################################################
        # TODO: initialize two MLPs here with the first one using inp_dim as input
        # dimension and hidden_dim_feedforward as output and the second with     #
        # hidden_dim_feedforward as input. You should figure out the output      #
        # dimesion of the second MLP.                                            #
        # HINT: Will the shape of input and output shape of the FeedForwardBlock #
        # change?                                                                #
        ##########################################################################

        self.linear1 = nn.Linear(inp_dim, hidden_dim_feedforward)
        self.linear2 = nn.Linear(hidden_dim_feedforward, inp_dim)

        ##########################################################################
        #               END OF COMPLETED CODE                                         #
        ##########################################################################

    def forward(self, x):
        """
        An implementation of the forward pass of the FeedForward block.

        args:
            x: a Tensor of shape (N, K, M) which is the output of 
               MultiHeadAttention
        returns:
            y: a Tensor of shape (N, K, M)
        """
        y = None
        ###########################################################################
        # TODO: Use the two MLP layers initialized in the init function to perform#
        # a forward pass. You should be using a ReLU layer after the first MLP and#
        # no activation after the second MLP                                      #
        ###########################################################################

        y = F.relu(self.linear1(x))
        y = self.linear2(y)

        ##########################################################################
        #               END OF COMPLETED CODE                                         #
        ##########################################################################
        return y


class EncoderBlock(nn.Module):
    def __init__(
        self, num_heads: int, emb_dim: int, feedforward_dim: int, dropout: float
    ):
        super().__init__()
        """
        This class implements the encoder block for the Transformer model, the 
        original paper used 6 of these blocks sequentially to train the final model. 
        Here, we will first initialize the required layers using the building  
        blocks we have already  implemented, and then finally write the forward     
        pass using these initialized layers, residual connections and dropouts.        
        
        As shown in the Figure 1 of the paper attention is all you need
        https://arxiv.org/pdf/1706.03762.pdf, the encoder consists of four components:
        
        1. MultiHead Attention
        2. FeedForward layer
        3. Residual connections after MultiHead Attention and feedforward layer
        4. LayerNorm
        
        The architecture is as follows:
        
       inp - multi_head_attention - out1 - layer_norm(out1 + inp) - dropout - out2 \ 
        - feedforward - out3 - layer_norm(out3 + out2) - dropout - out
        
        Here, inp is input of the MultiHead Attention of shape (N, K, M), out1, 
        out2 and out3 are the outputs of the corresponding layers and we add these 
        outputs to their respective inputs for implementing residual connections.

        args:
            num_heads: int value specifying the number of heads in the
                MultiHeadAttention block of the encoder

            emb_dim: int value specifying the embedding dimension of the input
                sequence

            feedforward_dim: int value specifying the number of hidden units in the 
                FeedForward layer of Transformer

            dropout: float value specifying the dropout value


        """

        if emb_dim % num_heads != 0:
            raise ValueError(
                f"""The value emb_dim = {emb_dim} is not divisible
                             by num_heads = {num_heads}. Please select an
                             appropriate value."""
            )

        ##########################################################################
        # TODO: Initialize the following layers:                                 #
        # 1. One MultiHead Attention block using num_heads as number of heads and#
        #    emb_dim as the input dimension. You should also be able to compute  #
        #    the output dimension of MultiheadHead attention given num_heads and #
        #    emb_dim.                                                            #
        #    Hint: use the logic that you concatenate the output from each       #
        #    SingleHeadAttention inside the MultiHead Attention block and choose #
        #    the output dimension such that the concatenated tensor and the input#
        #    tensor have the same embedding dimension.                           #
        #                                                                        #
        # 2. Two LayerNorm layers with input dimension equal to emb_dim          #
        # 3. One feedForward block taking input as emb_dim and hidden units as   #
        #    feedforward_dim                                                     #
        # 4. A Dropout layer with given dropout parameter                        #
        ##########################################################################

        # multiHead-attention block
        self.multihead_attention = MultiHeadAttention(num_heads, emb_dim, emb_dim)
        # LayerNorm layers
        self.norm1 = LayerNormalization(emb_dim)
        self.norm2 = LayerNormalization(emb_dim)
        # feedForward block
        self.feedforward = FeedForwardBlock(emb_dim, feedforward_dim)
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        ##########################################################################
        #               END OF COMPLETED CODE                                         #
        ##########################################################################

    def forward(self, x):

        """

        An implementation of the forward pass of the EncoderBlock of the
        Transformer model.
        args:
            x: a Tensor of shape (N, K, M) as input sequence
        returns:
            y: a Tensor of shape (N, K, M) as the output of the forward pass
        """
        y = None
        ##########################################################################
        # TODO: Use the layer initialized in the init function to complete the   #
        # forward pass. As Multihead Attention takes in 3 inputs, use the same   #
        # input thrice as the input. Follow the Figure 1 in Attention is All you #
        # Need paper to complete the rest of the forward pass. You can also take #
        # reference from the architecture written in the fucntion documentation. #
        ##########################################################################
      
        attn_output = self.multihead_attention(x, x, x) # multiHead-attention block       
        attn_output = self.norm1(x + attn_output)  # add & norm 1        
        attn_output = self.dropout(attn_output)  # apply dropout        
        feedforward_output = self.feedforward(attn_output)  # feedforward block       
        output = self.norm2(attn_output + feedforward_output)  # add & norm 2             
        y = self.dropout(output)  # dropout

        ##########################################################################
        #               END OF COMPLETED CODE                                         #
        ##########################################################################
        return y



class Encoder(nn.Module):
    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        feedforward_dim: int,
        num_layers: int,
        dropout: float,
    ):
        """
        The class encapsulates the implementation of the final Encoder that use
        multiple EncoderBlock layers.

        args:
            num_heads: int representing number of heads to be used in the
                EncoderBlock
            emb_dim: int repreesenting embedding dimension for the Transformer
                model
            feedforward_dim: int representing hidden layer dimension for the
                feed forward block

        """

        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderBlock(num_heads, emb_dim, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, src_seq: Tensor):
        for _layer in self.layers:
            src_seq = _layer(src_seq)

        return src_seq




def position_encoding_simple(K: int, M: int) -> Tensor:
    """
    An implementation of the simple positional encoding using uniform intervals
    for a sequence.

    args:
        K: int representing sequence length
        M: int representing embedding dimension for the sequence
           
    return:
        y: a Tensor of shape (1, K, M)
    """
    y = None
    ##############################################################################
    # TODO: Given the length of input sequence K, construct a 1D Tensor of length#
    # K with nth element as n/K, where n starts from 0. Replicate this tensor M  #
    # times to create a tensor of the required output shape                      #
    ##############################################################################

    pos = torch.arange(K).unsqueeze(1) / K
    y = pos.repeat(1, M).unsqueeze(0)

    ##############################################################################
    #               END OF COMPLETED CODE                                             #
    ##############################################################################
    return y



def position_encoding_sinusoid(K: int, M: int) -> Tensor:

    """
    An implementation of the sinousoidal positional encodings.

    args:
        K: int representing sequence length
        M: int representing embedding dimension for the sequence
           
    return:
        y: a Tensor of shape (1, K, M)

    """
    y = None
    ##############################################################################
    # TODO: Given the length of input sequence K and embedding dimension M       #
    # construct a tesnor of shape (K, M) where the value along the dimensions    #
    # follow the equations given in the notebook. Make sure to keep in mind the  #
    # alternating sines and cosines along the embedding dimension M.             #
    ##############################################################################
    
    temp=torch.arange(M).reshape(1,M)
    a = 2*torch.floor(temp / M)
    p = torch.arange(K).reshape(K,1)

    y = torch.zeros(1,K, M)
    y[:,:, 0::2] = torch.sin(p / torch.pow(10000, a[0, 0::2]))
    y[:,:, 1::2] = torch.cos(p / torch.pow(10000, a[0, 1::2]))
    return y


class Transformer_encoder(nn.Module):
    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        feedforward_dim: int,
        dropout: float,
        num_enc_layers: int,
        vocab_len: int,
        n_classes: int
    ):
        super().__init__()

        self.emb_layer = nn.Embedding(vocab_len,emb_dim)

        ##############################################################################

        self.position_encoding = position_encoding_sinusoid
        self.avg_pool = nn.AdaptiveAvgPool1d(1) # average pooling
        self.fc = nn.Linear(emb_dim, n_classes) # fc Layer for classification

        # init encoder layers
        self.encoder = Encoder(
            num_heads=num_heads,
            emb_dim=emb_dim,
            feedforward_dim=feedforward_dim,
            num_layers=num_enc_layers,
            dropout=dropout
        )

        ##############################################################################
        


    def forward(self, ques_b) -> Tensor:
      
    ##############################################################################

        # pass through embedding layer
        x = self.emb_layer(ques_b)
        # add position encoding
        N, K = x.shape[0], x.shape[1]
        position_encoding = self.position_encoding(K, self.emb_layer.embedding_dim).to(x.device)
        x = x + position_encoding[:, :K, :]

        x = self.encoder(x) # pass through encoder
        x_avg_pool = self.avg_pool(x.permute(0, 2, 1)).squeeze(2) # apply average pooling
        y = self.fc(x_avg_pool) # classification

        return y

    ##############################################################################
    #               END OF COMPLETED CODE                                             #
    ##############################################################################




