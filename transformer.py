import tensorflow as tf

def scaled_dot_product(q, k, v, mask=None):
    """
    q: query vector having shape (batch_size, num_heads, seq_len, head_dim)
    k: key vector having shape (batch_size, num_heads, seq_len, head_dim)
    v: value vector having shape (batch_size, num_heads, seq_len, head_dim)
    mask: mask vector having shape (batch_size, num_heads, seq_len, seq_len)
    """
    d_k = q.shape[-1]
    k = tf.transpose(k, perm=[0,1,3,2])
    scaled = tf.matmul(q,k)/tf.math.sqrt(float(d_k))
    if mask is not None:
        mask = tf.expand_dims(mask, axis=1)
        scaled += mask
    attention = tf.nn.softmax(scaled, axis=-1)
    values = tf.matmul(attention, v)
    return values, attention


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = input_dim//num_heads
        self.qkv_layer = tf.keras.layers.Dense(3*input_dim, activation="linear")
        self.linear_layer = tf.keras.layers.Dense(output_dim, activation="linear")
    
    def call(self, x, mask=None):
        batch_size, seq_len, input_dim = x.shape
        x = self.qkv_layer(x)
        x = tf.reshape(x, [batch_size, seq_len, self.num_heads, 3*self.head_dim])
        x = tf.transpose(x, perm=[0,2,1,3])
        q, k, v = tf.split(x, num_or_size_splits=3, axis=-1)
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = tf.reshape(values, [batch_size, seq_len, self.num_heads*self.head_dim])
        out = self.linear_layer(values)
        return out

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_sequence_length, d_model):
        super(PositionalEncoding,self).__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        
    def call(self, x):
        batch_size = x.shape[0]
        pos = tf.reshape(tf.range(0,self.max_sequence_length, dtype=tf.float32), [-1,1])
        dim_index = tf.range(0, self.d_model,2, dtype=tf.float32)
        denominator = tf.pow(10000, dim_index/self.d_model)
        even_PE = tf.sin(pos/denominator)
        odd_PE = tf.cos(pos/denominator)
        out = tf.reshape(tf.stack([even_PE, odd_PE],axis = 2),(1, self.max_sequence_length, self.d_model))
        out = tf.tile(out, [batch_size, 1, 1])
        return out+x

class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, parameter_shape, eps=1e-5):
        super(LayerNormalization,self).__init__()
        self.parameter_shape = parameter_shape
        self.eps = eps
        self.gamma = self.add_weight("gamma", shape=parameter_shape, dtype=tf.float32, initializer="ones", trainable=True)
        self.beta = self.add_weight("beta", shape=parameter_shape, dtype=tf.float32, initializer="zeros", trainable=True)
        
    def call(self, x):
        dims = [-(i+1) for i in range(len(self.parameter_shape))]
        mean = tf.reduce_mean(x, axis=dims, keepdims=True)
        var = tf.math.reduce_variance(x, axis = dims, keepdims=True)
        std = tf.sqrt(var + self.eps)
        y = (x-mean)/std
        out = self.gamma * y + self.beta
        return out

class PositionWiseFFN(tf.keras.layers.Layer):
    def __init__(self, emb_dim, ffn_hidden, dropout):
        super(PositionWiseFFN, self).__init__()
        self.emb_dim = emb_dim
        self.ffn_hidden = ffn_hidden
        self.dropout = dropout
        self.layer1 = tf.keras.layers.Dense(ffn_hidden, activation="relu")
        self.layer2 = tf.keras.layers.Dense(emb_dim)
        self.dropout = tf.keras.layers.Dropout(rate=dropout)
    def call(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, emb_dim, ffn_hidden, num_heads, dropout):
        super(EncoderLayer,self).__init__()
        self.emb_dim = emb_dim
        self.ffn_hidden = ffn_hidden
        self.num_heads = num_heads
        self.dropout = dropout
        self.mha = MultiHeadAttention(input_dim=emb_dim, output_dim=emb_dim, num_heads=num_heads)
        self.dropout1 = tf.keras.layers.Dropout(rate=dropout)
        self.layernorm1 = LayerNormalization(parameter_shape=[emb_dim])
        self.ffn = PositionWiseFFN(emb_dim, ffn_hidden, dropout)
        self.dropout2 = tf.keras.layers.Dropout(rate=dropout)
        self.layernorm2 = LayerNormalization(parameter_shape=[emb_dim])

    def call(self, x, self_attention_mask):
        #x of dim batchs_size x seq_len x emb_dim
        residual_x = x
        x = self.mha(x, mask=self_attention_mask)
        x = self.dropout1(x)
        x = self.layernorm1(x + residual_x)
        residual_x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.layernorm2(x + residual_x)
        return x

class Encoder(tf.keras.layers.Layer):
    def __init__(self, emb_dim, ffn_hidden, num_heads, dropout, num_layers):
        super(Encoder,self).__init__()
        self.emb_dim = emb_dim
        self.ffn_hidden = ffn_hidden
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = num_layers
        
    def call(self, x, self_attention_mask):
        layers = [EncoderLayer(self.emb_dim, self.ffn_hidden, self.num_heads, self.dropout) for _ in range(self.num_layers)]
        for layer in layers:
            x = layer(x, self_attention_mask)
        return x

class MultiHeadCrossAttention(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = input_dim//num_heads
        self.q_layer = tf.keras.layers.Dense(input_dim, activation="linear")
        self.kv_layer = tf.keras.layers.Dense(2*input_dim, activation="linear")
        self.linear_layer = tf.keras.layers.Dense(output_dim, activation="linear")
        
    def call(self, x, y, mask = None):
        batch_size, seq_len, input_dim = x.shape
        kv = self.kv_layer(x)
        q = self.q_layer(y)
        kv = tf.reshape(kv , [batch_size, seq_len, self.num_heads, 2*self.head_dim])
        q = tf.reshape(q, [batch_size, seq_len, self.num_heads, self.head_dim])
        kv = tf.transpose(kv, perm= [0, 2, 1, 3])
        q = tf.transpose(q, perm = [0, 2, 1, 3])
        k, v = tf.split(kv, num_or_size_splits = 2, axis = -1)
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = tf.reshape(values, [batch_size, seq_len, self.num_heads*self.head_dim])
        out = self.linear_layer(values)
        return out
        

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, emb_dim, ffn_hidden, num_heads, dropout):
        super(DecoderLayer, self).__init__()
        self.emb_dim = emb_dim
        self.ffn_hidden = ffn_hidden
        self.num_heads = num_heads
        self.dropout = dropout
        self.mha = MultiHeadAttention(input_dim=emb_dim, output_dim=emb_dim, num_heads=num_heads)
        self.dropout1 = tf.keras.layers.Dropout(rate=dropout)
        self.layernorm1 = LayerNormalization(parameter_shape=[emb_dim])
        self.mhca = MultiHeadCrossAttention(input_dim=emb_dim, output_dim=emb_dim, num_heads=num_heads)
        self.dropout2 = tf.keras.layers.Dropout(rate=dropout)
        self.layernorm2 = LayerNormalization(parameter_shape=[emb_dim])
        self.ffn = PositionWiseFFN(emb_dim, ffn_hidden, dropout)
        self.dropout3 = tf.keras.layers.Dropout(rate=dropout)
        self.layernorm3 = LayerNormalization(parameter_shape=[emb_dim])
    
    def call(self, x, y, self_attention_mask, cross_attention_mask):
        residual_y = y
        y = self.mha(y, mask=self_attention_mask)
        y = self.dropout1(y)
        y = self.layernorm1(y + residual_y)
        residual_y = y
        y = self.mhca(x, y, mask=cross_attention_mask)
        y = self.dropout2(y)
        y = self.layernorm2(y + residual_y)
        residual_y = y
        y = self.ffn(y)
        y = self.dropout3(y)
        y = self.layernorm3(y + residual_y)
        return y

class Decoder(tf.keras.layers.Layer):
    def __init__(self, emb_dim, ffn_hidden, num_heads, dropout, num_layers):
        super(Decoder,self).__init__()
        self.emb_dim = emb_dim
        self.ffn_hidden = ffn_hidden
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = num_layers
        self.layers = [DecoderLayer(self.emb_dim, self.ffn_hidden, self.num_heads, self.dropout) for _ in range(self.num_layers)]
    
    def call(self, x, y, self_attention_mask, cross_attention_mask):
        for layer in self.layers:
            y = layer(x, y, self_attention_mask, cross_attention_mask)
        return y

class Transformer(tf.keras.layers.Layer):
    def __init__(self, seq_len, target_vocab_size, emb_dim, ffn_hidden, num_heads, dropout, num_layers):
        super(Transformer, self).__init__()
        self.seq_len = seq_len
        self.target_vocab_size = target_vocab_size
        self.emb_dim = emb_dim
        self.ffn_hidden = ffn_hidden
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = num_layers
        self.x_PE = PositionalEncoding(self.seq_len, self.emb_dim)
        self.y_PE = PositionalEncoding(self.seq_len, self.emb_dim)
        self.x_dropout = tf.keras.layers.Dropout(rate = self.dropout)
        self.y_dropout = tf.keras.layers.Dropout(rate = self.dropout)
        self.encoder = Encoder(self.emb_dim, self.ffn_hidden, self.num_heads, self.dropout, self.num_layers)
        self.decoder = Decoder(self.emb_dim, self.ffn_hidden, self.num_heads, self.dropout, self.num_layers)
        self.linear_layer = tf.keras.layers.Dense(self.target_vocab_size,activation="linear")
        
    def call(self, x, y, encoder_self_attention_mask=None, decoder_self_attention_mask=None, decoder_cross_attention_mask=None):
        x = self.x_PE(x)
        x = self.x_dropout(x)
        x = self.encoder(x, encoder_self_attention_mask)
        
        y = self.y_PE(y)
        y = self.y_dropout(y)
        y = self.decoder(x, y, decoder_self_attention_mask, decoder_cross_attention_mask)
        y = self.linear_layer(y)
        return tf.nn.softmax(y)