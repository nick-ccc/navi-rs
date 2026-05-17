use core::{f32, panic};
use std::{usize, vec};

use burn::{
    module::{Module, Param},
    nn::{
        Embedding, EmbeddingConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig,
        PaddingConfig1d,
        conv::{Conv1d, Conv1dConfig},
    },
    tensor::{Distribution, Int, Tensor, activation::softmax, backend::Backend, s},
};

#[derive(Debug, Clone)]
pub struct ModelDimensions {
    pub n_mels: usize,
    pub n_audio_ctx: usize,
    pub n_audio_state: usize,
    pub n_audio_head: usize,
    pub n_audio_layer: usize,
    pub n_vocab: usize,
    pub n_text_ctx: usize,
    pub n_text_state: usize,
    pub n_text_head: usize,
    pub n_text_layer: usize,
}

fn sinusoids_positional_embedding<B: Backend>(
    n_audio_ctx: usize,
    n_audio_state: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    if !n_audio_state.is_multiple_of(2) {
        panic!("audio state length must be divisible by two")
    }

    let half = n_audio_state / 2;

    // Generate 1 x n_audio_ctx positional tensor
    let positions = Tensor::<B, 1, Int>::arange(0..n_audio_ctx as i64, device)
        .float()
        .unsqueeze::<2>()
        .swap_dims(1, 0);

    // Generate 1 x n_audio_state / 2 size tensor for frequency scaling
    let div_term = Tensor::<B, 1, Int>::arange(0..half as i64, device)
        .float()
        .mul_scalar(-(10000.0_f32.ln()) / half as f32)
        .exp()
        .unsqueeze::<2>();

    let sinusoid = positions * div_term;
    let sin = sinusoid.clone().sin();
    let cos = sinusoid.cos();

    // dim: n_audio_state x n_audio_ctx
    Tensor::cat(vec![sin, cos], 1)
}

#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    n_head: usize,
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    out: Linear<B>,
}

impl<B: Backend> MultiHeadAttention<B> {
    /// Creates new MultiHeadAttention block
    ///
    /// Produces linear configurations for later linear layers based on the state (dimnesion of the model)
    pub fn new(n_state: usize, n_head: usize, device: &B::Device) -> Self {
        let query = LinearConfig::new(n_state, n_state).init(device);
        let key = LinearConfig::new(n_state, n_state)
            .with_bias(false)
            .init(device);
        let value = LinearConfig::new(n_state, n_state).init(device);
        let out = LinearConfig::new(n_state, n_state).init(device);

        Self {
            n_head,
            query,
            key,
            value,
            out,
        }
    }

    /// Forward pass of multi-head self-attention.
    ///
    /// Applies linear projections to input `x` to produce query, key, and value,
    /// computes scaled dot-product attention, and applies a final output projection.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        xa: Option<Tensor<B, 3>>,
        mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        let query = self.query.forward(x.clone()); // [B, T, n_state]

        // toggles based on attention model
        let key = if let Some(v) = &xa {
            self.key.forward(v.clone())
        } else {
            self.key.forward(x.clone())
        };
        let value = if let Some(v) = &xa {
            self.value.forward(v.clone())
        } else {
            self.value.forward(x.clone())
        };

        // For now, just returning q as a placeholder
        let attention = self.qkv_attention(query, key, value, mask);

        self.out.forward(attention)
    }

    /// Perfroms the attention calculation for the multiheaded attention block
    ///
    /// Based on the formula in Attention is all you need - using scaled dot product attention:
    ///     Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    ///
    /// Inputs:
    /// * query: [batch_size, query_length, model_dim (n_state)]
    /// * key/value: [batch_size, key_length, model_dim (n_state)]
    /// * mask (optional): [query_length, key_length]
    ///
    /// Process:
    /// * Split into heads → [batch_size, number_of_heads (self.n_head), sequence_length, dimension_per_head]
    /// * Compute QK^T, apply mask, softmax
    /// * Weight values and merge heads → [batch_size, sequence_length, D]
    fn qkv_attention(
        &self,
        query: Tensor<B, 3>,
        key: Tensor<B, 3>,
        value: Tensor<B, 3>,
        mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        let [n_batch, n_qctx, n_state] = query.dims();
        let [_, n_ctx, _] = key.dims();

        let scale = (n_state as f64 / self.n_head as f64).powf(-0.25);
        let head_dim = n_state / self.n_head;

        let query = query
            .reshape([n_batch, n_qctx, self.n_head, head_dim])
            .permute([0, 2, 1, 3])
            .mul_scalar(scale);

        let key = key
            .reshape([n_batch, n_ctx, self.n_head, head_dim])
            .permute([0, 2, 1, 3])
            .mul_scalar(scale)
            .transpose();

        let value = value
            .reshape([n_batch, n_ctx, self.n_head, head_dim])
            .permute([0, 2, 1, 3]);

        // Q * K.transpose()
        let query_key = query.matmul(key);

        // apply the mask if one exists - and adjust size to context
        let query_key = if let Some(mask) = mask {
            query_key + mask.slice([0..n_qctx, 0..n_ctx]).unsqueeze::<4>()
        } else {
            query_key
        };

        let weight = softmax(query_key, 3);

        // Return output softmax(Q*K^T) * V
        weight.matmul(value).permute([0, 2, 1, 3]).flatten(2, 3)
    }
}

#[derive(Module, Debug)]
pub struct ResidualAttentionBlock<B: Backend> {
    attention_layer: MultiHeadAttention<B>,
    attention_layer_normalization: LayerNorm<B>,
    multilayer_percpetron: MultiLayerPerceptron<B>,
    multilayer_percpetron_normalization: LayerNorm<B>,
    cross_attention_layer: Option<MultiHeadAttention<B>>,
    cross_attention_layer_normalization: Option<LayerNorm<B>>,
}

impl<B: Backend> ResidualAttentionBlock<B> {
    /// Creates new ResidualAttentionBlock
    pub fn new(n_state: usize, n_head: usize, cross_attention: bool, device: &B::Device) -> Self {
        let attention_layer = MultiHeadAttention::new(n_state, n_head, device);
        let attention_layer_normalization = LayerNormConfig::new(n_state).init(device);

        // Optional cross attention layer
        let cross_attention_layer = if cross_attention {
            Some(MultiHeadAttention::new(n_state, n_head, device))
        } else {
            None
        };
        let cross_attention_layer_normalization = if cross_attention {
            Some(LayerNormConfig::new(n_state).init(device))
        } else {
            None
        };

        let multilayer_percpetron = MultiLayerPerceptron::new(n_state, device);
        let multilayer_percpetron_normalization = LayerNormConfig::new(n_state).init(device);

        Self {
            attention_layer,
            attention_layer_normalization,
            multilayer_percpetron,
            multilayer_percpetron_normalization,
            cross_attention_layer,
            cross_attention_layer_normalization,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        xa: Option<Tensor<B, 3>>,
        mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        // Apply first linear layer
        let x = x.clone()
            + self.attention_layer.forward(
                self.attention_layer_normalization.forward(x),
                None,
                mask.clone(),
            );

        // Apply cross attention if applicable
        let x = if let (Some(cross_attention_layer), Some(cross_attention_layer_normalization)) = (
            &self.cross_attention_layer,
            &self.cross_attention_layer_normalization,
        ) {
            x.clone()
                + cross_attention_layer.forward(
                    cross_attention_layer_normalization.forward(x),
                    xa.clone(),
                    None,
                )
        } else {
            x
        };

        // Apply last linear layer and return resulting tensor
        let x = self.multilayer_percpetron.forward(x);
        x.clone() + self.multilayer_percpetron.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct MultiLayerPerceptron<B: Backend> {
    linear_layer_1: Linear<B>,
    gelu: Gelu,
    linear_layer_2: Linear<B>,
}

impl<B: burn::tensor::backend::Backend> MultiLayerPerceptron<B> {
    pub fn new(n_state: usize, device: &B::Device) -> Self {
        let linear_layer_1 = LinearConfig::new(n_state, 4 * n_state).init(device);
        let gelu = Gelu::new();
        let linear_layer_2 = LinearConfig::new(4 * n_state, n_state).init(device);

        Self {
            linear_layer_1,
            gelu,
            linear_layer_2,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.linear_layer_1.forward(x);
        let x = self.gelu.forward(x);
        let x = self.linear_layer_2.forward(x);

        x
    }
}

#[derive(Module, Debug)]
pub struct AudioEncoder<B: Backend> {
    pub conv1: Conv1d<B>,
    pub gelu1: Gelu,
    pub conv2: Conv1d<B>,
    pub gelu2: Gelu,
    pub positional_embedding: Tensor<B, 2>,
    pub blocks: Vec<ResidualAttentionBlock<B>>,
    pub layer_normalization_post: LayerNorm<B>,
}

impl<B: Backend> AudioEncoder<B> {
    pub fn new(
        n_mels: usize,
        n_audio_ctx: usize,
        n_audio_state: usize,
        n_audio_head: usize,
        n_audio_layer: usize,
        device: &B::Device,
    ) -> Self {
        let conv1 = Conv1dConfig::new(n_mels, n_audio_state, 3)
            .with_padding(PaddingConfig1d::Explicit(1))
            .init(device);

        let gelu1 = Gelu::new();

        let conv2 = Conv1dConfig::new(n_audio_state, n_audio_state, 3)
            .with_stride(2)
            .with_padding(PaddingConfig1d::Explicit(1))
            .init(device);
        let gelu2 = Gelu::new();

        let positional_embedding =
            sinusoids_positional_embedding(n_audio_ctx, n_audio_state, device);

        let blocks: Vec<ResidualAttentionBlock<B>> = (0..n_audio_layer)
            .map(|_| ResidualAttentionBlock::new(n_audio_state, n_audio_head, true, device))
            .collect();

        let layer_normalization_post = LayerNormConfig::new(n_audio_state).init(device);

        Self {
            conv1,
            gelu1,
            conv2,
            gelu2,
            positional_embedding,
            blocks,
            layer_normalization_post,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.gelu1.forward(self.conv1.forward(x));
        let x = self.gelu2.forward(self.conv2.forward(x));
        let x = x.permute([0, 2, 1]);

        //if x.dims()[1..] == self.positional_embedding.dims() {
        //    panic!("Incorrect audio shape");
        //}

        let k = x.dims()[1];

        let x = x + self
            .positional_embedding
            .clone()
            .slice(0..k)
            .unsqueeze::<3>();

        let mut x = x;
        for block in &self.blocks {
            x = block.forward(x, None, None);
        }

        self.layer_normalization_post.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct TextDecoder<B: Backend> {
    pub token_embedding: Embedding<B>,
    pub positional_emebedding: Param<Tensor<B, 2>>,
    pub blocks: Vec<ResidualAttentionBlock<B>>,
    pub layer_normalization_post: LayerNorm<B>,
    pub mask: Tensor<B, 2>,
}

impl<B: Backend> TextDecoder<B> {
    pub fn new(
        n_vocab: usize,
        n_text_ctx: usize,
        n_text_state: usize,
        n_text_head: usize,
        n_text_layer: usize,
        device: &B::Device,
    ) -> Self {
        let embedding_config = EmbeddingConfig::new(n_vocab, n_text_state);
        let token_embedding = embedding_config.init(device);

        let positional_emebedding = Param::from_tensor(Tensor::<B, 2>::random(
            [n_text_ctx, n_text_state],
            Distribution::Default,
            device,
        ));

        let blocks: Vec<ResidualAttentionBlock<B>> = (0..n_text_layer)
            .map(|_| ResidualAttentionBlock::new(n_text_state, n_text_head, true, device))
            .collect();

        let layer_normalization_post = LayerNormConfig::new(n_text_state).init(device);
        let mask =
            Tensor::<B, 2>::full([n_text_ctx, n_text_ctx], f32::NEG_INFINITY, device).triu(1);

        Self {
            token_embedding,
            positional_emebedding,
            blocks,
            layer_normalization_post,
            mask,
        }
    }

    // Forward pass on text decoder
    //
    // Parameters:
    //  *  x: Tesnor<B,2>: 2D tesnor of shape (batch_size, <= n_ctx)
    //  * xa: Tesnor<B,3>: 3D tesnor of shape (batch_size, n_audio_ctx, n_audio_state)
    //                      of the audio features to be attended on
    pub fn forward(&self, x: Tensor<B, 2, Int>, xa: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_, seq_len] = x.dims();

        // could chcek seq_len is valid here

        // Why turn into int..
        let x = self.token_embedding.forward(x)
            + self
                .positional_emebedding
                .val()
                .slice([0..seq_len])
                .unsqueeze::<3>();

        let mut x = x;
        for block in &self.blocks {
            x = block.forward(x, Some(xa.clone()), Some(self.mask.clone()))
        }

        let x = self.layer_normalization_post.forward(x);

        // retrun logits
        x.matmul(
            self.token_embedding
                .weight
                .val()
                .transpose()
                .unsqueeze::<3>(),
        )
    }
}

#[derive(Module, Debug)]
pub struct Whisper<B: Backend> {
    encoder: AudioEncoder<B>,
    decoder: TextDecoder<B>,
}

impl<B: Backend> Whisper<B> {
    pub fn new(model_dims: &ModelDimensions, device: &B::Device) -> Self {
        let encoder = AudioEncoder::new(
            model_dims.n_mels,
            model_dims.n_audio_ctx,
            model_dims.n_audio_state,
            model_dims.n_audio_head,
            model_dims.n_audio_layer,
            device,
        );

        let decoder = TextDecoder::new(
            model_dims.n_vocab,
            model_dims.n_text_ctx,
            model_dims.n_text_state,
            model_dims.n_text_head,
            model_dims.n_text_layer,
            device,
        );

        Self { encoder, decoder }
    }

    pub fn forward(&self, mel: Tensor<B, 3>, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.forward_decoder(tokens, self.forward_encoder(mel))
    }

    pub fn forward_encoder(&self, mel: Tensor<B, 3>) -> Tensor<B, 3> {
        self.encoder.forward(mel)
    }

    pub fn forward_decoder(
        &self,
        tokens: Tensor<B, 2, Int>,
        encoder_output: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        self.decoder.forward(tokens, encoder_output)
    }

    pub fn forward_decoder_from_token_vector(
        &self,
        tokens: &Vec<u32>,
        encoder_output: Tensor<B, 3>,
    ) -> u32 {
        let seq_len = tokens.len();
        let device = encoder_output.device();

        let token_ints: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let tokens = Tensor::<B, 1, Int>::from_data(token_ints.as_slice(), &device);
        let tokens = tokens.reshape([1, seq_len]);

        let logits = self.decoder.forward(tokens, encoder_output);
        let last_token_logits = logits.slice(s![0..1, -1..]);
        let argmax_tensor = last_token_logits.argmax(2);

        let next_token_id = argmax_tensor
            .into_data()
            .to_vec::<i32>()
            .expect("Failed to read argmax tensor")[0] as u32;

        next_token_id
    }
}
