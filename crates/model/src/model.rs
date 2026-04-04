use burn::{
    module::Module,
    nn::{
        conv::{Conv1d, Conv1dConfig},
        LayerNorm, LayerNormConfig,
        Linear, LinearConfig,
    },
    tensor::{backend::Backend, Tensor},
};

#[derive(Debug, Clone)]
pub struct ModelDimensions {
    n_mels: u32,
    n_audio_ctx: u32,
    n_audio_state: u32,
    n_audio_head: u32,
    n_audio_layer: u32,
    n_vocab: u32,
    n_text_ctx: u32,
    n_text_state: u32,
    n_text_head: u32,
    n_text_layer: u32,
}

#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    pub n_head: usize,
    pub query: Linear<B>,
    pub key: Linear<B>,
    pub value: Linear<B>,
    pub out: Linear<B>,
}

impl<B: Backend> MultiHeadAttention<B> {
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

    /// Forward pass
    ///
    /// x shape: [batch, seq_len, n_state]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let q = self.query.forward(x.clone()); // [B, T, n_state]
        let k = self.key.forward(x.clone());
        let v = self.value.forward(x);

        // Note: actual multi-head attention requires:
        // 1. reshape to [B, T, n_head, head_dim]
        // 2. transpose to [B, n_head, T, head_dim]
        // 3. scaled dot-product attention (SDPA)
        // 4. combine heads and project with self.out

        // For now, just returning q as a placeholder
        q
    }
}




// #[derive(Module, Debug)]
// pub struct ResidualAttentionBlock<B: Backend> {
//     todo: i32,
// }

/// Sinusoidal positional embeddings `[n_audio_ctx, n_audio_state]`.
///
/// Encodes each position using sine/cosine functions at different
/// frequencies
///
/// Shapes:
/// - positions: `[T, 1]`
/// - div_term: `[1, D/2]`
/// - output: `[T, D]`
///
/// Notes:
/// - `n_audio_state` must be even
/// - uses broadcasting `[T,1] * [1,D/2]`
fn sinusoids <B: Backend>(
    n_audio_ctx: usize, 
    n_audio_state: usize, 
    device: &B::Device
) -> Tensor<B, 2> {
    let half = n_audio_state / 2;
    
    // Generate n_audio_ctx positional tensor
    let positions = Tensor::<B, 1, burn::tensor::Int>::arange(0..n_audio_ctx as i64, device)
        .float()
        .unsqueeze::<2>();

    // Generate n_audio_state / 2 size tensor for frequency scaling
    let div_term = Tensor::<B, 1, burn::tensor::Int>::arange(0..half as i64, device)
        .float()
        .mul_scalar(-(10000.0_f32.ln()) / half as f32)
        .exp()
        .unsqueeze::<2>()
        .swap_dims(0, 1);

    let sinusoid = positions * div_term;
    let sin = sinusoid.clone().sin();
    let cos = sinusoid.cos();

    Tensor::cat(vec![sin, cos], 1)
}
    

#[derive(Module, Debug)]
pub struct AudioEncoder<B: Backend> {
    pub conv1: Conv1d<B>,
    pub conv2: Conv1d<B>,
    pub positional_embedding: Tensor<B, 2>, // [n_ctx, n_state]
    pub blocks: Vec<ResidualAttentionBlock<B>>,
    pub ln_post: LayerNorm<B>,
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
            .with_padding(1)
            .init(device);

        let conv2 = Conv1dConfig::new(n_audio_state, n_audio_state, 3)
            .with_stride(2)
            .with_padding(1)
            .init(device);

        // Self {
        //     conv1,
        //     conv2,
        //     positional_embedding,
        //     blocks,
        //     ln_post,
        // }
    }
}