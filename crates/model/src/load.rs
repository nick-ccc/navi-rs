use burn::{
    module::Module,
    nn::{
        self, PaddingConfig1d,
        conv::{Conv1d, Conv1dConfig, Conv1dRecord},
    },
    tensor::{Bool, Int, Tensor, activation::relu, backend::Backend},
};
