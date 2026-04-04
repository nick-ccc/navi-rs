use burn::{
    module::Module,
    nn::{
        self,
        conv::{Conv1d, Conv1dConfig, Conv1dRecord},
        PaddingConfig1d,
    },
    tensor::{activation::relu, backend::Backend, Bool, Int, Tensor},
};
