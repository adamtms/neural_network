use super::layer_interface::Layer;
use crate::matrix::Matrix;

pub struct FlattenLayer {
    input_size: [usize; 2],
    output_size: [usize; 2]
}

impl FlattenLayer {
    pub fn new() -> FlattenLayer {
        FlattenLayer {input_size: [0, 0], output_size: [0, 0]}
    }
}

impl Layer for FlattenLayer {
    fn initialize(&mut self, input_size: [usize; 2]) {
        self.input_size = input_size;
        self.output_size = [1, input_size[0] * input_size[1]];
    }
    fn forward(&mut self, inputs: &Matrix) -> Matrix {
        Matrix::from_vec(inputs.get_data().clone(), self.output_size[0], self.output_size[1])
    }
    fn backwards(&mut self, output_error: &Matrix, _learning_rate: f64) -> Matrix {
        Matrix::from_vec(output_error.get_data().clone(), self.input_size[0], self.input_size[1])
    }
    fn get_size(&self) -> [usize; 2] {
        self.output_size
    }
}
