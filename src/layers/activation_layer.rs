use super::layer_interface::Layer;
use crate::matrix::Matrix;
use crate::activation_function::ActivationFunction;

pub struct ActivationLayer {
    activation_function: Box<dyn ActivationFunction>,
    last_input: Matrix,
    size: [usize; 2]
}

impl ActivationLayer {
    pub fn new(activation_function: Box<dyn ActivationFunction>) -> ActivationLayer {
        ActivationLayer {activation_function, last_input: Matrix::new(0, 0), size: [0, 0]}
    }
}

impl Layer for ActivationLayer {
    fn initialize(&mut self, input_size: [usize; 2]) {
        self.size = input_size;
    }
    fn forward(&mut self, inputs: &Matrix) -> Matrix {
        self.last_input = inputs.clone();
        self.activation_function.as_ref().forward(inputs)
    }
    fn backwards(&mut self, output_error: &Matrix, _learning_rate: f64) -> Matrix {
        self.activation_function.as_ref().backwards(&self.last_input).elementwise_mul(output_error).unwrap().clone()
    }
    fn get_last_input(&self) -> &Matrix {
        &self.last_input
    }
    fn get_size(&self) -> [usize; 2] {
        self.size
    }
}
