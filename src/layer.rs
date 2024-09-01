use crate::matrix::Matrix;
use crate::activation_function::ActivationFunction;

pub trait Layer {
    fn forward(&mut self, inputs: &Matrix) -> Matrix;
    fn backwards(&mut self, output_error: &Matrix, learning_rate: f64) -> Matrix;
    fn initialize(&mut self, input_size: usize) {}
    fn get_last_input(&self) -> &Matrix;
    fn get_size(&self) -> usize;
}

impl Layer for Box<dyn Layer> {
    fn forward(&mut self, inputs: &Matrix) -> Matrix {
        self.as_mut().forward(inputs)
    }
    fn initialize(&mut self, input_size: usize) {
        self.as_mut().initialize(input_size)
    }
    fn backwards(&mut self, output_error: &Matrix, learning_rate: f64) -> Matrix {
        self.as_mut().backwards(output_error, learning_rate)
    }
    fn get_last_input(&self) -> &Matrix {
        self.as_ref().get_last_input()
    }
    fn get_size(&self) -> usize {
        self.as_ref().get_size()
    }
}

pub struct DenseLayer {
    size: usize,
    input_size: usize,
    weights: Matrix,
    biases: Matrix,
    last_input: Matrix
}

impl DenseLayer {
    pub fn new(size: usize) -> DenseLayer {
        let matrix = Matrix::new(0, 0);
        DenseLayer {size, input_size: 0, weights: matrix.clone(), biases: matrix.clone(), last_input: matrix}
    }
}

impl Layer for DenseLayer {
    fn initialize(&mut self, input_size: usize) {
        self.input_size = input_size;
        self.weights = Matrix::new_random(self.input_size, self.size);
        self.biases = Matrix::new_random(1, self.size);
    }
    fn forward(&mut self, inputs: &Matrix) -> Matrix {
        self.last_input = inputs.clone();
        Matrix::mul(inputs, &self.weights).unwrap().add_matrix(&self.biases).unwrap().clone()
    }
    fn backwards(&mut self, output_error: &Matrix, learning_rate: f64) -> Matrix {
        let input_error = Matrix::mul(output_error, &Matrix::transpose(&self.weights));
        let weights_error = Matrix::mul(&Matrix::transpose(&self.last_input), output_error);
        self.weights.sub_matrix(weights_error.unwrap().mul_scalar(learning_rate));
        self.biases.sub_matrix(output_error.clone().mul_scalar(learning_rate));
        input_error.unwrap()
    }
    fn get_last_input(&self) -> &Matrix {
        &self.last_input
    }
    fn get_size(&self) -> usize {
        self.size
    }
}

pub struct ActivationLayer {
    activation_function: Box<dyn ActivationFunction>,
    last_input: Matrix,
    size: usize
}

impl ActivationLayer {
    pub fn new(activation_function: Box<dyn ActivationFunction>) -> ActivationLayer {
        ActivationLayer {activation_function, last_input: Matrix::new(0, 0), size: 0}
    }
}

impl Layer for ActivationLayer {
    fn initialize(&mut self, input_size: usize) {
        self.size = input_size;
    }
    fn forward(&mut self, inputs: &Matrix) -> Matrix {
        self.last_input = inputs.clone();
        self.activation_function.as_ref().forward(inputs)
    }
    fn backwards(&mut self, output_error: &Matrix, learning_rate: f64) -> Matrix {
self.activation_function.as_ref().backwards(&self.last_input).elementwise_mul(output_error).unwrap().clone()
    }
    fn get_last_input(&self) -> &Matrix {
        &self.last_input
    }
    fn get_size(&self) -> usize {
        self.size
    }
}
