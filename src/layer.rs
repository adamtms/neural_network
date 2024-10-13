use crate::matrix::Matrix;
use crate::activation_function::ActivationFunction;

pub trait Layer {
    fn forward(&mut self, inputs: &Matrix) -> Matrix;
    fn backwards(&mut self, output_error: &Matrix, learning_rate: f64) -> Matrix;
    fn initialize(&mut self, _input_size: [usize; 2]) {}
    fn get_size(&self) -> [usize; 2];
}

impl Layer for Box<dyn Layer> {
    fn forward(&mut self, inputs: &Matrix) -> Matrix {
        self.as_mut().forward(inputs)
    }
    fn initialize(&mut self, input_size: [usize; 2]) {
        self.as_mut().initialize(input_size)
    }
    fn backwards(&mut self, output_error: &Matrix, learning_rate: f64) -> Matrix {
        self.as_mut().backwards(output_error, learning_rate)
    }
    fn get_size(&self) -> [usize; 2] {
        self.as_ref().get_size()
    }
}

pub struct DenseLayer {
    size: [usize; 2],
    input_size: [usize; 2],
    weights: Matrix,
    biases: Matrix,
    last_input: Matrix
}

impl DenseLayer {
    pub fn new(size: usize) -> DenseLayer {
        let matrix = Matrix::new(0, 0);
        DenseLayer {size: [1, size], input_size: [0,0], weights: matrix.clone(), biases: matrix.clone(), last_input: matrix}
    }
}

impl Layer for DenseLayer {
    fn initialize(&mut self, input_size: [usize; 2]) {
        if input_size[0] != 1 {
            panic!("DenseLayer input size must be [1, n]");
        }
        self.input_size = input_size;
        self.weights = Matrix::new_random(self.input_size[1], self.size[1]);
        self.biases = Matrix::new_random(1, self.size[1]);
    }
    fn forward(&mut self, inputs: &Matrix) -> Matrix {
        self.last_input = inputs.clone();
        Matrix::mul(inputs, &self.weights).unwrap().add_matrix(&self.biases).unwrap().clone()
    }

    fn backwards(&mut self, output_error: &Matrix, learning_rate: f64) -> Matrix {
        let input_error = Matrix::mul(output_error, &Matrix::transpose(&self.weights));
        let weights_error = Matrix::mul(&Matrix::transpose(&self.last_input), output_error);
        let _ = self.weights.sub_matrix(weights_error.unwrap().mul_scalar(learning_rate));
        let _ = self.biases.sub_matrix(output_error.clone().mul_scalar(learning_rate));
        input_error.unwrap()
    }
    fn get_size(&self) -> [usize; 2] {
        self.size
    }
}

pub struct ConvolutionalLayer {
    kernel: Matrix,
    stride: usize,
    padding: usize,
    output_size: [usize; 2],
    last_input: Matrix
}

impl ConvolutionalLayer {
    pub fn new(kernel: Matrix, stride: usize, padding: usize) -> ConvolutionalLayer {
        ConvolutionalLayer {kernel, stride, padding, output_size: [0, 0], last_input: Matrix::new(0, 0)}
    }
    fn get_weight_error(&self, output_error: &Matrix, learning_rate: f64) -> Matrix {
        let mut result: Vec<f64> = Vec::with_capacity(self.kernel.get_num_rows() * self.kernel.get_num_cols());
        for i in 0..output_error.get_num_rows() {
            for j in 0..output_error.get_num_cols() {
                let mut counter = 0;
                for k in 0..self.kernel.get_num_rows() {
                    for l in 0..self.kernel.get_num_cols() {
                        let mut final_k = i * self.stride + k;
                        if final_k < self.padding || final_k >= self.last_input.get_num_rows() {
                            continue;
                        }
                        final_k -= self.padding;
                        let mut final_l = j * self.stride + l;
                        if final_l < self.padding || final_l >= self.last_input.get_num_cols() {
                            continue;
                        }
                        final_l -= self.padding;
                        result[counter] += self.last_input.get(final_k, final_l) * output_error.get(i, j);
                        counter += 1;
                    }
                }
            }
        }
        result.iter_mut().for_each(|x| *x = (*x)*learning_rate);
        Matrix::from_vec(result, self.kernel.get_num_rows(), self.kernel.get_num_cols())
    }
    fn get_input_error(&self, output_error: &Matrix) -> Matrix {
        let mut result: Matrix = Matrix::new(self.last_input.get_num_rows(), self.last_input.get_num_cols());
        for i in 0..output_error.get_num_rows() {
            for j in 0..output_error.get_num_cols() {
                for k in 0..self.kernel.get_num_rows() {
                    for l in 0..self.kernel.get_num_cols() {
                        let mut final_k = i * self.stride + k;
                        if final_k < self.padding || final_k >= self.last_input.get_num_rows() {
                            continue;
                        }
                        final_k -= self.padding;
                        let mut final_l = j * self.stride + l;
                        if final_l < self.padding || final_l >= self.last_input.get_num_cols() {
                            continue;
                        }
                        final_l -= self.padding;
                        result.set(final_k, final_l, result.get(final_k, final_l) + self.kernel.get(k, l) * output_error.get(i, j));
                    }
                }
            }
        }
        result
    }
}

impl Layer for ConvolutionalLayer {
    fn initialize(&mut self, input_size: [usize; 2]) {
        let output_num_rows = (input_size[0] - self.kernel.get_num_rows() + 2 * self.padding) / self.stride + 1;
        let output_num_cols = (input_size[1] - self.kernel.get_num_cols() + 2 * self.padding) / self.stride + 1;
        self.output_size = [output_num_rows, output_num_cols];
    }
    fn forward(&mut self, inputs: &Matrix) -> Matrix {
        self.last_input = inputs.clone();
        Matrix::convolve(inputs, &self.kernel, self.stride, self.padding)
    }
    fn backwards(&mut self, output_error: &Matrix, learning_rate: f64) -> Matrix {
        let weight_error = self.get_weight_error(output_error, learning_rate);
        let input_error = self.get_input_error(output_error);
        let result = self.kernel.sub_matrix(&weight_error);
        match result {
                Ok(_) => (),
                Err(_) => panic!("Error in ConvolutionalLayer implementation of get_weight_error")
            }
        input_error
    }
    fn get_size(&self) -> [usize; 2] {
        self.output_size
    }
}

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
    fn get_size(&self) -> [usize; 2] {
        self.size
    }
}
