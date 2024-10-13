use super::layer_interface::Layer;
use crate::matrix::Matrix;

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
