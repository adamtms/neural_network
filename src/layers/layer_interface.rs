use crate::matrix::Matrix;

pub trait Layer {
    fn forward(&mut self, inputs: &Matrix) -> Matrix;
    fn backwards(&mut self, output_error: &Matrix, learning_rate: f64) -> Matrix;
    fn initialize(&mut self, _input_size: [usize; 2]) {}
    fn get_last_input(&self) -> &Matrix;
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
    fn get_last_input(&self) -> &Matrix {
        self.as_ref().get_last_input()
    }
    fn get_size(&self) -> [usize; 2] {
        self.as_ref().get_size()
    }
}
