use crate::layer::Layer;
use crate::activation_function::ActivationFunction;

pub struct NN{
    layers: Vec<Box<dyn Layer>>,
    learning_rate: f64,
    layer_sizes: Vec<usize>
}

impl NN{
    pub fn new(input_size: usize, learning_rate: f64) -> NN {
        let layer_sizes = vec![input_size];
        NN {layers: Vec::new(), learning_rate, layer_sizes}
    }
    pub fn add(&mut self, mut layer: Box<dyn Layer>){
        layer.as_mut().initialize(*self.layer_sizes.last().unwrap());
        self.layers.push(layer);
    }
    pub fn predict(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
        let mut outputs = inputs.clone();
        for layer in self.layers.iter_mut() {
            outputs = layer.forward(&outputs);
        }
        outputs
    }
    pub fn train(&mut self, x_train: &Vec<Vec<f64>>, y_train: &Vec<Vec<f64>>, epochs: u64) -> Result<Vec<f64>, String>{
        if x_train.len() != y_train.len() {
            return Err("x_train and y_train must have the same length".to_owned());
        }
        let mut errors = Vec::new();
        let input_size: f64 = f64::try_from(i32::try_from(y_train[0].len()).unwrap()).unwrap();
        for _ in 0..epochs {
            let mut error = 0.0;
            for i in 0..x_train.len() {
                let mut outputs = x_train[i].clone();
                for layer in self.layers.iter_mut() {
                    outputs = layer.forward(&outputs);
                }
                let row_error: f64 = y_train[i].iter().zip(outputs.iter()).map(|(y, o)| (y - o).powi(2)).sum();
                error += row_error;

                let mut back_error = y_train[i].iter().zip(outputs.iter()).map(|(y, o)| 2.0 * (y - o) / input_size).collect();
                for layer in self.layers.iter_mut().rev() {
                    back_error = layer.backwards(&back_error, self.learning_rate);
                }
            }
            error = error / f64::try_from(i32::try_from(x_train.len()).unwrap()).unwrap();
            errors.push(error);
            println!("{:?}", error);
        }
        Ok(errors)
    }
}
