use crate::layer::Layer;
use crate::matrix::Matrix;

pub struct NN{
    layers: Vec<Box<dyn Layer>>,
    learning_rate: f64,
    layer_sizes: Vec<usize>
}

fn mse(y_true: &Matrix, y_pred: &Matrix) -> f64 {
    let y_true = y_true.get_data();
    let y_pred = y_pred.get_data();
    y_true.iter().zip(y_pred.iter()).map(|(t, p)| (t-p).powi(2)).sum()
}

fn mse_derivative(y_true: &mut Matrix, y_pred: &Matrix, size: f64) -> Matrix {
    y_true.sub_matrix(y_pred).unwrap().mul_scalar(-2.0 / size).clone()
}

impl NN{
    pub fn new(input_size: usize, learning_rate: f64) -> NN {
        let layer_sizes = vec![input_size];
        NN {layers: Vec::new(), learning_rate, layer_sizes}
    }
    pub fn add(&mut self, mut layer: Box<dyn Layer>){
        layer.as_mut().initialize(*self.layer_sizes.last().unwrap());
        self.layer_sizes.push(layer.get_size());
        self.layers.push(layer);
    }
    pub fn predict(&mut self, inputs: &Matrix) -> Matrix {
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
        for epoch in 0..epochs {
            let mut err = 0.0;
            for i in 0..x_train.len() {
                let mut outputs = Matrix::from_vec(x_train[i].clone(), 1, x_train[i].len());
                for layer in self.layers.iter_mut() {
                    outputs = layer.forward(&outputs);
                }
                let mut y_true = Matrix::from_vec(y_train[i].clone(), 1, y_train[i].len());
                err += mse(&outputs, &y_true);

                let mut error = mse_derivative(&mut y_true, &outputs, input_size);
                for layer in self.layers.iter_mut().rev() {
                    error = layer.backwards(&error, self.learning_rate);
                }

            }
            err = err / f64::try_from(i32::try_from(x_train.len()).unwrap()).unwrap();
            errors.push(err);
            if epochs % 100 == 0 {
                println!("{:?} Error: {:?}", epoch, err);
            }

        }
        Ok(errors)
    }
}
