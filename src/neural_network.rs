use crate::layer::Layer;
use crate::activation_function::ActivationFunction;

pub struct NN<'r>{
    layers: Vec<Layer>,
    activation_function: &'r dyn ActivationFunction
}

impl<'r> NN<'r> {
    pub fn new(sizes: Vec<usize>, function: &'r dyn ActivationFunction) -> NN {
        let mut layers = Vec::with_capacity(sizes.len());
        for i in 1..sizes.len() {
            layers.push(Layer::new(sizes[i], sizes[i - 1]));
        }

        NN {layers, activation_function: function}
    }
    pub fn activate(&self, inputs: &Vec<f64>) -> Vec<f64> {
        let mut outputs = inputs.clone();
        for layer in self.layers.iter() {
            outputs = layer.activate(&outputs);
            outputs = self.activation_function.activate(&outputs);
        }
        outputs
    }
}
