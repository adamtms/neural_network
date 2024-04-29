use crate::layer::Layer;

pub struct NN {
    layers: Vec<Layer>
}

impl NN {
    pub fn new(sizes: Vec<usize>) -> NN {
        let mut layers = Vec::with_capacity(sizes.len());
        for i in 1..sizes.len() {
            layers.push(Layer::new(sizes[i], sizes[i - 1]));
        }
        NN {layers}
    }
    pub fn activate(&self, inputs: &Vec<f64>) -> Vec<f64> {
        let mut outputs = inputs.clone();
        for layer in self.layers.iter() {
            outputs = layer.activate(&outputs);
        }
        outputs
    }
}
