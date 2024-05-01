use rand::Rng;
use crate::activation_function::ActivationFunction;

pub trait Layer {
    fn forward(&mut self, inputs: &Vec<f64>) -> Vec<f64>;
    fn backwards(&mut self, output_error: &Vec<f64>, learning_rate: f64) -> Vec<f64>;
    fn initialize(&mut self, input_size: usize) {}
    fn get_last_input(&self) -> &Vec<f64>;
}

impl Layer for Box<dyn Layer> {
    fn forward(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
        self.as_mut().forward(inputs)
    }
    fn initialize(&mut self, input_size: usize) {
        self.as_mut().initialize(input_size)
    }
    fn backwards(&mut self, output_error: &Vec<f64>, learning_rate: f64) -> Vec<f64> {
        self.as_mut().backwards(output_error, learning_rate)
    }
    fn get_last_input(&self) -> &Vec<f64> {
        self.as_ref().get_last_input()
    }
}


struct Node {
    weights: Vec<f64>,
    bias: f64
}

impl Node {
    fn new(num_weights: usize) -> Node {
        let mut rng = rand::thread_rng();
        Node {weights: (0..num_weights).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect(),
              bias: rng.gen::<f64>() * 2.0 - 1.0}
    }
    fn forward(&self, inputs: &Vec<f64>) -> f64 {
        inputs.iter().zip(self.weights.iter()).map(|(x, y)| x * y + self.bias).sum()
    }
}

pub struct DenseLayer {
    size: usize,
    input_size: usize,
    nodes: Vec<Node>,
    last_input: Vec<f64>
}

impl DenseLayer {
    pub fn new(size: usize) -> DenseLayer {
        DenseLayer {size, input_size: 0, nodes: Vec::new(), last_input: Vec::new()}
    }
}

impl Layer for DenseLayer {
    fn forward(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
        self.last_input = inputs.clone();
        self.nodes.iter().map(|node| node.forward(inputs)).collect()
    }
    fn initialize(&mut self, input_size: usize) {
        self.input_size = input_size;
        self.nodes = (0..self.size).map(|_| Node::new(input_size)).collect();
    }
    fn backwards(&mut self, output_error: &Vec<f64>, learning_rate: f64) -> Vec<f64> {
        let mut input_error = Vec::new();
        for i in 0..self.input_size {
            input_error.push(output_error.iter().zip(self.nodes.iter()).map(|(error, node)| error*node.weights[i]).sum())
        }
        let weights_error: Vec<f64> = output_error.iter().zip(self.last_input.iter()).map(|(error, input)| error*input).collect();
        for i in 0..self.size {
            self.nodes[i].bias -= output_error[i] * learning_rate;
            for j in 0..self.input_size {
                self.nodes[i].weights[j] += weights_error[i] * learning_rate;
            }
        }
        input_error
    }
    fn get_last_input(&self) -> &Vec<f64> {
        &self.last_input
    }
}

pub struct ActivationLayer {
    activation_function: Box<dyn ActivationFunction>,
    last_input: Vec<f64>
}

impl ActivationLayer {
    pub fn new(activation_function: Box<dyn ActivationFunction>) -> ActivationLayer {
        ActivationLayer {activation_function, last_input: Vec::new()}
    }
}

impl Layer for ActivationLayer {
    fn forward(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
        self.last_input = inputs.clone();
        self.activation_function.as_ref().forward(inputs)
    }
    fn backwards(&mut self, output_error: &Vec<f64>, learning_rate: f64) -> Vec<f64> {
        output_error.iter().zip(self.get_last_input().iter())
            .map(|(error, input)| self.activation_function.as_ref().derivative(*input)*error).collect()
    }
    fn get_last_input(&self) -> &Vec<f64> {
        &self.last_input
    }
}
