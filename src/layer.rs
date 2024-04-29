use rand::Rng;

struct Node {
    weights: Vec<f64>
}

impl Node {
    fn new(num_weights: usize) -> Node {
        let mut rng = rand::thread_rng();
        Node {weights: (0..num_weights).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect()}
    }
    fn activate(&self, inputs: &Vec<f64>) -> f64 {
        inputs.iter().zip(self.weights.iter()).map(|(x, y)| x * y).sum()
    }
}

pub struct Layer {
    nodes: Vec<Node>
}

impl Layer {
    pub fn new(size: usize, previous_layer_size: usize) -> Layer {
        Layer {nodes: (0..size).map(|_| Node::new(previous_layer_size)).collect()}
    }
    pub fn activate(&self, inputs: &Vec<f64>) -> Vec<f64> {
        self.nodes.iter().map(|node| node.activate(inputs)).collect()
    }
}
