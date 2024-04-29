use rand::Rng;

struct Node {
    weights: Vec<f64>
}

impl Node {
    fn new(num_weights: usize) -> Node {
        let mut rng = rand::thread_rng();
        let mut weights = Vec::with_capacity(num_weights);
        for _ in 0..num_weights {
            let value: f64 = rng.gen();
            weights.push(value);
        }
        Node {weights}
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
        let mut nodes = Vec::with_capacity(size);
        for _ in 0..size {
            nodes.push(Node::new(previous_layer_size));
        }
        Layer {nodes}
    }
    pub fn activate(&self, inputs: &Vec<f64>) -> Vec<f64> {
        self.nodes.iter().map(|node| node.activate(inputs)).collect()
    }
}
