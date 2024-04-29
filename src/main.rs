use neural_network::neural_network::NN;
use neural_network::activation_function::*;

fn main() {
    let nn = NN::new(vec![2, 432, 900, 300, 50, 1], &Sigmoid{});
    let inputs = vec![0.5, 0.5];
    let outputs = nn.activate(&inputs);
    println!("{:?}", outputs);
}
