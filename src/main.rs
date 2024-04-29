use neural_network::neural_network::NN;

fn main() {
    let nn = NN::new(vec![2, 3, 1]);
    let inputs = vec![0.5, 0.5];
    let outputs = nn.activate(&inputs);
    println!("{:?}", outputs);
}
