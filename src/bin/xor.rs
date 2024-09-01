use neural_network::layer::*;
use neural_network::matrix::Matrix;
use neural_network::neural_network::NN;
use neural_network::activation_function::*;

fn main() {
    let mut nn: NN = NN::new([1, 2], 0.1);
    let layer_sizes = vec![4, 8, 4];
    for i in 1..layer_sizes.len() {
        nn.add(Box::new(DenseLayer::new(layer_sizes[i])));
        nn.add(Box::new(ActivationLayer::new(Box::new(ReLU{}))));
    }

    nn.add(Box::new(DenseLayer::new(2)));
    nn.add(Box::new(ActivationLayer::new(Box::new(Sigmoid{}))));
    let x_train = vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]];
    let y_train = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 1.0], vec![1.0, 0.0]];
    match nn.train(&x_train, &y_train, 10000) {
        Ok(_) => println!("Training complete"),
        Err(e) => println!("Error: {}", e)
    }
    for inputs in x_train.iter() {
        let input_matrix = Matrix::from_vec(inputs.clone(), 1, inputs.len());
        println!("Input: {:?}, Model_Output: {:?}", inputs, nn.predict(&input_matrix).get_data());
    }
}
