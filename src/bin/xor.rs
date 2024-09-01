use neural_network::layer::*;
use neural_network::matrix::Matrix;
use neural_network::neural_network::NN;
use neural_network::activation_function::*;

fn vec_to_matrix(vec: Vec<f64>) -> Matrix {
    let len = vec.len();
    Matrix::from_vec(vec, 1, len)
}

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
    let x_train = x_train.iter().map(|x| vec_to_matrix(x.clone())).collect();
    let y_train = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 1.0], vec![1.0, 0.0]];
    let y_train = y_train.iter().map(|x| vec_to_matrix(x.clone())).collect();

    match nn.train(&x_train, &y_train, 10000) {
        Ok(_) => println!("Training complete"),
        Err(e) => println!("Error: {}", e)
    }
    for inputs in x_train.iter() {
        println!("Input: {:?}, Model_Output: {:?}", inputs.get_data(), nn.predict(inputs).get_data());
    }
}
