use crate::matrix::Matrix;

#[cfg(not(windows))]
macro_rules! main_separator{
    ()=>{"/"}
}

#[cfg(windows)]
macro_rules! main_separator{
    ()=>{r#"\"#}
}

macro_rules! get_mnist_file_path {
    ($file:literal) => {
        concat!("..", main_separator!(), "mnist_dataset", main_separator!(), $file)
    };
}

const TRAIN_IMAGES: &'static [u8] = include_bytes!(get_mnist_file_path!("train-images-60k"));
const TRAIN_LABELS: &'static [u8] = include_bytes!(get_mnist_file_path!("train-labels-60k"));
const TEST_IMAGES: &'static [u8] = include_bytes!(get_mnist_file_path!("test-images-10k"));
const TEST_LABELS: &'static [u8] = include_bytes!(get_mnist_file_path!("test-labels-10k"));


pub struct MnistDataset {
    num_of_images: usize,
    num_of_rows: usize,
    num_of_cols: usize,
    images: Vec<Matrix>,
    labels: Vec<u8>
}

fn load_dataset(images: &[u8], labels: &[u8]) -> Result<MnistDataset, std::io::Error> {
    let images: Vec<u8> = images.to_vec();
    let labels: Vec<u8> = labels.to_vec();
    let num_of_images: usize = i32::from_be_bytes(images[4..8].try_into().unwrap()).try_into().unwrap();
    let num_of_rows: usize = i32::from_be_bytes(images[8..12].try_into().unwrap()).try_into().unwrap();
    let num_of_cols: usize = i32::from_be_bytes(images[12..16].try_into().unwrap()).try_into().unwrap();
    let bytes_per_image = num_of_rows * num_of_cols;
    let mut image_data = Vec::with_capacity(num_of_images);
    let mut label_data = Vec::with_capacity(num_of_images);
    for i in 0..num_of_images {
        let image_start: usize = 16 + i * bytes_per_image;
        let data: Vec<u8> = images[image_start..(image_start+bytes_per_image)].try_into().unwrap();
        let data: Vec<f64> = data.iter().map(|&x| x as f64).collect();
        image_data.push(Matrix::from_vec(data, num_of_rows, num_of_cols));
        let label: u8 = labels[8+i];
        label_data.push(label);
    }
    Ok(MnistDataset {
        num_of_images,
        num_of_rows,
        num_of_cols,
        images: image_data,
        labels: label_data
    })
}

pub fn load_train_dataset() -> Result<MnistDataset, std::io::Error> {
    load_dataset(TRAIN_IMAGES, TRAIN_LABELS)
}

pub fn load_test_dataset() -> Result<MnistDataset, std::io::Error> {
    load_dataset(TEST_IMAGES, TEST_LABELS)
}
