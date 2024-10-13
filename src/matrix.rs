use rand::Rng;

pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Matrix {
        Matrix {rows, cols, data: vec![0.0; rows*cols]}
    }

    pub fn equals(&self, other: &Matrix) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }
        self.data.iter().zip(other.data.iter()).all(|(a, b)| a == b)
    }

    pub fn get_num_rows(&self) -> usize {
        self.rows
    }

    pub fn get_num_cols(&self) -> usize {
        self.cols
    }

    pub fn new_random(rows: usize, cols: usize) -> Matrix {
        let mut rng = rand::thread_rng();
        let data  = (0..rows*cols).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect();
        Matrix{rows, cols, data}
    }

    pub fn from_vec(data: Vec<f64>, rows: usize, cols: usize) -> Matrix {
        if data.len() != rows*cols {
            panic!("Data length must match matrix dimensions");
        }
        Matrix {rows, cols, data}
    }

    pub fn from_nested_vec(data: Vec<Vec<f64>>) -> Result<Matrix, String> {
        let rows = data.len();
        if rows == 0 {
            return Err("Matrix cannot have 0 rows".to_string());
        }
        let mut values = Vec::new();
        let cols = data[0].len();
        for column in data.iter() {
            if column.len() != cols {
                return Err("All columns must have the same length".to_string());
            }
            values.extend(column.iter());
        }
        Ok(Matrix{rows, cols, data: values})
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row*self.cols + col]
    }

    pub fn get_data(&self) -> &Vec<f64> {
        &self.data
    }

    pub fn get_data_mut(&mut self) -> &mut Vec<f64> {
        &mut self.data
    }

    pub fn set(&mut self, row: usize, col: usize, value: f64) -> &mut Matrix{
        self.data[row*self.cols + col] = value;
        self
    }

    pub fn add_matrix(&mut self, other: &Matrix) -> Result<&mut Matrix, String>{
        if self.rows != other.rows || self.cols != other.cols {
            return Err("Matrix dimensions must match".to_string());
        }

        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i*self.cols + j] += other.data[i*self.cols + j];
            }
        }
        Ok(self)
    }

    pub fn sub_matrix(&mut self, other: &Matrix) -> Result<&mut Matrix, String>{
        if self.rows != other.rows || self.cols != other.cols {
            return Err("Matrix dimensions must match".to_string());
        }

        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i*self.cols + j] -= other.data[i*self.cols + j];
            }
        }
        Ok(self)
    }

    pub fn add_scalar(&mut self, scalar: f64) -> &mut Matrix {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i*self.cols + j] += scalar;
            }
        }
        self
    }

    pub fn sub_scalar(&mut self, scalar: f64) -> &mut Matrix {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i*self.cols + j] -= scalar;
            }
        }
        self
    }

    pub fn mul_scalar(&mut self, scalar: f64) -> &mut Matrix {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i*self.cols + j] *= scalar;
            }
        }
        self
    }

    pub fn elementwise_mul(&mut self, other: &Matrix) -> Result<&mut Matrix, String> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err("Matrix dimensions must match".to_string());
        }
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i*self.cols + j] *= other.data[i*self.cols + j];
            }
        }
        Ok(self)
    }

    pub fn transpose(matrix: &Matrix) -> Matrix {
        let mut result = Matrix::new(matrix.cols, matrix.rows);
        for i in 0..matrix.rows {
            for j in 0..matrix.cols {
                result.set(j, i, matrix.get(i, j));
            }
        }
        result
    }

    pub fn mul(matrix_1: &Matrix, matrix_2: &Matrix) -> Result<Matrix, String> {
        if matrix_1.cols != matrix_2.rows {
            return Err("Matrix dimensions must match".to_string());
        }
        let mut result = Matrix::new(matrix_1.rows, matrix_2.cols);
        for row in 0..matrix_1.rows {
            for col in 0..matrix_2.cols {
                let elem = (0..matrix_1.cols).map(|i| matrix_1.get(row, i) * matrix_2.get(i, col)).sum();
                result.set(row, col, elem);
            }
        }
        Ok(result)
    }

    pub fn convolve(matrix: &Matrix, kernel: &Matrix, stride: usize, padding: usize) -> Matrix {
        let mut result = Matrix::new((matrix.rows + 2*padding as usize - kernel.rows)/stride as usize + 1,
                                     (matrix.cols + 2*padding as usize - kernel.cols)/stride as usize + 1);
        for i_result in 0..result.rows {
            for j_result in 0..result.cols {
                let mut sum = 0.0;
                // doesnt include padding
                let i_start = i_result*stride;
                let j_start = j_result*stride;
                for i_kernel in 0..kernel.rows {
                    for j_kernel in 0..kernel.cols {
                        if i_start + i_kernel < padding || j_start + j_kernel < padding {
                            continue;
                        }
                        // now it includes padding
                        let i_matrix = i_start + i_kernel - padding;
                        let j_matrix = j_start + j_kernel - padding;
                        if i_matrix >= matrix.rows || j_matrix >= matrix.cols {
                            continue;
                        }
                        sum += matrix.get(i_matrix, j_matrix) * kernel.get(i_kernel, j_kernel);
                    }
                }
                result.set(i_result, j_result, sum);
            }
        }
        result
    }

    pub fn clone(&self) -> Matrix {
        Matrix::from_vec(self.data.clone(), self.rows, self.cols)
    }
}

#[cfg(test)]
mod test_convolutions {
    use super::*;

    #[test]
    fn test_basic() {
        let matrix = Matrix::from_vec((1..10).map(|i| i as f64).collect(), 3, 3);
        let kernel = Matrix::from_vec((10..14).map(|i| i as f64).collect(), 2, 2);
        let result = Matrix::convolve(&matrix, &kernel, 1, 0);
        let expected = Matrix::from_vec(vec![145.0, 191.0, 283.0, 329.0], 2, 2);
        assert!(result.equals(&expected));
    }

    #[test]
    fn test_padding() {
        let matrix = Matrix::from_vec(vec![5.0, 4.0, 1.0, 2.0, 3.0, 4.0], 2, 3);
        let kernel = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let result = Matrix::convolve(&matrix, &kernel, 1, 1);
        let expected = Matrix::from_vec(vec![20.0, 31.0, 16.0, 3.0, 18.0, 31.0, 31.0, 13.0, 4.0, 8.0, 11.0, 4.0], 3, 4);
        assert!(result.equals(&expected));
    }

    #[test]
    fn test_stride() {
        let matrix = Matrix::from_vec((0..16).map(|i| i as f64).collect(), 4, 4);
        let kernel = Matrix::from_vec((0..4).map(|i| i as f64).collect(), 2, 2);
        let result = Matrix::convolve(&matrix, &kernel, 2, 0);
        let expected = Matrix::from_vec(vec![24.0, 36.0, 72.0, 84.0], 2, 2);
        assert!(result.equals(&expected));
    }
}
