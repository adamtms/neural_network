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

    pub fn new_random(rows: usize, cols: usize) -> Matrix {
        let mut rng = rand::thread_rng();
        let data  = (0..rows*cols).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect();
        Matrix{rows, cols, data}
    }

    pub fn from_vec(data: Vec<f64>, rows: usize, cols: usize) -> Matrix {
        Matrix {rows, cols, data}
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row*self.cols + col]
    }

    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        self.data[row*self.cols + col] = value;
    }

    pub fn add_matrix(&mut self, other: &Matrix) -> Result<(), String>{
        if self.rows != other.rows || self.cols != other.cols {
            return Err("Matrix dimensions must match".to_string());
        }

        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i*self.cols + j] += other.data[i*self.cols + j];
            }
        }
        Ok(())
    }

    pub fn sub_matrix(&mut self, other: &Matrix) -> Result<(), String>{
        if self.rows != other.rows || self.cols != other.cols {
            return Err("Matrix dimensions must match".to_string());
        }

        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i*self.cols + j] -= other.data[i*self.cols + j];
            }
        }
        Ok(())
    }

    pub fn add_scalar(&mut self, scalar: f64) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i*self.cols + j] += scalar;
            }
        }
    }

    pub fn sub_scalar(&mut self, scalar: f64) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i*self.cols + j] -= scalar;
            }
        }
    }

    pub fn elementwise_mul(&mut self, other: &Matrix) -> Result<(), String> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err("Matrix dimensions must match".to_string());
        }
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i*self.cols + j] *= other.data[i*self.cols + j];
            }
        }
        Ok(())
    }

    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j));
            }
        }
        result
    }

    pub fn map(&mut self, f: fn(f64) -> f64) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i*self.cols + j] = f(self.data[i*self.cols + j]);
            }
        }
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
}
