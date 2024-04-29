pub trait ActivationFunction {
    fn activate(&self, input: &Vec<f64>) -> Vec<f64>{
        input.iter().map(|x| self.function(*x)).collect()
    }
    fn function(&self, x: f64) -> f64;
    fn derivative(&self, x: f64) -> f64;
}

pub struct Sigmoid;
impl ActivationFunction for Sigmoid {
    fn function(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn derivative(&self, x: f64) -> f64 {
        let s = self.function(x);
        s * (1.0 - s)
    }
}

pub struct ReLU;
impl ActivationFunction for ReLU {
    fn function(&self, x: f64) -> f64 {
        if x > 0.0 {
            x
        } else {
            0.0
        }
    }

    fn derivative(&self, x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

pub struct LeakyReLU{
    alpha: f64
}

impl ActivationFunction for LeakyReLU {
    fn function(&self, x: f64) -> f64 {
        if x > 0.0 {
            x
        } else {
            self.alpha * x
        }
    }

    fn derivative(&self, x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            self.alpha
        }
    }
}
pub struct Tanh;
impl ActivationFunction for Tanh {
    fn function(&self, x: f64) -> f64 {
        x.tanh()
    }

    fn derivative(&self, x: f64) -> f64 {
        let t = self.function(x);
        1.0 - t * t
    }
}
