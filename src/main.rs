use std::fs;
use std::io::{Write, BufWriter};

use ndarray::array;

mod runge_kutta;
use runge_kutta as rk;


struct Logistic {
    growth_rate: f64,
    capacity: f64,
}

impl rk::Derivative for Logistic {
    fn evaluate(&self, _: f64, y: &rk::Array1d) -> rk::Array1d {
         return self.growth_rate * y * (1.0 - y / self.capacity);
    }
}


fn main() {
    let arr_x = rk::Array1d::linspace(0.0, 25.0, 21);
    let initial_values: rk::Array1d = array![0.01];

    let obj = Logistic {
        growth_rate: 0.5,
        capacity: 1.0,
    };

    let solution = runge_kutta::integrate(&arr_x, &initial_values, 0.01, &obj);
    let res = solution.column(0);

    let file = fs::File::create("/home/gmonteir/integrate/res.txt").unwrap();
    let mut writer = BufWriter::new(file);
    for kk in 0..arr_x.len() {
        writeln!(writer, "{} {}", arr_x[kk], res[kk]).unwrap();
    }
}
