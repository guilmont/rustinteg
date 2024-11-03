use ndarray::array;

use rustinteg::runge_kutta::{self, VecXd};

struct Logistic {
    growth_rate: f64,
    capacity: f64,
}

impl runge_kutta::Derivative for Logistic {
    fn evaluate(&self, _: f64, y: &VecXd) -> VecXd {
         return self.growth_rate * y * (1.0 - y / self.capacity);
    }
}

fn main() {
    let arr_x = VecXd::linspace(0.0, 25.0, 21);
    let initial_values: VecXd = array![0.01];

    let obj = Logistic {
        growth_rate: 0.5,
        capacity: 1.0,
    };

    let solution = runge_kutta::integrate(&arr_x, &initial_values, 0.01, &obj);
    let res = solution.column(0);

    for kk in 0..arr_x.len() {
        println!("{} {}", arr_x[kk], res[kk]);
    }
}
