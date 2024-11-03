use ndarray;

pub type VecXd = ndarray::Array1<f64>;
pub type MatXd = ndarray::Array2<f64>;

pub trait Derivative {
    fn evaluate(&self, x: f64, y: &VecXd) -> VecXd;
}

pub fn integrate(arr_x: &VecXd, initial_values: &VecXd,  tolerance: f64, obj: &dyn Derivative) -> MatXd {

    let ak: VecXd = ndarray::array![0.0, 2.0/9.0, 1.0/3.0, 3.0/4.0, 1.0, 5.0/6.0];
    let bkl: MatXd = ndarray::array![
        [ 0.0,        0.0,          0.0,        0.0,       0.0       ],
        [ 2.0/9.0,    0.0,          0.0,        0.0,       0.0       ],
	    [ 1.0/12.0,   1.0/4.0,      0.0,        0.0,       0.0       ],
	    [ 69.0/128.0, -243.0/128.0, 135.0/64.0, 0.0,       0.0       ],
	    [ -17.0/12.0, 27.0/4.0,     -27.0/5.0,  16.0/15.0, 0.0       ],
	    [ 65.0/432.0, -5.0/16.0,    13.0/16.0,  4.0/27.0,  5.0/144.0 ],
    ];
    let chk: VecXd = ndarray::array![ 47.0/450.0, 0.0, 12.0/25.0, 32.0/225.0, 1.0/30.0, 6.0/25.0 ];
    let ctk: VecXd = ndarray::array![ 1.0/150.0, 0.0, -3.0/100.0, 16.0/75.0, 1.0/20.0, -6.0/25.0 ];

    let vec_size = initial_values.len();
    let mut k0 = VecXd::zeros(vec_size);
    let mut k1 = VecXd::zeros(vec_size);
    let mut k2 = VecXd::zeros(vec_size);
    let mut k3 = VecXd::zeros(vec_size);
    let mut k4 = VecXd::zeros(vec_size);
    let mut k5 = VecXd::zeros(vec_size);
    let mut vec_error = VecXd::zeros(vec_size);

    let mut h: f64;
    let mut trunc_error: f64;
    let mut y = initial_values.clone();

    let mut output = MatXd::zeros((arr_x.len(), vec_size));
    output.row_mut(0).assign(&y);

    for kk in 1..arr_x.len() {
        let mut x = arr_x[kk-1];
        let end_x = arr_x[kk];
        h = end_x - x;

        while x < end_x {
            trunc_error = 1.0;
            while trunc_error > tolerance {
                k0.assign(&(h * obj.evaluate(x + ak[0] * h, &y)));
                k1.assign(&(h * obj.evaluate(x + ak[1] * h, &(&y + bkl[[1,0]] * &k0))));
                k2.assign(&(h * obj.evaluate(x + ak[2] * h, &(&y + bkl[[2,0]] * &k0 + bkl[[2,1]] * &k1))));
                k3.assign(&(h * obj.evaluate(x + ak[3] * h, &(&y + bkl[[3,0]] * &k0 + bkl[[3,1]] * &k1 + bkl[[3,2]] * &k2))));
                k4.assign(&(h * obj.evaluate(x + ak[4] * h, &(&y + bkl[[4,0]] * &k0 + bkl[[4,1]] * &k1 + bkl[[4,2]] * &k2 + bkl[[4,3]] * &k3))));
                k5.assign(&(h * obj.evaluate(x + ak[5] * h, &(&y + bkl[[5,0]] * &k0 + bkl[[5,1]] * &k1 + bkl[[5,2]] * &k2 + bkl[[5,3]] * &k3 + bkl[[5,4]] * &k4))));

                // Adapting step based on estimated error
                vec_error.assign(&(ctk[0] * &k0 + ctk[1] * &k1 + ctk[2] * &k2 + ctk[3] * &k3 + ctk[4] * &k4 + ctk[5] * &k5));
                trunc_error = vec_error.dot(&vec_error).sqrt();
                h = f64::min(end_x - x, 0.9 * h *(tolerance / trunc_error).powf(0.2));
            }
            // Calculate new position based on results above.
            x += h;
            y.assign(&(&y + chk[0] * &k0 + chk[1] * &k1 + chk[2] * &k2 + chk[3] * &k3 + chk[4] * &k4 + chk[5] * &k5));
        }
        // Save result into output array
        output.row_mut(kk).assign(&y);
    }

    return output;
}
