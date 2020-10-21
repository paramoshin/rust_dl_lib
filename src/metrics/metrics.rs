use crate::tensor::tensor::Tensor;

pub fn accuracy_score(y_true: Tensor, y_pred: Tensor) -> f64 {
    let t = y_true
        .data
        .iter()
        .zip(&y_pred.data)
        .filter(|&(a, b)| a == b)
        .count() as f64;
    t / y_true.shape[0] as f64
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_accuracy_score() {
        let y_true_data = vec![1.0, 1.0, 1.0, 1.0];
        let y_true = Tensor::new(y_true_data, &[4]).unwrap();

        let y_pred_data = vec![1.0, 0.0, 1.0, 0.0];
        let y_pred = Tensor::new(y_pred_data, &[4]).unwrap();

        assert_eq!(0.5, accuracy_score(y_true, y_pred))
    }

    #[test]
    fn test_accuracy_score_7() {
        let y_true_data = vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let y_true = Tensor::new(y_true_data, &[10]).unwrap();

        let y_pred_data = vec![1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let y_pred = Tensor::new(y_pred_data, &[10]).unwrap();

        assert_eq!(0.7, accuracy_score(y_true, y_pred))
    }
}
