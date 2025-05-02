from typing import Tuple

import numpy as np
import scipy.linalg

from .kalman_filter import KalmanFilter


class UnscentedKalmanFilter(KalmanFilter):
    """
    An Unscented Kalman Filter implementation based on the standard KalmanFilter class.

    It uses the Unscented Transform to handle potential non-linearities,
    although the base motion and observation models here are linear.
    The state space and models are inherited from the base KalmanFilter class.
    """

    def __init__(self, alpha=1e-3, beta=2.0, kappa=0.0):
        """
        Initializes the Unscented Kalman Filter.

        Args:
            alpha (float): UKF scaling parameter. Controls the spread of sigma points.
                           Typically 1e-4 <= alpha <= 1.
            beta (float): UKF parameter. Used to incorporate prior knowledge of the
                          distribution's type (Gaussian optimal is beta=2).
            kappa (float): UKF secondary scaling parameter. Usually set to 0 or 3-n.
        """
        super().__init__()
        self._n = self._motion_mat.shape[0]  # State dimension (8)
        self._m = self._update_mat.shape[0]  # Measurement dimension (4)

        # UKF parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        # Calculate lambda and weights
        self.lambda_ = self.alpha**2 * (self._n + self.kappa) - self._n
        self._initialize_weights()

    def _initialize_weights(self):
        """Calculate sigma point weights."""
        n = self._n
        lambda_ = self.lambda_

        self.Wc = np.full(2 * n + 1, 1.0 / (2.0 * (n + lambda_)))
        self.Wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lambda_)))

        self.Wc[0] = lambda_ / (n + lambda_) + (1.0 - self.alpha**2 + self.beta)
        self.Wm[0] = lambda_ / (n + lambda_)

    def _sigma_points(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> np.ndarray:
        """
        Generates sigma points for the unscented transform.

        Args:
            mean (ndarray): State mean vector (n_dim).
            covariance (ndarray): State covariance matrix (n_dim x n_dim).

        Returns:
            ndarray: Array of sigma points (2*n+1, n_dim).
        """
        n = self._n
        lambda_ = self.lambda_

        sigmas = np.zeros((2 * n + 1, n))

        try:
            # Use Cholesky decomposition
            # Adding a small epsilon for numerical stability if covariance is near singular
            epsilon = 1e-9
            L = scipy.linalg.cholesky(
                (n + lambda_) * covariance + epsilon * np.eye(n), lower=True
            )
        except scipy.linalg.LinAlgError:
            # Fallback if Cholesky fails (e.g., matrix not positive definite)
            # Use eigenvalue decomposition, handling potential small negative eigenvalues
            eigvals, eigvecs = np.linalg.eigh(covariance)
            eigvals[eigvals < 0] = 0 # Ensure non-negativity
            sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            L = np.sqrt(n + lambda_) * sqrt_cov


        sigmas[0] = mean
        for i in range(n):
            sigmas[i + 1] = mean + L[:, i]
            sigmas[n + i + 1] = mean - L[:, i]

        return sigmas

    def predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Unscented Kalman Filter prediction step.

        Args:
            mean (ndarray): The n dimensional mean vector of the object state
                at the previous time step.
            covariance (ndarray): The nxn dimensional covariance matrix of the
                object state at the previous time step.

        Returns:
            Tuple[ndarray, ndarray]: Returns the mean vector and covariance
                matrix of the predicted state.
        """
        # 1. Generate Sigma Points
        sigmas = self._sigma_points(mean, covariance)

        # 2. Propagate Sigma Points through motion model (f)
        # In this case, f(x) = _motion_mat @ x
        sigmas_pred = sigmas @ self._motion_mat.T

        # 3. Calculate Predicted Mean
        mean_pred = np.sum(self.Wm[:, None] * sigmas_pred, axis=0)

        # 4. Calculate Predicted Covariance
        y = sigmas_pred - mean_pred[None, :]
        # Efficient covariance calculation: sum(Wc[i] * y[i].outer(y[i]))
        cov_pred = y.T @ np.diag(self.Wc) @ y

        # 5. Add Process Noise
        # Calculate motion_cov based on the *original* mean for consistency with base class
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        covariance_pred = cov_pred + motion_cov

        return mean_pred, covariance_pred

    def project(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project state distribution to measurement space using Unscented Transform.

        Args:
            mean (ndarray): The state's mean vector (n dimensional).
            covariance (ndarray): The state's covariance matrix (nxn dimensional).

        Returns:
            Tuple[ndarray, ndarray]: Returns the projected mean and
                covariance matrix in measurement space.
        """
        # 1. Generate Sigma Points for the *given* state (mean, covariance)
        sigmas = self._sigma_points(mean, covariance)

        # 2. Transform Sigma Points through measurement model (h)
        # In this case, h(x) = _update_mat @ x
        sigmas_meas = sigmas @ self._update_mat.T # Shape (2n+1, m)

        # 3. Calculate Predicted Measurement Mean
        meas_mean_pred = np.sum(self.Wm[:, None] * sigmas_meas, axis=0) # Shape (m,)

        # 4. Calculate Predicted Measurement Covariance (Innovation Covariance part 1)
        y_meas = sigmas_meas - meas_mean_pred[None, :] # Shape (2n+1, m)
        cov_meas_pred = y_meas.T @ np.diag(self.Wc) @ y_meas # Shape (m, m)

        # 5. Add Measurement Noise
        # Calculate innovation_cov based on the *input* mean for consistency
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std)) # Shape (m, m)

        projected_cov = cov_meas_pred + innovation_cov

        return meas_mean_pred, projected_cov


    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Unscented Kalman Filter correction step.

        Args:
            mean (ndarray): The predicted state's mean vector (n dimensional).
            covariance (ndarray): The predicted state's covariance matrix (nxn dimensional).
            measurement (ndarray): The m-dimensional measurement vector.

        Returns:
            Tuple[ndarray, ndarray]: Returns the measurement-corrected state
                distribution (mean and covariance).
        """
        # 1. Generate Sigma Points for the *predicted* state (mean, covariance)
        sigmas = self._sigma_points(mean, covariance) # Shape (2n+1, n)

        # 2. Transform Sigma Points through measurement model (h)
        # h(x) = _update_mat @ x
        sigmas_meas = sigmas @ self._update_mat.T # Shape (2n+1, m)

        # 3. Calculate Predicted Measurement Mean
        meas_mean_pred = np.sum(self.Wm[:, None] * sigmas_meas, axis=0) # Shape (m,)

        # 4. Calculate Predicted Measurement Covariance (Innovation Covariance)
        y_meas = sigmas_meas - meas_mean_pred[None, :] # Shape (2n+1, m)
        cov_meas_pred = y_meas.T @ np.diag(self.Wc) @ y_meas # Shape (m, m)

        # Add Measurement Noise (calculated based on the *predicted* mean)
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov_noise = np.diag(np.square(std)) # Shape (m, m)
        cov_meas_pred += innovation_cov_noise

        # 5. Calculate Cross-Covariance between state and measurement
        y_state = sigmas - mean[None, :] # Shape (2n+1, n)
        Pxy = y_state.T @ np.diag(self.Wc) @ y_meas # Shape (n, m)

        # 6. Calculate Kalman Gain
        # K = Pxy * inv(cov_meas_pred)
        # Using solve for potentially better numerical stability than inv()
        try:
            kalman_gain = np.linalg.solve(cov_meas_pred.T, Pxy.T).T # Shape (n, m)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if matrix is singular
            kalman_gain = Pxy @ np.linalg.pinv(cov_meas_pred)


        # 7. Calculate Innovation
        innovation = measurement - meas_mean_pred # Shape (m,)

        # 8. Update State Mean
        new_mean = mean + kalman_gain @ innovation # Shape (n,)

        # 9. Update State Covariance
        # new_covariance = covariance - kalman_gain @ cov_meas_pred @ kalman_gain.T
        # More numerically stable form:
        new_covariance = covariance - kalman_gain @ Pxy.T

        return new_mean, new_covariance

    def multi_predict(
        self, means: np.ndarray, covariances: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run UKF prediction step (Non-vectorized version).

        Args:
            means (ndarray): The NxN_DIM dimensional mean matrix.
            covariances (ndarray): The NxN_DIMxN_DIM dimensional covariance matrices.

        Returns:
            Tuple[ndarray, ndarray]: Returns the predicted means and covariances.
        """
        n_tracks = len(means)
        new_means = np.zeros_like(means)
        new_covariances = np.zeros_like(covariances)
        for i in range(n_tracks):
            new_means[i], new_covariances[i] = self.predict(means[i], covariances[i])
        return new_means, new_covariances

    # initiate method is inherited from KalmanFilter and should work as is.