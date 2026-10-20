import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter1D:
    def __init__(self, dt=1.0, process_noise=1.0, measurement_noise=10.0):
        # pos += velocity * dt
        self.F = np.array([[1, dt],
                          [0, 1]])

        self.H = np.array([[1, 0]]) # position
        # process noise (movement model)
        self.Q = np.eye(2) * process_noise
        # sensor noise (GPS, camera)
        self.R = np.array([measurement_noise])
        # state 0 and covariance
        self.x = np.zeros((2, 1))
        self.P = np.eye(2) * 1000

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0, 0]

    def update(self, measurement):
        z = np.array([[measurement]])
        y = z - self.H @ self.x

        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P

        return self.x[0, 0]

np.random.seed(42)
dt = 1.0
n_steps =50
true_velocity = 2.0

true_positions = np.array([true_velocity * t for t in range(n_steps)])
measurments = true_positions + np.random.normal(0, 5, n_steps)

kf = KalmanFilter1D(dt=dt, process_noise=0.1, measurement_noise=25.0)

predictions = []
filtered = []

for z in measurments:
    pred = kf.predict()
    predictions.append(pred)
    filt = kf.update(z)
    filtered.append(filt)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

ax = axes[0]
ax.plot(true_positions, 'g-', linewidth=2, label='True pos')
ax.scatter(range(n_steps), measurments, s=15, alpha=0.5, color='red', label='GPS Measurments (noise)')
ax.plot(filtered, 'b-', linewidth=2, label='Kalman filtered')
ax.set_xlabel('Time (steps)')
ax.set_ylabel('Position (m)')
ax.set_title('Kalman Filter 1D - following pos')
ax.legend()
ax.grid(alpha=0.3)

ax2 = axes[1]
error_raw = np.abs(measurments - true_positions)
error_kf = np.abs(np.array(filtered) - true_positions)
ax2.plot(error_raw, 'r-', alpha=0.7, label=f'GPS Error (mean={error_raw.mean():.2f}m)')
ax2.plot(error_kf, 'b-', label=f'Kalman Error (mean={error_kf.mean():.2f}m)')
ax2.set_xlabel('Time (steps)')
ax2.set_ylabel('Absolute Error (m)')
ax2.set_title('Error comparison GPS vs Kalman')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/kalman_1d_0.1_25.0.png', dpi=150)
plt.show()

print(f"Mean error GPS: {error_raw.mean():.2f}m")
print(f"Mean error Kalman: {error_kf.mean():.2f}m")
print(f"Improvement: {(1 - error_kf.mean()/error_raw.mean())*100:.1f}%")