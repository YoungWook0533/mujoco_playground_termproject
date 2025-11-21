import jax.numpy as jp

L = 0.2854*2*1.875  # Distance between wheels
R = 0.1651    # radius of wheels

def FK(wheel_vel: jp.ndarray) -> jp.ndarray:
        """Forward Kinematics calculation.

        Args:
            wheel_vel (jp.ndarray): wheel angular velocities [w_left, w_right]

        Returns:
            jp.ndarray: base linear and angular velocity [v, w]
        """
        w_left, w_right = wheel_vel
        v = (w_left + w_right) * R / 2.0
        w = (w_right - w_left) * R / L
        return jp.array([v, w])

def IK(target_vel: jp.ndarray, wheel_vel: jp.ndarray, dt: float) -> jp.ndarray:
        """Inverse Kinematics calculation.

        Args:
            target_vel (jp.ndarray): target linear and angular velocity [v, w]

        Returns:
            jp.ndarray: desired wheel angular velocities [w_left, w_right]
        """
        
        base_vel_lim = jp.array([0.5, 1.0])  # [linear_vel, angular_vel]
        base_acc_lim = jp.array([1.0, 5.0])  # [linear_acc, angular_acc]
        
        v = target_vel[0]
        w = target_vel[1]
        
        # Limit the linear and angular velocities
        v = jp.clip(target_vel[0], -base_vel_lim[0], base_vel_lim[0])
        w = jp.clip(target_vel[1], -base_vel_lim[1], base_vel_lim[1])
        
        base_vel = FK(wheel_vel)                       # [v_now, ω_now]
        delta = jp.array([v, w]) - base_vel
        delta = jp.clip(delta, -base_acc_lim * dt, base_acc_lim * dt)
        v_safe, w_safe = base_vel + delta

        w_left = (v_safe - 0.5 * L * w_safe) / R
        w_right = (v_safe + 0.5 * L * w_safe) / R
        return jp.array([w_left, w_right])