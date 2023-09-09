from jax import numpy as jnp


def ypr_to_rotation_matrix(angles):
    yaw, pitch, roll = angles  # convert degrees to radian

    # rotation matrix for yaw
    R_yaw = jnp.array(
        [
            [jnp.cos(yaw), -jnp.sin(yaw), 0],
            [jnp.sin(yaw), jnp.cos(yaw), 0],
            [0, 0, 1],
        ]
    )

    # rotation matrix for pitch
    R_pitch = jnp.array(
        [
            [jnp.cos(pitch), 0, jnp.sin(pitch)],
            [0, 1, 0],
            [-jnp.sin(pitch), 0, jnp.cos(pitch)],
        ]
    )

    # rotation matrix for roll
    R_roll = jnp.array(
        [
            [1, 0, 0],
            [0, jnp.cos(roll), -jnp.sin(roll)],
            [0, jnp.sin(roll), jnp.cos(roll)],
        ]
    )

    # Combined rotation matrix is R = R_yaw * R_pitch * R_roll
    R = jnp.dot(R_yaw, jnp.dot(R_pitch, R_roll))

    return R


def euler_to_rotation_matrix(theta):
    R_x = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, jnp.cos(theta[0]), -jnp.sin(theta[0])],
            [0.0, jnp.sin(theta[0]), jnp.cos(theta[0])],
        ]
    )

    R_y = jnp.array(
        [
            [jnp.cos(theta[1]), 0, jnp.sin(theta[1])],
            [0, 1, 0],
            [-jnp.sin(theta[1]), 0, jnp.cos(theta[1])],
        ]
    )

    R_z = jnp.array(
        [
            [jnp.cos(theta[2]), -jnp.sin(theta[2]), 0],
            [jnp.sin(theta[2]), jnp.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )

    R = jnp.dot(R_z, jnp.dot(R_y, R_x))

    return R


def tag_to_camera_coordinates(tag_position, camera_position, camera_orientation):
    # Compute the rotation matrix from Yaw, Pitch, Roll
    R = camera_orientation.as_matrix()

    # Translate to the camera's coordinate system
    tag_position = tag_position.reshape(-1, 1)  # Make it a column vector
    camera_position = camera_position.reshape(-1, 1)  # Make it a column vector
    tag_in_camera_coordinates_xyz = jnp.dot(R, tag_position - camera_position)

    # Permute axes to get (u, v, w) from (x, y, z)
    permutation_matrix = jnp.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    tag_in_camera_coordinates_uvw = jnp.dot(
        permutation_matrix, tag_in_camera_coordinates_xyz
    ).flatten() * jnp.array([1, -1, 1])

    return tag_in_camera_coordinates_uvw.reshape(-1, 1)


@staticmethod
def project_to_image(tag_in_camera_coordinates, K):
    jnp.vstack([tag_in_camera_coordinates, 1])

    # Perform projection
    projected_homogeneous = jnp.dot(K, tag_in_camera_coordinates)

    # Convert to inhomogeneous coordinates
    projected_position = projected_homogeneous[:-1] / projected_homogeneous[-1]

    return projected_position


def apply_distortion(uv_pixel, K, distortion_coeffs):
    """
    Apply radial and tangential distortion to a pixel point (u, v).
    :param uv_pixel: 2x1 array containing the pixel coordinates [u, v]
    :param K: 3x3 camera intrinsic matrix
    :param distortion_coeffs: 1x5 array containing the distortion coefficients
        [k1, k2, p1, p2, k3]
    :return: 2x1 array containing the distorted pixel coordinates
        [u_dist_pixel, v_dist_pixel]
    """

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Normalize the pixel coordinates to metric coordinates
    u, v = uv_pixel
    u_normalized = (u - cx) / fx
    v_normalized = (v - cy) / fy

    # Apply distortion
    u_dist_normalized, v_dist_normalized = apply_distortion_normalized(
        jnp.array([u_normalized, v_normalized]), distortion_coeffs
    )

    # Convert back to pixel coordinates
    u_dist_pixel = u_dist_normalized * fx + cx
    v_dist_pixel = v_dist_normalized * fy + cy

    return jnp.array([u_dist_pixel, v_dist_pixel])


def apply_distortion_normalized(uv, distortion_coeffs):
    """
    Apply radial and tangential distortion to normalized metric coordinates (u, v).
    This function assumes u and v are already normalized to be relative to the
    aprincipal point and focal length.
    """

    u, v = uv
    r2 = u**2 + v**2

    k1, k2, p1, p2, k3 = distortion_coeffs

    # Radial distortion
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3

    # Tangential distortion
    dx = 2 * p1 * u * v + p2 * (r2 + 2 * u**2)
    dy = p1 * (r2 + 2 * v**2) + 2 * p2 * u * v

    u_dist = u * radial + dx
    v_dist = v * radial + dy

    return jnp.array([u_dist, v_dist])
