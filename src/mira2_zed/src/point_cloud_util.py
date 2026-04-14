import numpy as np
from typing import List, Tuple, Optional


def is_yellow_lime_green(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Filter points based on yellow to lime green/darker yellow color range.
    Uses np.where for vectorized operations.

    Args:
        r, g, b: Arrays of red, green, blue values (0-255)

    Returns:
        Boolean array indicating which points are in the yellow-green range
    """
    is_yellow = (r > 100) & (g > 100) & (b < 100)
    is_lime = (r > 50) & (g > 150) & (b < 100)
    is_dark_yellow = (r > 80) & (g > 80) & (g < 150) & (b < 50)

    return is_yellow | is_lime | is_dark_yellow


def filter_points_by_color(
    points: np.ndarray, colors: np.ndarray, color_range: str = "yellow_lime_green"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter points based on color range using vectorized np.where operations.

    Args:
        points: Array of shape (N, 3) containing xyz coordinates
        colors: Array of shape (N, 3) containing RGB values (0-255)
        color_range: Color range to filter ('yellow_lime_green', 'all')

    Returns:
        Tuple of (filtered_points, filtered_colors)
    """
    if len(points) == 0:
        return points, colors

    if color_range == "yellow_lime_green":
        r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]
        mask = is_yellow_lime_green(r, g, b)
    else:
        mask = np.ones(len(points), dtype=bool)

    return points[mask], colors[mask]


def separate_clusters_by_color(
    points: np.ndarray, colors: np.ndarray, color_threshold: float = 30.0
) -> List[np.ndarray]:
    """
    Separate point clusters based on color similarity using vectorized operations.

    Args:
        points: Array of shape (N, 3) containing xyz coordinates
        colors: Array of shape (N, 3) containing RGB values (0-255)
        color_threshold: Euclidean distance threshold for color similarity

    Returns:
        List of point arrays, each representing a color-based cluster
    """
    if len(points) == 0:
        return []

    colors_normalized = colors.astype(np.float32) / 255.0
    visited = np.zeros(len(points), dtype=bool)
    clusters = []

    i = 0
    while i < len(points):
        if visited[i]:
            i += 1
            continue

        mask = ~visited
        color_dists = np.linalg.norm(
            colors_normalized[mask] - colors_normalized[i], axis=1
        )
        cluster_indices = np.where(color_dists < (color_threshold / 255.0))[0]

        if len(cluster_indices) == 0:
            cluster_indices = [i]

        original_indices = np.where(mask)[0][cluster_indices]
        visited[original_indices] = True

        if len(original_indices) > 0:
            clusters.append(points[original_indices])

        i += 1

    return clusters


def cluster_with_hdbscan(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    min_cluster_size: int = 10,
    min_samples: int = 5,
    cluster_selection_epsilon: float = 0.0,
    color_weight: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster points using HDBSCAN algorithm with optional color features.

    Args:
        points: Array of shape (N, 3) containing xyz coordinates
        colors: Optional array of shape (N, 3) containing RGB values (0-255)
        min_cluster_size: Minimum number of points to form a cluster
        min_samples: Minimum samples for core points
        cluster_selection_epsilon: Cluster selection epsilon for HDBSCAN
        color_weight: Weight for color features in combined feature vector

    Returns:
        Tuple of (cluster_labels, feature_matrix)
        - cluster_labels: Array of shape (N,) with cluster labels (-1 for noise)
        - feature_matrix: The feature matrix used for clustering
    """
    try:
        import fast_hdbscan
    except ImportError:
        raise ImportError(
            "fast_hdbscan package not installed. Install with: pip install fast-hdbscan"
        )

    if len(points) == 0:
        return np.array([], dtype=int), np.array([])

    features = points.copy()

    if colors is not None:
        colors_normalized = (colors.astype(np.float32) / 255.0) * color_weight
        features = np.hstack([features, colors_normalized])

    clusterer = fast_hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )

    cluster_labels = clusterer.fit_predict(features)

    return cluster_labels, features


def segment_plane_ransac(
    points: np.ndarray,
    distance_threshold: float = 0.01,
    ransac_n: int = 3,
    num_iterations: int = 1000,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Segment plane from point cloud using RANSAC via Open3D.

    Args:
        points: Array of shape (N, 3) containing xyz coordinates
        distance_threshold: Maximum distance a point can have to the plane
        ransac_n: Number of points to randomly sample
        num_iterations: Number of RANSAC iterations

    Returns:
        Tuple of (plane_coefficients, inlier_indices)
        - plane_coefficients: [a, b, c, d] where ax + by + cz + d = 0
        - inlier_indices: Indices of points belonging to the plane
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError(
            "open3d package not installed. Install with: pip install open3d"
        )

    if len(points) < 3:
        return None, None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )

    inlier_indices = np.array(inliers)

    return plane_model, inlier_indices


def segment_multiple_planes(
    points: np.ndarray,
    distance_threshold: float = 0.01,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    min_inliers: int = 100,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Segment multiple planes from point cloud iteratively using RANSAC.

    Args:
        points: Array of shape (N, 3) containing xyz coordinates
        distance_threshold: Maximum distance a point can have to the plane
        ransac_n: Number of points to randomly sample
        num_iterations: Number of RANSAC iterations per plane
        min_inliers: Minimum number of inliers to accept a plane

    Returns:
        List of tuples (plane_coefficients, inlier_indices) for each plane
    """
    remaining_points = points.copy()
    remaining_indices = np.arange(len(points))
    planes = []

    for _ in range(10):
        if len(remaining_points) < min_inliers:
            break

        plane_model, inliers = segment_plane_ransac(
            remaining_points,
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )

        if plane_model is None or len(inliers) < min_inliers:
            break

        planes.append((plane_model, remaining_indices[inliers]))

        mask = np.ones(len(remaining_points), dtype=bool)
        mask[inliers] = False
        remaining_points = remaining_points[mask]
        remaining_indices = remaining_indices[mask]

    return planes


def extract_rgb_from_pointcloud2(data: np.ndarray, fields: List[dict]) -> np.ndarray:
    """
    Extract RGB/RGBA colors from PointCloud2 binary data.

    Args:
        data: Binary data blob from PointCloud2 message
        fields: List of PointField dictionaries

    Returns:
        Array of shape (N, 3) containing RGB values (0-255)
    """
    rgb_field = None
    for field in fields:
        if field["name"] in ("rgb", "rgba"):
            rgb_field = field
            break

    if rgb_field is None:
        return np.array([])

    offset = rgb_field["offset"]
    dtype = np.float32

    rgb_data = np.frombuffer(data, dtype=dtype)
    n_points = len(rgb_data) // 4

    rgb_array = rgb_data.reshape(n_points, 4)

    r = (rgb_array[:, 0] * 255).astype(np.uint8) & 0xFF
    g = (rgb_array[:, 1] * 255).astype(np.uint8) & 0xFF
    b = (rgb_array[:, 2] * 255).astype(np.uint8) & 0xFF

    return np.column_stack([r, g, b])


def extract_xyzi_from_pointcloud2(
    data: np.ndarray, fields: List[dict], point_step: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract xyz coordinates and intensity from PointCloud2 binary data.

    Args:
        data: Binary data blob from PointCloud2 message
        fields: List of PointField dictionaries
        point_step: Length of a point in bytes

    Returns:
        Tuple of (points, intensity)
        - points: Array of shape (N, 3) containing xyz coordinates
        - intensity: Array of shape (N,) containing intensity values
    """
    n_points = len(data) // point_step
    points = np.zeros((n_points, 3), dtype=np.float32)
    intensity = np.zeros(n_points, dtype=np.float32)

    for field in fields:
        name = field["name"]
        offset = field["offset"]
        count = field["count"]

        if name == "x":
            points[:, 0] = np.frombuffer(
                data, dtype=np.float32, count=n_points * count, offset=offset
            )
        elif name == "y":
            points[:, 1] = np.frombuffer(
                data, dtype=np.float32, count=n_points * count, offset=offset
            )
        elif name == "z":
            points[:, 2] = np.frombuffer(
                data, dtype=np.float32, count=n_points * count, offset=offset
            )
        elif name == "intensity":
            intensity = np.frombuffer(
                data, dtype=np.float32, count=n_points * count, offset=offset
            )

    return points, intensity
