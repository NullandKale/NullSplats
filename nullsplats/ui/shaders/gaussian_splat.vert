#version 330 core

// Quad vertex positions (per-vertex attribute)
layout (location = 0) in vec2 quad_vertex;  // (-1,-1), (1,-1), (1,1), (-1,1)

// Per-instance attributes (using instanced rendering)
layout (location = 1) in vec3 gaussian_mean;
layout (location = 2) in vec3 gaussian_scale;
layout (location = 3) in vec4 gaussian_rotation;  // quaternion
layout (location = 4) in float gaussian_opacity;
layout (location = 5) in vec3 gaussian_sh_dc;  // DC band of SH (RGB)

// Uniforms
uniform mat4 view;
uniform mat4 projection;
uniform vec3 camera_pos;
uniform float point_scale;  // Global scale multiplier
uniform vec3 scale_bias;    // Log-scale bias for each axis
uniform vec2 viewport_size;  // Actual viewport dimensions

// Outputs to fragment shader
out vec2 coordxy;  // local coordinate in quad, unit in pixel
out vec3 splat_color;
out float splat_opacity;
out vec3 conic;  // Conic matrix elements: (a, b, c) for power calculation

void main() {
    // Get instance-specific Gaussian parameters
    vec3 center = gaussian_mean;
    vec3 scale = gaussian_scale;
    vec4 rot = gaussian_rotation;
    
    // Build rotation matrix from quaternion
    // q = (w, x, y, z) but we store as (x, y, z, w)
    float qx = rot.x;
    float qy = rot.y;
    float qz = rot.z;
    float qw = rot.w;
    
    // Quaternion to rotation matrix
    mat3 R = mat3(
        1.0 - 2.0 * (qy*qy + qz*qz), 2.0 * (qx*qy - qw*qz), 2.0 * (qx*qz + qw*qy),
        2.0 * (qx*qy + qw*qz), 1.0 - 2.0 * (qx*qx + qz*qz), 2.0 * (qy*qz - qw*qx),
        2.0 * (qx*qz - qw*qy), 2.0 * (qy*qz + qw*qx), 1.0 - 2.0 * (qx*qx + qy*qy)
    );
    
    // Build scale matrix (scales ARE stored as log in PLY, must exponentiate)
    // Apply point_scale multiplier for user control
    vec3 exp_scale = exp((scale + scale_bias) * 0.5) * point_scale;
    mat3 S = mat3(
        exp_scale.x, 0.0, 0.0,
        0.0, exp_scale.y, 0.0,
        0.0, 0.0, exp_scale.z
    );
    
    // Build 3D covariance matrix: Σ = R * S * S^T * R^T
    mat3 RS = R * S;
    mat3 cov3d = RS * transpose(RS);
    
    // Transform center to view space
    vec4 view_center = view * vec4(center, 1.0);
    vec3 t = view_center.xyz;
    
    // Compute Jacobian of perspective projection
    // For perspective: J = [[fx/z, 0, -fx*x/z^2], [0, fy/z, -fy*y/z^2]]
    float focal_x = projection[0][0] * viewport_size.x * 0.5;
    float focal_y = projection[1][1] * viewport_size.y * 0.5;
    
    if (t.z >= 0.0) {
        // Gaussian is behind the camera
        gl_Position = vec4(0.0, 0.0, -10.0, 1.0);
        splat_opacity = 0.0;
        return;
    }
    float depth = max(-t.z, 1e-4);

    // Build 2x3 Jacobian matrix (stored as 3x3 with third row for convenience)
    mat3 J = mat3(
        focal_x / depth, 0.0, 0.0,
        0.0, focal_y / depth, 0.0,
        -(focal_x * t.x) / (depth * depth), -(focal_y * t.y) / (depth * depth), 0.0
    );
    
    // Extract view matrix rotation (upper 3x3)
    mat3 W = mat3(view);
    
    // Project covariance to 2D: Σ' = J * W * Σ * W^T * J^T
    mat3 T = J * W;
    mat3 Vrk = T * cov3d * transpose(T);
    
    // Extract 2x2 covariance in screen space
    mat2 cov2d = mat2(
        Vrk[0][0], Vrk[0][1],
        Vrk[1][0], Vrk[1][1]
    );
    
    // Add a small value to diagonal for numerical stability
    cov2d[0][0] += 0.3;
    cov2d[1][1] += 0.3;
    
    // Compute determinant and inverse of 2D covariance
    float det = cov2d[0][0] * cov2d[1][1] - cov2d[0][1] * cov2d[1][0];
    
    if (det <= 0.0) {
        // Degenerate covariance, skip this Gaussian
        gl_Position = vec4(0.0, 0.0, -10.0, 1.0);  // Off screen
        splat_opacity = 0.0;
        return;
    }

    float det_inv = 1.0 / det;
    
    // Inverse of 2D covariance (for Gaussian evaluation)
    mat2 cov2d_inv = mat2(
        cov2d[1][1] * det_inv, -cov2d[0][1] * det_inv,
        -cov2d[1][0] * det_inv, cov2d[0][0] * det_inv
    );
    
    // Store conic form for power calculation
    conic.x = cov2d_inv[0][0];
    conic.y = cov2d_inv[0][1];
    conic.z = cov2d_inv[1][1];
    
    // Compute extent of Gaussian (3 sigma)
    float mid = 0.5 * (cov2d[0][0] + cov2d[1][1]);
    float delta = sqrt(max(0.1, mid * mid - det));
    float lambda_max = max(0.0, mid + delta);
    float radius_pixels = 3.0 * sqrt(lambda_max);
    radius_pixels = ceil(max(radius_pixels, 1e-4));
    radius_pixels = max(radius_pixels, 1.0);
    vec2 quadwh_scr = vec2(radius_pixels);
    vec2 quadwh_ndc = quadwh_scr / viewport_size * 2.0;
    
    // Project center to screen space
    vec4 screen_center = projection * view_center;
    screen_center /= screen_center.w;  // Perspective divide
    screen_center.y = -screen_center.y;
    
    // Position quad vertex in NDC space
    screen_center.xy = screen_center.xy + quad_vertex * quadwh_ndc;
    
    // Pass pixel coordinates to fragment shader
    coordxy = quad_vertex * quadwh_scr;
    coordxy.y = -coordxy.y;
    
    gl_Position = screen_center;
    
    // Convert SH DC to color (logit -> sigmoid) so DC band matches gsplat expectations
    const float SH_C0 = 0.28209479177387814;
    vec3 color_linear = 0.5 + SH_C0 * gaussian_sh_dc;
    splat_color = clamp(color_linear, 0.0, 1.0);
    
    // Pass opacity directly (will be sigmoid-activated in fragment shader)
    splat_opacity = gaussian_opacity;
}
