#version 330 core

// Quad vertex positions (per-vertex attribute)
layout (location = 0) in vec2 quad_vertex;  // (-1,-1), (1,-1), (1,1), (-1,1)

// Per-instance attributes (using instanced rendering)
layout (location = 1) in vec3 gaussian_mean;
layout (location = 2) in vec3 gaussian_scale;
layout (location = 3) in vec4 gaussian_rotation;  // quaternion (x,y,z,w)
layout (location = 4) in float gaussian_opacity;
layout (location = 5) in vec3 gaussian_sh_dc;  // DC band of SH (RGB)

// Uniforms
uniform mat4 view;
uniform mat4 projection;
uniform vec3 camera_pos;
uniform float point_scale;   // Global scale multiplier
uniform vec3 scale_bias;     // Log-scale bias for each axis
uniform vec2 viewport_size;  // Actual viewport dimensions

// Outputs to fragment shader
out vec2 coordxy;  // local coordinate in quad, unit in pixel
out vec3 splat_color;
out float splat_opacity;
out vec3 conic;  // Conic matrix elements: (a, b, c) for power calculation

const float SH_C0 = 0.28209479177387814;
const float CULL_THRESHOLD = 1.3;

mat3 computeCov3D(vec3 scale_std, vec4 quat_xyzw) {
    vec4 q = vec4(quat_xyzw.w, quat_xyzw.x, quat_xyzw.y, quat_xyzw.z); // wxyz
    float r = q.x;
    float x = q.y;
    float y = q.z;
    float z = q.w;

    mat3 S = mat3(0.0);
    S[0][0] = scale_std.x;
    S[1][1] = scale_std.y;
    S[2][2] = scale_std.z;

    mat3 R = mat3(
        1.0 - 2.0 * (y*y + z*z), 2.0 * (x*y - r*z), 2.0 * (x*z + r*y),
        2.0 * (x*y + r*z), 1.0 - 2.0 * (x*x + z*z), 2.0 * (y*z - r*x),
        2.0 * (x*z - r*y), 2.0 * (y*z + r*x), 1.0 - 2.0 * (x*x + y*y)
    );

    mat3 M = S * R;
    return transpose(M) * M;
}

vec3 computeCov2D(vec4 mean_view, float focal_x, float focal_y, float tan_fovx, float tan_fovy, mat3 cov3D, mat4 viewmatrix) {
    vec4 t = mean_view;

    float limx = 1.3 * tan_fovx;
    float limy = 1.3 * tan_fovy;
    float txtz = t.x / t.z;
    float tytz = t.y / t.z;
    t.x = clamp(txtz, -limx, limx) * t.z;
    t.y = clamp(tytz, -limy, limy) * t.z;

    mat3 J = mat3(
        focal_x / t.z, 0.0, -(focal_x * t.x) / (t.z * t.z),
        0.0, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
        0.0, 0.0, 0.0
    );
    mat3 W = transpose(mat3(viewmatrix));
    mat3 T = W * J;
    mat3 cov = transpose(T) * transpose(cov3D) * T;

    // Low-pass filter: enforce minimum on-screen size.
    cov[0][0] += 0.3;
    cov[1][1] += 0.3;
    return vec3(cov[0][0], cov[0][1], cov[1][1]);
}

void main() {
    vec3 center = gaussian_mean;
    vec3 scale = gaussian_scale;
    vec4 rot = gaussian_rotation;

    vec3 scale_std = exp(scale + scale_bias) * point_scale;
    mat3 cov3d = computeCov3D(scale_std, rot);

    vec4 view_center = view * vec4(center, 1.0);
    vec4 screen_center = projection * view_center;
    screen_center.xyz = screen_center.xyz / screen_center.w;
    screen_center.w = 1.0;

    if (screen_center.x < -CULL_THRESHOLD || screen_center.x > CULL_THRESHOLD ||
        screen_center.y < -CULL_THRESHOLD || screen_center.y > CULL_THRESHOLD ||
        screen_center.z < -CULL_THRESHOLD || screen_center.z > CULL_THRESHOLD) {
        gl_Position = vec4(0.0, 0.0, -1.0, 0.0);
        return;
    }

    float tan_half_y = 1.0 / max(abs(projection[1][1]), 1e-6);
    float tan_half_x = tan_half_y * (viewport_size.x / max(viewport_size.y, 1.0));
    float focal = viewport_size.y / (2.0 * tan_half_y);

    vec3 cov2d = computeCov2D(
        view_center,
        focal,
        focal,
        tan_half_x,
        tan_half_y,
        cov3d,
        view
    );

    float det = (cov2d.x * cov2d.z - cov2d.y * cov2d.y);
    if (det <= 0.0) {
        gl_Position = vec4(0.0, 0.0, -1.0, 0.0);
        return;
    }
    float det_inv = 1.0 / det;
    conic = vec3(cov2d.z * det_inv, -cov2d.y * det_inv, cov2d.x * det_inv);

    vec2 quadwh_scr = vec2(3.0 * sqrt(cov2d.x), 3.0 * sqrt(cov2d.z));
    vec2 wh = vec2(2.0 * tan_half_x * focal, 2.0 * tan_half_y * focal);
    vec2 quadwh_ndc = quadwh_scr / wh * 2.0;

    screen_center.xy = screen_center.xy + quad_vertex * quadwh_ndc;
    coordxy = quad_vertex * quadwh_scr;
    gl_Position = screen_center;

    vec3 color_linear = 0.5 + SH_C0 * gaussian_sh_dc;
    splat_color = clamp(color_linear, 0.0, 1.0);

    splat_opacity = gaussian_opacity;
}
