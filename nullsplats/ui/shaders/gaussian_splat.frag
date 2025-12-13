#version 330 core

// Inputs from vertex shader
in vec2 coordxy;  // local coordinate in quad, unit in pixel
in vec3 splat_color;
in float splat_opacity;
in vec3 conic;  // Conic matrix elements

// Output color
uniform float opacity_bias;

out vec4 FragColor;

void main() {
    // Compute Gaussian power using conic representation
    // Power = -0.5 * (conic.x * x^2 + 2 * conic.y * x * y + conic.z * y^2)
    float power = -0.5 * (conic.x * coordxy.x * coordxy.x + conic.z * coordxy.y * coordxy.y)
                  - conic.y * coordxy.x * coordxy.y;
    
    // Early discard if power is positive
    if (power > 0.0) {
        discard;
    }
    
    // Opacity is already in [0,1] from the PLY; apply bias in linear domain.
    float alpha = clamp(splat_opacity + opacity_bias, 0.0, 1.0);
    
    // Check if contribution is negligible
    float minPower = log(1.0 / 255.0 / alpha);
    if (power < minPower) {
        discard;
    }
    
    // Compute final opacity
    float opacity = min(0.99, alpha * exp(power));
    
    // Output with straight alpha (not premultiplied)
    FragColor = vec4(splat_color, opacity);
}
