float getKValue(int l, int m)
{
    const float s_factorial_serials[9] =
    { 1.0f, 1.0f, 2.0f, 6.0f, 24.0f, 120.0f, 720.0f, 5040.0f, 40320.0f };
    float factor = sqrt((2 * l + 1) / (4.0f * PI));
    return sqrt(factor * s_factorial_serials[l - abs(m)] / s_factorial_serials[l + abs(m)]);
}

void fillPVauleTablle(inout float p_value[15], float x) {
    float x2 = x * x;
    float x2_s7 = 7.0f * x2;
    float x2_1 = 1.0f - x2;
    float x2_1sqrt = sqrt(x2_1);
    float x2_1_2_3rd = x2_1sqrt * x2_1;
    float inv_x2_1_2_3rd = 1.0f / x2_1_2_3rd;
    float x2_1sqr = x2_1 * x2_1;

    // l = 0, m = 0
    p_value[0] = 1.0f;
    // l = 1, m = 0
    p_value[1] = x;
    // l = 1, m = 1
    p_value[2] = -x2_1sqrt;
    float a00 = x * x2_1sqrt;
    // l = 2, m = 0
    p_value[3] = 0.5f * (3.0f * x2 - 1.0f);
    // l = 2, m = 1
    p_value[4] = -3.0f * a00;
    // l = 2, m = 2
    p_value[5] = 3.0f * x2_1;
    float a01 = (-5.0f * x2 + 1.0f) * x2_1sqrt;
    // l = 3, m = 0
    p_value[6] = 0.5f * x * (5.0f * x2 - 3.0f);
    // l = 3, m = 1
    p_value[7] = 1.5f * a01;
    // l = 3, m = 2
    p_value[8] = 15.0f * x * x2_1;
    // l = 3, m = 3
    p_value[9] = -15.0f * x2_1_2_3rd;
    float a02 = (-x2_s7 + 3.0f) * a00;
    // l = 4, m = 0
    p_value[10] = 0.125f * ((35.0f * x2 - 30.0f) * x2 + 3.0f);
    // l = 4, m = 1
    p_value[11] = 2.5f * a02;
    // l = 4, m = 2
    p_value[12] = 7.5f * (x2_s7 - 1.0f) * x2_1;
    // l = 4, m = 3
    p_value[13] = -105.0f * x * x2_1_2_3rd;
    // l = 4, m = 4
    p_value[14] = 105.0f * x2_1sqr;
}

void fillPreCalculateCoeffs(inout float coeffs[15], float theta) {
    const float sqrt2 = sqrt(2.0f);
    float cos_theta = cos(theta);

    float p_value[15];
    fillPVauleTablle(p_value, cos_theta);

    coeffs[0] = getKValue(0, 0) * p_value[0]; // l = 0, m = 0
    coeffs[1] = getKValue(1, 0) * p_value[1]; // l = 1, m = 0
    coeffs[2] = sqrt2 * getKValue(1, 1) * p_value[2]; // l = 1, m = 1
    coeffs[3] = getKValue(2, 0) * p_value[3]; // l = 2, m = 0
    coeffs[4] = sqrt2 * getKValue(2, 1) * p_value[4]; // l = 2, m = 1
    coeffs[5] = sqrt2 * getKValue(2, 2) * p_value[5]; // l = 2, m = 2
    coeffs[6] = getKValue(3, 0)* p_value[6]; // l = 3, m = 0
    coeffs[7] = sqrt2 * getKValue(3, 1) * p_value[7]; // l = 3, m = 1
    coeffs[8] = sqrt2 * getKValue(3, 2) * p_value[8]; // l = 3, m = 2
    coeffs[9] = sqrt2 * getKValue(3, 3) * p_value[9]; // l = 3, m = 3
    coeffs[10] = getKValue(4, 0)* p_value[10]; // l = 4, m = 0
    coeffs[11] = sqrt2 * getKValue(4, 1) * p_value[11]; // l = 4, m = 1
    coeffs[12] = sqrt2 * getKValue(4, 2) * p_value[12]; // l = 4, m = 2
    coeffs[13] = sqrt2 * getKValue(4, 3) * p_value[13]; // l = 4, m = 3
    coeffs[14] = sqrt2 * getKValue(4, 4) * p_value[14]; // l = 4, m = 4
}

void fillYVauleTablle(inout float y_value[25], float theta, float phi) {
    const float sqrt2 = sqrt(2.0f);

    float cos_theta = cos(theta);
    float sin_phi = sin(phi);
    float cos_phi = cos(phi);
    float sin_2phi = sin(2.0f * phi);
    float cos_2phi = cos(2.0f * phi);
    float sin_3phi = sin(3.0f * phi);
    float cos_3phi = cos(3.0f * phi);
    float sin_4phi = sin(4.0f * phi);
    float cos_4phi = cos(4.0f * phi);

    float p_value[15];
    fillPVauleTablle(p_value, cos_theta);

    float a11 = sqrt2 * getKValue(1, 1) * p_value[2];
    float a22 = sqrt2 * getKValue(2, 2) * p_value[5];
    float a21 = sqrt2 * getKValue(2, 1) * p_value[4];
    float a33 = sqrt2 * getKValue(3, 3) * p_value[9];
    float a32 = sqrt2 * getKValue(3, 2) * p_value[8];
    float a31 = sqrt2 * getKValue(3, 1) * p_value[7];
    float a44 = sqrt2 * getKValue(4, 4) * p_value[14];
    float a43 = sqrt2 * getKValue(4, 3) * p_value[13];
    float a42 = sqrt2 * getKValue(4, 2) * p_value[12];
    float a41 = sqrt2 * getKValue(4, 1) * p_value[11];

    // l = 0, m = 0
    y_value[0] = getKValue(0, 0) * p_value[0];
    // l = 1, m = -1
    y_value[1] = sin_phi * a11;
    // l = 1, m = 0
    y_value[2] = getKValue(1, 0) * p_value[1];
    // l = 1, m = 1
    y_value[3] = cos_phi * a11;
    // l = 2, m = -2
    y_value[4] = sin_2phi * a22;
    // l = 2, m = -1
    y_value[5] = sin_phi * a21;
    // l = 2, m = 0
    y_value[6] = getKValue(2, 0) * p_value[3];
    // l = 2, m = 1
    y_value[7] = cos_phi * a21;
    // l = 2, m = 2
    y_value[8] = cos_2phi * a22;
    // l = 3, m = -3
    y_value[9] = sin_3phi * a33;
    // l = 3, m = -2
    y_value[10] = sin_2phi * a32;
    // l = 3, m = -1
    y_value[11] = sin_phi * a31;
    // l = 3, m = 0
    y_value[12] = getKValue(3, 0) * p_value[6];
    // l = 3, m = 1
    y_value[13] = cos_phi * a31;
    // l = 3, m = 2
    y_value[14] = cos_2phi * a32;
    // l = 3, m = 3
    y_value[15] = cos_3phi * a33;
    // l = 4, m = -4
    y_value[16] = sin_4phi * a44;
    // l = 4, m = -3
    y_value[17] = sin_3phi * a43;
    // l = 4, m = -2
    y_value[18] = sin_2phi * a42;
    // l = 4, m = -1
    y_value[19] = sin_phi * a41;
    // l = 4, m = 0
    y_value[20] = getKValue(4, 0) * p_value[10];
    // l = 4, m = 1
    y_value[21] = cos_phi * a41;
    // l = 4, m = 2
    y_value[22] = cos_2phi * a42;
    // l = 4, m = 3
    y_value[23] = cos_3phi * a43;
    // l = 4, m = 4
    y_value[24] = cos_4phi * a44;
}

void fillYVauleTablle(inout float y_value[25], in float pre_calculate_coeffs[15], float phi) {
    float sin_phi = sin(phi);
    float cos_phi = cos(phi);
    float sin_2phi = sin(2.0f * phi);
    float cos_2phi = cos(2.0f * phi);
    float sin_3phi = sin(3.0f * phi);
    float cos_3phi = cos(3.0f * phi);
    float sin_4phi = sin(4.0f * phi);
    float cos_4phi = cos(4.0f * phi);

    float a11 = pre_calculate_coeffs[2];
    float a22 = pre_calculate_coeffs[5];
    float a21 = pre_calculate_coeffs[4];
    float a33 = pre_calculate_coeffs[9];
    float a32 = pre_calculate_coeffs[8];
    float a31 = pre_calculate_coeffs[7];
    float a44 = pre_calculate_coeffs[14];
    float a43 = pre_calculate_coeffs[13];
    float a42 = pre_calculate_coeffs[12];
    float a41 = pre_calculate_coeffs[11];

    // l = 0, m = 0
    y_value[0] = pre_calculate_coeffs[0];
    // l = 1, m = -1
    y_value[1] = sin_phi * a11;
    // l = 1, m = 0
    y_value[2] = pre_calculate_coeffs[1];
    // l = 1, m = 1
    y_value[3] = cos_phi * a11;
    // l = 2, m = -2
    y_value[4] = sin_2phi * a22;
    // l = 2, m = -1
    y_value[5] = sin_phi * a21;
    // l = 2, m = 0
    y_value[6] = pre_calculate_coeffs[3];
    // l = 2, m = 1
    y_value[7] = cos_phi * a21;
    // l = 2, m = 2
    y_value[8] = cos_2phi * a22;
    // l = 3, m = -3
    y_value[9] = sin_3phi * a33;
    // l = 3, m = -2
    y_value[10] = sin_2phi * a32;
    // l = 3, m = -1
    y_value[11] = sin_phi * a31;
    // l = 3, m = 0
    y_value[12] = pre_calculate_coeffs[6];
    // l = 3, m = 1
    y_value[13] = cos_phi * a31;
    // l = 3, m = 2
    y_value[14] = cos_2phi * a32;
    // l = 3, m = 3
    y_value[15] = cos_3phi * a33;
    // l = 4, m = -4
    y_value[16] = sin_4phi * a44;
    // l = 4, m = -3
    y_value[17] = sin_3phi * a43;
    // l = 4, m = -2
    y_value[18] = sin_2phi * a42;
    // l = 4, m = -1
    y_value[19] = sin_phi * a41;
    // l = 4, m = 0
    y_value[20] = pre_calculate_coeffs[10];
    // l = 4, m = 1
    y_value[21] = cos_phi * a41;
    // l = 4, m = 2
    y_value[22] = cos_2phi * a42;
    // l = 4, m = 3
    y_value[23] = cos_3phi * a43;
    // l = 4, m = 4
    y_value[24] = cos_4phi * a44;
}

// ray is normalized vector. org is origin of ray within [-1, 1] range.
float getIntersection(vec2 org, vec2 ray)
{
    vec2 ray_offset = (vec2(1.0f) - sign(abs(ray))) * 1e-20;
    vec2 adj_ray = ray + ray_offset;
    vec2 inv_ray = 1.0f / adj_ray;
    vec2 factor_a = abs(inv_ray);
    vec2 factor_b = -org * inv_ray;
    vec2 factor = factor_a + factor_b;
    return min(factor.x, factor.y);
}

float getIntersection(vec3 org, vec3 ray)
{
    vec3 ray_offset = (vec3(1.0f) - sign(abs(ray))) * 1e-20;
    vec3 adj_ray = ray + ray_offset;
    vec3 inv_ray = 1.0f / adj_ray;
    vec3 factor_a = abs(inv_ray);
    vec3 factor_b = -org * inv_ray;
    vec3 factor = factor_a + factor_b;
    return min(min(factor.x, factor.y), factor.z);
}

vec2 getIntersection(vec2 org, vec2 inv_ray, vec2 box_min, vec2 box_max)
{
    vec2 temp_t_min = (box_min - org) * inv_ray;
    vec2 temp_t_max = (box_max - org) * inv_ray;

    vec2 t_min = min(temp_t_min, temp_t_max);
    vec2 t_max = max(temp_t_min, temp_t_max);

    vec2 t_result = max(vec2(max(t_min.x, t_min.y), min(t_max.x, t_max.y)), 0.0f);

    return t_result.x > t_result.y ? vec2(0.0f) : t_result;
}

float getAngle(ivec2 coords) {
    return atan(coords.y, coords.x == 0 ? 0.01f : coords.x);
}

float alignAngle(float input_angle, float reference_angle) {
    float delta_angle =
        input_angle - reference_angle;

    float result = input_angle;
    if (delta_angle > PI) {
        result = input_angle - 2.0f * PI;
    }
    else if (delta_angle < -PI) {
        result = input_angle + 2.0f * PI;
    }

    return result;
}
