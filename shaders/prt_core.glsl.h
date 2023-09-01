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
    p_value[10] = 0.625f * ((x2_s7 - 6.0f) * x2 + 3.0f);
    // l = 4, m = 1
    p_value[11] = 2.5f * a02;
    // l = 4, m = 2
    p_value[12] = 7.5f * (x2_s7 - 1.0f) * x2_1;
    // l = 4, m = 3
    p_value[13] = -105.0f * x * x2_1_2_3rd;
    // l = 4, m = 4
    p_value[14] = 105.0f * x2_1sqr;
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
    vec3 inv_ray = 1.0f / ray;
    vec3 factor_a = abs(inv_ray);
    vec3 factor_b = -org * inv_ray;
    vec3 factor = factor_a + factor_b;
    return min(min(factor.x, factor.y), factor.z);
}