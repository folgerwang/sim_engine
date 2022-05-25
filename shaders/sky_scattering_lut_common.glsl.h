shared vec2 sum_intensity[kScatteringLutGroupSampleCount];

uint getPrefixSumIndex(uint sample_idx, uint pass_index_pow2) {
    return (sample_idx & ~(pass_index_pow2 - 1)) - 1;
}

// input array [0, 1, 2, 3, 4, 5, 6, 7, ....]
// 1st pass : [0, 0+1, 2, 2+3, 4, 4+5, 6, 6+7, ....]
// 2nd pass : [0, 0+1, 0+1+2, 0+1+2+3, 4, 4+5, 4+5+6, 4+5+6+7, ....]
// 3rd pass : [0, 0+1, 0+1+2, 0+1+2+3, 0+1+2+3+4, 0+1+2+3+4+5, ....]
// ......
void generateIncrementSumArray(uint local_index) {
    for (uint i_pass_pow2 = 1; i_pass_pow2 < kScatteringLutGroupSampleCount; i_pass_pow2++) {
        if ((local_index & i_pass_pow2) != 0) {
            sum_intensity[local_index] += sum_intensity[getPrefixSumIndex(local_index, i_pass_pow2)];
        }
    }
    memoryBarrierShared();
    barrier();
}
