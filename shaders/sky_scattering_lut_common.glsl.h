shared vec2 sum_intensity[kScatteringLutGroupSampleCount];

uint getPrefixSumIndex(uint sample_idx, uint pass_index_pow2) {
    return (sample_idx & ~(pass_index_pow2 - 1)) - 1;
}

// Parallel inclusive prefix-sum (Hillis-Steele up-sweep) over the shared
// sum_intensity array.  Each pass p doubles the span of accumulated values:
//
// input  : [0, 1, 2, 3, 4, 5, 6, 7, ....]
// pass 1 : [0, 0+1, 2, 2+3, 4, 4+5, 6, 6+7, ....]         (span 2)
// pass 2 : [0, 0+1, 0+1+2, 0+1+2+3, 4, 4+5, 4+5+6, 4+5+6+7, ....]  (span 4)
// pass 3 : [0, 0+1, ..., 0+..+3, 0+..+4, 0+..+5, 0+..+6, 0+..+7]   (span 8)
// ......
//
// CRITICAL: i_pass_pow2 must double each iteration (<<= 1), NOT increment (++).
// The condition `(local_index & i_pass_pow2) != 0` and getPrefixSumIndex both
// rely on i_pass_pow2 being an exact power of 2.  If it is not (e.g. 3, 5, 6…),
// getPrefixSumIndex returns (i & ~(i_pass_pow2-1)) - 1 which can underflow to
// 0xFFFFFFFF for some lanes, causing an out-of-bounds shared-memory read that
// corrupts the entire column.  With the old (broken) LUT geometry every optical
// depth was ~0 so the corruption was invisible; with correct geometry the
// garbage values produce NaN/Inf optical depths and a completely black sky.
void generateIncrementSumArray(uint local_index) {
    for (uint i_pass_pow2 = 1; i_pass_pow2 < kScatteringLutGroupSampleCount; i_pass_pow2 <<= 1) {
        if ((local_index & i_pass_pow2) != 0) {
            sum_intensity[local_index] += sum_intensity[getPrefixSumIndex(local_index, i_pass_pow2)];
        }
        // Barrier INSIDE the loop: each pass reads values written by the previous pass.
        // Without this, threads in different warps/waves can race — a thread in warp 1
        // can begin pass (p+1) while warp 0 is still writing pass p results, producing
        // stale reads and a corrupted prefix sum.  The barrier must be unconditional so
        // ALL threads (not just the ones that executed the if-branch) synchronize before
        // the next iteration.
        memoryBarrierShared();
        barrier();
    }
}
