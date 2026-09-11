// tile_detail.glsl.h — runtime 1 m terrain detail sampling.
//
// Include AFTER global_definition.glsl.h in a shader that has
// TILE_PARAMS_SET bound with the tile resource descriptor set
// (TerrainDetailStream provides the two bindings below).
//
// The world is split into kDetailTilesPerSide^2 detail tiles of
// kDetailTileMeters; a 5x5 ring around the camera is resident in
// detail_height_tiles (R16 array, kDetailTileRes^2 per layer, texel
// centers on integer world meters).  detail_slot_map maps world tile
// index -> array layer (-1 = not resident).
//
// Heights blend detail -> base over kDetailFadeStart/EndMeters of camera
// distance, so both the resident-region boundary and LOD changes stay
// smooth.  Detail tiles are bit-identical along shared borders (see
// terrain_detail_worker.py), so no cross-tile blending is needed.

layout(set = TILE_PARAMS_SET, binding = TERRAIN_DETAIL_HEIGHT_INDEX)
    uniform sampler2DArray detail_height_tiles;
layout(std430, set = TILE_PARAMS_SET, binding = TERRAIN_DETAIL_TABLE_INDEX)
    readonly buffer TerrainDetailTable {
    // Layout mirrors TerrainDetailStream::TableCpu.
    int detail_slot_map[kDetailTilesPerSide * kDetailTilesPerSide];
    int detail_color_slot[kDetailTilesPerSide * kDetailTilesPerSide];
    // Separate from detail_color_slot even though both are written by
    // the same tile load: the surface tile is produced by a later stage
    // of the worker than the colour tile, so a world generated before
    // surface tiles existed has colour on disk and no _surf.png beside
    // it.  One shared slot index would then either lose the colour or
    // claim a surface that is not there.
    int detail_surf_slot[kDetailTilesPerSide * kDetailTilesPerSide];
};

// 0 = pure base map, 1 = pure detail.
float terrainDetailFade(vec2 pos_xz_ws, vec3 camera_pos_ws) {
    float d = distance(camera_pos_ws.xz, pos_xz_ws);
    return 1.0f - smoothstep(kDetailFadeStartMeters,
                             kDetailFadeEndMeters, d);
}

// Returns the rendered terrain height (meters) at a world XZ position.
// base_h: the base rock-layer height (meters) sampled by the caller.
float terrainDetailHeight(vec2 pos_xz_ws, float base_h, float fade) {
    if (fade <= 0.0f) return base_h;
    vec2 rel = (pos_xz_ws + vec2(kTerrainMapMeters * 0.5f)) / kDetailTileMeters;
    ivec2 t = ivec2(floor(rel));
    if (any(lessThan(t, ivec2(0))) ||
        any(greaterThanEqual(t, ivec2(kDetailTilesPerSide))))
        return base_h;
    int slot = detail_slot_map[t.y * kDetailTilesPerSide + t.x];
    if (slot < 0) return base_h;
    // During streaming a neighbouring tile may not be resident yet.
    // Fade to the same base height at that boundary instead of creating
    // a vertical step. Include diagonal neighbours for corner continuity.
    vec2 local_m = (rel - vec2(t)) * kDetailTileMeters;
    const float border_m = 32.0f;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            ivec2 neighbour = t + ivec2(dx, dy);
            bool resident = false;
            if (all(greaterThanEqual(neighbour, ivec2(0))) &&
                all(lessThan(neighbour, ivec2(kDetailTilesPerSide))))
                resident = detail_slot_map[neighbour.y * kDetailTilesPerSide + neighbour.x] >= 0;
            if (!resident) {
                vec2 lo = vec2(dx, dy) * kDetailTileMeters;
                vec2 hi = lo + vec2(kDetailTileMeters);
                float d = length(local_m - clamp(local_m, lo, hi));
                fade = min(fade, smoothstep(0.0f, border_m, d));
            }
        }
    }
    // Texel k center at tile-origin + k * cell (res 1025 texels span the
    // tile; cell = kDetailTileMeters / (kDetailTileRes - 1)).
    vec2 texels = (rel - vec2(t)) * float(kDetailTileRes - 1);
    vec2 uv = (texels + 0.5f) / float(kDetailTileRes);
    float hd = texture(detail_height_tiles, vec3(uv, float(slot))).x * kTerrainHeightAmpMeters;
    return mix(base_h, hd, fade);
}
