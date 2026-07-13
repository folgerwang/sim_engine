// tile_detail.glsl.h — runtime 1 m terrain detail sampling.
//
// Include AFTER global_definition.glsl.h in a shader that has
// TILE_PARAMS_SET bound with the tile resource descriptor set
// (TerrainDetailStream provides the two bindings below).
//
// The world is split into kDetailTilesPerSide^2 detail tiles of
// kDetailTileMeters; a 3x3 ring around the camera is resident in
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
    // Texel k center at tile-origin + k * cell (res 2049 texels span the
    // tile; cell = kDetailTileMeters / (kDetailTileRes - 1)).
    vec2 texels = (rel - vec2(t)) * float(kDetailTileRes - 1);
    vec2 uv = (texels + 0.5f) / float(kDetailTileRes);
    float hd = texture(detail_height_tiles, vec3(uv, float(slot))).x * kTerrainHeightAmpMeters;
    return mix(base_h, hd, fade);
}
