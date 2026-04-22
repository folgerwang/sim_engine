#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"

// --- cluster_debug.vert ------------------------------------------------
// "Nanite-lite" cluster visualisation -- paints every triangle with a
// flat color derived from the cluster it was assigned to by
// helper::buildClusterMesh().
//
// Vertex layout fed by the C++ ClusterDebugDraw helper:
//   binding 0, location VINPUT_POSITION = 0 : vec3  position  (reused from the mesh's LOD-0 position buffer)
//   binding 1, location 15                 : uint  cluster_id (one per vertex -- every vertex in a
//                                                              triangle shares the cluster id the
//                                                              triangle was packed into; the buffer
//                                                              is built by ClusterDebugDraw from
//                                                              the helper::ClusterMesh sidecar)
// The camera view-proj comes from the usual VIEW_PARAMS_SET binding, and the
// model transform rides on a push-constant struct shared with the frag stage.
// ----------------------------------------------------------------------

layout(push_constant) uniform ClusterDebugUniformBufferObject {
    ClusterDebugParams params;
};

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX) readonly buffer CameraInfoBuffer {
    ViewCameraInfo camera_info;
};

layout(location = VINPUT_POSITION) in vec3 in_position;
layout(location = 15)              in uint in_cluster_id;

layout(location = 0) flat out uint v_cluster_id;

void main() {
    vec3 position_ws = (params.transform * vec4(in_position, 1.0f)).xyz;
    gl_Position = camera_info.view_proj * vec4(position_ws, 1.0);
    v_cluster_id = in_cluster_id;
}
