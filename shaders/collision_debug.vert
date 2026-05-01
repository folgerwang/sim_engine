#version 450
#extension GL_ARB_separate_shader_objects : enable
#include "global_definition.glsl.h"

// --- collision_debug.vert ---------------------------------------------
// Renders the per-CollisionMesh triangle list as a debug visualisation.
// Each vertex carries the segmentation id of the CollisionMesh it
// belongs to (the C++ helper writes the same `mesh_id` into every
// vertex of the mesh) so the fragment shader can hash that into a flat
// per-mesh colour -- effectively an "instance segmentation" view of
// the physics world. The positions are already in world space (the
// FBX collision builder bakes node transforms into vertex_position_),
// so the push-constant transform is normally identity but kept for
// symmetry with cluster_debug.vert.
//
// Vertex layout fed by the C++ CollisionDebugDraw helper:
//   binding 0, location VINPUT_POSITION = 0 : vec3  position
//   binding 1, location 15                  : uint  segmentation_id
//                                                  (one per vertex;
//                                                  every vertex of the
//                                                  same mesh shares it)
// The camera view-proj comes from the usual VIEW_PARAMS_SET binding.
// ----------------------------------------------------------------------

layout(push_constant) uniform CollisionDebugUniformBufferObject {
    ClusterDebugParams params;   // reused: same { mat4 transform } shape
};

layout(std430, set = VIEW_PARAMS_SET, binding = VIEW_CAMERA_BUFFER_INDEX)
    readonly buffer CameraInfoBuffer {
        ViewCameraInfo camera_info;
};

layout(location = VINPUT_POSITION) in vec3 in_position;
layout(location = 15)              in uint in_triangle_id;

layout(location = 0) flat out uint v_triangle_id;

void main() {
    vec3 position_ws = (params.transform * vec4(in_position, 1.0f)).xyz;
    gl_Position = camera_info.view_proj * vec4(position_ws, 1.0);
    v_triangle_id = in_triangle_id;
}
