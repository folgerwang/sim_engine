# EnTT (vendored via CMake download)

The single-header `entt/entt.hpp` is fetched automatically at CMake configure
time (pinned to v3.13.2), mirroring the project's miniaudio convention. It is
intentionally NOT committed; `.gitignore` excludes it. To refresh or pin a new
version, edit the `entt` download block in the root `CMakeLists.txt`.

If your build machine has no network, drop a v3.13.x `entt.hpp` into
`third_parties/entt/entt/entt.hpp` manually and re-run CMake.
