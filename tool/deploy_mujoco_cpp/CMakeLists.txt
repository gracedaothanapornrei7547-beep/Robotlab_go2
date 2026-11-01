cmake_minimum_required(VERSION 3.12)
project(mujoco_deploy_go2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    /home/rum/build/mujoco/include
    /home/rum/build/mujoco/build/_deps/glfw3-src/include
    /home/rum/env_isaaclab/lib/python3.11/site-packages/torch/include
    /home/rum/env_isaaclab/lib/python3.11/site-packages/torch/include/torch/csrc/api/include
)

link_directories(
    /home/rum/build/mujoco/build/lib
    /home/rum/env_isaaclab/lib/python3.11/site-packages/torch/lib
)

add_executable(mujoco_deploy_go2 main.cpp)

target_link_libraries(mujoco_deploy_go2
    mujoco
    /home/rum/build/mujoco/build/lib/libglfw3.a
    torch torch_cpu c10
    GL X11 pthread dl m
)

set_target_properties(mujoco_deploy_go2 PROPERTIES
    BUILD_RPATH "/home/rum/build/mujoco/build/lib;/home/rum/env_isaaclab/lib/python3.11/site-packages/torch/lib"
    INSTALL_RPATH "/home/rum/build/mujoco/build/lib;/home/rum/env_isaaclab/lib/python3.11/site-packages/torch/lib"
)

