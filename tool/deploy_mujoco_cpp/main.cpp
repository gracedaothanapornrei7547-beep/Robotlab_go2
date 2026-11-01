#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <cmath>
#include <algorithm>
#include <iomanip>

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <torch/script.h>

// MuJoCo visualization globals
mjModel* m = nullptr;
mjData* d = nullptr;
mjvCamera cam;
mjvOption opt;
mjvScene scn;
mjrContext con;

bool button_left = false;
bool button_middle = false;
bool button_right = false;
double lastx = 0;
double lasty = 0;

// Command and control variables
float cmd_vx = 0.0f;
float cmd_vy = 0.0f;
float cmd_wz = 0.0f;
float current_cmd_vx = 0.0f;
float current_cmd_vy = 0.0f;
float current_cmd_wz = 0.0f;
float target_heading = 0.0f;
bool enable_heading_lock = true;
const float heading_control_stiffness = 1.0f;
bool reset_requested = false;

// 慢动作回放与暂停变量
float time_scale = 1.0f; // 1.0 = 原速, 0.5 = 2倍慢速, 0.25 = 4倍慢速, 0.1 = 10倍超慢动作
bool is_paused = false;
bool step_single_frame = false;

// Global policy pointer for reset
torch::jit::script::Module* p_policy = nullptr;

inline float quat_to_yaw(const double q[4]) {
    double w = q[0], x = q[1], y = q[2], z = q[3];
    double siny_cosp = 2.0 * (w * z + x * y);
    double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
    return static_cast<float>(std::atan2(siny_cosp, cosy_cosp));
}

inline float wrap_to_pi(float angle) {
    while (angle > static_cast<float>(M_PI)) angle -= static_cast<float>(2.0 * M_PI);
    while (angle < static_cast<float>(-M_PI)) angle += static_cast<float>(2.0 * M_PI);
    return angle;
}

// Robot Lab mapping (Strictly matching IsaacLab JOINT_NAMES: FL, FR, RL, RR)
const std::vector<std::string> robotlab_joint_names = {
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"
};

// Nominal default joint angles:
// FL: [0.1, 0.8, -1.5], FR: [-0.1, 0.8, -1.5], RL: [0.1, 1.0, -1.5], RR: [-0.1, 1.0, -1.5]
const float default_joint_pos_robotlab[12] = {
    0.1f,  0.8f, -1.5f,  // FL
   -0.1f,  0.8f, -1.5f,  // FR
    0.1f,  1.0f, -1.5f,  // RL (官方原版标准名义角)
   -0.1f,  1.0f, -1.5f   // RR (官方原版标准名义角)
};

// Action scale: 0.25 for all 12 joints (matching trained MoE-CTS MDP config)
const float action_scale[12] = {
    0.25f, 0.25f, 0.25f, // FL
    0.25f, 0.25f, 0.25f, // FR
    0.25f, 0.25f, 0.25f, // RL
    0.25f, 0.25f, 0.25f  // RR
};

int mujoco_to_robotlab[12];
int robotlab_to_mujoco[12];
float default_joint_pos_mujoco[12];

// Mouse button callback
void mouse_button(GLFWwindow* window, int button, int action, int mods) {
    button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);
    glfwGetCursorPos(window, &lastx, &lasty);
}

// Mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos) {
    if (!button_left && !button_middle && !button_right) return;

    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    int width, height;
    glfwGetWindowSize(window, &width, &height);

    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

    mjtMouse action;
    if (button_right) action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if (button_left) action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else action = mjMOUSE_ZOOM;

    mjv_moveCamera(m, action, dx / height, dy / height, &scn, &cam);
}

// Scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset) {
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &scn, &cam);
}

// Keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods) {
    if (act == GLFW_PRESS || act == GLFW_REPEAT) {
        if (key == GLFW_KEY_ESCAPE) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        } else if (key == GLFW_KEY_UP || key == GLFW_KEY_W) {
            cmd_vx = std::min(cmd_vx + 0.2f, 2.0f);
            std::cout << "🎮 [C++ Controller] 前向速度 -> " << cmd_vx << " m/s (vy=" << cmd_vy << ", wz=" << cmd_wz << ")" << std::endl;
        } else if (key == GLFW_KEY_DOWN || key == GLFW_KEY_S) {
            cmd_vx = std::max(cmd_vx - 0.2f, -0.5f); // 训练倒车包线限速至 -0.5 m/s，防止超出分布
            std::cout << "🎮 [C++ Controller] 前向速度 -> " << cmd_vx << " m/s (vy=" << cmd_vy << ", wz=" << cmd_wz << ")" << std::endl;
        } else if (key == GLFW_KEY_A || key == GLFW_KEY_J) {
            cmd_vy = std::min(cmd_vy + 0.1f, 1.0f);
            std::cout << "🎮 [C++ Controller] 横向平移(左移 A) -> " << cmd_vy << " m/s (vx=" << cmd_vx << ", wz=" << cmd_wz << ")" << std::endl;
        } else if (key == GLFW_KEY_D || key == GLFW_KEY_L) {
            cmd_vy = std::max(cmd_vy - 0.1f, -1.0f);
            std::cout << "🎮 [C++ Controller] 横向平移(右移 D) -> " << cmd_vy << " m/s (vx=" << cmd_vx << ", wz=" << cmd_wz << ")" << std::endl;
        } else if (key == GLFW_KEY_Q || key == GLFW_KEY_LEFT) {
            cmd_wz = std::min(cmd_wz + 0.15f, 1.5f);
            std::cout << "🎮 [C++ Controller] 航向旋转(逆时针自转 Q) -> " << cmd_wz << " rad/s (vx=" << cmd_vx << ", vy=" << cmd_vy << ")" << std::endl;
        } else if (key == GLFW_KEY_E || key == GLFW_KEY_RIGHT) {
            cmd_wz = std::max(cmd_wz - 0.15f, -1.5f);
            std::cout << "🎮 [C++ Controller] 航向旋转(顺时针自转 E) -> " << cmd_wz << " rad/s (vx=" << cmd_vx << ", vy=" << cmd_vy << ")" << std::endl;
        } else if (key == GLFW_KEY_H && act == GLFW_PRESS) {
            enable_heading_lock = !enable_heading_lock;
            std::cout << "🧭 [C++ Controller] 航向闭环锁定开关: " << (enable_heading_lock ? "【开启】" : "【关闭】") << std::endl;
        } else if (key == GLFW_KEY_SPACE) {
            cmd_vx = 0.0f;
            cmd_vy = 0.0f;
            cmd_wz = 0.0f;
            if (d != nullptr) {
                target_heading = quat_to_yaw(d->qpos + 3);
            }
            std::cout << "⏸️ [C++ Controller] 急停立正指令 (vx=0, vy=0, wz=0, 锁定当前航向)" << std::endl;
        } else if (key == GLFW_KEY_R) {
            reset_requested = true;
            std::cout << "🔄 [C++ Controller] 请求重置复位" << std::endl;
        } else if (key == GLFW_KEY_LEFT_BRACKET || key == GLFW_KEY_MINUS) {
            time_scale = std::max(0.05f, time_scale * 0.5f);
            std::cout << "⏱️ [C++ Controller] 仿真放慢 -> " << time_scale << "x (" << (1.0f / time_scale) << "倍慢动作)" << std::endl;
        } else if (key == GLFW_KEY_RIGHT_BRACKET || key == GLFW_KEY_EQUAL) {
            time_scale = std::min(2.0f, time_scale * 2.0f);
            std::cout << "⏱️ [C++ Controller] 仿真加速 -> " << time_scale << "x (" << (1.0f / time_scale) << "倍慢动作)" << std::endl;
        } else if (key == GLFW_KEY_P && act == GLFW_PRESS) {
            is_paused = !is_paused;
            std::cout << (is_paused ? "⏸️ [C++ Controller] 画面已暂停 (按 P 继续，按 F 或 . 键单帧微调步进)" : "▶️ [C++ Controller] 恢复正常运行") << std::endl;
        } else if ((key == GLFW_KEY_F || key == GLFW_KEY_PERIOD) && act == GLFW_PRESS) {
            step_single_frame = true;
            std::cout << "⏯️ [C++ Controller] 单步逐帧微调 20ms" << std::endl;
        }
    }
}

void reset_robot() {
    mj_resetData(m, d);
    d->qpos[0] = 0.0;
    d->qpos[1] = 0.0;
    d->qpos[2] = 0.34; // 优雅标准站立高度，脚底平贴地面无穿模碰撞
    d->qpos[3] = 1.0;  // quat [w, x, y, z]
    d->qpos[4] = 0.0;
    d->qpos[5] = 0.0;
    d->qpos[6] = 0.0;

    for (int i = 0; i < 12; ++i) {
        d->qpos[7 + i] = default_joint_pos_mujoco[i];
    }
    for (int i = 0; i < m->nv; ++i) {
        d->qvel[i] = 0.0;
    }
    mj_forward(m, d);
    target_heading = quat_to_yaw(d->qpos + 3);
    current_cmd_vx = 0.0f;
    current_cmd_vy = 0.0f;
    current_cmd_wz = 0.0f;

    if (p_policy != nullptr) {
        try {
            p_policy->get_method("reset")({});
        } catch (...) {
        }
    }
}

int main(int argc, char** argv) {
    std::string scene_path = "/home/rum/himloco_assets/mujoco_menagerie/unitree_go2/scene.xml";
    std::string model_path = "/home/rum/go2_rl_robotlab/deploy/pre_train/go2/go2_moe_cts_176k_0.6984.pt";

    if (argc > 1) {
        cmd_vx = std::stof(argv[1]);
    }
    if (argc > 2) {
        model_path = argv[2];
    }
    if (argc > 3) {
        cmd_vy = std::stof(argv[3]);
    }
    if (argc > 4) {
        cmd_wz = std::stof(argv[4]);
    }
    if (argc > 5) {
        scene_path = argv[5];
    }
    const char* env_ts = std::getenv("TIME_SCALE");
    if (env_ts) {
        time_scale = std::stof(env_ts);
    }
    if (argc > 6) {
        time_scale = std::stof(argv[6]);
    }

    std::cout << "======================================================================\n";
    std::cout << "⚡ Unitree Go2 全向 MoE-CTS C++ 真实物理 MuJoCo 部署引擎\n";
    std::cout << "  - 场景文件: " << scene_path << "\n";
    std::cout << "  - 模型文件: " << model_path << "\n";
    std::cout << "  - 初始指令: vx=" << cmd_vx << " m/s, vy=" << cmd_vy << " m/s, wz=" << cmd_wz << " rad/s\n";
    std::cout << "  - 仿真速率: " << time_scale << "x (" << (1.0f / time_scale) << "倍慢动作)\n";
    std::cout << "======================================================================\n";

    // 1. 加载 MuJoCo 模型
    char error[1000] = "Could not load XML model";
    m = mj_loadXML(scene_path.c_str(), 0, error, 1000);
    if (!m) {
        std::cerr << "❌ MuJoCo 模型加载失败: " << error << std::endl;
        return 1;
    }
    d = mj_makeData(m);

    // 清除 Menagerie 被动阻尼，严格对齐 IsaacLab PhysX
    for (int i = 6; i < m->nv; ++i) {
        m->dof_damping[i] = 0.0;
    }

    // 2. 构建关节拓扑映射
    for (int i = 0; i < 12; ++i) {
        std::string name = robotlab_joint_names[i];
        int mj_id = mj_name2id(m, mjOBJ_JOINT, name.c_str()) - 1; // 0-based joint index (skip freejoint)
        robotlab_to_mujoco[i] = mj_id;
        mujoco_to_robotlab[mj_id] = i;
    }
    for (int i = 0; i < 12; ++i) {
        default_joint_pos_mujoco[i] = default_joint_pos_robotlab[mujoco_to_robotlab[i]];
    }

    // 3. 加载 LibTorch 策略
    torch::jit::script::Module policy;
    try {
        policy = torch::jit::load(model_path);
        policy.eval();
        p_policy = &policy;
        std::cout << "✅ 成功加载 C++ TorchScript 策略模型!" << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "❌ C++ 加载策略失败: " << e.msg() << std::endl;
        return 1;
    }

    // 4. 初始化 GLFW 视窗
    if (!glfwInit()) {
        std::cerr << "❌ 无法初始化 GLFW" << std::endl;
        return 1;
    }

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Unitree Go2 C++ MuJoCo MoE-CTS Omni Viewer", NULL, NULL);
    if (!window) {
        glfwTerminate();
        std::cerr << "❌ 无法创建 GLFW 窗口 (请检查 DISPLAY 环境变量)" << std::endl;
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // 开启垂直同步 60 FPS

    // 初始化 MuJoCo 渲染上下文
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // 设置事件回调
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    // 初始重置
    reset_robot();

    // 控制环路变量
    std::vector<float> last_action(12, 0.0f);
    std::vector<float> obs(45, 0.0f);
    float desired_pos[12];
    for (int i = 0; i < 12; ++i) desired_pos[i] = default_joint_pos_mujoco[i];

    const float kp = 20.0f;
    const float kd = 0.5f;
    const float torque_limit = 23.5f;

    const int decimation = 10; // 50Hz 控制 (sim_dt=0.002s, 0.002 * 10 = 0.02s)
    int step_counter = 0;

    std::cout << "\n🎮 全向交互控制操作指南 (老大专属标准键位):\n";
    std::cout << "  [W / S 或 ↑ / ↓]: 前进 / 后退 (+0.2 m/s / -0.2 m/s)\n";
    std::cout << "  [A / D]: 横向平移(左移 / 右移) (A: +0.1 m/s, D: -0.1 m/s，最高 ±1.0 m/s)\n";
    std::cout << "  [Q / E 或 ← / →]: 航向自转(逆时针 / 顺时针) (Q: +0.15 rad/s, E: -0.15 rad/s，最高 ±1.5 rad/s)\n";
    std::cout << "  [H 键]: 开关航向闭环锁定 (默认【关闭】，按 H 可按需开启)\n";
    std::cout << "  [空格键]: 急停归零 (原地立正)\n";
    std::cout << "  [R 键]: 初始姿态复位\n";
    std::cout << "  [ [ / - 键]: ⏱️ 放慢运行速度 (0.5x -> 0.25x -> 0.1x 慢动作回放)\n";
    std::cout << "  [ ] / = 键]: ⏱️ 加快运行速度 (恢复原速或倍速)\n";
    std::cout << "  [P 键]: ⏸️ 暂停 / 继续播放\n";
    std::cout << "  [F / . 键]: ⏯️ 暂停时单步逐帧微调步进 (20ms/帧)\n";
    std::cout << "  [鼠标拖拽]: 3D 自由环绕视角与滚轮缩放\n";
    std::cout << "======================================================================\n\n";

    auto last_telemetry_time = std::chrono::steady_clock::now();

    while (!glfwWindowShouldClose(window)) {
        if (is_paused && !step_single_frame) {
            cam.lookat[0] = d->qpos[0];
            cam.lookat[1] = d->qpos[1];
            cam.lookat[2] = d->qpos[2];

            mjrRect viewport = {0, 0, 0, 0};
            glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

            mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
            mjr_render(viewport, &scn, &con);

            glfwSwapBuffers(window);
            glfwPollEvents();
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
            continue;
        }

        if (step_single_frame) {
            step_single_frame = false;
        }

        if (reset_requested) {
            reset_robot();
            std::fill(last_action.begin(), last_action.end(), 0.0f);
            for (int i = 0; i < 12; ++i) desired_pos[i] = default_joint_pos_mujoco[i];
            reset_requested = false;
        }

        // 1. 控制器 50 Hz 触发
        if (step_counter % decimation == 0) {
            auto start_infer = std::chrono::high_resolution_clock::now();

            // 0..2: 机身角速度 (MuJoCo freejoint 的 qvel[3:6] 默认即为机身局部固连坐标系 body-fixed frame，无需也不能二次旋转)
            obs[0] = static_cast<float>(d->qvel[3] * 0.25);
            obs[1] = static_cast<float>(d->qvel[4] * 0.25);
            obs[2] = static_cast<float>(d->qvel[5] * 0.25);

            // 3..5: 投影重力 projected_gravity (世界坐标系 [0, 0, -1] 旋转变换至局部机身坐标系)
            mjtNum q_inv[4];
            mju_negQuat(q_inv, d->qpos + 3);
            mjtNum grav[3] = {0.0, 0.0, -1.0};
            mjtNum proj_grav[3];
            mju_rotVecQuat(proj_grav, grav, q_inv);
            obs[3] = static_cast<float>(proj_grav[0]);
            obs[4] = static_cast<float>(proj_grav[1]);
            obs[5] = static_cast<float>(proj_grav[2]);

            // 6..8: 速度指令平滑斜坡加速 (严格对齐 MDP 训练阶段 max_accel limits)
            const float dt_ctrl = 0.02f;
            const float max_accel_x = 2.0f;
            const float max_accel_y = 1.5f;
            const float max_accel_yaw = 3.0f;

            float diff_vx = cmd_vx - current_cmd_vx;
            current_cmd_vx += std::clamp(diff_vx, -max_accel_x * dt_ctrl, max_accel_x * dt_ctrl);

            float diff_vy = cmd_vy - current_cmd_vy;
            current_cmd_vy += std::clamp(diff_vy, -max_accel_y * dt_ctrl, max_accel_y * dt_ctrl);

            float diff_wz = cmd_wz - current_cmd_wz;
            current_cmd_wz += std::clamp(diff_wz, -max_accel_yaw * dt_ctrl, max_accel_yaw * dt_ctrl);

            // 闭环航向锁定：严格对齐 IsaacLab heading_command 协议，消除走偏
            float current_yaw = quat_to_yaw(d->qpos + 3);
            float active_wz = current_cmd_wz;
            if (enable_heading_lock && std::abs(cmd_wz) < 1e-4f) {
                // 直线/纯横移模式：航向闭环负反馈抗偏漂（纠偏限幅 ±1.0 rad/s，强力抑制起步左旋倾向）
                float heading_error = wrap_to_pi(target_heading - current_yaw);
                active_wz = std::clamp(2.0f * heading_error, -1.0f, 1.0f);
            } else {
                // 手动自转或关闭航向锁：目标航向跟随当前朝向
                target_heading = current_yaw;
            }
            obs[6] = current_cmd_vx;
            obs[7] = current_cmd_vy;
            obs[8] = active_wz;

            // 9..20: 关节相对位置 (robot_lab 顺序: FL, FR, RL, RR)
            for (int i = 0; i < 12; ++i) {
                int mj_idx = robotlab_to_mujoco[i];
                float q = static_cast<float>(d->qpos[7 + mj_idx]);
                obs[9 + i] = q - default_joint_pos_robotlab[i];
            }

            // 21..32: 关节速度 * 0.05 (robot_lab 顺序: FL, FR, RL, RR)
            for (int i = 0; i < 12; ++i) {
                int mj_idx = robotlab_to_mujoco[i];
                obs[21 + i] = static_cast<float>(d->qvel[6 + mj_idx] * 0.05);
            }

            // 33..44: 上一次动作 last_action
            for (int i = 0; i < 12; ++i) {
                obs[33 + i] = last_action[i];
            }

            // C++ LibTorch 极速推理
            torch::NoGradGuard no_grad;
            auto input_tensor = torch::from_blob(obs.data(), {1, 45}, torch::kFloat32);
            at::Tensor output = policy.forward({input_tensor}).toTensor();
            float* action_ptr = output.data_ptr<float>();

            // 更新目标关节角与期望动作
            for (int i = 0; i < 12; ++i) {
                float a = std::clamp(action_ptr[i], -100.0f, 100.0f);
                last_action[i] = a;
                float q_des_rl = default_joint_pos_robotlab[i] + a * action_scale[i];
                int mj_idx = robotlab_to_mujoco[i];
                desired_pos[mj_idx] = q_des_rl;
            }

            auto end_infer = std::chrono::high_resolution_clock::now();
            auto infer_us = std::chrono::duration_cast<std::chrono::microseconds>(end_infer - start_infer).count();

            // 每 0.5 秒输出一次 C++ 实时性能审计
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - last_telemetry_time).count() > 0.5) {
                printf("📊 [C++ Telemetry] 位姿: X=%+.2fm, Y=%+.2fm, Z=%.2fm, Yaw=%+.1f° | 指令: vx=%+.2f, vy=%+.2f, wz=%+.2f | 推理耗时: %ld μs\n",
                       d->qpos[0], d->qpos[1], d->qpos[2], current_yaw * 180.0f / static_cast<float>(M_PI),
                       cmd_vx, cmd_vy, active_wz, infer_us);
                last_telemetry_time = now;
            }
        }

        // 2. 500 Hz 电机 PD 扭矩控制
        for (int i = 0; i < 12; ++i) {
            float q = static_cast<float>(d->qpos[7 + i]);
            float dq = static_cast<float>(d->qvel[6 + i]);
            float tau = kp * (desired_pos[i] - q) - kd * dq;
            tau = std::clamp(tau, -torque_limit, torque_limit);
            d->ctrl[i] = tau;
        }

        // 3. 物理仿真单步步进
        mj_step(m, d);
        step_counter++;

        // 慢动作调速延时 (当 time_scale < 0.99f 时按比例放慢物理演进)
        if (step_counter % decimation == 0 && time_scale < 0.99f) {
            int target_us = static_cast<int>(20000.0f / time_scale);
            int sleep_us = target_us - 20000;
            if (sleep_us > 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
            }
        }

        // 4. 60 FPS 图形渲染刷新
        if (step_counter % 8 == 0) {
            // 相机智能跟踪狗子质心
            cam.lookat[0] = d->qpos[0];
            cam.lookat[1] = d->qpos[1];
            cam.lookat[2] = d->qpos[2];

            mjrRect viewport = {0, 0, 0, 0};
            glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

            mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
            mjr_render(viewport, &scn, &con);

            glfwSwapBuffers(window);
            glfwPollEvents();
        }
    }

    // 资源释放
    mjv_freeScene(&scn);
    mjr_freeContext(&con);
    mj_deleteData(d);
    mj_deleteModel(m);
    glfwDestroyWindow(window);
    glfwTerminate();

    std::cout << "\n✅ C++ MuJoCo 部署视窗安全退出。\n";
    return 0;
}
