/**
 * @file state_machine.hpp
 * @brief for robot to switch control state by user command input
 * @author mazunwang
 * @version 1.0
 * @date 2024-05-29
 * 
 * @copyright Copyright (c) 2024  DeepRobotics
 * 
 */
#pragma once

#include "state_base.h"
#include "idle_state.hpp"
#include "standup_state.hpp"
#include "joint_damping_state.hpp"

// #ifdef USE_ONNX
//     #include "rl_control_state_onnx.hpp"
// #else   
//     #include "rl_control_state.hpp"
// #endif

#include "rl_control_state_onnx.hpp"

#include "skydroid_gamepad_interface.hpp"
#include "retroid_gamepad_interface.hpp"
#include "keyboard_interface.hpp"
#ifdef USE_RAISIM
    #include "simulation/jueying_raisim_simulation.hpp"
#endif
#ifdef USE_PYBULLET
    #include "simulation/simulation_interface.hpp"
#endif

#ifdef USE_MJCPP
    #include "simulation/mujoco_interface.hpp"
#endif

#include "hardware/hardware_interface.hpp"
#include "data_streaming.hpp"

class StateMachine{
private:
    std::shared_ptr<StateBase> current_controller_;
    std::shared_ptr<StateBase> idle_controller_;
    std::shared_ptr<StateBase> standup_controller_;
    std::shared_ptr<StateBase> rl_controller_;
    std::shared_ptr<StateBase> joint_damping_controller_;

    StateName current_state_name_, next_state_name_;

    std::shared_ptr<UserCommandInterface> user_command_ptr_;
    std::shared_ptr<RobotInterface> robot_interface_ptr_;
    std::shared_ptr<ControlParameters> control_parameter_ptr_;

    std::shared_ptr<DataStreaming> data_stream_ptr_;

    void GetDataStreaming(){
        if(!robot_interface_ptr_) return;
        VecXf pos = robot_interface_ptr_->GetJointPosition();
        VecXf vel = robot_interface_ptr_->GetJointVelocity();
        VecXf tau = robot_interface_ptr_->GetJointTorque();
        Vec3f rpy = robot_interface_ptr_->GetImuRpy();
        Vec3f acc = robot_interface_ptr_->GetImuAcc();
        Vec3f omg = robot_interface_ptr_->GetImuOmega();
        MatXf jc = robot_interface_ptr_->GetJointCommand();

        data_stream_ptr_->InsertInterfaceTime(robot_interface_ptr_->GetInterfaceTimeStamp());
        data_stream_ptr_->InsertJointData("q", pos);
        data_stream_ptr_->InsertJointData("dq", vel);
        data_stream_ptr_->InsertJointData("tau", tau);
        data_stream_ptr_->InsertJointData("q_cmd", jc.col(1));
        data_stream_ptr_->InsertJointData("tau_ff", jc.col(4));

        data_stream_ptr_->InsertImuData("rpy", rpy);
        data_stream_ptr_->InsertImuData("acc", acc);
        data_stream_ptr_->InsertImuData("omg", omg);

        if(!user_command_ptr_) return;
        auto cmd = user_command_ptr_->GetUserCommand();
        data_stream_ptr_->InsertCommandData("target_mode", float(cmd.target_mode));

        data_stream_ptr_->InsertStateData("current_state", StateBase::msfb_.current_state);
       
        data_stream_ptr_->SendData();
    }

    std::shared_ptr<StateBase> GetNextStatePtr(StateName state_name){
        switch(state_name){
            case StateName::kInvalid:{
                return nullptr;
            }
            case StateName::kIdle:{
                return idle_controller_;
            }
            case StateName::kStandUp:{
                return standup_controller_;
            }
            case StateName::kRLControl:{
                return rl_controller_;
            }
            case StateName::kJointDamping:{
                return joint_damping_controller_;
            }
            default:{
                std::cerr << "error state name" << std::endl;
            }
        }
        return nullptr;
    }
public:
    StateMachine(RobotType robot_type){
        const std::string activation_key = "~/raisim/activation.raisim";
        std::string urdf_path = "";
        std::string mjcf_path = "";
        #ifdef BUILD_SIMULATION
            user_command_ptr_ = std::make_shared<KeyboardInterface>();
        #else
            user_command_ptr_ = std::make_shared<RetroidGamepadInterface>(12121);
        #endif
        // user_command_ptr_ = std::make_shared<KeyboardInterface>();
        // user_command_ptr_ = std::make_shared<RetroidGamepadInterface>(12121);
        if(robot_type == RobotType::Lite3){
            urdf_path = GetAbsPath()+"/../third_party/URDF_model/lite3_urdf/Lite3/urdf/Lite3.urdf";
            mjcf_path = GetAbsPath()+"third_party/URDF_model/Lite3/Lite3_mjcf/mjcf/Lite3.xml";
            #ifdef USE_RAISIM
                robot_interface_ptr_ = std::make_shared<JueyingRaisimSimulation>(activation_key, urdf_path, "Lite3_sim");

            #elif defined(USE_MJCPP)
                robot_interface_ptr_ = std::make_shared<MujocoInterface>("Lite3", mjcf_path);
                std::cout << "Using MujocoInterface CPP " << std::endl;
                std::cout << "mjcf_path: " << mjcf_path << std::endl;
            #elif defined(USE_PYBULLET)
                robot_interface_ptr_ = std::make_shared<SimulationInterface>("Lite3");
            #else
                robot_interface_ptr_ = std::make_shared<HardwareInterface>("Lite3");
            #endif
            control_parameter_ptr_ = std::make_shared<ControlParameters>(robot_type);
        }else{
            std::cerr << "error" << std::endl;
        }

        std::shared_ptr<ControllerData> data_ptr = std::make_shared<ControllerData>();
        data_ptr->robot_interface_ptr = robot_interface_ptr_;
        data_ptr->user_command_ptr = user_command_ptr_;
        data_ptr->control_parameter_ptr = control_parameter_ptr_;
        data_stream_ptr_ = std::make_shared<DataStreaming>(false, false);
        data_ptr->data_stream_ptr = data_stream_ptr_;

        idle_controller_ = std::make_shared<IdleState>(robot_type, "idle_state", data_ptr);
        standup_controller_ = std::make_shared<StandUpState>(robot_type, "standup_state", data_ptr);

        // 测试ONNX，后续需要改成参数控制
        // rl_controller_ = std::make_shared<RLControlState>(robot_type, "rl_control", data_ptr);
        // #ifdef USE_ONNX
        //     rl_controller_ = std::make_shared<RLControlStateONNX>(robot_type, "rl_control", data_ptr);
        // #else
        //     rl_controller_ = std::make_shared<RLControlState>(robot_type, "rl_control", data_ptr);
        // #endif
        rl_controller_ = std::make_shared<RLControlStateONNX>(robot_type, "rl_control", data_ptr);
        


        joint_damping_controller_ = std::make_shared<JointDampingState>(robot_type, "joint_damping", data_ptr);

        current_controller_ = idle_controller_;
        current_state_name_ = kIdle;
        next_state_name_ = kIdle;
   
        // std::cout << "Controller will be enabled in 3 seconds!!!" << std::endl;
        // std::this_thread::sleep_for(std::chrono::seconds(3)); //for safety 

        robot_interface_ptr_->Start();
        std::cout << "Robot interface started" << std::endl;
        user_command_ptr_->Start();
        
        current_controller_->OnEnter();  
    }
    ~StateMachine(){}

    void Run(){
        int cnt = 0;
        static double time_record = 0;
        int state_machine_run_period =2500; // 5us for 200hz
        auto state_machine_start_time = std::chrono::steady_clock::now();
        auto state_machine_end_time = state_machine_start_time;
		bool enter_condition = false;
        while(true){
            if(robot_interface_ptr_->GetInterfaceTimeStamp()!= time_record){
				enter_condition = true;
                state_machine_start_time = std::chrono::steady_clock::now();
                time_record = robot_interface_ptr_->GetInterfaceTimeStamp(); // get current robot data time stamp
                current_controller_ -> Run(); // calling each StateMachine Run() method

                // lose control ownership, set to Damping mode
                if(current_controller_->LoseControlJudge()) next_state_name_ = StateName::kJointDamping; 
                else next_state_name_ = current_controller_ -> GetNextStateName(); // check StateMachine change
                
                if(next_state_name_ != current_state_name_){
                    current_controller_ -> OnExit();
                    std::cout << current_controller_ -> state_name_ << " ------------> ";
                    current_controller_ = GetNextStatePtr(next_state_name_);
                    std::cout << current_controller_ -> state_name_ << std::endl;
                    current_controller_ ->OnEnter();
                    current_state_name_ = next_state_name_; 
                }
                ++cnt;
                this->GetDataStreaming(); // get obs
                state_machine_end_time = std::chrono::steady_clock::now();
            }
			if(enter_condition)
			{
				enter_condition = false; 
				auto time_elapse=std::chrono::duration_cast<std::chrono::microseconds>(state_machine_end_time-state_machine_start_time).count();
            	std::this_thread::sleep_for(std::chrono::microseconds(state_machine_run_period) - std::chrono::microseconds(time_elapse)); // 5ms -> 200hz
			}
			else
			{
				std::this_thread::sleep_for(std::chrono::microseconds(state_machine_run_period));
			}
//			std::cout<<"State machine run time: "<< std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-state_machine_start_time).count() << " us" << std::endl;
        }

        user_command_ptr_->Stop();
        robot_interface_ptr_->Stop();
    }

};
