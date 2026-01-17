/**
 * @file rl_control_state_onnx.hpp
 * @brief rl policy runnning state using onnx
 * @author Bo (Percy) Peng
 * @version 1.0
 * @date 2025-08-10
 * 
 * @copyright Copyright (c) 2025  DeepRobotics * 
 */




#pragma once

#include "state_base.h"
#include "policy_runner_base.hpp"
#include "lite3_test_policy_runner_onnx.hpp"



class RLControlStateONNX : public StateBase
{
private:
    RobotBasicState rbs_;
    int state_run_cnt_;

    std::shared_ptr<PolicyRunnerBase> policy_ptr_;
    std::shared_ptr<Lite3TestPolicyRunnerONNX> test_policy_;


    
    std::thread run_policy_thread_;
    bool start_flag_ = true;

    float policy_cost_time_ = 1;

    void UpdateRobotObservation(){
        rbs_.base_rpy     = robot_interface_ptr_->GetImuRpy();
        rbs_.base_rot_mat = RpyToRm(rbs_.base_rpy);
        rbs_.projected_gravity = RmToProjectedGravity(rbs_.base_rot_mat);
        rbs_.base_omega   = robot_interface_ptr_->GetImuOmega();
        rbs_.base_acc     = robot_interface_ptr_->GetImuAcc();
        rbs_.joint_pos    = robot_interface_ptr_->GetJointPosition();
        rbs_.joint_vel    = robot_interface_ptr_->GetJointVelocity();
        rbs_.joint_tau    = robot_interface_ptr_->GetJointTorque();
        rbs_.cmd_vel_normlized = Vec3f(user_command_ptr_->GetUserCommand().forward_vel_scale, // X
                                    user_command_ptr_->GetUserCommand().side_vel_scale,       // Y
                                    user_command_ptr_->GetUserCommand().turnning_vel_scale);  // turn left or right
        
    }

    void PolicyRunner(){
        int run_cnt_record = -1;
        int inference_period=20000; // 20000us for himloco
		auto inference_start_time = std::chrono::steady_clock::now();
		auto inference_end_time = inference_start_time;
		bool enter_condition = false;
        while (start_flag_){
            if(state_run_cnt_ != run_cnt_record){
				enter_condition = true;
                inference_start_time = std::chrono::steady_clock::now();
                timespec start_timestamp, end_timestamp;
                clock_gettime(CLOCK_MONOTONIC,&start_timestamp);
                auto ra = policy_ptr_->GetRobotAction(rbs_);
                MatXf res = ra.ConvertToMat();
                robot_interface_ptr_->SetJointCommand(res); // (current torque, not last torque, video content slip of the tongue)
                run_cnt_record = state_run_cnt_;
                clock_gettime(CLOCK_MONOTONIC,&end_timestamp);
                policy_cost_time_ = (end_timestamp.tv_sec-start_timestamp.tv_sec)*1e3 
                                    +(end_timestamp.tv_nsec-start_timestamp.tv_nsec)/1e6;
                std::cout << "cost_time:  " << policy_cost_time_ << " ms\n";
				inference_end_time = std::chrono::steady_clock::now();
            }
			 if(enter_condition)
			{
				enter_condition = false;
				auto time_elapse = std::chrono::duration_cast<std::chrono::microseconds>(inference_end_time - inference_start_time).count();
				std::this_thread::sleep_for(std::chrono::microseconds(inference_period) - std::chrono::microseconds(time_elapse)); // 20ms -> 50hz
			}
			else{
            std::this_thread::sleep_for(std::chrono::microseconds(inference_period));
			}
//			std::cout<<"Inference time: "<< std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-inference_start_time).count() << " us" << std::endl;
        }
			
    }

public:
    RLControlStateONNX(const RobotType& robot_type, const std::string& state_name, 
        std::shared_ptr<ControllerData> data_ptr):StateBase(robot_type, state_name, data_ptr){
        std::memset(&rbs_, 0, sizeof(rbs_));
        test_policy_ = std::make_shared<Lite3TestPolicyRunnerONNX>("test_onnx");
        policy_ptr_ = test_policy_;
        if(!policy_ptr_){
            std::cerr << "[ERROR] Failed to initialize ONNX policy runner." << std::endl;
            exit(0);
        }  
        policy_ptr_->DisplayPolicyInfo();
        }
    ~RLControlStateONNX(){}

    virtual void OnEnter() {
        state_run_cnt_ = -1;
        start_flag_ = true;
        run_policy_thread_ = std::thread(std::bind(&RLControlStateONNX::PolicyRunner, this));
        policy_ptr_->OnEnter();
        StateBase::msfb_.UpdateCurrentState(RobotMotionState::RLControlMode);
        user_command_ptr_->SetMotionStateFeedback(StateBase::msfb_);
    };

    virtual void OnExit() { 
        start_flag_ = false;
        run_policy_thread_.join();
        state_run_cnt_ = -1;
    }

    virtual void Run() {
        UpdateRobotObservation();
        data_stream_ptr_->InsertScopeData(0, policy_cost_time_);
        state_run_cnt_++;
    }

    virtual bool LoseControlJudge() {
        if(user_command_ptr_->GetUserCommand().target_mode == int(RobotMotionState::JointDamping)) return true;
        return PostureUnsafeCheck();
    }

    bool PostureUnsafeCheck(){
        Vec3f rpy = robot_interface_ptr_->GetImuRpy();
        if(fabs(rpy(0)) > 30./180*M_PI || fabs(rpy(1)) > 45./180*M_PI){
            std::cout << "posture value: " << 180./M_PI*rpy.transpose() << std::endl;
            return true;
        }
        return false;
    }

    virtual StateName GetNextStateName() {
        return StateName::kRLControl;
    }
};


