/**
 * @file lite3_test_policy_runner_onnx.hpp
 * @brief 
 * @author Bo (Percy) Peng
 * @version 1.0
 * @date 2025-08-10
 * 
 * @copyright Copyright (c) 2025  DeepRobotics
 * 
 */



#pragma once


#include "policy_runner_base.hpp"
#include <onnxruntime_cxx_api.h>

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <iostream>

using namespace types;

class Lite3TestPolicyRunnerONNX : public PolicyRunnerBase {
private:
	std::string model_path_;

	Ort::Env env_;
	Ort::SessionOptions session_options_;
	Ort::Session session_;
	Ort::MemoryInfo memory_info_;

	
	std::vector<const char*> input_names_;
	std::vector<const char*> output_names_;

	const int obs_dim_ = 45;
	const int obs_history_dim_ = 270; // 45 * 6
	const int obs_history_length_ = 6;
	const int act_dim_ = 12;
	const int motor_num = 12;

	VecXf current_obs_;
	VecXf obs_history_model_; // ! bad variable name
	std::vector<VecXf> obs_history_;
	VecXf joint_pos_rl = VecXf(act_dim_);// in rl squenece
	VecXf joint_vel_rl = VecXf(act_dim_);
	
	VecXf last_action, tmp_action, action, tmp_obs_history_;
	VecXf projected_gravity;

	VecXf dof_pos_default_robot, dof_pos_default_policy;
	VecXf kp_, kd_;
	Vec3f max_cmd_vel_, gravity_direction = Vec3f(0., 0., -1.);

	

	// joint mapping
	std::vector<int> robot2policy_idx, policy2robot_idx;

	// ? should be base angular velocity scale
	float omega_scale_ = 0.25;
	float dof_vel_scale_ = 0.05;
	Vec3f user_cmd_vel_scale_ =  Vec3f(2,2,0.25) ;
	// Z style
	std::vector<std::string> robot_order = {
		"FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
		"FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
		"HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
		"HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint"};
	// Z style
	std::vector<std::string> policy_order = {
		"FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
		"FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
		"HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
		"HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint"};

	std::vector<float> action_scale_robot = {0.25, 0.25, 0.25,
											 0.25, 0.25, 0.25,
											 0.25, 0.25, 0.25,
											 0.25, 0.25, 0.25};

	RobotAction ra;
	


public:
	Lite3TestPolicyRunnerONNX(std::string policy_name) : PolicyRunnerBase(policy_name),
		env_(ORT_LOGGING_LEVEL_WARNING, "ONNXPolicy"),
		session_options_(),
		memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
		session_(nullptr)
		{
		
		
		// .onnx model需要单独生成
		model_path_ = GetAbsPath() + "/../policy/ppo/himloco.onnx";        
		
		// 调试信息
		std::cout << "[ONNX INIT] Loading model: " << model_path_ << std::endl;

		session_options_.SetIntraOpNumThreads(1);
		session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

		session_ = Ort::Session(env_, model_path_.c_str(), session_options_);
		std::cout << "[ONNX INIT] Model loaded successfully.\n"; 
		

		input_names_ = {"obs"};
		output_names_ = {"action"};


		dof_pos_default_policy.setZero(12);
		dof_pos_default_policy <<  0.0000, -0.8000, 1.6000,
								   0.0000, -0.8000, 1.6000,
								   0.0000, -0.8000, 1.6000,
								   0.0000, -0.8000, 1.6000;

		dof_pos_default_robot.setZero(12);
		dof_pos_default_robot <<  0.0000, -0.8000, 1.6000,
								  0.0000, -0.8000, 1.6000,
								  0.0000, -0.8000, 1.6000,
								  0.0000, -0.8000, 1.6000;

		kp_ = 30.*VecXf::Ones(12);
		kd_ = 1. *VecXf::Ones(12);
		obs_history_.resize(obs_history_length_);
		for(int i =0;i<obs_history_.size();i++)
		{
			obs_history_[i].setZero(obs_dim_);
		}
		// velocity scale
		// HIMLoco: 2, 2, 0.25
		max_cmd_vel_ << 0.8, 0.8, 0.8;

		tmp_action = VecXf(act_dim_);
		last_action = VecXf(act_dim_);
		tmp_obs_history_ = VecXf(obs_history_dim_);
		tmp_obs_history_.setZero(obs_history_dim_);
		for(int i = 0;i<obs_history_length_;i++)
		{
			tmp_obs_history_[i*obs_history_length_+8] = -1.0; // process projected gravity z
		}
		
		ra.goal_joint_pos = VecXf::Zero(act_dim_);
		ra.goal_joint_vel = VecXf::Zero(act_dim_);
		ra.tau_ff = VecXf::Zero(act_dim_);
		ra.kp = kp_;
		ra.kd = kd_;
		
		robot2policy_idx = generate_permutation(robot_order, policy_order);
		policy2robot_idx = generate_permutation(policy_order, robot_order);
		for (int i = 0; i < act_dim_; ++i){
			std::cout << "robot2policy_idx[" << i << "]: " << robot2policy_idx[i] << std::endl;
			std::cout << "policy2robot_idx[" << i << "]: " << policy2robot_idx[i] << std::endl;
		}

		// Test the model
		for (int i = 0; i < 2; ++i) {
			VecXf dummy_input = VecXf::Ones(obs_history_dim_);
			std::array<int64_t, 2> input_shape{1, obs_history_dim_};

			Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
				memory_info_, dummy_input.data(), obs_history_dim_, input_shape.data(), input_shape.size());

			auto reaction = session_.Run(Ort::RunOptions{nullptr},
											   input_names_.data(), &input_tensor, 1,
											   output_names_.data(), 1);
			std::cout << policy_name_ << " ONNX policy network test success" << std::endl;
		}

		decimation_ = 4; // 4 for HIMLoco
	}

	~Lite3TestPolicyRunnerONNX() {}

	std::vector<int> generate_permutation(
		const std::vector<std::string>& from, 
		const std::vector<std::string>& to, 
		int default_index = 0) 
	{
		std::unordered_map<std::string, int> idx_map;
		for (int i = 0; i < from.size(); ++i) {
			idx_map[from[i]] = i;
		}

		std::vector<int> perm;
		for (const auto& name : to) {
			auto it = idx_map.find(name);
			if (it != idx_map.end()) {
				perm.push_back(it->second);
			} else {
				perm.push_back(default_index);  // 如果找不到，就填默认值
			}
		}

		return perm;
	}

	void DisplayPolicyInfo() override {
		std::cout << "ONNX policy: " << policy_name_ << "\n";
		std::cout << "path: " << model_path_ << "\n";
		std::cout << "obs_dim: " << obs_dim_ << ", action_dim: " << act_dim_ << "\n";
	}

	void OnEnter() override {
		run_cnt_ = 0;
		current_obs_.setZero(obs_dim_);
		for(int i =0; i < obs_history_length_; i++)
		{
			obs_history_.push_back(current_obs_);
			// ! Set projected_gravity z to -1.0
			obs_history_[i][8]= -1.0f; 
		}
		std::cout << "[ONNX ENTER] PolicyRunner entered: " << policy_name_ << std::endl;
	}

	RobotAction GetRobotAction(const RobotBasicState& ro) override {
		Vec3f base_omega = ro.base_omega * omega_scale_;
		Vec3f projected_gravity = ro.base_rot_mat.inverse() * gravity_direction;
		// ? Get 0.8 * vel_percentage, 0.8m/s is speed limit
		Vec3f cmd_vel = ro.cmd_vel_normlized.cwiseProduct(max_cmd_vel_);
		Vec3f cmd_vel_scaled = cmd_vel.cwiseProduct(user_cmd_vel_scale_);

		// TODO: check the joint mapping in IsaacLab
		for (int i = 0; i < act_dim_; ++i){
			joint_pos_rl(i) = ro.joint_pos(robot2policy_idx[i]); // scale equals to 1.0
			joint_vel_rl(i) = ro.joint_vel(robot2policy_idx[i]) * dof_vel_scale_;
		}

		current_obs_.setZero(obs_dim_);
		current_obs_ << cmd_vel_scaled,
						base_omega,
						projected_gravity,
						joint_pos_rl,
						joint_vel_rl,
						last_action;
		std::cout<<"###### current obs #######" <<std::endl;
		std::cout<< current_obs_<<std::endl;
		std::cout<<"##########################" <<std::endl;

		// TODO: obs stack new -> old 
		obs_history_.erase(obs_history_.begin()); // erase the old
		obs_history_.push_back(current_obs_);

		std::vector<VecXf> temp_obs_history;
		for(auto it = obs_history_.rbegin();it != obs_history_.rend();it++)
		{
			// new -> old
			temp_obs_history.push_back(*it);
		}

		// ! bad style
		for (int i =0;i < obs_history_length_;i++)
		{
			for(int j = 0 ; j < obs_dim_; j++)
			{
				tmp_obs_history_( i * obs_dim_ + j) =temp_obs_history[i](j);
			}
		}
		std::cout<<"******* ewest obs in stack *******" <<std::endl;
		for(int i =0 ;i< obs_dim_;i++)
		{
			std::cout<<tmp_obs_history_[i]<<std::endl;
		}
		std::cout<<"***********************************" <<std::endl;
	   
		std::array<int64_t, 2> input_shape{1, obs_history_dim_}; // change to 270 for himloco

		Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
			memory_info_, 
			tmp_obs_history_.data(), tmp_obs_history_.size(), 
			input_shape.data(), input_shape.size());
		
		std::vector<Ort::Value> inputs;
		inputs.emplace_back(std::move(input_tensor));  // forbidden copy construct

		auto output_tensors = session_.Run(Ort::RunOptions{nullptr},
										   input_names_.data(), 
										   inputs.data(), 1,
										   output_names_.data(), 1);

		float* action_data = output_tensors[0].GetTensorMutableData<float>();
		Eigen::Map<Eigen::MatrixXf> act(action_data, act_dim_, 1);
		action = VecXf(act);
		last_action = action;
		
		for (int i = 0; i < act_dim_; ++i){
			tmp_action(i) = action(policy2robot_idx[i]);
			tmp_action(i) *= action_scale_robot[i];
		}
		
		//tmp_action(0) = -tmp_action(1);
		//tmp_action(1) = -tmp_action(1);
		//tmp_action(2) = -tmp_action(2);
		//tmp_action(3) = -tmp_action(3);
		//tmp_action(4) = -tmp_action(4);
		//tmp_action(5) = -tmp_action(5);
		//tmp_action(6) = -tmp_action(6);
		//tmp_action(7) = -tmp_action(7);
		//tmp_action(8) = -tmp_action(8);
		//tmp_action(9) = -tmp_action(9);
		//tmp_action(10) = -tmp_action(10);
		//tmp_action(11) = -tmp_action(11);
		tmp_action += dof_pos_default_robot;

		std::cout<< "-------action---------"<<std::endl;
		std::cout<<tmp_action<<std::endl;
		std::cout<< "----------------------"<<std::endl;
		// Output desired joint pos
		ra.goal_joint_pos = tmp_action;
		ra.goal_joint_vel = VecXf::Zero(act_dim_);
		ra.tau_ff = VecXf::Zero(act_dim_);
		ra.kp = kp_;
		ra.kd = kd_;
		++run_cnt_;

		return ra;
	}
};

