# Huawei Tech Arena 2024 Phase 1

<p align="center"> Server fleet management optimisation, Phase 1
    <br> 
</p>

## 🧐 Problem Statement <a name = "problem_statement"></a>

The Tech Arena 2024 challenge involves optimizing server fleet management across four data centers. The goal is to develop a model that recommends actions—buying, moving, holding, or dismissing servers—at each time step to maximize three objectives: server utilization, lifespan, and profit. Each data center has capacity constraints, and the servers come in two types (CPU and GPU) with varying attributes like energy consumption, costs, and lifespans.

Participants must balance these factors over 168 discrete time steps, adhering to constraints while responding to dynamic, stochastic demand. Solutions are evaluated on cumulative scores, and participants submit their models as JSON files, tested against known and secret demand scenarios.

## 💡 Idea / Solution <a name = "idea"></a>

- Designed a custom OpenAI gym environment that keeps track of the current state of the server fleet
- Optimized the environment by fixing values of action properties depending on action type to reduce action space, which
decreased the training time for the model to start producing satisfactory scores
- Train a reinforcement learning model on the custom environment based on the Proximal Policy Optimization (PPO) algorithm to manage the server fleet and maximise the score over 168 timesteps
- Enforces constraints like datacenter capacity by invalidating actions every timestep using action masking, which enables variable action space
- Constructed training and testing pipelines to evaluate performance of the model using the provided utility functions

## 📊 Results
Top 15 out of 100+ participating teams, selected to participate in Phase 2 of the contest in Dublin, Ireland.

## ⛏️ Tech Stack
- Stable Baselines3
- OpenAI Gymnasium

## 🎉 Acknowledgments <a name = "acknowledgments"></a>
Credits to my team members [Tom Smail](https://github.com/tomSmail), [Gracie Zhou](https://github.com/Pidongg) and [Oliver Greenwood](https://github.com/ogreenwood672). 

