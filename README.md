# 🧠 Agentic Health AI System

An autonomous, agent-based health assistant that provides personalized **diet**, **fitness**, and **health insights** using a **Think–Act–Observe (TAO)** reasoning framework.

This system goes beyond static recommendations by using **agentic reasoning loops** that adapt outputs based on user context and historical data.

---

## ✨ Key Features

- **Agentic Reasoning (TAO Loop)**
  - Think → Act → Observe → Refine
  - Controlled iteration with explicit stopping conditions

- **Multiple Specialized Agents**
  - **Diet Agent**: Generates structured 7-day, day-wise meal plans
  - **Fitness Agent**: Creates progressive workout plans
  - **Health Agent**: Analyzes lifestyle risks and habits
  - **Mess Food Optimizer**: Optimizes hostel/mess food for nutrition and budget

- **Explainable Outputs**
  - Each agent exposes reasoning steps and confidence
  - No black-box decisions

- **Streamlit-based UI**
  - Simple, interactive web interface
  - No CLI interaction required

---

## 🧠 Agentic Design Overview

Each agent follows a **TAO (Think–Act–Observe)** loop:

1. **Think**
   - Analyze user profile and recent logs
   - Decide the most valuable goal

2. **Act**
   - Generate a solution based on the goal

3. **Observe**
   - Evaluate output quality and confidence
   - Decide whether another iteration is needed

Task-aware stopping logic ensures:
- No infinite loops
- No unnecessary reasoning
- Predictable output size (web-safe)

---

## 🗂 Project Structure

```text
.
├── app.py                 # Streamlit application (entry point)
├── agent_bridge.py        # Bridge between Streamlit and agent system
├── lifestyle_final.py     # Core agent logic and TAO framework
├── requirements.txt       # Dependencies
├── README.md              # Project overview
├── RUN.md                 # Setup and deployment instructions
