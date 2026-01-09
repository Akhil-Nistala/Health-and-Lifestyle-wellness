import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, TypedDict
from dataclasses import dataclass, field, asdict
from pathlib import Path
import re
import sys
from enum import Enum
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import uuid
import pickle

# ---------------- CONFIGURATION ------------------
load_dotenv()

class Config:
    """Configuration constants"""
    MODEL = "llama-3.1-8b-instant"
    MAX_TAO_LOOPS = 5
    MAX_TOKENS = 2000
    DATA_DIR = Path("health_ai_data")
    BACKUP_DIR = DATA_DIR / "backups"
    SESSION_ID = str(uuid.uuid4())[:8]

# ---------------- DATA MODELS ------------------
class DietType(Enum):
    VEGETARIAN = "vegetarian"
    NON_VEGETARIAN = "non_vegetarian"
    EGGETARIAN = "eggetarian"
    VEGAN = "vegan"

@dataclass
class UserProfile:
    """Complete user profile"""
    user_id: str
    name: str
    age: int
    weight: float  # kg
    height: float  # cm
    diet_type: DietType
    fitness_goal: str
    activity_level: str  # sedentary, light, moderate, active, very_active
    is_student: bool = False
    has_mess_access: bool = False
    weekly_budget: float = 500.0  # INR
    medical_conditions: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        if isinstance(self.diet_type, str):
            try:
                self.diet_type = DietType(self.diet_type.lower())
            except:
                self.diet_type = DietType.NON_VEGETARIAN
        self.updated_at = datetime.now().isoformat()
    
    @property
    def bmi(self) -> float:
        """Calculate BMI"""
        if self.height == 0:
            return 0.0
        height_m = self.height / 100
        return round(self.weight / (height_m ** 2), 1)
    
    @property
    def bmi_category(self) -> str:
        """Get BMI category"""
        bmi = self.bmi
        if bmi == 0:
            return "Unknown"
        elif bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Healthy"
        elif bmi < 30:
            return "Overweight"
        else:
            return "Obese"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['diet_type'] = self.diet_type.value
        data['bmi'] = self.bmi
        data['bmi_category'] = self.bmi_category
        return data

@dataclass
class DailyLog:
    """Daily health log entry"""
    log_id: str
    user_id: str
    date: str  # YYYY-MM-DD format
    day_of_week: str
    timestamp: str
    
    # Sleep metrics
    sleep_hours: float
    sleep_quality: int  # 1-10
    sleep_notes: str = ""
    
    # Nutrition
    meals: Dict[str, str] = field(default_factory=dict)
    water_intake_liters: float = 2.0
    supplements_taken: List[str] = field(default_factory=list)
    
    # Activity
    steps_count: int = 0
    workout_duration_min: int = 0
    workout_type: str = ""
    
    # Metrics
    energy_level: int = 5  # 1-10
    mood_level: int = 5  # 1-10
    stress_level: int = 5  # 1-10
    focus_level: int = 5  # 1-10
    
    # Symptoms & Notes
    symptoms: List[str] = field(default_factory=list)
    notes: str = ""
    
    @property
    def meals_eaten(self) -> int:
        """Count meals actually eaten (not skipped)"""
        skip_terms = ['', 'skip', 'skipped', 'none', 'no', 'not']
        return sum(1 for meal in self.meals.values() 
                  if meal.lower() not in skip_terms and meal.strip() != '')
    
    @property
    def sleep_score(self) -> float:
        """Calculate sleep score (0-100)"""
        base_score = min(self.sleep_hours / 8 * 70, 70)  # 70% for duration
        quality_score = (self.sleep_quality / 10) * 30  # 30% for quality
        return round(base_score + quality_score, 1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['meals_eaten'] = self.meals_eaten
        data['sleep_score'] = self.sleep_score
        return data

@dataclass
class MessMenu:
    """Today's mess menu"""
    menu_id: str
    user_id: str
    date: str
    meal_time: str  # breakfast, lunch, dinner
    items: List[str]
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# ---------------- DATABASE MANAGER ------------------
class DatabaseManager:
    """Manages data storage and retrieval"""
    
    def __init__(self):
        Config.DATA_DIR.mkdir(exist_ok=True)
        Config.BACKUP_DIR.mkdir(exist_ok=True)
        
        self.profiles_file = Config.DATA_DIR / "user_profiles.pkl"
        self.logs_file = Config.DATA_DIR / "daily_logs.pkl"
        self.mess_menus_file = Config.DATA_DIR / "mess_menus.pkl"
        
        # Load existing data
        self.profiles = self._load_data(self.profiles_file, {})
        self.logs = self._load_data(self.logs_file, {})
        self.mess_menus = self._load_data(self.mess_menus_file, {})
    
    def _load_data(self, filepath: Path, default: Any) -> Any:
        """Load data from file"""
        try:
            if filepath.exists():
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"⚠️ Warning: Could not load {filepath}: {e}")
        return default
    
    def _save_data(self, filepath: Path, data: Any):
        """Save data to file with backup"""
        try:
            if filepath.exists():
                backup_file = Config.BACKUP_DIR / f"{filepath.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
                with open(backup_file, 'wb') as f:
                    pickle.dump(data, f)
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"❌ Error saving data: {e}")
            raise
    
    def save_profile(self, profile: UserProfile) -> bool:
        """Save user profile"""
        try:
            self.profiles[profile.user_id] = profile
            self._save_data(self.profiles_file, self.profiles)
            return True
        except Exception as e:
            print(f"❌ Error saving profile: {e}")
            return False
    
    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile"""
        return self.profiles.get(user_id)
    
    def save_log(self, log: DailyLog) -> bool:
        """Save daily log"""
        try:
            if log.user_id not in self.logs:
                self.logs[log.user_id] = []
            
            # Remove existing log for same date
            self.logs[log.user_id] = [l for l in self.logs[log.user_id] if l.date != log.date]
            self.logs[log.user_id].append(log)
            self._save_data(self.logs_file, self.logs)
            return True
        except Exception as e:
            print(f"❌ Error saving log: {e}")
            return False
    
    def get_log_by_date(self, user_id: str, date: str) -> Optional[DailyLog]:
        """Get log by specific date"""
        user_logs = self.logs.get(user_id, [])
        for log in user_logs:
            if log.date == date:
                return log
        return None
    
    def save_mess_menu(self, menu: MessMenu) -> bool:
        """Save mess menu"""
        try:
            if menu.user_id not in self.mess_menus:
                self.mess_menus[menu.user_id] = []
            
            # Remove existing menu for same date and meal time
            self.mess_menus[menu.user_id] = [
                m for m in self.mess_menus[menu.user_id] 
                if not (m.date == menu.date and m.meal_time == menu.meal_time)
            ]
            
            self.mess_menus[menu.user_id].append(menu)
            self._save_data(self.mess_menus_file, self.mess_menus)
            return True
        except Exception as e:
            print(f"❌ Error saving mess menu: {e}")
            return False
    
    def get_user_logs(self, user_id: str, days: int = 30) -> List[DailyLog]:
        """Get user's recent logs"""
        logs = self.logs.get(user_id, [])
        if not logs:
            return []
        
        sorted_logs = sorted(logs, key=lambda x: x.date, reverse=True)
        return sorted_logs[:min(days, len(sorted_logs))]
    
    def get_today_mess_menu(self, user_id: str) -> Optional[List[MessMenu]]:
        """Get today's mess menus"""
        today = datetime.now().strftime("%Y-%m-%d")
        menus = self.mess_menus.get(user_id, [])
        return [m for m in menus if m.date == today]

# ---------------- LLM SERVICE ------------------
class LLMService:
    """Manages LLM interactions with error handling"""
    
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.llm = ChatGroq(
            api_key=api_key,
            model=Config.MODEL,
            temperature=0.7,
            max_tokens=Config.MAX_TOKENS
        )
    
    def call_with_retry(self, prompt: str, max_retries: int = 2) -> str:
        """Call LLM with retry logic"""
        for attempt in range(max_retries):
            try:
                if len(prompt) > 3500:
                    prompt = prompt[:3500] + "\n[Content truncated...]"
                
                response = self.llm.invoke([HumanMessage(content=prompt)])
                return response.content.strip()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                print(f"⚠️ LLM call failed (attempt {attempt + 1}): {str(e)[:100]}")
                import time
                time.sleep(1)
        return "Error: Could not get response from LLM"
    
    def extract_json(self, text: str) -> Dict:
        """Extract JSON from LLM response"""
        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"⚠️ JSON extraction failed: {str(e)[:100]}")
        
        return {"error": "Could not parse JSON response", "raw_text": text[:200]}

# ---------------- TAO AGENT FRAMEWORK ------------------
class AgentState(TypedDict):
    """State for autonomous agent"""
    profile: Dict[str, Any]
    recent_logs: List[Dict[str, Any]]
    reasoning_trace: List[Dict[str, Any]]
    current_goal: str
    constraints: Dict[str, Any]
    action_history: List[Dict[str, Any]]
    iteration: int
    max_iterations: int
    should_continue: bool
    final_output: List[str]
    confidence: float
    metadata: Dict[str, Any]
    mess_menu: Optional[Dict[str, Any]] = None
    execution_mode: str = "cli"  # NEW: Track execution mode

class TAOAgent:
    """Base agent with Think-Act-Observe framework"""
    
    def __init__(self, agent_type: str, llm_service: LLMService, execution_mode: str = "cli"):
        self.agent_type = agent_type
        self.llm = llm_service
        self.execution_mode = execution_mode  # NEW: Track execution mode
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow"""
        graph = StateGraph(AgentState)
        
        graph.add_node("think", self._think)
        graph.add_node("act", self._act)
        graph.add_node("observe", self._observe)
        
        graph.set_entry_point("think")
        graph.add_edge("think", "act")
        graph.add_edge("act", "observe")
        graph.add_conditional_edges("observe", self._decide_continuation)
        
        return graph.compile()
    
    def _think(self, state: AgentState) -> AgentState:
        """THINK: Analyze and plan"""
        try:
            think_prompt = self._build_think_prompt(state)
            response = self.llm.call_with_retry(think_prompt)
            decision = self.llm.extract_json(response)
            
            state["current_goal"] = decision.get("goal", "analyze_patterns")
            
            state["reasoning_trace"].append({
                "phase": "think",
                "iteration": state["iteration"],
                "analysis": decision.get("analysis", ""),
                "goal": state["current_goal"],
                "confidence": decision.get("confidence", 0.5)
            })
            
        except Exception as e:
            state["current_goal"] = "error_recovery"
            state["reasoning_trace"].append({
                "phase": "think",
                "iteration": state["iteration"],
                "error": str(e)[:100],
                "goal": "error_recovery"
            })
        
        return state
    
    def _act(self, state: AgentState) -> AgentState:
        """ACT: Execute planned action"""
        try:
            act_prompt = self._build_act_prompt(state)
            response = self.llm.call_with_retry(act_prompt)
            
            state["action_history"].append({
                "iteration": state["iteration"],
                "goal": state["current_goal"],
                "action": state["current_goal"],
                "output": response[:500] + "..." if len(response) > 500 else response
            })
            
            if "final_output" not in state:
                state["final_output"] = []
            
            if state["iteration"] == 0:
                state["final_output"] = [response]
            else:
                state["final_output"].append(f"\n\n[Deep Analysis - Iteration {state['iteration'] + 1}]:\n{response}")
            
        except Exception as e:
            state["action_history"].append({
                "iteration": state["iteration"],
                "error": str(e)[:100],
                "action": "failed"
            })
            if "final_output" not in state:
                state["final_output"] = [f"Error in analysis: {str(e)[:100]}"]
        
        return state
    
    def _observe(self, state: AgentState) -> AgentState:
        """OBSERVE: Evaluate results and learn"""
        try:
            observe_prompt = self._build_observe_prompt(state)
            response = self.llm.call_with_retry(observe_prompt)
            observation = self.llm.extract_json(response)
            
            satisfaction = observation.get("satisfaction_score", 0.5)
            new_confidence = observation.get("confidence", state.get("confidence", 0.5))
            
            state["reasoning_trace"].append({
                "phase": "observe",
                "iteration": state["iteration"],
                "observation": observation.get("assessment", ""),
                "satisfaction": satisfaction,
                "learning": observation.get("learnings", "")
            })
            
            state["confidence"] = new_confidence
            
            # CRITICAL FIX: Diet agent runs exactly ONE iteration
            if self.agent_type == "diet":
                state["should_continue"] = False
                print(f"🤖 Diet agent: Stopping after 1 iteration (web-optimized)")
            else:
                max_reached = state["iteration"] >= state["max_iterations"] - 1
                high_satisfaction = satisfaction >= 0.8
                low_confidence = new_confidence < 0.3 and state["iteration"] >= 2
                
                state["should_continue"] = (
                    observation.get("should_continue", False) and 
                    not max_reached and 
                    not high_satisfaction and 
                    not low_confidence
                )
            
        except Exception as e:
            state["should_continue"] = state["iteration"] < 1
            state["reasoning_trace"].append({
                "phase": "observe",
                "iteration": state["iteration"],
                "error": str(e)[:100]
            })
        
        state["iteration"] += 1
        return state
    
    def _decide_continuation(self, state: AgentState) -> str:
        """Decide whether to continue TAO loop"""
        return "think" if state.get("should_continue", True) else END
    
    def _build_think_prompt(self, state: AgentState) -> str:
        """Build THINK phase prompt"""
        profile = state["profile"]
        logs = state["recent_logs"]
        
        return f"""
        You are a {self.agent_type} health AI agent in the THINK phase.
        
        USER PROFILE:
        Name: {profile.get('name', 'User')}
        Diet: {profile.get('diet_type', 'mixed')}
        Goal: {profile.get('fitness_goal', 'general health')}
        BMI: {profile.get('bmi', 22)} ({profile.get('bmi_category', 'Healthy')})
        
        RECENT HEALTH DATA (Last {len(logs)} days):
        {json.dumps(logs, indent=2)[:1500]}
        
        CURRENT ITERATION: {state['iteration'] + 1}/{state['max_iterations']}
        
        Analyze the situation and decide what specific goal to pursue next.
        Consider what would provide most value to the user right now.
        
        Return JSON with:
        {{
            "goal": "specific_goal_name",
            "analysis": "brief analysis",
            "confidence": 0.0-1.0,
            "reasoning": "why this goal is appropriate"
        }}
        
        Available goal types: analyze_patterns, identify_gaps, create_plan, assess_risks
        """
    
    def _build_act_prompt(self, state: AgentState) -> str:
        """Build ACT phase prompt"""
        goal = state.get("current_goal", "analyze_patterns")
        execution_mode = state.get("execution_mode", "cli")
        
        prompt_templates = {
            "diet": self._diet_act_prompt,
            "fitness": self._fitness_act_prompt,
            "health": self._health_act_prompt,
            "mess_optimizer": self._mess_optimizer_act_prompt
        }
        
        template = prompt_templates.get(self.agent_type, self._generic_act_prompt)
        return template(state, goal, execution_mode)
    
    def _build_observe_prompt(self, state: AgentState) -> str:
        """Build OBSERVE phase prompt"""
        last_action = state["action_history"][-1] if state["action_history"] else {}
        
        return f"""
        You are in the OBSERVE phase of the TAO loop.
        
        ITERATION: {state['iteration'] + 1}/{state['max_iterations']}
        
        LAST ACTION:
        Goal: {state.get('current_goal', 'N/A')}
        
        Evaluate if we should continue or stop:
        
        Return JSON:
        {{
            "assessment": "brief evaluation",
            "satisfaction_score": 0.0-1.0,
            "confidence": 0.0-1.0,
            "learnings": "key insights",
            "should_continue": true/false,
            "reason": "why continue or stop"
        }}
        """
    
    # Agent-specific prompt templates
    def _diet_act_prompt(self, state: AgentState, goal: str, execution_mode: str) -> str:
        profile = state["profile"]
        logs = state["recent_logs"]
        
        mode_note = "WEB MODE" if execution_mode == "web" else "CLI MODE"
        
        return f"""
        As a DIET AND NUTRITION expert, create a personalized diet plan.
        
        USER INFORMATION:
        - Name: {profile.get('name', 'User')}
        - Age: {profile.get('age', 25)}
        - Diet Type: {profile.get('diet_type', 'mixed')}
        - Weight: {profile.get('weight', 70)} kg
        - Height: {profile.get('height', 170)} cm
        - BMI: {profile.get('bmi', 22)} ({profile.get('bmi_category', 'Healthy')})
        - Fitness Goal: {profile.get('fitness_goal', 'general health')}
        - Allergies: {', '.join(profile.get('allergies', ['None']))}
        
        RECENT EATING PATTERNS (Last {len(logs)} days):
        Average meals per day: {sum(log.get('meals_eaten', 0) for log in logs) / max(len(logs), 1):.1f}
        Average energy level: {sum(log.get('energy_level', 5) for log in logs) / max(len(logs), 1):.1f}/10
        
        CURRENT ANALYSIS FOCUS: {goal}
        EXECUTION MODE: {mode_note}
        
        CRITICAL INSTRUCTIONS:
        1. Provide exactly 7 days (Day 1 through Day 7)
        2. Each day must be a complete, independent daily plan
        3. NO weekly summaries or merged nutrition targets
        4. Each day must have identical structure
        5. Keep outputs concise for web display
        
        CREATE A DETAILED 7-DAY MEAL PLAN WITH THE FOLLOWING STRUCTURE:
        
        1. DAILY MEAL STRUCTURE (REQUIRED FOR ALL DAYS):
        For EACH day (Day 1 through Day 7):
        - Breakfast (time, specific foods, portion sizes)
        - Lunch (time, specific foods, portion sizes)
        - Dinner (time, specific foods, portion sizes)
        - 2 Snacks (timing and suggestions)
        
        2. NUTRITIONAL TARGETS PER DAY (REQUIRED):
        For EACH day:
        - Daily calorie target: ____ calories
        - Protein: ____ grams
        - Carbohydrates: ____ grams
        - Fats: ____ grams
        - Key micronutrients focus: ____
        
        {"3. ADDITIONAL SECTIONS FOR CLI MODE ONLY (SKIP IN WEB MODE):" if execution_mode == "cli" else "3. WEB MODE: Keep output concise"}
        {"   - 3 simple recipes for " + profile.get('diet_type', 'mixed') + " diet" if execution_mode == "cli" else ""}
        {"   - Cooking instructions" if execution_mode == "cli" else ""}
        {"   - Meal prep tips" if execution_mode == "cli" else ""}
        {"   - Grocery shopping list (organized by category)" if execution_mode == "cli" else ""}
        {"   - Estimated cost" if execution_mode == "cli" else ""}
        {"   - Progress tracking guidelines" if execution_mode == "cli" else ""}
        
        4. FORMATTING RULES:
        - Use clear headers: "DAY 1:", "DAY 2:", etc.
        - Use bullet points for readability
        - Keep each day's section self-contained
        - Do NOT create weekly summaries
        - Do NOT merge nutrition data across days
        
        FAILURE CONDITIONS (AVOID THESE):
        - DO NOT create weekly summaries
        - DO NOT merge nutrition targets across days
        - DO NOT skip daily nutritional targets
        - DO NOT create vague or incomplete daily plans
        
        Make it SPECIFIC, PRACTICAL, and TAILORED to {profile.get('diet_type', 'mixed')} diet.
        Provide clear, actionable steps for each day.
        """
    
    def _fitness_act_prompt(self, state: AgentState, goal: str, execution_mode: str) -> str:
        profile = state["profile"]
        logs = state["recent_logs"]
        
        return f"""
        You are a FITNESS AND EXERCISE expert. Create a COMPREHENSIVE, PROGRESSIVE workout plan.
        
        USER INFORMATION:
        - Name: {profile.get('name', 'User')}
        - Age: {profile.get('age', 25)}
        - Fitness Goal: {profile.get('fitness_goal', 'general fitness')}
        - Activity Level: {profile.get('activity_level', 'moderate')}
        - Weight: {profile.get('weight', 70)} kg
        - Height: {profile.get('height', 170)} cm
        - BMI: {profile.get('bmi', 22)} ({profile.get('bmi_category', 'Healthy')})
        
        RECENT ACTIVITY DATA (Last {len(logs)} days):
        Average sleep: {sum(log.get('sleep_hours', 7) for log in logs) / max(len(logs), 1):.1f} hours
        Average energy: {sum(log.get('energy_level', 5) for log in logs) / max(len(logs), 1):.1f}/10
        Workout days: {sum(1 for log in logs if log.get('workout_duration_min', 0) > 0)}/{len(logs)}
        
        CURRENT ANALYSIS FOCUS: {goal}
        
        CREATE A DETAILED 4-WEEK FITNESS PROGRAM WITH THE FOLLOWING SECTIONS:
        
        1. PROGRAM OVERVIEW:
           - Weekly focus and objectives
           - Expected outcomes
           - Equipment needed (minimal)
           - Time commitment per day
        
        2. WEEKLY SCHEDULE (Monday-Sunday):
           - Day 1: Strength Training (exercises, sets, reps, rest)
           - Day 2: Cardio (type, duration, intensity)
           - Day 3: Active Recovery (activities)
           - Day 4: Strength Training (different focus)
           - Day 5: Cardio (variation)
           - Day 6: Flexibility/Mobility
           - Day 7: Rest or Light Activity
        
        3. EXERCISE LIBRARY WITH DETAILS:
           For each exercise include:
           - Proper form description
           - Common mistakes to avoid
           - Progression options
           - Regressions if too difficult
           - Targeted muscle groups
        
        4. WARM-UP ROUTINE (5-10 minutes):
           - Dynamic stretches
           - Activation exercises
           - Mobility work
        
        5. COOL-DOWN ROUTINE (5-10 minutes):
           - Static stretches
           - Breathing exercises
           - Recovery techniques
        
        6. PROGRESSION SYSTEM:
           - How to increase difficulty each week
           - When to add weight/reps
           - Deload week schedule
           - Plateaus prevention
        
        Make it SPECIFIC with exact exercises, sets, reps, and rest times.
        Focus on SAFETY and PROPER FORM.
        Include both GYM and HOME options.
        """
    
    def _health_act_prompt(self, state: AgentState, goal: str, execution_mode: str) -> str:
        profile = state["profile"]
        logs = state["recent_logs"]
        
        return f"""
        You are a HEALTH AND WELLNESS analyst. Provide COMPREHENSIVE health assessment.
        
        USER INFORMATION:
        - Name: {profile.get('name', 'User')}
        - Age: {profile.get('age', 25)}
        - Weight: {profile.get('weight', 70)} kg
        - Height: {profile.get('height', 170)} cm
        - BMI: {profile.get('bmi', 22)} ({profile.get('bmi_category', 'Healthy')})
        - Diet Type: {profile.get('diet_type', 'mixed')}
        - Medical Conditions: {', '.join(profile.get('medical_conditions', ['None']))}
        - Allergies: {', '.join(profile.get('allergies', ['None']))}
        
        RECENT HEALTH METRICS (Last {len(logs)} days):
        Average sleep: {sum(log.get('sleep_hours', 7) for log in logs) / max(len(logs), 1):.1f} hours
        Sleep quality: {sum(log.get('sleep_quality', 5) for log in logs) / max(len(logs), 1):.1f}/10
        Average energy: {sum(log.get('energy_level', 5) for log in logs) / max(len(logs), 1):.1f}/10
        Average mood: {sum(log.get('mood_level', 5) for log in logs) / max(len(logs), 1):.1f}/10
        Average stress: {sum(log.get('stress_level', 5) for log in logs) / max(len(logs), 1):.1f}/10
        
        CURRENT ANALYSIS FOCUS: {goal}
        
        PROVIDE A DETAILED HEALTH ANALYSIS WITH THE FOLLOWING SECTIONS:
        
        1. OVERALL HEALTH ASSESSMENT:
           - Current health status summary
           - Strengths in current lifestyle
           - Areas needing improvement
           - Risk level (Low/Medium/High)
        
        2. SLEEP ANALYSIS:
           - Sleep pattern evaluation
           - Sleep quality assessment
           - Impact on daily energy
           - Specific improvement suggestions
        
        3. NUTRITION ASSESSMENT:
           - Current eating pattern analysis
           - Nutrient intake estimation
           - Potential deficiencies
           - Meal timing optimization
        
        4. ACTIVITY AND EXERCISE EVALUATION:
           - Current activity level analysis
           - Exercise consistency
           - Recovery adequacy
        
        5. ACTIONABLE RECOMMENDATIONS:
           - Top 3 immediate changes to make
           - Weekly improvement plan
           - Daily health habits to establish
        
        IMPORTANT: Do not provide medical diagnosis.
        Focus on LIFESTYLE IMPROVEMENTS and PREVENTIVE MEASURES.
        Be SPECIFIC and ACTIONABLE.
        Include CLEAR, MEASURABLE goals.
        """
    
    def _mess_optimizer_act_prompt(self, state: AgentState, goal: str, execution_mode: str) -> str:
        """Mess optimizer specific prompt"""
        profile = state["profile"]
        mess_menu = state.get("mess_menu", {})
        
        if not mess_menu:
            return """
            No mess menu provided. Please upload today's mess menu first.
            Use the 'Upload Mess Menu' option to add your mess food items.
            """
        
        meal_time = mess_menu.get("meal_time", "lunch")
        items = mess_menu.get("items", [])
        notes = mess_menu.get("notes", "")
        
        return f"""
        You are a MESS FOOD OPTIMIZATION expert specializing in Indian hostel food.
        
        USER PROFILE:
        - Name: {profile.get('name', 'User')}
        - Diet Type: {profile.get('diet_type', 'mixed')}
        - Fitness Goal: {profile.get('fitness_goal', 'general health')}
        - Weight: {profile.get('weight', 70)} kg
        - Height: {profile.get('height', 170)} cm
        - BMI: {profile.get('bmi', 22)} ({profile.get('bmi_category', 'Healthy')})
        - Allergies: {', '.join(profile.get('allergies', ['None']))}
        - Budget: ₹{profile.get('weekly_budget', 500)}/week extra for supplements
        
        TODAY'S MESS MENU ({meal_time.upper()}):
        {json.dumps(items, indent=2)}
        
        Additional Notes: {notes}
        
        CURRENT ANALYSIS FOCUS: {goal}
        
        CREATE A DETAILED MESS FOOD OPTIMIZATION PLAN WITH THE FOLLOWING SECTIONS:
        
        1. MENU ANALYSIS & SCORING:
           Rate each item on a scale of 1-10 for:
           - Nutritional value (protein, fiber, vitamins)
           - Suitability for {profile.get('fitness_goal', 'general health')} goal
           - {profile.get('diet_type', 'mixed')} diet compatibility
           - Value for money
        
        2. RECOMMENDED SELECTIONS:
           - MUST EAT items (highest nutritional value)
           - AVOID items (low nutrition, high empty calories)
           - MODERATE items (eat in moderation)
           - Portion size recommendations for each
        
        3. PROTEIN OPTIMIZATION STRATEGY:
           - Protein-rich items to prioritize
           - Protein supplements needed (if any)
           - How to combine items for complete protein
           - Target: {profile.get('weight', 70) * 1.2:.0f}g protein/day recommendation
        
        4. MEAL ENHANCEMENT HACKS:
           - Simple additions to improve nutrition
           - Flavor enhancement without extra calories
           - How to make bland items more nutritious
        
        5. COST-BENEFIT ANALYSIS:
           - Cost of selected mess items
           - Cost of supplements
           - Total nutritional value achieved
        
        Be VERY SPECIFIC with Indian hostel context.
        Include exact food names from the menu.
        Prioritize PRACTICALITY and AFFORDABILITY.
        Consider LIMITED COOKING FACILITIES in hostel.
        """
    
    def _generic_act_prompt(self, state: AgentState, goal: str, execution_mode: str) -> str:
        return f"""
        You are a {self.agent_type} health advisor. Provide detailed, personalized guidance.
        
        USER: {state['profile'].get('name', 'User')}
        CURRENT FOCUS: {goal}
        
        Provide comprehensive, actionable advice that is:
        1. Personalized to the user's specific situation
        2. Practical and immediately implementable
        3. Based on evidence and best practices
        4. Includes specific steps and timelines
        5. Considers real-world constraints
        
        Format with clear sections, bullet points, and specific recommendations.
        """
    
    def run(self, profile: Dict, logs: List[Dict], max_iterations: int = 3, mess_menu: Optional[Dict] = None) -> Dict:
        """Execute the TAO agent with dynamic iterations"""
        if not logs:
            actual_max = 2
        elif len(logs) < 3:
            actual_max = 3
        else:
            actual_max = min(4, max_iterations)
        
        initial_state: AgentState = {
            "profile": profile,
            "recent_logs": logs[-7:],
            "reasoning_trace": [],
            "current_goal": "analyze_patterns",
            "constraints": {"max_iterations": actual_max},
            "action_history": [],
            "iteration": 0,
            "max_iterations": actual_max,
            "should_continue": True,
            "final_output": [],
            "confidence": 0.5,
            "metadata": {
                "agent_type": self.agent_type,
                "start_time": datetime.now().isoformat(),
                "data_points": len(logs),
                "execution_mode": self.execution_mode
            },
            "mess_menu": mess_menu,
            "execution_mode": self.execution_mode  # Pass execution mode to state
        }
        
        try:
            final_state = self.graph.invoke(initial_state)
            
            # GUARANTEED RETURN BLOCK - Always returns structured dictionary
            final_output = ""
            if isinstance(final_state.get("final_output"), list) and final_state["final_output"]:
                if len(final_state["final_output"]) == 1:
                    final_output = final_state["final_output"][0]
                else:
                    final_output = "\n".join(str(item) for item in final_state["final_output"])
            else:
                final_output = "No detailed analysis generated. Please try again with more data."
            
            # Always return structured response
            return {
                "success": True,
                "agent_type": self.agent_type,
                "output": final_output,
                "iterations": final_state.get("iteration", 0),
                "max_iterations": actual_max,
                "confidence": final_state.get("confidence", 0.5),
                "reasoning_summary": [
                    {
                        "phase": trace.get("phase"),
                        "iteration": trace.get("iteration"),
                        "insight": trace.get("analysis") or trace.get("observation") or trace.get("learning") or ""
                    }
                    for trace in final_state.get("reasoning_trace", [])
                ],
                "metadata": {
                    **final_state.get("metadata", {}),
                    "end_time": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            # Error handling with guaranteed return
            return {
                "success": False,
                "error": str(e),
                "agent_type": self.agent_type,
                "output": f"Agent execution failed: {str(e)[:100]}",
                "iterations": initial_state.get("iteration", 0),
                "max_iterations": actual_max,
                "confidence": 0.0,
                "reasoning_summary": [],
                "metadata": {
                    "agent_type": self.agent_type,
                    "error": str(e),
                    "end_time": datetime.now().isoformat()
                }
            }

# ---------------- MAIN APPLICATION ------------------
class HealthAISystem:
    """Main production system"""
    
    def __init__(self, execution_mode: str = "cli"):
        self.db = DatabaseManager()
        self.execution_mode = execution_mode  # Track execution mode
        try:
            self.llm_service = LLMService()
            
            # Initialize agents with execution mode
            self.agents = {
                "diet": TAOAgent("diet", self.llm_service, execution_mode),
                "fitness": TAOAgent("fitness", self.llm_service, execution_mode),
                "health": TAOAgent("health", self.llm_service, execution_mode),
                "mess_optimizer": TAOAgent("mess_optimizer", self.llm_service, execution_mode)
            }
        except Exception as e:
            print(f"❌ Failed to initialize LLM service: {e}")
            print("Please check your GROQ_API_KEY in .env file")
            sys.exit(1)
        
        self.current_user_id = None
    
    def safe_input(self, prompt: str) -> str:
        """Safe input that works in both CLI and web modes"""
        if self.execution_mode == "web":
            # In web mode, return default or empty
            # For boolean choices, default to no/empty
            if prompt.lower().startswith("save") or "? (y/n)" in prompt.lower():
                return "n"
            if "select:" in prompt.lower():
                return ""  # Let main menu handle default
            return ""
        return input(prompt).strip()
    
    def setup_profile(self) -> bool:
        """Comprehensive profile setup"""
        if self.execution_mode == "web":
            print("Web mode: Profile setup handled by Streamlit UI")
            return False
        
        print("\n" + "="*60)
        print("👤 USER PROFILE SETUP")
        print("="*60)
        
        user_id = f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print("\n📝 BASIC INFORMATION")
        print("-"*30)
        name = input("Full Name: ").strip() or "User"
        
        while True:
            try:
                age = int(input("Age: ").strip() or "25")
                if 10 <= age <= 100:
                    break
                print("Please enter age 10-100")
            except:
                print("Please enter a valid number")
        
        while True:
            try:
                weight = float(input("Weight (kg): ").strip() or "70")
                if 30 <= weight <= 200:
                    break
                print("Please enter weight 30-200 kg")
            except:
                print("Please enter a valid number")
        
        while True:
            try:
                height = float(input("Height (cm): ").strip() or "170")
                if 100 <= height <= 250:
                    break
                print("Please enter height 100-250 cm")
            except:
                print("Please enter a valid number")
        
        print("\n🥗 DIET TYPE")
        print("-"*30)
        print("1. Vegetarian")
        print("2. Non-Vegetarian")
        print("3. Eggetarian")
        print("4. Vegan")
        
        diet_map = {
            "1": DietType.VEGETARIAN,
            "2": DietType.NON_VEGETARIAN,
            "3": DietType.EGGETARIAN,
            "4": DietType.VEGAN
        }
        
        while True:
            choice = input("Choose (1-4): ").strip()
            if choice in diet_map:
                diet_type = diet_map[choice]
                break
            print("Invalid choice")
        
        fitness_goal = input("\nFitness goal (e.g., lose weight, build muscle, improve endurance): ").strip() or "Improve overall health"
        
        print("\n📊 ACTIVITY LEVEL")
        print("-"*30)
        print("1. Sedentary (office job, little exercise)")
        print("2. Light (light exercise 1-3 days/week)")
        print("3. Moderate (moderate exercise 3-5 days/week)")
        print("4. Active (hard exercise 6-7 days/week)")
        print("5. Very Active (athlete, physical job)")
        
        activity_levels = ["sedentary", "light", "moderate", "active", "very_active"]
        while True:
            choice = input("Choose (1-5): ").strip()
            if choice in ["1", "2", "3", "4", "5"]:
                activity_level = activity_levels[int(choice) - 1]
                break
            print("Invalid choice")
        
        print("\n🏠 HOSTEL INFORMATION")
        print("-"*30)
        is_student = input("Hostel student? (y/n): ").strip().lower() == 'y'
        has_mess = False
        weekly_budget = 500.0
        
        if is_student:
            has_mess = input("Have mess access? (y/n): ").strip().lower() == 'y'
            if has_mess:
                try:
                    weekly_budget = float(input(f"Extra food budget/week (₹): ").strip() or "500")
                except:
                    weekly_budget = 500.0
        
        medical_input = input("\nMedical conditions (comma separated or 'none'): ").strip()
        medical_conditions = [c.strip() for c in medical_input.split(",")] if medical_input.lower() != 'none' and medical_input else []
        
        allergy_input = input("Food allergies (comma separated or 'none'): ").strip()
        allergies = [a.strip() for a in allergy_input.split(",")] if allergy_input.lower() != 'none' and allergy_input else []
        
        profile = UserProfile(
            user_id=user_id,
            name=name,
            age=age,
            weight=weight,
            height=height,
            diet_type=diet_type,
            fitness_goal=fitness_goal,
            activity_level=activity_level,
            is_student=is_student,
            has_mess_access=has_mess,
            weekly_budget=weekly_budget,
            medical_conditions=medical_conditions,
            allergies=allergies
        )
        
        if self.db.save_profile(profile):
            self.current_user_id = user_id
            print(f"\n✅ Profile created!")
            print(f"   Name: {name}")
            print(f"   BMI: {profile.bmi} ({profile.bmi_category})")
            print(f"   Diet: {diet_type.value}")
            print(f"   Goal: {fitness_goal}")
            return True
        else:
            print("❌ Failed to save profile")
            return False
    
    def upload_mess_menu(self):
        """Upload today's mess menu"""
        if not self.current_user_id:
            print("Please setup profile first!")
            return
        
        profile = self.db.get_profile(self.current_user_id)
        if not profile.has_mess_access:
            print("❌ You don't have mess access in your profile.")
            return
        
        print("\n" + "="*60)
        print("📋 UPLOAD TODAY'S MESS MENU")
        print("="*60)
        
        today = datetime.now().strftime("%Y-%m-%d")
        print(f"\n📅 Date: {today}")
        
        print("\n🍽️ SELECT MEAL TIME:")
        print("1. Breakfast")
        print("2. Lunch")
        print("3. Dinner")
        
        meal_map = {
            "1": "breakfast",
            "2": "lunch",
            "3": "dinner"
        }
        
        while True:
            choice = self.safe_input("\nChoose (1-3): ")
            if choice in meal_map:
                meal_time = meal_map[choice]
                break
            print("Invalid choice")
        
        print(f"\n📝 Enter {meal_time} menu items (one per line, type 'done' when finished):")
        items = []
        while True:
            item = self.safe_input(f"Item {len(items) + 1}: ")
            if item.lower() == 'done':
                if len(items) == 0:
                    print("Please enter at least one item")
                    continue
                break
            if item:
                items.append(item)
        
        notes = self.safe_input("\nAdditional notes (e.g., 'limited quantity', 'extra spicy'): ")
        
        menu_id = f"menu_{today}_{meal_time}_{self.current_user_id}"
        menu = MessMenu(
            menu_id=menu_id,
            user_id=self.current_user_id,
            date=today,
            meal_time=meal_time,
            items=items,
            notes=notes
        )
        
        if self.db.save_mess_menu(menu):
            print(f"\n✅ {meal_time.capitalize()} menu saved!")
            print(f"   Items: {len(items)}")
            print(f"   Date: {today}")
        else:
            print("❌ Failed to save menu")
    
    def run_mess_optimizer(self):
        """Run mess food optimizer with today's menu"""
        if not self.current_user_id:
            print("Please setup profile first!")
            return
        
        profile = self.db.get_profile(self.current_user_id)
        if not profile.has_mess_access:
            print("❌ You don't have mess access in your profile.")
            return
        
        # Get today's mess menus
        today_menus = self.db.get_today_mess_menu(self.current_user_id)
        
        if not today_menus:
            print("\n❌ No mess menu uploaded for today.")
            print("Please use 'Upload Mess Menu' option first.")
            return
        
        print("\n" + "="*60)
        print("🤖 MESS FOOD OPTIMIZER")
        print("="*60)
        
        # Let user choose which meal to optimize
        print("\n📋 TODAY'S UPLOADED MENUS:")
        for i, menu in enumerate(today_menus, 1):
            print(f"{i}. {menu.meal_time.capitalize()}: {', '.join(menu.items[:3])}...")
        
        print(f"{len(today_menus) + 1}. Optimize ALL meals")
        
        while True:
            try:
                choice = int(self.safe_input(f"\nSelect menu to optimize (1-{len(today_menus) + 1}): "))
                if 1 <= choice <= len(today_menus) + 1:
                    break
                print(f"Please enter 1-{len(today_menus) + 1}")
            except:
                print("Please enter a valid number")
        
        selected_menus = []
        if choice == len(today_menus) + 1:
            selected_menus = today_menus
            print(f"\n🚀 Optimizing ALL {len(today_menus)} meals...")
        else:
            selected_menus = [today_menus[choice - 1]]
            print(f"\n🚀 Optimizing {selected_menus[0].meal_time}...")
        
        # Get logs for context
        logs = self.db.get_user_logs(self.current_user_id, days=7)
        logs_dict = [log.to_dict() for log in logs] if logs else []
        profile_dict = profile.to_dict()
        
        results = []
        for menu in selected_menus:
            print(f"\n{'='*60}")
            print(f"🍽️  OPTIMIZING {menu.meal_time.upper()}")
            print("="*60)
            print(f"Menu: {', '.join(menu.items)}")
            
            menu_dict = menu.to_dict()
            agent = self.agents["mess_optimizer"]
            
            result = agent.run(profile_dict, logs_dict, mess_menu=menu_dict)
            
            if result["success"]:
                print(f"\n✅ OPTIMIZATION COMPLETE")
                print(f"Iterations: {result['iterations']}/{result['max_iterations']}")
                print(f"Confidence: {result['confidence']:.2f}")
                print("="*60)
                
                output = result["output"]
                lines = output.split('\n')
                
                if len(lines) > 40:
                    print('\n'.join(lines[:40]))
                    print("\n... [Output truncated - showing first 40 lines] ...")
                else:
                    print(output)
                
                results.append({
                    "meal_time": menu.meal_time,
                    "result": result,
                    "menu_items": menu.items
                })
                
                # Ask to save
                save = self.safe_input(f"\n💾 Save {menu.meal_time} optimization? (y/n): ")
                if save.lower() == 'y':
                    self._save_mess_optimization(result, profile.name, menu.meal_time, menu.items)
            else:
                print(f"❌ Optimization failed for {menu.meal_time}: {result.get('error', 'Unknown error')}")
        
        # Show summary if multiple meals
        if len(results) > 1:
            self._show_mess_optimization_summary(results, profile.name)
    
    def _save_mess_optimization(self, result: Dict, username: str, meal_time: str, menu_items: List[str]):
        """Save mess optimization to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{username}_mess_optimizer_{meal_time}_{timestamp}.txt"
        filepath = Config.DATA_DIR / filename
        
        content = f"""
MESS FOOD OPTIMIZATION REPORT
{'='*60}
User: {username}
Meal: {meal_time.capitalize()}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Menu: {', '.join(menu_items[:5])}{'...' if len(menu_items) > 5 else ''}
Iterations: {result['iterations']}/{result['max_iterations']}
Confidence: {result['confidence']:.2f}
{'='*60}

{result['output']}

{'='*60}
AGENT REASONING:
{'='*60}
"""
        
        for i, step in enumerate(result.get('reasoning_summary', []), 1):
            content += f"\n{i}. [{step.get('phase', 'unknown').upper()}] (Iteration {step.get('iteration', 0)}):\n"
            content += f"   {step.get('insight', '')}\n"
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Optimization saved to: {filepath}")
        except Exception as e:
            print(f"❌ Save failed: {e}")
    
    def _show_mess_optimization_summary(self, results: List[Dict], username: str):
        """Show summary of all meal optimizations"""
        print("\n" + "="*60)
        print("📊 DAILY MESS OPTIMIZATION SUMMARY")
        print("="*60)
        print(f"User: {username}")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        print("="*60)
        
        total_protein_estimate = 0
        total_calorie_estimate = 0
        total_supplement_cost = 0
        
        for result in results:
            meal_time = result["meal_time"]
            output = result["result"]["output"]
            
            # Extract estimates from output (simplified parsing)
            protein_match = re.search(r'(\d+)g.*[Pp]rotein', output)
            calorie_match = re.search(r'(\d+)\s*[Cc]alories', output)
            cost_match = re.search(r'₹(\d+)', output)
            
            protein = int(protein_match.group(1)) if protein_match else 0
            calories = int(calorie_match.group(1)) if calorie_match else 0
            cost = int(cost_match.group(1)) if cost_match else 0
            
            total_protein_estimate += protein
            total_calorie_estimate += calories
            total_supplement_cost += cost
            
            print(f"\n{meal_time.upper()}:")
            print(f"  Estimated protein: {protein}g")
            print(f"  Estimated calories: {calories}")
            print(f"  Supplement cost: ₹{cost}")
        
        print("\n" + "="*60)
        print("📈 DAILY TOTALS:")
        print("="*60)
        print(f"Total protein: {total_protein_estimate}g")
        print(f"Total calories: {total_calorie_estimate}")
        print(f"Total supplement cost: ₹{total_supplement_cost}")
        print(f"Budget remaining: ₹{max(0, 500 - total_supplement_cost)}")
        
        # Save summary
        save = self.safe_input("\n💾 Save daily summary? (y/n): ")
        if save.lower() == 'y':
            self._save_daily_summary(results, username, total_protein_estimate, total_calorie_estimate, total_supplement_cost)
    
    def _save_daily_summary(self, results: List[Dict], username: str, total_protein: int, total_calories: int, total_cost: int):
        """Save daily optimization summary"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{username}_daily_mess_summary_{timestamp}.txt"
        filepath = Config.DATA_DIR / filename
        
        content = f"""
DAILY MESS OPTIMIZATION SUMMARY
{'='*60}
User: {username}
Date: {datetime.now().strftime('%Y-%m-%d')}
Total Meals Optimized: {len(results)}
{'='*60}

DAILY TOTALS:
- Total Protein: {total_protein}g
- Total Calories: {total_calories}
- Total Supplement Cost: ₹{total_cost}
- Budget Remaining: ₹{max(0, 500 - total_cost)}

{'='*60}
MEAL-BY-MEAL ANALYSIS:
{'='*60}
"""
        
        for result in results:
            meal_time = result["meal_time"]
            menu_items = result["menu_items"]
            
            content += f"\n{meal_time.upper()}:\n"
            content += f"Menu: {', '.join(menu_items)}\n"
            content += f"Key Recommendations from analysis\n"
            content += "-" * 30 + "\n"
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Daily summary saved to: {filepath}")
        except Exception as e:
            print(f"❌ Save failed: {e}")
    
    def daily_checkin(self):
        """Comprehensive daily health check-in"""
        if not self.current_user_id:
            print("Please setup profile first!")
            return
        
        profile = self.db.get_profile(self.current_user_id)
        if not profile:
            print("Profile not found!")
            return
        
        print("\n" + "="*60)
        print(f"📝 DAILY CHECK-IN")
        print("="*60)
        
        today = datetime.now().strftime("%Y-%m-%d")
        date_input = self.safe_input(f"Date (YYYY-MM-DD) [Today: {today}]: ")
        if not date_input:
            date_input = today
        
        try:
            date_obj = datetime.strptime(date_input, "%Y-%m-%d")
            day_of_week = date_obj.strftime("%A")
        except:
            print("Invalid date format. Using today's date.")
            date_input = today
            day_of_week = datetime.now().strftime("%A")
        
        existing = self.db.get_log_by_date(self.current_user_id, date_input)
        if existing:
            overwrite = self.safe_input(f"Log exists for {date_input}. Overwrite? (y/n): ")
            if overwrite.lower() != 'y':
                return
        
        log_id = f"log_{date_input}_{self.current_user_id}"
        
        print(f"\n📅 Recording for {date_input} ({day_of_week})")
        print("-"*30)
        
        print("\n😴 SLEEP")
        sleep_hours = self._get_float("Sleep hours: ", 7.0, 0, 24)
        sleep_quality = self._get_int("Sleep quality (1-10): ", 7, 1, 10)
        sleep_notes = self.safe_input("Sleep notes: ")
        
        print("\n🍽️ MEALS")
        meals = {}
        for meal in ["breakfast", "lunch", "dinner"]:
            ate = self.safe_input(f"  {meal.title()}? (y/n): ")
            if ate.lower() == 'y':
                food = self.safe_input(f"    What: ")
                meals[meal] = food
            else:
                meals[meal] = "skipped"
        
        water = self._get_float("Water (liters): ", 2.0, 0, 10)
        
        print("\n🏃 ACTIVITY")
        steps = self._get_int("Steps: ", 5000, 0, 50000)
        
        worked_out = self.safe_input("Workout today? (y/n): ")
        workout_duration = 0
        workout_type = ""
        if worked_out.lower() == 'y':
            workout_duration = self._get_int("Duration (min): ", 30, 1, 240)
            workout_type = self.safe_input("Type (e.g., cardio, strength, yoga): ")
        
        print("\n📊 METRICS (1-10)")
        energy = self._get_int("Energy: ", 5, 1, 10)
        mood = self._get_int("Mood: ", 5, 1, 10)
        stress = self._get_int("Stress: ", 5, 1, 10)
        focus = self._get_int("Focus: ", 5, 1, 10)
        
        symptoms_input = self.safe_input("\nSymptoms (comma separated): ")
        symptoms = [s.strip() for s in symptoms_input.split(",")] if symptoms_input else []
        
        notes = self.safe_input("Notes: ")
        
        log = DailyLog(
            log_id=log_id,
            user_id=self.current_user_id,
            date=date_input,
            day_of_week=day_of_week,
            timestamp=datetime.now().isoformat(),
            sleep_hours=sleep_hours,
            sleep_quality=sleep_quality,
            sleep_notes=sleep_notes,
            meals=meals,
            water_intake_liters=water,
            steps_count=steps,
            workout_duration_min=workout_duration,
            workout_type=workout_type,
            energy_level=energy,
            mood_level=mood,
            stress_level=stress,
            focus_level=focus,
            symptoms=symptoms,
            notes=notes
        )
        
        if self.db.save_log(log):
            print(f"\n✅ Log saved!")
            print(f"   Sleep: {sleep_hours}h ({sleep_quality}/10)")
            print(f"   Meals: {log.meals_eaten}/3")
            print(f"   Energy: {energy}/10")
            print(f"   Steps: {steps:,}")
            if workout_duration > 0:
                print(f"   Workout: {workout_duration}min {workout_type}")
        else:
            print("❌ Failed to save log")
    
    def _get_int(self, prompt: str, default: int, min_val: int, max_val: int) -> int:
        """Get integer input"""
        while True:
            try:
                value = self.safe_input(prompt)
                if not value:
                    return default
                value = int(value)
                if min_val <= value <= max_val:
                    return value
                print(f"Enter {min_val}-{max_val}")
            except:
                print("Enter a number")
    
    def _get_float(self, prompt: str, default: float, min_val: float, max_val: float) -> float:
        """Get float input"""
        while True:
            try:
                value = self.safe_input(prompt)
                if not value:
                    return default
                value = float(value)
                if min_val <= value <= max_val:
                    return value
                print(f"Enter {min_val}-{max_val}")
            except:
                print("Enter a number")
    
    def run_analysis(self, analysis_type: str) -> Dict:
        """Run comprehensive analysis - returns results for web mode"""
        if not self.current_user_id:
            if self.execution_mode == "web":
                return {"success": False, "error": "Profile not set up"}
            print("Please setup profile first!")
            return None
        
        profile = self.db.get_profile(self.current_user_id)
        logs = self.db.get_user_logs(self.current_user_id, days=14)
        if not logs:
            if self.execution_mode == "web":
                return {"success": False, "error": "No health data available"}
            print("No health data. Complete a check-in first.")
            return None
        
        profile_dict = profile.to_dict()
        logs_dict = [log.to_dict() for log in logs]
        
        if self.execution_mode == "cli":
            print(f"\n{'='*60}")
            print(f"🤖 {analysis_type.upper()} ANALYSIS")
            print("="*60)
            print(f"User: {profile.name}")
            print(f"Data: {len(logs)} days")
            print("-"*60)
            print("🚀 Starting TAO agent...")
            print("Agent will decide optimal number of reasoning loops")
            print("-"*60)
        
        agent = self.agents.get(analysis_type)
        if not agent:
            if self.execution_mode == "web":
                return {"success": False, "error": f"Unknown analysis type: {analysis_type}"}
            print(f"❌ Unknown analysis type")
            return None
        
        result = agent.run(profile_dict, logs_dict)
        
        if self.execution_mode == "cli":
            if result["success"]:
                print(f"\n✅ ANALYSIS COMPLETE")
                print("="*60)
                print(f"Iterations: {result['iterations']}/{result['max_iterations']}")
                print(f"Confidence: {result['confidence']:.2f}")
                print("="*60)
                
                output = result["output"]
                if output and output != "No detailed analysis generated. Please try again with more data.":
                    lines = output.split('\n')
                    
                    if len(lines) > 50:
                        print('\n'.join(lines[:50]))
                        print("\n" + "="*60)
                        print("📄 OUTPUT CONTINUES...")
                        print("="*60)
                        print('\n'.join(lines[50:100]) if len(lines) > 100 else '\n'.join(lines[50:]))
                        
                        if len(lines) > 100:
                            print(f"\n... and {len(lines) - 100} more lines")
                    else:
                        print(output)
                    
                    save = self.safe_input("\n💾 Save full analysis to file? (y/n): ")
                    if save.lower() == 'y':
                        self._save_result(result, profile.name)
                else:
                    print("⚠️  No detailed analysis generated.")
                    print("Please try with more data.")
            else:
                print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
        
        return result
    
    def _save_result(self, result: Dict, username: str):
        """Save result to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{username}_{result['agent_type']}_{timestamp}.txt"
        filepath = Config.DATA_DIR / filename
        
        content = f"""
HEALTH AI ANALYSIS REPORT
{'='*60}
User: {username}
Type: {result['agent_type'].title()}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Iterations: {result['iterations']}/{result['max_iterations']}
Confidence: {result['confidence']:.2f}
{'='*60}

{result['output']}

{'='*60}
AGENT REASONING:
{'='*60}
"""
        
        for i, step in enumerate(result.get('reasoning_summary', []), 1):
            content += f"\n{i}. [{step.get('phase', 'unknown').upper()}] (Iteration {step.get('iteration', 0)}):\n"
            content += f"   {step.get('insight', '')}\n"
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Analysis saved to: {filepath}")
        except Exception as e:
            print(f"❌ Save failed: {e}")
    
    def show_dashboard(self):
        """Show health dashboard"""
        if not self.current_user_id:
            print("Please setup profile first!")
            return
        
        profile = self.db.get_profile(self.current_user_id)
        logs = self.db.get_user_logs(self.current_user_id, days=7)
        
        print("\n" + "="*60)
        print(f"📊 DASHBOARD - {profile.name}")
        print("="*60)
        
        print(f"\n👤 PROFILE")
        print("-"*30)
        print(f"Name: {profile.name}")
        print(f"Age: {profile.age} | Weight: {profile.weight}kg")
        print(f"Height: {profile.height}cm | BMI: {profile.bmi} ({profile.bmi_category})")
        print(f"Diet: {profile.diet_type.value} | Goal: {profile.fitness_goal}")
        
        if logs:
            print(f"\n📈 RECENT ({len(logs)} days)")
            print("-"*30)
            
            for log in logs[:5]:
                print(f"{log.date} ({log.day_of_week[:3]}):")
                print(f"  Sleep: {log.sleep_hours}h ({log.sleep_quality}/10)")
                print(f"  Meals: {log.meals_eaten}/3 | Energy: {log.energy_level}/10")
                if log.steps_count > 0:
                    print(f"  Steps: {log.steps_count:,}")
                if log.workout_duration_min > 0:
                    print(f"  Workout: {log.workout_duration_min}min")
                print()
        else:
            print("\n📊 No data yet. Complete a check-in!")
    
    def run(self):
        """Main application loop - CLI only"""
        if self.execution_mode == "web":
            print("Web mode: Use Streamlit interface")
            return
        
        print("\n" + "="*60)
        print("🏥 HEALTH AI - PRODUCTION SYSTEM")
        print("="*60)
        print(f"Session: {Config.SESSION_ID}")
        print("="*60)
        
        if not self.setup_profile():
            return
        
        while True:
            profile = self.db.get_profile(self.current_user_id)
            
            print("\n" + "="*60)
            print(f"🏥 MAIN MENU - {profile.name}")
            print("="*60)
            print(f"Diet: {profile.diet_type.value} | BMI: {profile.bmi}")
            print("="*60)
            
            print("\n1. 📝 Daily Check-in")
            print("2. 🍽️ Diet Plan Analysis")
            print("3. 💪 Fitness Plan Analysis")
            print("4. ⚕️ Health Risk Analysis")
            
            if profile.has_mess_access:
                print("5. 📋 Upload Mess Menu")
                print("6. 🤖 Run Mess Optimizer")
                print("7. 📊 View Dashboard")
                print("0. ❌ Exit")
            else:
                print("5. 📊 View Dashboard")
                print("0. ❌ Exit")
            
            print("="*60)
            
            choice = self.safe_input("\nSelect: ")
            
            if choice == "0":
                print("\n👋 Thank you! Stay healthy!")
                break
            elif choice == "1":
                self.daily_checkin()
            elif choice == "2":
                self.run_analysis("diet")
            elif choice == "3":
                self.run_analysis("fitness")
            elif choice == "4":
                self.run_analysis("health")
            elif choice == "5" and profile.has_mess_access:
                self.upload_mess_menu()
            elif choice == "6" and profile.has_mess_access:
                self.run_mess_optimizer()
            elif (choice == "5" and not profile.has_mess_access) or (choice == "7" and profile.has_mess_access):
                self.show_dashboard()
            else:
                print("❌ Invalid option")
            
            if choice != "0":
                input("\nPress Enter to continue...")

# ---------------- MAIN ------------------
def main():
    """Entry point"""
    try:
        system = HealthAISystem(execution_mode="cli")
        system.run()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ System error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
