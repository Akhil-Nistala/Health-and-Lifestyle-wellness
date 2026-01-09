import streamlit as st
import os
import json
from datetime import datetime, date
from typing import Dict, List, Any, Optional
from pathlib import Path
import uuid
import pandas as pd
from agent_bridge import AgentBridge

# ========== CONFIGURATION ==========
class Config:
    MODEL = "llama-3.1-8b-instant"
    DATA_DIR = Path("health_ai_web_data")
    SESSION_ID = str(uuid.uuid4())[:8]

# Ensure data directory exists
Config.DATA_DIR.mkdir(exist_ok=True)

# ========== HELPER FUNCTION FOR AGENT METADATA ==========
def display_agent_metadata(metadata: Dict, agent_type: str):
    """Display agent execution metadata with defensive checks"""
    if not metadata:
        return
    
    with st.expander(f"🤖 Agent Execution Details ({agent_type})"):
        col1, col2 = st.columns(2)
        with col1:
            iterations = metadata.get('iterations', 0)
            max_iterations = metadata.get('max_iterations', 3)
            if iterations is not None and max_iterations is not None:
                st.metric("TAO Iterations", 
                         f"{iterations}/{max_iterations}")
        
        with col2:
            confidence = metadata.get('confidence', 0.5)
            if confidence is not None:
                st.metric("Confidence Score", f"{confidence:.2f}")
        
        # Show reasoning trace if available
        reasoning_trace = metadata.get('reasoning_trace', [])
        if reasoning_trace:
            st.subheader("Reasoning Trace")
            for i, step in enumerate(reasoning_trace, 1):
                if isinstance(step, dict):
                    with st.expander(f"Step {i}: {step.get('phase', 'unknown').title()}"):
                        st.write(f"**Iteration:** {step.get('iteration', 'N/A')}")
                        insight = step.get('insight', '')
                        if insight:
                            st.write(f"**Insight:** {insight}")

# ========== DATA MODELS ==========
class DietType:
    VEGETARIAN = "vegetarian"
    NON_VEGETARIAN = "non_vegetarian"
    EGGETARIAN = "eggetarian"
    VEGAN = "vegan"

def get_diet_type_enum(value: str) -> str:
    """Safely convert string to DietType"""
    value_lower = value.lower()
    if value_lower in ["vegetarian", "non-vegetarian", "eggetarian", "vegan"]:
        return value_lower
    return "non_vegetarian"

class UserProfile:
    def __init__(
        self,
        user_id: str,
        name: str,
        age: int,
        weight: float,
        height: float,
        diet_type: str,
        fitness_goal: str,
        activity_level: str,
        is_student: bool = False,
        has_mess_access: bool = False,
        weekly_budget: float = 500.0,
        medical_conditions: List[str] = None,
        allergies: List[str] = None,
        created_at: str = None
    ):
        self.user_id = user_id
        self.name = name
        self.age = age
        self.weight = weight
        self.height = height
        self.diet_type = get_diet_type_enum(diet_type)
        self.fitness_goal = fitness_goal
        self.activity_level = activity_level
        self.is_student = is_student
        self.has_mess_access = has_mess_access
        self.weekly_budget = weekly_budget
        self.medical_conditions = medical_conditions or []
        self.allergies = allergies or []
        self.created_at = created_at or datetime.now().isoformat()
    
    @property
    def bmi(self) -> float:
        if self.height == 0:
            return 0.0
        height_m = self.height / 100
        return round(self.weight / (height_m ** 2), 1)
    
    @property
    def bmi_category(self) -> str:
        bmi = self.bmi
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Healthy"
        elif bmi < 30:
            return "Overweight"
        else:
            return "Obese"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'name': self.name,
            'age': self.age,
            'weight': self.weight,
            'height': self.height,
            'diet_type': self.diet_type,
            'fitness_goal': self.fitness_goal,
            'activity_level': self.activity_level,
            'is_student': self.is_student,
            'has_mess_access': self.has_mess_access,
            'weekly_budget': self.weekly_budget,
            'medical_conditions': self.medical_conditions,
            'allergies': self.allergies,
            'created_at': self.created_at,
            'bmi': self.bmi,
            'bmi_category': self.bmi_category
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(
            user_id=data['user_id'],
            name=data['name'],
            age=data['age'],
            weight=data['weight'],
            height=data['height'],
            diet_type=data['diet_type'],
            fitness_goal=data['fitness_goal'],
            activity_level=data['activity_level'],
            is_student=data.get('is_student', False),
            has_mess_access=data.get('has_mess_access', False),
            weekly_budget=data.get('weekly_budget', 500.0),
            medical_conditions=data.get('medical_conditions', []),
            allergies=data.get('allergies', []),
            created_at=data.get('created_at')
        )

# ========== DATABASE MANAGER ==========
class DatabaseManager:
    def __init__(self):
        self.profiles_file = Config.DATA_DIR / "profiles.json"
        self.logs_file = Config.DATA_DIR / "logs.json"
        self.mess_menus_file = Config.DATA_DIR / "mess_menus.json"
        self._load_data()
    
    def _load_data(self):
        """Load all data from JSON files"""
        try:
            # Load profiles
            if self.profiles_file.exists():
                with open(self.profiles_file, 'r', encoding='utf-8') as f:
                    profiles_data = json.load(f)
                    self.profiles = {user_id: UserProfile.from_dict(data) 
                                   for user_id, data in profiles_data.items()}
            else:
                self.profiles = {}
            
            # Load logs
            if self.logs_file.exists():
                with open(self.logs_file, 'r', encoding='utf-8') as f:
                    self.logs = json.load(f)
            else:
                self.logs = {}
            
            # Load mess menus
            if self.mess_menus_file.exists():
                with open(self.mess_menus_file, 'r', encoding='utf-8') as f:
                    self.mess_menus = json.load(f)
            else:
                self.mess_menus = {}
                
        except Exception as e:
            print(f"Error loading data: {str(e)[:200]}")
            self.profiles = {}
            self.logs = {}
            self.mess_menus = {}
    
    def _save_profiles(self):
        """Save profiles to file"""
        try:
            profiles_data = {user_id: profile.to_dict() 
                           for user_id, profile in self.profiles.items()}
            with open(self.profiles_file, 'w', encoding='utf-8') as f:
                json.dump(profiles_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving profiles: {str(e)[:200]}")
    
    def _save_logs(self):
        """Save logs to file"""
        try:
            with open(self.logs_file, 'w', encoding='utf-8') as f:
                json.dump(self.logs, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving logs: {str(e)[:200]}")
    
    def _save_mess_menus(self):
        """Save mess menus to file"""
        try:
            with open(self.mess_menus_file, 'w', encoding='utf-8') as f:
                json.dump(self.mess_menus, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving mess menus: {str(e)[:200]}")
    
    def save_profile(self, profile: UserProfile) -> bool:
        """Save or update user profile"""
        try:
            self.profiles[profile.user_id] = profile
            self._save_profiles()
            return True
        except Exception as e:
            print(f"Error saving profile: {str(e)[:200]}")
            return False
    
    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by ID"""
        return self.profiles.get(user_id)
    
    def save_log(self, user_id: str, log_data: Dict[str, Any]) -> bool:
        """Save daily log"""
        try:
            if user_id not in self.logs:
                self.logs[user_id] = []
            
            # Add log date and timestamp
            log_data['log_date'] = log_data.get('date', datetime.now().strftime('%Y-%m-%d'))
            log_data['timestamp'] = datetime.now().isoformat()
            
            self.logs[user_id].append(log_data)
            self._save_logs()
            return True
        except Exception as e:
            print(f"Error saving log: {str(e)[:200]}")
            return False
    
    def get_user_logs(self, user_id: str, days: int = 30) -> List[Dict]:
        """Get user's recent logs"""
        user_logs = self.logs.get(user_id, [])
        # Return last N logs
        return user_logs[-days:] if user_logs else []
    
    def save_mess_menu(self, user_id: str, date_str: str, menu_data: Dict[str, Any]) -> bool:
        """Save mess menu for a specific date"""
        try:
            if user_id not in self.mess_menus:
                self.mess_menus[user_id] = {}
            
            menu_data['date'] = date_str
            menu_data['timestamp'] = datetime.now().isoformat()
            self.mess_menus[user_id][date_str] = menu_data
            self._save_mess_menus()
            return True
        except Exception as e:
            print(f"Error saving mess menu: {str(e)[:200]}")
            return False
    
    def get_mess_menu(self, user_id: str, date_str: str) -> Optional[Dict]:
        """Get mess menu for a specific date"""
        user_menus = self.mess_menus.get(user_id, {})
        return user_menus.get(date_str)

# ========== LLM SERVICE ==========
class LLMService:
    def __init__(self):
        # Try multiple ways to get the API key
        self.api_key = None
        
        # Method 1: Try to get from streamlit secrets (only works in Streamlit Cloud)
        try:
            import streamlit as st
            self.api_key = st.secrets.get("GROQ_API_KEY")
        except:
            pass
        
        # Method 2: Try environment variable
        if not self.api_key:
            self.api_key = os.getenv("GROQ_API_KEY")
        
        # Method 3: Prompt user to enter it (but only if not already in session)
        if not self.api_key and 'api_key_entered' not in st.session_state:
            with st.sidebar.expander("🔑 API Key Setup", expanded=False):
                st.warning("GROQ_API_KEY not found. Please enter it below:")
                self.api_key = st.text_input(
                    "Enter your Groq API Key (get it from https://console.groq.com/keys):",
                    type="password",
                    key="api_key_input"
                )
                if self.api_key:
                    st.session_state.api_key_entered = True
                    st.success("API Key saved for this session!")
                    st.rerun()
        
        if not self.api_key:
            # Don't show error immediately, allow user to enter in sidebar
            self.llm = None
            return
        
        self.llm = self._init_llm()
    
    def _init_llm(self):
        """Initialize LLM with error handling"""
        try:
            # Try to import langchain_groq
            from langchain_groq import ChatGroq
            return ChatGroq(
                groq_api_key=self.api_key,
                model=Config.MODEL,
                temperature=0.7,
                max_tokens=2000
            )
        except ImportError:
            st.warning("""
            ⚠️ langchain-groq not installed!
            
            Install it with: `pip install langchain-groq`
            
            For now, using mock responses for demonstration.
            """)
            return MockLLM()
        except Exception as e:
            st.warning(f"⚠️ Error initializing LLM: {str(e)[:200]}")
            return MockLLM()
    
    def call_llm(self, prompt: str) -> str:
        """Call LLM with error handling"""
        if not self.llm:
            return "⚠️ LLM service not available. Please install langchain-groq and provide a valid GROQ_API_KEY in the sidebar."
        
        try:
            if isinstance(self.llm, MockLLM):
                return self.llm.call_llm(prompt)
            
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt[:4000])])
            return response.content.strip()
        except Exception as e:
            return f"❌ Error generating response: {str(e)[:200]}"

class MockLLM:
    """Mock LLM for demonstration when real LLM is not available"""
    def call_llm(self, prompt: str) -> str:
        """Generate a mock response based on the prompt"""
        if "diet plan" in prompt.lower():
            return self._generate_diet_plan(prompt)
        elif "workout" in prompt.lower():
            return self._generate_workout_plan(prompt)
        elif "health" in prompt.lower() and "analysis" in prompt.lower():
            return self._generate_health_analysis(prompt)
        elif "mess" in prompt.lower() and "optim" in prompt.lower():
            return self._generate_mess_optimization(prompt)
        else:
            return self._generate_general_response(prompt)
    
    def _generate_diet_plan(self, prompt: str) -> str:
        return """📋 PERSONALIZED 7-DAY DIET PLAN

DAY 1:
🥞 Breakfast (8 AM): 2 idlis with sambar + 1 cup milk + 1 banana (350 cal)
🍛 Lunch (1 PM): 2 rotis + 1 cup dal + 1 cup mixed vegetable curry + salad (550 cal)
🍽️ Dinner (7 PM): 1 cup brown rice + 1 cup chicken curry (or paneer) + salad (600 cal)
📊 Daily Targets: 2,200 cal | Protein: 110g | Carbs: 275g | Fats: 73g

DAY 2:
🥣 Breakfast: Oats with milk + nuts + fruits (400 cal)
🍲 Lunch: Rice + sambar + vegetable curry + curd (600 cal)
🍛 Dinner: Roti + paneer bhurji + salad (500 cal)
📊 Daily Targets: 2,200 cal | Protein: 115g | Carbs: 270g | Fats: 70g

DAY 3-7: Similar rotation with variety...

HYDRATION: 3-4 liters daily
SUPPLEMENTS: Multivitamin, Vitamin D"""

    def _generate_workout_plan(self, prompt: str) -> str:
        return """💪 4-WEEK PERSONALIZED WORKOUT PLAN

WEEK 1 SCHEDULE:
Monday (Upper Body): Push-ups 3x12, Rows 3x12, Shoulder Press 3x10
Tuesday (Cardio + Core): Jumping jacks 3x30s, Plank 3x30s, Bicycle crunches 3x20
Wednesday (Lower Body): Squats 3x15, Lunges 3x12, Glute bridges 3x15
Thursday (Active Recovery): Yoga/stretching
Friday (Full Body HIIT): Circuit training
Saturday (Cardio): 30-min brisk walk
Sunday: Rest

PROGRESSION: Increase reps by 10% each week!
SAFETY: Always warm up, maintain proper form."""

    def _generate_health_analysis(self, prompt: str) -> str:
        return """⚕️ HEALTH & WELLNESS ANALYSIS

OVERALL ASSESSMENT: Good health with room for improvement
BMI: 22.5 (Healthy range)

SLEEP ANALYSIS:
- Current: 7 hours nightly
- Target: 7-8 hours
- Quality: 7/10
- Recommendation: Consistent bedtime routine

NUTRITION ASSESSMENT:
✅ Strengths: Good variety
⚠️ Areas: Increase protein by 20g daily
💡 Action: Add protein source to each meal

ACTIVITY EVALUATION:
- Current: Moderate (3-4 days/week)
- Goal: Build muscle
- Recommendation: 4 strength + 2 cardio sessions weekly

ACTIONABLE RECOMMENDATIONS:
1. Add 30g protein to breakfast
2. Walk 10,000 steps daily
3. Sleep by 11 PM consistently

⚠️ Disclaimer: Not medical advice. Consult healthcare provider."""

    def _generate_mess_optimization(self, prompt: str) -> str:
        return """🍽️ MESS MEAL OPTIMIZATION PLAN

BREAKFAST OPTIMIZATION:
✅ Choose: Idli (2) + Sambar + Boiled egg
❌ Avoid: Fried items (puri/bhature)
Portion: Medium sambar, 2 idlis
Nutrition: 350 cal, 15g protein

LUNCH OPTIMIZATION:
✅ Choose: Rice + Dal + Mixed veg + Salad
✅ Add: Extra dal or chicken curry
Portion: 50% veggies, 25% protein, 25% carbs
Nutrition: 550 cal, 25g protein

DINNER OPTIMIZATION:
✅ Choose: Roti + Paneer/Chicken curry + Salad
❌ Limit: Rice at dinner
Portion: 2 rotis, 1 bowl curry
Nutrition: 450 cal, 20g protein

DAILY TOTALS:
Calories: ~1,350 | Protein: ~60g
Supplement: Protein powder post-workout

COST ANALYSIS:
Mess food: Included
Supplements: ₹500/month
Value: Excellent (balanced nutrition)"""

    def _generate_general_response(self, prompt: str) -> str:
        return f"""🤖 AI RESPONSE DEMONSTRATION

This is a mock response. For real AI:
1. Get Groq API key: https://console.groq.com/keys
2. Install: pip install langchain-groq
3. Add API key in sidebar

General health tips:
1. Stay hydrated (8-10 glasses daily)
2. Include protein in every meal
3. Get 7-8 hours quality sleep
4. Move for 30+ minutes daily
5. Eat colorful vegetables

Consistency is key!"""

# ========== HELPER FUNCTIONS ==========
def calculate_sleep_score(hours: float, quality: int) -> float:
    base_score = min(hours / 8 * 70, 70)
    quality_score = (quality / 10) * 30
    return round(base_score + quality_score, 1)

def calculate_ideal_weight(height_cm: float) -> str:
    height_m = height_cm / 100
    min_weight = round(18.5 * (height_m ** 2), 1)
    max_weight = round(24.9 * (height_m ** 2), 1)
    return f"{min_weight} - {max_weight}"

def calculate_calories(profile: UserProfile) -> int:
    """Calculate daily calorie needs"""
    if profile.activity_level == "sedentary":
        multiplier = 1.2
    elif profile.activity_level == "light":
        multiplier = 1.375
    elif profile.activity_level == "moderate":
        multiplier = 1.55
    elif profile.activity_level == "active":
        multiplier = 1.725
    else:  # very_active
        multiplier = 1.9
    
    # Mifflin-St Jeor Equation (simplified)
    bmr = 10 * profile.weight + 6.25 * profile.height - 5 * profile.age + 5
    return round(bmr * multiplier)

def calculate_protein(profile: UserProfile) -> int:
    """Calculate daily protein needs"""
    if "build muscle" in profile.fitness_goal.lower():
        return round(profile.weight * 1.6)
    elif "lose weight" in profile.fitness_goal.lower():
        return round(profile.weight * 1.2)
    else:
        return round(profile.weight * 0.8)

# ========== STREAMLIT APP ==========
def main():
    # Page configuration
    st.set_page_config(
        page_title="Health AI Assistant",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        border-left: 5px solid #1E88E5;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-color: #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .profile-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 4px solid #1E88E5;
    }
    .stButton > button {
        width: 100%;
    }
    .agent-output {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
        max-height: 600px;
        overflow-y: auto;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize services in session state
    if 'db' not in st.session_state:
        st.session_state.db = DatabaseManager()
    
    # Initialize LLM service
    if 'llm' not in st.session_state:
        st.session_state.llm = LLMService()
    
    # Initialize Agent Bridge - with web mode
    if 'agent_bridge' not in st.session_state:
        st.session_state.agent_bridge = AgentBridge(streamlit_db=st.session_state.db)
    
    # Initialize session states for storing AI responses
    if 'diet_plan_response' not in st.session_state:
        st.session_state.diet_plan_response = None
    if 'diet_plan_metadata' not in st.session_state:
        st.session_state.diet_plan_metadata = None
    if 'workout_plan_response' not in st.session_state:
        st.session_state.workout_plan_response = None
    if 'workout_plan_metadata' not in st.session_state:
        st.session_state.workout_plan_metadata = None
    if 'health_analysis_response' not in st.session_state:
        st.session_state.health_analysis_response = None
    if 'health_analysis_metadata' not in st.session_state:
        st.session_state.health_analysis_metadata = None
    if 'mess_optimization_response' not in st.session_state:
        st.session_state.mess_optimization_response = None
    if 'mess_optimization_metadata' not in st.session_state:
        st.session_state.mess_optimization_metadata = None
    
    # Initialize navigation state
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = "👤 Profile Setup"
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Health AI Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("### Navigation")
        
        # Check if user is logged in
        if 'user_id' not in st.session_state or not st.session_state.user_id:
            page_options = ["👤 Profile Setup", "ℹ️ About"]
        else:
            page_options = [
                "📊 Dashboard",
                "📝 Daily Check-in", 
                "🍽️ Diet Analysis",
                "💪 Fitness Analysis",
                "⚕️ Health Analysis",
                "📋 Mess Menu Upload",
                "🤖 Mess Optimizer",
                "👤 Profile Management"
            ]
        
        # Navigation selectbox
        selected_page = st.selectbox(
            "Choose a section:", 
            page_options,
            index=page_options.index(st.session_state.selected_page) if st.session_state.selected_page in page_options else 0
        )
        
        # Update session state when selection changes
        if selected_page != st.session_state.selected_page:
            st.session_state.selected_page = selected_page
            st.rerun()
        
        st.markdown("---")
        
        # User info if logged in
        if 'user_id' in st.session_state and st.session_state.user_id:
            profile = st.session_state.db.get_profile(st.session_state.user_id)
            if profile:
                st.markdown(f"**Logged in as:** {profile.name}")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("BMI", f"{profile.bmi}", profile.bmi_category)
                with col2:
                    st.metric("Weight", f"{profile.weight}kg")
                
                if st.button("🚪 Logout", use_container_width=True, type="secondary"):
                    for key in list(st.session_state.keys()):
                        if key not in ['db', 'llm', 'agent_bridge', 'selected_page']:
                            del st.session_state[key]
                    st.session_state.selected_page = "👤 Profile Setup"
                    st.rerun()
        
        st.markdown("---")
        st.markdown("### Agent System Status")
        
        # Check agent bridge availability
        agent_status = "🟢 Active" if st.session_state.agent_bridge.is_available() else "🔴 Inactive"
        st.write(f"TAO Agent: {agent_status}")
        
        if not st.session_state.agent_bridge.is_available():
            st.warning("Agent system not available. Using basic LLM.")
        
        st.markdown("---")
        st.markdown("### About")
        st.info("AI-powered health assistant with autonomous TAO agents for personalized fitness and nutrition guidance.")
    
    # Page routing
    if st.session_state.selected_page == "👤 Profile Setup":
        show_profile_setup()
    elif st.session_state.selected_page == "📊 Dashboard":
        show_dashboard()
    elif st.session_state.selected_page == "📝 Daily Check-in":
        show_daily_checkin()
    elif st.session_state.selected_page == "🍽️ Diet Analysis":
        show_diet_analysis()
    elif st.session_state.selected_page == "💪 Fitness Analysis":
        show_fitness_analysis()
    elif st.session_state.selected_page == "⚕️ Health Analysis":
        show_health_analysis()
    elif st.session_state.selected_page == "📋 Mess Menu Upload":
        show_mess_menu_upload()
    elif st.session_state.selected_page == "🤖 Mess Optimizer":
        show_mess_optimizer()
    elif st.session_state.selected_page == "👤 Profile Management":
        show_profile_management()
    elif st.session_state.selected_page == "ℹ️ About":
        show_about()

def show_profile_setup():
    """User profile creation page"""
    st.markdown("## 👤 Create Your Profile")
    
    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name", placeholder="John Doe")
            age = st.number_input("Age", min_value=10, max_value=100, value=25)
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
            height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
        
        with col2:
            diet_type = st.selectbox(
                "Diet Type",
                ["Vegetarian", "Non-Vegetarian", "Eggetarian", "Vegan"]
            )
            
            fitness_goal = st.selectbox(
                "Fitness Goal",
                ["Lose Weight", "Build Muscle", "Improve Endurance", "Maintain Health", "Gain Weight"]
            )
            
            activity_level = st.select_slider(
                "Activity Level",
                options=["Sedentary", "Light", "Moderate", "Active", "Very Active"]
            )
        
        st.markdown("### Additional Information")
        col3, col4 = st.columns(2)
        
        with col3:
            is_student = st.checkbox("I'm a hostel student")
            if is_student:
                has_mess_access = st.checkbox("I have mess access", value=True)
                weekly_budget = st.number_input("Weekly food budget (₹)", min_value=0, max_value=5000, value=500)
            else:
                has_mess_access = False
                weekly_budget = 500
        
        with col4:
            medical_input = st.text_input("Medical Conditions (comma separated)", placeholder="None")
            allergy_input = st.text_input("Food Allergies (comma separated)", placeholder="None")
        
        submitted = st.form_submit_button("Create Profile", use_container_width=True)
        
        if submitted:
            if not name or name.strip() == "":
                st.error("Please enter your name!")
                return
            
            # Process medical conditions and allergies
            medical_conditions = [c.strip() for c in medical_input.split(",") if c.strip()] if medical_input else []
            allergies = [a.strip() for a in allergy_input.split(",") if a.strip()] if allergy_input else []
            
            # Create user ID
            user_id = f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create profile object
            profile = UserProfile(
                user_id=user_id,
                name=name,
                age=int(age),
                weight=float(weight),
                height=float(height),
                diet_type=diet_type,
                fitness_goal=fitness_goal,
                activity_level=activity_level.lower(),
                is_student=is_student,
                has_mess_access=has_mess_access if is_student else False,
                weekly_budget=float(weekly_budget),
                medical_conditions=medical_conditions,
                allergies=allergies
            )
            
            # Save profile
            if st.session_state.db.save_profile(profile):
                st.session_state.user_id = user_id
                st.session_state.selected_page = "📊 Dashboard"
                st.success(f"✅ Profile created successfully for {name}!")
                
                # Show summary
                st.markdown(f"""
                <div class="success-box">
                <h4>Profile Summary:</h4>
                <p><strong>Name:</strong> {name}</p>
                <p><strong>Age:</strong> {age}</p>
                <p><strong>BMI:</strong> {profile.bmi} ({profile.bmi_category})</p>
                <p><strong>Diet:</strong> {diet_type}</p>
                <p><strong>Goal:</strong> {fitness_goal}</p>
                <p><strong>Activity Level:</strong> {activity_level}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.balloons()
                st.rerun()
            else:
                st.error("❌ Failed to save profile. Please try again.")

def show_dashboard():
    """Main dashboard page"""
    if 'user_id' not in st.session_state:
        st.warning("Please create a profile first!")
        st.session_state.selected_page = "👤 Profile Setup"
        st.rerun()
        return
    
    profile = st.session_state.db.get_profile(st.session_state.user_id)
    if not profile:
        st.error("Profile not found!")
        st.session_state.selected_page = "👤 Profile Setup"
        st.rerun()
        return
    
    st.markdown(f"## 📊 Dashboard - Welcome back, {profile.name}!")
    
    # Stats Cards in a grid
    st.markdown("### Health Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
        <h3 style="color: #1E88E5;">{profile.bmi}</h3>
        <p style="color: #666; font-size: 0.9rem;">BMI</p>
        <p style="color: {'#43A047' if profile.bmi_category == 'Healthy' else '#E53935'}; font-weight: bold;">
        {profile.bmi_category}
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        calories = calculate_calories(profile)
        st.markdown(f"""
        <div class="metric-card">
        <h3 style="color: #1E88E5;">{calories:,}</h3>
        <p style="color: #666; font-size: 0.9rem;">Daily Calories</p>
        <p style="color: #43A047; font-size: 0.8rem;">Recommended</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        protein = calculate_protein(profile)
        st.markdown(f"""
        <div class="metric-card">
        <h3 style="color: #1E88E5;">{protein}g</h3>
        <p style="color: #666; font-size: 0.9rem;">Daily Protein</p>
        <p style="color: #43A047; font-size: 0.8rem;">Target</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        ideal_weight = calculate_ideal_weight(profile.height)
        st.markdown(f"""
        <div class="metric-card">
        <h3 style="color: #1E88E5;">{profile.weight}</h3>
        <p style="color: #666; font-size: 0.9rem;">Current Weight</p>
        <p style="color: #666; font-size: 0.8rem;">Ideal: {ideal_weight}kg</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Profile Information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Profile Information")
        st.markdown(f"""
        <div class="card">
        <p><strong>👤 Name:</strong> {profile.name}</p>
        <p><strong>🎂 Age:</strong> {profile.age} years</p>
        <p><strong>🥗 Diet Type:</strong> {profile.diet_type.title()}</p>
        <p><strong>🎯 Fitness Goal:</strong> {profile.fitness_goal}</p>
        <p><strong>🏃 Activity Level:</strong> {profile.activity_level.title()}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Health Details")
        st.markdown(f"""
        <div class="card">
        <p><strong>💰 Weekly Budget:</strong> ₹{profile.weekly_budget}</p>
        <p><strong>🎓 Student:</strong> {'Yes' if profile.is_student else 'No'}</p>
        <p><strong>🏫 Mess Access:</strong> {'Yes' if profile.has_mess_access else 'No'}</p>
        <p><strong>🏥 Medical Conditions:</strong> {', '.join(profile.medical_conditions) if profile.medical_conditions else 'None'}</p>
        <p><strong>⚠️ Allergies:</strong> {', '.join(profile.allergies) if profile.allergies else 'None'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Recent Logs
    logs = st.session_state.db.get_user_logs(st.session_state.user_id, days=7)
    if logs:
        st.markdown("### Recent Activity (Last 7 Days)")
        
        # Create DataFrame for logs
        log_data = []
        for log in logs:
            log_date = log.get('log_date', log.get('date', 'Unknown'))
            sleep_hours = log.get('sleep_hours', 0)
            sleep_quality = log.get('sleep_quality', 0)
            energy = log.get('energy_level', 0)
            steps = log.get('steps_count', 0)
            meals = log.get('meals_eaten', 0)
            
            log_data.append({
                'Date': log_date,
                'Sleep (hrs)': sleep_hours,
                'Sleep Quality': sleep_quality,
                'Energy': energy,
                'Steps': steps,
                'Meals': meals
            })
        
        if log_data:
            df = pd.DataFrame(log_data)
            st.dataframe(df, use_container_width=True)
    
    # Quick Actions
    st.markdown("### Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📝 Daily Check-in", use_container_width=True):
            st.session_state.selected_page = "📝 Daily Check-in"
            st.rerun()
    
    with col2:
        if st.button("🍽️ Diet Plan", use_container_width=True):
            st.session_state.selected_page = "🍽️ Diet Analysis"
            st.rerun()
    
    with col3:
        if st.button("💪 Workout Plan", use_container_width=True):
            st.session_state.selected_page = "💪 Fitness Analysis"
            st.rerun()
    
    with col4:
        if profile.has_mess_access:
            if st.button("📋 Mess Menu", use_container_width=True):
                st.session_state.selected_page = "📋 Mess Menu Upload"
                st.rerun()
        else:
            st.button("📋 Mess Menu", disabled=True, use_container_width=True, 
                     help="Mess features require student status with mess access")

def show_daily_checkin():
    """Daily health check-in page"""
    if 'user_id' not in st.session_state:
        st.warning("Please create a profile first!")
        st.session_state.selected_page = "👤 Profile Setup"
        st.rerun()
        return
    
    st.markdown("## 📝 Daily Health Check-in")
    
    today = datetime.now().strftime("%Y-%m-%d")
    st.markdown(f"### Date: {today}")
    
    with st.form("daily_checkin_form"):
        # Sleep Section
        st.markdown("#### 😴 Sleep")
        col1, col2 = st.columns(2)
        with col1:
            sleep_hours = st.slider("Sleep Hours", 0.0, 24.0, 7.0, 0.5)
        with col2:
            sleep_quality = st.slider("Sleep Quality (1-10)", 1, 10, 7)
        sleep_notes = st.text_area("Sleep Notes", placeholder="Any sleep issues or observations...")
        
        # Meals Section
        st.markdown("#### 🍽️ Meals")
        col1, col2, col3 = st.columns(3)
        with col1:
            breakfast = st.text_input("Breakfast", placeholder="What did you eat?")
        with col2:
            lunch = st.text_input("Lunch", placeholder="What did you eat?")
        with col3:
            dinner = st.text_input("Dinner", placeholder="What did you eat?")
        
        water_intake = st.slider("Water Intake (liters)", 0.0, 10.0, 2.0, 0.5)
        
        # Activity Section
        st.markdown("#### 🏃 Activity")
        steps = st.number_input("Steps Today", 0, 50000, 5000)
        
        workout_today = st.checkbox("Did you workout today?")
        workout_duration = 0
        workout_type = ""
        if workout_today:
            col1, col2 = st.columns(2)
            with col1:
                workout_duration = st.number_input("Workout Duration (minutes)", 1, 240, 30)
            with col2:
                workout_type = st.text_input("Workout Type", placeholder="Cardio, Strength, Yoga...")
        
        # Metrics Section
        st.markdown("#### 📊 Health Metrics (1-10)")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            energy = st.slider("Energy Level", 1, 10, 5)
        with col2:
            mood = st.slider("Mood", 1, 10, 5)
        with col3:
            stress = st.slider("Stress Level", 1, 10, 5)
        with col4:
            focus = st.slider("Focus Level", 1, 10, 5)
        
        symptoms = st.text_input("Any symptoms? (comma separated)", placeholder="headache, fatigue, etc.")
        notes = st.text_area("Additional Notes", placeholder="Any other observations...")
        
        submitted = st.form_submit_button("Save Daily Log", use_container_width=True)
        
        if submitted:
            # Prepare log data
            meals_dict = {
                'breakfast': breakfast,
                'lunch': lunch,
                'dinner': dinner
            }
            meals_eaten = sum(1 for meal in meals_dict.values() if meal.strip())
            
            log_data = {
                'date': today,
                'sleep_hours': sleep_hours,
                'sleep_quality': sleep_quality,
                'sleep_notes': sleep_notes,
                'meals': meals_dict,
                'meals_eaten': meals_eaten,
                'water_intake_liters': water_intake,
                'steps_count': steps,
                'workout_duration_min': workout_duration if workout_today else 0,
                'workout_type': workout_type if workout_today else "",
                'energy_level': energy,
                'mood_level': mood,
                'stress_level': stress,
                'focus_level': focus,
                'symptoms': [s.strip() for s in symptoms.split(",") if s.strip()] if symptoms else [],
                'notes': notes
            }
            
            # Save log
            if st.session_state.db.save_log(st.session_state.user_id, log_data):
                st.success("✅ Daily check-in saved successfully!")
                
                # Show summary
                sleep_score = calculate_sleep_score(sleep_hours, sleep_quality)
                st.markdown(f"""
                <div class="success-box">
                <h4>Today's Summary:</h4>
                <p><strong>Sleep Score:</strong> {sleep_score}/100</p>
                <p><strong>Meals Eaten:</strong> {meals_eaten}/3</p>
                <p><strong>Water Intake:</strong> {water_intake}L</p>
                <p><strong>Steps:</strong> {steps:,}</p>
                <p><strong>Overall Energy:</strong> {energy}/10</p>
                {f'<p><strong>Workout:</strong> {workout_duration}min {workout_type}</p>' if workout_today else ''}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("❌ Failed to save log. Please try again.")

def show_diet_analysis():
    """Diet analysis page"""
    if 'user_id' not in st.session_state:
        st.warning("Please create a profile first!")
        st.session_state.selected_page = "👤 Profile Setup"
        st.rerun()
        return
    
    profile = st.session_state.db.get_profile(st.session_state.user_id)
    
    st.markdown("## 🍽️ Personalized Diet Analysis")
    
    # Profile summary
    with st.expander("Your Profile Info", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Name:** {profile.name}")
            st.write(f"**Age:** {profile.age}")
            st.write(f"**Weight:** {profile.weight} kg")
            st.write(f"**Height:** {profile.height} cm")
        with col2:
            st.write(f"**BMI:** {profile.bmi} ({profile.bmi_category})")
            st.write(f"**Diet:** {profile.diet_type.title()}")
            st.write(f"**Goal:** {profile.fitness_goal}")
            st.write(f"**Activity:** {profile.activity_level.title()}")
    
    # Only show form if we don't have a response yet
    if st.session_state.diet_plan_response is None:
        # Additional preferences
        st.markdown("### Additional Preferences")
        col1, col2 = st.columns(2)
        with col1:
            cuisine_pref = st.multiselect(
                "Preferred Cuisines",
                ["Indian", "Italian", "Chinese", "Mexican", "Mediterranean", "Other"],
                default=["Indian"]
            )
            cooking_time = st.select_slider(
                "Max Cooking Time",
                options=["15 min", "30 min", "45 min", "60 min", "90 min+"],
                value="30 min"
            )
        
        with col2:
            meal_prep = st.checkbox("Interested in meal prep", value=True)
            budget = st.number_input("Weekly Food Budget (₹)", min_value=0, value=int(profile.weekly_budget))
        
        # Generate diet plan button
        if st.button("Generate Personalized Diet Plan", type="primary", use_container_width=True):
            # Try to use agent system first
            agent_result = None
            agent_available = st.session_state.agent_bridge.is_available()
            
            if agent_available:
                with st.spinner("🤖 AI Agent is analyzing your profile and creating a personalized plan..."):
                    # Get profile and logs
                    profile_dict = profile.to_dict()
                    logs = st.session_state.db.get_user_logs(st.session_state.user_id, days=7)
                    logs_dict = logs if logs else []
                    
                    agent_result = st.session_state.agent_bridge.run_agent_analysis(
                        agent_type='diet',
                        profile=profile_dict,
                        logs=logs_dict
                    )
                
                if agent_result and agent_result.get('success'):
                    response = agent_result['output']
                    metadata = agent_result.get('metadata', {})
                else:
                    # Fall back to direct LLM call with warning
                    st.warning("⚠️ Agent analysis failed, using standard LLM...")
                    with st.spinner("🤖 AI is creating your personalized diet plan..."):
                        # Prepare fallback prompt
                        prompt = f"""
                        Create a detailed 7-day meal plan for:
                        
                        PERSONAL INFORMATION:
                        - Name: {profile.name}
                        - Age: {profile.age}
                        - Weight: {profile.weight} kg
                        - Height: {profile.height} cm
                        - BMI: {profile.bmi} ({profile.bmi_category})
                        - Diet Type: {profile.diet_type}
                        - Fitness Goal: {profile.fitness_goal}
                        - Activity Level: {profile.activity_level}
                        - Allergies: {', '.join(profile.allergies) if profile.allergies else 'None'}
                        
                        PREFERENCES:
                        - Preferred Cuisines: {', '.join(cuisine_pref)}
                        - Max Cooking Time: {cooking_time}
                        - Meal Prep: {'Yes' if meal_prep else 'No'}
                        - Weekly Budget: ₹{budget}
                        
                        Please provide a personalized diet plan with exactly 7 days, each day independent.
                        """
                        response = st.session_state.llm.call_llm(prompt)
                        metadata = {}
            else:
                # Use direct LLM call if agent not available
                with st.spinner("🤖 AI is creating your personalized diet plan..."):
                    prompt = f"""
                    Create a detailed 7-day meal plan for:
                    
                    PERSONAL INFORMATION:
                    - Name: {profile.name}
                    - Age: {profile.age}
                    - Weight: {profile.weight} kg
                    - Height: {profile.height} cm
                    - BMI: {profile.bmi} ({profile.bmi_category})
                    - Diet Type: {profile.diet_type}
                    - Fitness Goal: {profile.fitness_goal}
                    - Activity Level: {profile.activity_level}
                    - Allergies: {', '.join(profile.allergies) if profile.allergies else 'None'}
                    
                    PREFERENCES:
                    - Preferred Cuisines: {', '.join(cuisine_pref)}
                    - Max Cooking Time: {cooking_time}
                    - Meal Prep: {'Yes' if meal_prep else 'No'}
                    - Weekly Budget: ₹{budget}
                    
                    Please provide a personalized diet plan with exactly 7 days, each day independent.
                    """
                    response = st.session_state.llm.call_llm(prompt)
                    metadata = {}
            
            st.session_state.diet_plan_response = response
            st.session_state.diet_plan_metadata = metadata  # Store metadata for display
            st.rerun()
    
    # Display response if available
    if st.session_state.diet_plan_response:
        st.markdown("### 📋 Your Personalized Diet Plan")
        
        # Display in a scrollable container
        with st.container():
            st.markdown('<div class="agent-output">', unsafe_allow_html=True)
            st.markdown(st.session_state.diet_plan_response)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Display agent metadata if available
        if hasattr(st.session_state, 'diet_plan_metadata') and st.session_state.diet_plan_metadata:
            display_agent_metadata(st.session_state.diet_plan_metadata, "Diet Analysis")
        
        # Download and regenerate buttons
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Diet Plan",
                data=st.session_state.diet_plan_response,
                file_name=f"diet_plan_{profile.name}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col2:
            if st.button("🔄 Generate Alternative", use_container_width=True):
                st.session_state.diet_plan_response = None
                st.session_state.diet_plan_metadata = None
                st.rerun()
        
        # Clear response button
        if st.button("🧹 Clear Response", use_container_width=True):
            st.session_state.diet_plan_response = None
            st.session_state.diet_plan_metadata = None
            st.rerun()

def show_fitness_analysis():
    """Fitness analysis page"""
    if 'user_id' not in st.session_state:
        st.warning("Please create a profile first!")
        st.session_state.selected_page = "👤 Profile Setup"
        st.rerun()
        return
    
    profile = st.session_state.db.get_profile(st.session_state.user_id)
    
    st.markdown("## 💪 Fitness & Workout Analysis")
    
    # Only show form if we don't have a response yet
    if st.session_state.workout_plan_response is None:
        # Fitness assessment
        st.markdown("### Assess Your Fitness Level")
        
        with st.form("fitness_assessment_form"):
            col1, col2 = st.columns(2)
            with col1:
                pushups = st.number_input("Max Pushups", 0, 100, 20)
                plank_time = st.number_input("Plank Time (seconds)", 0, 600, 60)
            with col2:
                running_5k = st.number_input("5K Run Time (minutes)", 0, 120, 30)
                flexibility = st.select_slider(
                    "Flexibility",
                    options=["Poor", "Average", "Good", "Excellent"],
                    value="Average"
                )
            
            # Equipment availability
            st.markdown("### Equipment Available")
            col1, col2, col3 = st.columns(3)
            with col1:
                has_gym = st.checkbox("Gym Access", value=True)
            with col2:
                has_weights = st.checkbox("Dumbbells/Weights", value=False)
            with col3:
                has_yogamat = st.checkbox("Yoga Mat", value=True)
            
            submitted = st.form_submit_button("Generate Workout Plan", use_container_width=True)
            
            if submitted:
                # Try to use agent system first
                agent_result = None
                agent_available = st.session_state.agent_bridge.is_available()
                
                if agent_available:
                    with st.spinner("🤖 AI Agent is creating your personalized workout plan..."):
                        # Get profile and logs
                        profile_dict = profile.to_dict()
                        logs = st.session_state.db.get_user_logs(st.session_state.user_id, days=7)
                        logs_dict = logs if logs else []
                        
                        agent_result = st.session_state.agent_bridge.run_agent_analysis(
                            agent_type='fitness',
                            profile=profile_dict,
                            logs=logs_dict
                        )
                    
                    if agent_result and agent_result.get('success'):
                        response = agent_result['output']
                        metadata = agent_result.get('metadata', {})
                    else:
                        # Fall back to direct LLM call with warning
                        st.warning("⚠️ Agent analysis failed, using standard LLM...")
                        with st.spinner("🤖 Creating your personalized workout plan..."):
                            # Prepare fallback prompt
                            equipment = []
                            if has_gym:
                                equipment.append("Gym equipment")
                            if has_weights:
                                equipment.append("Dumbbells/weights")
                            if has_yogamat:
                                equipment.append("Yoga mat")
                            
                            prompt = f"""
                            Create a personalized workout plan for:
                            
                            USER PROFILE:
                            - Name: {profile.name}
                            - Age: {profile.age}
                            - Weight: {profile.weight} kg
                            - Height: {profile.height} cm
                            - Fitness Goal: {profile.fitness_goal}
                            - Activity Level: {profile.activity_level}
                            
                            CURRENT FITNESS LEVEL:
                            - Max Pushups: {pushups}
                            - Plank Time: {plank_time} seconds
                            - 5K Run Time: {running_5k} minutes
                            - Flexibility: {flexibility}
                            
                            EQUIPMENT AVAILABLE: {', '.join(equipment) if equipment else 'None (bodyweight only)'}
                            
                            Please provide a personalized workout plan.
                            """
                            response = st.session_state.llm.call_llm(prompt)
                            metadata = {}
                else:
                    # Use direct LLM call if agent not available
                    with st.spinner("🤖 Creating your personalized workout plan..."):
                        equipment = []
                        if has_gym:
                            equipment.append("Gym equipment")
                        if has_weights:
                            equipment.append("Dumbbells/weights")
                        if has_yogamat:
                            equipment.append("Yoga mat")
                        
                        prompt = f"""
                        Create a personalized workout plan for:
                        
                        USER PROFILE:
                        - Name: {profile.name}
                        - Age: {profile.age}
                        - Weight: {profile.weight} kg
                        - Height: {profile.height} cm
                        - Fitness Goal: {profile.fitness_goal}
                        - Activity Level: {profile.activity_level}
                        
                        CURRENT FITNESS LEVEL:
                        - Max Pushups: {pushups}
                        - Plank Time: {plank_time} seconds
                        - 5K Run Time: {running_5k} minutes
                        - Flexibility: {flexibility}
                        
                        EQUIPMENT AVAILABLE: {', '.join(equipment) if equipment else 'None (bodyweight only)'}
                        
                        Please provide a personalized workout plan.
                        """
                        response = st.session_state.llm.call_llm(prompt)
                        metadata = {}
                
                st.session_state.workout_plan_response = response
                st.session_state.workout_plan_metadata = metadata
                st.rerun()
    
    # Display response if available
    if st.session_state.workout_plan_response:
        st.markdown("### 🏋️ Your Personalized Workout Plan")
        
        # Display in a scrollable container
        with st.container():
            st.markdown('<div class="agent-output">', unsafe_allow_html=True)
            st.markdown(st.session_state.workout_plan_response)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Display agent metadata if available
        if hasattr(st.session_state, 'workout_plan_metadata') and st.session_state.workout_plan_metadata:
            display_agent_metadata(st.session_state.workout_plan_metadata, "Fitness Analysis")
        
        # Download and action buttons
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Workout Plan",
                data=st.session_state.workout_plan_response,
                file_name=f"workout_plan_{profile.name}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col2:
            if st.button("🔄 Generate Alternative", use_container_width=True):
                st.session_state.workout_plan_response = None
                st.session_state.workout_plan_metadata = None
                st.rerun()
        
        # Clear response button
        if st.button("🧹 Clear Response", use_container_width=True):
            st.session_state.workout_plan_response = None
            st.session_state.workout_plan_metadata = None
            st.rerun()

def show_health_analysis():
    """Health analysis page"""
    if 'user_id' not in st.session_state:
        st.warning("Please create a profile first!")
        st.session_state.selected_page = "👤 Profile Setup"
        st.rerun()
        return
    
    profile = st.session_state.db.get_profile(st.session_state.user_id)
    
    st.markdown("## ⚕️ Health & Wellness Analysis")
    
    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ IMPORTANT DISCLAIMER:</strong><br>
    This analysis provides general health guidance only.<br>
    It is NOT a substitute for professional medical advice, diagnosis, or treatment.<br>
    Always consult a qualified healthcare provider for medical concerns.
    </div>
    """, unsafe_allow_html=True)
    
    # Only show generate button if we don't have a response yet
    if st.session_state.health_analysis_response is None:
        if st.button("Generate Comprehensive Health Analysis", type="primary", use_container_width=True):
            # Get recent logs for context
            logs = st.session_state.db.get_user_logs(st.session_state.user_id, days=7)
            
            # Try to use agent system first
            agent_result = None
            agent_available = st.session_state.agent_bridge.is_available()
            
            if agent_available:
                with st.spinner("🤖 AI Agent is analyzing your health profile..."):
                    # Get profile and logs
                    profile_dict = profile.to_dict()
                    logs_dict = logs if logs else []
                    
                    agent_result = st.session_state.agent_bridge.run_agent_analysis(
                        agent_type='health',
                        profile=profile_dict,
                        logs=logs_dict
                    )
                
                if agent_result and agent_result.get('success'):
                    response = agent_result['output']
                    metadata = agent_result.get('metadata', {})
                else:
                    # Fall back to direct LLM call with warning
                    st.warning("⚠️ Agent analysis failed, using standard LLM...")
                    with st.spinner("🤖 Analyzing your health profile..."):
                        # Calculate averages from logs
                        avg_sleep = 7.0
                        avg_energy = 5.0
                        avg_mood = 5.0
                        avg_stress = 5.0
                        
                        if logs:
                            sleep_hours = [log.get('sleep_hours', 7) for log in logs]
                            energy_levels = [log.get('energy_level', 5) for log in logs]
                            mood_levels = [log.get('mood_level', 5) for log in logs]
                            stress_levels = [log.get('stress_level', 5) for log in logs]
                            
                            if sleep_hours:
                                avg_sleep = sum(sleep_hours) / len(sleep_hours)
                            if energy_levels:
                                avg_energy = sum(energy_levels) / len(energy_levels)
                            if mood_levels:
                                avg_mood = sum(mood_levels) / len(mood_levels)
                            if stress_levels:
                                avg_stress = sum(stress_levels) / len(stress_levels)
                        
                        # Prepare the prompt (for fallback)
                        prompt = f"""
                        Provide a comprehensive health and wellness analysis for:
                        
                        PERSONAL INFORMATION:
                        - Name: {profile.name}
                        - Age: {profile.age}
                        - Weight: {profile.weight} kg
                        - Height: {profile.height} cm
                        - BMI: {profile.bmi} ({profile.bmi_category})
                        - Diet Type: {profile.diet_type}
                        - Fitness Goal: {profile.fitness_goal}
                        - Activity Level: {profile.activity_level}
                        - Medical Conditions: {', '.join(profile.medical_conditions) if profile.medical_conditions else 'None reported'}
                        - Allergies: {', '.join(profile.allergies) if profile.allergies else 'None reported'}
                        
                        RECENT HEALTH METRICS (7-day average):
                        - Sleep: {avg_sleep:.1f} hours per night
                        - Energy Level: {avg_energy:.1f}/10
                        - Mood: {avg_mood:.1f}/10
                        - Stress Level: {avg_stress:.1f}/10
                        
                        Please provide a health analysis.
                        """
                        response = st.session_state.llm.call_llm(prompt)
                        metadata = {}
            else:
                # Use direct LLM call if agent not available
                with st.spinner("🤖 Analyzing your health profile..."):
                    # Calculate averages from logs
                    avg_sleep = 7.0
                    avg_energy = 5.0
                    avg_mood = 5.0
                    avg_stress = 5.0
                    
                    if logs:
                        sleep_hours = [log.get('sleep_hours', 7) for log in logs]
                        energy_levels = [log.get('energy_level', 5) for log in logs]
                        mood_levels = [log.get('mood_level', 5) for log in logs]
                        stress_levels = [log.get('stress_level', 5) for log in logs]
                        
                        if sleep_hours:
                            avg_sleep = sum(sleep_hours) / len(sleep_hours)
                        if energy_levels:
                            avg_energy = sum(energy_levels) / len(energy_levels)
                        if mood_levels:
                            avg_mood = sum(mood_levels) / len(mood_levels)
                        if stress_levels:
                            avg_stress = sum(stress_levels) / len(stress_levels)
                    
                    prompt = f"""
                    Provide a comprehensive health and wellness analysis for:
                    
                    PERSONAL INFORMATION:
                    - Name: {profile.name}
                    - Age: {profile.age}
                    - Weight: {profile.weight} kg
                    - Height: {profile.height} cm
                    - BMI: {profile.bmi} ({profile.bmi_category})
                    - Diet Type: {profile.diet_type}
                    - Fitness Goal: {profile.fitness_goal}
                    - Activity Level: {profile.activity_level}
                    - Medical Conditions: {', '.join(profile.medical_conditions) if profile.medical_conditions else 'None reported'}
                    - Allergies: {', '.join(profile.allergies) if profile.allergies else 'None reported'}
                    
                    RECENT HEALTH METRICS (7-day average):
                    - Sleep: {avg_sleep:.1f} hours per night
                    - Energy Level: {avg_energy:.1f}/10
                    - Mood: {avg_mood:.1f}/10
                    - Stress Level: {avg_stress:.1f}/10
                    
                    Please provide a health analysis.
                    """
                    response = st.session_state.llm.call_llm(prompt)
                    metadata = {}
            
            st.session_state.health_analysis_response = response
            st.session_state.health_analysis_metadata = metadata
            st.rerun()
    
    # Display response if available
    if st.session_state.health_analysis_response:
        st.markdown("### 📊 Your Health & Wellness Analysis")
        
        # Display in a scrollable container
        with st.container():
            st.markdown('<div class="agent-output">', unsafe_allow_html=True)
            st.markdown(st.session_state.health_analysis_response)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Display agent metadata if available
        if hasattr(st.session_state, 'health_analysis_metadata') and st.session_state.health_analysis_metadata:
            display_agent_metadata(st.session_state.health_analysis_metadata, "Health Analysis")
        
        # Download and action buttons
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Health Report",
                data=st.session_state.health_analysis_response,
                file_name=f"health_report_{profile.name}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col2:
            if st.button("🔄 Generate Alternative", use_container_width=True):
                st.session_state.health_analysis_response = None
                st.session_state.health_analysis_metadata = None
                st.rerun()
        
        # Clear response button
        if st.button("🧹 Clear Response", use_container_width=True):
            st.session_state.health_analysis_response = None
            st.session_state.health_analysis_metadata = None
            st.rerun()

def show_mess_menu_upload():
    """Mess menu upload page"""
    if 'user_id' not in st.session_state:
        st.warning("Please create a profile first!")
        st.session_state.selected_page = "👤 Profile Setup"
        st.rerun()
        return
    
    profile = st.session_state.db.get_profile(st.session_state.user_id)
    
    if not profile.has_mess_access:
        st.warning("You don't have mess access in your profile.")
        st.info("Update your profile in Profile Management to enable mess features.")
        return
    
    st.markdown("## 📋 Upload Mess Menu")
    
    # Date selection
    today = datetime.now().strftime("%Y-%m-%d")
    selected_date = st.date_input("Select Date", value=date.today())
    date_str = selected_date.strftime("%Y-%m-%d")
    
    # Check if menu already exists for this date
    existing_menu = st.session_state.db.get_mess_menu(st.session_state.user_id, date_str)
    
    st.markdown(f"### Menu for {date_str}")
    
    with st.form("mess_menu_form"):
        # Breakfast
        st.markdown("#### 🥞 Breakfast")
        breakfast_items = st.text_area(
            "Breakfast Items (one per line)",
            value="\n".join(existing_menu.get('breakfast', [])) if existing_menu else "",
            placeholder="Idli\nSambar\nChutney\nMilk\nFruit",
            height=100
        )
        breakfast_notes = st.text_input(
            "Breakfast Notes",
            value=existing_menu.get('breakfast_notes', '') if existing_menu else "",
            placeholder="e.g., limited quantity, extra milk available"
        )
        
        # Lunch
        st.markdown("#### 🍛 Lunch")
        lunch_items = st.text_area(
            "Lunch Items (one per line)",
            value="\n".join(existing_menu.get('lunch', [])) if existing_menu else "",
            placeholder="Rice\nDal\nVegetable Curry\nRoti\nSalad\nCurd",
            height=120
        )
        lunch_notes = st.text_input(
            "Lunch Notes",
            value=existing_menu.get('lunch_notes', '') if existing_menu else "",
            placeholder="e.g., extra spicy, chicken available"
        )
        
        # Dinner
        st.markdown("#### 🍽️ Dinner")
        dinner_items = st.text_area(
            "Dinner Items (one per line)",
            value="\n".join(existing_menu.get('dinner', [])) if existing_menu else "",
            placeholder="Chapati\nPaneer Curry\nMixed Vegetables\nRice\nDal\nSalad",
            height=120
        )
        dinner_notes = st.text_input(
            "Dinner Notes",
            value=existing_menu.get('dinner_notes', '') if existing_menu else "",
            placeholder="e.g., limited paneer, extra salad"
        )
        
        # General notes
        general_notes = st.text_area(
            "General Notes",
            value=existing_menu.get('general_notes', '') if existing_menu else "",
            placeholder="Any additional information about today's menu..."
        )
        
        submitted = st.form_submit_button("Save Mess Menu", use_container_width=True)
        
        if submitted:
            # Process items
            breakfast_list = [item.strip() for item in breakfast_items.split('\n') if item.strip()]
            lunch_list = [item.strip() for item in lunch_items.split('\n') if item.strip()]
            dinner_list = [item.strip() for item in dinner_items.split('\n') if item.strip()]
            
            if not breakfast_list and not lunch_list and not dinner_list:
                st.error("Please enter at least one menu item!")
                return
            
            # Prepare menu data
            menu_data = {
                'breakfast': breakfast_list,
                'breakfast_notes': breakfast_notes,
                'lunch': lunch_list,
                'lunch_notes': lunch_notes,
                'dinner': dinner_list,
                'dinner_notes': dinner_notes,
                'general_notes': general_notes,
                'total_items': len(breakfast_list) + len(lunch_list) + len(dinner_list)
            }
            
            # Save menu
            if st.session_state.db.save_mess_menu(st.session_state.user_id, date_str, menu_data):
                st.success(f"✅ Mess menu saved for {date_str}!")
                
                # Show summary
                st.markdown(f"""
                <div class="success-box">
                <h4>Menu Summary:</h4>
                <p><strong>Date:</strong> {date_str}</p>
                <p><strong>Breakfast Items:</strong> {len(breakfast_list)}</p>
                <p><strong>Lunch Items:</strong> {len(lunch_list)}</p>
                <p><strong>Dinner Items:</strong> {len(dinner_list)}</p>
                <p><strong>Total Items:</strong> {menu_data['total_items']}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("❌ Failed to save menu. Please try again.")

def show_mess_optimizer():
    """Mess food optimizer page"""
    if 'user_id' not in st.session_state:
        st.warning("Please create a profile first!")
        st.session_state.selected_page = "👤 Profile Setup"
        st.rerun()
        return
    
    profile = st.session_state.db.get_profile(st.session_state.user_id)
    
    if not profile.has_mess_access:
        st.warning("You don't have mess access in your profile.")
        st.info("Update your profile in Profile Management to enable mess features.")
        return
    
    st.markdown("## 🤖 Mess Food Optimizer")
    
    # Date selection
    today = datetime.now().strftime("%Y-%m-%d")
    selected_date = st.date_input("Select Date", value=date.today())
    date_str = selected_date.strftime("%Y-%m-%d")
    
    # Get menu for selected date
    menu = st.session_state.db.get_mess_menu(st.session_state.user_id, date_str)
    
    if not menu:
        st.warning(f"No mess menu found for {date_str}.")
        st.info("Please upload a mess menu first using the 'Mess Menu Upload' section.")
        return
    
    # Display menu
    st.markdown(f"### Today's Menu ({date_str})")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🥞 Breakfast**")
        for item in menu.get('breakfast', []):
            st.write(f"• {item}")
        if menu.get('breakfast_notes'):
            st.caption(f"Note: {menu['breakfast_notes']}")
    
    with col2:
        st.markdown("**🍛 Lunch**")
        for item in menu.get('lunch', []):
            st.write(f"• {item}")
        if menu.get('lunch_notes'):
            st.caption(f"Note: {menu['lunch_notes']}")
    
    with col3:
        st.markdown("**🍽️ Dinner**")
        for item in menu.get('dinner', []):
            st.write(f"• {item}")
        if menu.get('dinner_notes'):
            st.caption(f"Note: {menu['dinner_notes']}")
    
    if menu.get('general_notes'):
        st.info(f"General Notes: {menu['general_notes']}")
    
    # Only show form if we don't have a response yet
    if st.session_state.mess_optimization_response is None:
        # Optimization options
        st.markdown("### Optimization Options")
        
        with st.form("mess_optimizer_form"):
            col1, col2 = st.columns(2)
            with col1:
                optimize_for = st.selectbox(
                    "Optimize for",
                    ["Maximum Nutrition", "Weight Loss", "Muscle Gain", "Energy Boost", "Budget Friendly"]
                )
            
            with col2:
                meal_pref = st.multiselect(
                    "Focus on Meals",
                    ["Breakfast", "Lunch", "Dinner"],
                    default=["Breakfast", "Lunch", "Dinner"]
                )
            
            # Extra constraints
            with st.expander("Additional Constraints"):
                col1, col2 = st.columns(2)
                with col1:
                    max_calories = st.number_input("Max Calories per Meal", 200, 1000, 600, 50)
                    min_protein = st.number_input("Min Protein per Meal (g)", 10, 50, 20, 5)
                with col2:
                    avoid_items = st.text_input("Items to Avoid", placeholder="e.g., fried food, sweets")
                    extra_budget = st.number_input("Extra Budget for Supplements (₹)", 0, 1000, 100, 50)
            
            submitted = st.form_submit_button("Generate Optimization Plan", use_container_width=True)
            
            if submitted:
                # Get recent logs for context
                logs = st.session_state.db.get_user_logs(st.session_state.user_id, days=3)
                
                # Try to use agent system first
                agent_result = None
                agent_available = st.session_state.agent_bridge.is_available()
                
                if agent_available:
                    with st.spinner("🤖 AI Agent is optimizing your mess meals..."):
                        # Get profile and logs
                        profile_dict = profile.to_dict()
                        logs_dict = logs if logs else []
                        
                        agent_result = st.session_state.agent_bridge.run_agent_analysis(
                            agent_type='mess_optimizer',
                            profile=profile_dict,
                            logs=logs_dict,
                            mess_menu=menu
                        )
                    
                    if agent_result and agent_result.get('success'):
                        response = agent_result['output']
                        metadata = agent_result.get('metadata', {})
                    else:
                        # Fall back to direct LLM call with warning
                        st.warning("⚠️ Agent analysis failed, using standard LLM...")
                        with st.spinner("🤖 Optimizing your mess meals..."):
                            # Prepare fallback prompt
                            recent_energy = 5
                            if logs:
                                recent_energy = sum(log.get('energy_level', 5) for log in logs) / len(logs)
                            
                            prompt = f"""
                            Optimize today's mess meals for:
                            
                            USER PROFILE:
                            - Name: {profile.name}
                            - Diet Type: {profile.diet_type}
                            - Fitness Goal: {profile.fitness_goal}
                            - Weight: {profile.weight} kg
                            - Weekly Budget: ₹{profile.weekly_budget}
                            - Allergies: {', '.join(profile.allergies) if profile.allergies else 'None'}
                            
                            OPTIMIZATION GOAL: {optimize_for}
                            FOCUS MEALS: {', '.join(meal_pref)}
                            RECENT ENERGY LEVEL: {recent_energy:.1f}/10
                            
                            CONSTRAINTS:
                            - Max Calories per Meal: {max_calories}
                            - Min Protein per Meal: {min_protein}g
                            - Items to Avoid: {avoid_items if avoid_items else 'None'}
                            - Extra Supplement Budget: ₹{extra_budget}
                            
                            TODAY'S MESS MENU:
                            BREAKFAST: {', '.join(menu.get('breakfast', []))}
                            Breakfast Notes: {menu.get('breakfast_notes', 'None')}
                            
                            LUNCH: {', '.join(menu.get('lunch', []))}
                            Lunch Notes: {menu.get('lunch_notes', 'None')}
                            
                            DINNER: {', '.join(menu.get('dinner', []))}
                            Dinner Notes: {menu.get('dinner_notes', 'None')}
                            
                            General Notes: {menu.get('general_notes', 'None')}
                            
                            Please provide mess meal optimization.
                            """
                            response = st.session_state.llm.call_llm(prompt)
                            metadata = {}
                else:
                    # Use direct LLM call if agent not available
                    with st.spinner("🤖 Optimizing your mess meals..."):
                        recent_energy = 5
                        if logs:
                            recent_energy = sum(log.get('energy_level', 5) for log in logs) / len(logs)
                        
                        prompt = f"""
                        Optimize today's mess meals for:
                        
                        USER PROFILE:
                        - Name: {profile.name}
                        - Diet Type: {profile.diet_type}
                        - Fitness Goal: {profile.fitness_goal}
                        - Weight: {profile.weight} kg
                        - Weekly Budget: ₹{profile.weekly_budget}
                        - Allergies: {', '.join(profile.allergies) if profile.allergies else 'None'}
                        
                        OPTIMIZATION GOAL: {optimize_for}
                        FOCUS MEALS: {', '.join(meal_pref)}
                        RECENT ENERGY LEVEL: {recent_energy:.1f}/10
                        
                        CONSTRAINTS:
                        - Max Calories per Meal: {max_calories}
                        - Min Protein per Meal: {min_protein}g
                        - Items to Avoid: {avoid_items if avoid_items else 'None'}
                        - Extra Supplement Budget: ₹{extra_budget}
                        
                        TODAY'S MESS MENU:
                        BREAKFAST: {', '.join(menu.get('breakfast', []))}
                        Breakfast Notes: {menu.get('breakfast_notes', 'None')}
                        
                        LUNCH: {', '.join(menu.get('lunch', []))}
                        Lunch Notes: {menu.get('lunch_notes', 'None')}
                        
                        DINNER: {', '.join(menu.get('dinner', []))}
                        Dinner Notes: {menu.get('dinner_notes', 'None')}
                        
                        General Notes: {menu.get('general_notes', 'None')}
                        
                        Please provide mess meal optimization.
                        """
                        response = st.session_state.llm.call_llm(prompt)
                        metadata = {}
                
                st.session_state.mess_optimization_response = response
                st.session_state.mess_optimization_metadata = metadata
                st.session_state.mess_optimization_date = date_str
                st.rerun()
    
    # Display response if available
    if st.session_state.mess_optimization_response:
        st.markdown("### 🍽️ Your Optimized Mess Plan")
        
        # Display in a scrollable container
        with st.container():
            st.markdown('<div class="agent-output">', unsafe_allow_html=True)
            st.markdown(st.session_state.mess_optimization_response)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Display agent metadata if available
        if hasattr(st.session_state, 'mess_optimization_metadata') and st.session_state.mess_optimization_metadata:
            display_agent_metadata(st.session_state.mess_optimization_metadata, "Mess Optimization")
        
        # Download and action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                label="📥 Download Optimization",
                data=st.session_state.mess_optimization_response,
                file_name=f"mess_optimization_{st.session_state.mess_optimization_date}.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col2:
            if st.button("🔄 Re-optimize", use_container_width=True):
                st.session_state.mess_optimization_response = None
                st.session_state.mess_optimization_metadata = None
                st.rerun()
        
        with col3:
            if st.button("🧹 Clear Response", use_container_width=True):
                st.session_state.mess_optimization_response = None
                st.session_state.mess_optimization_metadata = None
                st.rerun()

def show_profile_management():
    """Profile management page with edit functionality"""
    if 'user_id' not in st.session_state:
        st.warning("Please create a profile first!")
        st.session_state.selected_page = "👤 Profile Setup"
        st.rerun()
        return
    
    profile = st.session_state.db.get_profile(st.session_state.user_id)
    
    st.markdown(f"## 👤 Profile Management - {profile.name}")
    
    # Tab layout for profile management
    tab1, tab2, tab3 = st.tabs(["📋 View Profile", "✏️ Edit Profile", "📊 Health Stats"])
    
    with tab1:
        # Display current profile
        st.markdown("### Profile Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="profile-card">
            <h4 style="color: #1E88E5;">Personal Information</h4>
            <p><strong>👤 Name:</strong> {profile.name}</p>
            <p><strong>🎂 Age:</strong> {profile.age} years</p>
            <p><strong>⚖️ Weight:</strong> {profile.weight} kg</p>
            <p><strong>📏 Height:</strong> {profile.height} cm</p>
            <p><strong>📊 BMI:</strong> {profile.bmi} ({profile.bmi_category})</p>
            <p><strong>📅 Created:</strong> {datetime.fromisoformat(profile.created_at).strftime('%Y-%m-%d')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="profile-card">
            <h4 style="color: #1E88E5;">Health & Lifestyle</h4>
            <p><strong>🥗 Diet Type:</strong> {profile.diet_type.title()}</p>
            <p><strong>🎯 Fitness Goal:</strong> {profile.fitness_goal}</p>
            <p><strong>🏃 Activity Level:</strong> {profile.activity_level.title()}</p>
            <p><strong>🎓 Student:</strong> {'Yes' if profile.is_student else 'No'}</p>
            <p><strong>🏫 Mess Access:</strong> {'Yes' if profile.has_mess_access else 'No'}</p>
            <p><strong>💰 Weekly Budget:</strong> ₹{profile.weekly_budget}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Medical info
        st.markdown("### Medical Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="card">
            <h5>🏥 Medical Conditions</h5>
            {''.join(f'<p>• {condition}</p>' for condition in profile.medical_conditions) if profile.medical_conditions else '<p style="color: #666;">None reported</p>'}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="card">
            <h5>⚠️ Allergies</h5>
            {''.join(f'<p>• {allergy}</p>' for allergy in profile.allergies) if profile.allergies else '<p style="color: #666;">None reported</p>'}
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        # Edit profile form
        st.markdown("### Edit Your Profile")
        st.info("Update your profile information below. Changes will be saved immediately.")
        
        with st.form("edit_profile_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_name = st.text_input("Full Name", value=profile.name)
                new_age = st.number_input("Age", min_value=10, max_value=100, value=profile.age)
                new_weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=profile.weight)
                new_height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=profile.height)
            
            with col2:
                new_diet_type = st.selectbox(
                    "Diet Type",
                    ["Vegetarian", "Non-Vegetarian", "Eggetarian", "Vegan"],
                    index=["Vegetarian", "Non-Vegetarian", "Eggetarian", "Vegan"].index(profile.diet_type.title())
                )
                
                new_fitness_goal = st.selectbox(
                    "Fitness Goal",
                    ["Lose Weight", "Build Muscle", "Improve Endurance", "Maintain Health", "Gain Weight"],
                    index=["Lose Weight", "Build Muscle", "Improve Endurance", "Maintain Health", "Gain Weight"].index(profile.fitness_goal)
                )
                
                activity_levels = ["Sedentary", "Light", "Moderate", "Active", "Very Active"]
                current_activity = profile.activity_level.title()
                activity_index = activity_levels.index(current_activity) if current_activity in activity_levels else 2
                new_activity_level = st.select_slider(
                    "Activity Level",
                    options=activity_levels,
                    value=activity_levels[activity_index]
                )
            
            # Additional Information
            st.markdown("### Additional Information")
            col3, col4 = st.columns(2)
            
            with col3:
                new_is_student = st.checkbox("I'm a hostel student", value=profile.is_student)
                if new_is_student:
                    new_has_mess_access = st.checkbox("I have mess access", value=profile.has_mess_access)
                    new_weekly_budget = st.number_input("Weekly food budget (₹)", min_value=0, max_value=5000, 
                                                       value=int(profile.weekly_budget))
                else:
                    new_has_mess_access = False
                    new_weekly_budget = 500.0
            
            with col4:
                new_medical_input = st.text_input(
                    "Medical Conditions (comma separated)", 
                    value=", ".join(profile.medical_conditions) if profile.medical_conditions else ""
                )
                new_allergy_input = st.text_input(
                    "Food Allergies (comma separated)", 
                    value=", ".join(profile.allergies) if profile.allergies else ""
                )
            
            update_submitted = st.form_submit_button("Update Profile", use_container_width=True)
            
            if update_submitted:
                # Process medical conditions and allergies
                new_medical_conditions = [c.strip() for c in new_medical_input.split(",") if c.strip()] if new_medical_input else []
                new_allergies = [a.strip() for a in new_allergy_input.split(",") if a.strip()] if new_allergy_input else []
                
                # Create updated profile object
                updated_profile = UserProfile(
                    user_id=profile.user_id,
                    name=new_name,
                    age=int(new_age),
                    weight=float(new_weight),
                    height=float(new_height),
                    diet_type=new_diet_type,
                    fitness_goal=new_fitness_goal,
                    activity_level=new_activity_level.lower(),
                    is_student=new_is_student,
                    has_mess_access=new_has_mess_access if new_is_student else False,
                    weekly_budget=float(new_weekly_budget),
                    medical_conditions=new_medical_conditions,
                    allergies=new_allergies,
                    created_at=profile.created_at  # Keep original creation date
                )
                
                # Save updated profile
                if st.session_state.db.save_profile(updated_profile):
                    st.success("✅ Profile updated successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to update profile. Please try again.")
    
    with tab3:
        # Health Statistics
        st.markdown("### Health Statistics")
        
        # Calculate health metrics
        calories = calculate_calories(profile)
        protein = calculate_protein(profile)
        ideal_weight = calculate_ideal_weight(profile.height)
        
        # Metrics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Daily Calorie Needs", f"{calories:,}", "calories")
            st.metric("BMI Category", profile.bmi_category)
        
        with col2:
            st.metric("Protein Target", f"{protein}g", "per day")
            st.metric("Ideal Weight Range", ideal_weight, "kg")
        
        with col3:
            bmi_status = "healthy" if profile.bmi_category == "Healthy" else "needs attention"
            st.metric("BMI Status", bmi_status)
            st.metric("Activity Level", profile.activity_level.title())
        
        # Progress tracking
        st.markdown("### Progress Tracking")
        
        # Get logs for progress chart
        logs = st.session_state.db.get_user_logs(st.session_state.user_id, days=30)
        if logs:
            # Create progress data
            dates = []
            energies = []
            sleeps = []
            
            for log in logs:
                date_str = log.get('log_date', log.get('date', ''))
                if date_str:
                    dates.append(date_str)
                    energies.append(log.get('energy_level', 0))
                    sleeps.append(log.get('sleep_hours', 0))
            
            if dates:
                # Create a simple progress chart
                progress_data = {
                    'Date': dates[-10:],  # Last 10 entries
                    'Energy Level': energies[-10:],
                    'Sleep Hours': sleeps[-10:]
                }
                
                if len(dates) >= 2:
                    df = pd.DataFrame(progress_data)
                    st.line_chart(df.set_index('Date'))
                else:
                    st.info("More data needed for progress tracking. Continue logging daily!")
        else:
            st.info("No progress data available yet. Start logging daily check-ins!")
        
        # Health recommendations
        st.markdown("### Health Recommendations")
        
        if profile.bmi_category != "Healthy":
            if profile.bmi_category == "Underweight":
                st.warning("📈 **Recommendation:** Consider increasing calorie intake with healthy foods to reach a healthy weight range.")
            elif profile.bmi_category == "Overweight":
                st.warning("📉 **Recommendation:** Focus on balanced diet and regular exercise for healthy weight management.")
            else:  # Obese
                st.error("⚠️ **Recommendation:** Consult with a healthcare provider for personalized weight management guidance.")
        
        if profile.activity_level.lower() == "sedentary":
            st.info("🏃 **Recommendation:** Try to incorporate more movement into your daily routine for better health.")
    
    # Profile Management Actions
    st.markdown("---")
    st.markdown("### Profile Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Profile", use_container_width=True):
            st.rerun()
    
    with col2:
        # Export profile data as JSON
        profile_data = profile.to_dict()
        st.download_button(
            label="📥 Export Profile",
            data=json.dumps(profile_data, indent=2),
            file_name=f"health_profile_{profile.name}_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col3:
        if st.button("🗑️ Delete Profile", use_container_width=True, type="secondary"):
            st.warning("⚠️ This will permanently delete your profile and all associated data!")
            confirm_col1, confirm_col2 = st.columns(2)
            with confirm_col1:
                if st.button("✅ Confirm Delete", use_container_width=True, type="primary"):
                    # Delete profile from database
                    if profile.user_id in st.session_state.db.profiles:
                        del st.session_state.db.profiles[profile.user_id]
                        st.session_state.db._save_profiles()
                    
                    # Clear user session
                    for key in list(st.session_state.keys()):
                        if key not in ['db', 'llm', 'agent_bridge', 'selected_page']:
                            del st.session_state[key]
                    
                    st.session_state.selected_page = "👤 Profile Setup"
                    st.success("Profile deleted successfully!")
                    st.rerun()
            with confirm_col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.rerun()

def show_about():
    """About page"""
    st.markdown("## ℹ️ About Health AI Assistant")
    
    st.markdown("""
    ### 🎯 Purpose
    Health AI Assistant is your personal health and fitness companion that uses artificial intelligence 
    to provide personalized health guidance, diet plans, workout routines, and mess food optimization 
    specifically designed for students and working professionals.
    
    ### ✨ Key Features
    
    **🤖 Autonomous TAO Agents**
    - **Think-Act-Observe** framework for deep reasoning
    - Diet, fitness, health, and mess optimization agents
    - Dynamic iteration based on data quality
    - Transparent reasoning traces
    
    **👤 Profile Management**
    - Create detailed health profiles
    - Track BMI and health metrics
    - Store medical conditions and allergies
    - Edit and update profile information
    
    **📝 Daily Health Tracking**
    - Record sleep, meals, and activity
    - Track energy, mood, and stress levels
    - Monitor progress over time
    
    **🍽️ Diet & Nutrition**
    - Personalized 7-day meal plans
    - Daily nutritional targets
    - Concise web-optimized outputs
    - Diet-type specific recommendations
    
    **💪 Fitness & Exercise**
    - Custom workout plans for your fitness goals
    - Exercise instructions with proper form
    - Progressive overload scheduling
    
    **📋 Mess Food Optimization** (for students)
    - Upload daily mess menus
    - AI-powered meal optimization
    - Supplement recommendations
    - Cost analysis and budgeting
    
    ### 🔒 Privacy & Security
    - All data is stored locally on your device
    - No personal information is shared with third parties
    - You have full control over your data
    - JSON-based storage for easy backup
    
    ### 🛠️ Technology Stack
    - **Frontend**: Streamlit (Python web framework)
    - **AI Engine**: Groq API with Llama 3.1 model
    - **Agent Framework**: LangGraph with TAO (Think-Act-Observe) loops
    - **Data Storage**: Local JSON files
    - **Visualization**: Streamlit charts and metrics
    
    ### ⚠️ Important Disclaimer
    This application provides health and fitness suggestions based on AI analysis. 
    It is **NOT** a substitute for professional medical advice, diagnosis, or treatment. 
    Always consult qualified healthcare providers for medical concerns.
    
    ### 📞 Setup Instructions
    1. **Without API Key**: Runs with mock responses
    2. **With API Key**: 
       - Get Groq API key from: https://console.groq.com/keys
       - Install: pip install langchain-groq langgraph pydantic
       - Enter API key when prompted in the app
    
    ### 🏆 Agentic System Features
    - **Autonomous Reasoning**: Agents decide their own analysis depth
    - **Goal-Oriented Behavior**: Each agent pursues specific health objectives
    - **Multi-Step Planning**: Complex analysis broken into thinking steps
    - **Adaptive Learning**: Agents learn from previous iterations
    - **Transparent Decision Making**: Full reasoning traces available
    
    ### 📄 License
    This software is provided for educational and personal use.
    """)

# ========== RUN THE APP ==========
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please refresh the page or check your connection.")

