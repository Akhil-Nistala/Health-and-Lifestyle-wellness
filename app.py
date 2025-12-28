import streamlit as st
import os
import json
from datetime import datetime, date
from typing import Dict, List, Any, Optional
from pathlib import Path
import uuid
import re
from enum import Enum
#from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load environment
#load_dotenv()

# ========== CONFIGURATION ==========
class Config:
    MODEL = "llama-3.1-8b-instant"
    DATA_DIR = Path("health_ai_web_data")
    SESSION_ID = str(uuid.uuid4())[:8]

# Ensure data directory exists
Config.DATA_DIR.mkdir(exist_ok=True)

# ========== DATA MODELS ==========
class DietType(Enum):
    VEGETARIAN = "vegetarian"
    NON_VEGETARIAN = "non_vegetarian"
    EGGETARIAN = "eggetarian"
    VEGAN = "vegan"

@st.cache_data
def get_diet_type_enum(value: str) -> DietType:
    """Safely convert string to DietType enum"""
    try:
        return DietType(value.lower())
    except:
        return DietType.NON_VEGETARIAN

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
            'diet_type': self.diet_type.value,
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
            st.error(f"Error loading data: {str(e)[:200]}")
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
            st.error(f"Error saving profiles: {str(e)[:200]}")
    
    def _save_logs(self):
        """Save logs to file"""
        try:
            with open(self.logs_file, 'w', encoding='utf-8') as f:
                json.dump(self.logs, f, indent=2, ensure_ascii=False)
        except Exception as e:
            st.error(f"Error saving logs: {str(e)[:200]}")
    
    def _save_mess_menus(self):
        """Save mess menus to file"""
        try:
            with open(self.mess_menus_file, 'w', encoding='utf-8') as f:
                json.dump(self.mess_menus, f, indent=2, ensure_ascii=False)
        except Exception as e:
            st.error(f"Error saving mess menus: {str(e)[:200]}")
    
    def save_profile(self, profile: UserProfile) -> bool:
        """Save or update user profile"""
        try:
            self.profiles[profile.user_id] = profile
            self._save_profiles()
            return True
        except Exception as e:
            st.error(f"Error saving profile: {str(e)[:200]}")
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
            st.error(f"Error saving log: {str(e)[:200]}")
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
            st.error(f"Error saving mess menu: {str(e)[:200]}")
            return False
    
    def get_mess_menu(self, user_id: str, date_str: str) -> Optional[Dict]:
        """Get mess menu for a specific date"""
        user_menus = self.mess_menus.get(user_id, {})
        return user_menus.get(date_str)

# ========== LLM SERVICE ==========
class LLMService:
    def __init__(self):
        self.api_key = st.secrets["GROQ_API_KEY"]
        if not self.api_key:
            st.error("❌ GROQ_API_KEY not found in .env file!")
            st.info("Please create a `.env` file with: GROQ_API_KEY=your_key_here")
            self.llm = None
        else:
            self.llm = self._init_llm()
    
    def _init_llm(self):
        """Initialize LLM with error handling"""
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                groq_api_key=self.api_key,
                model=Config.MODEL,
                temperature=0.7,
                max_tokens=2000
            )
        except ImportError:
            st.error("❌ Please install langchain-groq: pip install langchain-groq")
            return None
        except Exception as e:
            st.error(f"❌ Error initializing LLM: {str(e)[:200]}")
            return None
    
    def call_llm(self, prompt: str) -> str:
        """Call LLM with error handling"""
        if not self.llm:
            return "⚠️ LLM service not available. Please check your GROQ_API_KEY and install langchain-groq."
        
        try:
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt[:4000])])
            return response.content.strip()
        except Exception as e:
            return f"❌ Error generating response: {str(e)[:200]}"

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
    # Basic BMR calculation
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
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🏥 Health AI Assistant</h1>', unsafe_allow_html=True)
    
    # Initialize services
    if 'db' not in st.session_state:
        st.session_state.db = DatabaseManager()
    
    if 'llm' not in st.session_state:
        st.session_state.llm = LLMService()
    
    # Initialize session states for storing AI responses
    if 'diet_plan_response' not in st.session_state:
        st.session_state.diet_plan_response = None
    if 'workout_plan_response' not in st.session_state:
        st.session_state.workout_plan_response = None
    if 'health_analysis_response' not in st.session_state:
        st.session_state.health_analysis_response = None
    if 'mess_optimization_response' not in st.session_state:
        st.session_state.mess_optimization_response = None
    
    # Sidebar Navigation
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3063/3063812.png", width=100)
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
                "👤 Profile"
            ]
        
        selected_page = st.selectbox("Choose a section:", page_options)
        
        st.markdown("---")
        
        # User info if logged in
        if 'user_id' in st.session_state and st.session_state.user_id:
            profile = st.session_state.db.get_profile(st.session_state.user_id)
            if profile:
                st.markdown(f"**Logged in as:** {profile.name}")
                if st.button("🚪 Logout", use_container_width=True):
                    for key in list(st.session_state.keys()):
                        if key not in ['db', 'llm']:
                            del st.session_state[key]
                    st.rerun()
        
        st.markdown("---")
        st.markdown("### About")
        st.info("AI-powered health assistant for personalized fitness and nutrition guidance.")
    
    # Page routing
    if selected_page == "👤 Profile Setup":
        show_profile_setup()
    elif selected_page == "📊 Dashboard":
        show_dashboard()
    elif selected_page == "📝 Daily Check-in":
        show_daily_checkin()
    elif selected_page == "🍽️ Diet Analysis":
        show_diet_analysis()
    elif selected_page == "💪 Fitness Analysis":
        show_fitness_analysis()
    elif selected_page == "⚕️ Health Analysis":
        show_health_analysis()
    elif selected_page == "📋 Mess Menu Upload":
        show_mess_menu_upload()
    elif selected_page == "🤖 Mess Optimizer":
        show_mess_optimizer()
    elif selected_page == "👤 Profile":
        show_profile()
    elif selected_page == "ℹ️ About":
        show_about()

def show_profile_setup():
    """User profile creation page"""
    st.markdown("## 👤 Create Your Profile")
    
    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name", placeholder="John Doe", key="name_input")
            age = st.number_input("Age", min_value=10, max_value=100, value=25, key="age_input")
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, key="weight_input")
            height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, key="height_input")
        
        with col2:
            diet_type = st.selectbox(
                "Diet Type",
                ["Vegetarian", "Non-Vegetarian", "Eggetarian", "Vegan"],
                key="diet_input"
            )
            
            fitness_goal = st.selectbox(
                "Fitness Goal",
                ["Lose Weight", "Build Muscle", "Improve Endurance", "Maintain Health", "Gain Weight"],
                key="goal_input"
            )
            
            activity_level = st.select_slider(
                "Activity Level",
                options=["Sedentary", "Light", "Moderate", "Active", "Very Active"],
                key="activity_input"
            )
        
        st.markdown("### Additional Information")
        col3, col4 = st.columns(2)
        
        with col3:
            is_student = st.checkbox("I'm a hostel student", key="student_check")
            if is_student:
                has_mess_access = st.checkbox("I have mess access", key="mess_check", value=True)
                weekly_budget = st.number_input("Weekly food budget (₹)", min_value=0, max_value=5000, value=500, key="budget_input")
            else:
                has_mess_access = False
                weekly_budget = 500
        
        with col4:
            medical_input = st.text_input("Medical Conditions (comma separated)", placeholder="None", key="medical_input")
            allergy_input = st.text_input("Food Allergies (comma separated)", placeholder="None", key="allergy_input")
        
        submitted = st.form_submit_button("Create Profile", use_container_width=True)
        
        if submitted:
            if not name or name == "John Doe":
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
                
                # Auto-navigate to dashboard after 2 seconds
                st.info("Redirecting to dashboard...")
                st.rerun()
            else:
                st.error("❌ Failed to save profile. Please try again.")

def show_dashboard():
    """Main dashboard page"""
    if 'user_id' not in st.session_state:
        st.warning("Please create a profile first!")
        show_profile_setup()
        return
    
    profile = st.session_state.db.get_profile(st.session_state.user_id)
    if not profile:
        st.error("Profile not found!")
        return
    
    st.markdown(f"## 📊 Dashboard - {profile.name}")
    
    # Stats Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("BMI", f"{profile.bmi}", profile.bmi_category)
    
    with col2:
        st.metric("Age", profile.age)
    
    with col3:
        st.metric("Weight", f"{profile.weight} kg")
    
    with col4:
        st.metric("Height", f"{profile.height} cm")
    
    st.markdown("---")
    
    # Profile Information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Profile Information")
        info_html = f"""
        <div class="card">
        <p><strong>Diet Type:</strong> {profile.diet_type.value.title()}</p>
        <p><strong>Fitness Goal:</strong> {profile.fitness_goal}</p>
        <p><strong>Activity Level:</strong> {profile.activity_level.title()}</p>
        <p><strong>Weekly Budget:</strong> ₹{profile.weekly_budget}</p>
        <p><strong>Student:</strong> {'Yes' if profile.is_student else 'No'}</p>
        <p><strong>Mess Access:</strong> {'Yes' if profile.has_mess_access else 'No'}</p>
        </div>
        """
        st.markdown(info_html, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Health Metrics")
        calories = calculate_calories(profile)
        protein = calculate_protein(profile)
        ideal_weight = calculate_ideal_weight(profile.height)
        
        metrics_html = f"""
        <div class="card">
        <p><strong>BMI Category:</strong> {profile.bmi_category}</p>
        <p><strong>Ideal Weight Range:</strong> {ideal_weight} kg</p>
        <p><strong>Daily Calorie Needs:</strong> {calories:,} calories</p>
        <p><strong>Protein Target:</strong> {protein}g per day</p>
        <p><strong>Medical Conditions:</strong> {', '.join(profile.medical_conditions) if profile.medical_conditions else 'None'}</p>
        <p><strong>Allergies:</strong> {', '.join(profile.allergies) if profile.allergies else 'None'}</p>
        </div>
        """
        st.markdown(metrics_html, unsafe_allow_html=True)
    
    # Recent Logs
    logs = st.session_state.db.get_user_logs(st.session_state.user_id, days=7)
    if logs:
        st.markdown("### Recent Activity")
        
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
            st.session_state.show_daily_checkin = True
            st.rerun()
    
    with col2:
        if st.button("🍽️ Diet Plan", use_container_width=True):
            st.session_state.show_diet_analysis = True
            st.rerun()
    
    with col3:
        if st.button("💪 Workout Plan", use_container_width=True):
            st.session_state.show_fitness_analysis = True
            st.rerun()
    
    with col4:
        if profile.has_mess_access:
            if st.button("📋 Mess Menu", use_container_width=True):
                st.session_state.show_mess_menu = True
                st.rerun()

def show_daily_checkin():
    """Daily health check-in page"""
    if 'user_id' not in st.session_state:
        st.warning("Please create a profile first!")
        show_profile_setup()
        return
    
    st.markdown("## 📝 Daily Health Check-in")
    
    today = datetime.now().strftime("%Y-%m-%d")
    st.markdown(f"### Date: {today}")
    
    with st.form("daily_checkin_form"):
        # Sleep Section
        st.markdown("#### 😴 Sleep")
        col1, col2 = st.columns(2)
        with col1:
            sleep_hours = st.slider("Sleep Hours", 0.0, 24.0, 7.0, 0.5, key="sleep_hours")
        with col2:
            sleep_quality = st.slider("Sleep Quality (1-10)", 1, 10, 7, key="sleep_quality")
        sleep_notes = st.text_area("Sleep Notes", placeholder="Any sleep issues or observations...", key="sleep_notes")
        
        # Meals Section
        st.markdown("#### 🍽️ Meals")
        col1, col2, col3 = st.columns(3)
        with col1:
            breakfast = st.text_input("Breakfast", placeholder="What did you eat?", key="breakfast")
        with col2:
            lunch = st.text_input("Lunch", placeholder="What did you eat?", key="lunch")
        with col3:
            dinner = st.text_input("Dinner", placeholder="What did you eat?", key="dinner")
        
        water_intake = st.slider("Water Intake (liters)", 0.0, 10.0, 2.0, 0.5, key="water")
        
        # Activity Section
        st.markdown("#### 🏃 Activity")
        steps = st.number_input("Steps Today", 0, 50000, 5000, key="steps")
        
        workout_today = st.checkbox("Did you workout today?", key="workout_check")
        workout_duration = 0
        workout_type = ""
        if workout_today:
            col1, col2 = st.columns(2)
            with col1:
                workout_duration = st.number_input("Workout Duration (minutes)", 1, 240, 30, key="workout_duration")
            with col2:
                workout_type = st.text_input("Workout Type", placeholder="Cardio, Strength, Yoga...", key="workout_type")
        
        # Metrics Section
        st.markdown("#### 📊 Health Metrics (1-10)")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            energy = st.slider("Energy Level", 1, 10, 5, key="energy")
        with col2:
            mood = st.slider("Mood", 1, 10, 5, key="mood")
        with col3:
            stress = st.slider("Stress Level", 1, 10, 5, key="stress")
        with col4:
            focus = st.slider("Focus Level", 1, 10, 5, key="focus")
        
        symptoms = st.text_input("Any symptoms? (comma separated)", placeholder="headache, fatigue, etc.", key="symptoms")
        notes = st.text_area("Additional Notes", placeholder="Any other observations...", key="notes")
        
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
        show_profile_setup()
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
            st.write(f"**Diet:** {profile.diet_type.value.title()}")
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
            with st.spinner("🤖 AI is creating your personalized diet plan..."):
                prompt = f"""
                Create a detailed 7-day meal plan for:
                
                PERSONAL INFORMATION:
                - Name: {profile.name}
                - Age: {profile.age}
                - Weight: {profile.weight} kg
                - Height: {profile.height} cm
                - BMI: {profile.bmi} ({profile.bmi_category})
                - Diet Type: {profile.diet_type.value}
                - Fitness Goal: {profile.fitness_goal}
                - Activity Level: {profile.activity_level}
                - Allergies: {', '.join(profile.allergies) if profile.allergies else 'None'}
                
                PREFERENCES:
                - Preferred Cuisines: {', '.join(cuisine_pref)}
                - Max Cooking Time: {cooking_time}
                - Meal Prep: {'Yes' if meal_prep else 'No'}
                - Weekly Budget: ₹{budget}
                
                Please provide:
                1. DAILY CALORIE TARGET: Based on {profile.weight}kg weight and {profile.fitness_goal} goal
                2. MACRONUTRIENT BREAKDOWN: Protein, Carbs, Fats in grams per day
                3. 7-DAY MEAL PLAN: Specific foods and portions for each meal
                4. GROCERY SHOPPING LIST: Organized by category with quantities
                5. MEAL PREP TIPS: Time-saving strategies
                6. HYDRATION PLAN: Water intake schedule
                7. SNACK IDEAS: Healthy snack options
                8. COST ESTIMATE: Approximate weekly cost
                
                Make it practical, affordable, and tailored to {profile.diet_type.value} diet.
                Include Indian options where possible.
                """
                
                response = st.session_state.llm.call_llm(prompt)
                st.session_state.diet_plan_response = response
                st.session_state.diet_plan_prompt = prompt
                st.rerun()
    
    # Display response if available
    if st.session_state.diet_plan_response:
        st.markdown("### 📋 Your Personalized Diet Plan")
        st.markdown(st.session_state.diet_plan_response)
        
        # Download and regenerate buttons (outside form)
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
                st.rerun()
        
        # Clear response button
        if st.button("🧹 Clear Response", use_container_width=True):
            st.session_state.diet_plan_response = None
            st.rerun()

def show_fitness_analysis():
    """Fitness analysis page"""
    if 'user_id' not in st.session_state:
        st.warning("Please create a profile first!")
        show_profile_setup()
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
                pushups = st.number_input("Max Pushups", 0, 100, 20, key="pushups")
                plank_time = st.number_input("Plank Time (seconds)", 0, 600, 60, key="plank")
            with col2:
                running_5k = st.number_input("5K Run Time (minutes)", 0, 120, 30, key="run")
                flexibility = st.select_slider(
                    "Flexibility",
                    options=["Poor", "Average", "Good", "Excellent"],
                    value="Average",
                    key="flex"
                )
            
            # Equipment availability
            st.markdown("### Equipment Available")
            col1, col2, col3 = st.columns(3)
            with col1:
                has_gym = st.checkbox("Gym Access", value=True, key="has_gym")
            with col2:
                has_weights = st.checkbox("Dumbbells/Weights", value=False, key="has_weights")
            with col3:
                has_yogamat = st.checkbox("Yoga Mat", value=True, key="has_yogamat")
            
            submitted = st.form_submit_button("Generate Workout Plan", use_container_width=True)
            
            if submitted:
                with st.spinner("🤖 Creating your personalized workout plan..."):
                    equipment = []
                    if has_gym:
                        equipment.append("Gym equipment")
                    if has_weights:
                        equipment.append("Dumbbells/weights")
                    if has_yogamat:
                        equipment.append("Yoga mat")
                    
                    prompt = f"""
                    Create a 4-week personalized workout plan for:
                    
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
                    
                    Please provide:
                    1. PROGRAM OVERVIEW: Weekly focus and objectives
                    2. WEEKLY SCHEDULE: Day-by-day workout plan (Monday-Sunday)
                    3. EXERCISE LIBRARY: Detailed instructions for each exercise
                    4. WARM-UP ROUTINE: 5-10 minute warm-up
                    5. COOL-DOWN ROUTINE: 5-10 minute cool-down
                    6. PROGRESSION SYSTEM: How to increase difficulty each week
                    7. HOME WORKOUT ALTERNATIVES: No-equipment options
                    8. SAFETY GUIDELINES: Injury prevention tips
                    9. TRACKING TEMPLATE: Workout log and progress tracking
                    
                    Focus on safety, proper form, and gradual progression.
                    Include both strength and cardio exercises.
                    """
                    
                    response = st.session_state.llm.call_llm(prompt)
                    st.session_state.workout_plan_response = response
                    st.rerun()
    
    # Display response if available
    if st.session_state.workout_plan_response:
        st.markdown("### 🏋️ Your Personalized Workout Plan")
        st.markdown(st.session_state.workout_plan_response)
        
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
                st.rerun()
        
        # Clear response button
        if st.button("🧹 Clear Response", use_container_width=True):
            st.session_state.workout_plan_response = None
            st.rerun()

def show_health_analysis():
    """Health analysis page"""
    if 'user_id' not in st.session_state:
        st.warning("Please create a profile first!")
        show_profile_setup()
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
            with st.spinner("🤖 Analyzing your health profile..."):
                # Get recent logs for context
                logs = st.session_state.db.get_user_logs(st.session_state.user_id, days=7)
                
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
                - Diet Type: {profile.diet_type.value}
                - Fitness Goal: {profile.fitness_goal}
                - Activity Level: {profile.activity_level}
                - Medical Conditions: {', '.join(profile.medical_conditions) if profile.medical_conditions else 'None reported'}
                - Allergies: {', '.join(profile.allergies) if profile.allergies else 'None reported'}
                
                RECENT HEALTH METRICS (7-day average):
                - Sleep: {avg_sleep:.1f} hours per night
                - Energy Level: {avg_energy:.1f}/10
                - Mood: {avg_mood:.1f}/10
                - Stress Level: {avg_stress:.1f}/10
                
                Please provide analysis in these areas:
                
                1. NUTRITIONAL ASSESSMENT:
                   - Current diet evaluation for {profile.diet_type.value}
                   - Potential nutrient deficiencies to watch for
                   - Meal timing and portion recommendations
                
                2. EXERCISE RECOMMENDATIONS:
                   - Suitable exercise types for {profile.fitness_goal}
                   - Frequency and intensity guidelines
                   - Recovery and rest recommendations
                
                3. SLEEP OPTIMIZATION:
                   - Sleep quality assessment
                   - Ideal sleep schedule
                   - Sleep hygiene tips
                
                4. STRESS MANAGEMENT:
                   - Stress level analysis
                   - Coping strategies
                   - Relaxation techniques
                
                5. LIFESTYLE IMPROVEMENTS:
                   - Daily routine suggestions
                   - Habit formation strategies
                   - Work-life balance tips
                
                6. PREVENTIVE MEASURES:
                   - Health screenings to consider
                   - Vaccination reminders (if applicable)
                   - Regular check-up schedule
                
                7. ACTIONABLE RECOMMENDATIONS:
                   - Top 3 immediate changes
                   - 30-day improvement plan
                   - Progress tracking methods
                
                IMPORTANT: Do NOT provide medical diagnosis or treatment plans.
                Focus on evidence-based lifestyle improvements and preventive health.
                Be specific, practical, and encouraging.
                """
                
                response = st.session_state.llm.call_llm(prompt)
                st.session_state.health_analysis_response = response
                st.rerun()
    
    # Display response if available
    if st.session_state.health_analysis_response:
        st.markdown("### 📊 Your Health & Wellness Analysis")
        st.markdown(st.session_state.health_analysis_response)
        
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
                st.rerun()
        
        # Clear response button
        if st.button("🧹 Clear Response", use_container_width=True):
            st.session_state.health_analysis_response = None
            st.rerun()

def show_mess_menu_upload():
    """Mess menu upload page"""
    if 'user_id' not in st.session_state:
        st.warning("Please create a profile first!")
        show_profile_setup()
        return
    
    profile = st.session_state.db.get_profile(st.session_state.user_id)
    
    if not profile.has_mess_access:
        st.warning("You don't have mess access in your profile.")
        st.info("Update your profile to enable mess features.")
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
            height=100,
            key="breakfast_items"
        )
        breakfast_notes = st.text_input(
            "Breakfast Notes",
            value=existing_menu.get('breakfast_notes', '') if existing_menu else "",
            placeholder="e.g., limited quantity, extra milk available",
            key="breakfast_notes"
        )
        
        # Lunch
        st.markdown("#### 🍛 Lunch")
        lunch_items = st.text_area(
            "Lunch Items (one per line)",
            value="\n".join(existing_menu.get('lunch', [])) if existing_menu else "",
            placeholder="Rice\nDal\nVegetable Curry\nRoti\nSalad\nCurd",
            height=120,
            key="lunch_items"
        )
        lunch_notes = st.text_input(
            "Lunch Notes",
            value=existing_menu.get('lunch_notes', '') if existing_menu else "",
            placeholder="e.g., extra spicy, chicken available",
            key="lunch_notes"
        )
        
        # Dinner
        st.markdown("#### 🍽️ Dinner")
        dinner_items = st.text_area(
            "Dinner Items (one per line)",
            value="\n".join(existing_menu.get('dinner', [])) if existing_menu else "",
            placeholder="Chapati\nPaneer Curry\nMixed Vegetables\nRice\nDal\nSalad",
            height=120,
            key="dinner_items"
        )
        dinner_notes = st.text_input(
            "Dinner Notes",
            value=existing_menu.get('dinner_notes', '') if existing_menu else "",
            placeholder="e.g., limited paneer, extra salad",
            key="dinner_notes"
        )
        
        # General notes
        general_notes = st.text_area(
            "General Notes",
            value=existing_menu.get('general_notes', '') if existing_menu else "",
            placeholder="Any additional information about today's menu...",
            key="general_notes"
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
        show_profile_setup()
        return
    
    profile = st.session_state.db.get_profile(st.session_state.user_id)
    
    if not profile.has_mess_access:
        st.warning("You don't have mess access in your profile.")
        st.info("Update your profile to enable mess features.")
        return
    
    st.markdown("## 🤖 Mess Food Optimizer")
    
    # Date selection
    today = datetime.now().strftime("%Y-%m-%d")
    selected_date = st.date_input("Select Date", value=date.today(), key="optimizer_date")
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
                    ["Maximum Nutrition", "Weight Loss", "Muscle Gain", "Energy Boost", "Budget Friendly"],
                    key="optimize_for"
                )
            
            with col2:
                meal_pref = st.multiselect(
                    "Focus on Meals",
                    ["Breakfast", "Lunch", "Dinner"],
                    default=["Breakfast", "Lunch", "Dinner"],
                    key="meal_pref"
                )
            
            # Extra constraints
            with st.expander("Additional Constraints"):
                col1, col2 = st.columns(2)
                with col1:
                    max_calories = st.number_input("Max Calories per Meal", 200, 1000, 600, 50, key="max_cal")
                    min_protein = st.number_input("Min Protein per Meal (g)", 10, 50, 20, 5, key="min_protein")
                with col2:
                    avoid_items = st.text_input("Items to Avoid", placeholder="e.g., fried food, sweets", key="avoid")
                    extra_budget = st.number_input("Extra Budget for Supplements (₹)", 0, 1000, 100, 50, key="extra_budget")
            
            submitted = st.form_submit_button("Generate Optimization Plan", use_container_width=True)
            
            if submitted:
                with st.spinner("🤖 Optimizing your mess meals..."):
                    # Get recent logs for context
                    logs = st.session_state.db.get_user_logs(st.session_state.user_id, days=3)
                    recent_energy = 5
                    if logs:
                        recent_energy = sum(log.get('energy_level', 5) for log in logs) / len(logs)
                    
                    prompt = f"""
                    Optimize today's mess meals for:
                    
                    USER PROFILE:
                    - Name: {profile.name}
                    - Diet Type: {profile.diet_type.value}
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
                    
                    Please provide:
                    
                    1. MEAL-BY-MEAL RECOMMENDATIONS:
                       - For each meal, specify exactly what to eat and in what quantities
                       - Rate each item (1-10) for nutritional value
                       - Portion size recommendations
                    
                    2. NUTRITIONAL ANALYSIS:
                       - Estimated calories per meal
                       - Protein, carbs, fat breakdown
                       - Key nutrients in each meal
                    
                    3. SUPPLEMENTATION PLAN:
                       - What supplements to add (if any)
                       - Exact quantities and approximate cost
                       - How to incorporate with mess food
                    
                    4. MEAL TIMING:
                       - Best time to eat each meal
                       - Pre/post-workout recommendations
                       - Hydration schedule
                    
                    5. COST ANALYSIS:
                       - Cost of selected mess items
                       - Supplement cost
                       - Total value for money
                    
                    6. ALTERNATIVES:
                       - What to choose if preferred items run out
                       - Quick hostel room additions
                    
                    Be VERY SPECIFIC with Indian hostel context.
                    Include exact food names from the menu.
                    Prioritize practicality and affordability.
                    """
                    
                    response = st.session_state.llm.call_llm(prompt)
                    st.session_state.mess_optimization_response = response
                    st.session_state.mess_optimization_date = date_str
                    st.rerun()
    
    # Display response if available
    if st.session_state.mess_optimization_response:
        st.markdown("### 🍽️ Your Optimized Mess Plan")
        st.markdown(st.session_state.mess_optimization_response)
        
        # Download and action buttons (outside form)
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
                st.rerun()
        
        with col3:
            if st.button("🧹 Clear Response", use_container_width=True):
                st.session_state.mess_optimization_response = None
                st.rerun()

def show_profile():
    """Profile management page"""
    if 'user_id' not in st.session_state:
        st.warning("Please create a profile first!")
        show_profile_setup()
        return
    
    profile = st.session_state.db.get_profile(st.session_state.user_id)
    
    st.markdown(f"## 👤 Your Profile - {profile.name}")
    
    # Display current profile
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Personal Information")
        info_html = f"""
        <div class="card">
        <p><strong>Name:</strong> {profile.name}</p>
        <p><strong>Age:</strong> {profile.age}</p>
        <p><strong>Weight:</strong> {profile.weight} kg</p>
        <p><strong>Height:</strong> {profile.height} cm</p>
        <p><strong>BMI:</strong> {profile.bmi} ({profile.bmi_category})</p>
        <p><strong>Created:</strong> {datetime.fromisoformat(profile.created_at).strftime('%Y-%m-%d')}</p>
        </div>
        """
        st.markdown(info_html, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Health & Lifestyle")
        info_html = f"""
        <div class="card">
        <p><strong>Diet Type:</strong> {profile.diet_type.value.title()}</p>
        <p><strong>Fitness Goal:</strong> {profile.fitness_goal}</p>
        <p><strong>Activity Level:</strong> {profile.activity_level.title()}</p>
        <p><strong>Student:</strong> {'Yes' if profile.is_student else 'No'}</p>
        <p><strong>Mess Access:</strong> {'Yes' if profile.has_mess_access else 'No'}</p>
        <p><strong>Weekly Budget:</strong> ₹{profile.weekly_budget}</p>
        </div>
        """
        st.markdown(info_html, unsafe_allow_html=True)
    
    # Medical info
    st.markdown("### Medical Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="card">
        <p><strong>Medical Conditions:</strong></p>
        {''.join(f'<p>• {condition}</p>' for condition in profile.medical_conditions) if profile.medical_conditions else '<p>None reported</p>'}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="card">
        <p><strong>Allergies:</strong></p>
        {''.join(f'<p>• {allergy}</p>' for allergy in profile.allergies) if profile.allergies else '<p>None reported</p>'}
        </div>
        """, unsafe_allow_html=True)
    
    # Edit profile option
    st.markdown("### Profile Management")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✏️ Create New Profile", use_container_width=True):
            st.info("Creating a new profile will replace your current one.")
            if st.button("✅ Confirm New Profile", use_container_width=True):
                del st.session_state.user_id
                st.rerun()
    
    with col2:
        if st.button("📊 View Health Stats", use_container_width=True):
            st.session_state.show_stats = True
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
    
    **👤 Profile Management**
    - Create detailed health profiles
    - Track BMI and health metrics
    - Store medical conditions and allergies
    
    **📝 Daily Health Tracking**
    - Record sleep, meals, and activity
    - Track energy, mood, and stress levels
    - Monitor progress over time
    
    **🍽️ Diet & Nutrition**
    - Personalized meal plans based on your diet type
    - Grocery shopping lists and recipes
    - Calorie and macronutrient targets
    
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
    - **Data Storage**: Local JSON files
    - **Visualization**: Plotly for charts and graphs
    
    ### ⚠️ Important Disclaimer
    This application provides health and fitness suggestions based on AI analysis. 
    It is **NOT** a substitute for professional medical advice, diagnosis, or treatment. 
    Always consult qualified healthcare providers for medical concerns.
    
    ### 📞 Support & Feedback
    For issues, suggestions, or feedback:
    - Check the [GitHub repository](#) for updates
    - Report issues through the issue tracker
    - Contact: support@healthai.com
    
    ### 📄 License
    This software is provided for educational and personal use.
    """)

# ========== RUN THE APP ==========
if __name__ == "__main__":
    # Check for required environment variable
    if not os.getenv("GROQ_API_KEY"):
        st.error("""
        ⚠️ **GROQ_API_KEY not found!**
        
        Please create a `.env` file in the same directory with:
        ```
        GROQ_API_KEY=your_groq_api_key_here
        ```
        
        Get your API key from: https://console.groq.com/keys
        """)
        st.stop()
    
    try:
        main()
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

        st.info("Please refresh the page or check your connection.")


