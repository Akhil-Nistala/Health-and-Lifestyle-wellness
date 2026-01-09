# agent_bridge.py - CORRECTED VERSION
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import pickle
from datetime import datetime
import hashlib

# Try to import agent system
try:
    # Add the agent system to path if needed
    sys.path.append(str(Path(__file__).parent))
    
    from lifestyle_final import (
        HealthAISystem, UserProfile, DailyLog, MessMenu,
        DietType, DatabaseManager as AgentDatabaseManager,
        Config as AgentConfig, TAOAgent, LLMService
    )
    AGENT_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Agent system not available: {e}")
    AGENT_SYSTEM_AVAILABLE = False

class AgentBridge:
    """Bridge between Streamlit app and agent system"""
    
    def __init__(self, streamlit_db=None):
        self.agent_system = None
        self.streamlit_db = streamlit_db
        
        if AGENT_SYSTEM_AVAILABLE:
            try:
                # Do NOT set os.environ here - let the agent system read from environment
                # Create temporary directory for agent data
                agent_data_dir = Path("agent_data")
                agent_data_dir.mkdir(exist_ok=True)
                AgentConfig.DATA_DIR = agent_data_dir
                
                # Initialize agent system in WEB mode
                self.agent_system = HealthAISystem(execution_mode="web")  # FIXED: Set web mode
                print("✅ Agent system initialized in WEB mode")
            except Exception as e:
                print(f"❌ Failed to initialize agent system: {e}")
                self.agent_system = None
    
    def is_available(self) -> bool:
        """Check if agent system is available"""
        return self.agent_system is not None and hasattr(self.agent_system, 'db')
    
    def _sync_data_to_agent_db(self, profile: Dict, logs: List[Dict]) -> bool:
        """Synchronize Streamlit data to agent database before analysis"""
        if not self.is_available():
            return False
        
        try:
            # 1. Convert and save profile to agent DB
            agent_profile = self.convert_to_agent_profile(profile)
            if not agent_profile:
                return False
            
            # Create agent UserProfile object
            profile_obj = UserProfile(
                user_id=agent_profile['user_id'],
                name=agent_profile['name'],
                age=agent_profile['age'],
                weight=agent_profile['weight'],
                height=agent_profile['height'],
                diet_type=agent_profile['diet_type'],
                fitness_goal=agent_profile['fitness_goal'],
                activity_level=agent_profile['activity_level'],
                is_student=agent_profile.get('is_student', False),
                has_mess_access=agent_profile.get('has_mess_access', False),
                weekly_budget=agent_profile.get('weekly_budget', 500.0),
                medical_conditions=agent_profile.get('medical_conditions', []),
                allergies=agent_profile.get('allergies', [])
            )
            
            # Save to agent DB
            self.agent_system.db.save_profile(profile_obj)
            
            # 2. Convert and save logs to agent DB
            for log_data in logs:
                try:
                    # Generate unique log ID based on date and user
                    log_date = log_data.get('log_date', log_data.get('date', ''))
                    if not log_date:
                        continue
                    
                    log_id = f"log_{log_date}_{profile['user_id']}"
                    
                    # Create agent DailyLog object
                    meals = log_data.get('meals', {})
                    if isinstance(meals, dict):
                        meals_dict = meals
                    else:
                        meals_dict = {
                            'breakfast': '',
                            'lunch': '',
                            'dinner': ''
                        }
                    
                    log_obj = DailyLog(
                        log_id=log_id,
                        user_id=profile['user_id'],
                        date=log_date,
                        day_of_week=datetime.strptime(log_date, "%Y-%m-%d").strftime("%A") 
                                   if log_date else "Unknown",
                        timestamp=datetime.now().isoformat(),
                        sleep_hours=log_data.get('sleep_hours', 7.0),
                        sleep_quality=log_data.get('sleep_quality', 5),
                        sleep_notes=log_data.get('sleep_notes', ''),
                        meals=meals_dict,
                        water_intake_liters=log_data.get('water_intake_liters', 2.0),
                        steps_count=log_data.get('steps_count', 0),
                        workout_duration_min=log_data.get('workout_duration_min', 0),
                        workout_type=log_data.get('workout_type', ''),
                        energy_level=log_data.get('energy_level', 5),
                        mood_level=log_data.get('mood_level', 5),
                        stress_level=log_data.get('stress_level', 5),
                        focus_level=log_data.get('focus_level', 5),
                        symptoms=log_data.get('symptoms', []),
                        notes=log_data.get('notes', '')
                    )
                    
                    # Save to agent DB
                    self.agent_system.db.save_log(log_obj)
                    
                except Exception as e:
                    print(f"⚠️ Could not sync log {log_date}: {str(e)[:100]}")
                    continue
            
            return True
            
        except Exception as e:
            print(f"❌ Data sync failed: {e}")
            return False
    
    def convert_to_agent_profile(self, streamlit_profile: Dict) -> Dict:
        """Convert Streamlit profile format to agent format"""
        try:
            # Map diet types
            diet_type_map = {
                "vegetarian": DietType.VEGETARIAN,
                "non_vegetarian": DietType.NON_VEGETARIAN,
                "non-vegetarian": DietType.NON_VEGETARIAN,
                "eggetarian": DietType.EGGETARIAN,
                "vegan": DietType.VEGAN
            }
            
            diet_type_str = streamlit_profile.get('diet_type', 'non_vegetarian').lower()
            diet_type = diet_type_map.get(diet_type_str, DietType.NON_VEGETARIAN)
            
            return {
                'user_id': streamlit_profile.get('user_id', ''),
                'name': streamlit_profile.get('name', 'User'),
                'age': streamlit_profile.get('age', 25),
                'weight': streamlit_profile.get('weight', 70.0),
                'height': streamlit_profile.get('height', 170.0),
                'diet_type': diet_type,
                'fitness_goal': streamlit_profile.get('fitness_goal', 'general health'),
                'activity_level': streamlit_profile.get('activity_level', 'moderate'),
                'is_student': streamlit_profile.get('is_student', False),
                'has_mess_access': streamlit_profile.get('has_mess_access', False),
                'weekly_budget': streamlit_profile.get('weekly_budget', 500.0),
                'medical_conditions': streamlit_profile.get('medical_conditions', []),
                'allergies': streamlit_profile.get('allergies', []),
                'bmi': streamlit_profile.get('bmi', 22.0),
                'bmi_category': streamlit_profile.get('bmi_category', 'Healthy')
            }
        except Exception as e:
            print(f"❌ Profile conversion error: {e}")
            return {}
    
    def convert_mess_menu(self, mess_menu: Dict) -> Dict:
        """Convert Streamlit mess menu to agent format"""
        try:
            return {
                'meal_time': 'lunch',  # Default, could be enhanced
                'items': mess_menu.get('breakfast', []) + 
                        mess_menu.get('lunch', []) + 
                        mess_menu.get('dinner', []),
                'notes': f"Breakfast: {mess_menu.get('breakfast_notes', '')} | "
                        f"Lunch: {mess_menu.get('lunch_notes', '')} | "
                        f"Dinner: {mess_menu.get('dinner_notes', '')}",
                'date': mess_menu.get('date', datetime.now().strftime('%Y-%m-%d'))
            }
        except Exception as e:
            print(f"❌ Mess menu conversion error: {e}")
            return {'items': [], 'notes': ''}
    
    def _normalize_metadata(self, agent_result: Dict) -> Dict:
        """Normalize agent metadata to match UI expectations"""
        if not isinstance(agent_result, dict):
            return {}
        
        # Extract metadata from different possible locations
        metadata = agent_result.get('metadata', {})
        reasoning = agent_result.get('reasoning_summary', agent_result.get('reasoning_trace', []))
        
        # Normalize reasoning trace format
        normalized_reasoning = []
        if isinstance(reasoning, list):
            for i, item in enumerate(reasoning):
                if isinstance(item, dict):
                    normalized_item = {
                        'phase': item.get('phase', 'unknown'),
                        'iteration': item.get('iteration', i),
                        'insight': item.get('insight') or 
                                   item.get('analysis') or 
                                   item.get('observation') or 
                                   item.get('learning') or 
                                   ''
                    }
                    normalized_reasoning.append(normalized_item)
        
        return {
            'iterations': agent_result.get('iterations', 0),
            'max_iterations': agent_result.get('max_iterations', 3),
            'confidence': agent_result.get('confidence', 0.5),
            'reasoning_trace': normalized_reasoning
        }
    
    def run_agent_analysis(self, agent_type: str, profile: Dict, logs: List[Dict], 
                          mess_menu: Optional[Dict] = None) -> Dict:
        """
        Run agent analysis and return results
        
        Returns:
            Dict with keys:
            - success: bool
            - output: str (the analysis text)
            - metadata: Dict with iteration_count, confidence, etc.
            - error: str (if failed)
        """
        if not self.is_available():
            return {
                'success': False,
                'error': 'Agent system not available',
                'output': '',
                'iterations': 0,
                'confidence': 0.0,
                'reasoning_trace': []
            }
        
        try:
            # CRITICAL FIX: Sync data from Streamlit to agent DB before analysis
            sync_success = self._sync_data_to_agent_db(profile, logs)
            if not sync_success:
                return {
                    'success': False,
                    'error': 'Failed to sync data to agent database',
                    'output': '',
                    'iterations': 0,
                    'confidence': 0.0,
                    'reasoning_trace': []
                }
            
            # Set current user for agent system
            self.agent_system.current_user_id = profile.get('user_id', '')
            
            # Run the appropriate analysis
            result = None
            
            if agent_type in ['diet', 'fitness', 'health']:
                # Use HealthAISystem for consistent agent execution
                result = self.agent_system.run_analysis(agent_type)
                
                # HealthAISystem.run_analysis() returns None in CLI mode
                # We need to simulate the agent execution
                if result is None:
                    # Fallback to direct TAOAgent execution
                    profile_dict = self.convert_to_agent_profile(profile)
                    logs_dict = []
                    for log in logs:
                        logs_dict.append({
                            'date': log.get('log_date', log.get('date', '')),
                            'sleep_hours': log.get('sleep_hours', 7.0),
                            'sleep_quality': log.get('sleep_quality', 5),
                            'energy_level': log.get('energy_level', 5),
                            'mood_level': log.get('mood_level', 5),
                            'stress_level': log.get('stress_level', 5),
                            'workout_duration_min': log.get('workout_duration_min', 0),
                            'steps_count': log.get('steps_count', 0),
                            'meals_eaten': log.get('meals_eaten', 0)
                        })
                    
                    # Initialize appropriate agent (in web mode)
                    llm_service = LLMService()
                    agent = TAOAgent(agent_type, llm_service, execution_mode="web")  # FIXED: Web mode
                    result = agent.run(profile_dict, logs_dict)
            
            elif agent_type == 'mess_optimizer' and mess_menu:
                # NOTE: Mess optimizer remains standalone because:
                # 1. It requires specific mess menu input not handled by HealthAISystem
                # 2. HealthAISystem.run_analysis() is designed for profile/logs only
                # 3. Mess optimizer has different prompt structure and constraints
                
                profile_dict = self.convert_to_agent_profile(profile)
                logs_dict = []
                for log in logs:
                    logs_dict.append({
                        'date': log.get('log_date', log.get('date', '')),
                        'sleep_hours': log.get('sleep_hours', 7.0),
                        'energy_level': log.get('energy_level', 5),
                        'steps_count': log.get('steps_count', 0)
                    })
                
                agent_menu = self.convert_mess_menu(mess_menu)
                
                # Run mess optimizer agent (in web mode)
                llm_service = LLMService()
                agent = TAOAgent("mess_optimizer", llm_service, execution_mode="web")  # FIXED: Web mode
                result = agent.run(profile_dict, logs_dict, mess_menu=agent_menu)
            else:
                return {
                    'success': False,
                    'error': f'Unknown agent type: {agent_type}',
                    'output': '',
                    'iterations': 0,
                    'confidence': 0.0,
                    'reasoning_trace': []
                }
            
            # Process and normalize result
            if result and isinstance(result, dict):
                success = result.get('success', False)
                output = result.get('output', '')
                metadata = self._normalize_metadata(result)
                
                return {
                    'success': success,
                    'output': output,
                    'metadata': metadata,
                    'iterations': result.get('iterations', 0),
                    'confidence': result.get('confidence', 0.5),
                    'reasoning_trace': result.get('reasoning_summary', []),
                    'error': result.get('error', '') if not success else ''
                }
            else:
                # Handle string or None results
                return {
                    'success': bool(result),
                    'output': str(result) if result else 'No output generated',
                    'metadata': {},
                    'iterations': 0,
                    'confidence': 0.0,
                    'reasoning_trace': [],
                    'error': 'Invalid result format from agent'
                }
                
        except Exception as e:
            print(f"❌ Agent analysis error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'output': f'Agent execution failed: {str(e)[:100]}',
                'iterations': 0,
                'confidence': 0.0,
                'reasoning_trace': []
            }