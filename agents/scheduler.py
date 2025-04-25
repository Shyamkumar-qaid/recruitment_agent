from crewai import Agent
from typing import Dict, Any, List, Optional
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from models import Session, AuditLog

load_dotenv()

class Scheduler:
    def __init__(self):
        self.agent = Agent(
            role='Interview Scheduler',
            goal='Coordinate and manage interview scheduling efficiently',
            backstory='Expert in calendar management and scheduling optimization',
            verbose=True,
            allow_delegation=False,
            # Ensure Ollama server is running and the model is pulled (e.g. ollama run phi3)
            llm=Ollama(model="phi3") # Replace with your desired Ollama model
        )
    
    def schedule_interview(self, candidate_id: int, interviewer_ids: List[int],
                         interview_type: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point for scheduling an interview."""
        session = Session()
        try:
            # Get availability data
            interviewer_slots = self._get_interviewer_availability(interviewer_ids)
            candidate_slots = self._get_candidate_availability(candidate_id)
            
            # Find common slots
            common_slots = self._find_common_slots(
                interviewer_slots,
                candidate_slots,
                constraints.get('duration_minutes', 60)
            )
            
            if not common_slots:
                return {
                    'status': 'no_slots',
                    'message': 'No common time slots found'
                }
            
            # Auto-book or propose slots based on strategy
            if constraints.get('auto_book', False):
                result = self._auto_book_slot(common_slots[0], candidate_id, interviewer_ids, interview_type)
            else:
                result = {
                    'status': 'slots_found',
                    'available_slots': common_slots[:5]  # Limit to top 5 slots
                }
            
            # Log scheduling attempt
            self._log_scheduling(session, candidate_id, 'schedule_attempt', result)
            
            return result
            
        except Exception as e:
            self._log_scheduling(
                session,
                candidate_id,
                'schedule_error',
                {'error': str(e)}
            )
            return {
                'status': 'error',
                'error': str(e)
            }
        finally:
            session.close()
    
    def confirm_booking(self, candidate_id: int, slot: datetime,
                       interviewer_ids: List[int], interview_type: str) -> Dict[str, Any]:
        """Confirm a proposed interview slot."""
        session = Session()
        try:
            # Verify slot is still available
            if not self._verify_slot_availability(slot, interviewer_ids):
                return {
                    'status': 'slot_taken',
                    'message': 'Selected time slot is no longer available'
                }
            
            # Book the meeting
            booking = self._book_meeting(slot, candidate_id, interviewer_ids, interview_type)
            
            # Log successful booking
            self._log_scheduling(session, candidate_id, 'booking_confirmed', booking)
            
            return {
                'status': 'success',
                'booking': booking
            }
            
        except Exception as e:
            self._log_scheduling(
                session,
                candidate_id,
                'booking_error',
                {'error': str(e)}
            )
            return {
                'status': 'error',
                'error': str(e)
            }
        finally:
            session.close()
    
    def _get_interviewer_availability(self, interviewer_ids: List[int]) -> Dict[int, List[datetime]]:
        """Fetch availability for all interviewers."""
        # Here you would implement calendar API integration
        return {}  # Placeholder implementation
    
    def _get_candidate_availability(self, candidate_id: int) -> List[datetime]:
        """Fetch candidate's provided availability."""
        # Here you would implement availability retrieval
        return []  # Placeholder implementation
    
    def _find_common_slots(self, interviewer_slots: Dict[int, List[datetime]],
                          candidate_slots: List[datetime], duration_minutes: int) -> List[datetime]:
        """Find mutually available time slots."""
        # Here you would implement slot matching logic
        return []  # Placeholder implementation
    
    def _verify_slot_availability(self, slot: datetime, interviewer_ids: List[int]) -> bool:
        """Verify a time slot is still available."""
        # Here you would implement real-time availability check
        return True  # Placeholder implementation
    
    def _auto_book_slot(self, slot: datetime, candidate_id: int,
                       interviewer_ids: List[int], interview_type: str) -> Dict[str, Any]:
        """Automatically book the first available slot."""
        return self._book_meeting(slot, candidate_id, interviewer_ids, interview_type)
    
    def _book_meeting(self, slot: datetime, candidate_id: int,
                     interviewer_ids: List[int], interview_type: str) -> Dict[str, Any]:
        """Book the meeting in calendar system."""
        # Here you would implement calendar event creation
        return {
            'meeting_id': 'placeholder_id',
            'datetime': slot.isoformat(),
            'duration_minutes': 60,
            'meeting_link': 'https://meet.example.com/placeholder',
            'calendar_event_ids': {'interviewer_1': 'event_id_1'}
        }
    
    def _log_scheduling(self, session: Session, candidate_id: int,
                       action: str, details: Dict[str, Any]) -> None:
        """Log scheduling activity in audit trail."""
        try:
            audit = AuditLog(
                candidate_id=candidate_id,
                action=f'scheduling_{action}',
                details=details,
                timestamp=datetime.utcnow()
            )
            session.add(audit)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error logging scheduling: {str(e)}")