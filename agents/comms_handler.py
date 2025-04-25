from crewai import Agent
from langchain_community.llms import Ollama
from typing import Dict, Any, List, Optional
import os
from dotenv import load_dotenv
from datetime import datetime
from models import Session, AuditLog

load_dotenv()

class CommsHandler:
    def __init__(self):
        self.agent = Agent(
            role='Communication Coordinator',
            goal='Manage all candidate communications securely and efficiently',
            backstory='Expert in handling recruitment communications and scheduling',
            verbose=True,
            allow_delegation=False,
            # Ensure Ollama server is running and the model is pulled (e.g. ollama run phi3)
            llm=Ollama(model="phi3") # Replace with your desired Ollama model
        )
        
    def send_communication(self, candidate_id: int, comm_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Send communication to candidate based on type and context."""
        session = Session()
        try:
            # Select appropriate template and channel
            template = self._select_template(comm_type)
            channel = self._determine_channel(context.get('contact_info', {}))
            
            # Personalize message
            message = self._personalize_message(template, context)
            
            # Send message based on type
            if comm_type in ['rejection', 'update']:
                result = self._send_simple_notification(candidate_id, message, channel)
            elif comm_type == 'assessment':
                result = self._send_assessment_link(candidate_id, message, context.get('assessment_link'))
            elif comm_type == 'schedule_interview':
                result = self._handle_interview_scheduling(
                    candidate_id,
                    context.get('interviewer_ids', []),
                    context.get('interview_type'),
                    context.get('duration_minutes', 60)
                )
            else:
                raise ValueError(f"Unsupported communication type: {comm_type}")
            
            # Log communication
            self._log_communication(session, candidate_id, comm_type, result)
            
            return {
                'status': 'success',
                'details': result
            }
            
        except Exception as e:
            self._log_communication(
                session,
                candidate_id,
                comm_type,
                {'status': 'error', 'error': str(e)}
            )
            return {
                'status': 'error',
                'error': str(e)
            }
        finally:
            session.close()
    
    def _select_template(self, comm_type: str) -> str:
        """Select appropriate message template based on communication type."""
        templates = {
            'rejection': "We appreciate your interest, but we've decided to move forward with other candidates...",
            'update': "Thank you for your patience. We're currently processing your application...",
            'assessment': "Please complete the following technical assessment...",
            'schedule_interview': "We'd like to schedule an interview..."
        }
        return templates.get(comm_type, "")
    
    def _determine_channel(self, contact_info: Dict[str, str]) -> str:
        """Determine best communication channel based on available contact info."""
        if contact_info.get('email'):
            return 'email'
        elif contact_info.get('phone'):
            return 'sms'
        else:
            raise ValueError("No valid contact information available")
    
    def _personalize_message(self, template: str, context: Dict[str, Any]) -> str:
        """Inject context data into message template."""
        # Here you would implement template variable substitution
        return template  # Placeholder implementation
    
    def _send_simple_notification(self, candidate_id: int, message: str, channel: str) -> Dict[str, Any]:
        """Send a simple notification via specified channel."""
        # Here you would implement actual sending logic
        return {
            'sent': True,
            'channel': channel,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _send_assessment_link(self, candidate_id: int, message: str, assessment_link: Optional[str]) -> Dict[str, Any]:
        """Send assessment link to candidate."""
        if not assessment_link:
            raise ValueError("Assessment link not provided")
        
        # Here you would implement actual sending logic
        return {
            'sent': True,
            'assessment_link': assessment_link,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _handle_interview_scheduling(self, candidate_id: int, interviewer_ids: List[int],
                                   interview_type: str, duration_minutes: int) -> Dict[str, Any]:
        """Handle interview scheduling process."""
        # This would integrate with the Scheduler component
        return {
            'status': 'pending',
            'message': 'Scheduling request sent to Scheduler'
        }
    
    def _log_communication(self, session: Session, candidate_id: int,
                          comm_type: str, details: Dict[str, Any]) -> None:
        """Log communication attempt in audit trail."""
        try:
            audit = AuditLog(
                candidate_id=candidate_id,
                action=f'communication_{comm_type}',
                details=details,
                timestamp=datetime.utcnow()
            )
            session.add(audit)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error logging communication: {str(e)}")