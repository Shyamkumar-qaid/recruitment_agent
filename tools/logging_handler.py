import logging
import structlog
from sentry_sdk import init as sentry_init
from typing import Dict, Any

class EnterpriseLogger:
    """
    Enterprise-grade logging handler with structured logging and Sentry integration
    
    Features:
    - JSON formatting for production
    - Console output for development
    - Automated Sentry error reporting
    - Async log processing
    - Request context tracking
    """
    
    def __init__(self, env: str = 'development', sentry_dsn: str = None):
        self.env = env
        self.sentry_dsn = sentry_dsn
        self._configure_logging()

    def _configure_logging(self):
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
        ]

        if self.env == 'production':
            processors.extend([
                structlog.processors.dict_tracebacks,
                structlog.processors.JSONRenderer()
            ])
            logging.basicConfig(level=logging.INFO)
        else:
            processors.extend([
                structlog.dev.ConsoleRenderer()
            ])
            logging.basicConfig(level=logging.DEBUG)

        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(
                logging.INFO if self.env == 'production' else logging.DEBUG
            ),
            cache_logger_on_first_use=True,
        )

        if self.sentry_dsn:
            sentry_init(
                dsn=self.sentry_dsn,
                environment=self.env,
                traces_sample_rate=1.0,
                profiles_sample_rate=1.0,
            )

    def get_logger(self, name: str = None) -> structlog.BoundLogger:
        """
        Get a configured logger instance with context binding
        """
        logger = structlog.get_logger(name)
        return logger.new(
            service='resume-analyzer',
            environment=self.env,
            version='1.0.0'
        )

    @staticmethod
    def bind_context(**kwargs: Dict[str, Any]) -> None:
        """
        Bind common context variables to all subsequent log entries
        """
        structlog.contextvars.bind_contextvars(**kwargs)

# Example usage:
# logger = EnterpriseLogger(env='development').get_logger(__name__)
# logger.info("system_startup", component="database")