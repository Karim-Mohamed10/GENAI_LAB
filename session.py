import uuid
from loguru import logger

class Session:
    """
    Context manager for setting the current session id.

    can be use it like this
        ```
            with Session("exploring-web") as s:
                print(s.session_id)
        ```
    """

    def __init__(self, session_id: str = None):
        """
        Initialize Session with a session_id (auto-generate if None).
        """
        #TODO: set self.session_id and use uuid if None
        raise NotImplementedError()
    
    def __enter__(self):
        logger.info(f"Entering session: {self.session_id}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info(f"Exiting session: {self.session_id}")
        if exc_type:
            logger.error(f"An exception occurred: {exc_val}")
        # Return False to propagate exception (if any)
        return False
