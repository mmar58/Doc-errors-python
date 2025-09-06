"""
Progress tracking system for long-running PDF analysis tasks
"""
import asyncio
import json
import time
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum

class ProgressStatus(Enum):
    STARTED = "started"
    EXTRACTING_TEXT = "extracting_text"
    MATCHING_PHRASES = "matching_phrases"
    ANALYZING_SINGLE_WORDS = "analyzing_single_words"
    QUERYING_MODELS = "querying_models"
    DISCOVERING_PHRASES = "discovering_phrases"
    MERGING_RESULTS = "merging_results"
    GENERATING_REPORT = "generating_report"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ProgressUpdate:
    job_id: str
    status: ProgressStatus
    message: str
    progress_percent: int
    current_step: str
    total_steps: int
    current_step_number: int
    details: Optional[str] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self):
        """Convert to dictionary with JSON-serializable values"""
        return {
            "job_id": self.job_id,
            "status": self.status.value,  # Convert enum to string
            "message": self.message,
            "progress_percent": self.progress_percent,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "current_step_number": self.current_step_number,
            "details": self.details,
            "timestamp": self.timestamp
        }

class ProgressTracker:
    def __init__(self):
        self._progress: Dict[str, ProgressUpdate] = {}
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}
    
    def get_progress(self, job_id: str) -> Optional[ProgressUpdate]:
        """Get current progress for a job"""
        return self._progress.get(job_id)
    
    def subscribe(self, job_id: str) -> asyncio.Queue:
        """Subscribe to progress updates for a job"""
        queue = asyncio.Queue()
        if job_id not in self._subscribers:
            self._subscribers[job_id] = []
        self._subscribers[job_id].append(queue)
        return queue
    
    def unsubscribe(self, job_id: str, queue: asyncio.Queue):
        """Unsubscribe from progress updates"""
        if job_id in self._subscribers:
            try:
                self._subscribers[job_id].remove(queue)
                if not self._subscribers[job_id]:
                    del self._subscribers[job_id]
            except ValueError:
                pass
    
    async def update_progress(self, update: ProgressUpdate):
        """Update progress and notify subscribers"""
        self._progress[update.job_id] = update
        
        # Notify all subscribers
        if update.job_id in self._subscribers:
            for queue in self._subscribers[update.job_id][:]:  # Copy list to avoid modification during iteration
                try:
                    await queue.put(update)
                except Exception as e:
                    print(f"[progress] Failed to notify subscriber: {e}")
                    # Remove failed subscriber
                    try:
                        self._subscribers[update.job_id].remove(queue)
                    except ValueError:
                        pass
    
    def cleanup_job(self, job_id: str):
        """Clean up progress data for completed job"""
        # Keep progress data for a while, but clean up subscribers
        if job_id in self._subscribers:
            del self._subscribers[job_id]

# Global progress tracker instance
progress_tracker = ProgressTracker()

# Progress step definitions
ANALYSIS_STEPS = [
    (ProgressStatus.EXTRACTING_TEXT, "Extracting text from PDF", 10),
    (ProgressStatus.MATCHING_PHRASES, "Matching phrases from lexicon", 25),
    (ProgressStatus.ANALYZING_SINGLE_WORDS, "Analyzing single words with AI", 45),
    (ProgressStatus.QUERYING_MODELS, "Getting AI feedback on matches", 65),
    (ProgressStatus.DISCOVERING_PHRASES, "Discovering additional issues", 80),
    (ProgressStatus.MERGING_RESULTS, "Merging and prioritizing findings", 90),
    (ProgressStatus.GENERATING_REPORT, "Generating PDF report", 95),
    (ProgressStatus.COMPLETED, "Analysis complete", 100)
]

def get_step_info(status: ProgressStatus) -> tuple[str, int, int]:
    """Get step information for a status"""
    for i, (step_status, message, percent) in enumerate(ANALYSIS_STEPS):
        if step_status == status:
            return message, percent, i + 1
    return "Processing", 0, 1

async def update_job_progress(job_id: str, status: ProgressStatus, details: Optional[str] = None):
    """Helper function to update job progress"""
    message, percent, step_num = get_step_info(status)
    
    update = ProgressUpdate(
        job_id=job_id,
        status=status,
        message=message,
        progress_percent=percent,
        current_step=message,
        total_steps=len(ANALYSIS_STEPS),
        current_step_number=step_num,
        details=details
    )
    
    print(f"[progress_tracker] Job {job_id}: Step {step_num}/{len(ANALYSIS_STEPS)} - {message} ({percent}%)")
    if details:
        print(f"[progress_tracker] Details: {details}")
    
    await progress_tracker.update_progress(update)
