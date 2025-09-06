import pandas as pd
import requests
import json
import os
import time
import logging
from typing import Optional, Dict, Any, List, Tuple, Union
import pickle
from pathlib import Path
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import signal
import sys

# =============================================================================
# CONFIGURATION - Edit these values directly
# =============================================================================
API_KEY = "s2_8fe4b2ba5e984913af78aea198072d70"  # Replace with your actual Abacus AI API key
BASE_URL = "https://routellm.abacus.ai/v1"  # Abacus AI API base URL
MAX_WORKERS = 10  # Number of parallel threads (adjust based on API limits)
# =============================================================================

class ParallelTorturedPhraseAnalyzer:
    def __init__(self, api_key: str, base_url: str = "https://api.abacus.ai/v1", max_workers: int = 10):
        """
        Initialize the analyzer with API credentials and setup logging.
        
        Args:
            api_key: Abacus AI API key
            base_url: Base URL for the API
            max_workers: Number of parallel threads (default: 10)
        """
        self.api_key = api_key
        self.base_url = base_url
        self.max_workers = max_workers
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('phrase_analysis.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Progress tracking
        self.progress_dir = Path("analysis_progress")
        self.progress_dir.mkdir(exist_ok=True)
        self.progress_file = self.progress_dir / "progress.pkl"
        self.results_file = self.progress_dir / "partial_results.pkl"
        self.lock = threading.Lock()
        
        # Graceful shutdown handling
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Rate limiting
        self.request_times = Queue()
        self.rate_limit_lock = threading.Lock()
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info("Shutdown signal received. Finishing current tasks...")
        self.shutdown_requested = True
    
    def _rate_limit(self):
        """Implement rate limiting to avoid overwhelming the API."""
        with self.rate_limit_lock:
            current_time = time.time()
            
            # Remove requests older than 1 second
            while not self.request_times.empty():
                if current_time - self.request_times.queue[0] > 1.0:
                    self.request_times.get()
                else:
                    break
            
            # If we have too many recent requests, wait
            if self.request_times.qsize() >= self.max_workers:
                sleep_time = 1.0 - (current_time - self.request_times.queue[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            self.request_times.put(current_time)

    def analyze_phrase_comprehensive(self, phrase_data: Tuple[int, str]) -> Dict[str, Any]:
        """
        Analyze a phrase using GPT-5 to get comprehensive information.
        
        Args:
            phrase_data: Tuple of (index, phrase)
            
        Returns:
            Dictionary containing all analysis fields including index
        """
        idx, phrase = phrase_data
        
        if self.shutdown_requested:
            return None
        
        # Apply rate limiting
        self._rate_limit()
        
        prompt = f"""
        Analyze this potentially problematic phrase from academic/technical writing: "{phrase}"
        
        Provide a JSON response with the following fields:
        1. "severity": integer (0=High problem, 1=Medium problem, 2=Low problem)
        2. "feedback": string (concise rewrite suggestion)
        3. "tags": list of strings (categories like "absolute", "hedging", "compliance", "risk", "clarity", "redundancy", "jargon")
        4. "weight": integer 1-10 (priority for fixing, 10=highest priority)
        5. "example_context": string (one short realistic example of how this phrase might be used)
        
        Consider:
        - Clarity and readability issues
        - Academic writing best practices
        - Potential for misunderstanding
        - Compliance with clear communication standards
        
        Respond only with valid JSON.
        """
        
        payload = {
            "model": "gpt-5",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert in academic writing and clear communication. Analyze phrases for clarity issues and provide structured feedback."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 500
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content'].strip()
            
            # Parse JSON response
            try:
                analysis = json.loads(content)
                
                # Validate and set defaults
                return {
                    'index': idx,
                    'original_phrase': phrase,
                    'severity': int(analysis.get('severity', 1)),
                    'feedback': str(analysis.get('feedback', 'Consider rephrasing for clarity')),
                    'tags': '|'.join(analysis.get('tags', ['unclear'])),
                    'weight': int(analysis.get('weight', 5)),
                    'enabled': True,
                    'source_version': 'v1.0',
                    'example_context': str(analysis.get('example_context', '')),
                    'locale_lang': 'en',
                    'regex_flag': False
                }
                
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse JSON for phrase '{phrase}': {e}")
                return self._get_fallback_analysis(idx, phrase)
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed for phrase '{phrase}': {e}")
            return self._get_fallback_analysis(idx, phrase)
        except Exception as e:
            self.logger.error(f"Unexpected error analyzing phrase '{phrase}': {e}")
            return self._get_fallback_analysis(idx, phrase)
    
    def _get_fallback_analysis(self, idx: int, phrase: str) -> Dict[str, Any]:
        """
        Provide fallback analysis when API fails.
        
        Args:
            idx: Index of the phrase
            phrase: The phrase to analyze
            
        Returns:
            Dictionary with default analysis values
        """
        return {
            'index': idx,
            'original_phrase': phrase,
            'severity': 1,  # Medium severity as default
            'feedback': f'Consider rephrasing "{phrase}" for better clarity',
            'tags': 'unclear|needs_review',
            'weight': 5,
            'enabled': True,
            'source_version': 'v1.0_fallback',
            'example_context': f'Example usage: "{phrase}" appears in technical documentation',
            'locale_lang': 'en',
            'regex_flag': False
        }
    
    def save_progress(self, processed_indices: set, results: Dict[int, Dict]):
        """
        Save current progress to disk.
        
        Args:
            processed_indices: Set of processed phrase indices
            results: Dictionary mapping indices to results
        """
        with self.lock:
            progress_data = {
                'processed_indices': processed_indices,
                'total_processed': len(processed_indices),
                'timestamp': datetime.now().isoformat()
            }
            
            # Save progress metadata
            with open(self.progress_file, 'wb') as f:
                pickle.dump(progress_data, f)
            
            # Save partial results
            with open(self.results_file, 'wb') as f:
                pickle.dump(results, f)
            
            self.logger.info(f"Progress saved: {len(processed_indices)} phrases processed")
    
    def load_progress(self) -> Tuple[set, Dict[int, Dict]]:
        """
        Load previous progress if available.
        
        Returns:
            Tuple of (processed_indices_set, results_dict)
        """
        if self.progress_file.exists() and self.results_file.exists():
            try:
                with open(self.progress_file, 'rb') as f:
                    progress_data = pickle.load(f)
                
                with open(self.results_file, 'rb') as f:
                    results = pickle.load(f)
                
                processed_indices = progress_data['processed_indices']
                self.logger.info(f"Resuming from previous progress: {len(processed_indices)} phrases already processed")
                return processed_indices, results
                
            except Exception as e:
                self.logger.warning(f"Could not load previous progress: {e}")
                return set(), {}
        
        return set(), {}
    
    def process_csv_parallel(self, input_file: str, output_file: str, save_interval: int = 50):
        """
        Process the CSV file with parallel analysis.
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file
            save_interval: Save progress every N completed phrases
        """
        try:
            # Read the input CSV
            df = pd.read_csv(input_file)
            total_phrases = len(df)
            self.logger.info(f"Loaded CSV with {total_phrases} phrases")
            
            # Check if we have previous progress
            processed_indices, results = self.load_progress()
            
            if processed_indices:
                self.logger.info(f"Resuming from previous session: {len(processed_indices)} phrases already processed")
            
            # Create list of phrases to process (skip already processed ones)
            phrases_to_process = []
            for idx, row in df.iterrows():
                if idx not in processed_indices:
                    phrases_to_process.append((idx, row['Tortured_Phrases']))
            
            if not phrases_to_process:
                self.logger.info("All phrases already processed!")
                self._create_output_csv(results, output_file)
                return
            
            self.logger.info(f"Processing {len(phrases_to_process)} remaining phrases with {self.max_workers} workers")
            
            # Process phrases in parallel
            completed_count = len(processed_indices)
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_phrase = {
                    executor.submit(self.analyze_phrase_comprehensive, phrase_data): phrase_data 
                    for phrase_data in phrases_to_process
                }
                
                # Process completed tasks
                for future in as_completed(future_to_phrase):
                    if self.shutdown_requested:
                        self.logger.info("Shutdown requested, cancelling remaining tasks...")
                        executor.shutdown(wait=False)
                        break
                    
                    try:
                        result = future.result()
                        if result is not None:
                            results[result['index']] = result
                            processed_indices.add(result['index'])
                            completed_count += 1
                            
                            # Log progress
                            if completed_count % 10 == 0:
                                progress_pct = (completed_count / total_phrases) * 100
                                self.logger.info(f"Progress: {completed_count}/{total_phrases} ({progress_pct:.1f}%)")
                            
                            # Save progress periodically
                            if completed_count % save_interval == 0:
                                self.save_progress(processed_indices, results)
                    
                    except Exception as e:
                        phrase_data = future_to_phrase[future]
                        self.logger.error(f"Error processing phrase {phrase_data[1]}: {e}")
            
            # Final save
            self.save_progress(processed_indices, results)
            
            # Create output CSV
            self._create_output_csv(results, output_file)
            
            # Clean up progress files after successful completion
            if not self.shutdown_requested and len(results) == total_phrases:
                if self.progress_file.exists():
                    self.progress_file.unlink()
                if self.results_file.exists():
                    self.results_file.unlink()
                self.logger.info("Analysis completed successfully! Progress files cleaned up.")
            
        except Exception as e:
            self.logger.error(f"Error processing CSV: {e}")
            raise
    
    def _create_output_csv(self, results: Dict[int, Dict], output_file: str):
        """Create the output CSV from results."""
        # Sort results by original index
        sorted_results = [results[idx] for idx in sorted(results.keys())]
        
        # Remove the index field from results for CSV output
        for result in sorted_results:
            result.pop('index', None)
        
        # Create output DataFrame
        output_df = pd.DataFrame(sorted_results)
        
        # Save to CSV
        output_df.to_csv(output_file, index=False)
        self.logger.info(f"Results saved to {output_file}")
        
        # Print summary statistics
        print("\n=== Analysis Summary ===")
        print(f"Total phrases analyzed: {len(output_df)}")
        print(f"Severity distribution:")
        severity_counts = output_df['severity'].value_counts().sort_index()
        for severity, count in severity_counts.items():
            severity_name = {0: 'High', 1: 'Medium', 2: 'Low'}[severity]
            print(f"  {severity_name} ({severity}): {count} phrases")

def main():
    """
    Main function to run the parallel analysis.
    """
    # Get API key - first try environment variable, then use configured value
    api_key = os.getenv('ABACUS_API_KEY', API_KEY)
    if not api_key or api_key == "your_api_key_here":
        print("Error: Please set your API key!")
        print("Option 1: Edit API_KEY variable at the top of this script")
        print("Option 2: Set ABACUS_API_KEY environment variable")
        print("You can get your API key from: https://abacus.ai/app/route-llm-apis")
        return
    
    # Configuration - use environment variable or configured value
    max_workers = int(os.getenv('MAX_WORKERS', str(MAX_WORKERS)))
    
    print(f"Starting parallel analysis with {max_workers} workers")
    print("You can adjust the number of workers by setting MAX_WORKERS environment variable")
    print("Recommended: Start with 10 workers and adjust based on API rate limits")
    
    # Initialize analyzer
    analyzer = ParallelTorturedPhraseAnalyzer(api_key, base_url=BASE_URL, max_workers=max_workers)
    
    # File paths
    input_file = "Tortured_Phrases_Lexicon_2.csv"
    output_file = f"enhanced_tortured_phrases_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print("Progress will be saved automatically. You can safely interrupt and resume.")
    print("Press Ctrl+C to gracefully stop the analysis.")
    
    try:
        # Process the CSV
        analyzer.process_csv_parallel(input_file, output_file, save_interval=25)
        
        print(f"\nAnalysis complete! Results saved to: {output_file}")
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted gracefully. Progress has been saved.")
        print("Run the script again to resume from where you left off.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()