"""
Example library manager for few-shot learning.

Stores verified correct extraction results with their images.
Claude can reference these examples to improve accuracy on new screenshots.
"""

import json
import base64
import logging
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class ExampleLibrary:
    """Manages verified extraction examples for few-shot prompting."""
    
    def __init__(self, library_dir: str = None):
        """
        Initialize example library.
        
        Args:
            library_dir: Directory to store examples. Defaults to ./examples/
        """
        if library_dir is None:
            library_dir = Path(__file__).parent.parent / "examples"
        
        self.library_dir = Path(library_dir)
        self.library_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ExampleLibrary initialized at {self.library_dir}")
    
    def save_example(
        self,
        screenshot_base64: str,
        extraction_result: dict,
        tags: List[str] = None,
        notes: str = None
    ) -> str:
        """
        Save a verified extraction example.
        
        Args:
            screenshot_base64: Base64-encoded screenshot
            extraction_result: Verified correct extraction JSON
            tags: Optional tags (e.g., ["glare", "partial_card", "high_kills"])
            notes: Optional notes about the example
            
        Returns:
            Example ID (timestamp-based)
        """
        example_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        example_dir = self.library_dir / example_id
        example_dir.mkdir(parents=True, exist_ok=True)
        
        # Save screenshot
        screenshot_path = example_dir / "screenshot.jpg"
        with open(screenshot_path, "wb") as f:
            f.write(base64.b64decode(screenshot_base64))
        
        # Save extraction result
        result_path = example_dir / "result.json"
        with open(result_path, "w") as f:
            json.dump(extraction_result, f, indent=2)
        
        # Save metadata
        metadata = {
            "example_id": example_id,
            "timestamp": datetime.now().isoformat(),
            "tags": tags or [],
            "notes": notes or "",
            "screenshot_size_bytes": len(base64.b64decode(screenshot_base64)),
            "num_teams": len(extraction_result.get("teams", [])),
            "total_kills": sum(
                t.get("total_kills", 0) 
                for t in extraction_result.get("teams", [])
            )
        }
        
        metadata_path = example_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved example {example_id} with tags: {tags}")
        return example_id
    
    def get_example(self, example_id: str) -> Optional[Dict]:
        """
        Retrieve a specific example.
        
        Args:
            example_id: Example ID
            
        Returns:
            Dict with screenshot_base64, result, metadata
        """
        example_dir = self.library_dir / example_id
        
        if not example_dir.exists():
            logger.warning(f"Example {example_id} not found")
            return None
        
        # Load screenshot
        screenshot_path = example_dir / "screenshot.jpg"
        with open(screenshot_path, "rb") as f:
            screenshot_base64 = base64.b64encode(f.read()).decode("utf-8")
        
        # Load result
        result_path = example_dir / "result.json"
        with open(result_path, "r") as f:
            result = json.load(f)
        
        # Load metadata
        metadata_path = example_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        return {
            "example_id": example_id,
            "screenshot_base64": screenshot_base64,
            "result": result,
            "metadata": metadata
        }
    
    def find_examples_by_tag(self, tag: str, limit: int = 2) -> List[Dict]:
        """
        Find examples by tag (e.g., "glare", "partial_card").
        
        Args:
            tag: Tag to search for
            limit: Max number of examples to return
            
        Returns:
            List of matching examples
        """
        examples = []
        
        for example_dir in sorted(self.library_dir.iterdir(), reverse=True):
            if not example_dir.is_dir():
                continue
            
            metadata_path = example_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            if tag in metadata.get("tags", []):
                example = self.get_example(example_dir.name)
                if example:
                    examples.append(example)
                    if len(examples) >= limit:
                        break
        
        logger.info(f"Found {len(examples)} examples with tag '{tag}'")
        return examples
    
    def list_all_examples(self) -> List[Dict]:
        """
        List all examples with metadata.
        
        Returns:
            List of example metadata
        """
        examples = []
        
        for example_dir in sorted(self.library_dir.iterdir(), reverse=True):
            if not example_dir.is_dir():
                continue
            
            metadata_path = example_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                examples.append(metadata)
        
        return examples
    
    def build_few_shot_prompt(self, tags: List[str] = None, limit: int = 2) -> str:
        """
        Build a few-shot prompt section with relevant examples.
        
        Args:
            tags: Tags to filter examples (e.g., ["glare", "partial_card"])
            limit: Max examples to include
            
        Returns:
            Prompt section with examples
        """
        examples = []
        
        if tags:
            # Find examples matching any of the tags
            for tag in tags:
                found = self.find_examples_by_tag(tag, limit=1)
                examples.extend(found)
                if len(examples) >= limit:
                    break
        else:
            # Get most recent examples
            all_examples = self.list_all_examples()[:limit]
            for metadata in all_examples:
                example = self.get_example(metadata["example_id"])
                if example:
                    examples.append(example)
        
        if not examples:
            return ""
        
        prompt = "\n\n=== REFERENCE EXAMPLES ===\n"
        
        for i, example in enumerate(examples, 1):
            prompt += f"\nEXAMPLE {i}:\n"
            prompt += f"Tags: {', '.join(example['metadata'].get('tags', []))}\n"
            prompt += f"Notes: {example['metadata'].get('notes', 'N/A')}\n"
            prompt += f"Teams: {example['metadata']['num_teams']}, Total Kills: {example['metadata']['total_kills']}\n"
            prompt += f"\nExpected output:\n"
            prompt += json.dumps(example["result"], indent=2)
        
        prompt += "\n=== END EXAMPLES ===\n"
        return prompt
    
    def delete_example(self, example_id: str) -> bool:
        """
        Delete an example.
        
        Args:
            example_id: Example ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        example_dir = self.library_dir / example_id
        
        if not example_dir.exists():
            logger.warning(f"Example {example_id} not found")
            return False
        
        import shutil
        shutil.rmtree(example_dir)
        logger.info(f"Deleted example {example_id}")
        return True
