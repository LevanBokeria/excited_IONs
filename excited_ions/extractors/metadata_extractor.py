"""Metadata extractor using LLMs."""

import json
from typing import Dict, Any
from datetime import datetime
from . import BaseLLMExtractor
from ..models import ExtractedData, ExperimentalResult, ExperimentalCondition


class MetadataExtractor(BaseLLMExtractor):
    """Extracts experimental metadata and results from paper abstracts and content."""
    
    async def extract(self, paper: Dict[str, Any]) -> ExtractedData:
        """Extract experimental data and metadata from a paper."""
        
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')
        content = paper.get('content', '')
        
        # Combine available text
        text_to_analyze = f"Title: {title}\n\nAbstract: {abstract}"
        if content and content != abstract:
            text_to_analyze += f"\n\nContent: {content[:5000]}"  # Limit content length
        
        system_prompt = """You are a scientific research expert specializing in experimental data extraction. 
        Your task is to extract structured experimental information from scientific papers. 
        Focus on identifying:
        1. Experimental results with specific numerical values and units
        2. Experimental conditions and parameters
        3. Methodologies used
        4. Materials and samples
        5. Key findings and conclusions
        6. Study limitations
        
        Return your analysis in JSON format with the following structure:
        {
            "experimental_results": [
                {
                    "measurement": "description of what was measured",
                    "value": "numerical value or description",
                    "unit": "unit of measurement if applicable",
                    "uncertainty": "uncertainty value if provided",
                    "method": "measurement method if specified",
                    "conditions": [
                        {
                            "parameter": "condition parameter",
                            "value": "condition value",
                            "unit": "condition unit"
                        }
                    ]
                }
            ],
            "methodologies": ["method 1", "method 2"],
            "materials": ["material 1", "material 2"],
            "key_findings": ["finding 1", "finding 2"],
            "limitations": ["limitation 1", "limitation 2"]
        }"""
        
        user_prompt = f"""Please analyze the following scientific paper and extract experimental data:

        {text_to_analyze}
        
        Extract the experimental results, methodologies, materials, key findings, and limitations as structured data."""
        
        try:
            response = await self._call_llm(user_prompt, system_prompt)
            
            # Parse JSON response
            extracted_json = json.loads(response)
            
            # Convert to ExtractedData model
            experimental_results = []
            for result_data in extracted_json.get('experimental_results', []):
                conditions = []
                for cond_data in result_data.get('conditions', []):
                    conditions.append(ExperimentalCondition(**cond_data))
                
                result = ExperimentalResult(
                    measurement=result_data.get('measurement', ''),
                    value=result_data.get('value', ''),
                    unit=result_data.get('unit'),
                    uncertainty=result_data.get('uncertainty'),
                    method=result_data.get('method'),
                    conditions=conditions
                )
                experimental_results.append(result)
            
            return ExtractedData(
                experimental_results=experimental_results,
                methodologies=extracted_json.get('methodologies', []),
                materials=extracted_json.get('materials', []),
                key_findings=extracted_json.get('key_findings', []),
                limitations=extracted_json.get('limitations', [])
            )
            
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response as JSON: {e}")
            # Return empty ExtractedData if parsing fails
            return ExtractedData()
        except Exception as e:
            print(f"Error in metadata extraction: {e}")
            return ExtractedData()