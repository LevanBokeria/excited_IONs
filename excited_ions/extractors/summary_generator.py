"""Summary generator using LLMs."""

from typing import Dict, Any, List
from datetime import datetime
from . import BaseLLMExtractor
from ..models import PaperSummary


class SummaryGenerator(BaseLLMExtractor):
    """Generates comprehensive summaries of scientific papers."""
    
    async def extract(self, paper: Dict[str, Any]) -> PaperSummary:
        """Generate a structured summary of a scientific paper."""
        
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')
        content = paper.get('content', '')
        extracted_data = paper.get('extracted_data', {})
        
        # Prepare context
        text_to_analyze = f"Title: {title}\n\nAbstract: {abstract}"
        if content and content != abstract:
            text_to_analyze += f"\n\nContent: {content[:8000]}"  # Limit content length
        
        # Add extracted data context if available
        if extracted_data:
            context_additions = []
            if extracted_data.get('key_findings'):
                context_additions.append(f"Key Findings: {'; '.join(extracted_data['key_findings'])}")
            if extracted_data.get('methodologies'):
                context_additions.append(f"Methodologies: {'; '.join(extracted_data['methodologies'])}")
            if context_additions:
                text_to_analyze += f"\n\nExtracted Information:\n{chr(10).join(context_additions)}"
        
        system_prompt = """You are a scientific research expert specializing in academic writing and research synthesis. 
        Your task is to create comprehensive, well-structured summaries of scientific papers that would be valuable 
        for researchers in the field. Focus on:
        1. Clear, concise executive summary
        2. Key contributions and novelty
        3. Methodology overview
        4. Results and findings summary
        5. Scientific significance and impact
        
        Write in a professional, academic tone suitable for researchers."""
        
        user_prompt = f"""Please create a comprehensive summary of the following scientific paper:

        {text_to_analyze}
        
        Provide:
        1. Executive Summary (2-3 sentences capturing the essence of the work)
        2. Key Contributions (list the main novel contributions)
        3. Methodology Summary (overview of experimental/computational methods)
        4. Results Summary (main findings and outcomes)
        5. Significance (why this work matters to the field)
        
        Format your response clearly with these sections."""
        
        try:
            response = await self._call_llm(user_prompt, system_prompt)
            
            # Parse the structured response
            sections = self._parse_summary_response(response)
            
            return PaperSummary(
                executive_summary=sections.get('executive_summary', ''),
                key_contributions=sections.get('key_contributions', []),
                methodology_summary=sections.get('methodology_summary', ''),
                results_summary=sections.get('results_summary', ''),
                significance=sections.get('significance', ''),
                generated_at=datetime.now()
            )
            
        except Exception as e:
            print(f"Error in summary generation: {e}")
            # Return basic summary if LLM processing fails
            return PaperSummary(
                executive_summary=f"Summary generation failed for: {title}",
                key_contributions=[],
                methodology_summary="",
                results_summary="",
                significance="",
                generated_at=datetime.now()
            )
    
    def _parse_summary_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into structured sections."""
        sections = {
            'executive_summary': '',
            'key_contributions': [],
            'methodology_summary': '',
            'results_summary': '',
            'significance': ''
        }
        
        lines = response.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for section headers
            lower_line = line.lower()
            if 'executive summary' in lower_line or 'summary' in lower_line:
                if current_section:
                    sections[current_section] = self._process_section_content(current_section, current_content)
                current_section = 'executive_summary'
                current_content = []
                # Check if content is on the same line
                if ':' in line:
                    remaining = line.split(':', 1)[1].strip()
                    if remaining:
                        current_content.append(remaining)
            elif 'key contribution' in lower_line or 'contribution' in lower_line:
                if current_section:
                    sections[current_section] = self._process_section_content(current_section, current_content)
                current_section = 'key_contributions'
                current_content = []
                if ':' in line:
                    remaining = line.split(':', 1)[1].strip()
                    if remaining:
                        current_content.append(remaining)
            elif 'methodology' in lower_line or 'methods' in lower_line:
                if current_section:
                    sections[current_section] = self._process_section_content(current_section, current_content)
                current_section = 'methodology_summary'
                current_content = []
                if ':' in line:
                    remaining = line.split(':', 1)[1].strip()
                    if remaining:
                        current_content.append(remaining)
            elif 'results' in lower_line or 'findings' in lower_line:
                if current_section:
                    sections[current_section] = self._process_section_content(current_section, current_content)
                current_section = 'results_summary'
                current_content = []
                if ':' in line:
                    remaining = line.split(':', 1)[1].strip()
                    if remaining:
                        current_content.append(remaining)
            elif 'significance' in lower_line or 'impact' in lower_line:
                if current_section:
                    sections[current_section] = self._process_section_content(current_section, current_content)
                current_section = 'significance'
                current_content = []
                if ':' in line:
                    remaining = line.split(':', 1)[1].strip()
                    if remaining:
                        current_content.append(remaining)
            else:
                if current_section:
                    current_content.append(line)
                else:
                    # If no section identified yet, assume executive summary
                    if not sections['executive_summary']:
                        current_section = 'executive_summary'
                        current_content = [line]
        
        # Process the last section
        if current_section and current_content:
            sections[current_section] = self._process_section_content(current_section, current_content)
        
        return sections
    
    def _process_section_content(self, section_type: str, content_lines: List[str]) -> Any:
        """Process content lines based on section type."""
        if section_type == 'key_contributions':
            # Extract list items
            contributions = []
            for line in content_lines:
                # Remove bullet points and numbers
                cleaned = line.strip('•-*').strip()
                if cleaned.startswith(('1.', '2.', '3.', '4.', '5.')):
                    cleaned = cleaned[2:].strip()
                if cleaned:
                    contributions.append(cleaned)
            return contributions
        else:
            # Join as single text
            return ' '.join(content_lines).strip()