"""Experiment idea generator using LLMs."""

from typing import Dict, Any, List
from datetime import datetime
from . import BaseLLMExtractor
from ..models import ExperimentIdea


class IdeationEngine(BaseLLMExtractor):
    """Generates novel experiment ideas based on paper analysis and literature context."""
    
    async def extract(self, paper: Dict[str, Any], related_papers: List[Dict[str, Any]] = None) -> ExperimentIdea:
        """Generate a novel experiment idea based on the paper and related literature."""
        
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')
        extracted_data = paper.get('extracted_data', {})
        summary = paper.get('summary', {})
        
        # Build context from the main paper
        paper_context = f"Title: {title}\n\nAbstract: {abstract}"
        
        if summary:
            if summary.get('key_contributions'):
                paper_context += f"\n\nKey Contributions: {'; '.join(summary['key_contributions'])}"
            if summary.get('significance'):
                paper_context += f"\n\nSignificance: {summary['significance']}"
        
        if extracted_data:
            if extracted_data.get('key_findings'):
                paper_context += f"\n\nKey Findings: {'; '.join(extracted_data['key_findings'])}"
            if extracted_data.get('limitations'):
                paper_context += f"\n\nLimitations: {'; '.join(extracted_data['limitations'])}"
        
        # Build context from related papers if provided
        related_context = ""
        if related_papers:
            related_context = "\n\nRelated Literature Context:\n"
            for i, related in enumerate(related_papers[:3], 1):  # Limit to 3 related papers
                related_title = related.get('title', f'Related Paper {i}')
                related_abstract = related.get('abstract', '')
                related_context += f"{i}. {related_title}\n   Abstract: {related_abstract[:300]}...\n\n"
        
        full_context = paper_context + related_context
        
        system_prompt = """You are a creative scientific research expert specializing in experimental design and innovation. 
        Your task is to generate novel, feasible experiment ideas that build upon existing research while addressing 
        identified gaps or limitations. Focus on:
        
        1. Identifying genuine research gaps or limitations from the current work
        2. Proposing innovative but feasible experimental approaches
        3. Building on existing methodologies while introducing novel elements
        4. Considering practical constraints and resource requirements
        5. Ensuring the proposed experiment would advance scientific knowledge
        
        Your experiment idea should be:
        - Novel and creative but scientifically sound
        - Technically feasible with current methods and reasonable resources
        - Clearly motivated by gaps or limitations in existing work
        - Well-defined in terms of hypothesis, methodology, and expected outcomes"""
        
        user_prompt = f"""Based on the following scientific paper and related literature, propose a novel experiment that would advance the field:

        {full_context}
        
        Please provide:
        1. Experiment Title: A clear, descriptive title
        2. Hypothesis: What you expect to find and why
        3. Methodology: Detailed experimental approach and methods
        4. Expected Outcomes: What results you anticipate and how they would advance knowledge
        5. Required Resources: Equipment, materials, expertise needed
        6. Timeline: Estimated duration for the study
        7. Rationale: Why this experiment is needed and how it builds on existing work
        
        Make sure the proposed experiment addresses genuine gaps or limitations in the current research."""
        
        try:
            response = await self._call_llm(user_prompt, system_prompt)
            
            # Parse the structured response
            sections = self._parse_ideation_response(response)
            
            # Extract related paper IDs/titles for reference
            related_refs = []
            if related_papers:
                for paper in related_papers:
                    ref = paper.get('title', '')
                    if paper.get('arxiv_id'):
                        ref += f" (arXiv:{paper['arxiv_id']})"
                    elif paper.get('doi'):
                        ref += f" (DOI:{paper['doi']})"
                    if ref:
                        related_refs.append(ref)
            
            return ExperimentIdea(
                title=sections.get('title', 'Novel Experiment Based on Literature Analysis'),
                hypothesis=sections.get('hypothesis', ''),
                methodology=sections.get('methodology', ''),
                expected_outcomes=sections.get('expected_outcomes', ''),
                required_resources=sections.get('required_resources', []),
                timeline=sections.get('timeline'),
                rationale=sections.get('rationale', ''),
                related_papers=related_refs[:5],  # Limit to 5 references
                generated_at=datetime.now()
            )
            
        except Exception as e:
            print(f"Error in experiment ideation: {e}")
            # Return basic idea if LLM processing fails
            return ExperimentIdea(
                title=f"Follow-up Study for: {title}",
                hypothesis="Further investigation needed to address study limitations",
                methodology="To be determined based on detailed analysis",
                expected_outcomes="Improved understanding of the research question",
                required_resources=["Standard laboratory equipment"],
                rationale="Generated due to LLM processing error",
                related_papers=[],
                generated_at=datetime.now()
            )
    
    def _parse_ideation_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into structured sections."""
        sections = {
            'title': '',
            'hypothesis': '',
            'methodology': '',
            'expected_outcomes': '',
            'required_resources': [],
            'timeline': '',
            'rationale': ''
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
            section_found = False
            
            for section_key in ['title', 'hypothesis', 'methodology', 'expected_outcomes', 'required_resources', 'timeline', 'rationale']:
                if section_key.replace('_', ' ') in lower_line or section_key in lower_line:
                    if section_key == 'expected_outcomes' and ('outcome' in lower_line or 'result' in lower_line):
                        section_found = True
                    elif section_key == 'required_resources' and ('resource' in lower_line or 'equipment' in lower_line or 'material' in lower_line):
                        section_found = True
                    elif section_key in lower_line:
                        section_found = True
                    
                    if section_found:
                        # Process previous section
                        if current_section:
                            sections[current_section] = self._process_ideation_content(current_section, current_content)
                        
                        current_section = section_key
                        current_content = []
                        
                        # Check if content is on the same line
                        if ':' in line:
                            remaining = line.split(':', 1)[1].strip()
                            if remaining:
                                current_content.append(remaining)
                        break
            
            if not section_found and current_section:
                current_content.append(line)
            elif not section_found and not current_section:
                # Assume first content is title if no section identified
                current_section = 'title'
                current_content = [line]
        
        # Process the last section
        if current_section and current_content:
            sections[current_section] = self._process_ideation_content(current_section, current_content)
        
        return sections
    
    def _process_ideation_content(self, section_type: str, content_lines: List[str]) -> Any:
        """Process content lines based on section type."""
        if section_type == 'required_resources':
            # Extract list items
            resources = []
            for line in content_lines:
                # Split by common delimiters and clean
                items = line.replace(',', '\n').replace(';', '\n').replace('•', '\n').replace('-', '\n').split('\n')
                for item in items:
                    cleaned = item.strip()
                    if cleaned and not cleaned.startswith(('1.', '2.', '3.')):
                        resources.append(cleaned)
                    elif cleaned.startswith(('1.', '2.', '3.')):
                        resources.append(cleaned[2:].strip())
            return resources
        else:
            # Join as single text
            return ' '.join(content_lines).strip()