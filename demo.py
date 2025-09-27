#!/usr/bin/env python3
"""
Demo script showing the excited_ions pipeline functionality.

This script demonstrates the pipeline with mock data since we can't access 
external APIs in this environment.
"""

import json
import asyncio
from datetime import datetime
from pathlib import Path

from excited_ions.models import *
from excited_ions.config import Config


def create_mock_paper_data():
    """Create mock paper data for demonstration."""
    return {
        'id': '2312.12345',
        'source': 'arxiv',
        'title': 'Quantum Machine Learning for Ion Trap Quantum Computing: Novel Approaches and Experimental Validation',
        'abstract': '''We present a novel approach to quantum machine learning using ion trap quantum computers. 
        Our method combines variational quantum algorithms with classical optimization to achieve quantum advantage 
        in specific machine learning tasks. We demonstrate experimentally that our approach can outperform classical 
        methods on benchmark datasets with quantum advantage becoming apparent for problems with specific structure. 
        The key innovation is a hybrid quantum-classical neural network architecture that leverages the unique 
        properties of trapped ions for quantum information processing. Our results show 15% improvement in accuracy 
        over classical baselines for structured optimization problems and 98.5% fidelity in quantum state preparation. 
        These findings open new avenues for practical quantum machine learning applications.''',
        'authors': [
            {'name': 'Alice Johnson', 'affiliation': 'MIT'},
            {'name': 'Bob Smith', 'affiliation': 'Stanford'},
            {'name': 'Carol Williams', 'affiliation': 'Caltech'}
        ],
        'publication_date': datetime(2023, 12, 20, 10, 0, 0),
        'subjects': ['quant-ph', 'cs.LG', 'cs.AI'],
        'url': 'http://arxiv.org/abs/2312.12345'
    }


def create_mock_extracted_data():
    """Create mock extracted experimental data."""
    return ExtractedData(
        experimental_results=[
            ExperimentalResult(
                measurement="Classification accuracy improvement",
                value=15,
                unit="percent",
                method="Hybrid quantum-classical neural network",
                conditions=[
                    ExperimentalCondition(parameter="Problem size", value=100, unit="features"),
                    ExperimentalCondition(parameter="Quantum circuit depth", value=10, unit="layers")
                ]
            ),
            ExperimentalResult(
                measurement="Quantum state fidelity",
                value=98.5,
                unit="percent",
                uncertainty=0.3,
                method="Process tomography"
            )
        ],
        methodologies=[
            "Variational Quantum Eigensolver (VQE)",
            "Quantum Approximate Optimization Algorithm (QAOA)",
            "Ion trap quantum computing",
            "Hybrid quantum-classical optimization"
        ],
        materials=[
            "Ytterbium-171 ions",
            "Paul trap setup",
            "Laser cooling system",
            "Classical computing cluster for optimization"
        ],
        key_findings=[
            "Quantum advantage achieved for structured optimization problems",
            "Hybrid architecture outperforms pure classical methods by 15%",
            "High fidelity quantum state preparation (98.5%) maintained throughout computation",
            "Scalability demonstrated up to 100-feature problems"
        ],
        limitations=[
            "Current approach limited to problems with specific structure",
            "Decoherence effects limit circuit depth to ~10 layers",
            "Classical optimization overhead increases with problem size",
            "Requires specialized quantum hardware (ion traps)"
        ]
    )


def create_mock_summary():
    """Create mock paper summary."""
    return PaperSummary(
        executive_summary="""This work demonstrates a breakthrough in quantum machine learning by developing 
        a hybrid quantum-classical neural network architecture specifically designed for ion trap quantum computers. 
        The authors achieve measurable quantum advantage on structured optimization problems, with 15% accuracy 
        improvements over classical baselines while maintaining 98.5% quantum state fidelity.""",
        key_contributions=[
            "Novel hybrid quantum-classical neural network architecture for ion traps",
            "Experimental demonstration of quantum advantage in machine learning tasks",
            "Scalable approach validated on problems up to 100 features",
            "High-fidelity quantum state preparation and manipulation protocols"
        ],
        methodology_summary="""The research combines variational quantum algorithms (VQE, QAOA) with classical 
        optimization in a hybrid architecture. Experiments were conducted on ytterbium-171 ion trap systems with 
        careful characterization of quantum states using process tomography. The approach was validated on 
        benchmark machine learning datasets with systematic comparison to classical baselines.""",
        results_summary="""Key results include 15% accuracy improvement over classical methods on structured 
        optimization problems, 98.5% fidelity in quantum state preparation, and successful scaling to 100-feature 
        problems. The quantum advantage becomes apparent for problems with specific mathematical structure that 
        can leverage quantum superposition and entanglement.""",
        significance="""This work represents a significant step toward practical quantum machine learning by 
        demonstrating measurable quantum advantage on real hardware. The hybrid approach addresses key limitations 
        of pure quantum methods while leveraging quantum resources effectively. Results suggest promising paths 
        for near-term quantum computing applications in machine learning and optimization."""
    )


def create_mock_experiment_idea():
    """Create mock experiment idea."""
    return ExperimentIdea(
        title="Quantum Transfer Learning for Multi-Ion Species in Heterogeneous Trap Arrays",
        hypothesis="""We hypothesize that using multiple ion species (e.g., Ca+, Sr+, Ba+) in heterogeneous 
        trap arrays will enable quantum transfer learning, where quantum models trained on one ion species 
        can be efficiently adapted to others. This should reduce training time by 70% and improve performance 
        on cross-species optimization tasks by leveraging species-specific quantum properties.""",
        methodology="""
        1. Design heterogeneous Paul trap array supporting 3+ ion species simultaneously
        2. Develop species-specific quantum feature encoding protocols
        3. Implement quantum transfer learning algorithms using variational quantum circuits
        4. Train base models on abundant Ca+ ion data, then transfer to Sr+ and Ba+ ions
        5. Compare transfer learning performance vs. training from scratch
        6. Characterize cross-species quantum correlations and their impact on learning
        7. Validate on molecular property prediction and quantum chemistry optimization tasks
        """,
        expected_outcomes="""We expect 70% reduction in training time for new ion species, 25% improvement 
        in cross-species task performance, and discovery of novel quantum correlations between different ion 
        species that can be exploited for enhanced quantum machine learning. This could lead to more efficient 
        quantum algorithms that leverage the unique properties of multiple atomic species.""",
        required_resources=[
            "Multi-species ion trap setup (Ca+, Sr+, Ba+)",
            "Wavelength-specific laser systems for each ion species",
            "Advanced quantum control electronics",
            "High-resolution imaging system for multi-species detection",
            "Classical computing cluster for hybrid optimization",
            "Quantum chemistry simulation software",
            "Specialized expertise in atomic physics and quantum control"
        ],
        timeline="18 months: 6 months setup and calibration, 8 months experiments, 4 months analysis and publication",
        rationale="""Current quantum machine learning approaches are limited to single ion species, missing 
        opportunities to leverage the diverse properties of different atoms. The demonstrated success with 
        single-species systems suggests that multi-species approaches could unlock new quantum advantages. 
        Transfer learning has shown great success in classical ML, and quantum transfer learning could be 
        even more powerful due to quantum superposition and entanglement across species boundaries.""",
        related_papers=[
            "Quantum Machine Learning for Ion Trap Quantum Computing (arXiv:2312.12345)",
            "Multi-species Trapped Ion Systems for Quantum Computing (Nature, 2023)",
            "Classical Transfer Learning in Optimization (ICML, 2023)"
        ]
    )


async def demonstrate_pipeline():
    """Demonstrate the complete pipeline with mock data."""
    
    print("🧪 excited_ions Pipeline Demonstration")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("./demo_output")
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Mock paper search and fetch
    print("\n1. 📚 Fetching Papers")
    print("   Searching arXiv for: 'quantum machine learning'")
    mock_paper = create_mock_paper_data()
    print(f"   ✓ Found paper: {mock_paper['title'][:60]}...")
    
    # Step 2: Extract metadata and experimental data
    print("\n2. 🔍 Extracting Metadata and Experimental Data")
    extracted_data = create_mock_extracted_data()
    print(f"   ✓ Extracted {len(extracted_data.experimental_results)} experimental results")
    print(f"   ✓ Identified {len(extracted_data.methodologies)} methodologies")
    print(f"   ✓ Found {len(extracted_data.key_findings)} key findings")
    
    # Step 3: Generate summary
    print("\n3. 📝 Generating Paper Summary")
    summary = create_mock_summary()
    print(f"   ✓ Generated executive summary ({len(summary.executive_summary)} characters)")
    print(f"   ✓ Identified {len(summary.key_contributions)} key contributions")
    
    # Step 4: Generate experiment idea
    print("\n4. 💡 Generating Novel Experiment Ideas")
    experiment_idea = create_mock_experiment_idea()
    print(f"   ✓ Generated experiment: {experiment_idea.title}")
    print(f"   ✓ Hypothesis: {experiment_idea.hypothesis[:100]}...")
    
    # Step 5: Create ProcessedPaper and save
    print("\n5. 💾 Saving Structured Results")
    processed_paper = ProcessedPaper(
        paper_id=mock_paper['id'],
        source_url=mock_paper['url'],
        metadata=PaperMetadata(
            title=mock_paper['title'],
            authors=[Author(name=a['name'], affiliation=a.get('affiliation')) 
                    for a in mock_paper['authors']],
            abstract=mock_paper['abstract'],
            arxiv_id=mock_paper['id'],
            publication_date=mock_paper['publication_date'],
            subjects=mock_paper['subjects']
        ),
        extracted_data=extracted_data,
        summary=summary,
        experiment_idea=experiment_idea,
        processing_status="completed",
        processed_at=datetime.now()
    )
    
    # Save to JSON
    output_file = output_dir / f"{mock_paper['id']}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_paper.model_dump(), f, indent=2, ensure_ascii=False, default=str)
    
    print(f"   ✓ Saved detailed results to: {output_file}")
    
    # Step 6: Display key results
    print("\n6. 📊 Results Summary")
    print(f"   📄 Paper: {processed_paper.metadata.title}")
    print(f"   👥 Authors: {', '.join([a.name for a in processed_paper.metadata.authors])}")
    print(f"   📅 Date: {processed_paper.metadata.publication_date.strftime('%Y-%m-%d')}")
    print(f"   🏷️  Subjects: {', '.join(processed_paper.metadata.subjects)}")
    
    print(f"\n   📋 Summary:")
    print(f"      {summary.executive_summary[:200]}...")
    
    print(f"\n   🔬 Key Findings:")
    for finding in extracted_data.key_findings[:2]:
        print(f"      • {finding}")
    
    print(f"\n   💡 Experiment Idea: {experiment_idea.title}")
    print(f"      Hypothesis: {experiment_idea.hypothesis[:150]}...")
    print(f"      Timeline: {experiment_idea.timeline}")
    
    print(f"\n   📁 Output: {len(list(output_dir.glob('*.json')))} files saved to {output_dir}/")
    
    print("\n✅ Pipeline demonstration completed successfully!")
    print("\nThis demonstrates the complete excited_ions workflow:")
    print("  1. Paper fetching from arXiv/PubMed")
    print("  2. LLM-based metadata extraction")
    print("  3. Intelligent summarization")
    print("  4. Novel experiment ideation")
    print("  5. Structured JSON output")
    
    return processed_paper


if __name__ == "__main__":
    # Run the demonstration
    result = asyncio.run(demonstrate_pipeline())
    
    print(f"\n📄 Try the CLI with: excited-ions --help")
    print(f"📄 View demo output: ls -la demo_output/")