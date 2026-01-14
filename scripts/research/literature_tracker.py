#!/usr/bin/env python3
"""
QBitaLabs Literature & Research Tracker
Automated monitoring of latest papers, datasets, and competitor activity

Sources:
- arXiv (cs.LG, q-bio, physics.chem-ph)
- bioRxiv (bioinformatics, systems biology)
- PubMed (drug discovery, quantum computing)
- GitHub (trending repos)
- TDC Leaderboards
"""

import os
import sys
import json
import logging
import argparse
import hashlib
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from urllib.parse import quote_plus
import time

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None
    BeautifulSoup = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Paper:
    """Represents a research paper."""
    title: str
    authors: List[str]
    abstract: str
    url: str
    source: str  # arxiv, biorxiv, pubmed
    published_date: str
    categories: List[str] = field(default_factory=list)
    relevance_score: float = 0.0
    keywords_matched: List[str] = field(default_factory=list)
    pdf_url: Optional[str] = None


@dataclass
class Dataset:
    """Represents a dataset or benchmark."""
    name: str
    description: str
    url: str
    source: str
    task_type: str
    size: Optional[str] = None
    license: Optional[str] = None
    last_updated: Optional[str] = None


@dataclass
class CompetitorActivity:
    """Tracks competitor announcements and releases."""
    company: str
    activity_type: str  # paper, funding, partnership, product
    title: str
    description: str
    url: str
    date: str
    impact_level: str  # high, medium, low


class ArxivTracker:
    """Tracks papers from arXiv."""

    BASE_URL = "http://export.arxiv.org/api/query"

    # Relevant categories for drug discovery and quantum computing
    CATEGORIES = [
        "cs.LG",      # Machine Learning
        "q-bio.BM",   # Biomolecules
        "q-bio.QM",   # Quantitative Methods
        "physics.chem-ph",  # Chemical Physics
        "quant-ph",   # Quantum Physics
        "cs.AI",      # Artificial Intelligence
        "stat.ML",    # Machine Learning (Statistics)
    ]

    # Keywords for relevance scoring
    KEYWORDS = [
        "drug discovery", "molecular", "protein", "binding affinity",
        "quantum", "drug-target", "ADMET", "toxicity", "solubility",
        "graph neural network", "GNN", "transformer", "foundation model",
        "AlphaFold", "ESM", "diffusion", "generative", "SMILES",
        "virtual screening", "docking", "pharmacokinetics",
        "digital twin", "clinical trial", "personalized medicine",
    ]

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir / "arxiv"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def search(self, days_back: int = 7, max_results: int = 100) -> List[Paper]:
        """Search arXiv for recent relevant papers."""
        if requests is None:
            logger.warning("requests library not installed - skipping arXiv")
            return []

        papers = []

        # Build search query
        category_query = " OR ".join([f"cat:{cat}" for cat in self.CATEGORIES])
        keyword_query = " OR ".join([f'"{kw}"' for kw in self.KEYWORDS[:10]])

        query = f"({category_query}) AND ({keyword_query})"

        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }

        try:
            logger.info(f"Searching arXiv for recent papers...")
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()

            # Parse XML response
            if BeautifulSoup:
                soup = BeautifulSoup(response.text, 'xml')
                entries = soup.find_all('entry')

                for entry in entries:
                    try:
                        paper = self._parse_entry(entry)
                        if paper and self._is_recent(paper.published_date, days_back):
                            paper.relevance_score = self._calculate_relevance(paper)
                            if paper.relevance_score > 0.3:
                                papers.append(paper)
                    except Exception as e:
                        logger.warning(f"Error parsing arXiv entry: {e}")

            logger.info(f"Found {len(papers)} relevant papers from arXiv")

        except Exception as e:
            logger.error(f"arXiv search failed: {e}")

        return papers

    def _parse_entry(self, entry) -> Optional[Paper]:
        """Parse an arXiv entry."""
        try:
            title = entry.find('title').text.strip().replace('\n', ' ')
            abstract = entry.find('summary').text.strip().replace('\n', ' ')

            authors = []
            for author in entry.find_all('author'):
                name = author.find('name')
                if name:
                    authors.append(name.text.strip())

            url = entry.find('id').text.strip()
            published = entry.find('published').text.strip()[:10]

            categories = []
            for cat in entry.find_all('category'):
                term = cat.get('term')
                if term:
                    categories.append(term)

            # Get PDF URL
            pdf_url = None
            for link in entry.find_all('link'):
                if link.get('title') == 'pdf':
                    pdf_url = link.get('href')
                    break

            return Paper(
                title=title,
                authors=authors[:5],  # Limit authors
                abstract=abstract[:1000],  # Limit abstract
                url=url,
                source="arxiv",
                published_date=published,
                categories=categories,
                pdf_url=pdf_url
            )
        except Exception as e:
            logger.warning(f"Error parsing entry: {e}")
            return None

    def _is_recent(self, date_str: str, days_back: int) -> bool:
        """Check if date is within days_back."""
        try:
            pub_date = datetime.strptime(date_str[:10], "%Y-%m-%d")
            cutoff = datetime.now() - timedelta(days=days_back)
            return pub_date >= cutoff
        except:
            return False

    def _calculate_relevance(self, paper: Paper) -> float:
        """Calculate relevance score based on keyword matching."""
        text = f"{paper.title} {paper.abstract}".lower()

        matched = []
        for kw in self.KEYWORDS:
            if kw.lower() in text:
                matched.append(kw)

        paper.keywords_matched = matched

        # Score based on number of keywords matched
        if len(matched) >= 5:
            return 1.0
        elif len(matched) >= 3:
            return 0.8
        elif len(matched) >= 2:
            return 0.5
        elif len(matched) >= 1:
            return 0.3
        return 0.0


class BioRxivTracker:
    """Tracks papers from bioRxiv."""

    BASE_URL = "https://api.biorxiv.org/details/biorxiv"

    COLLECTIONS = [
        "bioinformatics",
        "systems-biology",
        "pharmacology-and-toxicology",
        "biochemistry",
        "genomics",
    ]

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir / "biorxiv"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def search(self, days_back: int = 7) -> List[Paper]:
        """Search bioRxiv for recent papers."""
        if requests is None:
            logger.warning("requests library not installed - skipping bioRxiv")
            return []

        papers = []

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        url = f"{self.BASE_URL}/{start_date}/{end_date}/0/100"

        try:
            logger.info(f"Searching bioRxiv for papers from {start_date} to {end_date}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            data = response.json()

            for item in data.get("collection", []):
                try:
                    paper = self._parse_item(item)
                    if paper:
                        # Filter by relevance
                        relevance = self._calculate_relevance(paper)
                        if relevance > 0.3:
                            paper.relevance_score = relevance
                            papers.append(paper)
                except Exception as e:
                    logger.warning(f"Error parsing bioRxiv item: {e}")

            logger.info(f"Found {len(papers)} relevant papers from bioRxiv")

        except Exception as e:
            logger.error(f"bioRxiv search failed: {e}")

        return papers

    def _parse_item(self, item: Dict) -> Optional[Paper]:
        """Parse a bioRxiv item."""
        try:
            return Paper(
                title=item.get("title", ""),
                authors=item.get("authors", "").split("; ")[:5],
                abstract=item.get("abstract", "")[:1000],
                url=f"https://www.biorxiv.org/content/{item.get('doi')}",
                source="biorxiv",
                published_date=item.get("date", ""),
                categories=[item.get("category", "")],
                pdf_url=f"https://www.biorxiv.org/content/{item.get('doi')}.full.pdf"
            )
        except Exception as e:
            logger.warning(f"Error parsing bioRxiv item: {e}")
            return None

    def _calculate_relevance(self, paper: Paper) -> float:
        """Calculate relevance score."""
        keywords = [
            "drug", "molecule", "protein", "binding", "quantum",
            "machine learning", "deep learning", "neural network",
            "prediction", "screening", "target", "compound",
            "foundation model", "transformer", "graph",
        ]

        text = f"{paper.title} {paper.abstract}".lower()
        matched = [kw for kw in keywords if kw in text]
        paper.keywords_matched = matched

        return min(len(matched) / 3, 1.0)


class TDCLeaderboardTracker:
    """Tracks Therapeutics Data Commons leaderboards."""

    TDC_URL = "https://tdcommons.ai"

    # Key benchmarks to track
    BENCHMARKS = [
        {"name": "ADMET", "tasks": ["caco2_wang", "hia_hou", "pgp_broccatelli", "bioavailability_ma"]},
        {"name": "DTI", "tasks": ["davis", "kiba", "bindingdb_kd"]},
        {"name": "Tox21", "tasks": ["tox21"]},
        {"name": "ADME", "tasks": ["lipophilicity_astrazeneca", "solubility_aqsoldb"]},
        {"name": "DDI", "tasks": ["drugbank", "twosides"]},
    ]

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir / "tdc"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.leaderboard_file = self.output_dir / "leaderboards.json"

    def fetch_leaderboards(self) -> Dict[str, Any]:
        """Fetch current leaderboard standings."""
        # Note: TDC doesn't have a public API, so we track known benchmarks
        # In production, this would scrape or use their API if available

        leaderboards = {
            "last_updated": datetime.now().isoformat(),
            "benchmarks": {}
        }

        # Placeholder for benchmark data - would fetch from TDC
        for benchmark in self.BENCHMARKS:
            leaderboards["benchmarks"][benchmark["name"]] = {
                "tasks": benchmark["tasks"],
                "our_rank": None,
                "sota_model": "Unknown",
                "sota_score": None
            }

        # Save leaderboards
        with open(self.leaderboard_file, "w") as f:
            json.dump(leaderboards, f, indent=2)

        logger.info(f"Updated TDC leaderboard tracking")
        return leaderboards

    def compare_performance(self, our_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Compare our performance against SOTA."""
        comparison = {}

        # SOTA benchmarks (as of Jan 2026)
        sota = {
            "caco2_wang": {"metric": "MAE", "sota": 0.276, "model": "GNN-ADMET"},
            "lipophilicity": {"metric": "RMSE", "sota": 0.467, "model": "ChemBERTa"},
            "davis": {"metric": "MSE", "sota": 0.194, "model": "GraphDTA"},
            "kiba": {"metric": "MSE", "sota": 0.124, "model": "DeepDTA"},
        }

        for task, data in sota.items():
            ours = our_metrics.get(task)
            if ours:
                gap = ((ours - data["sota"]) / data["sota"]) * 100
                comparison[task] = {
                    "our_score": ours,
                    "sota_score": data["sota"],
                    "sota_model": data["model"],
                    "gap_percent": gap,
                    "status": "ABOVE_SOTA" if gap < 0 else "BELOW_SOTA"
                }

        return comparison


class CompetitorTracker:
    """Tracks competitor activity and announcements."""

    # Key competitors to monitor
    COMPETITORS = [
        {
            "name": "Recursion Pharmaceuticals",
            "focus": "AI drug discovery, phenomics",
            "funding": "$1.5B+",
            "stage": "Public (RXRX)"
        },
        {
            "name": "Isomorphic Labs",
            "focus": "AlphaFold3, protein structure",
            "funding": "$600M seed",
            "stage": "Private"
        },
        {
            "name": "Insilico Medicine",
            "focus": "Hybrid quantum-AI, KRAS targeting",
            "funding": "$500M+",
            "stage": "Clinical (Phase 2)"
        },
        {
            "name": "Schrödinger",
            "focus": "Physics-based simulation",
            "funding": "Public (SDGR)",
            "stage": "Clinical partnerships"
        },
        {
            "name": "Atomwise",
            "focus": "AI virtual screening",
            "funding": "$174M",
            "stage": "Partnerships"
        },
        {
            "name": "BenevolentAI",
            "focus": "Knowledge graphs, drug repurposing",
            "funding": "Public (BAI)",
            "stage": "Clinical"
        },
        {
            "name": "Relay Therapeutics",
            "focus": "Motion-based drug design",
            "funding": "Public (RLAY)",
            "stage": "Clinical (Phase 2)"
        },
        {
            "name": "Exscientia",
            "focus": "AI-first drug design",
            "funding": "Public (EXAI)",
            "stage": "Clinical (Phase 1)"
        },
    ]

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir / "competitors"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_competitor_profiles(self) -> List[Dict[str, Any]]:
        """Get competitor profiles."""
        return self.COMPETITORS

    def generate_competitive_analysis(self) -> Dict[str, Any]:
        """Generate competitive analysis report."""
        analysis = {
            "generated": datetime.now().isoformat(),
            "competitors": self.COMPETITORS,
            "qbitalabs_differentiators": [
                "Only hybrid quantum-classical + foundation model platform",
                "FDA-ready digital twin architecture",
                "End-to-end vertical integration",
                "Open-source data strategy (TDC integration)",
                "Apple Silicon optimization for cost efficiency"
            ],
            "threats": [
                "Isomorphic Labs' AlphaFold3 dominance in structure prediction",
                "Recursion's massive phenomics dataset",
                "Insilico's clinical validation (Phase 2)"
            ],
            "opportunities": [
                "Quantum advantage emerging in 2025-2026",
                "OpenFold3 Apache 2.0 license enables commercial use",
                "FDA digital twin guidance opening regulatory pathway",
                "Foundation model convergence (ESM3 + Geneformer + OpenFold3)"
            ]
        }

        # Save analysis
        analysis_file = self.output_dir / "competitive_analysis.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2)

        return analysis


class DatasetCatalog:
    """Catalogs available open-source datasets."""

    DATASETS = [
        # TDC Datasets
        Dataset("TDC-ADMET", "ADMET property prediction benchmarks", "https://tdcommons.ai/benchmark/admet_group/", "TDC", "property_prediction", "66 datasets"),
        Dataset("TDC-DTI", "Drug-target interaction datasets", "https://tdcommons.ai/multi_pred_tasks/dti/", "TDC", "interaction_prediction", "Davis, KIBA, BindingDB"),
        Dataset("TDC-Tox", "Toxicity prediction benchmarks", "https://tdcommons.ai/benchmark/toxicity/", "TDC", "toxicity", "Tox21, ClinTox"),

        # Molecular Datasets
        Dataset("ChEMBL", "Bioactivity database", "https://www.ebi.ac.uk/chembl/", "EBI", "bioactivity", "2.4M compounds"),
        Dataset("PDBbind", "Protein-ligand binding affinities", "http://www.pdbbind.org.cn/", "PDBbind", "binding_affinity", "23K complexes"),
        Dataset("ZINC250K", "Drug-like molecules", "https://zinc.docking.org/", "ZINC", "molecules", "250K molecules"),
        Dataset("MoleculeNet", "Molecular ML benchmarks", "https://moleculenet.org/", "Stanford", "benchmark", "Multiple tasks"),

        # Protein Datasets
        Dataset("UniProt", "Protein sequences and annotations", "https://www.uniprot.org/", "UniProt", "protein", "250M sequences"),
        Dataset("PDB", "Protein structures", "https://www.rcsb.org/", "RCSB", "structure", "200K+ structures"),
        Dataset("AlphaFold DB", "Predicted protein structures", "https://alphafold.ebi.ac.uk/", "DeepMind", "structure", "200M+ predictions"),

        # Genomics/Transcriptomics
        Dataset("GEO", "Gene expression data", "https://www.ncbi.nlm.nih.gov/geo/", "NCBI", "expression", "Millions of samples"),
        Dataset("TCGA", "Cancer genomics", "https://www.cancer.gov/ccg/research/genome-sequencing/tcga", "NCI", "cancer", "11K patients"),
        Dataset("GTEx", "Tissue expression", "https://gtexportal.org/", "NIH", "expression", "17K samples"),

        # Clinical/Patient
        Dataset("Synthea", "Synthetic patient records", "https://synthea.mitre.org/", "MITRE", "patient", "Unlimited synthetic"),
        Dataset("MIMIC-IV", "ICU patient data", "https://mimic.mit.edu/", "MIT", "clinical", "300K+ admissions"),

        # Pathway/Network
        Dataset("KEGG", "Biological pathways", "https://www.kegg.jp/", "KEGG", "pathway", "500+ pathways"),
        Dataset("STRING", "Protein interactions", "https://string-db.org/", "STRING", "network", "67M interactions"),
    ]

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir / "datasets"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_catalog(self) -> List[Dataset]:
        """Get full dataset catalog."""
        return self.DATASETS

    def save_catalog(self):
        """Save catalog to file."""
        catalog_file = self.output_dir / "dataset_catalog.json"
        catalog = [asdict(d) for d in self.DATASETS]

        with open(catalog_file, "w") as f:
            json.dump({
                "updated": datetime.now().isoformat(),
                "datasets": catalog
            }, f, indent=2)

        logger.info(f"Saved dataset catalog with {len(self.DATASETS)} datasets")
        return catalog_file

    def get_priority_datasets(self) -> List[Dataset]:
        """Get priority datasets for MVP."""
        priority_names = [
            "TDC-ADMET", "TDC-DTI", "ChEMBL", "PDBbind",
            "ZINC250K", "MoleculeNet", "Synthea"
        ]
        return [d for d in self.DATASETS if d.name in priority_names]


class ResearchTracker:
    """Main research tracking orchestrator."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize trackers
        self.arxiv = ArxivTracker(output_dir)
        self.biorxiv = BioRxivTracker(output_dir)
        self.tdc = TDCLeaderboardTracker(output_dir)
        self.competitors = CompetitorTracker(output_dir)
        self.datasets = DatasetCatalog(output_dir)

    def run_daily_scan(self, days_back: int = 7) -> Dict[str, Any]:
        """Run daily research scan."""
        logger.info("="*60)
        logger.info("Starting Daily Research Scan")
        logger.info("="*60)

        results = {
            "timestamp": datetime.now().isoformat(),
            "papers": [],
            "leaderboards": {},
            "competitors": {},
            "datasets": []
        }

        # Scan arXiv
        logger.info("\n[1/5] Scanning arXiv...")
        arxiv_papers = self.arxiv.search(days_back=days_back)
        results["papers"].extend([asdict(p) for p in arxiv_papers])

        # Scan bioRxiv
        logger.info("\n[2/5] Scanning bioRxiv...")
        biorxiv_papers = self.biorxiv.search(days_back=days_back)
        results["papers"].extend([asdict(p) for p in biorxiv_papers])

        # Update TDC leaderboards
        logger.info("\n[3/5] Updating TDC leaderboards...")
        results["leaderboards"] = self.tdc.fetch_leaderboards()

        # Competitive analysis
        logger.info("\n[4/5] Generating competitive analysis...")
        results["competitors"] = self.competitors.generate_competitive_analysis()

        # Update dataset catalog
        logger.info("\n[5/5] Updating dataset catalog...")
        self.datasets.save_catalog()
        results["datasets"] = [asdict(d) for d in self.datasets.get_priority_datasets()]

        # Save daily report
        report_file = self.output_dir / f"daily_scan_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_file, "w") as f:
            json.dump(results, f, indent=2)

        # Generate summary
        self._generate_summary(results)

        logger.info("\n" + "="*60)
        logger.info("Daily Research Scan Complete")
        logger.info(f"Report saved to: {report_file}")
        logger.info("="*60)

        return results

    def _generate_summary(self, results: Dict[str, Any]):
        """Generate human-readable summary."""
        papers = results.get("papers", [])

        # Sort by relevance
        papers.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        summary_lines = [
            "# Daily Research Summary",
            f"\nDate: {datetime.now().strftime('%Y-%m-%d')}",
            f"\n## Papers Found: {len(papers)}",
            ""
        ]

        # Top papers
        summary_lines.append("### Top 10 Most Relevant Papers\n")
        for i, paper in enumerate(papers[:10], 1):
            summary_lines.append(f"{i}. **{paper['title'][:80]}...**")
            summary_lines.append(f"   - Source: {paper['source']}")
            summary_lines.append(f"   - Relevance: {paper['relevance_score']:.2f}")
            summary_lines.append(f"   - Keywords: {', '.join(paper['keywords_matched'][:5])}")
            summary_lines.append(f"   - URL: {paper['url']}")
            summary_lines.append("")

        # Save summary
        summary_file = self.output_dir / f"summary_{datetime.now().strftime('%Y%m%d')}.md"
        with open(summary_file, "w") as f:
            f.write("\n".join(summary_lines))

        logger.info(f"Summary saved to: {summary_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="QBitaLabs Research Tracker")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./mvp/research",
        help="Output directory for research data"
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=7,
        help="Number of days to look back"
    )
    parser.add_argument(
        "--source",
        choices=["all", "arxiv", "biorxiv", "tdc", "competitors"],
        default="all",
        help="Source to scan"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    tracker = ResearchTracker(output_dir)

    if args.source == "all":
        tracker.run_daily_scan(days_back=args.days_back)
    elif args.source == "arxiv":
        papers = tracker.arxiv.search(days_back=args.days_back)
        print(f"Found {len(papers)} papers from arXiv")
    elif args.source == "biorxiv":
        papers = tracker.biorxiv.search(days_back=args.days_back)
        print(f"Found {len(papers)} papers from bioRxiv")
    elif args.source == "tdc":
        tracker.tdc.fetch_leaderboards()
    elif args.source == "competitors":
        tracker.competitors.generate_competitive_analysis()

    return 0


if __name__ == "__main__":
    sys.exit(main())
