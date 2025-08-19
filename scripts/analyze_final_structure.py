#!/usr/bin/env python3
"""
Project structure analysis script for NLP Sentiment Analysis.

This script analyzes the project structure, validates completeness,
and provides insights about the codebase organization and functionality.
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import ast
import importlib.util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProjectAnalyzer:
    """Analyzes the structure and completeness of the NLP project."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.analysis_results = {}
        
    def analyze_directory_structure(self) -> Dict[str, Any]:
        """Analyze the directory structure of the project."""
        logger.info("Analyzing directory structure...")
        
        structure = {}
        file_counts = defaultdict(int)
        total_size = 0
        
        for root, dirs, files in os.walk(self.project_root):
            root_path = Path(root)
            rel_path = root_path.relative_to(self.project_root)
            
            # Skip hidden directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            dir_info = {
                'files': [],
                'subdirs': dirs.copy(),
                'file_count': len(files),
                'total_size': 0
            }
            
            for file in files:
                if not file.startswith('.') and not file.endswith('.pyc'):
                    file_path = root_path / file
                    file_size = file_path.stat().st_size if file_path.exists() else 0
                    
                    file_info = {
                        'name': file,
                        'size': file_size,
                        'type': file_path.suffix or 'no_extension'
                    }
                    
                    dir_info['files'].append(file_info)
                    dir_info['total_size'] += file_size
                    total_size += file_size
                    
                    # Count by file type
                    file_type = file_path.suffix or 'no_extension'
                    file_counts[file_type] += 1
            
            structure[str(rel_path)] = dir_info
        
        return {
            'structure': structure,
            'file_counts': dict(file_counts),
            'total_size': total_size,
            'total_files': sum(file_counts.values())
        }
    
    def analyze_python_files(self) -> Dict[str, Any]:
        """Analyze Python files for imports, functions, classes, etc."""
        logger.info("Analyzing Python files...")
        
        python_files = list(self.project_root.rglob("*.py"))
        analysis = {
            'total_python_files': len(python_files),
            'files': {},
            'imports': defaultdict(set),
            'functions': [],
            'classes': [],
            'total_lines': 0,
            'total_functions': 0,
            'total_classes': 0
        }
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse AST
                try:
                    tree = ast.parse(content)
                except SyntaxError as e:
                    logger.warning(f"Syntax error in {py_file}: {e}")
                    continue
                
                file_info = {
                    'path': str(py_file.relative_to(self.project_root)),
                    'lines': len(content.splitlines()),
                    'functions': [],
                    'classes': [],
                    'imports': [],
                    'docstring': ast.get_docstring(tree),
                    'has_main': False
                }
                
                # Extract information from AST
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            import_name = alias.name
                            file_info['imports'].append(import_name)
                            analysis['imports'][import_name].add(str(py_file.relative_to(self.project_root)))
                    
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            import_name = node.module
                            file_info['imports'].append(import_name)
                            analysis['imports'][import_name].add(str(py_file.relative_to(self.project_root)))
                    
                    elif isinstance(node, ast.FunctionDef):
                        func_info = {
                            'name': node.name,
                            'line': node.lineno,
                            'args': [arg.arg for arg in node.args.args],
                            'docstring': ast.get_docstring(node),
                            'is_private': node.name.startswith('_'),
                            'is_magic': node.name.startswith('__') and node.name.endswith('__')
                        }
                        file_info['functions'].append(func_info)
                        analysis['functions'].append({**func_info, 'file': str(py_file.relative_to(self.project_root))})
                        
                        if node.name == 'main':
                            file_info['has_main'] = True
                    
                    elif isinstance(node, ast.ClassDef):
                        class_info = {
                            'name': node.name,
                            'line': node.lineno,
                            'bases': [base.id if hasattr(base, 'id') else str(base) for base in node.bases],
                            'docstring': ast.get_docstring(node),
                            'methods': []
                        }
                        
                        # Extract methods
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef):
                                method_info = {
                                    'name': item.name,
                                    'line': item.lineno,
                                    'is_private': item.name.startswith('_'),
                                    'is_magic': item.name.startswith('__') and item.name.endswith('__')
                                }
                                class_info['methods'].append(method_info)
                        
                        file_info['classes'].append(class_info)
                        analysis['classes'].append({**class_info, 'file': str(py_file.relative_to(self.project_root))})
                
                analysis['files'][str(py_file.relative_to(self.project_root))] = file_info
                analysis['total_lines'] += file_info['lines']
                analysis['total_functions'] += len(file_info['functions'])
                analysis['total_classes'] += len(file_info['classes'])
                
            except Exception as e:
                logger.warning(f"Error analyzing {py_file}: {e}")
        
        # Convert sets to lists for JSON serialization
        analysis['imports'] = {k: list(v) for k, v in analysis['imports'].items()}
        
        return analysis
    
    def analyze_package_structure(self) -> Dict[str, Any]:
        """Analyze the package structure and imports."""
        logger.info("Analyzing package structure...")
        
        packages = {}
        
        # Find all __init__.py files
        init_files = list(self.project_root.rglob("__init__.py"))
        
        for init_file in init_files:
            package_dir = init_file.parent
            package_name = str(package_dir.relative_to(self.project_root))
            
            # Get modules in this package
            modules = []
            for py_file in package_dir.glob("*.py"):
                if py_file.name != "__init__.py":
                    modules.append(py_file.stem)
            
            # Check if __init__.py has content
            try:
                with open(init_file, 'r', encoding='utf-8') as f:
                    init_content = f.read()
                
                has_content = len(init_content.strip()) > 0
                has_imports = "__import__" in init_content or "from" in init_content or "import" in init_content
                
            except Exception as e:
                has_content = False
                has_imports = False
                logger.warning(f"Error reading {init_file}: {e}")
            
            packages[package_name] = {
                'modules': modules,
                'init_has_content': has_content,
                'init_has_imports': has_imports,
                'module_count': len(modules)
            }
        
        return packages
    
    def check_documentation(self) -> Dict[str, Any]:
        """Check for documentation files and their content."""
        logger.info("Checking documentation...")
        
        doc_files = []
        doc_types = {
            '.md': 'markdown',
            '.rst': 'restructured_text',
            '.txt': 'plain_text',
            '.html': 'html'
        }
        
        # Look for common documentation files
        common_docs = ['README', 'CHANGELOG', 'LICENSE', 'CONTRIBUTING', 'INSTALL', 'USAGE']
        
        for root, dirs, files in os.walk(self.project_root):
            for file in files:
                file_path = Path(root) / file
                file_stem = file_path.stem.upper()
                file_suffix = file_path.suffix.lower()
                
                is_doc = (
                    file_stem in common_docs or
                    file_suffix in doc_types or
                    'doc' in str(file_path).lower()
                )
                
                if is_doc:
                    try:
                        size = file_path.stat().st_size
                        doc_info = {
                            'path': str(file_path.relative_to(self.project_root)),
                            'type': doc_types.get(file_suffix, 'unknown'),
                            'size': size,
                            'has_content': size > 0
                        }
                        doc_files.append(doc_info)
                    except Exception as e:
                        logger.warning(f"Error checking {file_path}: {e}")
        
        return {
            'documentation_files': doc_files,
            'total_doc_files': len(doc_files),
            'has_readme': any('readme' in doc['path'].lower() for doc in doc_files),
            'has_license': any('license' in doc['path'].lower() for doc in doc_files)
        }
    
    def check_configuration_files(self) -> Dict[str, Any]:
        """Check for configuration and setup files."""
        logger.info("Checking configuration files...")
        
        config_files = []
        config_patterns = [
            'setup.py', 'setup.cfg', 'pyproject.toml',
            'requirements.txt', 'requirements-dev.txt',
            'Pipfile', 'poetry.lock', 'environment.yml',
            'config.json', 'config.yaml', 'config.yml', 'config.ini',
            '.env', '.env.example', '.env.template',
            'Dockerfile', 'docker-compose.yml',
            '.gitignore', '.gitattributes',
            'Makefile', 'tox.ini', 'pytest.ini'
        ]
        
        for pattern in config_patterns:
            matches = list(self.project_root.glob(pattern))
            for match in matches:
                try:
                    size = match.stat().st_size
                    config_files.append({
                        'path': str(match.relative_to(self.project_root)),
                        'type': self._get_config_type(match.name),
                        'size': size,
                        'has_content': size > 0
                    })
                except Exception as e:
                    logger.warning(f"Error checking {match}: {e}")
        
        return {
            'configuration_files': config_files,
            'total_config_files': len(config_files),
            'has_setup_py': any('setup.py' in cfg['path'] for cfg in config_files),
            'has_requirements': any('requirements' in cfg['path'] for cfg in config_files),
            'has_gitignore': any('.gitignore' in cfg['path'] for cfg in config_files)
        }
    
    def _get_config_type(self, filename: str) -> str:
        """Determine the type of configuration file."""
        if 'requirements' in filename:
            return 'dependencies'
        elif filename in ['setup.py', 'setup.cfg', 'pyproject.toml']:
            return 'package_setup'
        elif filename in ['config.json', 'config.yaml', 'config.yml', 'config.ini']:
            return 'application_config'
        elif filename.startswith('.env'):
            return 'environment'
        elif filename in ['Dockerfile', 'docker-compose.yml']:
            return 'docker'
        elif filename.startswith('.git'):
            return 'git'
        else:
            return 'other'
    
    def validate_project_completeness(self) -> Dict[str, Any]:
        """Validate that the project is complete and functional."""
        logger.info("Validating project completeness...")
        
        issues = []
        recommendations = []
        score = 100
        
        # Check if utils package exists and is complete
        utils_dir = self.project_root / "utils"
        if not utils_dir.exists():
            issues.append("Missing utils package")
            score -= 20
        else:
            init_file = utils_dir / "__init__.py"
            if not init_file.exists():
                issues.append("utils package missing __init__.py")
                score -= 10
        
        # Check for main modules
        expected_modules = [
            "utils/text_preprocessor.py",
            "utils/model_utils.py",
            "utils/inference.py",
            "utils/evaluation.py"
        ]
        
        for module in expected_modules:
            module_path = self.project_root / module
            if not module_path.exists():
                issues.append(f"Missing module: {module}")
                score -= 5
            else:
                # Check if module has content
                try:
                    with open(module_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if len(content.strip()) < 100:  # Very minimal content
                        issues.append(f"Module {module} appears to be incomplete")
                        score -= 3
                except Exception:
                    issues.append(f"Cannot read module: {module}")
                    score -= 5
        
        # Check for test files
        test_files = list(self.project_root.glob("tests/*.py"))
        if len(test_files) < 3:
            recommendations.append("Add more test files for better coverage")
            score -= 5
        
        # Check for documentation
        readme_files = list(self.project_root.glob("README*"))
        if not readme_files:
            issues.append("Missing README file")
            score -= 10
        
        # Check for requirements
        req_files = list(self.project_root.glob("requirements*.txt"))
        if not req_files:
            recommendations.append("Add requirements.txt for dependencies")
            score -= 5
        
        return {
            'completeness_score': max(0, score),
            'issues': issues,
            'recommendations': recommendations,
            'is_complete': len(issues) == 0,
            'total_issues': len(issues),
            'total_recommendations': len(recommendations)
        }
    
    def generate_project_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive project summary."""
        logger.info("Generating project summary...")
        
        return {
            'project_name': 'NLP Sentiment Analysis',
            'project_root': str(self.project_root),
            'analysis_timestamp': str(datetime.now()),
            'structure': self.analyze_directory_structure(),
            'python_analysis': self.analyze_python_files(),
            'packages': self.analyze_package_structure(),
            'documentation': self.check_documentation(),
            'configuration': self.check_configuration_files(),
            'completeness': self.validate_project_completeness()
        }
    
    def print_summary_report(self) -> None:
        """Print a formatted summary report."""
        summary = self.generate_project_summary()
        
        print("=" * 80)
        print("🎯 NLP SENTIMENT ANALYSIS PROJECT ANALYSIS")
        print("=" * 80)
        
        # Project overview
        structure = summary['structure']
        python = summary['python_analysis']
        completeness = summary['completeness']
        
        print(f"\n📊 PROJECT OVERVIEW")
        print(f"   Project Root: {summary['project_root']}")
        print(f"   Total Files: {structure['total_files']}")
        print(f"   Total Size: {structure['total_size']:,} bytes")
        print(f"   Python Files: {python['total_python_files']}")
        print(f"   Total Lines of Code: {python['total_lines']:,}")
        
        # Code statistics
        print(f"\n💻 CODE STATISTICS")
        print(f"   Functions: {python['total_functions']}")
        print(f"   Classes: {python['total_classes']}")
        print(f"   Packages: {len(summary['packages'])}")
        print(f"   Unique Imports: {len(python['imports'])}")
        
        # File type breakdown
        print(f"\n📁 FILE TYPE BREAKDOWN")
        for file_type, count in structure['file_counts'].items():
            print(f"   {file_type}: {count}")
        
        # Project completeness
        print(f"\n✅ PROJECT COMPLETENESS")
        print(f"   Completeness Score: {completeness['completeness_score']}/100")
        print(f"   Status: {'✅ Complete' if completeness['is_complete'] else '⚠️ Issues Found'}")
        print(f"   Issues: {completeness['total_issues']}")
        print(f"   Recommendations: {completeness['total_recommendations']}")
        
        if completeness['issues']:
            print(f"\n❌ ISSUES FOUND:")
            for issue in completeness['issues']:
                print(f"   • {issue}")
        
        if completeness['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in completeness['recommendations']:
                print(f"   • {rec}")
        
        # Top-level directories
        print(f"\n📂 PROJECT STRUCTURE")
        for path, info in structure['structure'].items():
            if '/' not in path or path.count('/') == 0:
                print(f"   {path}: {info['file_count']} files, {len(info['subdirs'])} subdirs")
        
        print("\n" + "=" * 80)
    
    def save_analysis_report(self, output_file: Path) -> None:
        """Save the analysis report to a JSON file."""
        logger.info(f"Saving analysis report to {output_file}")
        
        summary = self.generate_project_summary()
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"✅ Analysis report saved to {output_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save report: {e}")


def main():
    """Main function for project analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze NLP Sentiment Analysis project structure and completeness"
    )
    
    parser.add_argument(
        '--project-root',
        type=Path,
        help='Path to project root directory (default: current script directory parent)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output file for detailed analysis report (JSON format)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress summary report output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Initialize analyzer
    analyzer = ProjectAnalyzer(project_root=args.project_root)
    
    # Generate and print summary
    if not args.quiet:
        analyzer.print_summary_report()
    
    # Save detailed report if requested
    if args.output:
        analyzer.save_analysis_report(args.output)
    
    # Return analysis for potential programmatic use
    return analyzer.generate_project_summary()


if __name__ == "__main__":
    try:
        summary = main()
        
        # Exit with error code if project has critical issues
        if summary['completeness']['completeness_score'] < 70:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)
