import os
from clinical_trials_client import ClinicalTrialsClient
from trials_analysis import ClinicalTrialsAnalyzer

def test_analyzer():
    "Test basic functionality of the ClinicalTrialsAnalyzer"
    print("Testing ClinicalTrialsAnalyzer...")
    
    # Initialize client and analyzer
    client = ClinicalTrialsClient(cache_enabled=True)
    analyzer = ClinicalTrialsAnalyzer(output_dir="test_reports")
    
    print(f"Output directory: {analyzer.output_dir}")
    if not os.path.exists(analyzer.output_dir):
        os.makedirs(analyzer.output_dir)
    
    # Get some test data
    print("\nFetching test data...")
    try:
        results = client.search_trials(condition="diabetes", status="RECRUITING", limit=20)
        studies = results.get("studies", [])
        print(f"Retrieved {len(studies)} studies for analysis")
        
        if len(studies) < 5:
            print("Warning: Small sample size may affect analysis quality")
            
    except Exception as e:
        print(f"Error fetching test data: {str(e)}")
        return
    
    # Test DataFrame creation
    print("\nTesting create_dataframe()...")
    try:
        df = analyzer.create_dataframe(studies)
        print(f"Created DataFrame with {df.shape[0]} rows and {df.shape[1]} columns")
        print(f"Columns: {', '.join(df.columns)}")
    except Exception as e:
        print(f"Error creating DataFrame: {str(e)}")
        return
    
    # Test phase analysis
    print("\nTesting analyze_by_phase()...")
    try:
        phase_results = analyzer.analyze_by_phase(df)
        print(f"Phase distribution: {phase_results['counts']}")
        print(f"Chart saved to: {phase_results['chart_path']}")
    except Exception as e:
        print(f"Error in phase analysis: {str(e)}")
    
    # Test status analysis
    print("\nTesting analyze_by_status()...")
    try:
        status_results = analyzer.analyze_by_status(df)
        print(f"Status distribution: {status_results['counts']}")
        print(f"Chart saved to: {status_results['chart_path']}")
    except Exception as e:
        print(f"Error in status analysis: {str(e)}")
    
    # Test enrollment statistics
    print("\nTesting enrollment_statistics()...")
    try:
        enrollment_results = analyzer.enrollment_statistics(df)
        print(f"Enrollment stats: {enrollment_results['statistics']}")
        print(f"Chart saved to: {enrollment_results['chart_path']}")
    except Exception as e:
        print(f"Error in enrollment analysis: {str(e)}")
    
    # Test full report generation
    print("\nTesting generate_report()...")
    try:
        report = analyzer.generate_report(studies, "diabetes")
        print(f"Generated report: {report['summary']}")
    except Exception as e:
        print(f"Error generating report: {str(e)}")

if __name__ == "__main__":
    test_analyzer()
