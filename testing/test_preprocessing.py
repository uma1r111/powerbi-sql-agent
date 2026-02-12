import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.query_preprocessor import preprocessor


def test_abbreviations():
    # Test quantity abbreviation
    query, corrections = preprocessor.preprocess("show qty in stock")
    assert "quantity" in query.lower()
    
    # Test customer abbreviation
    query, corrections = preprocessor.preprocess("top 10 custs")
    assert "customers" in query.lower()
    
    # Test revenue abbreviation
    query, corrections = preprocessor.preprocess("total rev by month")
    assert "revenue" in query.lower()

def test_typos():
    # Test common typo
    query, corrections = preprocessor.preprocess("show costumers")
    assert "customers" in query.lower()
    
    # Test misspelling
    query, corrections = preprocessor.preprocess("total revnue")
    assert "revenue" in query.lower()

def test_patterns():
    # Test top X pattern
    query, corrections = preprocessor.preprocess("top10 products")
    assert "top 10" in query.lower()
    
    # Test spacing
    query, corrections = preprocessor.preprocess("show  me   products")
    assert "show me products" == query.lower()

if __name__ == "__main__":
    test_abbreviations()
    test_typos()
    test_patterns()
    print("✅ All tests passed!")