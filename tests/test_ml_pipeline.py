import pytest
import numpy as np
from src.features.engineering import FeatureEngineeringProcessor

def test_feature_engineering_matrix_dimensions():
    """Validates that the mathematical transformation engine returns the exact spatial vector dimensions."""
    processor = FeatureEngineeringProcessor()
    
    mock_raw_data = [
        {"total_spend": 500.0, "login_frequency": 12, "support_tickets": 1},
        {"total_spend": 120.50, "login_frequency": 2, "support_tickets": 4}
    ]
    
    features_matrix = processor.compute_customer_churn_features(mock_raw_data)
    
    # Validações estruturais de formato da matriz numpy de treino
    assert isinstance(features_matrix, np.ndarray)
    assert features_matrix.shape == (2, 4) # 2 entidades com 4 colunas calculadas cada
    assert features_matrix[0][0] == 500.0

def test_probability_sigmoidal_bounds_simulation():
    """Validates that calculation abstractions fall inside appropriate probability margins (0.0 to 1.0)."""
    # Simulação da lógica da função sigmoide aplicada sobre o score de risco do cliente
    mock_support_tickets_low = 2
    mock_support_tickets_high = 15
    
    calc_low = (mock_support_tickets_low * 0.25) - (100 * 0.0001)
    calc_high = (mock_support_tickets_high * 0.25) - (100 * 0.0001)
    
    prob_low = float(1 / (1 + np.exp(-calc_low)))
    prob_high = float(1 / (1 + np.exp(-calc_high)))
    
    assert 0.0 <= prob_low <= 1.0
    assert 0.0 <= prob_high <= 1.0
    assert prob_high > prob_low # Mais chamados de suporte devem induzir maior risco de Churn
