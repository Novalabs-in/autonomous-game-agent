import pytest
import main

def test_qlearningagent_instantiation():
    # Verify that the class QLearningAgent is inspectable and loadable
    assert hasattr(main, 'QLearningAgent')

