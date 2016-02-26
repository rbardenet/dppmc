import pytest
import sys
sys.path.append("..")
import tools

def test_GradedOrder():
    trueAnswer = [[0,0,0], [0,0,1], [0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1],[0,0,2],[0,1,2]]
    cpt = 0
    it =  tools.GradedOrder(len(trueAnswer), 3)
    for l in it:
        assert tuple(trueAnswer[cpt]) == l
        cpt += 1
    with pytest.raises(StopIteration):
        it.__next__()
