import numpy as np,pandas as pd,torch
from ddos_fusion.models import Conv1,Model2,Model3,Model4
from ddos_fusion.reference import cognitiveD,diversityS

torch.set_num_threads(1)

def test_extracted_models_have_binary_forward_outputs():
    for model in [Conv1(8),Model2(8,16),Model3(8),Model4(8)]:
        model.eval();out=model(torch.randn(5,8))
        assert out.shape==(5,1) and bool(((out>=0)&(out<=1)).all())
        out.mean().backward()
        assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.parameters())


def test_cnn_feature_dimensions_are_instance_local():
    a,b=Conv1(8).eval(),Conv1(12).eval()
    assert a(torch.randn(4,8)).shape==b(torch.randn(4,12)).shape==(4,1)


def test_reference_diversity_preserves_four_model_symmetry():
    data=pd.DataFrame({'Conv1d':[0.1,0.2],'GRU':[0.3,0.4],'MLP':[0.5,0.6],'ResNets':[0.7,0.8]})
    d=cognitiveD(data).to_numpy(dtype=float)
    np.testing.assert_allclose(d,d.T);np.testing.assert_allclose(np.diag(d),0)
    assert len(diversityS(data))==4
