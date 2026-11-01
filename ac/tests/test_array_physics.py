import unittest
import numpy as np
from array_physics import codebook


class CodebookTests(unittest.TestCase):
    def test_step4_step6_sign_and_kron_are_related_by_conjugation_and_transpose(self):
        for n in (8,32):
            for angle in (-14,0,14):
                az=np.exp(1j*np.pi*np.arange(n)*np.sin(np.deg2rad(angle)))
                rt=np.kron(az,np.ones(n))/n
                los=codebook(n,angle)
                np.testing.assert_allclose(rt.reshape(n,n).T,los.reshape(n,n).conj(),atol=1e-12)
                rng=np.random.default_rng(1);h=rng.standard_normal((n,n))+1j*rng.standard_normal((n,n))
                los_field=np.sum(h*los.reshape(n,n).conj())
                rt_field=np.sum(h.conj().T*rt.reshape(n,n).conj())
                np.testing.assert_allclose(rt_field,los_field.conj(),atol=1e-12)


if __name__ == '__main__': unittest.main()
