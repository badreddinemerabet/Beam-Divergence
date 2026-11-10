"""Regression checks for normalization, near-source behavior and validation splits."""
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.special import airy
from beam_models import airy_norm, airy_intensity, top_hat, bessel_gaussian
from beam_recommender import grouped_split, physics_divergence, make_features
import lg01_multiwavelength_propagation as lg
import hg10_multiwavelength_propagation as hg

class NumericalChecks(unittest.TestCase):
    def test_airy_total_power(self):
        a=.08; scale=.05
        integral=quad(lambda s: airy(s)[0]**2*np.exp(2*a*s),-200,20,limit=1000)[0]
        self.assertAlmostEqual((airy_norm(scale,a)*scale*integral)**2,1.,places=7)
        self.assertTrue(np.isfinite(airy_intensity(0,0,100000,1064e-9,scale,a)))

    def test_top_hat_source_continuity_and_far_power(self):
        lam=1064e-9; radius=.05; source=1/(np.pi*radius**2)
        for z in [0,1e-12,1e-6,100,100000]:
            self.assertLessEqual(float(top_hat(0,z,lam,radius)),source*(1+1e-12))
        self.assertAlmostEqual(float(top_hat(0,1e-12,lam,radius)),source,places=8)
        z=100000.; factor=2*np.pi/lam*radius/z
        # Dimensionless radial integral captures >99.9% of Airy-pattern power.
        power=quad(lambda u: 2*np.pi*u/factor**2*float(top_hat(u/factor,z,lam,radius)),0,2000,limit=3000)[0]
        self.assertAlmostEqual(power,1.,delta=.001)

    def test_bessel_power_at_two_planes(self):
        for z in [0,100000]:
            w=.2*np.sqrt(1+(z/(np.pi*.2**2/1064e-9))**2)
            power=quad(lambda r:2*np.pi*r*float(bessel_gaussian(r,z,1064e-9,.05e-3,.2)),0,8*w,limit=1000)[0]
            self.assertAlmostEqual(power,1.,places=7)

    def test_higher_mode_normalization(self):
        for module in [lg,hg]:
            w=module.W0; a=np.linspace(-6*w,6*w,801); X,Y=np.meshgrid(a,a)
            I=module.intensity(X,Y,0,1064e-9)
            self.assertAlmostEqual(float(np.trapezoid(np.trapezoid(I,a,axis=1),a)),1.,places=7)

    def test_group_split_has_no_scenario_leakage(self):
        scenarios=[(j,j+1) for j in range(20) for _ in range(7)]
        tr,te=grouped_split(scenarios,np.random.default_rng(1))
        self.assertFalse({scenarios[i] for i in tr}&{scenarios[i] for i in te})
        self.assertEqual(len(tr)+len(te),len(scenarios))

    def test_recommender_matches_plot_envelope(self):
        s=(1064e-9,.01,1.,100000.,.2,.05e-3,.2,.05)
        self.assertAlmostEqual(physics_divergence('LG01',*s),float(lg.waist(s[3],s[0]))/s[3])
        with self.assertRaises(ValueError): physics_divergence('Gaussian',*s[:3],0,*s[4:])

    def test_unused_receiver_does_not_change_features(self):
        s=(1064e-9,.01,1.,100000.,.2,.05e-3,.2,.05)
        other=s[:4]+(2.,)+s[5:]
        np.testing.assert_array_equal(make_features('Airy',s),make_features('Airy',other))

if __name__=='__main__': unittest.main()
