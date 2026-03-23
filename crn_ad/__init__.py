"""
CRN_AD: Chemical Reaction Network with Automatic Differentiation

A pH-responsive chemical reaction network whose parameters (pKa values,
interaction strength, steric mismatch factor phi) are trained via JAX
autodiff so the network preferentially 'folds' (forms correct dimers)
only under a target sequence of pH stimuli.
"""
