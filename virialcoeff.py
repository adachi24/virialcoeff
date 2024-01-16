#%% Functions for the calculation of the second virial coefficient and the Boyle and critical temperatures with the monomer or rescaled dimer pair approximation
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import optimize

def ashbaugh_hatch(r: float, sigma: float, lambda_: float, epsilon: float = 0.8368) -> float:
    """
    Calculate the Ashbaugh-Hatch potential.

    Parameters
    ----------
    r: float
        Distance between two amino acids [nm].
    sigma: float
        Interaction length [nm].
    lambda_: float
        Dimensionless attractive interaction strength (hydrophobicity).
    epsilon: float
        Interaction strength [kJ/mol].

    Returns
    -------
    potential: float
        Ashbaugh-Hatch potential [kJ/mol].
    """
    if r < 2**(1/6) * sigma:
        potential = epsilon*(4*((sigma/r)**12-(sigma/r)**6)+1-lambda_)
    else:
        potential = lambda_*epsilon*4*((sigma/r)**12-(sigma/r)**6)
    return potential


def relative_permittivity(T: float, csalt: float = 0.15) -> float:
    """
    Calculate the relative permittivity for the HPS model.

    Parameters
    ----------
    T: float
        Temperature [K].
    csalt: float
        Salt ionic strength [mol/l].

    Returns
    -------
    epsilon_r: float
        Relative permittivity.
    """
    fac1 = 249.4 - 0.788*T + 0.00072*T**2
    fac2 = 1 - 0.2551*csalt + 0.05151*csalt**2 - 0.006889*csalt**3
    epsilon_r = fac1*fac2
    return epsilon_r


def debye_length(T: float, epsilon_r: float, csalt: float = 0.15) -> float:
    """
    Calculate the Debye length for the HPS model.

    Parameters
    ----------
    T: float
        Temperature [K].
    csalt: float
        Salt ionic strength [mol/l].
    epsilon_r: float
        Relative permittivity.

    Returns
    -------
    lambda_D: float
        Debye length [nm].
    """
    lambda_D = math.sqrt(3.953940922571158e-6*T*epsilon_r/csalt)  # epsilon0*kB / (2*e**2) = 3.953940922571158e-6
    return lambda_D


def debye_huckel(r: float, q1: float, q2: float, lambda_D: float, epsilon_r: float) -> float:
    """
    Calculate the electrostatic potential with the Debye-Huckel approximation.

    Parameters
    ----------
    r: float
        Distance between two amino acids [nm].
    q1, q2: float
        Normalized charges for two amino acids.
    lambda_D: float
        Debye length [nm].
    epsilon_r: float
        Relative permittivity.

    Returns
    -------
    potential: float
        Electrostatic potential [kJ/mol].
    """
    potential = 138.93545764438196*q1*q2*math.exp(-r/lambda_D) / (epsilon_r*r)  # e^2 / (4*pi*epsilon0) = 138.93545764438196
    return potential


def wang_frenkel(r: float, mu: float, epsilon_tilde: float, sigma_tilde: float) -> float:
    """
    Calculate the Wang-Frenkel potential.

    Parameters
    ----------
    r: float
        Distance between the amino acids [nm].
    mu: float
        Power for the potential.
    epsilon_tilde: float
        Overall constant [kJ/mol].
    sigma_tilde: float
        Interaction range [nm].

    Returns
    -------
    potential: float
        Wang-Frenkel potential [kJ/mol].
    """
    if r < 3*sigma_tilde:
        potential = epsilon_tilde*((sigma_tilde/r)**(2*mu)-1)*((3*sigma_tilde/r)**(2*mu)-1)**2
    else:
        potential = 0.0
    return potential


def interaction_dict(model: str) -> dict:
    """
    Get the dictionary of interaction potential between two amino acids for the given model.

    Parameters
    ----------
    model: str
        Model ('dignon', 'tesei', or 'mpipi').

    Returns
    -------
    interaction: dict
        Dictionary of the interaction potential between two amino acids [kJ/mol]
        (function of distance r [nm] and temperature T [K]).
    """
    interaction = {}
    if model in ('dignon', 'tesei'):
        df_param = pd.read_excel(f'params_{model}.xls')
        for aa1, q1, sigma1, lambda1 in zip(df_param['amino_acid'], df_param['charge'], df_param['sigma'], df_param['lambda']):
            for aa2, q2, sigma2, lambda2 in zip(df_param['amino_acid'], df_param['charge'], df_param['sigma'], df_param['lambda']):
                if np.isclose(q1, 0.0, atol=1e-5) or np.isclose(q2, 0.0, atol=1e-5):  # When electrostatic interaction is zero
                    sigma_mean = (sigma1 + sigma2) / 2
                    lambda_mean = (lambda1 + lambda2) / 2
                    interaction[aa1, aa2] = lambda r, T, sigma_mean=sigma_mean, lambda_mean=lambda_mean: ashbaugh_hatch(r, sigma_mean, lambda_mean)
                else:  # When electrostatic interaction is not zero
                    sigma_mean = (sigma1 + sigma2) / 2
                    lambda_mean = (lambda1 + lambda2) / 2
                    interaction[aa1, aa2] = lambda r, T, sigma_mean=sigma_mean, lambda_mean=lambda_mean, q1=q1, q2=q2: ashbaugh_hatch(r, sigma_mean, lambda_mean) + debye_huckel(r, q1, q2, debye_length(T, relative_permittivity(T)), relative_permittivity(T))
    elif model == 'mpipi':
        df_param1 = pd.read_excel('params_mpipi1.xls')
        df_param2 = pd.read_excel('params_mpipi2.xls')
        for aa1, q1 in zip(df_param1['amino_acid'], df_param1['charge']):
            for aa2, q2 in zip(df_param1['amino_acid'], df_param1['charge']):
                mu = df_param2['mu'][df_param2['pair'] == aa1 + '_' + aa2].values[0]
                epsilon_tilde = df_param2['epsilon_alpha'][df_param2['pair'] == aa1 + '_' + aa2].values[0]
                sigma_tilde = df_param2['sigma'][df_param2['pair'] == aa1 + '_' + aa2].values[0]
                if np.isclose(q1, 0.0, atol=1e-5) or np.isclose(q2, 0.0, atol=1e-5):  # When electrostatic interaction is zero
                    interaction[aa1, aa2] = lambda r, T, mu=mu, epsilon_tilde=epsilon_tilde, sigma_tilde=sigma_tilde: wang_frenkel(r, mu, epsilon_tilde, sigma_tilde)
                else:  # When electrostatic interaction is not zero
                    interaction[aa1, aa2] = lambda r, T, mu=mu, epsilon_tilde=epsilon_tilde, sigma_tilde=sigma_tilde, q1=q1, q2=q2: wang_frenkel(r, mu, epsilon_tilde, sigma_tilde) + debye_huckel(r, q1, q2, lambda_D=0.795, epsilon_r=80.0)
    return interaction


def virial_monomer(T: float, seq1: str, seq2: str, interaction: dict) -> float:
    """
    Calculate the second virial coefficient for a pair of sequences (seq1, seq2) with the monomer pair approximation.

    Parameters
    ----------
    T: float
        Temperature [K].
    seq1, seq2: str
        Amino acid sequences.
    interaction: dict
        Dictionary of the interaction potential between two amino acids [kJ/mol].

    Returns
    -------
    B: float
        Second virial coefficient within the monomer pair approximation [nm^3].
    """
    T_ = 8.31446261815324*1e-3*T  # Temperature [kJ/mol]
    B = 0.0  # Second virial coefficient [nm^3]

    aa_types1 = set(seq1)
    aa_types2 = set(seq2)
    for aa1 in aa_types1:
        for aa2 in aa_types2:
            B += 2*np.pi*integrate.quad(lambda r: r**2*(1-math.exp(-interaction[aa1, aa2](r, T)/T_)), 0.0, np.inf, epsrel=1e-5)[0]*seq1.count(aa1)*seq2.count(aa2)
    return B


def virial_resc_dimer(T: float, seq1: str, seq2: str, interaction: dict, alpha: float) -> float:
    """
    Calculate the second virial coefficient for a pair of sequences (seq1, seq2) with the rescaled dimer pair approximation.

    Parameters
    ----------
    T: float
        Temperature [K].
    seq1, seq2: str
        Amino acid sequences.
    interaction: dict
        Dictionary of the interaction potential between two amino acids [kJ/mol].
    alpha: float
        Rescaling factor.

    Returns
    -------
    B: float
        Second virial coefficient within the rescaled dimer pair approximation [nm^3].
    """
    T_ = 8.31446261815324*1e-3*T  # Temperature [kJ/mol]
    B = 0.0  # Second virial coefficient [nm^3]

    dimer_types1 = set([seq1[i : i + 2] for i in range(len(seq1)-1)])
    dimer_types2 = set([seq2[i : i + 2] for i in range(len(seq2)-1)])
    for dimer1 in dimer_types1:
        for dimer2 in dimer_types2:
            B += 2*np.pi*integrate.quad(lambda r: r**2*(1-math.exp(-alpha*(interaction[dimer1[0], dimer2[0]](r, T)+interaction[dimer1[0], dimer2[1]](r, T)+interaction[dimer1[1], dimer2[0]](r, T)+interaction[dimer1[1], dimer2[1]](r, T))/T_)), 0.0, np.inf, epsrel=1e-5)[0]*len(re.findall(f'(?={dimer1})', seq1))*len(re.findall(f'(?={dimer2})', seq2))
    return B


def boyle_temp_monomer(seq: str, interaction: dict, Tmin: float = 10.0, Tmax: float = 2000.0) -> float:
    """
    Calculate the Boyle temperature for a sequence with the monomer pair approximation.

    Parameters
    ----------
    seq: str
        Amino acid sequence.
    interaction: dict
        Dictionary of the interaction potential between two amino acids [kJ/mol].
    Tmin, Tmax: float
        Minimum and maximum temperatures [K].
    
    Returns
    -------
    TB: float
        Boyle temperature within the monomer pair approximation [K].
    """
    if virial_monomer(Tmin, seq, seq, interaction) < 0 and virial_monomer(Tmax, seq, seq, interaction) > 0:
        solution = optimize.root_scalar(virial_monomer, args=(seq, seq, interaction), bracket=[Tmin, Tmax])
        if solution.converged == True:
            TB = solution.root
        else:
            TB = np.nan
    else:
        TB = np.nan
    return TB


def boyle_temp_resc_dimer(seq: str, interaction: dict, alpha: float, Tmin: float = 10.0, Tmax: float = 2000.0) -> float:
    """
    Calculate the Boyle temperature for a sequence with the rescaled dimer pair approximation.

    Parameters
    ----------
    seq: str
        Amino acid sequence.
    interaction: dict
        Dictionary of the interaction potential between two amino acids [kJ/mol].
    alpha: float
        Rescaling factor.
    Tmin, Tmax: float
        Minimum and maximum temperatures [K].
    
    Returns
    -------
    TB: float
        Boyle temperature within the rescaled dimer pair approximation [K].
    """
    if virial_resc_dimer(Tmin, seq, seq, interaction, alpha) < 0 and virial_resc_dimer(Tmax, seq, seq, interaction, alpha) > 0:
        solution = optimize.root_scalar(virial_resc_dimer, args=(seq, seq, interaction, alpha), bracket=[Tmin, Tmax])
        if solution.converged == True:
            TB = solution.root
        else:
            TB = np.nan
    else:
        TB = np.nan
    return TB


def critical_temp_resc_dimer(seq: str, interaction: dict, alpha: float, l_lb: float, plot: bool = False, Tmin_TB: float = 0.8, Tmax_TB: float = 1.2) -> float:
    """
    Calculate the critical temperature for a sequence with the rescaled dimer pair approximation.

    Parameters
    ----------
    seq: str
        Amino acid sequence.
    interaction: dict
        Dictionary of the interaction potential between two amino acids [kJ/mol].
    alpha: float
        Rescaling factor.
    l_lb: float
        Lattice constant in the Flory-Huggins theory in units of the bond length lb (= 0.38 nm).
    plot: bool
        Whether to plot the fitting result.
    Tmin_TB, Tmax_TB: float
        Minimum and maximum temperatures divided by the Boyle temperature TB within the rescaled dimer pair approximation.
    
    Returns
    -------
    Tc: float
        Critical temperature within the rescaled dimer pair approximation [K].
    """
    TB = boyle_temp_resc_dimer(seq, interaction, alpha)  # Boyle temperature [K]
    N = len(seq)  # Number of amino acids in the sequence
    lb = 0.38  # Bond length [nm]
    l = l_lb*lb  # Lattice constant in the Flory-Huggins theory [nm]

    def virial_fit(T, A1, B1):  # A1 and B1 are the fitting parameters of the second virial coefficient B(T) = A1 - B1/T
        return A1 - B1 / T
    
    Ts = np.linspace(Tmin_TB*TB, Tmax_TB*TB, 10)
    Bs = np.array([virial_resc_dimer(T, seq, seq, interaction, alpha) for T in Ts])
    solution = optimize.curve_fit(virial_fit, Ts, Bs)
    A1, B1 = solution[0]
    A0, B0 = A1 / (N**2*l**3), B1 / (N**2*l**3)    
    Tc = B0 / (A0+1/math.sqrt(N)+1/(2*N))

    if plot:
        plt.plot(Ts, Bs, 'o')
        plt.plot(Ts, virial_fit(Ts, *solution[0]))
        plt.title(f'Tc = {Tc:.3g} K, A0 = {A0:.3g}, B0 = {B0:.3g}')
        plt.xlabel('T [K]')
        plt.ylabel('B^RDP [nm^3]')
        plt.show()

    return Tc


def effective_virial_monomer(T: float, seq1: str, seq2: str, interaction: dict) -> float:
    """
    Calculate the effective second virial coefficient for a pair of sequences (seq1, seq2) with the monomer pair approximation.

    Parameters
    ----------
    T: float
        Temperature [K].
    seq1, seq2: str
        Amino acid sequences.
    interaction: dict
        Dictionary of the interaction potential between two amino acids [kJ/mol].
    
    Returns
    -------
    Btilde: float
        Effective second virial coefficient within the monomer pair approximation [nm^3].
    """
    if seq1 == seq2:
        Btilde = 0.0
    else:
        Btilde = virial_monomer(T, seq1, seq2, interaction) - (virial_monomer(T, seq1, seq1, interaction)+virial_monomer(T, seq2, seq2, interaction))/2
    return Btilde


def effective_virial_resc_dimer(T: float, seq1: str, seq2: str, interaction: dict, alpha: float) -> float:
    """
    Calculate the effective second virial coefficient for a pair of sequences (seq1, seq2) with the rescaled dimer pair approximation.

    Parameters
    ----------
    T: float
        Temperature [K].
    seq1, seq2: str
        Amino acid sequences.
    interaction: dict
        Dictionary of the interaction potential between two amino acids [kJ/mol].
    alpha: float
        Rescaling factor.
    
    Returns
    -------
    Btilde: float
        Effective second virial coefficient within the rescaled dimer pair approximation [nm^3].
    """
    if seq1 == seq2:
        Btilde = 0.0
    else:
        Btilde = virial_resc_dimer(T, seq1, seq2, interaction, alpha) - (virial_resc_dimer(T, seq1, seq1, interaction, alpha)+virial_resc_dimer(T, seq2, seq2, interaction, alpha))/2
    return Btilde


def aa_types() -> list:
    """
    Get the list of amino acid types.

    Returns
    -------
    aa_types_: list
        List of amino acid types.
    """
    aa_types_ = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    return aa_types_


def dimer_types() -> list:
    """
    Get the list of dimer types.

    Returns
    -------
    dimer_types_: list
        List of dimer types.
    """
    dimer_types_ = []
    aa_types_ = aa_types()
    for aa1 in aa_types_:
        for aa2 in aa_types_:
            dimer_types_.append(aa1+aa2)
    return dimer_types_


def virial_matrix_monomer(T: float, interaction: dict) -> pd.DataFrame:
    """
    Calculate the virial coefficient matrix for amino acid monomers.

    Parameters
    ----------
    T: float
        Temperature [K].
    interaction: dict
        Dictionary of the interaction potential between two amino acids [kJ/mol].
    
    Returns
    -------
    vmat: pd.DataFrame
        Virial coefficient matrix [nm^3].
    """
    aa_types_ = aa_types()
    vmat = np.zeros((len(aa_types_), len(aa_types_)))
    for i, aa1 in enumerate(aa_types_):
        for j, aa2 in enumerate(aa_types_):
            vmat[i, j] = virial_monomer(T, aa1, aa2, interaction)
    vmat = pd.DataFrame(vmat, index=aa_types_, columns=aa_types_)
    return vmat


def virial_matrix_resc_dimer(T: float, interaction: dict, alpha: float) -> pd.DataFrame:
    """
    Calculate the virial coefficient matrix for amino acid dimers with the rescaled dimer pair approximation.

    Parameters
    ----------
    T: float
        Temperature [K].
    interaction: dict
        Dictionary of the interaction potential between two amino acids [kJ/mol].
    alpha: float
        Rescaling factor.
    
    Returns
    -------
    vmat: pd.DataFrame
        Virial coefficient matrix [nm^3].
    """
    dimer_types_ = dimer_types()
    vmat = np.zeros((len(dimer_types_), len(dimer_types_)))
    for i, dimer1 in enumerate(dimer_types_):
        for j, dimer2 in enumerate(dimer_types_):
            vmat[i, j] = virial_resc_dimer(T, dimer1, dimer2, interaction, alpha)
    vmat = pd.DataFrame(vmat, index=dimer_types_, columns=dimer_types_)
    return vmat