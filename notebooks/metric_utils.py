""" A set of predefined weight schemes that can be used with the agreement \
coefficients.
"""

import numpy as np
import pandas as pd

from copy import deepcopy

from scipy import stats


class Weights:
    """Methods for computing weights for a set of categories.

    The class can compute weights, using the methods:

    * bipolar
    * circular
    * identity
    * linear
    * ordinal
    * quadratic
    * radial
    * ratio
    """

    def __init__(self, categories):
        """Initialize a `Weights` class based on the rating categories.

        Parameters
        ----------
        categories : list, array-like or data frame
            A mandatory parameter representing the vector of all possible
            ratings. If the `categories` has numerical values, the vector is
            sorted. If the `categories` are categorical labels, the vector is
            a sequence of numbers in range `1…len(categories)`.

        Raises
        ------
        ValueError

            * The provided value for `categories` is not one of a list, a numpy
              array or a pandas dataframe.
            * Giving an unknown name of weights type.
        """
        if isinstance(categories, list):
            self.q = len(categories)
        elif isinstance(categories, (np.ndarray, pd.DataFrame)):
            self.q = categories.shape[-1]
        else:
            raise ValueError(
                "Valid input for `categories` is one of "
                "list, numpy array, or pandas data frame."
            )
        if all(isinstance(n, (int, float)) for n in categories):
            self.categ_vec = sorted(categories)
        else:
            self.categ_vec = list(range(1, len(categories) + 1))
        self.xmin, self.xmax = min(self.categ_vec), max(self.categ_vec)

    def __getitem__(self, item):
        if item == "bipolar":
            return self.bipolar()
        elif item == "circular":
            return self.circular()
        elif item == "identity":
            return self.identity()
        elif item == "linear":
            return self.linear()
        elif item == "ordinal":
            return self.ordinal()
        elif item == "quadratic":
            return self.quadratic()
        elif item == "radical":
            return self.radical()
        elif item == "ratio":
            return self.ratio()
        else:
            raise ValueError(f'"{item} is an unknown type of weights.')

    def __str__(self):
        return f"Weights for {self.q} categories."

    def bipolar(self):
        r"""Function for computing the Bipolar Weights

        Bipolar weights of a matrix :math:`\mathbf{W} \in \mathbb{R}^{q \times q}`,
        where :math:`q` is the number of `categories`, are defined for each cell
        :math:`w_{kl}, (k,l=1, \ldots, q)` by

        .. math::
            w_{kl} = \frac{(k - l)^2}{
                ((k + l - 2 \cdot \min(c))(2 \cdot \max(c) - k - l))}

        where :math:`c \in \mathbb{R}^q` is the vector of the `categories`.

        Returns
        -------
        :math:`\mathbb{R}^{q \times q}` matrix
            A square matrix of bipolar weights to be used for calculating the
            weighted coefficients.
        """
        weights = np.eye(self.q)
        for k in range(self.q):
            for el in range(self.q):
                if k != el:
                    weights[k][el] = pow(self.categ_vec[k] - self.categ_vec[el], 2) / (
                        (self.categ_vec[k] + self.categ_vec[el] - 2 * self.xmin)
                        * (2 * self.xmax - self.categ_vec[k] - self.categ_vec[el])
                    )
                else:
                    weights[k][el] = 0
        weights = 1 - weights / np.max(weights)
        return weights

    def circular(self):
        r"""Function for computing the Circular Weights

        Circular weights of a matrix :math:`\mathbf{W} \in \mathbb{R}^{q \times q}`,
        where :math:`q` is the number of `categories`, are defined for each cell
        :math:`w_{kl}, (k,l=1, \ldots, q)` by

        .. math::
            w_{kl} = \sin{(\frac{\pi (k - l)}{(\max(c) - \min(c) + 1)})^2}

        where :math:`c \in \mathbb{R}^q` is the vector of the `categories`.

        Returns
        -------
        :math:`\mathbb{R}^{q \times q}` matrix
            A square matrix of circular weights to be used for calculating the
            weighted coefficients.
        """
        weights = np.eye(self.q)
        U = self.xmax - self.xmin + 1
        for k in range(self.q):
            for el in range(self.q):
                weights[k][el] = pow(
                    np.sin(np.pi * (self.categ_vec[k] - self.categ_vec[el]) / U), 2
                )
        weights = 1 - weights / np.max(weights)
        return weights

    def identity(self):
        r"""Function for computing the Identity Weights.

        The identity weighted matrix :math:`\mathbf{W} \in \mathbb{R}^{q \times q}`,
        where :math:`q` is the number of `categories`, is the same as to calculate the
        coefficients without weights (unweighted). The weights are defined as
        :math:`w_{kk}=1, (k=1, \ldots, q)` and
        :math:`w_{kl}=0, (k,l=1, \ldots, q)` if :math:`k \neq l`.

        Returns
        -------
        :math:`\mathbb{R}^{q \times q}` matrix
            A square matrix of identity weights to be used for calculating the
            weighted coefficients.
        """
        weights = np.eye(self.q)
        return weights

    def linear(self):
        r"""Function for computing the Linear Weights.

        Linear weights of a matrix :math:`\mathbf{W} \in \mathbb{R}^{q \times q}`,
        where math:`q` is the number of `categories`, are defined for each cell
        :math:`w_{kl}, (k,l=1, \ldots, q)` by

        .. math::
            w_{kl} = 1 - \frac{|k - l|}{q - 1}

        Returns
        -------
        :math:`\mathbb{R}^{q \times q}` matrix
            A square matrix of linear weights to be used for calculating the
            weighted coefficients.
        """
        weights = np.eye(self.q)
        for k in range(self.q):
            for el in range(self.q):
                weights[k][el] = 1 - abs(self.categ_vec[k] - self.categ_vec[el]) / abs(
                    self.xmax - self.xmin
                )
        return weights

    def ordinal(self):
        r"""Function for computing the Ordinal Weights

        Ordinal weights of a matrix :math:`\mathbf{W} \in \mathbb{R}^{q \times q}`,
        where :math:`q` is the number of `categories`, are defined for each cell
        :math:`w_{kl}, (k,l=1, \ldots, q)` by

        .. math::
            w_{kl} = \frac{(\max(k, l) - \min(k, l) + 1) \cdot
                     (\max(k, l) - \min(k, l))}{2}

        Returns
        -------
        :math:`\mathbb{R}^{q \times q}` matrix
            A square matrix of ordinal weights to be used for calculating the
            weighted coefficients.
        """
        weights = np.eye(self.q)
        for k in range(self.q):
            for el in range(self.q):
                nkl = max(k, el) - min(k, el) + 1
                weights[k][el] = nkl * (nkl - 1) / 2
        weights = 1 - weights / np.max(weights)
        return weights

    def quadratic(self):
        r"""Function for computing the Quadratic Weights

        Quadratic weights of a matrix :math:`\mathbf{W} \in \mathbb{R}^{q \times q}`,
        where :math:`q` is the number of `categories`, are defined for each cell
        :math:`w_{kl}, (k,l=1, \ldots, q)` by

        .. math::
            w_{kl} = 1 - \frac{(k - l)^2}{(q - 1)^2}

        Returns
        -------
        :math:`\mathbb{R}^{q \times q}` matrix
            A square matrix of quadratic weights to be used for calculating the
            weighted coefficients.
        """
        weights = np.eye(self.q)
        diff = self.xmax - self.xmin
        for k in range(self.q):
            for el in range(self.q):
                weights[k][el] = 1 - pow(
                    (self.categ_vec[k] - self.categ_vec[el]) / diff, 2
                )
        return weights

    def radical(self):
        r"""Function for computing the Radical Weights

        Radical weights of a matrix :math:`\mathbf{W} \in \mathbb{R}^{q \times q}`,
        where :math:`q` is the number of `categories`, are defined for each cell
        :math:`w_{kl}, (k,l=1, \ldots, q)` by

        .. math::
            w_{kl} = 1 - \frac{|k - l|^{1/2}}{(\max(c) - \min(c))^{1/2}}

        where :math:`c \in \mathbb{R}^q` is the vector of the `categories`.

        Returns
        -------
        :math:`\mathbb{R}^{q \times q}` matrix
            A square matrix of radical weights to be used for calculating the
            weighted coefficients.
        """
        weights = np.eye(self.q)
        for k in range(self.q):
            for el in range(self.q):
                weights[k][el] = 1 - np.sqrt(
                    abs(self.categ_vec[k] - self.categ_vec[el])
                ) / np.sqrt(abs(self.xmax - self.xmin))
        return weights

    def ratio(self):
        r"""Function for computing the Ratio Weights

        Ratio weights of a matrix :math:`\mathbf{W} \in \mathbb{R}^{q \times q}`, where
        :math:`q` is the number of `categories`, are defined for each cell
        :math:`w_{kl}, (k,l=1, \ldots, q)` by

        .. math::
            w_{kl} = 1 - (\frac{k - l}{k + l})^2/(
                \frac{\max(c) - \min(c)}{\max(c) - \min(c)})^2

        where :math:`c \in \mathbb{R}^q` is the vector of the `categories`.

        Returns
        -------
        :math:`\mathbb{R}^{q \times q}` matrix
            A square matrix of ratio weights to be used for calculating the
            weighted coefficients.

        Raises
        ------
        ValueError
            In cases we have the 0 as a category. This will produce a division
            by zero error.
        """
        if 0 in self.categ_vec:
            raise ValueError(
                "You have 0 as a category. Please do not use"
                " 0 as a category because it produce a"
                " division by 0."
            )
        weights = np.eye(self.q)
        for k in range(self.q):
            for el in range(self.q):
                weights[k][el] = 1 - (
                    pow(
                        (self.categ_vec[k] - self.categ_vec[el])
                        / (self.categ_vec[k] + self.categ_vec[el]),
                        2,
                    )
                ) / pow((self.xmax - self.xmin) / (self.xmax + self.xmin), 2)
        return weights


class CAC:
    """ Chance-corrected Agreement Coefficients (CAC)

    Calculates various chance-corrected agreement coefficients (CAC) among 2 or
    more raters are provided. Among the CAC coefficients covered are

    * Brennan-Prediger coefficient,
    * Conger's kappa,
    * Fleiss' kappa,
    * Gwet's AC1/AC2 coefficients, and
    * Krippendorff's Alpha.

    Multiple sets of weights are proposed for computing weighted analyses.
    All of these statistical procedures are described in details in Gwet
    :cite:p:`Gwe14`.

    Parameters
    ----------
    ratings : DataFrame
        A data frame of ratings where each column represents one rater and
        each row one subject.
    weights : array-like, ndarray, or str, {"identity", "quadratic", "ordinal",\
    "linear", "radical", "ratio", "circular", "bipolar"}, default: "identity"
        A mandatory parameter that is either a string variable or a matrix.
        The string describes one of the predefined weights. If this
        parameter is a matrix then it must be a square matrix qxq where q
        is the number of possible categories where a subject can be
        classified. If some of the q possible categories are not used,
        then it is strongly advised to specify the complete list of
        possible categories as a vector in parameter ``categories``.
        Otherwise, the program may not work.
    categories : list or None, default None
        An optional vector parameter containing the list of all possible
        ratings. It may be useful in case some possible ratings are not
        used by any rater, they will still be used when calculating
        agreement coefficients. The default value is None. In this case,
        only categories reported by the raters are used in the calculations.
    confidence_level : float, default 0.95
        An optional parameter representing the confidence level associated
        with the confidence interval. Its default value is 0.95.
    N : int, default infinity
        An optional parameter representing the population size (if any).
        It may be used to perform the final population correction to the
        variance. Its default value is infinity.
    digits : int, default 5
        The number of digits to round the results.
    """

    def __init__(
        self,
        ratings,
        weights="identity",
        categories=None,
        confidence_level=0.95,
        N=np.inf,
        digits=5,
    ):
        weights_choices = (
            "identity",
            "quadratic",
            "ordinal",
            "linear",
            "radical",
            "ratio",
            "circular",
            "bipolar",
        )
        if not 0.9 <= confidence_level <= 0.99:
            raise ValueError("Please provide a value in range [0.90, 0.99].")
        self.confidence_level = confidence_level

        # Drop subjects with no ratings.
        self.ratings = ratings.dropna(how="all")
        self.ratings.replace(to_replace="", value=np.nan, inplace=True)
        self.n, self.r = self.ratings.shape  # subjects, raters
        self.f = self.n / N
        if categories is None:
            self.categories = sorted(self.ratings.stack().unique().tolist())
        else:
            self.categories = categories
        self.q = len(self.categories)
        if isinstance(weights, str):
            if weights not in weights_choices:
                raise ValueError(f"weights values can be any of {weights_choices}")
            self.weights_name = weights
            weights_functions = Weights(self.categories)
            self.weights_mat = weights_functions[self.weights_name]
        else:
            self.weights_name = "Custom Weights"
            self.weights_mat = np.asarray(weights)
            rows, cols = self.weights_mat.shape
            if not (rows == self.q and cols == self.q):
                raise ValueError(
                    f"Expected weights matrix shape is {self.q}x{self.q}. "
                    f"Given size is {rows}x{cols}."
                )
        self.digits = digits
        self.coefficient_value = 0
        self.coefficient_name = None
        self.confidence_interval = (0, 0)
        self.p_value = 0
        self.z = 0
        self.se = 0
        self.pa = 0
        self.pe = 0
        self.agreement = {
            "est": dict(
                coefficient_value=0,
                coefficient_name=None,
                confidence_interval=(0, 0),
                p_value=0,
                z=0,
                se=0,
                pa=0,
                pe=0,
            ),
            "weights": self.weights_mat,
            "categories": self.categories,
        }

    def __str__(self):
        class_path = f"{CAC.__module__}.{CAC.__name__}"
        subjects = f"Subjects: {self.n}"
        raters = f"Raters: {self.r}"
        categories = f"Categories: {self.categories}"
        weights_name = f'Weights: "{self.weights_name}"'
        _str = f"{class_path} {subjects}, {raters}, {categories}, {weights_name}"
        return f"<{_str}>"

    def __repr__(self):
        return self.__str__()

    def gwet(self):
        """Gwet's AC1/AC2 coefficient.

        The AC1 coefficient was suggested by Gwet :cite:p:`Gwe08` as a
        paradox-resistant alternative to Cohen’s Kappa. The percent chance agreement it
        is defined as the propensity for raters to agree on hard-to-score subjects and
        is calculated by multiplying the probability to agree when the rating is random
        by the probability to select a hard-to-score subject.

        The Gwet's AC2 coefficient is the one when using weights for the
        calculation.
        """
        agree_mat = np.zeros(shape=(self.n, self.q))
        for k in range(self.q):
            agree_mat[:, k] = self.ratings[self.ratings == self.categories[k]].count(
                axis=1
            )
        agree_mat_w = np.transpose(np.matmul(self.weights_mat, agree_mat.T))
        ri_vec = agree_mat.sum(axis=1)
        sum_q = (agree_mat * (agree_mat_w - 1)).sum(axis=1)
        n2more = sum(ri_vec >= 2)
        pa = sum(sum_q[ri_vec >= 2] / (ri_vec * (ri_vec - 1))[ri_vec >= 2]) / n2more
        pi_vec = (
            np.repeat(1 / self.n, self.n).reshape(self.n, 1)
            * (agree_mat / np.repeat(ri_vec, self.q).reshape(self.n, self.q))
        ).T.sum(axis=1)
        weights_mat_sum = np.sum(np.sum(self.weights_mat))
        if self.q >= 2:
            pe = weights_mat_sum * sum(pi_vec * (1 - pi_vec)) / (self.q * (self.q - 1))
        else:
            pe = 1 - 1e-15
        ac1 = (pa - pe) / (1 - pe)
        den_ivec = ri_vec * (ri_vec - 1)
        den_ivec = den_ivec - (den_ivec == 0)
        pa_ivec = sum_q / den_ivec
        pe_r2 = pe * (ri_vec >= 2)
        ac1_ivec = (self.n / n2more) * (pa_ivec - pe_r2) / (1 - pe)
        pe_ivec = (
            (weights_mat_sum / (self.q * (self.q - 1)))
            * np.matmul(agree_mat, (1 - pi_vec))
            / ri_vec
        )
        ac1_ivec_x = ac1_ivec - 2 * (1 - ac1) * (pe_ivec - pe) / (1 - pe)
        var_ac1 = (1 - self.f) / (self.n * (self.n - 1)) * sum((ac1_ivec_x - ac1) ** 2)
        stderr = np.sqrt(var_ac1)
        if stderr == 0.0:
            stderr = 1e-15
        p_value = 2 * (1 - stats.t.cdf(abs(ac1 / stderr), self.n - 1))
        lcb, ucb = stats.t.interval(
            self.confidence_level, df=self.n - 1, scale=stderr, loc=ac1
        )
        ucb = min(1, ucb)

        if weights_mat_sum == self.q:
            coeff_name = "AC1"
        else:
            coeff_name = "AC2"

        self.coefficient_value = np.round(ac1, self.digits)
        self.coefficient_name = coeff_name
        self.confidence_interval = (round(lcb, self.digits), round(ucb, self.digits))
        self.p_value = p_value
        self.z = round(ac1 / stderr, self.digits)
        self.se = round(stderr, self.digits)
        self.pa = round(pa, self.digits)
        self.pe = round(pe, self.digits)
        self.agreement["est"].update(
            dict(
                coefficient_name=coeff_name,
                pa=self.pa,
                pe=self.pe,
                se=self.se,
                z=self.z,
                coefficient_value=self.coefficient_value,
                confidence_interval=self.confidence_interval,
                p_value=self.p_value,
            )
        )
        return deepcopy(self.agreement)

    def fleiss(self):
        """Fleiss' generalized kappa coefficient.

        Fleiss :cite:p:`Fle71` defined the percent chance agreement the
        probability that any pair of raters classify a subject into the same category.

        Notes
        -----
        The calculation of the kappa coefficient here takes into account any
        missing values.
        """
        agree_mat = np.zeros(shape=(self.n, self.q))
        for c in range(self.q):
            agree_mat[:, c] = self.ratings[self.ratings == self.categories[c]].count(
                axis=1
            )
        agree_mat_w = np.transpose(np.matmul(self.weights_mat, agree_mat.T))
        ri_vec = agree_mat.sum(axis=1)
        sum_q = (agree_mat * (agree_mat_w - 1)).sum(axis=1)
        n2more = sum(ri_vec >= 2)
        pa = float(
            sum(sum_q[ri_vec >= 2] / (ri_vec * (ri_vec - 1))[ri_vec >= 2]) / n2more
        )
        pi_vec = (
            np.repeat(1 / self.n, self.n).reshape(self.n, 1)
            * (agree_mat / np.repeat(ri_vec, self.q).reshape(self.n, self.q))
        ).T.sum(axis=1)
        pe = float(
            np.sum(
                self.weights_mat
                * (pi_vec.reshape(self.q, 1) * pi_vec.reshape(1, self.q))
            )
        )
        fleiss_kappa = (pa - pe) / (1 - pe)
        den_ivec = ri_vec * (ri_vec - 1)
        den_ivec = den_ivec - (den_ivec == 0)
        pa_ivec = sum_q / den_ivec
        pe_r2 = pe * (ri_vec >= 2)
        kappa_ivec = (self.n / n2more) * (pa_ivec - pe_r2) / (1 - pe)
        pi_vec_wk_ = np.matmul(self.weights_mat, pi_vec)
        pi_vec_w_k = np.matmul(self.weights_mat.T, pi_vec)
        pi_vec_w = (pi_vec_wk_ + pi_vec_w_k) / 2
        pe_ivec = np.matmul(agree_mat, pi_vec_w) / ri_vec
        kappa_ivec_x = kappa_ivec - 2 * (1 - fleiss_kappa) * (pe_ivec - pe) / (1 - pe)
        var_fleiss = (
            (1 - self.f)
            / (self.n * (self.n - 1))
            * sum((kappa_ivec_x - fleiss_kappa) ** 2)
        )
        stderr = np.sqrt(var_fleiss)
        if stderr == 0.0:
            stderr = 1e-15
        p_value = float(2 * (1 - stats.t.cdf(abs(fleiss_kappa / stderr), self.n - 1)))
        lcb, ucb = stats.t.interval(
            self.confidence_level, df=self.n - 1, scale=stderr, loc=fleiss_kappa
        )
        ucb = min(1, ucb)

        self.coefficient_value = round(fleiss_kappa, self.digits)
        self.coefficient_name = "Fleiss' kappa"
        self.confidence_interval = (round(lcb, self.digits), round(ucb, self.digits))
        self.p_value = p_value
        self.z = round(fleiss_kappa / stderr, self.digits)
        self.se = round(stderr, self.digits)
        self.pa = round(pa, self.digits)
        self.pe = round(pe, self.digits)
        self.agreement["est"].update(
            dict(
                coefficient_name=self.coefficient_name,
                pa=self.pa,
                pe=self.pe,
                se=self.se,
                z=self.z,
                coefficient_value=self.coefficient_value,
                confidence_interval=self.confidence_interval,
                p_value=self.p_value,
            )
        )
        return deepcopy(self.agreement)

    def krippendorff(self):
        """Krippendorff’s alpha coefficient for an arbitrary number of raters.

        Krippendorff’s alpha :cite:p:`Kri70,Kri80` coefficient for an arbitrary number
        of raters (2, 3, +) when the input data represent the raw ratings reported for
        each subject and each rater.
        """
        agree_mat = np.zeros(shape=(self.n, self.q))
        for k in range(self.q):
            agree_mat[:, k] = self.ratings[self.ratings == self.categories[k]].count(
                axis=1
            )
        agree_mat_w = np.transpose(np.matmul(self.weights_mat, agree_mat.T))
        ri_vec = agree_mat.sum(axis=1)
        agree_mat = agree_mat[ri_vec >= 2]
        agree_mat_w = agree_mat_w[ri_vec >= 2]
        ri_vec = ri_vec[ri_vec >= 2]
        ri_mean = np.mean(ri_vec)
        n = agree_mat.shape[0]
        epsi = 1 / np.sum(ri_vec)
        sum_q = (agree_mat * (agree_mat_w - 1)).sum(axis=1)
        paprime = np.sum(sum_q / (ri_mean * (ri_vec - 1))) / n
        pa = float((1 - epsi) * paprime + epsi)
        pi_vec = np.matmul(np.repeat(1 / n, n).reshape(1, n), agree_mat / ri_mean).T
        pe = float(np.sum(self.weights_mat * np.matmul(pi_vec, pi_vec.T)))
        krippen_alpha = (pa - pe) / (1 - pe)
        krippen_alpha_est = np.round(krippen_alpha, self.digits)
        krippen_alpha_prime = (paprime - pe) / (1 - pe)
        pa_ivec = (
            sum_q / (ri_mean * (ri_vec - 1)) - pa * (ri_vec - ri_mean) / ri_mean
        ).reshape(-1, 1)
        krippen_ivec = (pa_ivec - pe) / (1 - pe)
        pi_vec_wk_ = np.matmul(self.weights_mat, pi_vec)
        pi_vec_w_k = np.matmul(self.weights_mat.T, pi_vec)
        pi_vec_w = (pi_vec_wk_ + pi_vec_w_k) / 2
        pe_ivec = np.matmul(agree_mat, pi_vec_w) / ri_mean - (
            pe * (ri_vec - ri_mean) / ri_mean
        ).reshape(-1, 1)
        krippen_ivec_x = krippen_ivec - 2 * (1 - krippen_alpha_prime) * (
            pe_ivec - pe
        ) / (1 - pe)
        var_krippen = (
            (1 - self.f)
            / (n * (n - 1))
            * sum((krippen_ivec_x - krippen_alpha_prime) ** 2)
        )
        stderr = np.sqrt(float(var_krippen.item()))
        if stderr == 0.0:
            stderr = 1e-15
        p_value = 2 * (1 - stats.t.cdf(abs(krippen_alpha / stderr), n - 1))
        lcb, ucb = stats.t.interval(
            self.confidence_level, df=n - 1, scale=stderr, loc=krippen_alpha
        )
        ucb = min(1, ucb)
        self.coefficient_value = round(krippen_alpha_est, self.digits)
        self.coefficient_name = "Krippendorff's Alpha"
        self.confidence_interval = (round(lcb, self.digits), round(ucb, self.digits))
        self.p_value = p_value
        self.z = round(krippen_alpha_est / stderr, self.digits)
        self.se = round(stderr, self.digits)
        self.pa = round(pa, self.digits)
        self.pe = round(pe, self.digits)
        self.agreement["est"].update(
            dict(
                coefficient_name=self.coefficient_name,
                pa=self.pa,
                pe=self.pe,
                se=self.se,
                z=self.z,
                coefficient_value=self.coefficient_value,
                confidence_interval=self.confidence_interval,
                p_value=self.p_value,
            )
        )
        return deepcopy(self.agreement)

    def conger(self):
        """Conger's generalized kappa coefficient.

        Conger :cite:p:`Con80` adopted the same percent agreement :math:`p_a` as Fleiss,
        but suggested estimating the percent chance agreement by averaging all
        :math:`r(r − 1) / 2` Cohen-type pairwise percent chance agreement estimates.

        .. versionadded:: 0.2.5
        """
        agree_mat = np.zeros(shape=(self.n, self.q))
        for k in range(self.q):
            agree_mat[:, k] = self.ratings[self.ratings == self.categories[k]].count(
                axis=1
            )
        classif_mat = np.zeros(shape=(self.r, self.q))
        for k in range(self.q):
            with_mis = self.ratings == self.categories[k]
            without_mis = with_mis.T.fillna(False)
            classif_mat[:, k] = without_mis.sum(axis=1)
        ri_vec = agree_mat.sum(axis=1)
        agree_mat_w = np.transpose(np.matmul(self.weights_mat, agree_mat.T))
        sum_q = (agree_mat * (agree_mat_w - 1)).sum(axis=1)
        n2more = sum(ri_vec >= 2)
        pa = sum(sum_q[ri_vec >= 2] / (ri_vec * (ri_vec - 1))[ri_vec >= 2]) / n2more
        ng_vec = classif_mat.sum(axis=1).reshape(-1, 1)
        pgk_mat = classif_mat / np.broadcast_to(ng_vec, (self.r, self.q))
        p_mean_k = pgk_mat.T.sum(axis=1) / self.r
        p_mean_k = p_mean_k.reshape(-1, 1)
        s2kl_mat = (
            np.matmul(pgk_mat.T, pgk_mat) - self.r * (p_mean_k * p_mean_k.T)
        ) / (self.r - 1)
        pe = np.sum(self.weights_mat * (p_mean_k * p_mean_k.T - s2kl_mat / self.r))
        conger_kappa = (pa - pe) / (1 - pe)
        # bkl_mat = (self.weights_mat + self.weights_mat.T) / 2
        lambda_ig_mat = np.zeros((self.n, self.r))
        is_numeric_ratings = (
            self.ratings.map(lambda x: isinstance(x, (int, float))).all(1).sum()
            == self.n
        )
        if is_numeric_ratings:
            epsi_ig_mat = 1 - self.ratings.isna()
        else:
            epsi_ig_mat = self.ratings.map(lambda x: isinstance(x, str))
        for k in range(self.q):
            lambda_ig_kmat = np.zeros((self.n, self.r))
            for lam in range(self.q):
                delta_ig_mat = self.ratings == self.categories[lam]
                lambda_ig_kmat += self.weights_mat[k][lam] * (
                    delta_ig_mat
                    - (epsi_ig_mat - ng_vec.T / self.n)
                    * np.broadcast_to(pgk_mat[:, lam], (self.n, self.r))
                )
            lambda_ig_kmat *= self.n / ng_vec.T
            lambda_ig_mat += lambda_ig_kmat * (
                self.r * pgk_mat[:, k].mean()
                - np.broadcast_to(pgk_mat[:, k], (self.n, self.r))
            )
        pe_ivec = lambda_ig_mat.sum(axis=1) / (self.r * (self.r - 1))
        den_ivec = ri_vec * (ri_vec - 1)
        den_ivec = den_ivec - (den_ivec == 0)
        pa_ivec = sum_q / den_ivec
        pe_r2 = pe * (ri_vec >= 2)
        conger_ivec = self.n / n2more * (pa_ivec - pe_r2) / (1 - pe)
        conger_ivec_x = conger_ivec - 2 * (1 - conger_kappa) * (pe_ivec - pe) / (1 - pe)

        var_conger = (
            (1 - self.f)
            / (self.n * (self.n - 1))
            * sum((conger_ivec_x - conger_kappa) ** 2)
        )
        stderr = np.sqrt(var_conger)
        if stderr == 0.0:
            stderr = 1e-15
        p_value = float(2 * (1 - stats.t.cdf(abs(conger_kappa / stderr), self.n - 1)))
        lcb, ucb = stats.t.interval(
            self.confidence_level, df=self.n - 1, scale=stderr, loc=conger_kappa
        )
        ucb = min(1, ucb)

        self.coefficient_value = round(conger_kappa, self.digits)
        self.coefficient_name = "Conger's kappa"
        self.confidence_interval = (round(lcb, self.digits), round(ucb, self.digits))
        self.p_value = p_value
        self.z = round(conger_kappa / stderr, self.digits)
        self.se = round(stderr, self.digits)
        self.pa = round(pa, self.digits)
        self.pe = round(float(pe), self.digits)
        self.agreement["est"].update(
            dict(
                coefficient_name=self.coefficient_name,
                pa=self.pa,
                pe=self.pe,
                se=self.se,
                z=self.z,
                coefficient_value=self.coefficient_value,
                confidence_interval=self.confidence_interval,
                p_value=self.p_value,
            )
        )
        return deepcopy(self.agreement)

    def bp(self):
        """Brennan-Prediger coefficient

        The agreement coefficient recommended by Brennan and Prediger :cite:p:`BP81`
        assumes that when the rating of a subject is a random process, the subject would
        be assigned to any of the :math:`q` categories with equal probability
        :math:`1/q`, resulting in a percent chance agreement of :math:`1/q`.

        .. versionadded:: 0.4.0
        """
        agree_mat = np.zeros(shape=(self.n, self.q))
        for c in range(self.q):
            agree_mat[:, c] = self.ratings[self.ratings == self.categories[c]].count(
                axis=1
            )
        agree_mat_w = np.transpose(np.matmul(self.weights_mat, agree_mat.T))
        ri_vec = agree_mat.sum(axis=1)
        sum_q = (agree_mat * (agree_mat_w - 1)).sum(axis=1)
        n2more = sum(ri_vec >= 2)
        pa = float(
            sum(sum_q[ri_vec >= 2] / (ri_vec * (ri_vec - 1))[ri_vec >= 2]) / n2more
        )
        if self.q >= 2:
            pe = np.sum(self.weights_mat) / (self.q**2)
        else:
            pe = 1e-15
        bp_coeff = (pa - pe) / (1 - pe)
        den_ivec = ri_vec * (ri_vec - 1)
        den_ivec = den_ivec - (den_ivec == 0)
        pa_ivec = sum_q / den_ivec
        pe_r2 = pe * (ri_vec >= 2)
        bp_ivec = (self.n / n2more) * (pa_ivec - pe_r2) / (1 - pe)

        var_bp = (1 - self.f) / (self.n * (self.n - 1)) * sum((bp_ivec - bp_coeff) ** 2)
        stderr = np.sqrt(var_bp)
        if stderr == 0.0:
            stderr = 1e-15
        p_value = 1 - stats.t.cdf(abs(bp_coeff / stderr), self.n - 1)
        lcb, ucb = stats.t.interval(
            self.confidence_level, df=self.n - 1, scale=stderr, loc=bp_coeff
        )
        ucb = min(1, ucb)

        self.coefficient_value = round(bp_coeff, self.digits)
        self.coefficient_name = "Brennan-Prediger"
        self.confidence_interval = (round(lcb, self.digits), round(ucb, self.digits))
        self.p_value = p_value
        self.z = round(bp_coeff / stderr, self.digits)
        self.se = round(stderr, self.digits)
        self.pa = round(pa, self.digits)
        self.pe = round(pe, self.digits)
        self.agreement["est"].update(
            dict(
                coefficient_name=self.coefficient_name,
                pa=self.pa,
                pe=self.pe,
                se=self.se,
                z=self.z,
                coefficient_value=self.coefficient_value,
                confidence_interval=self.confidence_interval,
                p_value=self.p_value,
            )
        )
        return deepcopy(self.agreement)

