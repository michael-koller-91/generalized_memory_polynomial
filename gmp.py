import numpy as np
import numpy.typing as npt


class MemoryPolynomial:
    def __init__(
        self, degree: int, mem_depth: int, alpha: float = 0.0, cross_terms: bool = True
    ):
        """Set the memory polynomial hyperparameters.

        Parameters
        ----------
        degree : int
            The polynomial degree.
        mem_depth : int
            The memory depth.
        alpha : float, optional
            The regularization parameter. (The default is 0.0 which yields the least squares solution.)
        cross_terms : bool, optional
            If False, cross terms of the memory polynomial are not used. (The default is True.)
        """
        self.degree = degree
        self.mem_depth = mem_depth
        self.alpha = alpha
        self.cross_terms = cross_terms

        self.coef = None

    def fit(self, mem_input: npt.NDArray, mem_output: npt.NDArray) -> npt.NDArray:
        """Determine the memory polynomial coefficients.

        Parameters
        ----------
        mem_input : ndarray
            The input of the memory polynomial.
        mem_output : ndarray
            The output of the memory polynomial. The goal is to determine the coefficients such that applying the memory
            polynomial to mem_input yields mem_output.

        Returns
        -------
        coef : ndarray
            The coefficients of the memory polynomial.
        """
        assert mem_input.ndim == 1, "Expected mem_input to have 1 dimension."
        assert mem_output.ndim == 1, "Expected mem_output to have 1 dimension."

        if self.cross_terms:
            X = self._make_X_cross_terms(mem_input)
        else:
            X = self._make_X_no_cross_terms(mem_input)

        self.coef = np.linalg.lstsq(
            X.dot(X.T.conj()) + self.alpha * np.identity(X.shape[0]),
            X.dot(mem_output[self.mem_depth - 1 :]),
            rcond=None,
        )[0]

        return self.coef

    def __call__(self, x: npt.NDArray) -> npt.NDArray:
        """Apply the memory polynomial to x."""
        assert x.ndim == 1, "Expected x to have 1 dimension."

        if self.coef is None:
            raise RuntimeError(
                "The memory polynomial coefficients have not been determined yet. Call the fit method first."
            )

        x0 = np.zeros(x.shape[0] + self.mem_depth - 1)
        x0[self.mem_depth - 1 :] = x
        if self.cross_terms:
            return self.coef @ self._make_X_cross_terms(x0)
        else:
            return self.coef @ self._make_X_no_cross_terms(x0)

    def _make_X_cross_terms(self, x):
        X = np.empty(
            (
                self.mem_depth + self.mem_depth**2 * (self.degree - 1),
                x.shape[0] - self.mem_depth + 1,
            )
        )

        for m in range(self.mem_depth):
            X[m, :] = x[self.mem_depth - m - 1 : x.shape[0] - m]

        idx = self.mem_depth - 1
        for m in range(self.mem_depth):
            for j in range(self.mem_depth):
                for k in range(1, self.degree):
                    idx += 1
                    X[idx, :] = (
                        x[self.mem_depth - m - 1 : x.shape[0] - m]
                        * np.abs(x[self.mem_depth - j - 1 : x.shape[0] - j]) ** k
                    )

        return X

    def _make_X_no_cross_terms(self, x):
        X = np.empty(
            (
                self.mem_depth * self.degree,
                x.shape[0] - self.mem_depth + 1,
            )
        )

        idx = -1
        for k in range(self.degree):
            for m in range(self.mem_depth):
                idx += 1
                X[idx, :] = (
                    x[self.mem_depth - m - 1 : x.shape[0] - m]
                    * np.abs(x[self.mem_depth - m - 1 : x.shape[0] - m]) ** k
                )

        return X


if __name__ == "__main__":
    """Perform some tests."""

    #
    # compare cross_terms = False with manual computation
    #
    # random data
    mem_in = np.array(
        [
            0.558477700124607,
            -0.715024260922966,
            3.163501150295815,
            1.863496209009447,
            0.285950580815500,
            -1.578691616589387,
            1.435843649651902,
            0.014396432983365,
            0.335509831859568,
            0.510374049149944,
        ]
    )
    mem_out = np.array(
        [
            -2.149234670459673,
            -1.122750053414559,
            0.437938005044204,
            0.539855703633882,
            -0.051937494465593,
            0.528598841250532,
            0.333067810149752,
            0.789186307487690,
            0.734953461595165,
            0.982835322902923,
        ]
    )
    mem_pol = MemoryPolynomial(degree=2, mem_depth=3, alpha=0, cross_terms=False)
    coef = np.sort(mem_pol.fit(mem_in, mem_out))
    coef_manual = np.sort(
        np.array(
            [
                0.822607438583439,
                -0.191077564064046,
                1.457358369083999,
                -0.433566823063547,
                1.326508432215479,
                -0.563958410228584,
            ]
        )
    )
    print("--- cross_terms = False ---")
    print("np.allclose(coef, coef_manual) =", np.allclose(coef, coef_manual))
    x = np.arange(1, 11)
    y_manual = np.array(
        [
            0.631529874519394,
            1.904696166931149,
            2.691123707074617,
            1.100345652505728,
            -2.867637996775505,
            -9.212827240769094,
            -17.935222079475032,
            -29.034822512893324,
            -42.511628541023974,
            -58.365640163866956,
        ]
    )
    y = mem_pol(x)
    print("np.allclose(y, y_manual) =", np.allclose(y, y_manual))

    del coef, coef_manual, y, y_manual

    #
    # compare cross_terms = True with manual computation
    #
    # random data
    mem_in = np.array(
        [
            -1.191565625468646,
            0.509379624489162,
            0.091693905857362,
            -1.452240886408114,
            0.607932770815961,
            -0.833525020031206,
            0.544291298400450,
            1.162257811034621,
            -0.905563628996942,
            -0.709044234050656,
            0.205585034976983,
            -0.682238828067065,
            -1.248210810667920,
            -2.565744268782611,
            0.237181916660355,
            2.570202006685949,
            -0.230271142399029,
            0.174262896942398,
            0.802316033177130,
            -0.122993270013892,
        ]
    )
    mem_out = np.array(
        [
            -1.018035453756015,
            0.163191367717437,
            -0.892785295327948,
            -0.192178761303400,
            -0.229659115715074,
            -0.969685973784051,
            -0.450777536308442,
            -0.694843291992499,
            -0.105731404994098,
            -0.564769529634786,
            0.908204157506864,
            0.142876888235028,
            -0.464574143776830,
            -1.388391204159283,
            0.449745405500624,
            1.056076997041570,
            -0.191782485456401,
            -0.321604653183496,
            0.456438037146792,
            0.283467323275400,
        ]
    )
    mem_pol = MemoryPolynomial(degree=2, mem_depth=3, alpha=0, cross_terms=True)
    coef = np.sort(mem_pol.fit(mem_in, mem_out))
    coef_manual = np.sort(
        np.array(
            [
                0.021949111183202,
                -0.121631285162029,
                0.103621306593167,
                0.330222635735031,
                -0.052757117769840,
                0.409502665247861,
                0.022060322166631,
                -0.479009960849836,
                -0.368256130600532,
                0.163943104928630,
                0.383130008542019,
                0.004703726985394,
            ]
        )
    )
    print("--- cross_terms = True ---")
    print("np.allclose(coef, coef_manual) =", np.allclose(coef, coef_manual))
    x = np.arange(1, 21)
    y_manual = 1e2 * np.array(
        [
            -0.000996821739788,
            0.005529242297971,
            0.029598215650246,
            0.069998039486257,
            0.126728713806006,
            0.199790238609493,
            0.289182613896717,
            0.394905839667678,
            0.516959915922376,
            0.655344842660812,
            0.810060619882986,
            0.981107247588896,
            1.168484725778544,
            1.372193054451930,
            1.592232233609052,
            1.828602263249912,
            2.081303143374510,
            2.350334873982844,
            2.635697455074917,
            2.937390886650727,
        ]
    )
    y = mem_pol(x)
    print("np.allclose(y, y_manual) =", np.allclose(y, y_manual))

    #
    # see if cross_terms = True performs at least as well as cross_terms = False
    #
    print("--- compare cross_terms = False with cross_terms = True ---")
    rng = np.random.default_rng(12345678)

    n_samples = 100_000
    degree = 5
    mem_depth = 4
    alpha = 1e-10

    # distort some random data
    mem_out = rng.standard_normal((2, n_samples))
    mem_in = mem_out + 0.1 * mem_out**2 + 0.01 * mem_out**3

    # try to reconstruct the non-distorted data
    mem_pol_cross = MemoryPolynomial(degree, mem_depth, alpha, cross_terms=True)
    mem_pol_cross.fit(mem_in[0, :], mem_out[0, :])
    yTrue = mem_pol_cross(mem_in[1, :])

    mem_pol = MemoryPolynomial(degree, mem_depth, alpha, cross_terms=False)
    mem_pol.fit(mem_in[0, :], mem_out[0, :])
    yFalse = mem_pol(mem_in[1, :])

    # it should hold: eT <= eF <= e0
    e0 = np.mean(np.abs(mem_in[1, :] - mem_out[1, :]))
    eT = np.mean(np.abs(yTrue - mem_out[1, :]))
    eF = np.mean(np.abs(yFalse - mem_out[1, :]))
    print("norm(mem_in[1, :] - mem_out[1, :]) =", e0)
    print("norm(yTrue - mem_out[1, :])        =", eT)
    print("norm(yFalse - mem_out[1, :])       =", eF)
