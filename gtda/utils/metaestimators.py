"""Meta-estimators."""
# License: GNU AGPLv3

from functools import reduce
from operator import and_

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import clone
from sklearn.utils.metaestimators import if_delegate_has_method

from gtda.utils import check_collection


class ForEachInput(BaseEstimator, TransformerMixin):
    """Meta-transformer for applying a fit-transformer to each input in a
    collection.

    If `transformer` possesses a ``fit_transform`` method,
    ``ForEachInput(transformer)`` also possesses a :meth:`fit_transform` method
    which, on each entry in its input ``X``, fit-transforms a clone of
    `transformer`. A collection (list or ndarray) of outputs is returned.

    Parameters
    ----------
    transformer : object
        The fit-transformer instance from which the transformer acting on
        collections is built. Should implement ``fit_transform``.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use in a joblib-parallel application of
        `transformer`'s ``fit_transform`` to each input. ``None`` means 1
        unless in a :obj:`joblib.parallel_backend` context. ``-1`` means using
        all processors.

    parallel_backend_prefer :  ``"processes"`` | ``"threads"`` | ``None``, \
        optional, default: ``None``
        Soft hint for the default joblib backend to use in a joblib-parallel
        application  of `transformer`'s ``fit_transform`` to each input. To be
        used in conjunction with `n_jobs`. The default process-based backend is
        "loky" and the default thread-based backend is "threading". See [1]_.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.decomposition import PCA
    >>> from gtda.utils import ForEachInput
    >>> rng = np.random.default_rng()

    Create a collection of 1000 2D inputs for a PCA, as a single 3D ndarray (we
    could also create a list of 2D inputs instead).

    >>> X = rng.random((1000, 100, 50))

    In the case of PCA, joblib parallelism can be very beneficial!

    >>> multi_pca = ForEachInput(PCA(n_components=3), n_jobs=-1)
    >>> Xt = multi_pca.fit_transform(X)

    Since all PCA outputs have the same shape, ``Xt`` is an  ndarray.
    >>> print(Xt.shape)
    (1000, 100, 3)

    See also
    --------
    gtda.mapper.utils.pipeline.transformer_from_callable_on_rows, \
    gtda.mapper.utils.decorators.method_to_transform

    References
    ----------
    .. [1] "Thread-based parallelism vs process-based parallelism", in
           `joblib documentation
           <https://joblib.readthedocs.io/en/latest/parallel.html>`_.

    """

    def __init__(self, transformer, n_jobs=None, parallel_backend_prefer=None):
        self.transformer = transformer
        self.n_jobs = n_jobs
        self.parallel_backend_prefer = parallel_backend_prefer

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : list of length n_samples, or ndarray of shape (n_samples, ...)
            Collection of inputs to be fit-transformed by `transformer`.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        check_collection(X, accept_sparse=True, accept_large_sparse=True,
                         force_all_finite=False)

        return self

    @if_delegate_has_method(delegate="transformer")
    def fit_transform(self, X, y=None):
        """Fit-transform a clone of `transformer` to each input in `X`.

        Parameters
        ----------
        X : list of length n_samples, or ndarray of shape (n_samples, ...)
            Collection of inputs to be fit-transformed by `transformer`.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : list of length n_samples, or ndarray of shape (n_samples, ...)
            Collection of outputs. It is a list unless all outputs have the
            same shape, in which case it is converted to an ndarray.

        """
        Xt = check_collection(X, accept_sparse=True, accept_large_sparse=True,
                              force_all_finite=False)

        Xt = Parallel(n_jobs=self.n_jobs, prefer=self.parallel_backend_prefer)(
            delayed(clone(self.transformer).fit_transform)(x) for x in Xt
            )

        x0_shape = Xt[0].shape
        if reduce(and_, (x.shape == x0_shape for x in Xt), True):
            Xt = np.asarray(Xt)

        return Xt
